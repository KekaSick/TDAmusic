"""
run_full_scale.py
-----------------
Финальный этап (Phase 5): Масштабирование на все 999 треков GTZAN.
- Очистка кэша / проверка MPS для однородности.
- Извлечение фичей (MERT, MuQ, Encodec, MIR).
- Обучение IncrementalPCA по батчам (решает проблему OOM).
- Подсчет Wasserstein с сэмплированием (по 2000 пар).
- Downstream-классификация (10 классов).
"""
import os
import sys
import shutil
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import embed_spaces
import preprocess
import pointcloud
import persistence
import controls
from distances import sampled_distances, bootstrap_gap, mantel

def clear_cache(cfg):
    cache_dir = cfg["paths"]["cache"]
    if os.path.exists(cache_dir):
        print(f"Clearing cache directory {cache_dir} to ensure homogeneity...")
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

def verify_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    print(f"=== DEVICE VERIFICATION ===")
    print(f"Using device: {device.upper()}")
    if device == "cpu":
        print("WARNING: GPU/MPS not available. Extraction will be extremely slow!")
    return device

def empty_cache(device):
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

def classification_bootstrap(y_true, y_pred, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs = []
    f1s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    
    acc_mean = np.mean(accs)
    acc_ci = (np.percentile(accs, 2.5), np.percentile(accs, 97.5))
    f1_mean = np.mean(f1s)
    f1_ci = (np.percentile(f1s, 2.5), np.percentile(f1s, 97.5))
    return acc_mean, acc_ci, f1_mean, f1_ci

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before running")
    parser.add_argument("--debug-subset", type=int, default=0, help="Run on N tracks for testing")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    
    if args.clear_cache:
        clear_cache(cfg)
        
    device = verify_device()
    
    # 1. Загрузка метаданных
    meta = pd.read_csv(cfg["paths"]["meta"])
    if args.debug_subset > 0:
        meta = meta.sample(args.debug_subset, random_state=42).copy()
        
    files = [os.path.join(cfg["paths"]["audio"], f) for f in meta["filename"]]
    labels = meta["genre"].values
    
    # 2. Извлечение (все 4 пространства)
    spaces = ["muq", "mert", "encodec", "mir"]
    valid_indices = {sp: [] for sp in spaces}
    
    for sp in spaces:
        print(f"\n--- Extracting space: {sp.upper()} ---")
        for i, fpath in tqdm(enumerate(files), total=len(files), desc=f"Extract {sp}", unit="track"):
            try:
                # embed_spaces.extract кэширует результат на диск
                _ = embed_spaces.extract(fpath, sp, cfg)
                valid_indices[sp].append(i)
                # Очистка памяти КАЖДЫЙ ТРЕК
                import gc
                gc.collect()
                empty_cache(device)
            except Exception as e:
                print(f"\nFailed to extract {fpath} for {sp}: {e}")
                
    # Используем только треки, успешно извлеченные во ВСЕХ пространствах
    common_idx = set(range(len(files)))
    for sp in spaces:
        common_idx = common_idx.intersection(set(valid_indices[sp]))
    common_idx = sorted(list(common_idx))
    
    files = [files[i] for i in common_idx]
    labels = labels[common_idx]
    print(f"\nValid tracks across all spaces: {len(files)}")
    
    # Stratified Train/Test Split
    stratify_labels = labels if args.debug_subset == 0 else None
    tr_idx, te_idx = train_test_split(
        np.arange(len(files)), test_size=cfg["data"]["test_size"],
        stratify=stratify_labels, random_state=cfg["seed"]
    )
    
    out_dir = cfg["paths"]["results"]
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    
    # Для хранения результатов
    wb_results = []
    clf_results = []
    dist_samples = {} # для Mantel
    
    # Конфигурация
    pc_cfg = cfg["pointcloud"]
    pc_cfg["takens_pca_dim"] = None  # Baseline
    pers_cfg = cfg["persistence"]
    
    rng = np.random.default_rng(cfg["seed"])
    
    best_f1 = -1
    best_conf_matrix = None
    best_conf_labels = None
    
    # 3. Основной цикл по пространствам
    for sp in spaces:
        print(f"\n========== Processing space: {sp.upper()} ==========")
        
        # 3.1 Incremental PCA Fitting
        print("Fitting PCAReducer (Incremental, Batching)...")
        reducer = preprocess.PCAReducer(dim=cfg["common"]["pca_dim"])
        
        # Ленивый загрузчик кадров для батчинга
        def frame_generator(indices):
            for i in indices:
                yield embed_spaces.extract(files[i], sp, cfg)
                
        # Собираем список, так как fit() ожидает list[np.ndarray], но он не съест OOM,
        # так как это просто ссылки на numpy-массивы. Но для экономии RAM:
        train_frames = [embed_spaces.extract(files[i], sp, cfg) for i in tr_idx]
        reducer.fit(train_frames, batch_size=20)
        
        # Освобождаем память
        del train_frames
        
        # 3.2 Извлечение диаграмм для всех треков (Train + Test)
        print("Computing diagrams & features for all tracks...")
        track_features = {
            "mean_pool": [],
            "dgm_real": [],
            "dgm_shuf": [],
            "dgm_rand": []
        }
        
        for i in tqdm(range(len(files)), desc="Diagrams", unit="track"):
            x = embed_spaces.extract(files[i], sp, cfg)
            x_pca = preprocess.prepare_track(x, cfg["spaces"].get(sp, {}), cfg["common"], reducer)
            
            # Mean pool
            track_features["mean_pool"].append(np.mean(x_pca, axis=0))
            
            # Real
            cloud_real = pointcloud.build_cloud(x_pca, pc_cfg)
            res_real = persistence.compute_diagrams(cloud_real, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
            track_features["dgm_real"].append(res_real["dgms"][1] if len(res_real["dgms"]) > 1 else np.empty((0,2)))
            
            # Shuffle
            x_shuf = controls.shuffle_frames(x_pca, rng)
            cloud_shuf = pointcloud.build_cloud(x_shuf, pc_cfg)
            res_shuf = persistence.compute_diagrams(cloud_shuf, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
            track_features["dgm_shuf"].append(res_shuf["dgms"][1] if len(res_shuf["dgms"]) > 1 else np.empty((0,2)))
            
            # Random
            x_rand = controls.random_like(x_pca, rng, match_autocorr=True)
            cloud_rand = pointcloud.build_cloud(x_rand, pc_cfg)
            res_rand = persistence.compute_diagrams(cloud_rand, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
            track_features["dgm_rand"].append(res_rand["dgms"][1] if len(res_rand["dgms"]) > 1 else np.empty((0,2)))

        # 3.3 Within/Between (Wasserstein)
        # Сэмплирование дистанций
        w_dists, b_dists, w_pairs, b_pairs = sampled_distances(track_features["dgm_real"], labels, n_within=2000, n_between=2000, seed=cfg["seed"])
        w_mean, b_mean, gap_mean = sampled_distances.within_between(w_arr=w_dists, b_arr=b_dists) if hasattr(sampled_distances, 'within_between') else (w_dists.mean(), b_dists.mean(), b_dists.mean() - w_dists.mean())
        _, gap_ci = bootstrap_gap(w_arr=w_dists, b_arr=b_dists, n_boot=1000, rng=rng)
        
        wb_results.append({
            "Space": sp,
            "Within": w_mean,
            "Between": b_mean,
            "Gap": gap_mean,
            "CI_2.5": gap_ci[0],
            "CI_97.5": gap_ci[1]
        })
        
        dist_samples[sp] = {
            "w_dists": w_dists,
            "b_dists": b_dists,
            "pairs": (w_pairs, b_pairs) # Важно: seed фиксирован, пары одинаковые!
        }
        
        # 3.4 Векторизация и Классификация
        print("Vectorizing diagrams for downstream task...")
        # Обучаем векторизатор на Train Real
        train_dgms = [track_features["dgm_real"][i] for i in tr_idx]
        vectorizer = persistence.DiagramVectorizer()
        vectorizer.fit(train_dgms, pers_cfg)
        
        # Векторизуем всё
        vec_real = np.array([vectorizer.transform(d) for d in track_features["dgm_real"]])
        vec_shuf = np.array([vectorizer.transform(d) for d in track_features["dgm_shuf"]])
        vec_rand = np.array([vectorizer.transform(d) for d in track_features["dgm_rand"]])
        vec_mean = np.array(track_features["mean_pool"])
        vec_concat = np.hstack([vec_real, vec_mean])
        
        y_train = labels[tr_idx]
        y_test = labels[te_idx]
        
        feature_sets = {
            "persistence": vec_real,
            "mean_pool": vec_mean,
            "persistence@shuffle": vec_shuf,
            "persistence@random": vec_rand,
            "concat(pers+mean)": vec_concat
        }
        
        for f_name, X in feature_sets.items():
            X_train, X_test = X[tr_idx], X[te_idx]
            
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            acc, acc_ci, f1, f1_ci = classification_bootstrap(y_test, y_pred, n_boot=1000, seed=cfg["seed"])
            
            clf_results.append({
                "Space": sp,
                "Features": f_name,
                "Accuracy": acc,
                "Acc_CI": f"[{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]",
                "Macro_F1": f1,
                "F1_CI": f"[{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]"
            })
            
            # Сохраняем лучшую матрицу ошибок только для persistence
            if f_name == "persistence":
                if f1 > best_f1:
                    best_f1 = f1
                    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
                    best_conf_matrix = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)
                    best_conf_labels = f"Best Channel: {sp} (Macro F1: {f1:.3f})"

    # 4. Mantel Test (на сэмплированных расстояниях)
    print("\n========== Running Mantel Test ==========")
    mantel_results = []
    sp_list = ["mert", "muq", "encodec"]
    # Объединяем w_dists и b_dists в один вектор для Mantel
    def get_full_dist_vector(sp_name):
        return np.concatenate([dist_samples[sp_name]["w_dists"], dist_samples[sp_name]["b_dists"]])
        
    for i in range(len(sp_list)):
        for j in range(i+1, len(sp_list)):
            sp1, sp2 = sp_list[i], sp_list[j]
            if sp1 in dist_samples and sp2 in dist_samples:
                v1 = get_full_dist_vector(sp1)
                v2 = get_full_dist_vector(sp2)
                # Корреляция Пирсона между массивами расстояний эквивалентна Mantel
                r = float(np.corrcoef(v1, v2)[0, 1])
                mantel_results.append({"Space 1": sp1, "Space 2": sp2, "Mantel_R": r})

    # 5. Сохранение таблиц
    pd.DataFrame(wb_results).to_csv(os.path.join(out_dir, "tables", "within_between_full.csv"), index=False)
    pd.DataFrame(clf_results).to_csv(os.path.join(out_dir, "tables", "classification_full.csv"), index=False)
    pd.DataFrame(mantel_results).to_csv(os.path.join(out_dir, "tables", "mantel_full.csv"), index=False)
    
    if best_conf_matrix is not None:
        best_conf_matrix.to_csv(os.path.join(out_dir, "tables", "confusion_best.csv"))
        with open(os.path.join(out_dir, "tables", "confusion_best_meta.txt"), "w") as f:
            f.write(best_conf_labels)

    print("\nALL TASKS COMPLETED SUCCESSFULLY. Results saved to results/tables/.")

if __name__ == "__main__":
    main()
