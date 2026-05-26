"""
run_classify_betti.py
---------------------
Финальная классификация с betti-векторизацией.
НЕ переизвлекает эмбеддинги — работает по кэшу.
Пересчитывает диаграммы, векторизует betti-кривой, классифицирует с bootstrap-CI.
Обновляет classification_full.csv и confusion_best.csv.

Два классификатора:
  - logreg: LogisticRegression (линейный baseline)
  - hgb:    HistGradientBoostingClassifier (нелинейный, gradient boosting)
Оба обучаются на тех же feature_sets и train/test split.
Колонка "classifier" различает строки в classification_full.csv.

Воспроизводимость:
  Каждый контроль для трека i, повтора k получает ИЗОЛИРОВАННЫЙ seed:
    shuffle -> default_rng(seed*1000 + i*100 + k)
    random  -> default_rng(seed*2000 + i*100 + k)
    iaaft   -> default_rng(seed*3000 + i*100 + k)
  Betti-вектор контроля = среднее по K=shuffle_repeats реализациям.
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import embed_spaces
import preprocess
import pointcloud
import persistence
import controls


def classification_bootstrap(y_true, y_pred, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    return (np.mean(accs), (np.percentile(accs, 2.5), np.percentile(accs, 97.5)),
            np.mean(f1s),  (np.percentile(f1s, 2.5),  np.percentile(f1s, 97.5)))


def _build_classifiers(seed):
    """Return dict {name: classifier} for all classifiers to evaluate."""
    return {
        "logreg": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed),
        "hgb": HistGradientBoostingClassifier(
            max_iter=200, random_state=seed),
    }


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    meta = pd.read_csv(cfg["paths"]["meta"])
    files = [os.path.join(cfg["paths"]["audio"], f) for f in meta["filename"]]
    labels = meta["genre"].values

    pc_cfg = dict(cfg["pointcloud"])
    pc_cfg["takens_pca_dim"] = None  # baseline
    pers_cfg = cfg["persistence"]
    assert pers_cfg["vectorization"] == "betti", \
        f"Expected betti, got {pers_cfg['vectorization']}. Fix config.yaml!"

    seed = cfg["seed"]
    n_shuffle = cfg["controls"]["shuffle_repeats"]
    match_autocorr = cfg["controls"]["random_match_autocorr"]
    iaaft_n_iter = cfg["controls"].get("iaaft_n_iter", 200)
    iaaft_tol = cfg["controls"].get("iaaft_tol", 1e-8)

    tr_idx, te_idx = train_test_split(
        np.arange(len(files)), test_size=cfg["data"]["test_size"],
        stratify=labels, random_state=seed)
    y_train, y_test = labels[tr_idx], labels[te_idx]

    spaces = ["muq", "mert", "encodec", "mir"]
    clf_results = []
    best_f1 = -1
    best_conf_matrix = None
    best_conf_labels = None

    out_dir = os.path.join(cfg["paths"]["results"], "tables")
    os.makedirs(out_dir, exist_ok=True)

    for sp in spaces:
        print(f"\n{'='*60}")
        print(f"  Space: {sp.upper()}")
        print(f"{'='*60}")

        # --- PCA (fit on train) ---
        reducer = preprocess.PCAReducer(dim=cfg["common"]["pca_dim"])
        train_frames = [embed_spaces.extract(files[i], sp, cfg) for i in tr_idx]
        reducer.fit(train_frames, batch_size=20)
        del train_frames

        # --- Diagrams ---
        dgm_real = []
        # Controls: accumulate betti vectors per repeat, then average
        dgm_shuf_repeats = []   # list of lists: [[dgm_k0, dgm_k1, ...] per track]
        dgm_rand_repeats = []
        dgm_iaaft_repeats = []
        mean_pool = []

        for i in tqdm(range(len(files)), desc=f"Diagrams {sp}", unit="track", leave=False):
            x = embed_spaces.extract(files[i], sp, cfg)
            x_pca = preprocess.prepare_track(x, cfg["spaces"].get(sp, {}), cfg["common"], reducer)
            mean_pool.append(np.mean(x_pca, axis=0))

            # --- Real ---
            cloud = pointcloud.build_cloud(x_pca, pc_cfg)
            res = persistence.compute_diagrams(cloud, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
            dgm_real.append(res["dgms"][1] if len(res["dgms"]) > 1 else np.empty((0, 2)))

            # --- Shuffle (K repeats, isolated seeds) ---
            track_shuf_dgms = []
            for k in range(n_shuffle):
                shuffle_rng = np.random.default_rng(seed * 1000 + i * 100 + k)
                x_s = controls.shuffle_frames(x_pca, shuffle_rng)
                cloud_s = pointcloud.build_cloud(x_s, pc_cfg)
                res_s = persistence.compute_diagrams(cloud_s, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
                track_shuf_dgms.append(res_s["dgms"][1] if len(res_s["dgms"]) > 1 else np.empty((0, 2)))
            dgm_shuf_repeats.append(track_shuf_dgms)

            # --- Random (K repeats, isolated seeds) ---
            track_rand_dgms = []
            for k in range(n_shuffle):
                random_rng = np.random.default_rng(seed * 2000 + i * 100 + k)
                x_r = controls.random_like(x_pca, random_rng, match_autocorr=match_autocorr)
                cloud_r = pointcloud.build_cloud(x_r, pc_cfg)
                res_r = persistence.compute_diagrams(cloud_r, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
                track_rand_dgms.append(res_r["dgms"][1] if len(res_r["dgms"]) > 1 else np.empty((0, 2)))
            dgm_rand_repeats.append(track_rand_dgms)

            # --- IAAFT (K repeats, isolated seeds) ---
            track_iaaft_dgms = []
            for k in range(n_shuffle):
                iaaft_rng = np.random.default_rng(seed * 3000 + i * 100 + k)
                x_ia = controls.iaaft_surrogate(x_pca, iaaft_rng, n_iter=iaaft_n_iter, tol=iaaft_tol)
                cloud_ia = pointcloud.build_cloud(x_ia, pc_cfg)
                res_ia = persistence.compute_diagrams(cloud_ia, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
                track_iaaft_dgms.append(res_ia["dgms"][1] if len(res_ia["dgms"]) > 1 else np.empty((0, 2)))
            dgm_iaaft_repeats.append(track_iaaft_dgms)

        # --- Vectorize (betti) ---
        print("Vectorizing (betti)...")
        train_dgms = [dgm_real[i] for i in tr_idx]
        vectorizer = persistence.DiagramVectorizer()
        vectorizer.fit(train_dgms, pers_cfg)

        vec_real = np.array([vectorizer.transform(d) for d in dgm_real])

        # Controls: average betti vector over K repeats per track
        vec_shuf = np.array([
            np.mean([vectorizer.transform(d) for d in dgm_shuf_repeats[i]], axis=0)
            for i in range(len(files))
        ])
        vec_rand = np.array([
            np.mean([vectorizer.transform(d) for d in dgm_rand_repeats[i]], axis=0)
            for i in range(len(files))
        ])
        vec_iaaft = np.array([
            np.mean([vectorizer.transform(d) for d in dgm_iaaft_repeats[i]], axis=0)
            for i in range(len(files))
        ])

        vec_mean = np.array(mean_pool)
        vec_concat = np.hstack([vec_real, vec_mean])

        feature_sets = {
            "persistence": vec_real,
            "mean_pool": vec_mean,
            "persistence@shuffle": vec_shuf,
            "persistence@random": vec_rand,
            "persistence@iaaft": vec_iaaft,
            "concat(pers+mean)": vec_concat,
        }

        classifiers = _build_classifiers(seed)

        for f_name, X in feature_sets.items():
            X_train, X_test = X[tr_idx], X[te_idx]

            for clf_name, clf in classifiers.items():
                # Fresh clone for each (feature_set, classifier) combo
                clf_inst = clone(clf)
                clf_inst.fit(X_train, y_train)
                y_pred = clf_inst.predict(X_test)

                acc, acc_ci, f1, f1_ci = classification_bootstrap(
                    y_test, y_pred, n_boot=1000, seed=seed)

                print(f"  [{clf_name:6s}] {f_name:25s}  acc={acc:.3f} "
                      f"[{acc_ci[0]:.3f},{acc_ci[1]:.3f}]  "
                      f"F1={f1:.3f} [{f1_ci[0]:.3f},{f1_ci[1]:.3f}]")

                clf_results.append({
                    "Space": sp, "Features": f_name,
                    "classifier": clf_name,
                    "Accuracy": round(acc, 4),
                    "Acc_CI": f"[{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]",
                    "Macro_F1": round(f1, 4),
                    "F1_CI": f"[{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]",
                })

                # Лучший канал = max F1 на persistence (across all classifiers)
                if f_name == "persistence" and f1 > best_f1:
                    best_f1 = f1
                    cm = confusion_matrix(y_test, y_pred,
                                          labels=clf_inst.classes_)
                    best_conf_matrix = pd.DataFrame(
                        cm, index=clf_inst.classes_, columns=clf_inst.classes_)
                    best_conf_labels = (f"Best Channel: {sp} "
                                        f"(classifier={clf_name}, "
                                        f"Macro F1: {f1:.3f})")

    # --- Save ---
    pd.DataFrame(clf_results).to_csv(
        os.path.join(out_dir, "classification_full.csv"), index=False)
    if best_conf_matrix is not None:
        best_conf_matrix.to_csv(os.path.join(out_dir, "confusion_best.csv"))
        with open(os.path.join(out_dir, "confusion_best_meta.txt"), "w") as f:
            f.write(best_conf_labels)

    print(f"\n{'='*60}")
    print(f"  DONE. Results saved to {out_dir}/")
    print(f"  Best channel: {best_conf_labels}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
