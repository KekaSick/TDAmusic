"""
run_fix_vectorization.py
------------------------
Диагностика + фикс вырожденной persistence-image векторизации.
Работает ТОЛЬКО по кэшу эмбеддингов — не переизвлекает.
Пересчитывает диаграммы и классификацию с разными схемами векторизации.
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import embed_spaces
import preprocess
import pointcloud
import persistence
import controls


def eval_f1(X_train, y_train, X_test, y_test):
    """Macro-F1 одного прогона LogisticRegression."""
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return f1_score(y_test, y_pred, average="macro")


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    meta = pd.read_csv(cfg["paths"]["meta"])
    files = [os.path.join(cfg["paths"]["audio"], f) for f in meta["filename"]]
    labels = meta["genre"].values

    pc_cfg = dict(cfg["pointcloud"])
    pc_cfg["takens_pca_dim"] = None
    pers_cfg = cfg["persistence"]
    rng = np.random.default_rng(cfg["seed"])

    tr_idx, te_idx = train_test_split(
        np.arange(len(files)), test_size=cfg["data"]["test_size"],
        stratify=labels, random_state=cfg["seed"])

    y_train = labels[tr_idx]
    y_test = labels[te_idx]

    spaces = ["muq", "mert", "encodec", "mir"]
    results = []

    for sp in spaces:
        print(f"\n{'='*60}")
        print(f"  Space: {sp.upper()}")
        print(f"{'='*60}")

        # --- PCA ---
        reducer = preprocess.PCAReducer(dim=cfg["common"]["pca_dim"])
        train_frames = [embed_spaces.extract(files[i], sp, cfg) for i in tr_idx]
        reducer.fit(train_frames, batch_size=20)
        del train_frames

        # --- Диаграммы ---
        dgm_real = []
        dgm_shuf = []
        dgm_rand = []
        mean_pool = []

        for i in tqdm(range(len(files)), desc=f"Diagrams {sp}", unit="track", leave=False):
            x = embed_spaces.extract(files[i], sp, cfg)
            x_pca = preprocess.prepare_track(x, cfg["spaces"].get(sp, {}), cfg["common"], reducer)
            mean_pool.append(np.mean(x_pca, axis=0))

            cloud = pointcloud.build_cloud(x_pca, pc_cfg)
            res = persistence.compute_diagrams(cloud, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
            dgm_real.append(res["dgms"][1] if len(res["dgms"]) > 1 else np.empty((0, 2)))

            x_s = controls.shuffle_frames(x_pca, rng)
            cloud_s = pointcloud.build_cloud(x_s, pc_cfg)
            res_s = persistence.compute_diagrams(cloud_s, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
            dgm_shuf.append(res_s["dgms"][1] if len(res_s["dgms"]) > 1 else np.empty((0, 2)))

            x_r = controls.random_like(x_pca, rng, match_autocorr=True)
            cloud_r = pointcloud.build_cloud(x_r, pc_cfg)
            res_r = persistence.compute_diagrams(cloud_r, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
            dgm_rand.append(res_r["dgms"][1] if len(res_r["dgms"]) > 1 else np.empty((0, 2)))

        train_dgms = [dgm_real[i] for i in tr_idx]

        # ======= ДИАГНОСТИКА (только для muq) =======
        if sp == "muq":
            print("\n--- DIAGNOSTIC: image vectorizer (old) ---")
            vec_old = persistence.DiagramVectorizer()
            vec_old.fit(train_dgms, pers_cfg)
            X_old = np.array([vec_old.transform(d) for d in dgm_real])
            stds = X_old.std(axis=0)
            print(f"  vector_dim     : {X_old.shape[1]}")
            print(f"  const pixels   : {(stds < 1e-6).sum()} / {X_old.shape[1]}")
            print(f"  mean std       : {stds.mean():.6f}")
            print(f"  max std        : {stds.max():.6f}")
            # доля энергии в топ-5 пикселях
            total_var = (stds**2).sum()
            top5 = np.sort(stds**2)[-5:].sum()
            print(f"  top-5 var share : {top5/total_var:.3f}" if total_var > 0 else "  total_var = 0!")
            print(f"  birth_range    : {vec_old._birth_range}")
            print(f"  pers_range     : {vec_old._pers_range}")
            b_w = vec_old._birth_range[1] - vec_old._birth_range[0]
            p_w = vec_old._pers_range[1] - vec_old._pers_range[0]
            print(f"  birth width    : {b_w:.4f}")
            print(f"  pers width     : {p_w:.4f}")
            print(f"  pixel_size(old): {max(b_w,p_w)/20:.6f}")
            print(f"  eff birth px   : {b_w / (max(b_w,p_w)/20):.1f}")
            print(f"  eff pers px    : {p_w / (max(b_w,p_w)/20):.1f}")

        # ======= ВАРИАНТ 1: BETTI =======
        print(f"\n  [BETTI] Vectorizing...")
        pers_betti = dict(pers_cfg)
        pers_betti["vectorization"] = "betti"
        vec_betti = persistence.DiagramVectorizer()
        vec_betti.fit(train_dgms, pers_betti)

        Xr_b = np.array([vec_betti.transform(d) for d in dgm_real])
        Xs_b = np.array([vec_betti.transform(d) for d in dgm_shuf])
        Xn_b = np.array([vec_betti.transform(d) for d in dgm_rand])

        f1_real_b = eval_f1(Xr_b[tr_idx], y_train, Xr_b[te_idx], y_test)
        f1_shuf_b = eval_f1(Xs_b[tr_idx], y_train, Xs_b[te_idx], y_test)
        f1_rand_b = eval_f1(Xn_b[tr_idx], y_train, Xn_b[te_idx], y_test)
        print(f"  [BETTI]  F1: real={f1_real_b:.3f}  shuffle={f1_shuf_b:.3f}  random={f1_rand_b:.3f}")

        results.append({"space": sp, "method": "betti",
                        "F1_real": f1_real_b, "F1_shuffle": f1_shuf_b, "F1_random": f1_rand_b})

        # ======= ВАРИАНТ 2: IMAGE FIXED =======
        print(f"  [IMAGE_FIX] Vectorizing...")
        pers_img = dict(pers_cfg)
        pers_img["vectorization"] = "image_fix"
        vec_img = persistence.DiagramVectorizer()
        vec_img.fit(train_dgms, pers_img)

        Xr_i = np.array([vec_img.transform(d) for d in dgm_real])
        Xs_i = np.array([vec_img.transform(d) for d in dgm_shuf])
        Xn_i = np.array([vec_img.transform(d) for d in dgm_rand])

        f1_real_i = eval_f1(Xr_i[tr_idx], y_train, Xr_i[te_idx], y_test)
        f1_shuf_i = eval_f1(Xs_i[tr_idx], y_train, Xs_i[te_idx], y_test)
        f1_rand_i = eval_f1(Xn_i[tr_idx], y_train, Xn_i[te_idx], y_test)
        print(f"  [IMAGE_FIX]  F1: real={f1_real_i:.3f}  shuffle={f1_shuf_i:.3f}  random={f1_rand_i:.3f}")

        results.append({"space": sp, "method": "image_fix",
                        "F1_real": f1_real_i, "F1_shuffle": f1_shuf_i, "F1_random": f1_rand_i})

        # ======= IMAGE (old, для сравнения) =======
        print(f"  [IMAGE_OLD] Vectorizing...")
        vec_old2 = persistence.DiagramVectorizer()
        vec_old2.fit(train_dgms, pers_cfg)  # vectorization: "image"

        Xr_o = np.array([vec_old2.transform(d) for d in dgm_real])
        Xs_o = np.array([vec_old2.transform(d) for d in dgm_shuf])
        Xn_o = np.array([vec_old2.transform(d) for d in dgm_rand])

        f1_real_o = eval_f1(Xr_o[tr_idx], y_train, Xr_o[te_idx], y_test)
        f1_shuf_o = eval_f1(Xs_o[tr_idx], y_train, Xs_o[te_idx], y_test)
        f1_rand_o = eval_f1(Xn_o[tr_idx], y_train, Xn_o[te_idx], y_test)
        print(f"  [IMAGE_OLD]  F1: real={f1_real_o:.3f}  shuffle={f1_shuf_o:.3f}  random={f1_rand_o:.3f}")

        results.append({"space": sp, "method": "image_old",
                        "F1_real": f1_real_o, "F1_shuffle": f1_shuf_o, "F1_random": f1_rand_o})

    df = pd.DataFrame(results)
    out_path = os.path.join(cfg["paths"]["results"], "tables", "vectorization_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to {out_path}")
    print(f"{'='*60}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
