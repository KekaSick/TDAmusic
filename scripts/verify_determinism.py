"""
verify_determinism.py
---------------------
Run the classification pipeline twice on files[:40] and verify
bit-for-bit identical results. Uses a single space (muq) to keep it fast.
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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


def run_once(cfg, files, labels, spaces, out_path):
    """One full run of the classification pipeline, saving CSV to out_path."""
    pc_cfg = dict(cfg["pointcloud"])
    pc_cfg["takens_pca_dim"] = None
    pers_cfg = cfg["persistence"]

    seed = cfg["seed"]
    n_shuffle = cfg["controls"]["shuffle_repeats"]
    match_autocorr = cfg["controls"]["random_match_autocorr"]
    iaaft_n_iter = cfg["controls"].get("iaaft_n_iter", 200)
    iaaft_tol = cfg["controls"].get("iaaft_tol", 1e-8)

    tr_idx, te_idx = train_test_split(
        np.arange(len(files)), test_size=cfg["data"]["test_size"],
        stratify=labels, random_state=seed)
    y_train, y_test = labels[tr_idx], labels[te_idx]

    clf_results = []

    for sp in spaces:
        print(f"  [{os.path.basename(out_path)}] Space: {sp}")

        reducer = preprocess.PCAReducer(dim=cfg["common"]["pca_dim"])
        train_frames = [embed_spaces.extract(files[i], sp, cfg) for i in tr_idx]
        reducer.fit(train_frames, batch_size=20)
        del train_frames

        dgm_real = []
        dgm_shuf_repeats = []
        dgm_rand_repeats = []
        dgm_iaaft_repeats = []
        mean_pool = []

        for i in range(len(files)):
            x = embed_spaces.extract(files[i], sp, cfg)
            x_pca = preprocess.prepare_track(x, cfg["spaces"].get(sp, {}), cfg["common"], reducer)
            mean_pool.append(np.mean(x_pca, axis=0))

            cloud = pointcloud.build_cloud(x_pca, pc_cfg)
            res = persistence.compute_diagrams(cloud, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
            dgm_real.append(res["dgms"][1] if len(res["dgms"]) > 1 else np.empty((0, 2)))

            track_shuf = []
            for k in range(n_shuffle):
                shuffle_rng = np.random.default_rng(seed * 1000 + i * 100 + k)
                x_s = controls.shuffle_frames(x_pca, shuffle_rng)
                cloud_s = pointcloud.build_cloud(x_s, pc_cfg)
                res_s = persistence.compute_diagrams(cloud_s, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
                track_shuf.append(res_s["dgms"][1] if len(res_s["dgms"]) > 1 else np.empty((0, 2)))
            dgm_shuf_repeats.append(track_shuf)

            track_rand = []
            for k in range(n_shuffle):
                random_rng = np.random.default_rng(seed * 2000 + i * 100 + k)
                x_r = controls.random_like(x_pca, random_rng, match_autocorr=match_autocorr)
                cloud_r = pointcloud.build_cloud(x_r, pc_cfg)
                res_r = persistence.compute_diagrams(cloud_r, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
                track_rand.append(res_r["dgms"][1] if len(res_r["dgms"]) > 1 else np.empty((0, 2)))
            dgm_rand_repeats.append(track_rand)

            track_iaaft = []
            for k in range(n_shuffle):
                iaaft_rng = np.random.default_rng(seed * 3000 + i * 100 + k)
                x_ia = controls.iaaft_surrogate(x_pca, iaaft_rng, n_iter=iaaft_n_iter, tol=iaaft_tol)
                cloud_ia = pointcloud.build_cloud(x_ia, pc_cfg)
                res_ia = persistence.compute_diagrams(cloud_ia, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
                track_iaaft.append(res_ia["dgms"][1] if len(res_ia["dgms"]) > 1 else np.empty((0, 2)))
            dgm_iaaft_repeats.append(track_iaaft)

        # Vectorize
        train_dgms = [dgm_real[i] for i in tr_idx]
        vectorizer = persistence.DiagramVectorizer()
        vectorizer.fit(train_dgms, pers_cfg)

        vec_real = np.array([vectorizer.transform(d) for d in dgm_real])
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

        for f_name, X in feature_sets.items():
            X_train, X_test = X[tr_idx], X[te_idx]
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc, acc_ci, f1, f1_ci = classification_bootstrap(
                y_test, y_pred, n_boot=1000, seed=seed)

            clf_results.append({
                "Space": sp, "Features": f_name,
                "Accuracy": round(acc, 4),
                "Acc_CI": f"[{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]",
                "Macro_F1": round(f1, 4),
                "F1_CI": f"[{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]",
            })

    df = pd.DataFrame(clf_results)
    df.to_csv(out_path, index=False)
    return df


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    meta = pd.read_csv(cfg["paths"]["meta"])

    # Stratified sample of 40 tracks (4 per genre) to ensure multi-class
    # Can't use plain [:40] — data is sorted by genre.
    rng_sample = np.random.default_rng(cfg["seed"])
    sample_idx = []
    for genre in sorted(meta["genre"].unique()):
        genre_idx = meta.index[meta["genre"] == genre].tolist()
        chosen = rng_sample.choice(genre_idx, size=min(4, len(genre_idx)), replace=False)
        sample_idx.extend(chosen.tolist())
    sample_idx.sort()
    meta = meta.loc[sample_idx].reset_index(drop=True)
    files = [os.path.join(cfg["paths"]["audio"], f) for f in meta["filename"]]
    labels = meta["genre"].values
    print(f"Stratified slice: {len(files)} tracks, genres: {dict(pd.Series(labels).value_counts())}")

    # Use a single space for speed
    spaces = ["muq"]

    os.makedirs("results/tables", exist_ok=True)
    path_1 = "results/tables/_determinism_run1.csv"
    path_2 = "results/tables/_determinism_run2.csv"

    print("=" * 60)
    print("  RUN 1")
    print("=" * 60)
    df1 = run_once(cfg, files, labels, spaces, path_1)

    print()
    print("=" * 60)
    print("  RUN 2")
    print("=" * 60)
    df2 = run_once(cfg, files, labels, spaces, path_2)

    print()
    print("=" * 60)
    print("  COMPARISON")
    print("=" * 60)

    # Read back from disk to ensure CSV serialization is identical
    text1 = open(path_1).read()
    text2 = open(path_2).read()

    if text1 == text2:
        print("  ✅ IDENTICAL — two runs produced bit-for-bit same CSV")
        print()
        print(df1.to_string(index=False))
    else:
        print("  ❌ MISMATCH — results differ between runs!")
        print()
        print("Run 1:")
        print(df1.to_string(index=False))
        print()
        print("Run 2:")
        print(df2.to_string(index=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
