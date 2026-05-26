"""
test_pipeline.py
----------------
Интеграционный тест: cached embeddings -> preprocess -> pointcloud -> persistence.
Проверяет ВСЕ 4 канала (mert, muq, encodec, mir).

Структурные ассерты (обязаны быть верны независимо от данных):
  - T_tgt одинаковая у всех каналов после ресемпла
  - fps после ресемпла совпадает у всех
  - pca_dim = 32 у DL-каналов, 15 у MIR
  - takens cloud shape = (N, window*d)
  - вектор имеет фиксированную длину vector_dim
  - при takens_pca_dim=20 облако (N, 20)

Предупреждения (НЕ ассерты):
  - Пустая H1 → WARNING, не падать. Пустая H1 может быть честным
    свойством данных; решение принимает человек.

    python test_pipeline.py
"""
import sys; sys.path.insert(0, "src")
import os
import yaml
import numpy as np
import logging

import preprocess, pointcloud, persistence

logging.basicConfig(level=logging.INFO,
                    format="%(name)-18s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("test_pipeline")


def run_pipeline_test(cfg, takens_pca_dim_override=None):
    """Прогнать полный тест.

    takens_pca_dim_override: если не None, перезаписать cfg pointcloud.takens_pca_dim.
    """
    spaces = list(cfg["spaces"].keys())
    target_fps = cfg["common"]["target_fps"]
    clip_s = cfg["data"]["clip_seconds"]
    standardize = cfg["common"].get("standardize_features", True)

    label = f"takens_pca_dim={takens_pca_dim_override}" \
        if takens_pca_dim_override is not None else "default"
    sep = '#' * 60
    print(f"\n{sep}")
    print(f"# TEST RUN: {label}")
    print(sep)

    # ═══════════════════════════════════════════════════════════════
    # 1. Загрузка кэшированных эмбеддингов
    # ═══════════════════════════════════════════════════════════════
    raw = {}
    for space in spaces:
        path = os.path.join(cfg["paths"]["cache"], space, "test.npy")
        if not os.path.exists(path):
            logger.error("Cache not found: %s", path)
            sys.exit(1)
        raw[space] = np.load(path)
        logger.info("loaded %-8s  shape=%-14s dtype=%s",
                     space, str(raw[space].shape), raw[space].dtype)

    # ═══════════════════════════════════════════════════════════════
    # 2. Preprocess: resample → scaler → normalize → PCA (per channel)
    # ═══════════════════════════════════════════════════════════════
    preprocessed = {}
    reducers = {}

    for space in spaces:
        space_cfg = cfg["spaces"][space]
        native_fps = space_cfg.get("native_fps", target_fps)

        # Resample
        resampled = preprocess.resample_fps(raw[space], native_fps, target_fps)

        # PCAReducer: scaler → normalize → PCA bundled
        reducer = preprocess.PCAReducer(
            cfg["common"]["pca_dim"],
            standardize=standardize,
            normalize=cfg["common"]["normalize"])
        reducer.fit([resampled])
        reduced = reducer.transform(resampled)

        preprocessed[space] = reduced
        reducers[space] = reducer

        logger.info("%-8s  raw=(%d,%d) → resample=(%d,%d) → PCA=(%d,%d)  "
                     "explained=%.4f",
                     space,
                     raw[space].shape[0], raw[space].shape[1],
                     resampled.shape[0], resampled.shape[1],
                     reduced.shape[0], reduced.shape[1],
                     reducer.explained)

    # ═══════════════════════════════════════════════════════════════
    # 3. ASSERT: одинаковая T_tgt и fps у всех каналов
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("CHECK: same T_tgt / fps across channels")
    print("="*60)

    t_tgts = {s: preprocessed[s].shape[0] for s in spaces}
    ref_T = list(t_tgts.values())[0]

    for s in spaces:
        actual_fps = t_tgts[s] / clip_s
        print(f"  {s:8s}  T_tgt={t_tgts[s]:5d}  fps={actual_fps:.2f}")
        assert abs(t_tgts[s] - ref_T) <= 1, \
            f"T_tgt mismatch: {s}={t_tgts[s]} vs {spaces[0]}={ref_T}"
        assert abs(actual_fps - target_fps) < 2, \
            f"fps mismatch for {s}: expected ~{target_fps}, got {actual_fps:.2f}"

    print("  ✓ all channels have the same T_tgt and fps")

    # ═══════════════════════════════════════════════════════════════
    # 4. ASSERT: pca_dim = 32 для DL, min(d, 32) для MIR
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("CHECK: pca_dim per channel")
    print("="*60)

    for s in spaces:
        actual_d = preprocessed[s].shape[1]
        if s == "mir":
            expected_d = min(raw[s].shape[1], cfg["common"]["pca_dim"])
        else:
            expected_d = cfg["common"]["pca_dim"]
        print(f"  {s:8s}  pca_dim={actual_d}  (expected {expected_d})")
        assert actual_d == expected_d, \
            f"{s}: pca_dim={actual_d}, expected {expected_d}"

    print("  ✓ pca_dim correct for all channels")

    # ═══════════════════════════════════════════════════════════════
    # 5. Point clouds (takens) + optional second PCA + diagrams
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("TAKENS CLOUDS + PERSISTENCE DIAGRAMS")
    print("="*60)

    pc_cfg = dict(cfg["pointcloud"])
    pc_cfg["method"] = "takens"  # force takens

    # Определить takens_pca_dim
    tpca_dim = takens_pca_dim_override if takens_pca_dim_override is not None \
        else pc_cfg.get("takens_pca_dim")

    all_dgms_h1 = []
    all_clouds = []

    for space in spaces:
        x = preprocessed[space]
        cloud = pointcloud.build_cloud(x, pc_cfg)

        # ASSERT: shape = (N, window * d)
        d_dim = x.shape[1]
        expected_D = pc_cfg["window"] * d_dim
        assert cloud.shape[1] == expected_D, \
            f"{space}: cloud D={cloud.shape[1]}, expected window*d={expected_D}"

        T = x.shape[0]
        raw_N = (T - pc_cfg["window"]) // pc_cfg["stride"] + 1
        expected_N = min(raw_N, pc_cfg["n_points"])
        assert cloud.shape[0] == expected_N, \
            f"{space}: cloud N={cloud.shape[0]}, expected {expected_N}"

        all_clouds.append(cloud)
        print(f"\n  {space:8s}  cloud_raw=({cloud.shape[0]}, {cloud.shape[1]})")

    # Опциональный второй PCA после Такенса
    if tpca_dim is not None:
        print(f"\n  Applying Takens PCA → {tpca_dim}D")
        takens_reducers = {}
        for i, space in enumerate(spaces):
            tr = preprocess.PCAReducer(tpca_dim, standardize=False,
                                       normalize="none")
            tr.fit([all_clouds[i]])
            all_clouds[i] = tr.transform(all_clouds[i])
            takens_reducers[space] = tr
            print(f"  {space:8s}  cloud_pca=({all_clouds[i].shape[0]}, "
                  f"{all_clouds[i].shape[1]})  "
                  f"explained={tr.explained:.4f}")

            # ASSERT: second PCA output
            assert all_clouds[i].shape[1] == tpca_dim, \
                f"{space}: takens PCA dim={all_clouds[i].shape[1]}, " \
                f"expected {tpca_dim}"

    # Compute diagrams
    for i, space in enumerate(spaces):
        cloud = all_clouds[i]
        res = persistence.compute_diagrams(
            cloud, maxdim=cfg["persistence"]["maxdim"],
            thresh=cfg["persistence"]["persistence_threshold"])

        h0 = res["dgms"][0]
        h1 = res["dgms"][1]
        all_dgms_h1.append(h1)

        cloud_label = f"({cloud.shape[0]}, {cloud.shape[1]})"
        print(f"\n  {space:8s}  cloud={cloud_label}")
        print(f"           H0: {len(h0)} points")
        print(f"           H1: {len(h1)} points")

        if len(h1) == 0:
            logger.warning("⚠️  %s: H1 is EMPTY. This may be a legitimate "
                           "property of the data. Inspect manually.", space)
        else:
            lifetimes = h1[:, 1] - h1[:, 0]
            print(f"           H1 max persistence: {lifetimes.max():.4f}")
            print(f"           H1 mean persistence: {lifetimes.mean():.4f}")

    # ═══════════════════════════════════════════════════════════════
    # 6. DiagramVectorizer (fit на диаграммах ВСЕХ каналов)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("DIAGRAM VECTORIZER")
    print("="*60)

    vectorizer = persistence.DiagramVectorizer()
    vectorizer.fit(all_dgms_h1, cfg["persistence"])

    print(f"  vector_dim = {vectorizer.vector_dim}")
    print(f"  birth_range = ({vectorizer._birth_range[0]:.4f}, "
          f"{vectorizer._birth_range[1]:.4f})")
    print(f"  pers_range  = ({vectorizer._pers_range[0]:.4f}, "
          f"{vectorizer._pers_range[1]:.4f})")

    for i, space in enumerate(spaces):
        vec = vectorizer.transform(all_dgms_h1[i])

        # ASSERT: fixed length
        assert vec.shape == (vectorizer.vector_dim,), \
            f"{space}: vector shape={vec.shape}, expected ({vectorizer.vector_dim},)"

        print(f"  {space:8s}  vector shape={vec.shape}  "
              f"norm={np.linalg.norm(vec):.4f}  "
              f"nonzero={np.count_nonzero(vec)}")

    print("  ✓ all vectors have fixed length")

    # ═══════════════════════════════════════════════════════════════
    # 7. Explained variance summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PCA EXPLAINED VARIANCE SUMMARY")
    print("="*60)

    for space in spaces:
        r = reducers[space]
        evr = r.explained_per_component
        n = min(5, len(evr))
        top = ", ".join(f"{v:.4f}" for v in evr[:n])
        print(f"  {space:8s}  total={r.explained:.4f}  "
              f"PC00..PC{n-1:02d}=[{top}]")

    print("\n" + "="*60)
    print(f"✅ ALL CHECKS PASSED ({label})")
    print("="*60)


def main():
    cfg = yaml.safe_load(open("config.yaml"))

    # Run 1: default (takens_pca_dim=null → no second PCA)
    run_pipeline_test(cfg)

    # Run 2: takens_pca_dim=20 (second PCA enabled)
    print("\n\n")
    run_pipeline_test(cfg, takens_pca_dim_override=20)


if __name__ == "__main__":
    main()
