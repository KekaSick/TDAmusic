"""
fit_reducer.py
--------------
Обучить PCAReducer ОДИН раз на Spotify-кэше MuQ и сохранить через joblib.
Все скрипты (loop, cycle, classify, viz) должны ГРУЗИТЬ этот сохранённый
reducer, а НЕ обучать заново — это гарантирует воспроизводимость.

    .venv/bin/python scripts/fit_reducer.py
"""
import os
import sys
sys.path.insert(0, "src")
import glob
import yaml
import numpy as np
import preprocess

CACHE_DIR = "cache/muq_spotify90"
REDUCER_PATH = "cache/pca_reducer_muq_spotify90.joblib"


def main():
    cfg = yaml.safe_load(open("config.yaml"))

    print("Loading cached MuQ embeddings...")
    all_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npy")))
    all_embs = [np.load(f) for f in all_files]
    print(f"  Loaded {len(all_embs)} tracks, "
          f"total frames: {sum(e.shape[0] for e in all_embs)}")

    print("Fitting PCAReducer (StandardScaler → unit_sphere → PCA)...")
    reducer = preprocess.PCAReducer(
        dim=cfg["spaces"]["muq"].get("pca_dim", cfg["common"]["pca_dim"]),
        standardize=cfg["common"].get("standardize_features", True),
        normalize=cfg["common"]["normalize"],
    )
    reducer.fit(all_embs)
    print(f"  PCA dim={reducer.dim}, explained={reducer.explained:.6f}")

    # Verify determinism: fit again, compare components
    reducer2 = preprocess.PCAReducer(
        dim=cfg["spaces"]["muq"].get("pca_dim", cfg["common"]["pca_dim"]),
        standardize=cfg["common"].get("standardize_features", True),
        normalize=cfg["common"]["normalize"],
    )
    reducer2.fit(all_embs)
    diff = np.max(np.abs(reducer.pca.components_ - reducer2.pca.components_))
    print(f"  Determinism check: max |components1 - components2| = {diff:.2e}")
    assert diff == 0.0, f"PCA NOT deterministic! diff={diff}"
    print("  ✓ PCA is deterministic (bit-identical)")

    # Save
    os.makedirs(os.path.dirname(REDUCER_PATH), exist_ok=True)
    reducer.save(REDUCER_PATH)
    fsize = os.path.getsize(REDUCER_PATH) / 1024
    print(f"\n→ Saved: {REDUCER_PATH} ({fsize:.0f} KB)")

    # Verify load
    r_loaded = preprocess.PCAReducer.load(REDUCER_PATH)
    diff2 = np.max(np.abs(reducer.pca.components_ - r_loaded.pca.components_))
    assert diff2 == 0.0, f"Save/load mismatch! diff={diff2}"
    print("  ✓ Load verified (bit-identical)")


if __name__ == "__main__":
    main()
