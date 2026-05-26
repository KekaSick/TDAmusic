"""
check_determinism.py
--------------------
Проверка абсолютного детерминизма пайплайна.
1. Проверяет облако одного трека дважды.
2. Проверяет `sampled_distances` (within/between) дважды.
"""
import os
import sys
import yaml
import numpy as np

sys.path.insert(0, "src")
import preprocess
import pointcloud
import persistence
import distances

def load_cfg():
    return yaml.safe_load(open("config.yaml"))

def test_pointcloud_determinism():
    print("Testing pointcloud determinism (maxmin should be disabled)...")
    cfg = load_cfg()
    assert cfg["pointcloud"]["subsample"] == "none", "subsample must be 'none'"
    
    # Сгенерируем синтетический PCA-ряд
    np.random.seed(42)
    x_pca = np.random.randn(2000, 32)
    
    cloud1, _, _ = pointcloud.build_cloud_with_indices(x_pca, cfg["pointcloud"])
    cloud2, _, _ = pointcloud.build_cloud_with_indices(x_pca, cfg["pointcloud"])
    
    assert np.array_equal(cloud1, cloud2), "Cloud is NOT deterministic!"
    print("  ✓ Pointcloud is bit-identical")

def test_distance_determinism():
    print("Testing distance sampling determinism...")
    cfg = load_cfg()
    
    np.random.seed(42)
    # 5 диаграмм, у каждой по 3 точки
    dgms = [np.random.rand(3, 2) for _ in range(5)]
    labels = np.array([0, 0, 1, 1, 0])
    
    w_dists1, b_dists1, w_pairs1, b_pairs1 = distances.sampled_distances(
        dgms, labels, n_within=10, n_between=10, seed=cfg["seed"]
    )
    
    w_dists2, b_dists2, w_pairs2, b_pairs2 = distances.sampled_distances(
        dgms, labels, n_within=10, n_between=10, seed=cfg["seed"]
    )
    
    assert np.array_equal(w_dists1, w_dists2), "Within dists NOT deterministic!"
    assert np.array_equal(b_dists1, b_dists2), "Between dists NOT deterministic!"
    assert np.array_equal(w_pairs1, w_pairs2), "Within pairs NOT deterministic!"
    assert np.array_equal(b_pairs1, b_pairs2), "Between pairs NOT deterministic!"
    print("  ✓ Distance sampling is bit-identical")

if __name__ == "__main__":
    test_pointcloud_determinism()
    test_distance_determinism()
    print("All determinism tests passed!")
