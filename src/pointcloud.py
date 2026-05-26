"""
pointcloud.py
-------------
Frame sequence to point cloud utilities.

Takens embedding moves temporal order into geometry. Direct clouds are kept for
visualization and diagnostics; Vietoris-Rips is invariant to point order there.
"""
from __future__ import annotations
import numpy as np


def takens(x: np.ndarray, window: int, stride: int) -> np.ndarray:
    """(T, d) -> (N, window*d). N = (T - window)//stride + 1."""
    T, d = x.shape
    idx = range(0, T - window + 1, stride)
    return np.stack([x[i:i + window].reshape(-1) for i in idx], axis=0)


def direct(x: np.ndarray) -> np.ndarray:
    return x


def build_cloud(x: np.ndarray, pc_cfg) -> np.ndarray:
    if pc_cfg["method"] == "takens":
        cloud = takens(x, pc_cfg["window"], pc_cfg["stride"])
    elif pc_cfg["method"] == "direct":
        cloud = direct(x)
    else:
        raise ValueError(pc_cfg["method"])
    return subsample(cloud, pc_cfg["n_points"], pc_cfg["subsample"])


def subsample(cloud: np.ndarray, n: int, mode: str) -> np.ndarray:
    if cloud.shape[0] <= n:
        return cloud
    if mode == "none":
        return cloud
    if mode == "uniform":
        idx = np.linspace(0, cloud.shape[0] - 1, n).astype(int)
        return cloud[idx]
    if mode == "maxmin":
        return _maxmin(cloud, n)
    raise ValueError(mode)


def _maxmin(cloud, n, seed=0):
    """Greedy farthest-point sampling."""
    rng = np.random.default_rng(seed)
    chosen = [rng.integers(cloud.shape[0])]
    d = np.linalg.norm(cloud - cloud[chosen[0]], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(cloud - cloud[i], axis=1))
    return cloud[chosen]


def _maxmin_with_indices(cloud, n, seed=0):
    """Greedy farthest-point sampling with source indices."""
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(cloud.shape[0]))]
    d = np.linalg.norm(cloud - cloud[chosen[0]], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(cloud - cloud[i], axis=1))
    idx = np.array(chosen)
    return cloud[idx], idx


def subsample_with_indices(cloud, n, mode):
    """Subsample while preserving source indices."""
    if cloud.shape[0] <= n:
        return cloud, np.arange(cloud.shape[0])
    if mode == "none":
        return cloud, np.arange(cloud.shape[0])
    if mode == "uniform":
        idx = np.linspace(0, cloud.shape[0] - 1, n).astype(int)
        return cloud[idx], idx
    if mode == "maxmin":
        return _maxmin_with_indices(cloud, n)
    raise ValueError(mode)


def build_cloud_with_indices(x, pc_cfg):
    """Build a cloud and keep the mapping back to frame indices."""
    if pc_cfg["method"] == "takens":
        window = pc_cfg["window"]
        stride = pc_cfg["stride"]
        cloud_full = takens(x, window, stride)
        T = x.shape[0]
        takens_starts = np.arange(0, T - window + 1, stride)
    elif pc_cfg["method"] == "direct":
        cloud_full = direct(x)
        takens_starts = np.arange(cloud_full.shape[0])
    else:
        raise ValueError(pc_cfg["method"])

    cloud_sub, sub_indices = subsample_with_indices(
        cloud_full, pc_cfg["n_points"], pc_cfg["subsample"])
    return cloud_sub, takens_starts, sub_indices
