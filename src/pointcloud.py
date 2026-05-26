"""
pointcloud.py
-------------
Превращение последовательности кадров в облако точек.

Две опции (обе нужны):
  * takens : sliding-window embedding. Порядок кадров -> геометрия облака.
             ТОЛЬКО здесь shuffle осмыслен. Периодичность периода T ->
             петля H1 (теорема Такенса).
  * direct : кадры как облако напрямую. Vietoris-Rips ИНВАРИАНТЕН к порядку
             точек, поэтому shuffle на direct ничего не меняет — это важный
             дидактический факт, а не баг.

Прореживание (subsample) до n_points: гомологии в высокой размерности на
длинных рядах нерешаемы. maxmin сохраняет геометрию лучше uniform.
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
    """Жадный maxmin (farthest point sampling) — равномерное покрытие."""
    rng = np.random.default_rng(seed)
    chosen = [rng.integers(cloud.shape[0])]
    d = np.linalg.norm(cloud - cloud[chosen[0]], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(cloud - cloud[i], axis=1))
    return cloud[chosen]


# ---- Варианты С ВОЗВРАТОМ ИНДЕКСОВ (для интерпретации cocycles) ----

def _maxmin_with_indices(cloud, n, seed=0):
    """Жадный maxmin — возвращает (subcloud, indices в исходном облаке)."""
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
    """Как subsample(), но возвращает (subcloud, indices)."""
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
    """Как build_cloud(), но возвращает (cloud, takens_starts, sub_indices).

    Позволяет протащить индекс точки обратно к секунде аудио:
      cocycle vertex i  →  sub_indices[i]  →  строка Такенс-облака
      →  takens_starts[sub_indices[i]]  →  стартовый кадр окна в PCA-ряду.

    takens_starts: массив стартовых кадров для каждой строки полного
                   Такенс-облака (до прореживания). Для direct — np.arange(T).
    sub_indices:   индексы выбранных строк в полном Такенс-/direct-облаке.
    """
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
