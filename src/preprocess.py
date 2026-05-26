"""
preprocess.py
-------------
Shared preprocessing for all representation spaces.

Order: resample frames, scale features, L2-normalize frames, then PCA. UMAP is
only for visualization and is never passed to persistent homology.
"""
from __future__ import annotations
import logging
import numpy as np
from scipy.signal import resample
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


def resample_fps(x: np.ndarray, src_fps: float, tgt_fps: float) -> np.ndarray:
    """Fourier-resample a frame sequence along time."""
    if abs(src_fps - tgt_fps) < 1e-6:
        return x
    t_tgt = max(1, int(round(x.shape[0] * tgt_fps / src_fps)))
    return resample(x, t_tgt, axis=0)


def normalize_frames(x: np.ndarray, mode: str = "unit_sphere") -> np.ndarray:
    if mode == "unit_sphere":
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return x / n
    if mode == "none":
        return x
    raise ValueError(mode)


class PCAReducer:
    """Fit StandardScaler + PCA on train frames and reuse them everywhere."""
    def __init__(self, dim: int, standardize: bool = True,
                 normalize: str = "unit_sphere", scaler_type: str = "standard"):
        self._requested_dim = dim
        self._standardize = standardize
        self._normalize = normalize
        self._scaler_type = scaler_type
        self.scaler = None
        self.pca = None

    def fit(self, frames_list: list[np.ndarray], batch_size: int = 50):
        """Fit StandardScaler + PCA on all frames (deterministic).

        batch_size is kept for API compatibility but ignored — full-batch PCA
        is used for determinism (IncrementalPCA depended on batch order).
        """
        all_frames = np.concatenate(frames_list, axis=0)
        actual_dim = min(self._requested_dim, all_frames.shape[1])
        if actual_dim < self._requested_dim:
            logger.warning(
                "PCA: requested %d components but data has %d features → using %d",
                self._requested_dim, all_frames.shape[1], actual_dim)

        if self._standardize:
            if self._scaler_type == "robust":
                from sklearn.preprocessing import RobustScaler
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
            self.scaler.fit(all_frames)
            all_frames = self.scaler.transform(all_frames)
            logger.info("%s: %d features", self.scaler.__class__.__name__, self.scaler.center_.shape[0] if self._scaler_type == "robust" else self.scaler.mean_.shape[0])

        all_frames = normalize_frames(all_frames, self._normalize)
        self.pca = PCA(n_components=actual_dim, svd_solver='full')
        self.pca.fit(all_frames)

        logger.info("PCA explained variance total: %.4f (%d components)",
                     self.explained, actual_dim)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply fitted scaler, frame normalization, and PCA."""
        if self.pca is None:
            raise RuntimeError("PCAReducer.fit() must be called before transform()")
        if self.scaler is not None:
            x = self.scaler.transform(x)
        x = normalize_frames(x, self._normalize)
        return self.pca.transform(x)

    @property
    def dim(self) -> int:
        """Actual PCA dimension after fitting."""
        if self.pca is None:
            return self._requested_dim
        return self.pca.n_components_

    @property
    def explained(self) -> float:
        return float(self.pca.explained_variance_ratio_.sum())

    @property
    def explained_per_component(self) -> np.ndarray:
        """PCA explained variance ratio per component."""
        return self.pca.explained_variance_ratio_

    def save(self, path: str):
        """Save the fitted reducer."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> PCAReducer:
        """Load a fitted reducer."""
        return joblib.load(path)


def prepare_track(x_raw, space_cfg, common_cfg, reducer: PCAReducer):
    """Preprocess one track for one representation space."""
    x = resample_fps(x_raw, space_cfg["native_fps"], common_cfg["target_fps"]) \
        if "native_fps" in space_cfg else x_raw
    x = reducer.transform(x)
    return x
