"""
persistence.py
--------------
Vietoris-Rips diagrams and fixed-range diagram vectorization.

DiagramVectorizer must be fitted once on train diagrams and then reused.
Otherwise per-track auto-scaling makes vectors incomparable.
"""
from __future__ import annotations
import logging
import warnings
import numpy as np
from ripser import ripser
import joblib

logger = logging.getLogger(__name__)


def compute_diagrams(cloud: np.ndarray, maxdim: int, thresh: float = 0.0,
                     do_cocycles: bool = False):
    """Return ripser output with optional persistence thresholding."""
    # Takens clouds commonly have D > N; suppress the expected ripser warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                message=".*more columns than rows.*",
                                category=UserWarning)
        res = ripser(cloud, maxdim=maxdim, do_cocycles=do_cocycles)
    dgms = res["dgms"]
    if thresh > 0:
        dgms = [_filter(d, thresh) for d in dgms]
    res["dgms"] = dgms
    return res


def _filter(dgm, thresh):
    if len(dgm) == 0:
        return dgm
    life = dgm[:, 1] - dgm[:, 0]
    return dgm[life >= thresh]


def _to_birth_persist(dgm):
    """(birth, death) -> (birth, persistence). Keeps all points including inf."""
    if len(dgm) == 0:
        return np.empty((0, 2))
    out = dgm.copy()
    out[:, 1] = out[:, 1] - out[:, 0]   # (birth, persistence)
    return out


def _to_bp_finite(dgm):
    """(birth, death) -> (birth, persistence), only finite deaths."""
    if len(dgm) == 0:
        return np.empty((0, 2))
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return np.empty((0, 2))
    out = finite.copy()
    out[:, 1] = out[:, 1] - out[:, 0]
    return out


class DiagramVectorizer:
    """Fixed-coordinate vectorizer for persistence diagrams."""

    def __init__(self):
        self._kind = None
        self._pimgr = None
        self._grid = None
        self._vector_dim = None
        self._birth_range = None
        self._pers_range = None

    def fit(self, dgms_train: list[np.ndarray], cfg: dict):
        """Estimate global diagram ranges from train diagrams."""
        self._kind = cfg["vectorization"]
        pixels = cfg["image_pixels"]

        bp_all = []
        for d in dgms_train:
            bp = _to_bp_finite(d)
            if len(bp) > 0:
                bp_all.append(bp)

        if bp_all:
            all_pts = np.concatenate(bp_all)
            b_min = float(all_pts[:, 0].min())
            b_max = float(all_pts[:, 0].max())
            p_min = float(all_pts[:, 1].min())
            p_max = float(all_pts[:, 1].max())
        else:
            logger.warning("DiagramVectorizer.fit: no finite points in train "
                           "diagrams, using default ranges [0, 1]")
            b_min, b_max = 0.0, 1.0
            p_min, p_max = 0.0, 1.0

        b_pad = max((b_max - b_min) * 0.05, 1e-3)
        p_pad = max((p_max - p_min) * 0.05, 1e-3)
        self._birth_range = (b_min - b_pad, b_max + b_pad)
        self._pers_range = (max(0.0, p_min - p_pad), p_max + p_pad)

        logger.info("DiagramVectorizer: birth_range=(%.4f, %.4f), "
                     "pers_range=(%.4f, %.4f)",
                     *self._birth_range, *self._pers_range)

        if self._kind == "image":
            self._fit_image(pixels, cfg)
        elif self._kind == "image_fix":
            self._fit_image_fix(pixels, cfg)
        elif self._kind == "betti":
            self._fit_betti(pixels)
        else:
            raise ValueError(f"Unknown vectorization: {self._kind}")

        logger.info("DiagramVectorizer: vector_dim=%d", self._vector_dim)
        return self

    def _fit_image(self, pixels, cfg):
        from persim import PersistenceImager

        b_width = self._birth_range[1] - self._birth_range[0]
        p_width = self._pers_range[1] - self._pers_range[0]
        pixel_size = max(b_width, p_width) / pixels

        self._pimgr = PersistenceImager(
            birth_range=self._birth_range,
            pers_range=self._pers_range,
            pixel_size=pixel_size,
        )
        self._pimgr.weight_params = {}

        test_bp = np.array([[
            (self._birth_range[0] + self._birth_range[1]) / 2,
            (self._pers_range[0] + self._pers_range[1]) / 2
        ]])
        test_img = self._pimgr.transform([test_bp])[0]
        self._vector_dim = test_img.flatten().shape[0]

    def _fit_image_fix(self, pixels, cfg):
        """Persistence image with fixed ranges and default persistence weighting."""
        from persim import PersistenceImager

        b_width = self._birth_range[1] - self._birth_range[0]
        p_width = self._pers_range[1] - self._pers_range[0]

        pixel_size = min(b_width, p_width) / pixels
        if pixel_size < 1e-8:
            pixel_size = max(b_width, p_width) / pixels

        logger.info("image_fix: pixel_size=%.6f, birth ~%d px, pers ~%d px",
                     pixel_size,
                     int(b_width / pixel_size) if pixel_size > 0 else 0,
                     int(p_width / pixel_size) if pixel_size > 0 else 0)

        self._pimgr = PersistenceImager(
            birth_range=self._birth_range,
            pers_range=self._pers_range,
            pixel_size=pixel_size,
        )
        test_bp = np.array([[
            (self._birth_range[0] + self._birth_range[1]) / 2,
            (self._pers_range[0] + self._pers_range[1]) / 2
        ]])
        test_img = self._pimgr.transform([test_bp])[0]
        self._vector_dim = test_img.flatten().shape[0]
        logger.info("image_fix: vector_dim=%d", self._vector_dim)

    def _fit_betti(self, pixels):
        lo = self._birth_range[0]
        hi = self._birth_range[1] + self._pers_range[1]
        self._grid = np.linspace(lo, hi, pixels)
        self._vector_dim = pixels

    def transform(self, dgm: np.ndarray) -> np.ndarray:
        """Transform one diagram into a fixed-length vector."""
        if self._kind in ("image", "image_fix"):
            return self._transform_image(dgm)
        elif self._kind == "betti":
            return self._transform_betti(dgm)
        raise ValueError(self._kind)

    def _transform_image(self, dgm):
        bp = _to_bp_finite(dgm)
        if len(bp) == 0:
            return np.zeros(self._vector_dim)
        img = self._pimgr.transform([bp])[0]
        return img.flatten()

    def _transform_betti(self, dgm):
        if len(dgm) == 0:
            return np.zeros(self._vector_dim)
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) == 0:
            return np.zeros(self._vector_dim)
        return np.array([((finite[:, 0] <= t) & (finite[:, 1] > t)).sum()
                         for t in self._grid], dtype=float)

    @property
    def vector_dim(self) -> int:
        """Output vector length."""
        if self._vector_dim is None:
            raise RuntimeError("DiagramVectorizer.fit() must be called first")
        return self._vector_dim

    def save(self, path: str):
        """Save the fitted vectorizer."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> DiagramVectorizer:
        """Load a fitted vectorizer."""
        return joblib.load(path)


# Backward-compatible standalone function.
# Kept for older exploratory scripts. Final results use DiagramVectorizer.

def vectorize(dgm, cfg):
    """Legacy one-off vectorization without fitted global ranges."""
    kind = cfg["vectorization"]
    if kind == "image":
        from persim import PersistenceImager
        pimgr = PersistenceImager(pixel_size=1.0 / cfg["image_pixels"])
        pimgr.weight_params = {}
        return pimgr.transform([_to_birth_persist(dgm)])[0].flatten()
    if kind == "betti":
        return _betti_curve(dgm, n=cfg["image_pixels"])
    raise ValueError(kind)


def _betti_curve(dgm, n):
    """Count live classes on a fixed threshold grid."""
    if len(dgm) == 0:
        return np.zeros(n)
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return np.zeros(n)
    lo, hi = finite[:, 0].min(), finite[:, 1].max()
    grid = np.linspace(lo, hi, n)
    return np.array([((finite[:, 0] <= t) & (finite[:, 1] > t)).sum()
                     for t in grid], dtype=float)
