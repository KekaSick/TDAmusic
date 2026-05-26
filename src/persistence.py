"""
persistence.py
--------------
Vietoris-Rips через ripser -> диаграммы H0/H1 -> векторизация для ML.

H1 (петли) музыкально наиболее значимо. H0 (компоненты) дёшево.
Отсечка persistence_threshold убирает коротко-живущий шум — он раздут
из-за контекстуализации (соседние кадры сильно скоррелированы через
self-attention, см. обзор), поэтому фильтрация важна.

Векторизация нужна, т.к. диаграмма — множество переменного размера, его
нельзя подать в классификатор напрямую.

ВАЖНО: DiagramVectorizer.fit() фиксирует birth_range/pers_range на
TRAIN-диаграммах. transform() переиспользует зафиксированные диапазоны.
Без этого вектора разных треков несопоставимы (каждый трек нормирован
на свой диапазон — см. §5 CONTRACTS.md).
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
    """Возвращает dict: {'dgms': [H0, H1, ...], 'cocycles': ...}."""
    # Takens embedding: (n_points, window*pca_dim) → D > N — ожидаемо,
    # ripser предупреждает зря. Подавляем.
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
    """Обучается на train-диаграммах, фиксирует birth_range/pers_range.
    Применяется ко всем — по той же логике, что PCA.

    Без фиксации диапазонов вектора разных треков несопоставимы:
    PersistenceImager с авто-подгонкой нормирует каждый трек на свой
    диапазон, и разница между треками теряется.
    """

    def __init__(self):
        self._kind = None
        self._pimgr = None       # для "image"
        self._grid = None        # для "betti"
        self._vector_dim = None
        self._birth_range = None
        self._pers_range = None

    def fit(self, dgms_train: list[np.ndarray], cfg: dict):
        """Вычисляет глобальные диапазоны по train-диаграммам.

        Для "image": создаёт PersistenceImager ОДИН раз с фиксированными
        birth_range/pers_range. Для "betti": фиксирует grid.
        """
        self._kind = cfg["vectorization"]
        pixels = cfg["image_pixels"]

        # Собрать все конечные (birth, persistence) из train
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

        # Padding + защита от вырожденных диапазонов
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

        # pixel_size для ~pixels пикселей в каждом измерении
        b_width = self._birth_range[1] - self._birth_range[0]
        p_width = self._pers_range[1] - self._pers_range[0]
        # Единый pixel_size — квадратные пиксели
        pixel_size = max(b_width, p_width) / pixels

        self._pimgr = PersistenceImager(
            birth_range=self._birth_range,
            pers_range=self._pers_range,
            pixel_size=pixel_size,
        )
        # Без весовой функции (как в оригинале)
        self._pimgr.weight_params = {}

        # Определить реальный размер выхода
        test_bp = np.array([[
            (self._birth_range[0] + self._birth_range[1]) / 2,
            (self._pers_range[0] + self._pers_range[1]) / 2
        ]])
        test_img = self._pimgr.transform([test_bp])[0]
        self._vector_dim = test_img.flatten().shape[0]

    def _fit_image_fix(self, pixels, cfg):
        """Fixed persistence image: раздельное разрешение по осям + штатное взвешивание."""
        from persim import PersistenceImager

        b_width = self._birth_range[1] - self._birth_range[0]
        p_width = self._pers_range[1] - self._pers_range[0]

        # Раздельное разрешение: обе оси покрываются ~pixels пикселями
        # PersistenceImager принимает скалярный pixel_size, так что
        # используем меньший из двух, чтобы обе оси были покрыты полностью
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
        # Штатное взвешивание (по умолчанию — линейная ramping по persistence)
        # НЕ обнуляем weight_params!

        test_bp = np.array([[
            (self._birth_range[0] + self._birth_range[1]) / 2,
            (self._pers_range[0] + self._pers_range[1]) / 2
        ]])
        test_img = self._pimgr.transform([test_bp])[0]
        self._vector_dim = test_img.flatten().shape[0]
        logger.info("image_fix: vector_dim=%d", self._vector_dim)

    def _fit_betti(self, pixels):
        # Фиксированная сетка по (birth, death) =
        # (birth_min, birth_max + max_persistence)
        lo = self._birth_range[0]
        hi = self._birth_range[1] + self._pers_range[1]
        self._grid = np.linspace(lo, hi, pixels)
        self._vector_dim = pixels

    def transform(self, dgm: np.ndarray) -> np.ndarray:
        """Превращает одну диаграмму в вектор фиксированной длины."""
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
        """Размер выходного вектора."""
        if self._vector_dim is None:
            raise RuntimeError("DiagramVectorizer.fit() must be called first")
        return self._vector_dim

    def save(self, path: str):
        """Сохранить обученный vectorizer (для viz/classify скриптов)."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> DiagramVectorizer:
        """Загрузить ранее обученный vectorizer."""
        return joblib.load(path)


# ---- backward-compatible standalone function ----
# ВНИМАНИЕ: используется ТОЛЬКО в run_pipeline.py (скелет, не критический путь).
# Все финальные результаты (results/tables/) получены через DiagramVectorizer.

def vectorize(dgm, cfg):
    """Диаграмма -> фиксированный вектор для классификатора.

    ⚠ ПРЕДУПРЕЖДЕНИЯ:
    1. Для mode="image" использует _to_birth_persist, которая ОСТАВЛЯЕТ точки
       с death=inf (характерны для H0). Это НЕКОРРЕКТНО для H0 — inf-persistence
       раздувает масштаб и искажает PersistenceImage. Для H1 inf-точки обычно
       отсутствуют, поэтому на практике проблема не проявляется, но поведение
       отличается от DiagramVectorizer, который всегда фильтрует inf через
       _to_bp_finite.
    2. Создаёт PersistenceImager с авто-подгонкой на КАЖДЫЙ вызов — диапазоны
       birth/persistence не фиксируются, что делает вектора разных треков
       несопоставимыми (каждый трек нормирован на свой диапазон).

    Для корректного кросс-сравнения треков используйте DiagramVectorizer с
    fit()/transform() — он фиксирует диапазоны на train-данных и фильтрует inf.

    НИ ОДИН финальный результат статьи не получен через эту функцию.
    Все результаты в results/tables/ сгенерированы через DiagramVectorizer
    (см. scripts/run_full_scale.py, scripts/run_classify_betti.py и др.).
    """
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
    """Простая Betti-кривая: число классов, живых на сетке порогов."""
    if len(dgm) == 0:
        return np.zeros(n)
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return np.zeros(n)
    lo, hi = finite[:, 0].min(), finite[:, 1].max()
    grid = np.linspace(lo, hi, n)
    return np.array([((finite[:, 0] <= t) & (finite[:, 1] > t)).sum()
                     for t in grid], dtype=float)
