"""
distances.py
------------
Сравнение диаграмм + значимость + кросс-пространственный анализ.

within vs between genre:
  гипотеза «похожие треки -> похожая топология» = среднее Вассерштейн-
  расстояние ВНУТРИ жанра значимо МЕНЬШЕ, чем МЕЖДУ жанрами. С bootstrap-CI,
  т.к. метрики диаграмм нестабильны в высокой размерности (см. обзор).

cross-space consistency (новое для 3+ пространств):
  видят ли MERT / MuQ / Encodec / MIR ОДНУ топологическую структуру музыки,
  или каждое — свою? Mantel-корреляция матриц попарных расстояний.
"""
from __future__ import annotations
import numpy as np


def pairwise_wasserstein(diagrams: list[np.ndarray]) -> np.ndarray:
    """Список H1-диаграмм -> симметричная матрица (n, n) расстояний."""
    from persim import wasserstein
    from tqdm import tqdm
    n = len(diagrams)
    D = np.zeros((n, n))
    for i in tqdm(range(n), desc="Wasserstein matrix", leave=False):
        for j in range(i + 1, n):
            d = wasserstein(diagrams[i], diagrams[j])
            D[i, j] = D[j, i] = d
    return D


def sampled_distances(diagrams: list[np.ndarray], labels: np.ndarray, n_within: int = 2000, n_between: int = 2000, seed: int = 42):
    """Случайно сэмплирует пары для расчета Wasserstein, чтобы избежать O(N^2) на 999 треках.
    Возвращает (within_dists, between_dists, within_pairs, between_pairs)."""
    from persim import wasserstein
    from tqdm import tqdm
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n = len(diagrams)
    
    # Находим все возможные пары
    iu = np.triu_indices(n, k=1)
    same = labels[iu[0]] == labels[iu[1]]
    
    idx_within = np.where(same)[0]
    idx_between = np.where(~same)[0]
    
    if len(idx_within) > n_within:
        idx_within = rng.choice(idx_within, n_within, replace=False)
    if len(idx_between) > n_between:
        idx_between = rng.choice(idx_between, n_between, replace=False)
        
    pairs_within = (iu[0][idx_within], iu[1][idx_within])
    pairs_between = (iu[0][idx_between], iu[1][idx_between])
    
    within_dists = []
    between_dists = []
    
    print(f"Calculating {len(idx_within)} within-pairs and {len(idx_between)} between-pairs...")
    for i, j in tqdm(zip(*pairs_within), total=len(idx_within), desc="Within", leave=False):
        within_dists.append(wasserstein(diagrams[i], diagrams[j]))
        
    for i, j in tqdm(zip(*pairs_between), total=len(idx_between), desc="Between", leave=False):
        between_dists.append(wasserstein(diagrams[i], diagrams[j]))
        
    return np.array(within_dists), np.array(between_dists), pairs_within, pairs_between

def within_between(D: np.ndarray = None, labels: np.ndarray = None, w_arr: np.ndarray = None, b_arr: np.ndarray = None):
    """Средние within/between + разность. Поддерживает как матрицу D, так и готовые массивы."""
    if D is not None and labels is not None:
        labels = np.asarray(labels)
        iu = np.triu_indices_from(D, k=1)
        same = labels[iu[0]] == labels[iu[1]]
        within = D[iu][same]
        between = D[iu][~same]
    else:
        within, between = w_arr, b_arr
        
    w = within.mean() if (within is not None and within.size) else np.nan
    b = between.mean() if (between is not None and between.size) else np.nan
    return w, b, b - w

def bootstrap_gap(D=None, labels=None, w_arr=None, b_arr=None, n_boot=1000, rng=None):
    """Bootstrap-CI разности (between - within). Поддерживает полные матрицы D или сэмплированные массивы."""
    rng = rng or np.random.default_rng(0)
    gaps = []
    
    if D is not None and labels is not None:
        n = D.shape[0]
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            Db = D[np.ix_(idx, idx)]
            _, _, gap = within_between(D=Db, labels=np.asarray(labels)[idx])
            if not np.isnan(gap):
                gaps.append(gap)
    else:
        # Прямой ресэмплинг сэмплированных массивов
        nw, nb = len(w_arr), len(b_arr)
        for _ in range(n_boot):
            w_boot = w_arr[rng.integers(0, nw, nw)] if nw > 0 else np.array([])
            b_boot = b_arr[rng.integers(0, nb, nb)] if nb > 0 else np.array([])
            _, _, gap = within_between(w_arr=w_boot, b_arr=b_boot)
            if not np.isnan(gap):
                gaps.append(gap)
                
    gaps = np.array(gaps)
    if gaps.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    return float(gaps.mean()), (float(np.percentile(gaps, 2.5)),
                                float(np.percentile(gaps, 97.5)))


def bootstrap_gap_by_track(D: np.ndarray, labels: np.ndarray,
                           n_boot: int = 9999, rng=None):
    """Кластерный бутстрап gap = mean(between) − mean(within).

    Кластер = трек: на каждой итерации ресэмплируются ИНДЕКСЫ ТРЕКОВ
    с возвратом, затем берётся подматрица D и разделяется на within/between.
    Это корректно учитывает зависимость пар, деливших общий трек.

    Возвращает
    ----------
    gap_observed : float
        Gap на полной (не ресэмплированной) выборке.
    ci : tuple[float, float]
        Percentile 2.5 / 97.5 доверительный интервал.
    gaps : np.ndarray
        Массив gap-значений по итерациям (длина ≤ n_boot).
    """
    rng = rng or np.random.default_rng(0)
    labels = np.asarray(labels)
    N = D.shape[0]

    def _gap_from_idx(idx):
        """Вычисляет gap для подматрицы D[idx, idx]."""
        Dsub = D[np.ix_(idx, idx)]
        lsub = labels[idx]
        n = len(idx)
        # k=1: исключаем диагональ — при повторных треках
        # диагональ = 0 (расстояние трека до самого себя) → раздувает gap
        iu = np.triu_indices(n, k=1)
        same = lsub[iu[0]] == lsub[iu[1]]
        within = Dsub[iu][same]
        between = Dsub[iu][~same]
        if within.size == 0 or between.size == 0:
            return np.nan
        return float(between.mean() - within.mean())

    # Observed gap на полной выборке (не среднее бутстрапа!)
    gap_observed = _gap_from_idx(np.arange(N))

    gaps = np.empty(n_boot)
    valid = 0
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        g = _gap_from_idx(idx)
        if not np.isnan(g):
            gaps[valid] = g
            valid += 1
    gaps = gaps[:valid]

    if gaps.size == 0:
        return gap_observed, (float("nan"), float("nan")), gaps

    ci = (float(np.percentile(gaps, 2.5)),
          float(np.percentile(gaps, 97.5)))
    return gap_observed, ci, gaps


def mantel_test(D1: np.ndarray, D2: np.ndarray,
                n_perm: int = 9999, rng=None,
                two_sided: bool = True) -> dict:
    """Тест Мантеля с пермутациями: проверяет значимость корреляции
    двух матриц расстояний.

    Нулевое распределение строится СИНХРОННОЙ перестановкой строк И столбцов
    одной матрицы (сохраняет структуру расстояний), а НЕ перемешиванием
    элементов верхнего треугольника.

    Parameters
    ----------
    D1, D2 : (N, N) симметричные матрицы расстояний
    n_perm : число пермутаций (по умолч. 9999)
    rng    : np.random.Generator (для воспроизводимости)
    two_sided : если True, p = P(|r_perm| >= |r_obs|); иначе P(r_perm >= r_obs)

    Returns
    -------
    dict с ключами: r, p, r_perm (np.ndarray), n_perm
    """
    rng = rng or np.random.default_rng(0)
    N = D1.shape[0]
    iu = np.triu_indices(N, k=1)
    a = D1[iu].astype(np.float64)
    b = D2[iu].astype(np.float64)

    # ── защита от вырожденной дисперсии ──
    if a.std() < 1e-12 or b.std() < 1e-12:
        return {"r": 0.0, "p": 1.0,
                "r_perm": np.zeros(n_perm), "n_perm": n_perm}

    r_obs = float(np.corrcoef(a, b)[0, 1])

    # ── нулевое распределение: синхронная перестановка строк+столбцов D2 ──
    r_perm = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(N)
        Dp = D2[np.ix_(p, p)]
        bp = Dp[iu].astype(np.float64)
        if bp.std() < 1e-12:
            r_perm[i] = 0.0
        else:
            r_perm[i] = float(np.corrcoef(a, bp)[0, 1])

    # ── p-value с поправкой +1 ──
    if two_sided:
        count = np.sum(np.abs(r_perm) >= np.abs(r_obs))
    else:
        count = np.sum(r_perm >= r_obs)
    p_value = float(count + 1) / float(n_perm + 1)

    return {"r": r_obs, "p": p_value,
            "r_perm": r_perm, "n_perm": n_perm}


def mantel_bootstrap_ci(D1: np.ndarray, D2: np.ndarray,
                        n_boot: int = 2000, rng=None,
                        alpha: float = 0.05) -> tuple[float, float]:
    """Бутстрап-CI для Mantel-корреляции r(D1, D2).

    Ресэмплирование на уровне ТРЕКОВ (индексов), а не элементов
    верхнего треугольника, чтобы сохранить структуру зависимости пар.

    Returns (ci_low, ci_high) — percentile CI.
    """
    rng = rng or np.random.default_rng(0)
    N = D1.shape[0]
    r_boot = np.empty(n_boot)
    valid = 0
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        D1b = D1[np.ix_(idx, idx)]
        D2b = D2[np.ix_(idx, idx)]
        iu = np.triu_indices(N, k=1)
        a = D1b[iu].astype(np.float64)
        b = D2b[iu].astype(np.float64)
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        r_boot[valid] = float(np.corrcoef(a, b)[0, 1])
        valid += 1
    r_boot = r_boot[:valid]
    if valid == 0:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(r_boot, 100 * alpha / 2))
    hi = float(np.percentile(r_boot, 100 * (1 - alpha / 2)))
    return (lo, hi)


def mantel(D1: np.ndarray, D2: np.ndarray) -> float:
    """Обратная совместимость: возвращает только r (Pearson).

    Для полного теста значимости используйте mantel_test().
    """
    return mantel_test(D1, D2)["r"]


# ── inline-тест на синтетике ──────────────────────────────────────
if __name__ == "__main__":
    print("=== Inline-тест bootstrap_gap_by_track ===")
    rng = np.random.default_rng(42)

    # 60 «треков», 3 жанра с разнесёнными центрами
    n_per_genre = 20
    genres = np.repeat(["rock", "jazz", "classical"], n_per_genre)
    centers = {"rock": 0.0, "jazz": 5.0, "classical": 10.0}

    # Эмбеддинг: d=4, каждый жанр сдвинут по первой координате
    embs = []
    for g in genres:
        embs.append(rng.standard_normal(4) + np.array([centers[g], 0, 0, 0]))
    embs = np.array(embs)

    # Евклидова матрица расстояний (прокси для Вассерштейна)
    from scipy.spatial.distance import squareform, pdist
    D = squareform(pdist(embs))

    gap_obs, ci, gaps = bootstrap_gap_by_track(D, genres, n_boot=4999, rng=rng)

    print(f"  N tracks     = {len(genres)}")
    print(f"  gap_observed = {gap_obs:.4f}")
    print(f"  95% CI       = [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  boot iters   = {len(gaps)}")
    print(f"  boot mean    = {gaps.mean():.4f}  (≠ gap_observed — это нормально)")

    assert gap_obs > 0, f"gap должен быть > 0, получили {gap_obs}"
    assert ci[0] > 0, f"нижняя граница CI должна быть > 0, получили {ci[0]}"
    print("\n✅ bootstrap_gap_by_track: gap > 0, CI не включает 0 — OK")
