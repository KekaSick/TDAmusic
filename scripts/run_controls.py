"""
run_controls.py
---------------
ЯДРО НАУЧНОЙ ЧАСТИ. Прогоняет контроли и within/between анализ
по всем 4 каналам на размеченном подмножестве.

Для каждого канала и каждого трека:
  1. real:    preprocess → takens → Ripser → H1
  2. shuffle: shuffle ИСХОДНОГО ряда ДО окна Такенса (×K повторов)
  3. random:  phase randomization спектра (×K повторов)
  4. iaaft:   IAAFT-суррогат (×K повторов)
  5. mean_pool: среднее по кадрам (baseline вектор)

Лестница контролей:
  shuffle (порядок) → phase (линейная структура) → iaaft (лин. + распределение)

Все три контроля делаются в K = controls.shuffle_repeats повторах
с детерминированным seed на трек+повтор. max-pers усредняется по повторам.

Метрики:
  - within vs between genre по Вассерштейну + bootstrap-CI
  - real↔shuffle расстояние vs real↔real (оценка эффекта shuffle)
  - max-persistence H1 real vs random vs iaaft vs shuffle
  - парные Wilcoxon + rank-biserial для всех пар
  - Mantel между DL-каналами (cross_space_compare)
  - H1 point counts

Результаты → results/tables/
Числа — как есть, без интерпретации.
"""
from __future__ import annotations
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*more columns than rows.*")
warnings.filterwarnings("ignore", category=UserWarning, module="ripser")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import embed_spaces, preprocess, pointcloud, persistence, controls, distances


def max_persistence_h1(dgm: np.ndarray) -> float:
    """Максимальная персистентность H1 (death - birth), finite only."""
    if len(dgm) == 0:
        return 0.0
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return 0.0
    return float((finite[:, 1] - finite[:, 0]).max())


def h1_count(dgm: np.ndarray, thresh: float = 0.0) -> int:
    """Число точек H1 (finite) с persistence >= thresh."""
    if len(dgm) == 0:
        return 0
    finite = dgm[np.isfinite(dgm[:, 1])]
    if thresh > 0 and len(finite) > 0:
        life = finite[:, 1] - finite[:, 0]
        finite = finite[life >= thresh]
    return len(finite)


def wasserstein_single(d1, d2):
    """Wasserstein distance between two diagrams."""
    from persim import wasserstein
    return wasserstein(d1, d2)


def process_space(space: str, cfg: dict, files: list, labels: np.ndarray,
                  tr_idx: np.ndarray, rng: np.random.Generator,
                  compute_random: bool = True, compute_iaaft: bool = True) -> dict:
    """Полная обработка одного пространства."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  SPACE: {space}")
    print(f"{'='*60}")

    space_cfg = cfg["spaces"][space]
    common_cfg = cfg["common"]
    pc_cfg = cfg["pointcloud"]
    pers_cfg = cfg["persistence"]
    n_shuffle = cfg["controls"]["shuffle_repeats"]
    match_autocorr = cfg["controls"]["random_match_autocorr"]
    iaaft_n_iter = cfg["controls"].get("iaaft_n_iter", 200)
    iaaft_tol = cfg["controls"].get("iaaft_tol", 1e-8)

    # --- 1. Load raw embeddings from cache ---
    print(f"  Loading {len(files)} cached embeddings...")
    raw = []
    for f in files:
        arr = embed_spaces.extract(f, space, cfg)
        raw.append(arr)

    # --- 2. Fit PCA on train ---
    print(f"  Fitting PCA on train ({len(tr_idx)} tracks)...")
    reducer = preprocess.PCAReducer(
        common_cfg["pca_dim"],
        standardize=common_cfg.get("standardize_features", True),
        normalize=common_cfg["normalize"],
        scaler_type=common_cfg.get("scaler", "standard"))
    tr_frames = [
        preprocess.resample_fps(
            raw[i],
            space_cfg.get("native_fps", common_cfg["target_fps"]),
            common_cfg["target_fps"])
        for i in tr_idx
    ]
    reducer.fit(tr_frames)
    print(f"    PCA explained variance: {reducer.explained:.4f} "
          f"({reducer.dim} components)")

    # --- 3. Preprocess all tracks ---
    print(f"  Preprocessing all tracks...")
    preprocessed = []
    for x_raw in raw:
        x = preprocess.prepare_track(x_raw, space_cfg, common_cfg, reducer)
        preprocessed.append(x)

    # --- 4. Build clouds + diagrams (real, shuffle, random, iaaft) ---
    print(f"  Computing diagrams (real + {n_shuffle}×shuffle + {n_shuffle}×random + {n_shuffle}×iaaft)...")

    diags_real = []        # H1 diagrams for real trajectories
    diags_shuffle = []     # list of lists: [[shuffle_1..shuffle_K] per track]
    diags_random = []      # list of lists: [[random_1..random_K] per track]
    diags_iaaft = []       # list of lists: [[iaaft_1..iaaft_K] per track]
    mean_pool_vecs = []    # mean-pool vectors
    h1_counts_real = []
    h1_counts_real_thresh = []
    max_pers_real = []
    max_pers_random = []   # mean max-pers over K repeats per track
    max_pers_shuffle = []  # mean max-pers over K repeats per track
    max_pers_iaaft = []    # mean max-pers over K repeats per track

    for i, x in enumerate(preprocessed):
        # --- Real ---
        cloud = pointcloud.build_cloud(x, pc_cfg)
        res = persistence.compute_diagrams(
            cloud, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
        dgm_real = res["dgms"][1]  # H1
        diags_real.append(dgm_real)
        h1_counts_real.append(h1_count(dgm_real))
        h1_counts_real_thresh.append(h1_count(dgm_real, thresh=0.05))
        max_pers_real.append(max_persistence_h1(dgm_real))

        # --- Mean pool ---
        mean_pool_vecs.append(controls.mean_pool(x))

        # --- Shuffle (K repeats) ---
        track_shuffle_dgms = []
        track_shuffle_max_pers = []
        for k in range(n_shuffle):
            shuffle_rng = np.random.default_rng(cfg["seed"] * 1000 + i * 100 + k)
            x_shuf = controls.shuffle_frames(x, shuffle_rng)
            cloud_shuf = pointcloud.build_cloud(x_shuf, pc_cfg)
            res_shuf = persistence.compute_diagrams(
                cloud_shuf, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
            track_shuffle_dgms.append(res_shuf["dgms"][1])
            track_shuffle_max_pers.append(max_persistence_h1(res_shuf["dgms"][1]))
        diags_shuffle.append(track_shuffle_dgms)
        max_pers_shuffle.append(np.mean(track_shuffle_max_pers))

        # --- Random (K repeats, phase randomization) ---
        if compute_random:
            track_random_dgms = []
            track_random_max_pers = []
            for k in range(n_shuffle):
                random_rng = np.random.default_rng(cfg["seed"] * 2000 + i * 100 + k)
                x_rand = controls.random_like(x, random_rng, match_autocorr)
                cloud_rand = pointcloud.build_cloud(x_rand, pc_cfg)
                res_rand = persistence.compute_diagrams(
                    cloud_rand, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
                track_random_dgms.append(res_rand["dgms"][1])
                track_random_max_pers.append(max_persistence_h1(res_rand["dgms"][1]))
            diags_random.append(track_random_dgms)
            max_pers_random.append(np.mean(track_random_max_pers))

        # --- IAAFT (K repeats) ---
        if compute_iaaft:
            track_iaaft_dgms = []
            track_iaaft_max_pers = []
            for k in range(n_shuffle):
                iaaft_rng = np.random.default_rng(cfg["seed"] * 3000 + i * 100 + k)
                x_iaaft = controls.iaaft_surrogate(
                    x, iaaft_rng, n_iter=iaaft_n_iter, tol=iaaft_tol)
                cloud_iaaft = pointcloud.build_cloud(x_iaaft, pc_cfg)
                res_iaaft = persistence.compute_diagrams(
                    cloud_iaaft, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
                track_iaaft_dgms.append(res_iaaft["dgms"][1])
                track_iaaft_max_pers.append(max_persistence_h1(res_iaaft["dgms"][1]))
            diags_iaaft.append(track_iaaft_dgms)
            max_pers_iaaft.append(np.mean(track_iaaft_max_pers))

        if (i + 1) % 10 == 0 or i == 0:
            print(f"    [{i+1}/{len(preprocessed)}] H1 points: {h1_count(dgm_real)}, "
                  f"max_pers: {max_pers_real[-1]:.4f}")

    # --- 5. Within vs between (real) ---
    print(f"  Computing pairwise Wasserstein (real)...")
    D_real = distances.pairwise_wasserstein(diags_real)
    w, b, gap = distances.within_between(D_real, labels)
    gap_obs, gap_ci, _gap_boots = distances.bootstrap_gap_by_track(
        D_real, labels,
        n_boot=cfg["distances"]["bootstrap"],
        rng=np.random.default_rng(cfg["seed"]))
    print(f"    within={w:.4f}, between={b:.4f}, gap={gap:.4f}")
    print(f"    track-level bootstrap gap: {gap_obs:.4f}, CI95={gap_ci}")

    # --- 6. Shuffle effect (per-track aggregation to avoid pseudo-replication) ---
    # For each track i: mean Wasserstein(real_i, shuffle_i_k) over k repeats
    # Compare to per-track within-genre mean distance from D_real
    print(f"  Computing shuffle effect (per-track)...")
    from scipy import stats

    n_tracks = len(diags_real)
    per_track_shuffle_dist = np.empty(n_tracks)
    for i in range(n_tracks):
        dists_i = [wasserstein_single(diags_real[i], dgm_shuf)
                   for dgm_shuf in diags_shuffle[i]]
        per_track_shuffle_dist[i] = np.mean(dists_i)

    # Per-track within-genre mean distance: for track i,
    # mean of D_real[i, j] for all j in same genre (j != i)
    per_track_within_dist = np.empty(n_tracks)
    for i in range(n_tracks):
        same_mask = (labels == labels[i])
        same_mask[i] = False  # exclude self
        if same_mask.sum() > 0:
            per_track_within_dist[i] = D_real[i, same_mask].mean()
        else:
            per_track_within_dist[i] = np.nan

    shuffle_mean = float(per_track_shuffle_dist.mean())
    shuffle_std = float(per_track_shuffle_dist.std())
    shuffle_sem = float(per_track_shuffle_dist.std() / np.sqrt(n_tracks))
    within_mean = float(per_track_within_dist.mean())
    within_std = float(per_track_within_dist.std())
    within_sem = float(per_track_within_dist.std() / np.sqrt(n_tracks))

    # Paired Wilcoxon (by track): shuffle > within?
    stat_shuffle, p_value_shuffle = stats.wilcoxon(
        per_track_shuffle_dist, per_track_within_dist,
        alternative="greater")

    # Effect size: matched-pairs rank-biserial correlation
    # r_rb = 1 - (2*T) / (n*(n+1)/2) where T = lesser of W+, W-
    diffs = per_track_shuffle_dist - per_track_within_dist
    nonzero_diffs = diffs[diffs != 0]
    n_eff = len(nonzero_diffs)
    if n_eff > 0:
        ranks = stats.rankdata(np.abs(nonzero_diffs))
        r_plus = ranks[nonzero_diffs > 0].sum()
        r_minus = ranks[nonzero_diffs < 0].sum()
        shuffle_effect_size = float((r_plus - r_minus) / (r_plus + r_minus))
    else:
        shuffle_effect_size = 0.0

    print(f"    real↔shuffle (per-track): {shuffle_mean:.4f} ± {shuffle_std:.4f} (SEM={shuffle_sem:.4f})")
    print(f"    real↔real within (per-track): {within_mean:.4f} ± {within_std:.4f} (SEM={within_sem:.4f})")
    print(f"    Wilcoxon paired p-value (shuffle > within): {p_value_shuffle:.6f}")
    print(f"    Effect size (rank-biserial r): {shuffle_effect_size:.4f}")

    # --- 7. Max-persistence: real vs shuffle vs random vs iaaft ---
    print(f"  Max-persistence H1: real vs shuffle vs random vs iaaft...")
    mp_real = np.array(max_pers_real)
    mp_shuffle = np.array(max_pers_shuffle)

    def _rank_biserial(a, b):
        """Matched-pairs rank-biserial r for paired Wilcoxon."""
        d = a - b
        nz = d[d != 0]
        if len(nz) == 0:
            return 0.0
        r = stats.rankdata(np.abs(nz))
        rp = r[nz > 0].sum()
        rm = r[nz < 0].sum()
        return float((rp - rm) / (rp + rm))

    t_stat_shuf_mp, p_value_shuf_mp = stats.wilcoxon(mp_real, mp_shuffle, alternative="two-sided")
    effect_real_vs_shuffle_mp = _rank_biserial(mp_real, mp_shuffle)

    mp_real_sem = float(mp_real.std() / np.sqrt(len(mp_real)))
    mp_shuffle_sem = float(mp_shuffle.std() / np.sqrt(len(mp_shuffle)))

    if compute_random:
        mp_random = np.array(max_pers_random)
        t_stat_rand, p_value_rand = stats.wilcoxon(mp_real, mp_random, alternative="two-sided")
        effect_real_vs_random = _rank_biserial(mp_real, mp_random)
        mp_random_mean = float(mp_random.mean())
        mp_random_std = float(mp_random.std())
        mp_random_sem = float(mp_random.std() / np.sqrt(len(mp_random)))
        p_value_rand = float(p_value_rand)
    else:
        mp_random_mean = None
        mp_random_std = None
        mp_random_sem = None
        p_value_rand = None
        effect_real_vs_random = None

    if compute_iaaft:
        mp_iaaft = np.array(max_pers_iaaft)
        t_stat_iaaft, p_value_iaaft = stats.wilcoxon(mp_real, mp_iaaft, alternative="two-sided")
        effect_real_vs_iaaft = _rank_biserial(mp_real, mp_iaaft)
        mp_iaaft_mean = float(mp_iaaft.mean())
        mp_iaaft_std = float(mp_iaaft.std())
        mp_iaaft_sem = float(mp_iaaft.std() / np.sqrt(len(mp_iaaft)))
        p_value_iaaft = float(p_value_iaaft)
    else:
        mp_iaaft_mean = None
        mp_iaaft_std = None
        mp_iaaft_sem = None
        p_value_iaaft = None
        effect_real_vs_iaaft = None

    print(f"    real   max-pers: {mp_real.mean():.4f} ± {mp_real.std():.4f} (SEM={mp_real_sem:.4f})")
    print(f"    shuffle max-pers: {mp_shuffle.mean():.4f} ± {mp_shuffle.std():.4f} (SEM={mp_shuffle_sem:.4f})")
    if compute_random:
        print(f"    random max-pers: {mp_random_mean:.4f} ± {mp_random_std:.4f} (SEM={mp_random_sem:.4f})")
        print(f"    Wilcoxon real vs random  p={p_value_rand:.6f}, r_rb={effect_real_vs_random:.4f}")
    if compute_iaaft:
        print(f"    iaaft  max-pers: {mp_iaaft_mean:.4f} ± {mp_iaaft_std:.4f} (SEM={mp_iaaft_sem:.4f})")
        print(f"    Wilcoxon real vs iaaft   p={p_value_iaaft:.6f}, r_rb={effect_real_vs_iaaft:.4f}")
    print(f"    Wilcoxon real vs shuffle p={p_value_shuf_mp:.6f}, r_rb={effect_real_vs_shuffle_mp:.4f}")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    return {
        "space": space,
        "n_tracks": len(files),
        "pca_explained": float(reducer.explained),
        "pca_dim_actual": int(reducer.dim),
        # within/between
        "within_mean": float(w),
        "between_mean": float(b),
        "gap": float(gap),
        "gap_obs_track": float(gap_obs),
        "gap_ci95_track": [float(gap_ci[0]), float(gap_ci[1])],
        # shuffle (per-track aggregated)
        "shuffle_dist_mean": shuffle_mean,
        "shuffle_dist_std": shuffle_std,
        "shuffle_dist_sem": shuffle_sem,
        "within_dist_mean": within_mean,
        "within_dist_std": within_std,
        "within_dist_sem": within_sem,
        "shuffle_p_value": float(p_value_shuffle),
        "shuffle_effect_size": shuffle_effect_size,
        # max persistence
        "max_pers_real_mean": float(mp_real.mean()),
        "max_pers_real_std": float(mp_real.std()),
        "max_pers_real_sem": mp_real_sem,
        "max_pers_shuffle_mean": float(mp_shuffle.mean()),
        "max_pers_shuffle_std": float(mp_shuffle.std()),
        "max_pers_shuffle_sem": mp_shuffle_sem,
        "max_pers_random_mean": mp_random_mean,
        "max_pers_random_std": mp_random_std,
        "max_pers_random_sem": mp_random_sem,
        "max_pers_iaaft_mean": mp_iaaft_mean,
        "max_pers_iaaft_std": mp_iaaft_std,
        "max_pers_iaaft_sem": mp_iaaft_sem,
        "max_pers_real_vs_shuffle_p": float(p_value_shuf_mp),
        "max_pers_real_vs_shuffle_effect": effect_real_vs_shuffle_mp,
        "max_pers_real_vs_random_p": p_value_rand,
        "max_pers_real_vs_random_effect": effect_real_vs_random,
        "max_pers_real_vs_iaaft_p": p_value_iaaft,
        "max_pers_real_vs_iaaft_effect": effect_real_vs_iaaft,
        # H1 counts
        "h1_count_mean": float(np.mean(h1_counts_real)),
        "h1_count_std": float(np.std(h1_counts_real)),
        "h1_count_sem": float(np.std(h1_counts_real) / np.sqrt(len(h1_counts_real))),
        "h1_count_thresh005_mean": float(np.mean(h1_counts_real_thresh)),
        "h1_count_thresh005_std": float(np.std(h1_counts_real_thresh)),
        "h1_count_thresh005_sem": float(np.std(h1_counts_real_thresh) / np.sqrt(len(h1_counts_real_thresh))),
        # distance matrix (for Mantel)
        "D_real": D_real,
        "elapsed_s": elapsed,
    }


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    rng = np.random.default_rng(cfg["seed"])

    meta = pd.read_csv(cfg["paths"]["meta"])
    files = [os.path.join(cfg["paths"]["audio"], f) for f in meta["filename"]]
    labels = meta[cfg["data"]["stratify_by"]].values
    genres = sorted(set(labels))

    print(f"Dataset: {len(files)} tracks, genres: {genres}")
    print(f"Config: pca_dim={cfg['common']['pca_dim']}, "
          f"method={cfg['pointcloud']['method']}, "
          f"window={cfg['pointcloud']['window']}, "
          f"stride={cfg['pointcloud']['stride']}, "
          f"n_points={cfg['pointcloud']['n_points']}, "
          f"shuffle_repeats={cfg['controls']['shuffle_repeats']}")

    # Train/test split
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(files))
    tr_idx, te_idx = train_test_split(
        idx, test_size=cfg["data"]["test_size"],
        stratify=labels, random_state=cfg["seed"])
    print(f"Train: {len(tr_idx)}, Test: {len(te_idx)}")

    # Process all spaces
    spaces = list(cfg["spaces"].keys())
    results = {}
    for space in spaces:
        results[space] = process_space(space, cfg, files, labels, tr_idx, rng)

    # --- Mantel matrix (cross_space_compare only) ---
    print(f"\n{'='*60}")
    print(f"  MANTEL CROSS-SPACE CONSISTENCY")
    print(f"{'='*60}")
    cross = cfg["common"].get("cross_space_compare", [])
    cross = [s for s in cross if s in results]  # filter to available
    n_perm_mantel = cfg["distances"].get("mantel_n_perm", 9999)
    n_boot_mantel = cfg["distances"].get("mantel_n_boot", 2000)
    mantel_rows = []  # list of dicts for DataFrame
    mantel_matrix = {}  # backward-compat summary dict
    for i, a in enumerate(cross):
        for j, b in enumerate(cross):
            if i < j:
                Da, Db = results[a]["D_real"], results[b]["D_real"]
                mantel_rng = np.random.default_rng(cfg["seed"])
                mt = distances.mantel_test(
                    Da, Db, n_perm=n_perm_mantel, rng=mantel_rng)
                ci_lo, ci_hi = distances.mantel_bootstrap_ci(
                    Da, Db, n_boot=n_boot_mantel,
                    rng=np.random.default_rng(cfg["seed"]))
                mantel_rows.append({
                    "pair": f"{a}_vs_{b}",
                    "r": float(mt["r"]),
                    "p": float(mt["p"]),
                    "ci_lo": float(ci_lo),
                    "ci_hi": float(ci_hi),
                })
                mantel_matrix[f"{a}_vs_{b}"] = {
                    "r": float(mt["r"]),
                    "p": float(mt["p"]),
                    "ci_lo": float(ci_lo),
                    "ci_hi": float(ci_hi),
                }
                print(f"  {a} ~ {b}: Mantel r={mt['r']:.4f}, "
                      f"p={mt['p']:.4f}, CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

    # --- Save results ---
    os.makedirs("results/tables", exist_ok=True)

    # Summary table
    rows = []
    for space in spaces:
        r = results[space]
        rows.append({
            "space": space,
            "n_tracks": r["n_tracks"],
            "pca_explained": f"{r['pca_explained']:.4f}",
            "pca_dim": r["pca_dim_actual"],
            # within/between
            "within_mean": f"{r['within_mean']:.4f}",
            "between_mean": f"{r['between_mean']:.4f}",
            "gap": f"{r['gap']:.4f}",
            "gap_obs_track": f"{r['gap_obs_track']:.4f}",
            "gap_ci95_track_lo": f"{r['gap_ci95_track'][0]:.4f}",
            "gap_ci95_track_hi": f"{r['gap_ci95_track'][1]:.4f}",
            # shuffle (per-track aggregated)
            "shuffle_dist_mean": f"{r['shuffle_dist_mean']:.4f}",
            "shuffle_dist_std": f"{r['shuffle_dist_std']:.4f}",
            "shuffle_dist_sem": f"{r['shuffle_dist_sem']:.4f}",
            "within_dist_mean": f"{r['within_dist_mean']:.4f}",
            "within_dist_std": f"{r['within_dist_std']:.4f}",
            "within_dist_sem": f"{r['within_dist_sem']:.4f}",
            "shuffle_p": f"{r['shuffle_p_value']:.2e}",
            "shuffle_effect_r": f"{r['shuffle_effect_size']:.4f}",
            # max persistence
            "max_pers_real_mean": f"{r['max_pers_real_mean']:.4f}",
            "max_pers_real_std": f"{r['max_pers_real_std']:.4f}",
            "max_pers_real_sem": f"{r['max_pers_real_sem']:.4f}",
            "max_pers_shuffle_mean": f"{r['max_pers_shuffle_mean']:.4f}",
            "max_pers_shuffle_std": f"{r['max_pers_shuffle_std']:.4f}",
            "max_pers_shuffle_sem": f"{r['max_pers_shuffle_sem']:.4f}",
            "max_pers_random_mean": f"{r['max_pers_random_mean']:.4f}",
            "max_pers_random_std": f"{r['max_pers_random_std']:.4f}",
            "max_pers_random_sem": f"{r['max_pers_random_sem']:.4f}",
            "max_pers_iaaft_mean": f"{r['max_pers_iaaft_mean']:.4f}",
            "max_pers_iaaft_std": f"{r['max_pers_iaaft_std']:.4f}",
            "max_pers_iaaft_sem": f"{r['max_pers_iaaft_sem']:.4f}",
            "real_vs_shuffle_p": f"{r['max_pers_real_vs_shuffle_p']:.2e}",
            "real_vs_shuffle_effect_r": f"{r['max_pers_real_vs_shuffle_effect']:.4f}",
            "real_vs_random_p": f"{r['max_pers_real_vs_random_p']:.2e}",
            "real_vs_random_effect_r": f"{r['max_pers_real_vs_random_effect']:.4f}",
            "real_vs_iaaft_p": f"{r['max_pers_real_vs_iaaft_p']:.2e}",
            "real_vs_iaaft_effect_r": f"{r['max_pers_real_vs_iaaft_effect']:.4f}",
            # H1 counts
            "h1_count_mean": f"{r['h1_count_mean']:.1f}",
            "h1_count_std": f"{r['h1_count_std']:.1f}",
            "h1_count_sem": f"{r['h1_count_sem']:.2f}",
            "h1_count_t005_mean": f"{r['h1_count_thresh005_mean']:.1f}",
            "h1_count_t005_std": f"{r['h1_count_thresh005_std']:.1f}",
            "h1_count_t005_sem": f"{r['h1_count_thresh005_sem']:.2f}",
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/tables/controls_summary.csv", index=False)
    print(f"\nSaved: results/tables/controls_summary.csv")

    # Mantel matrix (r, p, ci_lo, ci_hi per pair)
    if mantel_rows:
        mantel_df = pd.DataFrame(mantel_rows)
    else:
        mantel_df = pd.DataFrame(columns=["pair", "r", "p", "ci_lo", "ci_hi"])
    mantel_df.to_csv("results/tables/mantel_matrix.csv", index=False)
    print(f"Saved: results/tables/mantel_matrix.csv")

    # Full JSON results (without numpy arrays)
    json_results = {}
    for space in spaces:
        r = {k: v for k, v in results[space].items() if k != "D_real"}
        json_results[space] = r
    json_results["mantel"] = mantel_matrix
    with open("results/tables/controls_full.json", "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"Saved: results/tables/controls_full.json")

    # --- Print final summary table ---
    print(f"\n{'='*80}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\n--- Within vs Between (Wasserstein, H1) ---")
    print(f"{'Space':>10s}  {'Within':>10s}  {'Between':>10s}  {'Gap':>10s}  "
          f"{'Gap(trk)':>10s}  {'CI95 lo':>10s}  {'CI95 hi':>10s}")
    for space in spaces:
        r = results[space]
        print(f"{space:>10s}  {r['within_mean']:10.4f}  {r['between_mean']:10.4f}  "
              f"{r['gap']:10.4f}  {r['gap_obs_track']:10.4f}  "
              f"{r['gap_ci95_track'][0]:10.4f}  {r['gap_ci95_track'][1]:10.4f}")

    print(f"\n--- Shuffle Effect (Wasserstein, per-track aggregated) ---")
    print(f"{'Space':>10s}  {'real↔shuf':>16s}  {'SEM':>7s}  {'real↔real(w)':>16s}  {'SEM':>7s}  {'p-value':>12s}  {'r_rb':>7s}")
    for space in spaces:
        r = results[space]
        print(f"{space:>10s}  {r['shuffle_dist_mean']:7.4f}±{r['shuffle_dist_std']:.4f}  "
              f"{r['shuffle_dist_sem']:7.4f}  "
              f"{r['within_dist_mean']:7.4f}±{r['within_dist_std']:.4f}  "
              f"{r['within_dist_sem']:7.4f}  "
              f"{r['shuffle_p_value']:12.2e}  "
              f"{r['shuffle_effect_size']:7.4f}")

    print(f"\n--- Max Persistence H1: Real vs Shuffle vs Random vs IAAFT ---")
    print(f"{'Space':>10s}  {'Real':>16s}  {'SEM':>7s}  {'Shuffle':>16s}  {'SEM':>7s}  "
          f"{'Random':>16s}  {'SEM':>7s}  {'IAAFT':>16s}  {'SEM':>7s}")
    for space in spaces:
        r = results[space]
        print(f"{space:>10s}  "
              f"{r['max_pers_real_mean']:7.4f}±{r['max_pers_real_std']:.4f}  "
              f"{r['max_pers_real_sem']:7.4f}  "
              f"{r['max_pers_shuffle_mean']:7.4f}±{r['max_pers_shuffle_std']:.4f}  "
              f"{r['max_pers_shuffle_sem']:7.4f}  "
              f"{r['max_pers_random_mean']:7.4f}±{r['max_pers_random_std']:.4f}  "
              f"{r['max_pers_random_sem']:7.4f}  "
              f"{r['max_pers_iaaft_mean']:7.4f}±{r['max_pers_iaaft_std']:.4f}  "
              f"{r['max_pers_iaaft_sem']:7.4f}")
    print(f"{'Space':>10s}  {'p(R~Shuf)':>10s}  {'r_rb':>7s}  {'p(R~Rand)':>10s}  {'r_rb':>7s}  {'p(R~IAAFT)':>10s}  {'r_rb':>7s}")
    for space in spaces:
        r = results[space]
        print(f"{space:>10s}  "
              f"{r['max_pers_real_vs_shuffle_p']:10.2e}  "
              f"{r['max_pers_real_vs_shuffle_effect']:7.4f}  "
              f"{r['max_pers_real_vs_random_p']:10.2e}  "
              f"{r['max_pers_real_vs_random_effect']:7.4f}  "
              f"{r['max_pers_real_vs_iaaft_p']:10.2e}  "
              f"{r['max_pers_real_vs_iaaft_effect']:7.4f}")

    print(f"\n--- H1 Point Counts ---")
    print(f"{'Space':>10s}  {'H1 count':>16s}  {'SEM':>7s}  {'H1 (thresh>0.05)':>16s}  {'SEM':>7s}")
    for space in spaces:
        r = results[space]
        print(f"{space:>10s}  "
              f"{r['h1_count_mean']:7.1f}±{r['h1_count_std']:.1f}  "
              f"{r['h1_count_sem']:7.2f}        "
              f"{r['h1_count_thresh005_mean']:7.1f}±{r['h1_count_thresh005_std']:.1f}  "
              f"{r['h1_count_thresh005_sem']:7.2f}")

    print(f"\n--- Mantel (DL channels only: {cross}) ---")
    for pair, m in mantel_matrix.items():
        print(f"  {pair}: r={m['r']:.4f}, p={m['p']:.4f}, "
              f"CI=[{m['ci_lo']:.4f}, {m['ci_hi']:.4f}]")

    print(f"\nDone. All results in results/tables/")


if __name__ == "__main__":
    main()
