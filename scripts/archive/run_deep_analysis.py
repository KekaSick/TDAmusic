"""
run_deep_analysis.py
--------------------
Три отдельных прогона на подмножестве 60 треков.

ПРОГОН 1: MuQ layer ablation (8, 10, 12) — переизвлечение + полные контроли.
ПРОГОН 2: Takens PCA dim=20 vs baseline(null) — все 4 канала.
ПРОГОН 3: Within/between по парам жанров — 3 пары × 4 канала.

Каждый прогон — отдельная таблица в results/tables/.
"""
from __future__ import annotations
import os, sys, time, json, warnings, copy
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import embed_spaces, preprocess, pointcloud, persistence, controls, distances

from scipy import stats


# ====== Utility functions (same as run_controls.py) ======

def max_persistence_h1(dgm):
    if len(dgm) == 0: return 0.0
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0: return 0.0
    return float((finite[:, 1] - finite[:, 0]).max())

def h1_count(dgm, thresh=0.0):
    if len(dgm) == 0: return 0
    finite = dgm[np.isfinite(dgm[:, 1])]
    if thresh > 0 and len(finite) > 0:
        life = finite[:, 1] - finite[:, 0]
        finite = finite[life >= thresh]
    return len(finite)

def wasserstein_single(d1, d2):
    from persim import wasserstein
    return wasserstein(d1, d2)


def run_controls_block(preprocessed, labels, cfg, pc_cfg_override=None):
    """Run full controls block on preprocessed data. Returns dict of metrics.
    
    pc_cfg_override: if set, overrides cfg["pointcloud"] for this run.
    """
    pc_cfg = pc_cfg_override or cfg["pointcloud"]
    pers_cfg = cfg["persistence"]
    n_shuffle = cfg["controls"]["shuffle_repeats"]
    match_autocorr = cfg["controls"]["random_match_autocorr"]

    diags_real = []
    diags_shuffle = []
    diags_random = []
    h1_counts_real = []
    max_pers_real = []
    max_pers_random = []
    max_pers_shuffle = []

    for i, x in enumerate(preprocessed):
        # --- Real ---
        cloud = pointcloud.build_cloud(x, pc_cfg)
        # Optional: second PCA after Takens
        if pc_cfg.get("takens_pca_dim") is not None:
            from sklearn.decomposition import PCA as TPCA
            tdim = pc_cfg["takens_pca_dim"]
            if cloud.shape[1] > tdim:
                tpca = TPCA(n_components=tdim)
                cloud = tpca.fit_transform(cloud)

        res = persistence.compute_diagrams(
            cloud, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
        dgm_real = res["dgms"][1]
        diags_real.append(dgm_real)
        h1_counts_real.append(h1_count(dgm_real))
        max_pers_real.append(max_persistence_h1(dgm_real))

        # --- Shuffle ---
        track_shuffle_max_pers = []
        track_shuffle_dgms = []
        for s in range(n_shuffle):
            shuffle_rng = np.random.default_rng(cfg["seed"] * 1000 + i * 100 + s)
            x_shuf = controls.shuffle_frames(x, shuffle_rng)
            cloud_shuf = pointcloud.build_cloud(x_shuf, pc_cfg)
            if pc_cfg.get("takens_pca_dim") is not None:
                tdim = pc_cfg["takens_pca_dim"]
                if cloud_shuf.shape[1] > tdim:
                    tpca_s = TPCA(n_components=tdim)
                    cloud_shuf = tpca_s.fit_transform(cloud_shuf)
            res_shuf = persistence.compute_diagrams(
                cloud_shuf, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
            track_shuffle_dgms.append(res_shuf["dgms"][1])
            track_shuffle_max_pers.append(max_persistence_h1(res_shuf["dgms"][1]))
        diags_shuffle.append(track_shuffle_dgms)
        max_pers_shuffle.append(np.mean(track_shuffle_max_pers))

        # --- Random ---
        random_rng = np.random.default_rng(cfg["seed"] * 2000 + i)
        x_rand = controls.random_like(x, random_rng, match_autocorr)
        cloud_rand = pointcloud.build_cloud(x_rand, pc_cfg)
        if pc_cfg.get("takens_pca_dim") is not None:
            tdim = pc_cfg["takens_pca_dim"]
            if cloud_rand.shape[1] > tdim:
                tpca_r = TPCA(n_components=tdim)
                cloud_rand = tpca_r.fit_transform(cloud_rand)
        res_rand = persistence.compute_diagrams(
            cloud_rand, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
        diags_random.append(res_rand["dgms"][1])
        max_pers_random.append(max_persistence_h1(res_rand["dgms"][1]))

        if (i + 1) % 10 == 0 or i == 0:
            print(f"    [{i+1}/{len(preprocessed)}] H1={h1_count(dgm_real)}, "
                  f"maxP={max_pers_real[-1]:.4f}", flush=True)

    # --- Pairwise Wasserstein ---
    D_real = distances.pairwise_wasserstein(diags_real)
    w, b, gap = distances.within_between(D_real, labels)
    gap_mean, gap_ci = distances.bootstrap_gap(
        D_real, labels, cfg["distances"]["bootstrap"],
        np.random.default_rng(cfg["seed"]))

    # --- Shuffle effect ---
    real_vs_shuffle_dists = []
    for i in range(len(diags_real)):
        for dgm_shuf in diags_shuffle[i]:
            d = wasserstein_single(diags_real[i], dgm_shuf)
            real_vs_shuffle_dists.append(d)
    real_vs_shuffle_dists = np.array(real_vs_shuffle_dists)
    iu = np.triu_indices_from(D_real, k=1)
    same_genre = labels[iu[0]] == labels[iu[1]]
    real_vs_real_within = D_real[iu][same_genre]

    _, p_shuffle = stats.mannwhitneyu(
        real_vs_shuffle_dists, real_vs_real_within, alternative="greater")

    # --- Max pers ---
    mp_real = np.array(max_pers_real)
    mp_random = np.array(max_pers_random)
    mp_shuffle = np.array(max_pers_shuffle)
    _, p_rand = stats.wilcoxon(mp_real, mp_random, alternative="two-sided")
    _, p_shuf_mp = stats.wilcoxon(mp_real, mp_shuffle, alternative="two-sided")

    return {
        "within": float(w), "between": float(b), "gap": float(gap),
        "gap_boot": float(gap_mean),
        "gap_ci95_lo": float(gap_ci[0]), "gap_ci95_hi": float(gap_ci[1]),
        "shuffle_dist_mean": float(real_vs_shuffle_dists.mean()),
        "within_dist_mean": float(real_vs_real_within.mean()),
        "shuffle_p": float(p_shuffle),
        "max_pers_real": float(mp_real.mean()),
        "max_pers_real_std": float(mp_real.std()),
        "max_pers_random": float(mp_random.mean()),
        "max_pers_random_std": float(mp_random.std()),
        "max_pers_shuffle": float(mp_shuffle.mean()),
        "real_vs_random_p": float(p_rand),
        "real_vs_shuffle_p": float(p_shuf_mp),
        "h1_count_mean": float(np.mean(h1_counts_real)),
        "h1_count_std": float(np.std(h1_counts_real)),
        "D_real": D_real,
    }


def preprocess_space(space_name, cfg, files, tr_idx, raw=None, cache_dir_override=None):
    """Load raw embeddings and run PCA. Returns (preprocessed, reducer, raw)."""
    space_cfg = cfg["spaces"][space_name]
    common_cfg = cfg["common"]

    if raw is None:
        print(f"  Loading {len(files)} cached embeddings...", flush=True)
        raw = []
        # override cache dir for loading
        orig_cache = cfg["paths"]["cache"]
        if cache_dir_override:
            cfg["paths"]["cache"] = cache_dir_override
        for f in files:
            arr = embed_spaces.extract(f, space_name, cfg)
            raw.append(arr)
        if cache_dir_override:
            cfg["paths"]["cache"] = orig_cache

    print(f"  Fitting PCA on train ({len(tr_idx)} tracks)...", flush=True)
    reducer = preprocess.PCAReducer(
        common_cfg["pca_dim"],
        standardize=common_cfg.get("standardize_features", True),
        normalize=common_cfg["normalize"])
    tr_frames = [
        preprocess.resample_fps(
            raw[i],
            space_cfg.get("native_fps", common_cfg["target_fps"]),
            common_cfg["target_fps"])
        for i in tr_idx
    ]
    reducer.fit(tr_frames)
    print(f"    PCA explained: {reducer.explained:.4f} ({reducer.dim} dims)", flush=True)

    preprocessed = []
    for x_raw in raw:
        x = preprocess.prepare_track(x_raw, space_cfg, common_cfg, reducer)
        preprocessed.append(x)

    return preprocessed, reducer, raw


# =====================================================================
# ПРОГОН 1: MuQ layer ablation
# =====================================================================

def extract_muq_layer(wav, sr, cfg, layer):
    """Extract MuQ at a specific layer."""
    from muq import MuQ
    key = "muq"
    if key not in embed_spaces._MODELS:
        mid = cfg["spaces"]["muq"]["model_id"]
        embed_spaces._MODELS[key] = MuQ.from_pretrained(mid).to(embed_spaces._device()).eval()
    model = embed_spaces._MODELS[key]
    import torch
    wavs = torch.tensor(wav).unsqueeze(0).to(embed_spaces._device())
    with torch.no_grad():
        out = model(wavs, output_hidden_states=True)
    h = out.hidden_states[layer].squeeze(0)
    return h.cpu().numpy()


def run_muq_ablation(cfg, files, labels, tr_idx):
    """ПРОГОН 1: MuQ layers 8, 10, 12."""
    print(f"\n{'#'*70}")
    print(f"  ПРОГОН 1: MuQ Layer Ablation (layers 8, 10, 12)")
    print(f"{'#'*70}")

    layers = [8, 10, 12]
    results_rows = []

    for layer in layers:
        cache_sub = f"muq_l{layer}"
        cache_dir = os.path.join(cfg["paths"]["cache"], cache_sub)
        os.makedirs(cache_dir, exist_ok=True)
        print(f"\n--- Layer {layer} (cache: {cache_sub}) ---", flush=True)

        # Check what's cached
        cached = sum(1 for f in files
                     if os.path.exists(os.path.join(cache_dir,
                        os.path.splitext(os.path.basename(f))[0] + ".npy")))
        need = len(files) - cached
        print(f"  Cached: {cached}/{len(files)}, need to extract: {need}", flush=True)

        # Extract missing
        if need > 0:
            sr = cfg["data"]["sample_rate"]
            import librosa as _lr
            for fi, fpath in enumerate(files):
                base = os.path.splitext(os.path.basename(fpath))[0]
                npy_path = os.path.join(cache_dir, f"{base}.npy")
                if os.path.exists(npy_path):
                    continue
                wav, _ = _lr.load(fpath, sr=sr, mono=True)
                arr = extract_muq_layer(wav, sr, cfg, layer)
                np.save(npy_path, arr)
                if (fi + 1) % 10 == 0:
                    print(f"    Extracted [{fi+1}/{len(files)}]", flush=True)

        # If layer 10, we can also use existing muq cache
        if layer == 10:
            # Check if muq_l10 has data, else copy from muq/
            orig_dir = os.path.join(cfg["paths"]["cache"], "muq")
            for f in files:
                base = os.path.splitext(os.path.basename(f))[0]
                src = os.path.join(orig_dir, f"{base}.npy")
                dst = os.path.join(cache_dir, f"{base}.npy")
                if not os.path.exists(dst) and os.path.exists(src):
                    import shutil
                    shutil.copy2(src, dst)

        # Load from cache
        raw = []
        for f in files:
            base = os.path.splitext(os.path.basename(f))[0]
            npy_path = os.path.join(cache_dir, f"{base}.npy")
            raw.append(np.load(npy_path))

        # Preprocess with muq config
        preprocessed, reducer, _ = preprocess_space("muq", cfg, files, tr_idx, raw=raw)
        print(f"  Running controls block...", flush=True)
        t0 = time.time()
        r = run_controls_block(preprocessed, labels, cfg)
        elapsed = time.time() - t0
        print(f"  Layer {layer} done in {elapsed:.1f}s", flush=True)

        results_rows.append({
            "layer": layer,
            "gap": f"{r['gap']:.4f}",
            "gap_boot": f"{r['gap_boot']:.4f}",
            "gap_ci95": f"[{r['gap_ci95_lo']:.4f}, {r['gap_ci95_hi']:.4f}]",
            "shuffle_p": f"{r['shuffle_p']:.2e}",
            "max_pers_real": f"{r['max_pers_real']:.4f}±{r['max_pers_real_std']:.4f}",
            "max_pers_random": f"{r['max_pers_random']:.4f}±{r['max_pers_random_std']:.4f}",
            "real_vs_random_p": f"{r['real_vs_random_p']:.2e}",
            "real_vs_shuffle_p": f"{r['real_vs_shuffle_p']:.2e}",
            "h1_count": f"{r['h1_count_mean']:.1f}±{r['h1_count_std']:.1f}",
            "pca_expl": f"{reducer.explained:.4f}",
        })

    df = pd.DataFrame(results_rows)
    df.to_csv("results/tables/ablation_muq_layers.csv", index=False)
    print(f"\n  Saved: results/tables/ablation_muq_layers.csv")
    print(df.to_string(index=False))
    return df


# =====================================================================
# ПРОГОН 2: Takens PCA dim=20 vs baseline
# =====================================================================

def run_takens_pca_compare(cfg, files, labels, tr_idx, baseline_results):
    """ПРОГОН 2: all 4 channels with takens_pca_dim=20 vs baseline(null)."""
    print(f"\n{'#'*70}")
    print(f"  ПРОГОН 2: Takens PCA dim=20 vs baseline (null)")
    print(f"{'#'*70}")

    spaces = list(cfg["spaces"].keys())
    results_rows = []

    pc_cfg_pca20 = copy.deepcopy(cfg["pointcloud"])
    pc_cfg_pca20["takens_pca_dim"] = 20

    for space in spaces:
        print(f"\n--- {space} (takens_pca_dim=20) ---", flush=True)
        preprocessed, reducer, _ = preprocess_space(space, cfg, files, tr_idx)
        t0 = time.time()
        r = run_controls_block(preprocessed, labels, cfg, pc_cfg_override=pc_cfg_pca20)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s", flush=True)

        # Baseline values from run_controls.py results
        b = baseline_results.get(space, {})

        results_rows.append({
            "space": space,
            # baseline
            "gap_baseline": b.get("gap", ""),
            "gap_ci95_baseline": b.get("gap_ci95", ""),
            "real_vs_random_p_baseline": b.get("real_vs_random_p", ""),
            "shuffle_p_baseline": b.get("shuffle_p", ""),
            "max_pers_real_baseline": b.get("max_pers_real", ""),
            # pca20
            "gap_pca20": f"{r['gap']:.4f}",
            "gap_boot_pca20": f"{r['gap_boot']:.4f}",
            "gap_ci95_pca20": f"[{r['gap_ci95_lo']:.4f}, {r['gap_ci95_hi']:.4f}]",
            "real_vs_random_p_pca20": f"{r['real_vs_random_p']:.2e}",
            "shuffle_p_pca20": f"{r['shuffle_p']:.2e}",
            "max_pers_real_pca20": f"{r['max_pers_real']:.4f}±{r['max_pers_real_std']:.4f}",
            "max_pers_random_pca20": f"{r['max_pers_random']:.4f}±{r['max_pers_random_std']:.4f}",
            "max_pers_shuffle_pca20": f"{r['max_pers_shuffle']:.4f}",
            "real_vs_shuffle_p_pca20": f"{r['real_vs_shuffle_p']:.2e}",
            "h1_count_pca20": f"{r['h1_count_mean']:.1f}±{r['h1_count_std']:.1f}",
            "shuffle_breaks_h1": "YES" if r['shuffle_p'] < 0.001 else "NO ⚠️",
        })

    df = pd.DataFrame(results_rows)
    df.to_csv("results/tables/takens_pca_compare.csv", index=False)
    print(f"\n  Saved: results/tables/takens_pca_compare.csv")
    print(df.to_string(index=False))
    return df


# =====================================================================
# ПРОГОН 3: within/between по парам жанров
# =====================================================================

def run_genre_pairs(cfg, files, labels, tr_idx):
    """ПРОГОН 3: break between into 3 genre pairs."""
    print(f"\n{'#'*70}")
    print(f"  ПРОГОН 3: Within/Between по парам жанров")
    print(f"{'#'*70}")

    genres = sorted(set(labels))
    pairs = [(genres[i], genres[j]) for i in range(len(genres))
             for j in range(i+1, len(genres))]
    print(f"  Genres: {genres}")
    print(f"  Pairs: {pairs}")

    spaces = list(cfg["spaces"].keys())
    results_rows = []

    for space in spaces:
        print(f"\n--- {space} ---", flush=True)
        preprocessed, reducer, _ = preprocess_space(space, cfg, files, tr_idx)

        # Compute real diagrams only (no shuffle/random needed)
        pc_cfg = cfg["pointcloud"]
        pers_cfg = cfg["persistence"]
        diags_real = []
        for i, x in enumerate(preprocessed):
            cloud = pointcloud.build_cloud(x, pc_cfg)
            res = persistence.compute_diagrams(
                cloud, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
            diags_real.append(res["dgms"][1])

        D = distances.pairwise_wasserstein(diags_real)

        # Within per genre
        iu = np.triu_indices_from(D, k=1)
        for g in genres:
            mask_g = labels == g
            idx_g = np.where(mask_g)[0]
            dists_within = []
            for a in range(len(idx_g)):
                for b in range(a+1, len(idx_g)):
                    dists_within.append(D[idx_g[a], idx_g[b]])
            results_rows.append({
                "space": space,
                "type": "within",
                "genre_a": g,
                "genre_b": g,
                "mean_dist": f"{np.mean(dists_within):.4f}",
                "std_dist": f"{np.std(dists_within):.4f}",
                "n_pairs": len(dists_within),
            })

        # Between per pair
        for ga, gb in pairs:
            idx_a = np.where(labels == ga)[0]
            idx_b = np.where(labels == gb)[0]
            dists_between = []
            for a in idx_a:
                for b in idx_b:
                    dists_between.append(D[a, b])
            results_rows.append({
                "space": space,
                "type": "between",
                "genre_a": ga,
                "genre_b": gb,
                "mean_dist": f"{np.mean(dists_between):.4f}",
                "std_dist": f"{np.std(dists_between):.4f}",
                "n_pairs": len(dists_between),
            })

        print(f"  Done.", flush=True)

    df = pd.DataFrame(results_rows)
    df.to_csv("results/tables/genre_pairs.csv", index=False)
    print(f"\n  Saved: results/tables/genre_pairs.csv")

    # Print nicely
    for space in spaces:
        sub = df[df["space"] == space]
        print(f"\n  {space}:")
        for _, row in sub.iterrows():
            tag = f"{row['genre_a']}-{row['genre_b']}" if row["type"] == "between" else f"{row['genre_a']} (within)"
            print(f"    {tag:25s}  {row['mean_dist']} ± {row['std_dist']}  (n={row['n_pairs']})")
    return df


# =====================================================================
# MAIN
# =====================================================================

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    rng = np.random.default_rng(cfg["seed"])
    meta = pd.read_csv(cfg["paths"]["meta"])
    files = [os.path.join(cfg["paths"]["audio"], f) for f in meta["filename"]]
    labels = meta[cfg["data"]["stratify_by"]].values

    from sklearn.model_selection import train_test_split
    idx = np.arange(len(files))
    tr_idx, te_idx = train_test_split(
        idx, test_size=cfg["data"]["test_size"],
        stratify=labels, random_state=cfg["seed"])

    os.makedirs("results/tables", exist_ok=True)

    # Load baseline results from Stage 3 for comparison in Прогон 2
    baseline_results = {}
    try:
        with open("results/tables/controls_full.json") as f:
            bj = json.load(f)
        for sp in ["mert", "muq", "encodec", "mir"]:
            if sp in bj:
                b = bj[sp]
                baseline_results[sp] = {
                    "gap": f"{b['gap']:.4f}",
                    "gap_ci95": f"[{b['gap_ci95'][0]:.4f}, {b['gap_ci95'][1]:.4f}]",
                    "real_vs_random_p": f"{b['max_pers_real_vs_random_p']:.2e}",
                    "shuffle_p": f"{b['shuffle_p_value']:.2e}",
                    "max_pers_real": f"{b['max_pers_real_mean']:.4f}±{b['max_pers_real_std']:.4f}",
                    "max_pers_random": f"{b['max_pers_random_mean']:.4f}±{b['max_pers_random_std']:.4f}",
                }
    except Exception as e:
        print(f"  Warning: could not load baseline results: {e}")

    # --- ПРОГОН 1 ---
    run_muq_ablation(cfg, files, labels, tr_idx)

    # --- ПРОГОН 2 ---
    run_takens_pca_compare(cfg, files, labels, tr_idx, baseline_results)

    # --- ПРОГОН 3 ---
    run_genre_pairs(cfg, files, labels, tr_idx)

    print(f"\n{'='*70}")
    print(f"  ALL DONE. Results in results/tables/")
    print(f"    - ablation_muq_layers.csv")
    print(f"    - takens_pca_compare.csv")
    print(f"    - genre_pairs.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
