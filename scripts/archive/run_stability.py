"""
run_stability.py
----------------
Multi-seed stability of "loop = repeat" significance on CYCLE representatives.
Two sources of variation, tested independently:
  A) PCA bootstrap: resample 199 training tracks (with replacement) → different
     PCA reducer → different cloud → different cycle → different p-value.
  B) Maxmin seed: fixed PCA, vary maxmin starting point (seed 0..9) →
     different 300-point subsample → different cycle → different p-value.

Output: fraction of seeds where each track is significant (p < 0.05).

    .venv/bin/python scripts/run_stability.py
"""
import os
from tqdm import tqdm
import sys
sys.path.insert(0, "src")
import yaml
import glob
import csv
import time
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import mannwhitneyu
import dionysus as d

import preprocess
import pointcloud

# ======================================================================
# Config
# ======================================================================

CACHE_DIR = "cache/muq_spotify90"
CLIP_SEC = 90
N_SEEDS = 10

TRACKS = [
    "pop_29_Something_Just_Like_This_The_Chainsmokers",
    "electronic_21_Turn_Down_for_What_DJ_Snake",
    "hip-hop_30_The_Thrill_Wiz_Khalifa",
    "reggae_14_Dile_Don_Omar",
    "hip-hop_21_Passionfruit_Drake",
    "reggae_30_frente_al_mar_Beéle",
    "hip-hop_07_HUMBLE__Kendrick_Lamar",
    "electronic_10_Closer_Nine_Inch_Nails",
]


def load_cfg():
    return yaml.safe_load(open("config.yaml"))


# ======================================================================
# Core: cycle extraction (Dionysus homology only — no ripser/cocycle)
# ======================================================================

def compute_cycle(cloud):
    """Compute representative cycle of most persistent H1 feature."""
    from scipy.spatial.distance import pdist
    dists = pdist(cloud)
    max_radius = float(np.percentile(dists, 30))

    f = d.fill_rips(cloud, 2, max_radius)
    f.sort()

    m = d.homology_persistence(f, prime=2, progress=False)
    dgms = d.init_diagrams(m, f)
    h1 = [pt for pt in dgms[1] if pt.death < float("inf")]

    if not h1:
        # Try larger radius
        max_radius = float(np.percentile(dists, 50))
        f = d.fill_rips(cloud, 2, max_radius)
        f.sort()
        m = d.homology_persistence(f, prime=2, progress=False)
        dgms = d.init_diagrams(m, f)
        h1 = [pt for pt in dgms[1] if pt.death < float("inf")]

    if not h1:
        return {"verts": [], "persistence": 0, "birth": 0, "death": 0}

    best = max(h1, key=lambda p: p.death - p.birth)
    death_idx = best.data
    birth_idx = m.pair(death_idx)
    chain = m[birth_idx]

    verts = set()
    for entry in chain:
        s = f[entry.index]
        vs = list(s)
        if len(vs) == 2:
            verts.update(vs)

    return {
        "verts": sorted(verts),
        "persistence": float(best.death - best.birth),
        "birth": float(best.birth),
        "death": float(best.death),
    }


# ======================================================================
# Vertex → seconds + chroma test (same as run_cycle_analysis.py)
# ======================================================================

def vertices_to_seconds(vertex_ids, sub_indices, takens_starts,
                        window, target_fps):
    seconds = []
    for v in vertex_ids:
        takens_row = sub_indices[v]
        start_frame = takens_starts[takens_row]
        sec = (start_frame + window / 2) / target_fps
        seconds.append(sec)
    return np.sort(seconds)


def chroma_test(wav, sr, seconds, window_sec, hop_length=512,
                distant_thresh=5.0, n_random=500, seed=42):
    chroma_full = librosa.feature.chroma_cens(y=wav, sr=sr, hop_length=hop_length)
    chroma_fps = sr / hop_length
    n_chroma = chroma_full.shape[1]
    clip_dur = len(wav) / sr

    half_win = window_sec / 2
    def get_chroma(sec):
        sf = max(0, int((sec - half_win) * chroma_fps))
        ef = min(n_chroma, int((sec + half_win) * chroma_fps) + 1)
        if ef <= sf:
            ef = sf + 1
        return chroma_full[:, sf:ef].mean(axis=1)

    loop_chroma = np.array([get_chroma(s) for s in seconds])

    distant_sims = []
    for i in range(len(seconds)):
        for j in range(i + 1, len(seconds)):
            if abs(seconds[i] - seconds[j]) >= distant_thresh:
                sim = 1.0 - cosine_dist(loop_chroma[i], loop_chroma[j])
                distant_sims.append(sim)

    rng = np.random.default_rng(seed)
    all_times = np.linspace(window_sec / 2, clip_dur - window_sec / 2, 200)
    all_chroma = np.array([get_chroma(s) for s in all_times])

    random_sims = []
    attempts = 0
    while len(random_sims) < n_random and attempts < n_random * 20:
        a, b = rng.choice(len(all_times), 2, replace=False)
        if abs(all_times[a] - all_times[b]) >= distant_thresh:
            sim = 1.0 - cosine_dist(all_chroma[a], all_chroma[b])
            random_sims.append(sim)
        attempts += 1

    distant_sims = np.array(distant_sims)
    random_sims = np.array(random_sims)

    p = np.nan
    if len(distant_sims) >= 3 and len(random_sims) >= 3:
        _, p = mannwhitneyu(distant_sims, random_sims, alternative="greater")

    return {
        "p_value": float(p),
        "significant": bool(p < 0.05) if not np.isnan(p) else False,
        "n_distant": len(distant_sims),
    }


# ======================================================================
# Build cloud with custom maxmin seed
# ======================================================================

def build_cloud_custom_seed(x_pca, pc_cfg, maxmin_seed=0):
    """build_cloud_with_indices but with configurable maxmin seed."""
    window = pc_cfg["window"]
    stride = pc_cfg["stride"]
    n_points = pc_cfg["n_points"]

    cloud_full = pointcloud.takens(x_pca, window, stride)
    T = x_pca.shape[0]
    takens_starts = np.arange(0, T - window + 1, stride)

    if cloud_full.shape[0] <= n_points:
        return cloud_full, takens_starts, np.arange(cloud_full.shape[0])

    # maxmin with custom seed
    cloud_sub, sub_indices = pointcloud._maxmin_with_indices(
        cloud_full, n_points, seed=maxmin_seed)
    return cloud_sub, takens_starts, sub_indices


# ======================================================================
# Main
# ======================================================================

def main():
    cfg = load_cfg()
    sr = cfg["data"]["sample_rate"]
    target_fps = cfg["common"]["target_fps"]
    hop_length = cfg["spaces"]["mir"]["hop_length"]
    pc_cfg = cfg["pointcloud"].copy()
    pc_cfg["takens_pca_dim"] = None
    window = pc_cfg["window"]
    window_sec = window / target_fps

    # Load all embeddings (for PCA fitting)
    print("Loading cached MuQ embeddings...")
    all_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npy")))
    all_embs = [np.load(f) for f in all_files]
    n_tracks_total = len(all_embs)
    print(f"  {n_tracks_total} tracks")

    # Load fixed reducer for Source B
    REDUCER_PATH = "cache/pca_reducer_muq_spotify90.joblib"
    fixed_reducer = preprocess.PCAReducer.load(REDUCER_PATH)
    print(f"  Fixed PCA: dim={fixed_reducer.dim}, explained={fixed_reducer.explained:.4f}")

    # Preload audio for 8 tracks
    print("Loading audio...")
    track_audio = {}
    for track_base in TRACKS:
        wav_path = None
        for genre in ["pop", "electronic", "hip-hop", "reggae"]:
            c = os.path.join("data/top50musicSpotify", genre, track_base + ".wav")
            if os.path.exists(c):
                wav_path = c
                break
        if wav_path is None:
            print(f"  SKIP: {track_base} — wav not found")
            continue
        wav_full, _ = librosa.load(wav_path, sr=sr, mono=True)
        total_sec = len(wav_full) / sr
        if total_sec > CLIP_SEC:
            start = int((total_sec - CLIP_SEC) / 2 * sr)
            wav = wav_full[start:start + int(CLIP_SEC * sr)]
        else:
            wav = wav_full
        track_audio[track_base] = wav
        # Pre-compute chroma (expensive, do once)
    print(f"  Loaded audio for {len(track_audio)} tracks")

    # Preload embeddings for 8 tracks
    track_embs = {}
    for tb in TRACKS:
        p = os.path.join(CACHE_DIR, f"{tb}.npy")
        if os.path.exists(p):
            track_embs[tb] = np.load(p)

    # ================================================================
    # SOURCE A: PCA bootstrap (10 seeds)
    # ================================================================
    print(f"\n{'='*80}")
    print(f"SOURCE A: PCA bootstrap ({N_SEEDS} seeds)")
    print(f"{'='*80}")

    results_a = {tb: [] for tb in TRACKS}
    pbar_a = tqdm(total=N_SEEDS * len(TRACKS), desc="Source A (PCA)", unit="run")

    for seed_idx in range(N_SEEDS):
        pca_seed = seed_idx * 7 + 1  # diverse seeds

        # Bootstrap: resample training tracks with replacement
        rng = np.random.default_rng(pca_seed)
        boot_indices = rng.choice(n_tracks_total, n_tracks_total, replace=True)
        boot_embs = [all_embs[i] for i in boot_indices]

        # Fit PCA on bootstrap sample
        reducer = preprocess.PCAReducer(
            dim=cfg["common"]["pca_dim"],
            standardize=cfg["common"].get("standardize_features", True),
            normalize=cfg["common"]["normalize"],
        )
        reducer.fit(boot_embs)

        for tb in TRACKS:
            pbar_a.set_postfix_str(f"seed={seed_idx} {tb[:25]}")
            if tb not in track_embs or tb not in track_audio:
                results_a[tb].append({"p": np.nan, "sig": False, "span": 0,
                                       "n_verts": 0, "pers": 0})
                pbar_a.update(1)
                continue

            x_pca = preprocess.prepare_track(
                track_embs[tb], cfg["spaces"]["muq"], cfg["common"], reducer)
            cloud, t_starts, sub_idx = build_cloud_custom_seed(
                x_pca, pc_cfg, maxmin_seed=0)  # fixed maxmin seed

            cy = compute_cycle(cloud)
            if not cy["verts"]:
                results_a[tb].append({"p": np.nan, "sig": False, "span": 0,
                                       "n_verts": 0, "pers": cy["persistence"]})
                pbar_a.update(1)
                continue

            secs = vertices_to_seconds(cy["verts"], sub_idx, t_starts,
                                        window, target_fps)
            span = float(secs.max() - secs.min()) if len(secs) > 1 else 0

            ct = chroma_test(track_audio[tb], sr, secs, window_sec, hop_length)

            results_a[tb].append({
                "p": ct["p_value"], "sig": ct["significant"],
                "span": span, "n_verts": len(cy["verts"]),
                "pers": cy["persistence"],
            })
            pbar_a.update(1)

    pbar_a.close()

    # ================================================================
    # SOURCE B: Maxmin seed (10 seeds, fixed PCA)
    # ================================================================
    print(f"\n{'='*80}")
    print(f"SOURCE B: Maxmin seed ({N_SEEDS} seeds, fixed PCA)")
    print(f"{'='*80}")

    results_b = {tb: [] for tb in TRACKS}
    pbar_b = tqdm(total=N_SEEDS * len(TRACKS), desc="Source B (maxmin)", unit="run")

    for seed_idx in range(N_SEEDS):
        maxmin_seed = seed_idx

        for tb in TRACKS:
            pbar_b.set_postfix_str(f"seed={seed_idx} {tb[:25]}")
            if tb not in track_embs or tb not in track_audio:
                results_b[tb].append({"p": np.nan, "sig": False, "span": 0,
                                       "n_verts": 0, "pers": 0})
                pbar_b.update(1)
                continue

            x_pca = preprocess.prepare_track(
                track_embs[tb], cfg["spaces"]["muq"], cfg["common"],
                fixed_reducer)
            cloud, t_starts, sub_idx = build_cloud_custom_seed(
                x_pca, pc_cfg, maxmin_seed=maxmin_seed)

            cy = compute_cycle(cloud)
            if not cy["verts"]:
                results_b[tb].append({"p": np.nan, "sig": False, "span": 0,
                                       "n_verts": 0, "pers": cy["persistence"]})
                pbar_b.update(1)
                continue

            secs = vertices_to_seconds(cy["verts"], sub_idx, t_starts,
                                        window, target_fps)
            span = float(secs.max() - secs.min()) if len(secs) > 1 else 0

            ct = chroma_test(track_audio[tb], sr, secs, window_sec, hop_length)

            results_b[tb].append({
                "p": ct["p_value"], "sig": ct["significant"],
                "span": span, "n_verts": len(cy["verts"]),
                "pers": cy["persistence"],
            })
            pbar_b.update(1)

    pbar_b.close()

    # ================================================================
    # Aggregate
    # ================================================================
    print(f"\n\n{'='*100}")
    print("STABILITY SUMMARY")
    print(f"{'='*100}")

    out_rows = []
    hdr = (f"{'Track':42s} | {'A sig':>5} {'A span':>12} | "
           f"{'B sig':>5} {'B span':>12} | {'Verdict':>14}")
    print(hdr)
    print("-" * len(hdr))

    for tb in TRACKS:
        # Source A
        a_sigs = [r["sig"] for r in results_a[tb] if not np.isnan(r["p"])]
        a_spans = [r["span"] for r in results_a[tb] if r["span"] > 0]
        a_frac = sum(a_sigs) / len(a_sigs) if a_sigs else 0
        a_span_mean = np.mean(a_spans) if a_spans else 0
        a_span_std = np.std(a_spans) if a_spans else 0

        # Source B
        b_sigs = [r["sig"] for r in results_b[tb] if not np.isnan(r["p"])]
        b_spans = [r["span"] for r in results_b[tb] if r["span"] > 0]
        b_frac = sum(b_sigs) / len(b_sigs) if b_sigs else 0
        b_span_mean = np.mean(b_spans) if b_spans else 0
        b_span_std = np.std(b_spans) if b_spans else 0

        # Verdict (use max of both sources)
        max_frac = max(a_frac, b_frac)
        if max_frac >= 0.8:
            verdict = "STABLE SIG"
        elif max_frac <= 0.2:
            verdict = "STABLE NONSIG"
        else:
            verdict = "UNSTABLE"

        a_sig_str = f"{sum(a_sigs)}/{len(a_sigs)}" if a_sigs else "n/a"
        b_sig_str = f"{sum(b_sigs)}/{len(b_sigs)}" if b_sigs else "n/a"
        a_sp_str = f"{a_span_mean:.1f}±{a_span_std:.1f}s"
        b_sp_str = f"{b_span_mean:.1f}±{b_span_std:.1f}s"

        t = tb[:42]
        print(f"{t:42s} | {a_sig_str:>5} {a_sp_str:>12} | "
              f"{b_sig_str:>5} {b_sp_str:>12} | {verdict:>14}")

        out_rows.append({
            "track": tb,
            "genre": tb.split("_")[0],
            "a_n_sig": sum(a_sigs) if a_sigs else 0,
            "a_n_total": len(a_sigs),
            "a_frac_sig": a_frac,
            "a_span_mean": a_span_mean,
            "a_span_std": a_span_std,
            "b_n_sig": sum(b_sigs) if b_sigs else 0,
            "b_n_total": len(b_sigs),
            "b_frac_sig": b_frac,
            "b_span_mean": b_span_mean,
            "b_span_std": b_span_std,
            "verdict": verdict,
        })

    # Overall
    n_stable_sig = sum(1 for r in out_rows if r["verdict"] == "STABLE SIG")
    n_stable_nonsig = sum(1 for r in out_rows if r["verdict"] == "STABLE NONSIG")
    n_unstable = sum(1 for r in out_rows if r["verdict"] == "UNSTABLE")

    print(f"\nOverall: {n_stable_sig} stable-sig, {n_stable_nonsig} stable-nonsig, "
          f"{n_unstable} unstable out of {len(out_rows)} tracks")

    # Which source is more unstable?
    a_flips = sum(1 for r in out_rows if 0.2 < r["a_frac_sig"] < 0.8)
    b_flips = sum(1 for r in out_rows if 0.2 < r["b_frac_sig"] < 0.8)
    print(f"Source A (PCA bootstrap): {a_flips} tracks with unstable significance")
    print(f"Source B (maxmin seed):   {b_flips} tracks with unstable significance")

    # Save CSV
    out_dir = os.path.join(cfg["paths"]["results"], "tables")
    os.makedirs(out_dir, exist_ok=True)
    csv_out = os.path.join(out_dir, "stability_cycle.csv")
    fieldnames = list(out_rows[0].keys())
    with open(csv_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\n→ {csv_out}")

    # Save per-seed detail
    csv_detail = os.path.join(out_dir, "stability_cycle_detail.csv")
    detail_rows = []
    for source, results in [("A_pca", results_a), ("B_maxmin", results_b)]:
        for tb in TRACKS:
            for seed_idx, r in enumerate(results[tb]):
                detail_rows.append({
                    "source": source, "track": tb, "seed": seed_idx,
                    "p_value": r["p"], "significant": r["sig"],
                    "span": r["span"], "n_verts": r["n_verts"],
                    "persistence": r["pers"],
                })
    det_fields = ["source", "track", "seed", "p_value", "significant",
                   "span", "n_verts", "persistence"]
    with open(csv_detail, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=det_fields)
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"→ {csv_detail}")

    print("\nDone!")


if __name__ == "__main__":
    main()
