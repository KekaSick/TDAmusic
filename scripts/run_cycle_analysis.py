"""
run_cycle_analysis.py  (v3 — full population, no cherry-picking)
-----------------------------------------------------------------
Cycle vs Cocycle on FIXED environment (numpy 1.26.4, sklearn/ripser/librosa OK).
Uses PROPER PCAReducer from preprocess.py and build_cloud_with_indices from
pointcloud.py — same cloud as the main pipeline.

v3: loads ALL tracks from loop_spotify.csv instead of a hardcoded list of 8.
This eliminates circular analysis / double-dipping that was present in v2.

Computes:
  1. Ripser cocycle (original method from run_loop_spotify)
  2. Dionysus cocycle (cohomology persistence)
  3. Dionysus cycle (homology persistence — geometrically correct)

All THREE on the SAME cloud, SAME diagram — differences are in the representative.

    .venv/bin/python scripts/run_cycle_analysis.py
"""
import os
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
from scipy.spatial.distance import pdist, cdist, cosine as cosine_dist
from scipy.stats import false_discovery_control
import ripser as ripser_lib
import dionysus as d
from joblib import Parallel, delayed

import preprocess
import pointcloud

# Config

# 90s center-crop — matches the physical cache in cache/muq_spotify90/.
# This is NOT a tunable hyperparameter; it's fixed by the embedding cache.
CLIP_SEC = 90
CACHE_DIR = "cache/muq_spotify90"


def load_cfg():
    return yaml.safe_load(open("config.yaml"))


def load_tracks_from_csv(csv_path):
    """Load ALL track basenames from loop_spotify.csv.

    Returns list of basenames (without .wav extension) for every row
    that has a valid H1 loop (max_persistence > 0).
    No filtering by significance — that would be circular analysis.
    """
    tracks = []
    with open(csv_path, "r") as fh:
        for row in csv.DictReader(fh):
            basename = row["basename"]
            # Strip .wav extension to get the track key
            track_base = basename.replace(".wav", "")
            # Only skip tracks with no H1 features at all
            pers = float(row.get("max_persistence", 0))
            if pers > 0:
                tracks.append(track_base)
    return tracks


# Representatives extraction

def extract_all_representatives(cloud, max_radius=None):
    """Compute all 3 representative types from same cloud.

    Returns dict with ripser_cocycle, dionysus_cocycle, dionysus_cycle info,
    plus verified diagrams.
    """
    n = cloud.shape[0]

    # ---- 1. Ripser cocycle ----
    r = ripser_lib.ripser(cloud, maxdim=1, do_cocycles=True)
    rdgm = r["dgms"][1]
    if len(rdgm) == 0:
        return None
    rlife = rdgm[:, 1] - rdgm[:, 0]
    rk = int(np.argmax(rlife))
    r_birth, r_death = float(rdgm[rk, 0]), float(rdgm[rk, 1])
    rcc = r["cocycles"][1][rk]
    ripser_verts = sorted(set(np.unique(rcc[:, :2].astype(int))))
    ripser_edges = [(int(e[0]), int(e[1])) for e in rcc[:, :2]]

    # ---- Dionysus filtration (used for both cocycle and cycle) ----
    # Radius must be >= death of most persistent H1
    if max_radius is None:
        max_radius = r_death * 1.1  # 10% margin above ripser death
    t0 = time.time()
    f = d.fill_rips(cloud, 2, max_radius)
    f.sort()
    filt_time = time.time() - t0

    # ---- 2. Dionysus cocycle ----
    cp = d.cohomology_persistence(f, prime=2, keep_cocycles=True)
    dgms_co = d.init_diagrams(cp, f)
    h1_co = [pt for pt in dgms_co[1] if pt.death < float("inf")]
    if not h1_co:
        return None
    best_co = max(h1_co, key=lambda p: p.death - p.birth)
    cc = cp.cocycle(best_co.data)
    dio_co_edges = []
    dio_co_verts = set()
    for entry in cc:
        s = f[entry.index]
        vs = list(s)
        if len(vs) == 2:
            dio_co_edges.append(vs)
            dio_co_verts.update(vs)

    # ---- 3. Dionysus cycle ----
    m = d.homology_persistence(f, prime=2, progress=False)
    dgms_ho = d.init_diagrams(m, f)
    h1_ho = [pt for pt in dgms_ho[1] if pt.death < float("inf")]
    if not h1_ho:
        return None
    best_ho = max(h1_ho, key=lambda p: p.death - p.birth)
    death_idx = best_ho.data
    birth_idx = m.pair(death_idx)
    chain = m[birth_idx]
    dio_cy_edges = []
    dio_cy_verts = set()
    for entry in chain:
        s = f[entry.index]
        vs = list(s)
        if len(vs) == 2:
            dio_cy_edges.append(vs)
            dio_cy_verts.update(vs)

    # ---- Verify diagrams ----
    match_co = (abs(r_birth - best_co.birth) < 1e-3 and
                abs(r_death - best_co.death) < 1e-3)
    match_ho = (abs(r_birth - best_ho.birth) < 1e-3 and
                abs(r_death - best_ho.death) < 1e-3)

    return {
        "n_points": n,
        "max_radius": max_radius,
        "filtration_size": len(f),
        "filt_time": filt_time,
        # Diagram
        "birth": r_birth,
        "death": r_death,
        "persistence": r_death - r_birth,
        "diagram_match_co": match_co,
        "diagram_match_ho": match_ho,
        # Ripser cocycle
        "ripser_co_verts": sorted(ripser_verts),
        "ripser_co_edges": ripser_edges,
        "n_ripser_co": len(ripser_verts),
        # Dionysus cocycle
        "dio_co_verts": sorted(dio_co_verts),
        "dio_co_edges": dio_co_edges,
        "n_dio_co": len(dio_co_verts),
        # Dionysus cycle
        "dio_cy_verts": sorted(dio_cy_verts),
        "dio_cy_edges": dio_cy_edges,
        "n_dio_cy": len(dio_cy_verts),
    }


# Vertex to seconds mapping

def vertices_to_seconds(vertex_ids, sub_indices, takens_starts,
                        window, target_fps):
    """Map cloud vertex IDs to time (center of Takens window)."""
    seconds = []
    for v in vertex_ids:
        takens_row = sub_indices[v]
        start_frame = takens_starts[takens_row]
        sec = (start_frame + window / 2) / target_fps
        seconds.append(sec)
    return np.sort(seconds)


# Chroma similarity test

def _mean_distant_sim(chroma_vecs, times, distant_thresh):
    """Mean cosine similarity among distant pairs (time gap >= threshold)."""
    sims = []
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if abs(times[i] - times[j]) >= distant_thresh:
                sims.append(1.0 - cosine_dist(chroma_vecs[i], chroma_vecs[j]))
    return float(np.mean(sims)) if sims else np.nan, len(sims)


def chroma_test(wav, sr, seconds, window_sec, hop_length=512,
                distant_thresh=5.0, n_perm=2000, seed=42):
    """Permutation test: loop vertices more chroma-similar than random sets?

    H0: vertices of the most-persistent H1 loop are not more chromatically
        similar than a random set of the same size with the same temporal
        spread constraint.

    Test statistic: mean cosine similarity among distant pairs (gap >=
    distant_thresh) of the loop vertices.

    Null distribution: the same statistic computed on R random vertex sets
    of the same size, drawn from all valid time-points, keeping only those
    with at least one distant pair.

    p = (#{null >= observed} + 1) / (R + 1).
    """
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

    # --- Observed statistic ---
    k = len(seconds)
    observed_stat, n_distant = _mean_distant_sim(
        np.array([get_chroma(s) for s in seconds]), seconds, distant_thresh)

    if np.isnan(observed_stat) or n_distant < 1:
        return {
            "n_distant_pairs": n_distant,
            "distant_sim_mean": observed_stat,
            "null_mean": np.nan,
            "p_value": np.nan,
        }

    # --- Pool of candidate time-points & their chroma vectors ---
    all_times = np.linspace(window_sec / 2, clip_dur - window_sec / 2, max(200, k + 10))
    all_chroma = np.array([get_chroma(s) for s in all_times])

    # --- Null distribution via permutation ---
    rng = np.random.default_rng(seed)
    null_stats = []
    for _ in range(n_perm):
        idx = rng.choice(len(all_times), size=k, replace=False)
        rand_times = all_times[idx]
        rand_chroma = all_chroma[idx]
        stat, cnt = _mean_distant_sim(rand_chroma, rand_times, distant_thresh)
        if not np.isnan(stat) and cnt >= 1:
            null_stats.append(stat)

    null_stats = np.array(null_stats)
    if len(null_stats) == 0:
        return {
            "n_distant_pairs": n_distant,
            "distant_sim_mean": observed_stat,
            "null_mean": np.nan,
            "p_value": np.nan,
        }

    p_value = float((np.sum(null_stats >= observed_stat) + 1) / (len(null_stats) + 1))

    return {
        "n_distant_pairs": n_distant,
        "distant_sim_mean": observed_stat,
        "null_mean": float(null_stats.mean()),
        "p_value": p_value,
    }


# Visualization

def plot_3methods(cloud, rep, sub_indices, takens_starts, target_fps,
                  window, track_name, out_path):
    """3-panel: ripser cocycle, dio cocycle, dio cycle on same 2D PCA."""
    cloud_c = cloud - cloud.mean(axis=0)
    _, _, Vt = np.linalg.svd(cloud_c, full_matrices=False)
    proj = cloud_c @ Vt[:2].T

    fig, axes = plt.subplots(1, 4, figsize=(26, 6))

    panels = [
        (axes[0], rep["ripser_co_edges"], rep["ripser_co_verts"],
         f"Ripser cocycle ({rep['n_ripser_co']} verts)", "green"),
        (axes[1], rep["dio_co_edges"], rep["dio_co_verts"],
         f"Dionysus cocycle ({rep['n_dio_co']} verts)", "red"),
        (axes[2], rep["dio_cy_edges"], rep["dio_cy_verts"],
         f"Dionysus cycle ({rep['n_dio_cy']} verts)", "blue"),
    ]
    for ax, edges, verts, title, color in panels:
        ax.scatter(proj[:, 0], proj[:, 1], c="gray", s=8, alpha=0.3, zorder=1)
        for e in edges:
            ax.plot(proj[e, 0], proj[e, 1], "-", c=color, lw=0.6, alpha=0.3, zorder=2)
        if verts:
            ax.scatter(proj[verts, 0], proj[verts, 1], c=color, s=15, zorder=3)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")

    # Panel 4: Timeline
    ax = axes[3]
    for verts, color, yoff, label in [
        (rep["ripser_co_verts"], "green", 0.6, "Ripser co"),
        (rep["dio_co_verts"], "red", 0.0, "Dio co"),
        (rep["dio_cy_verts"], "blue", -0.6, "Dio cycle"),
    ]:
        secs = vertices_to_seconds(verts, sub_indices, takens_starts,
                                    window, target_fps)
        ax.eventplot([secs], lineoffsets=yoff, linelengths=0.4,
                      colors=[color], label=label)
    ax.set_xlabel("Time (seconds)")
    ax.set_yticks([0.6, 0.0, -0.6])
    ax.set_yticklabels(["Ripser co", "Dio co", "Dio cycle"])
    ax.set_xlim(0, 90)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Timeline")

    fig.suptitle(f"{track_name}\npers={rep['persistence']:.4f}, "
                  f"diag: {'✓' if rep['diagram_match_co'] and rep['diagram_match_ho'] else '✗'}",
                  fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Main


def process_track(idx, n_total, track_base, cfg, sr, target_fps, hop_length, pc_cfg, window, window_sec, reducer, existing):
    print(f"\n[{idx+1}/{n_total}] {'='*60}")
    print(f"Track: {track_base}")

    cache_path = os.path.join(CACHE_DIR, f"{track_base}.npy")
    if not os.path.exists(cache_path):
        print(f"  SKIP: cache not found")
        
        return None
    emb = np.load(cache_path)
    print(f"  Embedding: {emb.shape}")

    # ---- PROPER preprocessing: resample → PCAReducer.transform ----
    x_pca = preprocess.prepare_track(emb, cfg["spaces"]["muq"], cfg["common"], reducer)
    print(f"  After PCA: {x_pca.shape}")

    # ---- Build cloud (same as main pipeline) ----
    cloud, takens_starts, sub_indices = pointcloud.build_cloud_with_indices(
        x_pca, pc_cfg)
    print(f"  Cloud: {cloud.shape}")

    # ---- All 3 representatives from SAME cloud ----
    print(f"  Computing 3 representatives...")
    t0 = time.time()
    rep = extract_all_representatives(cloud)
    elapsed = time.time() - t0
    if rep is None:
        print(f"  SKIP: no H1 features")
        return None

    print(f"  Diagram: birth={rep['birth']:.4f}, death={rep['death']:.4f}, "
          f"pers={rep['persistence']:.4f}")
    print(f"  Diagram match: co={rep['diagram_match_co']}, "
          f"ho={rep['diagram_match_ho']}")
    print(f"  Ripser cocycle:  {rep['n_ripser_co']} verts")
    print(f"  Dionysus cocycle: {rep['n_dio_co']} verts")
    print(f"  Dionysus cycle:   {rep['n_dio_cy']} verts")
    print(f"  Filtration: {rep['filtration_size']} simplices, "
          f"{elapsed:.1f}s total")

    # ---- Map to seconds ----
    secs_ripser = vertices_to_seconds(
        rep["ripser_co_verts"], sub_indices, takens_starts, window, target_fps)
    secs_dio_co = vertices_to_seconds(
        rep["dio_co_verts"], sub_indices, takens_starts, window, target_fps)
    secs_dio_cy = vertices_to_seconds(
        rep["dio_cy_verts"], sub_indices, takens_starts, window, target_fps)

    span_ripser = float(secs_ripser.max() - secs_ripser.min()) if len(secs_ripser) > 1 else 0
    span_dio_co = float(secs_dio_co.max() - secs_dio_co.min()) if len(secs_dio_co) > 1 else 0
    span_dio_cy = float(secs_dio_cy.max() - secs_dio_cy.min()) if len(secs_dio_cy) > 1 else 0

    print(f"\n  Spans:  ripser_co={span_ripser:.1f}s  "
          f"dio_co={span_dio_co:.1f}s  dio_cycle={span_dio_cy:.1f}s")

    # ---- Load audio (discover genre from track name) ----
    wav_path = None
    # Extract genre from track basename (format: genre_NN_Title_Artist)
    track_genre = track_base.split("_")[0]
    candidate = os.path.join("data/top50musicSpotify", track_genre,
                              track_base + ".wav")
    if os.path.exists(candidate):
        wav_path = candidate
    else:
        # Fallback: scan all genre dirs
        for genre_dir in sorted(os.listdir("data/top50musicSpotify")):
            candidate = os.path.join("data/top50musicSpotify", genre_dir,
                                      track_base + ".wav")
            if os.path.exists(candidate):
                wav_path = candidate
                break
    if wav_path is None:
        print(f"  SKIP chroma: wav not found")
        
        return None

    wav_full, _ = librosa.load(wav_path, sr=sr, mono=True)
    total_sec = len(wav_full) / sr
    if total_sec > CLIP_SEC:
        start = int((total_sec - CLIP_SEC) / 2 * sr)
        wav = wav_full[start:start + int(CLIP_SEC * sr)]
    else:
        wav = wav_full

    # ---- Chroma tests (permutation test, raw p) ----
    n_perm = cfg["distances"]["chroma_n_perm"]
    ct_ripser = chroma_test(wav, sr, secs_ripser, window_sec, hop_length,
                            n_perm=n_perm)
    ct_dio_co = chroma_test(wav, sr, secs_dio_co, window_sec, hop_length,
                            n_perm=n_perm)
    ct_dio_cy = chroma_test(wav, sr, secs_dio_cy, window_sec, hop_length,
                            n_perm=n_perm)

    print(f"\n  Chroma p-values (raw permutation):")
    print(f"    Ripser cocycle:  p={ct_ripser['p_value']:.6f}")
    print(f"    Dionysus cocycle: p={ct_dio_co['p_value']:.6f}")
    print(f"    Dionysus cycle:   p={ct_dio_cy['p_value']:.6f}")

    # Compare with previous ripser results from CSV
    ex = existing.get(track_base + ".wav", {})
    prev_p_str = ex.get("p_value", "nan")
    prev_p = float(prev_p_str) if prev_p_str else np.nan
    prev_verts_str = ex.get("n_loop_vertices", 0)
    prev_verts = int(prev_verts_str) if prev_verts_str else 0
    prev_span_str = ex.get("loop_span_sec", 0)
    prev_span = float(prev_span_str) if prev_span_str else 0.0

    r_row = {
        "track": track_base,
        "genre": track_base.split("_")[0],
        "persistence": rep["persistence"],
        "diagram_match": rep["diagram_match_co"] and rep["diagram_match_ho"],
        # Ripser cocycle
        "n_ripser_co": rep["n_ripser_co"],
        "span_ripser_co": span_ripser,
        "p_ripser_co": ct_ripser["p_value"],
        # Dionysus cocycle
        "n_dio_co": rep["n_dio_co"],
        "span_dio_co": span_dio_co,
        "p_dio_co": ct_dio_co["p_value"],
        # Dionysus cycle
        "n_dio_cy": rep["n_dio_cy"],
        "span_dio_cy": span_dio_cy,
        "p_dio_cy": ct_dio_cy["p_value"],
        # FDR-adjusted p-values and significance — filled after loop
        "p_fdr_ripser_co": np.nan,
        "p_fdr_dio_co": np.nan,
        "p_fdr_dio_cy": np.nan,
        "sig_ripser_co": False,
        "sig_dio_co": False,
        "sig_dio_cy": False,
        # Previous (from loop_spotify.csv)
        "prev_n_verts": prev_verts,
        "prev_span": prev_span,
        "prev_p": prev_p,
    }
    return r_row



def main():
    cfg = load_cfg()
    sr = cfg["data"]["sample_rate"]
    target_fps = cfg["common"]["target_fps"]
    hop_length = cfg["spaces"]["mir"]["hop_length"]
    pc_cfg = cfg["pointcloud"].copy()
    pc_cfg["takens_pca_dim"] = None  # no extra PCA on takens
    window = pc_cfg["window"]
    window_sec = window / target_fps

    # ---- Load pre-fitted PCA reducer (deterministic, saved once) ----
    REDUCER_PATH = "cache/pca_reducer_muq_spotify90.joblib"
    if not os.path.exists(REDUCER_PATH):
        print(f"ERROR: {REDUCER_PATH} not found. Run: .venv/bin/python scripts/archive/fit_reducer.py")
        sys.exit(1)
    print(f"Loading saved PCA reducer from {REDUCER_PATH}...")
    reducer = preprocess.PCAReducer.load(REDUCER_PATH)
    print(f"  PCA dim={reducer.dim}, explained={reducer.explained:.4f}")

    # ---- Load ALL tracks from loop_spotify.csv (no cherry-picking) ----
    csv_path = os.path.join(cfg["paths"]["results"], "tables", "loop_spotify.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run: .venv/bin/python scripts/run_loop_spotify.py")
        sys.exit(1)
    all_tracks = load_tracks_from_csv(csv_path)
    print(f"\nLoaded {len(all_tracks)} tracks from {csv_path} (no filtering by significance)")

    # Load existing ripser results for comparison
    existing = {}
    with open(csv_path, "r") as fh:
        for row in csv.DictReader(fh):
            existing[row["basename"]] = row

    # Output dirs
    out_figs = os.path.join(cfg["paths"]["results"], "figures")
    out_tables = os.path.join(cfg["paths"]["results"], "tables")
    os.makedirs(out_figs, exist_ok=True)
    os.makedirs(out_tables, exist_ok=True)

    csv_out = os.path.join(out_tables, "cycle_vs_cocycle_full.csv")

    # Load existing progress if any
    done_tracks = set()
    results = []
    if os.path.exists(csv_out):
        print(f"\nLoading existing progress from {csv_out}...")
        with open(csv_out, "r") as fh:
            for row in csv.DictReader(fh):
                # Convert numeric fields back
                for k, v in row.items():
                    if k not in ["track", "genre"]:
                        try:
                            if v == "True": row[k] = True
                            elif v == "False": row[k] = False
                            elif v == "" or v == "nan": row[k] = np.nan
                            else: row[k] = float(v)
                        except ValueError:
                            pass
                results.append(row)
                done_tracks.add(row["track"])
        print(f"  Found {len(done_tracks)} already processed tracks.")

    tracks_to_run = [t for t in all_tracks if t not in done_tracks]
    print(f"Tracks remaining to run: {len(tracks_to_run)}")

    if len(tracks_to_run) > 0:
        print(f"\n--- Running {len(tracks_to_run)} tracks dynamically (saving as they complete) ---")
        try:
            # return_as="generator" yields results immediately as each worker finishes a track!
            results_generator = Parallel(n_jobs=-1, verbose=10, return_as="generator")(
                delayed(process_track)(
                    all_tracks.index(track_base), len(all_tracks), track_base, cfg, sr, target_fps, 
                    hop_length, pc_cfg, window, window_sec, reducer, existing
                ) for track_base in tracks_to_run
            )
            
            for r in results_generator:
                if r is not None:
                    results.append(r)
                    
                    # Save immediately after EVERY finished track
                    fieldnames = list(results[0].keys())
                    tmp_csv = csv_out + ".tmp"
                    with open(tmp_csv, "w", newline="") as fh:
                        writer = csv.DictWriter(fh, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(results)
                    os.replace(tmp_csv, csv_out)

        except Exception as e:
            print(f"\n[CRASH] A worker failed (likely OOM or Timeout): {e}")
            print("Progress up to the LAST fully completed track is safe in the CSV.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Execution stopped by user.")
            print("Progress up to the LAST fully completed track is safe in the CSV.")
            sys.exit(1)

    n_skipped = len(all_tracks) - len(results)
    # ==================================================================
    # FDR correction (Benjamini-Hochberg) across ALL tracks × 3 methods
    # ==================================================================
    print(f"\n\nProcessed: {len(results)} tracks, skipped: {n_skipped}")

    # Collect all raw p-values with their (row_idx, method) index
    p_col_names = ["p_ripser_co", "p_dio_co", "p_dio_cy"]
    fdr_col_names = ["p_fdr_ripser_co", "p_fdr_dio_co", "p_fdr_dio_cy"]
    sig_col_names = ["sig_ripser_co", "sig_dio_co", "sig_dio_cy"]

    raw_p_list = []  # (row_idx, method_idx, p_raw)
    for ri, r in enumerate(results):
        for mi, pc in enumerate(p_col_names):
            p = r[pc]
            if not np.isnan(p):
                raw_p_list.append((ri, mi, p))

    if raw_p_list:
        raw_p_arr = np.array([x[2] for x in raw_p_list])
        # Benjamini-Hochberg FDR correction
        fdr_p_arr = false_discovery_control(raw_p_arr, method='bh')
        for (ri, mi, _), fdr_p in zip(raw_p_list, fdr_p_arr):
            results[ri][fdr_col_names[mi]] = float(fdr_p)
            results[ri][sig_col_names[mi]] = bool(fdr_p < 0.05)

    # ---- Summary table ----
    print("\n" + "=" * 140)
    print("3-METHOD COMPARISON TABLE (FULL POPULATION — NO CHERRY-PICKING)")
    print("=" * 140)
    hdr = (f"{'Track':40s} | {'n_rip':>5} {'n_dco':>5} {'n_dcy':>5} | "
           f"{'sp_rip':>6} {'sp_dco':>6} {'sp_dcy':>6} | "
           f"{'p_rip':>8} {'p_dco':>8} {'p_dcy':>8} | "
           f"{'fdr_rip':>8} {'fdr_dco':>8} {'fdr_dcy':>8}")
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        t = r["track"][:40]
        print(f"{t:40s} | {r['n_ripser_co']:5d} {r['n_dio_co']:5d} {r['n_dio_cy']:5d} | "
              f"{r['span_ripser_co']:5.1f}s {r['span_dio_co']:5.1f}s {r['span_dio_cy']:5.1f}s | "
              f"{r['p_ripser_co']:8.4f} {r['p_dio_co']:8.4f} {r['p_dio_cy']:8.4f} | "
              f"{r['p_fdr_ripser_co']:8.4f} {r['p_fdr_dio_co']:8.4f} {r['p_fdr_dio_cy']:8.4f}")

    # Save CSV — NEW file, do NOT overwrite old cycle_vs_cocycle.csv
    csv_out = os.path.join(out_tables, "cycle_vs_cocycle_full.csv")
    fieldnames = list(results[0].keys()) if results else []
    with open(csv_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n→ {csv_out}")

    # ---- P-value distribution summary ----
    print("\n" + "=" * 70)
    print("P-VALUE DISTRIBUTION: RAW vs FDR-CORRECTED")
    print("=" * 70)
    print(f"  Total tracks processed: {len(results)}")
    print(f"  Total tests (tracks × 3 methods): {len(raw_p_list)}")
    print(f"  FDR method: Benjamini-Hochberg")

    for method, p_col, fdr_col, sig_col in [
        ("Ripser cocycle", "p_ripser_co", "p_fdr_ripser_co", "sig_ripser_co"),
        ("Dionysus cocycle", "p_dio_co", "p_fdr_dio_co", "sig_dio_co"),
        ("Dionysus cycle", "p_dio_cy", "p_fdr_dio_cy", "sig_dio_cy"),
    ]:
        p_vals = [r[p_col] for r in results if not np.isnan(r[p_col])]
        n_raw_sig = sum(1 for r in results
                       if not np.isnan(r[p_col]) and r[p_col] < 0.05)
        n_fdr_sig = sum(1 for r in results if r[sig_col])
        n_valid = len(p_vals)
        n_nan = len(results) - n_valid
        print(f"\n  {method}:")
        print(f"    Valid p-values: {n_valid}, NaN (too few pairs): {n_nan}")
        if n_valid > 0:
            print(f"    Raw p < 0.05:   {n_raw_sig}/{n_valid} "
                  f"({100*n_raw_sig/n_valid:.1f}%)")
            print(f"    FDR p < 0.05:   {n_fdr_sig}/{n_valid} "
                  f"({100*n_fdr_sig/n_valid:.1f}%)")
        else:
            print("    N/A")
        if p_vals:
            p_arr = np.array(p_vals)
            bins = [0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
            counts, _ = np.histogram(p_arr, bins=bins)
            print(f"    Raw p distribution:")
            for i in range(len(bins) - 1):
                print(f"      [{bins[i]:.3f}, {bins[i+1]:.3f}): {counts[i]}")

    # ---- Verdict ----
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Check diagram consistency
    all_match = all(r["diagram_match"] for r in results)
    n_match = sum(1 for r in results if r["diagram_match"])
    print(f"  Diagrams match: {n_match}/{len(results)} "
          f"{'(ALL ✓)' if all_match else '(SOME MISMATCH ✗)'}")

    # Check p-value agreement between methods (after FDR)
    n_agree = sum(1 for r in results
                   if (r["sig_ripser_co"] == r["sig_dio_cy"]))
    print(f"  Ripser_co vs Cycle agree on significance (FDR): "
          f"{n_agree}/{len(results)}")

    # Honest summary: how many survive FDR?
    total_fdr_sig = sum(1 for r in results
                        if r["sig_ripser_co"] or r["sig_dio_co"]
                        or r["sig_dio_cy"])
    print(f"  Tracks significant by ANY method (FDR<0.05): "
          f"{total_fdr_sig}/{len(results)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
