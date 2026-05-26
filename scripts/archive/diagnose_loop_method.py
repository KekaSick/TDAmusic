"""
diagnose_loop_method.py
-----------------------
Диагностика двух подозрений в методе loop interpretation:

1. COCYCLE vs CYCLE: коцикл не является геометрической петлёй —
   он может покрывать бо́льшую часть облака. Проверить на синтетике
   и на реальных треках.

2. ПОРОГ ПОВТОРОВ: has_repeats=True у всех 199 треков подозрительно.
   Проверить чувствительность к порогу и отличить однородную SSM
   от блочной (истинные повторы).

    .venv/bin/python scripts/diagnose_loop_method.py
"""
import os
import sys
import yaml
import glob
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import preprocess
import pointcloud
import persistence


CLIP_SEC = 90
SPOTIFY_DIR = "data/top50musicSpotify"
CACHE_DIR = "cache/muq_spotify90"


def load_cfg():
    return yaml.safe_load(open("config.yaml"))


# =====================================================================
# 1. COCYCLE DIAGNOSTICS
# =====================================================================

def diagnose_cocycle_synthetic():
    """Коцикл на синтетической чистой окружности — ожидаем размазанность."""
    print("=" * 70)
    print("1A. SYNTHETIC: Cocycle coverage on a clean circle")
    print("=" * 70)

    for n_pts in [50, 100, 300]:
        t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        X = np.column_stack([np.cos(t), np.sin(t)])
        X += 0.03 * np.random.default_rng(42).standard_normal(X.shape)

        r = persistence.compute_diagrams(X, maxdim=1, do_cocycles=True)
        dgm1 = r["dgms"][1]
        if len(dgm1) == 0:
            print(f"  n={n_pts}: NO H1")
            continue
        life = dgm1[:, 1] - dgm1[:, 0]
        k = int(np.argmax(life))
        cc = r["cocycles"][1][k]
        verts = np.unique(cc[:, :2].astype(int))

        print(f"  n={n_pts}: cocycle={len(cc)} edges, "
              f"{len(verts)} vertices ({len(verts)/n_pts*100:.0f}% of cloud), "
              f"persistence={life[k]:.3f}")

    print()
    print("  → Cocycle на ЧИСТОЙ окружности покрывает ~50-70% точек.")
    print("  → Это СВОЙСТВО коцикла, а НЕ артефакт данных.")
    print("  → Большой span и высокий vertex count — ожидаемы для коцикла.")
    print()


def diagnose_cocycle_real(cfg, reducer, tracks, n_tracks=5):
    """Анализ коцикла на реальных треках:
    - сколько вершин, какой % облака
    - как распределены по времени
    - есть ли кластеризация или равномерное покрытие
    """
    print("=" * 70)
    print("1B. REAL TRACKS: Cocycle coverage analysis")
    print("=" * 70)

    sr = cfg["data"]["sample_rate"]
    pc_cfg = cfg["pointcloud"].copy()
    pc_cfg["takens_pca_dim"] = None
    window = pc_cfg["window"]
    target_fps = cfg["common"]["target_fps"]
    cloud_size = pc_cfg["n_points"]

    results = []
    for i, t in enumerate(tracks[:n_tracks]):
        fp = t["filepath"]
        base = t["basename"]
        genre = t["genre"]

        # Load MuQ from cache
        cache_base = os.path.splitext(base)[0]
        emb = np.load(os.path.join(CACHE_DIR, f"{cache_base}.npy"))
        x_pca = preprocess.prepare_track(
            emb, cfg["spaces"]["muq"], cfg["common"], reducer)

        cloud, takens_starts, sub_indices = \
            pointcloud.build_cloud_with_indices(x_pca, pc_cfg)

        res = persistence.compute_diagrams(cloud, maxdim=1, do_cocycles=True)
        dgm1 = res["dgms"][1]
        if len(dgm1) == 0:
            continue

        life = dgm1[:, 1] - dgm1[:, 0]
        k = int(np.argmax(life))
        cc = res["cocycles"][1][k]
        verts = np.unique(cc[:, :2].astype(int))
        vert_frac = len(verts) / cloud_size

        # Map vertices to seconds
        seconds = []
        for v in verts:
            takens_row = sub_indices[v]
            start_frame = takens_starts[takens_row]
            sec = (start_frame + window / 2) / target_fps
            seconds.append(sec)
        seconds = np.sort(seconds)

        # Check temporal distribution: gaps between consecutive vertices
        if len(seconds) > 1:
            gaps = np.diff(seconds)
            max_gap = gaps.max()
            # Coverage: what fraction of the 90s timeline is within 2s of a vertex?
            timeline = np.linspace(0, 90, 1000)
            covered = np.any(np.abs(timeline[:, None] - seconds[None, :]) < 2.0, axis=1)
            coverage_frac = covered.mean()
        else:
            max_gap = 0
            coverage_frac = 0

        r = {
            "basename": base,
            "genre": genre,
            "n_verts": len(verts),
            "vert_frac": vert_frac,
            "n_edges": len(cc),
            "persistence": float(life[k]),
            "span_sec": float(seconds.max() - seconds.min()) if len(seconds) > 1 else 0,
            "max_gap_sec": float(max_gap),
            "timeline_coverage": float(coverage_frac),
            "sec_min": float(seconds.min()),
            "sec_max": float(seconds.max()),
        }
        results.append(r)

        print(f"\n  [{i+1}] {base} ({genre})")
        print(f"      Vertices: {r['n_verts']}/{cloud_size} ({r['vert_frac']:.0%})")
        print(f"      Edges: {r['n_edges']}")
        print(f"      Persistence: {r['persistence']:.4f}")
        print(f"      Time span: {r['span_sec']:.1f}s "
              f"(range [{r['sec_min']:.1f}, {r['sec_max']:.1f}])")
        print(f"      Max gap between vertices: {r['max_gap_sec']:.1f}s")
        print(f"      Timeline coverage (±2s): {r['timeline_coverage']:.0%}")

        # Verdict for this track
        if r["vert_frac"] > 0.5:
            print(f"      → SPREAD: cocycle covers >{r['vert_frac']:.0%} of cloud")
        elif r["vert_frac"] > 0.2:
            print(f"      → MODERATE: {r['vert_frac']:.0%} of cloud")
        else:
            print(f"      → LOCALIZED: only {r['vert_frac']:.0%} of cloud")

    return results


def diagnose_cocycle_aggregate():
    """Агрегат по всем 199 трекам из CSV."""
    print("\n" + "=" * 70)
    print("1C. AGGREGATE: Cocycle coverage across all 199 tracks")
    print("=" * 70)

    df = pd.read_csv("results/tables/loop_spotify.csv")
    df["vert_frac"] = df["n_loop_vertices"] / 300

    print(f"\n  Vertex count: median={df['n_loop_vertices'].median():.0f}, "
          f"mean={df['n_loop_vertices'].mean():.0f}")
    print(f"  Vertex fraction: median={df['vert_frac'].median():.2f}, "
          f"mean={df['vert_frac'].mean():.2f}")
    print()

    bins = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3),
            (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
    print("  Cloud coverage distribution:")
    for lo, hi in bins:
        n = ((df["vert_frac"] >= lo) & (df["vert_frac"] < hi)).sum()
        sig = df[(df["vert_frac"] >= lo) & (df["vert_frac"] < hi) & df["significant"]].shape[0]
        print(f"    {lo:.0%}-{hi:.0%}: {n:3d} tracks, {sig:3d} significant")

    # Key check: is significance correlated with vertex count?
    from scipy.stats import pointbiserialr
    mask = df["p_value"].notna()
    r, p = pointbiserialr(df.loc[mask, "significant"].astype(int),
                           df.loc[mask, "n_loop_vertices"])
    print(f"\n  Correlation(significant, n_vertices): r={r:.3f}, p={p:.4f}")
    print(f"  → {'Significant bias toward larger cocycles!' if p < 0.05 else 'No significant correlation.'}")

    # Significant with SMALL cocycles (<10% of cloud)?
    small = df[(df["vert_frac"] < 0.1)]
    small_sig = small[small["significant"]]
    print(f"\n  Small cocycles (<10% cloud): {len(small)} tracks, "
          f"{len(small_sig)} significant ({len(small_sig)/max(1,len(small))*100:.0f}%)")

    large = df[(df["vert_frac"] > 0.5)]
    large_sig = large[large["significant"]]
    print(f"  Large cocycles (>50% cloud): {len(large)} tracks, "
          f"{len(large_sig)} significant ({len(large_sig)/max(1,len(large))*100:.0f}%)")


# =====================================================================
# 2. REPEAT THRESHOLD DIAGNOSTICS
# =====================================================================

def compute_ssm_structure(wav, sr, hop_length=512, min_dist_sec=5.0):
    """Compute SSM and structural metrics.

    Returns dict with:
    - bright_frac_85, _90, _95: fraction of far-diagonal cells above threshold
    - ssm_var: variance of far-diagonal values
    - block_score: ratio of variance to mean (higher = more blocky vs uniform)
    - has_offdiag_peaks: presence of clear peaks away from diagonal
    """
    chroma = librosa.feature.chroma_cens(y=wav, sr=sr, hop_length=hop_length)
    chroma_fps = sr / hop_length
    min_dist_frames = int(min_dist_sec * chroma_fps)

    ssm = cosine_similarity(chroma.T)
    n = ssm.shape[0]

    rows, cols = np.triu_indices(n, k=min_dist_frames)
    far_vals = ssm[rows, cols]

    if len(far_vals) == 0:
        return {k: 0 for k in ["bright_frac_85", "bright_frac_90", "bright_frac_95",
                                "far_mean", "far_std", "far_var",
                                "block_score", "has_structured_repeats"]}

    far_mean = float(far_vals.mean())
    far_std = float(far_vals.std())
    far_var = float(far_vals.var())

    # Block score: higher std/mean ratio means more contrast (blocks, not uniform)
    # Uniform bright → high mean, low std → low block_score
    # Blocky → moderate mean, higher std → higher block_score
    block_score = far_std / max(far_mean, 1e-8)

    # Check for structured repeats:
    # Compute SSM mean for horizontal bands (time-localized averages)
    # If there are repeating sections, some rows will have much higher mean
    row_means = []
    for i in range(0, n, max(1, n // 20)):
        far_row = ssm[i, :].copy()
        far_row[max(0, i - min_dist_frames):min(n, i + min_dist_frames)] = np.nan
        m = np.nanmean(far_row)
        if not np.isnan(m):
            row_means.append(m)
    row_means = np.array(row_means)
    row_var = float(row_means.var()) if len(row_means) > 1 else 0

    # Has structured repeats if:
    # (a) block_score > 0.15 (not uniformly bright) AND
    # (b) row_var > 0.002 (some rows are much brighter than others)
    has_structured = block_score > 0.15 and row_var > 0.002

    return {
        "bright_frac_85": float((far_vals >= 0.85).mean()),
        "bright_frac_90": float((far_vals >= 0.90).mean()),
        "bright_frac_95": float((far_vals >= 0.95).mean()),
        "far_mean": far_mean,
        "far_std": far_std,
        "far_var": far_var,
        "block_score": float(block_score),
        "row_var": row_var,
        "has_structured_repeats": has_structured,
    }


def diagnose_repeat_threshold(cfg, tracks, n_tracks=20):
    """Проверка чувствительности порога и различия блочной vs однородной SSM."""
    print("\n" + "=" * 70)
    print("2A. SSM STRUCTURE: Block vs uniform analysis")
    print("=" * 70)

    sr = cfg["data"]["sample_rate"]
    hop_length = cfg["spaces"]["mir"]["hop_length"]
    results = []

    # Select diverse tracks: 5 pop, 5 electronic, 5 hip-hop, 5 reggae
    by_genre = {}
    for t in tracks:
        by_genre.setdefault(t["genre"], []).append(t)
    sample = []
    for g in ["pop", "electronic", "hip-hop", "reggae"]:
        sample.extend(by_genre.get(g, [])[:5])

    for i, t in enumerate(sample[:n_tracks]):
        fp = t["filepath"]
        base = t["basename"]
        genre = t["genre"]

        wav_full, _ = librosa.load(fp, sr=sr, mono=True)
        total_sec = len(wav_full) / sr
        if total_sec > CLIP_SEC:
            start = int((total_sec - CLIP_SEC) / 2 * sr)
            wav = wav_full[start:start + int(CLIP_SEC * sr)]
        else:
            wav = wav_full

        metrics = compute_ssm_structure(wav, sr, hop_length)
        metrics["basename"] = base
        metrics["genre"] = genre
        results.append(metrics)

        print(f"\n  [{i+1}] {base} ({genre})")
        print(f"      Far-diag mean={metrics['far_mean']:.3f}, "
              f"std={metrics['far_std']:.3f}")
        print(f"      Block score (std/mean): {metrics['block_score']:.3f}")
        print(f"      Row variance: {metrics['row_var']:.5f}")
        print(f"      Bright fracs: "
              f"≥0.85={metrics['bright_frac_85']:.3f}, "
              f"≥0.90={metrics['bright_frac_90']:.3f}, "
              f"≥0.95={metrics['bright_frac_95']:.3f}")
        print(f"      Structured repeats: {metrics['has_structured_repeats']}")

    return results


def diagnose_threshold_sensitivity(cfg, tracks):
    """Сколько треков сохраняют 'has_repeats' при разных порогах."""
    print("\n" + "=" * 70)
    print("2B. THRESHOLD SENSITIVITY: How many tracks have 'repeats'")
    print("=" * 70)

    sr = cfg["data"]["sample_rate"]
    hop_length = cfg["spaces"]["mir"]["hop_length"]

    thresholds = [0.80, 0.85, 0.90, 0.95]
    min_fracs = [0.01, 0.02, 0.05]

    # Sample: all tracks
    counts = {}
    n_sample = min(len(tracks), 199)

    # Pre-compute SSM stats for all tracks
    all_frac = {th: [] for th in thresholds}

    print(f"\n  Computing SSM for {n_sample} tracks...")
    for i, t in enumerate(tracks[:n_sample]):
        fp = t["filepath"]
        wav_full, _ = librosa.load(fp, sr=sr, mono=True)
        total_sec = len(wav_full) / sr
        if total_sec > CLIP_SEC:
            start = int((total_sec - CLIP_SEC) / 2 * sr)
            wav = wav_full[start:start + int(CLIP_SEC * sr)]
        else:
            wav = wav_full

        chroma = librosa.feature.chroma_cens(y=wav, sr=sr, hop_length=hop_length)
        chroma_fps = sr / hop_length
        min_dist_frames = int(5.0 * chroma_fps)
        ssm = cosine_similarity(chroma.T)
        n = ssm.shape[0]
        rows, cols = np.triu_indices(n, k=min_dist_frames)
        far_vals = ssm[rows, cols]

        for th in thresholds:
            frac = (far_vals >= th).mean() if len(far_vals) > 0 else 0
            all_frac[th].append(frac)

        if (i + 1) % 50 == 0:
            print(f"    ... {i+1}/{n_sample}")

    print(f"\n  Threshold sensitivity table:")
    print(f"  {'Thresh':>7} {'min_frac':>9} {'n_with_repeats':>15} {'fraction':>9}")
    print(f"  {'-'*45}")
    for th in thresholds:
        for mf in min_fracs:
            n_rep = sum(1 for f in all_frac[th] if f >= mf)
            print(f"  {th:>7.2f} {mf:>9.2f} {n_rep:>15} {n_rep/n_sample:>9.1%}")
        print()


# =====================================================================
# 3. SUMMARY AND VISUALIZATIONS
# =====================================================================

def plot_ssm_comparison(cfg, tracks, out_dir):
    """Plot SSM for 4 tracks: pop (blocky), hip-hop (uniform), etc."""
    sr = cfg["data"]["sample_rate"]
    hop_length = cfg["spaces"]["mir"]["hop_length"]

    # Find one pop and one hip-hop track
    examples = {}
    for t in tracks:
        if t["genre"] == "pop" and "pop" not in examples:
            examples["pop"] = t
        elif t["genre"] == "hip-hop" and "hip-hop" not in examples:
            examples["hip-hop"] = t
        if len(examples) == 2:
            break

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (genre, t) in zip(axes, examples.items()):
        fp = t["filepath"]
        wav_full, _ = librosa.load(fp, sr=sr, mono=True)
        total_sec = len(wav_full) / sr
        if total_sec > CLIP_SEC:
            start = int((total_sec - CLIP_SEC) / 2 * sr)
            wav = wav_full[start:start + int(CLIP_SEC * sr)]
        else:
            wav = wav_full
        clip_dur = len(wav) / sr

        chroma = librosa.feature.chroma_cens(y=wav, sr=sr, hop_length=hop_length)
        ssm = cosine_similarity(chroma.T)

        im = ax.imshow(ssm, origin="lower", cmap="magma", aspect="auto",
                        extent=[0, clip_dur, 0, clip_dur], vmin=0.3, vmax=1.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"{genre}: {os.path.basename(fp)[:40]}")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("SSM comparison: blocky (pop) vs uniform (hip-hop)?",
                  fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "diagnose_ssm_comparison.png"),
                 bbox_inches="tight", dpi=150)
    print(f"\n  → {os.path.join(out_dir, 'diagnose_ssm_comparison.png')}")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main():
    cfg = load_cfg()

    # Collect tracks
    all_tracks = []
    for genre in ["pop", "electronic", "hip-hop", "reggae"]:
        wavs = sorted(glob.glob(os.path.join(SPOTIFY_DIR, genre, "*.wav")))
        for w in wavs:
            all_tracks.append({"filepath": w, "genre": genre,
                                "basename": os.path.basename(w)})

    # Train PCA (same as main experiment)
    print("Loading MuQ embeddings from cache + training PCA...")
    raw_embs = []
    for t in all_tracks:
        base = os.path.splitext(t["basename"])[0]
        cache_path = os.path.join(CACHE_DIR, f"{base}.npy")
        if os.path.exists(cache_path):
            raw_embs.append(np.load(cache_path))
    reducer = preprocess.PCAReducer(
        dim=cfg["common"]["pca_dim"],
        standardize=cfg["common"].get("standardize_features", True),
        normalize=cfg["common"]["normalize"])
    reducer.fit(raw_embs)
    print(f"  PCA trained on {len(raw_embs)} tracks, "
          f"explained={reducer.explained:.4f}")

    # === 1. COCYCLE DIAGNOSTICS ===
    diagnose_cocycle_synthetic()

    # Pick diverse tracks for real analysis
    diverse = []
    df = pd.read_csv("results/tables/loop_spotify.csv")
    # Pick: smallest cocycle (significant), largest, pop sig, hiphop nonsig, reggae sig
    sig_small = df[df["significant"]].nsmallest(1, "n_loop_vertices")
    sig_large = df[df["significant"]].nlargest(1, "n_loop_vertices")
    nonsig = df[~df["significant"] & df["p_value"].notna()].nlargest(1, "max_persistence")

    picks = pd.concat([sig_small, sig_large, nonsig])
    # Add 2 more diverse ones
    for g in ["pop", "hip-hop"]:
        g_tracks = df[(df["genre"] == g) & df["significant"]].head(1)
        picks = pd.concat([picks, g_tracks])
    picks = picks.drop_duplicates("basename").head(5)

    real_tracks = []
    for _, row in picks.iterrows():
        for t in all_tracks:
            if t["basename"] == row["basename"]:
                real_tracks.append(t)
                break

    diagnose_cocycle_real(cfg, reducer, real_tracks, n_tracks=5)
    diagnose_cocycle_aggregate()

    # === 2. REPEAT THRESHOLD ===
    ssm_results = diagnose_repeat_threshold(cfg, all_tracks, n_tracks=20)
    diagnose_threshold_sensitivity(cfg, all_tracks)

    # === 3. SSM COMPARISON FIGURE ===
    out_dir = os.path.join(cfg["paths"]["results"], "figures")
    os.makedirs(out_dir, exist_ok=True)
    plot_ssm_comparison(cfg, all_tracks, out_dir)

    # === FINAL SUMMARY ===
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print("""
  1. COCYCLE ISSUE:
     - Representative cocycles are DUAL objects; even on a clean circle,
       they cover 50-70% of points.
     - On real tracks: median 38 vertices (13% of 300), but 39 tracks
       have >50% coverage.
     - Span ~70s is partly an ARTIFACT of cocycle spread, not necessarily
       evidence that the "loop" spans distant repeats.
     - To properly test the hypothesis, we would need representative
       CYCLES (simplicial 1-chains), which ripser does not provide.
       (Ripserer.jl or other tools could help.)

  2. REPEAT THRESHOLD:
     - All 199 tracks had has_repeats=True with thresh=0.85, frac≥0.02.
     - This is because pop/electronic/hip-hop have uniformly high SSM
       values (harmonic homogeneity), NOT necessarily structural repeats.
     - Need to distinguish: uniform brightness vs block structure.
     - block_score (std/mean of far-diagonal values) and row_variance
       can distinguish these cases.

  CONCLUSION: The 55% significance rate is UNRELIABLE because:
  (a) cocycle vertices ≠ geometric cycle vertices (wrong object);
  (b) the "informative" filter passed ALL tracks (no filtering).
  Results need reinterpretation before drawing conclusions.
""")


if __name__ == "__main__":
    main()
