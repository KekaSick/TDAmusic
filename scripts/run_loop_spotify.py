"""
run_loop_spotify.py
-------------------
Тест «петля = музыкальный повтор» на ПОЛНЫХ Spotify-треках (центральные 90 сек).

Условия улучшены vs GTZAN:
  * полные треки (не 30-сек клипы) — дальние повторы вмещаются;
  * жанры с явной повторяющейся структурой (pop, electronic, hip-hop, reggae);
  * центральные 90 сек — куплет + припев, без деградации на краях;
  * агрегат по 50+ трекам.

    .venv/bin/python scripts/run_loop_spotify.py
"""
import os
import sys

# Offline mode — идентично embed_spaces.py
os.environ["HF_HOME"] = "/Users/mverzhbitskiy/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
_fake_home = os.path.join(os.getcwd(), "cache")
os.makedirs(os.path.join(_fake_home, ".keras"), exist_ok=True)
os.environ["HOME"] = _fake_home
os.environ["KERAS_HOME"] = os.path.join(_fake_home, ".keras")

import yaml
import glob
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import mannwhitneyu
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import preprocess
import pointcloud
import persistence


# ======================================================================
# Config
# ======================================================================

CLIP_SEC = 90            # центральные 90 секунд
TARGET_GENRES = ["pop", "electronic", "hip-hop", "reggae"]
SPOTIFY_DIR = "data/top50musicSpotify"
CACHE_DIR = "cache/muq_spotify90"   # отдельный кэш для 90-сек фрагментов
DISTANT_THRESH = 5.0     # секунды
CLOSE_THRESH = 2.0
SSM_REPEAT_THRESH = 0.85  # порог для "трек имеет повторы"
SSM_REPEAT_MIN_FRAC = 0.02  # доля ярких off-diagonal ячеек


def load_cfg():
    return yaml.safe_load(open("config.yaml"))


# ======================================================================
# MuQ extraction (центральные 90 секунд, отдельный кэш)
# ======================================================================

def extract_muq_center(filepath, cfg, clip_sec=CLIP_SEC):
    """Извлечь MuQ-10 для центральных clip_sec секунд трека.

    Кэшируется в CACHE_DIR (отдельно от основного cache/muq/).
    НЕ трогает кэш GTZAN-эмбеддингов.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(filepath))[0]
    cache_path = os.path.join(CACHE_DIR, f"{base}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    sr = cfg["data"]["sample_rate"]
    wav, _ = librosa.load(filepath, sr=sr, mono=True)
    total_sec = len(wav) / sr

    # Центральный фрагмент
    if total_sec > clip_sec:
        start_sec = (total_sec - clip_sec) / 2
        start_sample = int(start_sec * sr)
        end_sample = start_sample + int(clip_sec * sr)
        wav = wav[start_sample:end_sample]
    # Если трек короче — берём весь

    # MuQ extraction (модель держится в _MUQ_MODEL между вызовами)
    import torch
    from muq import MuQ

    device = _device()
    layer = cfg["spaces"]["muq"]["layer"]

    global _MUQ_MODEL
    if _MUQ_MODEL is None:
        model_id = cfg["spaces"]["muq"]["model_id"]
        print(f"  Loading MuQ model (offline, from cache) ...", flush=True)
        _MUQ_MODEL = MuQ.from_pretrained(model_id).to(device).eval()

    wavs = torch.tensor(wav).unsqueeze(0).to(device)
    with torch.no_grad():
        out = _MUQ_MODEL(wavs, output_hidden_states=True)
    h = out.hidden_states[layer].squeeze(0).cpu().numpy()

    np.save(cache_path, h)
    # Cleanup tensors (but NOT the model)
    del wavs, out
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    return h


_MUQ_MODEL = None  # module-level model cache


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ======================================================================
# Helpers (shared with run_loop_interpretation.py logic)
# ======================================================================

def cocycle_vertices_to_seconds(cocycle, sub_indices, takens_starts,
                                 window, target_fps):
    """Cocycle вершины → центры окон (секунды)."""
    vertex_ids = np.unique(cocycle[:, :2].astype(int).ravel())
    seconds = []
    for v in vertex_ids:
        takens_row = sub_indices[v]
        start_frame = takens_starts[takens_row]
        center_sec = (start_frame + window / 2) / target_fps
        seconds.append(center_sec)
    return np.sort(seconds), vertex_ids


def extract_chroma_at_windows(wav, sr, seconds, window_sec, hop_length=512):
    """Извлечь усреднённый chroma вектор для каждого момента."""
    chroma_full = librosa.feature.chroma_cens(y=wav, sr=sr, hop_length=hop_length)
    chroma_fps = sr / hop_length
    n_chroma = chroma_full.shape[1]
    result = []
    half_win = window_sec / 2.0
    for sec in seconds:
        start_frame = max(0, int((sec - half_win) * chroma_fps))
        end_frame = min(n_chroma, int((sec + half_win) * chroma_fps) + 1)
        if end_frame <= start_frame:
            end_frame = start_frame + 1
        chunk = chroma_full[:, start_frame:end_frame]
        result.append(chunk.mean(axis=1))
    return np.array(result)


def has_distant_repeats(wav, sr, hop_length=512, thresh=SSM_REPEAT_THRESH,
                         min_frac=SSM_REPEAT_MIN_FRAC, min_dist_sec=5.0):
    """Проверить, есть ли у трека дальние повторы (яркие off-diagonal в SSM)."""
    chroma = librosa.feature.chroma_cens(y=wav, sr=sr, hop_length=hop_length)
    chroma_fps = sr / hop_length
    min_dist_frames = int(min_dist_sec * chroma_fps)

    ssm = cosine_similarity(chroma.T)
    n = ssm.shape[0]

    # Считаем долю ярких ячеек далеко от диагонали
    total_far = 0
    bright_far = 0
    for i in range(n):
        for j in range(i + min_dist_frames, n):
            total_far += 1
            if ssm[i, j] >= thresh:
                bright_far += 1

    if total_far == 0:
        return False, 0.0
    frac = bright_far / total_far
    return frac >= min_frac, frac


def has_distant_repeats_fast(wav, sr, hop_length=512, thresh=SSM_REPEAT_THRESH,
                              min_frac=SSM_REPEAT_MIN_FRAC, min_dist_sec=5.0):
    """Быстрая версия: vectorized SSM check."""
    chroma = librosa.feature.chroma_cens(y=wav, sr=sr, hop_length=hop_length)
    chroma_fps = sr / hop_length
    min_dist_frames = int(min_dist_sec * chroma_fps)

    ssm = cosine_similarity(chroma.T)
    n = ssm.shape[0]

    # Маска: далеко от диагонали
    rows, cols = np.triu_indices(n, k=min_dist_frames)
    far_vals = ssm[rows, cols]
    total_far = len(far_vals)
    bright_far = (far_vals >= thresh).sum()

    if total_far == 0:
        return False, 0.0
    frac = bright_far / total_far
    return frac >= min_frac, float(frac)


# ======================================================================
# Per-track analysis
# ======================================================================

def analyze_track(filepath, x_pca, wav, sr, cfg, pc_cfg):
    """Полный анализ одного трека: cocycle → seconds → similarity."""
    window = pc_cfg["window"]
    target_fps = cfg["common"]["target_fps"]
    hop_length = cfg["spaces"]["mir"]["hop_length"]
    window_sec = window / target_fps
    clip_dur = len(wav) / sr

    # Build cloud with indices
    cloud, takens_starts, sub_indices = \
        pointcloud.build_cloud_with_indices(x_pca, pc_cfg)

    # Ripser
    res = persistence.compute_diagrams(cloud, maxdim=1, do_cocycles=True)
    dgm1 = res["dgms"][1]

    result = {
        "filepath": filepath,
        "basename": os.path.basename(filepath),
        "duration_sec": clip_dur,
        "n_h1_classes": len(dgm1),
        "max_persistence": 0.0,
        "has_repeats": False,
        "repeat_frac": 0.0,
        "n_loop_vertices": 0,
        "loop_seconds": [],
        "n_close_pairs": 0,
        "n_distant_pairs": 0,
        "close_sim_mean": np.nan,
        "close_sim_std": np.nan,
        "distant_loop_sim_mean": np.nan,
        "distant_loop_sim_std": np.nan,
        "distant_random_sim_mean": np.nan,
        "distant_random_sim_std": np.nan,
        "p_value": np.nan,
        "significant": False,
        "loop_span_sec": 0.0,  # max(seconds) - min(seconds)
    }

    if len(dgm1) == 0:
        return result

    life = dgm1[:, 1] - dgm1[:, 0]
    k = int(np.argmax(life))
    result["max_persistence"] = float(life[k])

    cocycle = res["cocycles"][1][k]
    seconds, vertices = cocycle_vertices_to_seconds(
        cocycle, sub_indices, takens_starts, window, target_fps)

    result["n_loop_vertices"] = len(vertices)
    result["loop_seconds"] = seconds.tolist()
    result["loop_span_sec"] = float(seconds.max() - seconds.min()) if len(seconds) > 1 else 0.0

    # Check if track has distant repeats
    has_rep, rep_frac = has_distant_repeats_fast(wav, sr, hop_length)
    result["has_repeats"] = has_rep
    result["repeat_frac"] = rep_frac

    # Chroma at loop moments (window-averaged)
    loop_chroma = extract_chroma_at_windows(wav, sr, seconds, window_sec, hop_length)

    # Close vs distant pairs
    close_sims = []
    distant_sims = []
    for i in range(len(seconds)):
        for j in range(i + 1, len(seconds)):
            dt = abs(seconds[i] - seconds[j])
            sim = 1.0 - cosine_dist(loop_chroma[i], loop_chroma[j])
            if dt < CLOSE_THRESH:
                close_sims.append(sim)
            elif dt >= DISTANT_THRESH:
                distant_sims.append(sim)

    result["n_close_pairs"] = len(close_sims)
    result["n_distant_pairs"] = len(distant_sims)

    if close_sims:
        result["close_sim_mean"] = float(np.mean(close_sims))
        result["close_sim_std"] = float(np.std(close_sims))

    if distant_sims:
        result["distant_loop_sim_mean"] = float(np.mean(distant_sims))
        result["distant_loop_sim_std"] = float(np.std(distant_sims))

    # Random distant pairs (control)
    rng = np.random.default_rng(42)
    n_random = max(500, len(distant_sims) * 10)
    all_times = np.linspace(window_sec / 2, clip_dur - window_sec / 2, 200)
    all_chroma = extract_chroma_at_windows(wav, sr, all_times, window_sec, hop_length)

    random_distant = []
    attempts = 0
    while len(random_distant) < n_random and attempts < n_random * 20:
        a, b = rng.choice(len(all_times), 2, replace=False)
        dt = abs(all_times[a] - all_times[b])
        if dt >= DISTANT_THRESH:
            sim = 1.0 - cosine_dist(all_chroma[a], all_chroma[b])
            random_distant.append(sim)
        attempts += 1
    random_distant = np.array(random_distant)

    result["distant_random_sim_mean"] = float(random_distant.mean())
    result["distant_random_sim_std"] = float(random_distant.std())

    # Mann-Whitney
    if len(distant_sims) >= 3 and len(random_distant) >= 3:
        stat, p = mannwhitneyu(distant_sims, random_distant, alternative="greater")
        result["p_value"] = float(p)
        result["significant"] = p < 0.05

    return result


# ======================================================================
# Visualization
# ======================================================================

def plot_track_panel(filepath, wav, sr, seconds, cocycle, sub_indices,
                     takens_starts, window, target_fps, result, out_path):
    """Timeline + chromagram + SSM для одного трека."""
    hop_length = 512
    clip_dur = len(wav) / sr
    chroma = librosa.feature.chroma_cens(y=wav, sr=sr, hop_length=hop_length)
    chroma_fps = sr / hop_length

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

    # 1. Chromagram + loop moments
    ax1 = fig.add_subplot(gs[0, 0])
    times_chroma = np.arange(chroma.shape[1]) / chroma_fps
    ax1.pcolormesh(times_chroma, np.arange(12), chroma, cmap="magma", shading="auto")
    for sec in seconds:
        ax1.axvline(sec, color="cyan", alpha=0.7, lw=1.0, ls="--")
    ax1.set_ylabel("Pitch class")
    ax1.set_xlabel("Time (s)")
    ax1.set_title("Chromagram + H1 loop moments")
    ax1.set_yticks(np.arange(12))
    ax1.set_yticklabels(["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"],
                         fontsize=7)

    # 2. SSM
    ax2 = fig.add_subplot(gs[0, 1])
    ssm = cosine_similarity(chroma.T)
    im = ax2.imshow(ssm, origin="lower", cmap="magma", aspect="auto",
                     extent=[0, clip_dur, 0, clip_dur])
    for sec in seconds:
        ax2.axvline(sec, color="cyan", alpha=0.4, lw=0.6)
        ax2.axhline(sec, color="cyan", alpha=0.4, lw=0.6)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Time (s)")
    ax2.set_title("Self-Similarity (chroma)")
    plt.colorbar(im, ax=ax2, shrink=0.8)

    # 3. Timeline with loop edges
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axhline(0, color="gray", lw=0.5)
    ax3.set_xlim(0, clip_dur)
    ax3.set_ylim(-0.5, 0.5)
    for sec in seconds:
        ax3.axvline(sec, color="red", alpha=0.8, lw=1.5)
    # Draw arcs for cocycle edges
    if cocycle is not None:
        for edge_i, edge_j, val in cocycle:
            ei, ej = int(edge_i), int(edge_j)
            sec_i = (takens_starts[sub_indices[ei]] + window / 2) / target_fps
            sec_j = (takens_starts[sub_indices[ej]] + window / 2) / target_fps
            ax3.annotate("", xy=(sec_j, 0), xytext=(sec_i, 0),
                          arrowprops=dict(arrowstyle="-",
                                          connectionstyle="arc3,rad=0.3",
                                          color="blue", alpha=0.3, lw=0.8))
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Loop")
    ax3.set_yticks([])
    ax3.set_title(f"Loop edges (span={result['loop_span_sec']:.1f}s)")

    # 4. Text summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    txt = (f"Track: {os.path.basename(filepath)}\n"
           f"Duration: {clip_dur:.1f}s\n"
           f"Max H1 pers: {result['max_persistence']:.4f}\n"
           f"Vertices: {result['n_loop_vertices']}, "
           f"Span: {result['loop_span_sec']:.1f}s\n"
           f"Has repeats: {result['has_repeats']} "
           f"(frac={result['repeat_frac']:.3f})\n\n"
           f"Distant loop sim: {result['distant_loop_sim_mean']:.4f} "
           f"± {result['distant_loop_sim_std']:.4f} "
           f"(n={result['n_distant_pairs']})\n"
           f"Distant random sim: {result['distant_random_sim_mean']:.4f} "
           f"± {result['distant_random_sim_std']:.4f}\n"
           f"p-value: {result['p_value']:.4f}\n"
           f"Significant: {result['significant']}")
    ax4.text(0.05, 0.95, txt, transform=ax4.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(f"{os.path.basename(filepath)} — H1 Loop Interpretation",
                 fontsize=12, fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

def main():
    cfg = load_cfg()
    sr = cfg["data"]["sample_rate"]
    pc_cfg = cfg["pointcloud"].copy()
    pc_cfg["takens_pca_dim"] = None
    target_fps = cfg["common"]["target_fps"]
    window = pc_cfg["window"]

    # ---- Collect track paths ----
    all_tracks = []
    for genre in TARGET_GENRES:
        wavs = sorted(glob.glob(os.path.join(SPOTIFY_DIR, genre, "*.wav")))
        for w in wavs:
            all_tracks.append({"filepath": w, "genre": genre,
                                "basename": os.path.basename(w)})
    print(f"Total tracks from {TARGET_GENRES}: {len(all_tracks)}")

    # ---- Extract MuQ embeddings (center 90s) ----
    print("\n=== Extracting MuQ-10 embeddings (center 90s) ===")
    raw_embeddings = {}
    for i, t in enumerate(all_tracks):
        fp = t["filepath"]
        print(f"  [{i+1}/{len(all_tracks)}] {t['basename']}...", end="", flush=True)
        emb = extract_muq_center(fp, cfg, clip_sec=CLIP_SEC)
        raw_embeddings[fp] = emb
        print(f" shape={emb.shape}")

    # ---- Train PCA on all these tracks ----
    print("\n=== Training PCA reducer ===")
    reducer = preprocess.PCAReducer(
        dim=cfg["common"]["pca_dim"],
        standardize=cfg["common"].get("standardize_features", True),
        normalize=cfg["common"]["normalize"])
    reducer.fit(list(raw_embeddings.values()))
    print(f"  PCA dim={reducer.dim}, explained={reducer.explained:.4f}")

    # ---- Analyze each track ----
    print("\n=== Analyzing tracks ===")
    results = []
    # Track best/worst for figures
    best_sig = None
    best_nonsig = None

    for i, t in enumerate(all_tracks):
        fp = t["filepath"]
        genre = t["genre"]
        print(f"\n[{i+1}/{len(all_tracks)}] {t['basename']} ({genre})")

        # Preprocess
        x_raw = raw_embeddings[fp]
        x_pca = preprocess.prepare_track(
            x_raw, cfg["spaces"]["muq"], cfg["common"], reducer)
        print(f"  PCA shape: {x_pca.shape}")

        # Load wav (center crop for chroma)
        wav_full, _ = librosa.load(fp, sr=sr, mono=True)
        total_sec = len(wav_full) / sr
        if total_sec > CLIP_SEC:
            start = int((total_sec - CLIP_SEC) / 2 * sr)
            wav = wav_full[start:start + int(CLIP_SEC * sr)]
        else:
            wav = wav_full
        clip_dur = len(wav) / sr

        # Analyze
        r = analyze_track(fp, x_pca, wav, sr, cfg, pc_cfg)
        r["genre"] = genre
        results.append(r)

        sig = "***" if r["significant"] else ""
        print(f"  max_pers={r['max_persistence']:.4f}, "
              f"vertices={r['n_loop_vertices']}, "
              f"span={r['loop_span_sec']:.1f}s, "
              f"repeats={r['has_repeats']}, "
              f"p={r['p_value']:.4f} {sig}")

        # Track best significant and non-significant for figure
        if r["significant"] and r["n_distant_pairs"] >= 5:
            if best_sig is None or r["p_value"] < best_sig["p_value"]:
                best_sig = {**r, "x_pca": x_pca, "wav": wav}
        if not r["significant"] and r["n_distant_pairs"] >= 5:
            if best_nonsig is None or r["max_persistence"] > best_nonsig["max_persistence"]:
                best_nonsig = {**r, "x_pca": x_pca, "wav": wav}

    # ---- Save results table ----
    out_dir_tables = os.path.join(cfg["paths"]["results"], "tables")
    out_dir_figs = os.path.join(cfg["paths"]["results"], "figures")
    os.makedirs(out_dir_tables, exist_ok=True)
    os.makedirs(out_dir_figs, exist_ok=True)

    df = pd.DataFrame(results)
    # Select columns for CSV
    cols = ["basename", "genre", "duration_sec", "max_persistence",
            "has_repeats", "repeat_frac", "n_loop_vertices", "loop_span_sec",
            "n_close_pairs", "n_distant_pairs",
            "close_sim_mean", "distant_loop_sim_mean", "distant_loop_sim_std",
            "distant_random_sim_mean", "distant_random_sim_std",
            "p_value", "significant"]
    df[cols].to_csv(os.path.join(out_dir_tables, "loop_spotify.csv"), index=False)
    print(f"\n→ {os.path.join(out_dir_tables, 'loop_spotify.csv')}")

    # ---- Aggregate statistics ----
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    n_total = len(df)
    n_with_h1 = len(df[df["max_persistence"] > 0])
    n_with_repeats = len(df[df["has_repeats"]])
    n_informative = len(df[df["has_repeats"] & (df["n_distant_pairs"] >= 3)])
    n_significant = len(df[df["has_repeats"] & df["significant"]])

    print(f"Total tracks: {n_total}")
    print(f"Tracks with H1: {n_with_h1}")
    print(f"Tracks with distant repeats (informative): {n_with_repeats}")
    print(f"Informative + enough distant pairs: {n_informative}")
    print(f"Hypothesis confirmed (p<0.05, informative): {n_significant}")

    if n_informative > 0:
        frac_sig = n_significant / n_informative
        print(f"Fraction significant: {frac_sig:.1%}")
    else:
        frac_sig = 0
        print("No informative tracks!")

    # Средний эффект среди информативных
    info_df = df[df["has_repeats"] & (df["n_distant_pairs"] >= 3)]
    if len(info_df) > 0:
        mean_loop = info_df["distant_loop_sim_mean"].mean()
        mean_rand = info_df["distant_random_sim_mean"].mean()
        print(f"\nMean distant loop sim (informative): {mean_loop:.4f}")
        print(f"Mean distant random sim (informative): {mean_rand:.4f}")
        print(f"Mean effect (loop - random): {mean_loop - mean_rand:.4f}")

    # Close similarity — are loops local or spanning?
    mean_close = df["close_sim_mean"].dropna().mean()
    mean_span = df["loop_span_sec"].mean()
    print(f"\nMean close similarity: {mean_close:.4f}")
    print(f"Mean loop span: {mean_span:.1f}s")

    # Qualitative comparison with GTZAN
    # NB: числовое сравнение sim-значений между GTZAN и Spotify некорректно
    # (разные PCA-пространства). Сравниваем только качественный вывод.
    print("\n--- Qualitative comparison with GTZAN ---")
    print("GTZAN: hypothesis NOT confirmed (p=0.23) on 30s classical clip")
    print("  Conditions: short clip (no room for distant repeats), classical genre")
    print("  (develops rather than repeats). PCA trained on GTZAN.")
    if len(info_df) > 0:
        qual = "CONFIRMED" if frac_sig > 0.5 else (
            "PARTIALLY" if n_significant > 0 else "NOT CONFIRMED")
        print(f"Spotify: hypothesis {qual} on 90s clips with repetitive genres")
        print(f"  {n_significant}/{n_informative} informative tracks significant (p<0.05)")
        print(f"  NB: PCA trained on Spotify — numbers not directly comparable to GTZAN")

    # ---- Visualizations ----
    # 1. Panel for best significant track
    if best_sig is not None:
        fp = best_sig["filepath"]
        wav_viz = best_sig["wav"]
        x_pca = best_sig["x_pca"]
        cloud, takens_starts, sub_indices = \
            pointcloud.build_cloud_with_indices(x_pca, pc_cfg)
        res_rip = persistence.compute_diagrams(cloud, maxdim=1, do_cocycles=True)
        dgm1 = res_rip["dgms"][1]
        life = dgm1[:, 1] - dgm1[:, 0]
        k = int(np.argmax(life))
        cocycle = res_rip["cocycles"][1][k]
        seconds, _ = cocycle_vertices_to_seconds(
            cocycle, sub_indices, takens_starts, window, target_fps)
        plot_track_panel(fp, wav_viz, sr, seconds, cocycle, sub_indices,
                          takens_starts, window, target_fps, best_sig,
                          os.path.join(out_dir_figs, "loop_spotify_significant.png"))
        print(f"\n→ {os.path.join(out_dir_figs, 'loop_spotify_significant.png')}")

    # 2. Panel for best non-significant track
    if best_nonsig is not None:
        fp = best_nonsig["filepath"]
        wav_viz = best_nonsig["wav"]
        x_pca = best_nonsig["x_pca"]
        cloud, takens_starts, sub_indices = \
            pointcloud.build_cloud_with_indices(x_pca, pc_cfg)
        res_rip = persistence.compute_diagrams(cloud, maxdim=1, do_cocycles=True)
        dgm1 = res_rip["dgms"][1]
        life = dgm1[:, 1] - dgm1[:, 0]
        k = int(np.argmax(life))
        cocycle = res_rip["cocycles"][1][k]
        seconds, _ = cocycle_vertices_to_seconds(
            cocycle, sub_indices, takens_starts, window, target_fps)
        plot_track_panel(fp, wav_viz, sr, seconds, cocycle, sub_indices,
                          takens_starts, window, target_fps, best_nonsig,
                          os.path.join(out_dir_figs, "loop_spotify_nonsignificant.png"))
        print(f"→ {os.path.join(out_dir_figs, 'loop_spotify_nonsignificant.png')}")

    # 3. Aggregate boxplot
    fig_agg, ax = plt.subplots(figsize=(10, 6))
    data_box = []
    labels_box = []

    close_vals = df["close_sim_mean"].dropna().values
    if len(close_vals) > 0:
        data_box.append(close_vals)
        labels_box.append(f"Close loop\n(<{CLOSE_THRESH}s)")

    dist_loop = info_df["distant_loop_sim_mean"].dropna().values
    if len(dist_loop) > 0:
        data_box.append(dist_loop)
        labels_box.append(f"Distant loop\n(>{DISTANT_THRESH}s)")

    dist_rand = info_df["distant_random_sim_mean"].dropna().values
    if len(dist_rand) > 0:
        data_box.append(dist_rand)
        labels_box.append(f"Distant random\n(>{DISTANT_THRESH}s)")

    if data_box:
        bp = ax.boxplot(data_box, tick_labels=labels_box, patch_artist=True,
                          widths=0.5)
        colors = ["#4CAF50", "#2196F3", "#9E9E9E"][:len(data_box)]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Cosine similarity (chroma)")
        ax.set_title(f"Spotify Loop Interpretation — Aggregate "
                      f"({n_significant}/{n_informative} significant, "
                      f"n={n_total} tracks)")
        ax.grid(axis="y", alpha=0.3)
        fig_agg.tight_layout()
        fig_agg.savefig(os.path.join(out_dir_figs, "loop_spotify_aggregate.png"),
                         bbox_inches="tight", dpi=150)
        print(f"→ {os.path.join(out_dir_figs, 'loop_spotify_aggregate.png')}")
    plt.close(fig_agg)

    # ---- Save text report ----
    report_path = os.path.join(out_dir_tables, "loop_spotify_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("H1 LOOP INTERPRETATION — SPOTIFY FULL TRACKS (90s center)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Genres: {TARGET_GENRES}\n")
        f.write(f"Clip: center {CLIP_SEC}s\n")
        f.write(f"Total tracks: {n_total}\n")
        f.write(f"Tracks with H1: {n_with_h1}\n")
        f.write(f"Tracks with distant repeats: {n_with_repeats}\n")
        f.write(f"Informative (repeats + ≥3 distant pairs): {n_informative}\n")
        f.write(f"Significant (p<0.05): {n_significant}\n")
        if n_informative > 0:
            f.write(f"Fraction significant: {frac_sig:.1%}\n\n")
            f.write(f"Mean distant loop sim: {mean_loop:.4f}\n")
            f.write(f"Mean distant random sim: {mean_rand:.4f}\n")
            f.write(f"Mean effect: {mean_loop - mean_rand:.4f}\n\n")
        f.write(f"Mean close similarity: {mean_close:.4f}\n")
        f.write(f"Mean loop span: {mean_span:.1f}s\n\n")
        f.write("-" * 70 + "\n")
        f.write("Qualitative comparison with GTZAN:\n")
        f.write("  GTZAN: NOT confirmed (p=0.23) on 30s classical clip.\n")
        f.write("  Conditions were unfavorable: short clip (no room for distant\n")
        f.write("  repeats), classical genre (develops, does not repeat verbatim).\n")
        f.write("  PCA was trained on GTZAN data.\n\n")
        if len(info_df) > 0:
            qual = "CONFIRMED" if frac_sig > 0.5 else (
                "PARTIALLY" if n_significant > 0 else "NOT CONFIRMED")
            f.write(f"  Spotify: {qual} on 90s clips with repetitive genres.\n")
            f.write(f"  {n_significant}/{n_informative} informative tracks "
                    f"significant (p<0.05).\n")
            f.write(f"  NB: PCA trained on Spotify data — similarity values are\n")
            f.write(f"  in a different space, not directly comparable to GTZAN.\n")
        f.write("\n")

        # Key question: local or spanning?
        local_loops = len(df[df["loop_span_sec"] < 10])
        spanning_loops = len(df[df["loop_span_sec"] >= 30])
        f.write("-" * 70 + "\n")
        f.write("Loop locality:\n")
        f.write(f"  Loops spanning <10s: {local_loops}/{n_total}\n")
        f.write(f"  Loops spanning ≥30s: {spanning_loops}/{n_total}\n")
        f.write(f"  Mean span: {mean_span:.1f}s\n")
        f.write(f"  → H1 loops {'tend to be LOCAL' if mean_span < 20 else 'SPAN distant repeats'}\n")
        f.write("=" * 70 + "\n")
    print(f"→ {report_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
