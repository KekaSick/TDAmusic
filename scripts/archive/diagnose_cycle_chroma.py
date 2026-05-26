import os
import sys
import yaml
import glob
import numpy as np
import pandas as pd
import dionysus as d
import ripser as ripser_lib
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine as cosine_dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import preprocess
import pointcloud

CACHE_DIR = "cache/muq_spotify90"
SPOTIFY_DIR = "data/top50musicSpotify"
CLIP_SEC = 90

def get_both_representatives(X):
    r = ripser_lib.ripser(X, maxdim=1, do_cocycles=True)
    dgm1 = r["dgms"][1]
    if len(dgm1) == 0:
        return set(), set()
        
    life = dgm1[:, 1] - dgm1[:, 0]
    life[np.isinf(dgm1[:, 1])] = -1
    idx = int(np.argmax(life))
    if life[idx] < 0:
        return set(), set()
        
    r_death = float(dgm1[idx, 1])
    cc = r["cocycles"][1][idx]
    c_verts = set(np.unique(cc[:, :2].astype(int)))
    
    max_radius = r_death * 1.1
    f = d.fill_rips(X.astype(np.float64), 2, max_radius)
    m = d.homology_persistence(f, prime=2)
    dgms = d.init_diagrams(m, f)
    
    dgm1 = dgms[1]
    h1_ho = [pt for pt in dgm1 if pt.death < float("inf")]
    if not h1_ho:
        return c_verts, set()
        
    best_ho = max(h1_ho, key=lambda p: p.death - p.birth)
    death_idx = best_ho.data
    birth_idx = m.pair(death_idx)
    chain = m[birth_idx]
    
    cy_verts = set()
    for entry in chain:
        s = f[entry.index]
        if len(list(s)) == 2:
            cy_verts.update(list(s))
            
    return c_verts, cy_verts

def _mean_distant_sim(chroma_vecs, times, distant_thresh):
    sims = []
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if abs(times[i] - times[j]) >= distant_thresh:
                sims.append(1.0 - cosine_dist(chroma_vecs[i], chroma_vecs[j]))
    return float(np.mean(sims)) if sims else np.nan, len(sims)

def chroma_test(wav, sr, seconds, window_sec, hop_length=512, distant_thresh=5.0, n_perm=2000, seed=42):
    chroma_full = librosa.feature.chroma_cens(y=wav, sr=sr, hop_length=hop_length)
    chroma_fps = sr / hop_length
    n_chroma = chroma_full.shape[1]
    clip_dur = len(wav) / sr

    half_win = window_sec / 2
    def get_chroma(sec):
        sf = max(0, int((sec - half_win) * chroma_fps))
        ef = min(n_chroma, int((sec + half_win) * chroma_fps) + 1)
        if ef <= sf: ef = sf + 1
        return chroma_full[:, sf:ef].mean(axis=1)

    k = len(seconds)
    if k == 0:
        return np.nan
        
    observed_stat, n_distant = _mean_distant_sim(
        np.array([get_chroma(s) for s in seconds]), seconds, distant_thresh)

    if np.isnan(observed_stat) or n_distant < 1:
        return np.nan

    all_times = np.linspace(window_sec / 2, clip_dur - window_sec / 2, max(200, k + 10))
    all_chroma = np.array([get_chroma(s) for s in all_times])

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
        return np.nan

    p_value = float((np.sum(null_stats >= observed_stat) + 1) / (len(null_stats) + 1))
    return p_value

def main():
    print("=" * 70)
    print("CYCLE CHROMA TEST AND VERTEX DISTRIBUTION")
    print("=" * 70)
    
    cfg = yaml.safe_load(open("config.yaml"))
    target_fps = cfg["common"]["target_fps"]
    sr = cfg["data"]["sample_rate"]
    hop_length = cfg["spaces"]["mir"]["hop_length"]
    
    # 1. Load tracks & fit PCA
    all_tracks = []
    for genre in ["pop", "electronic", "hip-hop", "reggae"]:
        wavs = sorted(glob.glob(os.path.join(SPOTIFY_DIR, genre, "*.wav")))
        for w in wavs:
            all_tracks.append({"filepath": w, "genre": genre, "basename": os.path.basename(w)})
            
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
    
    # 2. Pick 15 tracks
    df = pd.read_csv("results/tables/loop_spotify.csv")
    picks = pd.concat([
        df[df["significant"]].nsmallest(3, "n_loop_vertices"),
        df[df["significant"]].nlargest(3, "n_loop_vertices"),
        df[~df["significant"] & df["p_value"].notna()].nlargest(3, "max_persistence")
    ])
    for g in ["pop", "electronic", "hip-hop", "reggae"]:
        picks = pd.concat([picks, df[df["genre"] == g].head(2)])
    picks = picks.drop_duplicates("basename").head(15)
    
    real_tracks = []
    for _, row in picks.iterrows():
        for t in all_tracks:
            if t["basename"] == row["basename"]:
                t["cocycle_p_prev"] = row.get("p_value", np.nan)
                real_tracks.append(t)
                break
                
    pc_cfg = cfg["pointcloud"].copy()
    pc_cfg["takens_pca_dim"] = None
    window = pc_cfg["window"]
    window_sec = window / target_fps
    
    res_df = []
    
    # Matplotlib setup
    n_tracks = len(real_tracks)
    cols = 3
    rows = int(np.ceil(n_tracks / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    
    for i, t in enumerate(real_tracks):
        base = t["basename"]
        print(f"\n[{i+1}/{n_tracks}] {base}")
        
        # Build cloud
        cache_base = os.path.splitext(base)[0]
        emb = np.load(os.path.join(CACHE_DIR, f"{cache_base}.npy"))
        x_pca = preprocess.prepare_track(emb, cfg["spaces"]["muq"], cfg["common"], reducer)
        cloud, takens_starts, sub_indices = pointcloud.build_cloud_with_indices(x_pca, pc_cfg)
        
        c_verts, cy_verts = get_both_representatives(cloud)
        
        # Map to seconds
        secs = []
        for v in cy_verts:
            sec = (takens_starts[sub_indices[int(v)]] + window/2) / target_fps
            secs.append(sec)
        secs = np.array(secs)
        
        if len(secs) == 0:
            print("  No cycle vertices found.")
            continue
            
        cy_span = max(secs) - min(secs) if len(secs) > 1 else 0.0
        
        # Chroma test
        wav_full, _ = librosa.load(t["filepath"], sr=sr, mono=True)
        total_sec = len(wav_full) / sr
        if total_sec > CLIP_SEC:
            start = int((total_sec - CLIP_SEC) / 2 * sr)
            wav = wav_full[start:start + int(CLIP_SEC * sr)]
        else:
            wav = wav_full
            
        cy_p_val = chroma_test(wav, sr, secs, window_sec, hop_length)
        
        # Bins (10-sec)
        bins = np.arange(0, 100, 10)
        hist, _ = np.histogram(secs, bins=bins)
        n_nonempty_bins = np.sum(hist > 0)
        max_bin_frac = np.max(hist) / len(secs) if len(secs) > 0 else 0
        
        # Plot
        ax = axes[i]
        ax.bar(bins[:-1], hist, width=10, align='edge', color='blue', alpha=0.7, edgecolor='black')
        ax.set_title(f"{base[:30]}\\np={cy_p_val:.4f} max_bin={max_bin_frac:.0%}", fontsize=9)
        ax.set_xlim(0, 90)
        ax.set_xticks(bins)
        if i % cols == 0: ax.set_ylabel("Vertices")
        
        # Save results
        print(f"  Cycle verts: {len(secs)} (span {cy_span:.1f}s)")
        print(f"  p-value: Cycle={cy_p_val:.4f}, Cocycle_prev={t['cocycle_p_prev']:.4f}")
        print(f"  Bins: {n_nonempty_bins} non-empty, max_bin_frac={max_bin_frac:.0%}")
        
        res_df.append({
            "basename": base,
            "genre": t["genre"],
            "cycle_verts": len(secs),
            "cycle_span": cy_span,
            "cycle_p_value": cy_p_val,
            "cocycle_p_prev": t["cocycle_p_prev"],
            "n_nonempty_bins": n_nonempty_bins,
            "max_bin_fraction": max_bin_frac
        })
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    fig_path = "results/figures/diagnose_cycle_vertex_histogram.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nSaved histogram to {fig_path}")
    
    df_out = pd.DataFrame(res_df)
    os.makedirs("results/tables", exist_ok=True)
    csv_path = "results/tables/diagnose_cycle_chroma.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    world1_count = 0
    world2_count = 0
    
    for row in res_df:
        cy_p = row["cycle_p_value"]
        max_frac = row["max_bin_fraction"]
        n_bins = row["n_nonempty_bins"]
        
        # Criteria: World 1 if p < 0.05 AND (max_frac >= 0.4 OR n_bins <= 3)
        # It's a heuristic for "clustered and significant"
        is_clustered = (max_frac >= 0.4 or n_bins <= 3)
        is_sig = (not np.isnan(cy_p)) and (cy_p < 0.05)
        
        if is_sig and is_clustered:
            world1_count += 1
            print(f"  {row['basename'][:30]:<30} -> WORLD 1 (Repeats) [p={cy_p:.3f}, clustered]")
        else:
            world2_count += 1
            reason = []
            if not is_sig: reason.append(f"p={cy_p:.3f}")
            if not is_clustered: reason.append(f"spread over {n_bins} bins")
            print(f"  {row['basename'][:30]:<30} -> WORLD 2 (Artifact) [{', '.join(reason)}]")
            
    print(f"\nTOTAL: World 1 = {world1_count}, World 2 = {world2_count}")
    
    if world1_count > world2_count * 2:
        print("CONCLUSION: Mostly WORLD 1. The cycle captures real clustered repeats!")
    elif world2_count > world1_count * 2:
        print("CONCLUSION: Mostly WORLD 2. The geometric cycle does NOT mean musical repeats (significant or clustered).")
    else:
        print("CONCLUSION: MIXED WORLD. Some tracks have clustered repeats, others are just geometric artifacts.")

if __name__ == "__main__":
    main()
