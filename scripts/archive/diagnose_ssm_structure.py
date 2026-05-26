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
from sklearn.metrics.pairwise import cosine_similarity

SPOTIFY_DIR = "data/top50musicSpotify"
CLIP_SEC = 90

def compute_ssm_structure(wav, sr, hop_length=512, min_dist_sec=5.0):
    chroma = librosa.feature.chroma_cens(y=wav, sr=sr, hop_length=hop_length)
    chroma_fps = sr / hop_length
    min_dist_frames = int(min_dist_sec * chroma_fps)
    ssm = cosine_similarity(chroma.T)
    n = ssm.shape[0]
    rows, cols = np.triu_indices(n, k=min_dist_frames)
    far_vals = ssm[rows, cols]
    if len(far_vals) == 0:
        return 0.0, 0.0
    far_mean = float(far_vals.mean())
    far_std = float(far_vals.std())
    block_score = far_std / max(far_mean, 1e-8)
    
    row_means = []
    for i in range(0, n, max(1, n // 20)):
        far_row = ssm[i, :].copy()
        far_row[max(0, i - min_dist_frames):min(n, i + min_dist_frames)] = np.nan
        m = np.nanmean(far_row)
        if not np.isnan(m):
            row_means.append(m)
    row_var = float(np.var(row_means)) if len(row_means) > 1 else 0.0
    return block_score, row_var

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    sr = cfg["data"]["sample_rate"]
    hop_length = cfg["spaces"]["mir"]["hop_length"]
    
    all_tracks = []
    for genre in ["pop", "electronic", "hip-hop", "reggae"]:
        wavs = sorted(glob.glob(os.path.join(SPOTIFY_DIR, genre, "*.wav")))
        for w in wavs:
            all_tracks.append({"filepath": w, "genre": genre, "basename": os.path.basename(w)})
            
    res_df = []
    from tqdm import tqdm
    print(f"Processing {len(all_tracks)} tracks for SSM structure...")
    for t in tqdm(all_tracks, desc="SSM structure"):
        wav_full, _ = librosa.load(t["filepath"], sr=sr, mono=True)
        total_sec = len(wav_full) / sr
        if total_sec > CLIP_SEC:
            start = int((total_sec - CLIP_SEC) / 2 * sr)
            wav = wav_full[start:start + int(CLIP_SEC * sr)]
        else:
            wav = wav_full
            
        bs, rv = compute_ssm_structure(wav, sr, hop_length)
        res_df.append({
            "basename": t["basename"],
            "genre": t["genre"],
            "block_score": bs,
            "row_var": rv
        })
            
    df = pd.DataFrame(res_df)
    
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/ssm_block_scores.csv", index=False)
    print("\nSaved block scores to results/tables/ssm_block_scores.csv")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["block_score"], bins=30, color="blue", alpha=0.7)
    axes[0].set_title("Block Score Distribution")
    axes[0].set_xlabel("block_score (std/mean)")
    axes[0].set_ylabel("Count")
    
    axes[1].hist(df["row_var"], bins=30, color="green", alpha=0.7)
    axes[1].set_title("Row Variance Distribution")
    axes[1].set_xlabel("row_var")
    
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/diagnose_ssm_histograms.png")
    print("Saved histograms to results/figures/diagnose_ssm_histograms.png")
    
    # Determine empirical thresholds (e.g. top 25% or mean + 0.5*std)
    # Let's use the 75th percentile for both as an initial strict filter
    bs_thresh = df["block_score"].quantile(0.75)
    rv_thresh = df["row_var"].quantile(0.75)
    
    df["has_repeats"] = (df["block_score"] > bs_thresh) & (df["row_var"] > rv_thresh)
    n_repeats = df["has_repeats"].sum()
    
    print("\n" + "="*70)
    print("SSM STRUCTURAL REPEATS CRITERIA (STAGE 3)")
    print("="*70)
    print(f"Block Score Threshold (75th perc): {bs_thresh:.4f}")
    print(f"Row Var Threshold (75th perc):     {rv_thresh:.6f}")
    print(f"Tracks passing BOTH thresholds:    {n_repeats} / {len(df)}")
    print(f"This leaves only the top ~{100 * n_repeats / len(df):.1f}% tracks with the strongest block structure.")

if __name__ == "__main__":
    main()
