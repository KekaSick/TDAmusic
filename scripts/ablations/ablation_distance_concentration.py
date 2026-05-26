import os
import sys
import copy
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from common import get_ablation_data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import embed_spaces
import preprocess
import pointcloud
import persistence
from run_controls import process_space

def compute_concentration(points, max_points=2000):
    if points is None or len(points) < 2:
        return np.nan, np.nan, np.nan
    
    if len(points) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        
    dists = pdist(points)
    if len(dists) == 0:
        return np.nan, np.nan, np.nan
    m = np.mean(dists)
    s = np.std(dists)
    return m, s, (s / m if m > 0 else np.nan)

def _process_file(f, space, cfg, reducer):
    try:
        # Raw
        x_raw = embed_spaces.extract(f, space, cfg)
        m_raw, s_raw, c_raw = compute_concentration(x_raw)
        
        # PCA
        x_pca = preprocess.prepare_track(x_raw, cfg["spaces"][space], cfg["common"], reducer)
        m_pca, s_pca, c_pca = compute_concentration(x_pca)
        
        # Takens
        cloud = pointcloud.build_cloud(x_pca, cfg["pointcloud"])
        m_tak, s_tak, c_tak = compute_concentration(cloud)
        
        return {
            "raw": (m_raw, s_raw, c_raw),
            "pca": (m_pca, s_pca, c_pca),
            "takens": (m_tak, s_tak, c_tak)
        }
    except Exception as e:
        print(f"Skipping {f} due to error: {e}")
        return None


def main():
    cfg, files, labels, tr_idx, te_idx = get_ablation_data()
    spaces = ["muq", "mert", "encodec"]
    
    results = []
    
    for space in tqdm(spaces, desc="Distance Concentration ablation"):
        print(f"\nProcessing {space.upper()}...")
        
        # We also need max persistence for correlation, so we just run process_space first to get baseline pers
        res = process_space(space, cfg, files, labels, tr_idx, np.random.default_rng(42), compute_random=False, compute_iaaft=False)
        pers_mean = res.get("max_pers_real_mean")
        
        reducer = preprocess.PCAReducer(
            cfg["common"]["pca_dim"],
            standardize=cfg["common"].get("standardize_features", True),
            normalize=cfg["common"]["normalize"]
        )
        # fit reducer on train
        raw_tr = [embed_spaces.extract(files[i], space, cfg) for i in tr_idx]
        tr_frames = [
            preprocess.resample_fps(
                r,
                cfg["spaces"][space].get("native_fps", cfg["common"]["target_fps"]),
                cfg["common"]["target_fps"]
            ) for r in raw_tr
        ]
        reducer.fit(tr_frames)
        
        metrics = {"raw": [], "pca": [], "takens": []}
        
        results_gen = Parallel(n_jobs=-1, return_as="generator")(
            delayed(_process_file)(f, space, cfg, reducer)
            for f in files
        )
        file_results = [r for r in tqdm(results_gen, total=len(files), desc=f"Concentration {space}", leave=False) if r is not None]
        
        for r in file_results:
            metrics["raw"].append(r["raw"])
            metrics["pca"].append(r["pca"])
            metrics["takens"].append(r["takens"])

            
        def agg(lst, idx):
            vals = [v[idx] for v in lst if not np.isnan(v[idx])]
            return np.mean(vals) if vals else np.nan
            
        row = {
            "space": space,
            "max_persistence": pers_mean,
            "raw_mean_dist": agg(metrics["raw"], 0),
            "raw_std_dist": agg(metrics["raw"], 1),
            "raw_concentration": agg(metrics["raw"], 2),
            "pca_mean_dist": agg(metrics["pca"], 0),
            "pca_std_dist": agg(metrics["pca"], 1),
            "pca_concentration": agg(metrics["pca"], 2),
            "takens_mean_dist": agg(metrics["takens"], 0),
            "takens_std_dist": agg(metrics["takens"], 1),
            "takens_concentration": agg(metrics["takens"], 2),
        }
        results.append(row)
        
    df = pd.DataFrame(results)
    out_path = "results/tables/ablations/ablation_distance_concentration.csv"
    df.to_csv(out_path, index=False)
    
    print("\n--- VERDICT ---")
    n_tracks = len(files)
    print(f"Sample: {n_tracks} tracks")
    
    # If concentration ratio is extremely low (std << mean, e.g. < 0.1) and persistence is high
    # it might be a high-dimensional artifact.
    artifact = False
    for _, row in df.iterrows():
        print(f"[{row['space']}] Takens concentration: {row['takens_concentration']:.4f}, max_persistence: {row['max_persistence']:.4f}")
        if row["takens_concentration"] < 0.15 and row["max_persistence"] > 0.5:
            artifact = True
            
    print(f"\nWARNING: subset verdict n={n_tracks}; thresholds are heuristic. Use the numbers, not only the binary verdict.")
    if artifact:
        print("VERDICT: strongly configuration-dependent / possible high-dimensional artifact")
    else:
        print("VERDICT: effect stable / distance concentration acceptable")

if __name__ == "__main__":
    main()
