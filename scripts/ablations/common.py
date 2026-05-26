import os
import sys
import yaml
import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from joblib import Parallel, delayed

# Ensure src and scripts are in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import embed_spaces
import preprocess
import pointcloud

def get_ablation_data():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    meta = pd.read_csv(cfg["paths"]["meta"])
    # Stratified sampling: max 10 per genre, random_state=42
    sub_meta = meta.groupby(cfg["data"]["stratify_by"], group_keys=False).apply(
        lambda x: x.sample(n=min(10, len(x)), random_state=42)
    ).reset_index(drop=True)
    
    files = [os.path.join(cfg["paths"]["audio"], f) for f in sub_meta["filename"]]
    labels = sub_meta[cfg["data"]["stratify_by"]].values
    
    idx = np.arange(len(files))
    tr_idx, te_idx = train_test_split(
        idx, test_size=cfg["data"]["test_size"], 
        stratify=labels, random_state=42
    )
    
    return cfg, files, labels, tr_idx, te_idx

_CLOUD_STATS_CACHE = {}

def _process_stat(f, space, cfg, tgt_fps, pc_cfg, pca_dim, native_fps):
    x_raw = embed_spaces.extract(f, space, cfg)
    t_tgt = max(1, int(round(x_raw.shape[0] * tgt_fps / native_fps)))
    
    if pc_cfg["method"] == "takens":
        n_before = max(0, (t_tgt - pc_cfg["window"]) // pc_cfg["stride"] + 1)
        takens_dim = pca_dim * pc_cfg["window"]
    else:
        n_before = t_tgt
        takens_dim = pca_dim
        
    n_after = min(n_before, pc_cfg["n_points"]) if pc_cfg["subsample"] != "none" else n_before
    return n_before, n_after, takens_dim

def extract_cloud_stats(space, cfg, files):
    """
    Extract dimensional and size stats for point clouds across all files.
    Calculates mean, std, min, max without heavy PCA fitting.
    """
    if len(files) == 0:
        return {}
        
    tgt_fps = cfg["common"]["target_fps"]
    pca_dim = cfg["common"]["pca_dim"]
    pc_cfg = cfg["pointcloud"]
    
    key = (
        space,
        tgt_fps,
        pca_dim,
        pc_cfg["method"],
        pc_cfg["window"],
        pc_cfg["stride"],
        pc_cfg["n_points"],
        pc_cfg["subsample"],
    )
    
    if key in _CLOUD_STATS_CACHE:
        return _CLOUD_STATS_CACHE[key]
        
    try:
        native_fps = cfg["spaces"][space].get("native_fps", tgt_fps)
        
        results_gen = Parallel(n_jobs=-1, return_as="generator")(
            delayed(_process_stat)(f, space, cfg, tgt_fps, pc_cfg, pca_dim, native_fps)
            for f in files
        )
        file_results = [r for r in tqdm(results_gen, total=len(files), desc="Extracting cloud stats", leave=False) if r is not None]
        
        if not file_results:
            return {}
            
        n_before_list = [r[0] for r in file_results]
        n_after_list = [r[1] for r in file_results]
        takens_dim = file_results[0][2]
            
        stats = {
            "takens_dim": takens_dim,
            "n_points_before_mean": float(np.mean(n_before_list)),
            "n_points_before_std": float(np.std(n_before_list)),
            "n_points_before_min": int(np.min(n_before_list)),
            "n_points_before_max": int(np.max(n_before_list)),
            "n_points_after_mean": float(np.mean(n_after_list)),
            "subsample_method": pc_cfg["subsample"]
        }
        
        _CLOUD_STATS_CACHE[key] = stats
        return stats
    except Exception as e:
        print(f"Warning: Failed to extract cloud stats: {e}")
        return {}

def extract_metrics(res, space=None, cfg=None, files=None):
    metrics = {
        "gap": res.get("gap"),
        "within": res.get("within_mean"),
        "between": res.get("between_mean"),
        "max_persistence_real": res.get("max_pers_real_mean"),
        "real_vs_shuffle_effect": res.get("max_pers_real_vs_shuffle_effect"),
        "real_vs_random_effect": res.get("max_pers_real_vs_random_effect"),
        "bootstrap_gap_obs": res.get("gap_obs_track"),
        "bootstrap_gap_ci": f"[{res.get('gap_ci95_track')[0]:.4f}, {res.get('gap_ci95_track')[1]:.4f}]" if "gap_ci95_track" in res else None,
    }
    
    if space and cfg and files:
        metrics.update(extract_cloud_stats(space, cfg, files))
        
    return metrics
