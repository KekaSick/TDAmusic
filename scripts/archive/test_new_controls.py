"""
test_new_controls.py
--------------------
Тестирование новых метрик на подвыборке из 50 треков GTZAN (MuQ).
"""
import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
from scipy.stats import wilcoxon

sys.path.insert(0, "src")
import pointcloud
import preprocess
import persistence
import controls
import embed_spaces

def calc_tda_metrics(dgm):
    if len(dgm) == 0:
        return [0.0, 0.0, 0.0, 0, 0]
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return [0.0, 0.0, 0.0, 0, 0]
        
    lifetimes = finite[:, 1] - finite[:, 0]
    max_pers = lifetimes.max()
    total_pers = lifetimes.sum()
    h1_count = len(lifetimes)
    h1_sig = np.sum(lifetimes >= 0.05)
    
    L = lifetimes.sum()
    if L > 0:
        p = lifetimes / L
        entropy = -np.sum(p * np.log2(p))
    else:
        entropy = 0.0
        
    return [max_pers, total_pers, entropy, h1_count, h1_sig]

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    import pandas as pd
    meta = pd.read_csv(cfg["paths"]["meta"]).head(50)
    files = [os.path.join(cfg["paths"]["audio"], f) for f in meta["filename"]]
    
    space = "muq"
    print(f"Loading {len(files)} tracks for {space}...")
    
    raw = [embed_spaces.extract(f, space, cfg) for f in tqdm(files, desc="Load")]
    
    reducer = preprocess.PCAReducer(dim=cfg["common"]["pca_dim"])
    reducer.fit(raw, batch_size=20)
    
    preprocessed = [preprocess.prepare_track(x, cfg["spaces"][space], cfg["common"], reducer) for x in raw]
    
    rng = np.random.default_rng(42)
    pc_cfg = cfg["pointcloud"]
    pers_cfg = cfg["persistence"]
    block_size = cfg["controls"].get("block_shuffle_frames", 75)
    
    metrics = {"Real": [], "Block": [], "Full": [], "PhaseRand": []}
    
    for x_pca in tqdm(preprocessed, desc="Processing"):
        # Real
        d_real = persistence.compute_diagrams(pointcloud.build_cloud(x_pca, pc_cfg), pers_cfg["maxdim"], pers_cfg["persistence_threshold"])["dgms"][1]
        metrics["Real"].append(calc_tda_metrics(d_real))
        
        # Block
        x_block = controls.block_shuffle_frames(x_pca, rng, block_size)
        d_block = persistence.compute_diagrams(pointcloud.build_cloud(x_block, pc_cfg), pers_cfg["maxdim"], pers_cfg["persistence_threshold"])["dgms"][1]
        metrics["Block"].append(calc_tda_metrics(d_block))
        
        # Full
        x_shuf = controls.shuffle_frames(x_pca, rng)
        d_shuf = persistence.compute_diagrams(pointcloud.build_cloud(x_shuf, pc_cfg), pers_cfg["maxdim"], pers_cfg["persistence_threshold"])["dgms"][1]
        metrics["Full"].append(calc_tda_metrics(d_shuf))
        
        # Phase Rand
        x_rand = controls.random_like(x_pca, rng, match_autocorr=True)
        d_rand = persistence.compute_diagrams(pointcloud.build_cloud(x_rand, pc_cfg), pers_cfg["maxdim"], pers_cfg["persistence_threshold"])["dgms"][1]
        metrics["PhaseRand"].append(calc_tda_metrics(d_rand))
        
    print("\n=== AVERAGE METRICS (50 tracks) ===")
    metric_names = ["MaxPers", "TotPers", "Entropy", "H1Count", "SigH1"]
    
    means = {k: np.mean(v, axis=0) for k, v in metrics.items()}
    
    for k, v in means.items():
        print(f"[{k:^10}] " + " | ".join(f"{name}: {val:.3f}" for name, val in zip(metric_names, v)))
        
    print("\n=== WILCOXON P-VALUES (vs Real) ===")
    real_arr = np.array(metrics["Real"])
    for control in ["Block", "Full", "PhaseRand"]:
        ctrl_arr = np.array(metrics[control])
        p_vals = []
        for i in range(5):
            try:
                res = wilcoxon(real_arr[:, i], ctrl_arr[:, i])
                p_vals.append(res.pvalue)
            except ValueError:
                p_vals.append(1.0)
        
        print(f"[{control:^10}] " + " | ".join(f"{name}: {val:.3e}" for name, val in zip(metric_names, p_vals)))

if __name__ == "__main__":
    main()
