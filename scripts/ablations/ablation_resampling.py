import os
import sys
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from common import get_ablation_data, extract_metrics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import preprocess
from run_controls import process_space

def _process_task(space, variant, cfg, files, labels, tr_idx, original_resample):
    if variant == "A_baseline":
        cfg_var = copy.deepcopy(cfg)
        cfg_var["common"]["target_fps"] = 25
        cfg_var["pointcloud"]["window"] = 16
        cfg_var["pointcloud"]["stride"] = 4
        preprocess.resample_fps = original_resample
    elif variant == "B_no_resample":
        cfg_var = copy.deepcopy(cfg)
        cfg_var["common"]["target_fps"] = 75
        cfg_var["pointcloud"]["window"] = 48
        cfg_var["pointcloud"]["stride"] = 12
        preprocess.resample_fps = original_resample
    elif variant == "C_linear_resample":
        cfg_var = copy.deepcopy(cfg)
        cfg_var["common"]["target_fps"] = 25
        cfg_var["pointcloud"]["window"] = 16
        cfg_var["pointcloud"]["stride"] = 4
        preprocess.resample_fps = linear_resample_fps
        
    try:
        res = process_space(space, cfg_var, files, labels, tr_idx, np.random.default_rng(42), compute_random=False, compute_iaaft=False)
        row = {"space": space, "variant": variant}
        row.update(extract_metrics(res, space, cfg_var, files))
        return row
    except Exception as e:
        print(f"Failed {variant} for {space}: {e}")
        return None


def linear_resample_fps(x: np.ndarray, src_fps: float, tgt_fps: float) -> np.ndarray:
    if abs(src_fps - tgt_fps) < 1e-6:
        return x
    t_tgt = max(1, int(round(x.shape[0] * tgt_fps / src_fps)))
    src_times = np.linspace(0, 1, x.shape[0])
    tgt_times = np.linspace(0, 1, t_tgt)
    res = np.zeros((t_tgt, x.shape[1]), dtype=x.dtype)
    for j in range(x.shape[1]):
        res[:, j] = np.interp(tgt_times, src_times, x[:, j])
    return res

def main():
    cfg, files, labels, tr_idx, te_idx = get_ablation_data()
    print("\n--- ABLATION 1: RESAMPLING CONFOUND ---")
    
    # Save original functions
    original_resample = preprocess.resample_fps
    
    # We will compute 3 variants for each space
    # A: Baseline (target 25fps)
    # B: No resample (target 75fps, window 48, stride 12)
    # C: Linear resample (target 25fps using custom interpolation)
    
    spaces = ["mert", "encodec"]
    results = []
    
    variants = ["A_baseline", "B_no_resample", "C_linear_resample"]
    tasks = [(space, variant) for space in spaces for variant in variants]
    
    try:
        results_gen = Parallel(n_jobs=-1, return_as="generator")(
            delayed(_process_task)(space, variant, cfg, files, labels, tr_idx, original_resample)
            for space, variant in tasks
        )
        results = [r for r in tqdm(results_gen, total=len(tasks), desc="Resampling ablation") if r is not None]
    finally:
        preprocess.resample_fps = original_resample
        
    df = pd.DataFrame(results)
    out_path = "results/tables/ablations/ablation_resampling.csv"
    df.to_csv(out_path, index=False)
    
    # Verdict
    print("\n--- VERDICT ---")
    if len(df) == 0:
        print("VERDICT: Execution failed, no results.")
        return
        
    try:
        n_tracks = len(files)
        print(f"Sample: {n_tracks} tracks")
        
        rank_flip = False
        for space in spaces:
            eff_muq_a_rows = df[(df["variant"]=="A_baseline") & (df["space"]==space)]["real_vs_shuffle_effect"].values
            eff_muq_b_rows = df[(df["variant"]=="B_no_resample") & (df["space"]==space)]["real_vs_shuffle_effect"].values
            
            if len(eff_muq_a_rows) > 0 and len(eff_muq_b_rows) > 0:
                diff = abs(eff_muq_a_rows[0] - eff_muq_b_rows[0])
                print(f"[{space}] baseline={eff_muq_a_rows[0]:.4f}, no_resample={eff_muq_b_rows[0]:.4f}, abs_diff={diff:.4f}")
                if diff > 0.1:
                    rank_flip = True
                    
        print(f"\nWARNING: subset verdict n={n_tracks}; threshold 0.1 is heuristic. Use the numbers, not only the binary verdict.")
        if rank_flip:
            print("VERDICT: moderately sensitive / resampling confound likely")
        else:
            print("VERDICT: effect stable / multiscale robust")
    except Exception as e:
        print(f"VERDICT: Error calculating verdict: {e}")
        
if __name__ == "__main__":
    main()
