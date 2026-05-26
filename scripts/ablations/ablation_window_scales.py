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
from run_controls import process_space

def _process_task(scale, space, variant, cfg, files, labels, tr_idx, fps):
    window = max(1, int(round(scale * fps)))
    stride = max(1, int(round(window / 4)))
    
    cfg_scale = copy.deepcopy(cfg)
    cfg_scale["pointcloud"]["window"] = window
    cfg_scale["pointcloud"]["stride"] = stride
    if variant == "fixed_takens_dim":
        cfg_scale["common"]["pca_dim"] = max(1, 512 // window)
        
    try:
        res = process_space(space, cfg_scale, files, labels, tr_idx, np.random.default_rng(42), compute_random=False, compute_iaaft=False)
        row = {"space": space, "timescale_sec": scale, "window": window, "stride": stride, "variant": variant}
        row.update(extract_metrics(res, space, cfg_scale, files))
        return row
    except Exception as e:
        print(f"Failed {variant} {space} at scale {scale}: {e}")
        return None


def main():
    cfg, files, labels, tr_idx, te_idx = get_ablation_data()
    spaces = ["muq", "mert"]
    scales_sec = [0.32, 0.64, 1.28, 2.56]
    fps = cfg["common"]["target_fps"] # 25
    
    tasks = [(scale, space, variant) for scale in scales_sec for space in spaces for variant in ["standard", "fixed_takens_dim"]]
    results_gen = Parallel(n_jobs=-1, return_as="generator")(
        delayed(_process_task)(scale, space, variant, cfg, files, labels, tr_idx, fps)
        for scale, space, variant in tasks
    )
    results = [r for r in tqdm(results_gen, total=len(tasks), desc="Window Scales ablation") if r is not None]
            
    df = pd.DataFrame(results)
    out_path = "results/tables/ablations/ablation_window_scales.csv"
    df.to_csv(out_path, index=False)
    
    print("\n--- VERDICT ---")
    n_tracks = len(files)
    print(f"Sample: {n_tracks} tracks")
    
    if len(df) == 0:
        print("VERDICT: Execution failed, no results.")
        return
        
    for variant in ["standard", "fixed_takens_dim"]:
        df_var = df[df["variant"] == variant]
        if len(df_var) == 0:
            continue
            
        print(f"\nVariant: {variant}")
        try:
            baseline_muq = df_var[(df_var["timescale_sec"]==0.64) & (df_var["space"]=="muq")]["real_vs_shuffle_effect"].values[0]
            baseline_mert = df_var[(df_var["timescale_sec"]==0.64) & (df_var["space"]=="mert")]["real_vs_shuffle_effect"].values[0]
            
            rank_flip = False
            for scale in scales_sec:
                eff_muq_rows = df_var[(df_var["timescale_sec"]==scale) & (df_var["space"]=="muq")]["real_vs_shuffle_effect"].values
                eff_mert_rows = df_var[(df_var["timescale_sec"]==scale) & (df_var["space"]=="mert")]["real_vs_shuffle_effect"].values
                
                if len(eff_muq_rows) > 0 and len(eff_mert_rows) > 0:
                    diff = eff_muq_rows[0] - eff_mert_rows[0]
                    print(f"[scale={scale}s] muq_effect={eff_muq_rows[0]:.4f}, mert_effect={eff_mert_rows[0]:.4f}, diff={diff:.4f}")
                    if diff < -0.05:
                        rank_flip = True
                        
            print(f"\nWARNING: subset verdict n={n_tracks}; diff threshold -0.05 is heuristic. Use the numbers, not only the binary verdict.")
            if rank_flip:
                print("VERDICT: moderately sensitive / results timescale-sensitive")
            else:
                print("VERDICT: effect stable / multiscale robust")
        except Exception as e:
            print(f"VERDICT: Error calculating verdict: {e}")

if __name__ == "__main__":
    main()
