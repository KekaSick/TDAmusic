import os
import sys
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from common import get_ablation_data, extract_metrics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import preprocess
from run_controls import process_space

def _process_task(space, variant, cfg, files, labels, tr_idx):
    cfg_var = copy.deepcopy(cfg)
    if variant == "A_StandardScaler_Only":
        cfg_var["common"]["standardize_features"] = True
        cfg_var["common"]["normalize"] = "none"
        cfg_var["common"]["scaler"] = "standard"
    elif variant == "B_StandardScaler_L2":
        cfg_var["common"]["standardize_features"] = True
        cfg_var["common"]["normalize"] = "unit_sphere"
        cfg_var["common"]["scaler"] = "standard"
    elif variant == "C_RobustScaler_L2":
        cfg_var["common"]["standardize_features"] = True
        cfg_var["common"]["normalize"] = "unit_sphere"
        cfg_var["common"]["scaler"] = "robust"
    elif variant == "D_No_Normalization":
        cfg_var["common"]["standardize_features"] = False
        cfg_var["common"]["normalize"] = "none"
    
    try:
        res = process_space(space, cfg_var, files, labels, tr_idx, np.random.default_rng(42), compute_random=False, compute_iaaft=False)
        row = {"space": space, "variant": variant}
        row.update(extract_metrics(res))
        return row
    except Exception as e:
        print(f"Failed {variant} for {space}: {e}")
        return None


def main():
    cfg, files, labels, tr_idx, te_idx = get_ablation_data()
    spaces = ["muq", "mert"]
    results = []
    
    variants = [
        "A_StandardScaler_Only", 
        "B_StandardScaler_L2", 
        "C_RobustScaler_L2", 
        "D_No_Normalization"
    ]
    tasks = [(space, variant) for space in spaces for variant in variants]
    
    results_gen = Parallel(n_jobs=-1, return_as="generator")(
        delayed(_process_task)(space, variant, cfg, files, labels, tr_idx)
        for space, variant in tasks
    )
    results = [r for r in tqdm(results_gen, total=len(tasks), desc="Normalization ablation") if r is not None]


            
    df = pd.DataFrame(results)
    out_path = "results/tables/ablations/ablation_normalization.csv"
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
        variants = df["variant"].unique()
        for variant in variants:
            eff_muq_rows = df[(df["variant"]==variant) & (df["space"]=="muq")]["real_vs_shuffle_effect"].values
            eff_mert_rows = df[(df["variant"]==variant) & (df["space"]=="mert")]["real_vs_shuffle_effect"].values
            if len(eff_muq_rows) > 0 and len(eff_mert_rows) > 0:
                print(f"[{variant}] muq_effect={eff_muq_rows[0]:.4f}, mert_effect={eff_mert_rows[0]:.4f}, diff={eff_muq_rows[0]-eff_mert_rows[0]:.4f}")
                if eff_muq_rows[0] - eff_mert_rows[0] < -0.05:
                    rank_flip = True
                    
        print(f"\nWARNING: subset verdict n={n_tracks}; -0.05 threshold is heuristic. Use the numbers, not only the binary verdict.")
        if rank_flip:
            print("VERDICT: moderately sensitive / results normalization-sensitive")
        else:
            print("VERDICT: effect stable / topology not driven by normalization")
    except Exception as e:
        print(f"VERDICT: Error calculating verdict: {e}")

if __name__ == "__main__":
    main()
