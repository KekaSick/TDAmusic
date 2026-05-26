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

def _process_task(dim, space, cfg, files, labels, tr_idx):
    cfg_dim = copy.deepcopy(cfg)
    cfg_dim["common"]["pca_dim"] = dim
    try:
        res = process_space(space, cfg_dim, files, labels, tr_idx, np.random.default_rng(42), compute_random=False, compute_iaaft=False)
        row = {"space": space, "pca_dim": dim, "explained_variance": res.get("pca_explained", 0.0)}
        row.update(extract_metrics(res, space, cfg_dim, files))
        return row
    except Exception as e:
        print(f"Failed {space} at dim {dim}: {e}")
        return None


def main():
    cfg, files, labels, tr_idx, te_idx = get_ablation_data()
    spaces = ["muq", "mert"]
    pca_dims = [16, 32, 64, 128]
    results = []
    
    tasks = [(dim, space) for dim in pca_dims for space in spaces]
    results_gen = Parallel(n_jobs=-1, return_as="generator")(
        delayed(_process_task)(dim, space, cfg, files, labels, tr_idx)
        for dim, space in tasks
    )
    results = [r for r in tqdm(results_gen, total=len(tasks), desc="PCA Dims ablation") if r is not None]
            
    df = pd.DataFrame(results)
    out_path = "results/tables/ablations/ablation_pca_dims.csv"
    df.to_csv(out_path, index=False)
    
    # Verdict
    print("\n--- VERDICT ---")
    
    if len(df) == 0:
        print("VERDICT: Execution failed, no results.")
        return
        
    try:
        n_tracks = len(files)
        print(f"Выборка: {n_tracks} треков")
        
        baseline_32_muq_rows = df[(df["pca_dim"]==32) & (df["space"]=="muq")]["real_vs_shuffle_effect"].values
        baseline_32_mert_rows = df[(df["pca_dim"]==32) & (df["space"]=="mert")]["real_vs_shuffle_effect"].values
        
        if len(baseline_32_muq_rows) > 0 and len(baseline_32_mert_rows) > 0:
            baseline_diff = baseline_32_muq_rows[0] - baseline_32_mert_rows[0]
        else:
            baseline_diff = 0.0
            
        rank_flip = False
        for dim in pca_dims:
            eff_muq_rows = df[(df["pca_dim"]==dim) & (df["space"]=="muq")]["real_vs_shuffle_effect"].values
            eff_mert_rows = df[(df["pca_dim"]==dim) & (df["space"]=="mert")]["real_vs_shuffle_effect"].values
            
            if len(eff_muq_rows) > 0 and len(eff_mert_rows) > 0:
                diff = eff_muq_rows[0] - eff_mert_rows[0]
                print(f"[PCA={dim}] muq_effect={eff_muq_rows[0]:.4f}, mert_effect={eff_mert_rows[0]:.4f}, diff={diff:.4f}")
                if diff < -0.05: # MuQ became worse than MERT by a margin
                    rank_flip = True
                    
        print(f"\nВНИМАНИЕ: вердикт на подвыборке n={n_tracks}, порог -0.05 эвристический; ориентируйся на числа, а не на binary-вердикт.")
        if rank_flip:
            print("VERDICT: moderately sensitive / results PCA-sensitive")
        else:
            print("VERDICT: effect stable / topology robust to dimensionality")
    except Exception as e:
        print(f"VERDICT: Error calculating verdict: {e}")

if __name__ == "__main__":
    main()
