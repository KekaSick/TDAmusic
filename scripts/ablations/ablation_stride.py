import os
import sys
import copy
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from common import get_ablation_data, extract_metrics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from run_controls import process_space

def _process_task(stride, space, cfg, files, labels, tr_idx):
    cfg_stride = copy.deepcopy(cfg)
    cfg_stride["pointcloud"]["stride"] = stride
    t0 = time.time()
    try:
        res = process_space(space, cfg_stride, files, labels, tr_idx, np.random.default_rng(42), compute_random=False, compute_iaaft=False)
        elapsed = time.time() - t0
        row = {
            "space": space, 
            "stride": stride, 
            "runtime_sec": elapsed
        }
        row.update(extract_metrics(res, space, cfg_stride, files))
        return row
    except Exception as e:
        print(f"Failed {space} stride {stride}: {e}")
        return None


def main():
    cfg, files, labels, tr_idx, te_idx = get_ablation_data()
    spaces = ["muq", "mert"]
    
    # window is fixed to 16 by default.
    window = cfg["pointcloud"]["window"]
    strides = [window // 2, window // 4, window // 8]
    strides = [max(1, s) for s in strides] # Ensure minimum stride of 1
    
    tasks = [(stride, space) for stride in strides for space in spaces]
    
    results_gen = Parallel(n_jobs=-1, return_as="generator")(
        delayed(_process_task)(stride, space, cfg, files, labels, tr_idx)
        for stride, space in tasks
    )
    results = [r for r in tqdm(results_gen, total=len(tasks), desc="Stride ablation") if r is not None]
            
    df = pd.DataFrame(results)
    out_path = "results/tables/ablations/ablation_stride.csv"
    df.to_csv(out_path, index=False)
    
    # Verdict
    print("\n--- VERDICT ---")
    n_tracks = len(files)
    print(f"Выборка: {n_tracks} треков")
    
    sensitive = False
    for space in spaces:
        effs = df[df["space"]==space]["max_persistence_real"].values
        diff = np.max(effs) - np.min(effs) if len(effs) > 0 else 0
        print(f"[{space.upper()}] Max_persistence spread: {diff:.4f} (min={np.min(effs):.4f}, max={np.max(effs):.4f})")
        if diff > 0.1:
            sensitive = True
            
    print(f"\nВНИМАНИЕ: вердикт на подвыборке n={n_tracks}, порог spread 0.1 эвристический; ориентируйся на числа, а не на binary-вердикт.")
    if sensitive:
        print("VERDICT: strongly configuration-dependent / sampling-density sensitive")
    else:
        print("VERDICT: effect stable / density robust")

if __name__ == "__main__":
    main()
