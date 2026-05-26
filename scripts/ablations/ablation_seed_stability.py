import os
import sys
import yaml
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from common import extract_metrics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from run_controls import process_space

def _process_task(seed, space):
    cfg, files, labels, tr_idx, te_idx = get_data_for_seed(seed)
    try:
        res = process_space(space, cfg, files, labels, tr_idx, np.random.default_rng(seed), compute_random=False, compute_iaaft=False)
        row = {"space": space, "seed": seed}
        row.update(extract_metrics(res, space, cfg, files))
        return row
    except Exception as e:
        print(f"Failed {space} at seed {seed}: {e}")
        return None


def get_data_for_seed(seed):
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    meta = pd.read_csv(cfg["paths"]["meta"])
    # 6-10 tracks per genre, keeping the *same* tracks but different splits?
    # "5 random seeds, 5 train/test splits."
    # Let's resample tracks too for maximum robustness check.
    sub_meta = meta.groupby(cfg["data"]["stratify_by"], group_keys=False).apply(
        lambda x: x.sample(n=min(10, len(x)), random_state=seed)
    ).reset_index(drop=True)
    
    files = [os.path.join(cfg["paths"]["audio"], f) for f in sub_meta["filename"]]
    labels = sub_meta[cfg["data"]["stratify_by"]].values
    
    idx = np.arange(len(files))
    tr_idx, te_idx = train_test_split(
        idx, test_size=cfg["data"]["test_size"], 
        stratify=labels, random_state=seed
    )
    
    return cfg, files, labels, tr_idx, te_idx

def main():
    spaces = ["muq", "mert"]
    seeds = [42, 123, 999, 2026, 7]
    
    tasks = [(seed, space) for seed in seeds for space in spaces]
    
    results_gen = Parallel(n_jobs=-1, return_as="generator")(
        delayed(_process_task)(seed, space)
        for seed, space in tasks
    )
    results = [r for r in tqdm(results_gen, total=len(tasks), desc="Seed ablation") if r is not None]
            
    df = pd.DataFrame(results)
    out_path = "results/tables/ablations/ablation_seed_stability.csv"
    df.to_csv(out_path, index=False)
    
    # Verdict
    print("\n--- SUMMARY STATS ---")
    variance_high = False
    
    cfg, files, _, _, _ = get_data_for_seed(seeds[0])
    n_tracks = len(files)
    print(f"Sample: {n_tracks} tracks per seed")
    
    for space in spaces:
        effs = df[df["space"]==space]["real_vs_shuffle_effect"].values
        m = np.mean(effs)
        s = np.std(effs)
        print(f"[{space.upper()}] Effect: {m:.4f} ± {s:.4f} (std={s:.4f})")
        if s > 0.15:
            variance_high = True
            
    print(f"\nWARNING: subset verdict n={n_tracks}; std threshold 0.15 is heuristic. Use the numbers, not only the binary verdict.")
    print("\n--- VERDICT ---")
    if variance_high:
        print("VERDICT: moderately sensitive / results split-sensitive")
    else:
        print("VERDICT: effect stable / stable across splits")

if __name__ == "__main__":
    main()
