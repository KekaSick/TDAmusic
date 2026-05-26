"""
estimate_cost.py
----------------
Оценка времени выполнения и потребления памяти Ripser на полных облаках
(без прореживания). Тестирует самый длинный, самый короткий и 3 средних трека.
Выводит предупреждение при риске OOM.
"""
import os
import sys
import glob
import time
import yaml
import numpy as np
import ripser
import psutil

sys.path.insert(0, "src")
import pointcloud
import preprocess

CACHE_DIR = "cache/muq_spotify90"
REDUCER_PATH = "cache/pca_reducer_muq_spotify90.joblib"

def load_cfg():
    return yaml.safe_load(open("config.yaml"))

def format_bytes(b):
    return f"{b / 1024 / 1024:.1f} MB"

def main():
    cfg = load_cfg()
    pc_cfg = cfg["pointcloud"].copy()
    assert pc_cfg["subsample"] == "none", "subsample must be 'none'"
    
    print("Loading cached MuQ embeddings...")
    all_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npy")))
    if not all_files:
        print("ERROR: No cached embeddings found.")
        sys.exit(1)
        
    lengths = []
    for f in all_files:
        emb = np.load(f)
        lengths.append((f, emb.shape[0]))
        
    lengths.sort(key=lambda x: x[1])
    
    # Pick shortest, longest, and 3 mid
    shortest = lengths[0]
    longest = lengths[-1]
    n = len(lengths)
    mids = [lengths[n//4], lengths[n//2], lengths[3*n//4]]
    
    test_tracks = [shortest] + mids + [longest]
    
    print(f"Loaded reducer from {REDUCER_PATH}")
    reducer = preprocess.PCAReducer.load(REDUCER_PATH)
    
    print("\n--- RIPSER COST ESTIMATION (subsample=none) ---")
    print(f"{'Type':<10} | {'Frames':<8} | {'Cloud Shape':<15} | {'Time (s)':<10} | {'Peak Mem (MB)'}")
    print("-" * 65)
    
    total_time = 0
    worst_time = 0
    
    for label, (f, frames) in zip(["Shortest", "Mid 1", "Mid 2", "Mid 3", "Longest"], test_tracks):
        emb = np.load(f)
        
        # PCA
        x_pca = preprocess.prepare_track(emb, cfg["spaces"]["muq"], cfg["common"], reducer)
        
        # Cloud
        cloud, _, _ = pointcloud.build_cloud_with_indices(x_pca, pc_cfg)
        
        # Ripser
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        
        t0 = time.time()
        res = ripser.ripser(cloud, maxdim=1)
        t1 = time.time()
        
        mem_after = process.memory_info().rss
        peak_mem = max(0, mem_after - mem_before)
        
        dt = t1 - t0
        total_time += dt
        worst_time = max(worst_time, dt)
        
        print(f"{label:<10} | {frames:<8} | {str(cloud.shape):<15} | {dt:<10.2f} | {format_bytes(peak_mem)}")
        
    print("-" * 65)
    avg_time = total_time / len(test_tracks)
    
    n_tracks = len(all_files) # 199
    # Each track: 1 real + 20 shuffle + 1 random = 22 runs
    n_runs_per_track = 1 + cfg["controls"].get("shuffle_repeats", 20) + 1
    total_runs = n_tracks * n_runs_per_track
    
    est_total_hours = (total_runs * avg_time) / 3600
    est_worst_hours = (total_runs * worst_time) / 3600
    
    print(f"\nExtrapolation for {n_tracks} tracks x {n_runs_per_track} runs/track = {total_runs} total runs")
    print(f"Average-case total time: ~{est_total_hours:.1f} hours")
    print(f"Worst-case total time:   ~{est_worst_hours:.1f} hours")
    
if __name__ == "__main__":
    main()
