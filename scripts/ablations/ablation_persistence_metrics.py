import os
import sys
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from common import get_ablation_data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import embed_spaces
import preprocess
import pointcloud
import persistence
import controls
from scipy import stats

def compute_persistence_metrics(dgm):
    if dgm is None or len(dgm) == 0:
        return {"max": 0.0, "top3_mean": 0.0, "entropy": 0.0, "total": 0.0, "count_05": 0, "betti_area": 0.0}
        
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return {"max": 0.0, "top3_mean": 0.0, "entropy": 0.0, "total": 0.0, "count_05": 0, "betti_area": 0.0}
        
    lifetimes = finite[:, 1] - finite[:, 0]
    lifetimes = np.sort(lifetimes)[::-1]
    
    max_p = lifetimes[0]
    top3 = np.mean(lifetimes[:min(3, len(lifetimes))])
    total = np.sum(lifetimes)
    count_05 = np.sum(lifetimes > 0.05)
    
    # Entropy
    L = np.sum(lifetimes)
    if L > 0:
        p = lifetimes / L
        entropy = -np.sum(p * np.log(p + 1e-12))
    else:
        entropy = 0.0
        
    # Betti curve area approx placeholder (filled later)
    betti_area = 0.0
    
    return {
        "max": float(max_p),
        "top3_mean": float(top3),
        "entropy": float(entropy),
        "total": float(total),
        "count_05": int(count_05),
        "betti_area": float(betti_area)
    }

def compute_effect_size(real_vals, shuf_vals):
    d = np.array(real_vals) - np.array(shuf_vals)
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0
    r = stats.rankdata(np.abs(nz))
    return float((r[nz > 0].sum() - r[nz < 0].sum()) / (r[nz > 0].sum() + r[nz < 0].sum()))

def _process_file(i, f, space, cfg_local, reducer):
    x_raw = embed_spaces.extract(f, space, cfg_local)
    x_pca = preprocess.prepare_track(x_raw, cfg_local["spaces"][space], cfg_local["common"], reducer)
    cloud = pointcloud.build_cloud(x_pca, cfg_local["pointcloud"])
    
    if cloud is None or len(cloud) < 2:
        return None
        
    # Real
    try:
        res = persistence.compute_diagrams(cloud, cfg_local["persistence"]["maxdim"], cfg_local["persistence"]["persistence_threshold"])
        dgm_r = res["dgms"][1] if len(res["dgms"]) > 1 else np.empty((0,2))
    except Exception as e:
        print(f"Failed to compute real diagram for {f}: {e}")
        dgm_r = np.empty((0,2))
        
    real_metrics = compute_persistence_metrics(dgm_r)
    
    # Shuffle (mean of K repeats)
    shuf_metrics_k = {"max": [], "top3_mean": [], "entropy": [], "total": [], "count_05": []}
    shuf_dgms_k = []
    
    n_shuffle = cfg_local["controls"]["shuffle_repeats"]
    
    for k in range(n_shuffle):
        rng = np.random.default_rng(cfg_local["seed"] * 1000 + i * 100 + k)
        x_s = controls.shuffle_frames(x_pca, rng)
        cloud_s = pointcloud.build_cloud(x_s, cfg_local["pointcloud"])
        if cloud_s is None or len(cloud_s) < 2:
            continue
        try:
            res_s = persistence.compute_diagrams(cloud_s, cfg_local["persistence"]["maxdim"], cfg_local["persistence"]["persistence_threshold"])
            dgm_s = res_s["dgms"][1] if len(res_s["dgms"]) > 1 else np.empty((0,2))
        except Exception as e:
            dgm_s = np.empty((0,2))
        
        shuf_dgms_k.append(dgm_s)
        m = compute_persistence_metrics(dgm_s)
        for mk in shuf_metrics_k:
            shuf_metrics_k[mk].append(m[mk])
            
    avg_shuf = {mk: float(np.mean(shuf_metrics_k[mk])) if shuf_metrics_k[mk] else 0.0 for mk in shuf_metrics_k}
    avg_shuf["betti_area"] = 0.0
    
    return {
        "index": i,
        "dgm_real": dgm_r,
        "real_metrics": real_metrics,
        "shuf_dgms": shuf_dgms_k,
        "shuf_metrics": avg_shuf
    }

def main():
    cfg, files, labels, tr_idx, te_idx = get_ablation_data()
    spaces = ["muq", "mert"]
    
    results = []
    
    for space in tqdm(spaces, desc="Persistence Metrics ablation"):
        print(f"\nProcessing {space.upper()}...")
        
        cfg_local = copy.deepcopy(cfg)
        cfg_local["persistence"]["vectorization"] = "betti"
        
        reducer = preprocess.PCAReducer(
            cfg_local["common"]["pca_dim"],
            standardize=cfg_local["common"].get("standardize_features", True),
            normalize=cfg_local["common"]["normalize"],
            scaler_type=cfg_local["common"].get("scaler", "standard")
        )
        raw_tr = [embed_spaces.extract(files[i], space, cfg_local) for i in tr_idx]
        tr_frames = [
            preprocess.resample_fps(
                r,
                cfg_local["spaces"][space].get("native_fps", cfg_local["common"]["target_fps"]),
                cfg_local["common"]["target_fps"]
            ) for r in raw_tr
        ]
        reducer.fit(tr_frames)
        
        vectorizer = persistence.DiagramVectorizer()
        
        all_metrics_real = []
        all_metrics_shuf = []
        
        dgms_real = []
        dgms_shuf = []
        valid_indices = []
        
        n_shuffle = cfg_local["controls"]["shuffle_repeats"]
        
        results_gen = Parallel(n_jobs=-1, return_as="generator")(
            delayed(_process_file)(i, f, space, cfg_local, reducer)
            for i, f in enumerate(files)
        )
        file_results = [r for r in tqdm(results_gen, total=len(files), desc=f"Persistence Metrics {space}", leave=False) if r is not None]
        
        for r in file_results:
            dgms_real.append(r["dgm_real"])
            all_metrics_real.append(r["real_metrics"])
            valid_indices.append(r["index"])
            dgms_shuf.append(r["shuf_dgms"])
            all_metrics_shuf.append(r["shuf_metrics"])

            
        if not all_metrics_real:
            continue
            
        # Compute Betti area
        # Fit vectorizer only on train diagrams
        try:
            train_dgms = [d for d, orig_i in zip(dgms_real, valid_indices) if orig_i in set(tr_idx)]
            if len(train_dgms) > 0:
                vectorizer.fit(train_dgms, cfg_local["persistence"])
            else:
                vectorizer.fit(dgms_real, cfg_local["persistence"])
                
            betti_curves_r = [vectorizer.transform(d) for d in dgms_real]
            
            for i, m in enumerate(all_metrics_real):
                m["betti_area"] = float(np.sum(betti_curves_r[i]))
                
            for i, m in enumerate(all_metrics_shuf):
                shuf_areas = [float(np.sum(vectorizer.transform(d))) for d in dgms_shuf[i]]
                m["betti_area"] = float(np.mean(shuf_areas)) if shuf_areas else 0.0
        except Exception as e:
            print(f"Failed Betti computation for {space}: {e}")
            
        row = {"space": space}
        
        for k in all_metrics_real[0].keys():
            real_vals = [m[k] for m in all_metrics_real]
            shuf_vals = [m[k] for m in all_metrics_shuf]
            
            row[f"real_{k}_mean"] = float(np.mean(real_vals))
            row[f"shuf_{k}_mean"] = float(np.mean(shuf_vals))
            row[f"effect_{k}"] = compute_effect_size(real_vals, shuf_vals)
            
        results.append(row)
        
    df = pd.DataFrame(results)
    out_path = "results/tables/ablations/ablation_persistence_metrics.csv"
    df.to_csv(out_path, index=False)
    
    print("\n--- VERDICT ---")
    if len(df) == 0:
        print("VERDICT: Execution failed, no results.")
        return
        
    muq_row = df[df["space"]=="muq"].iloc[0]
    n_tracks = len(files)
    print(f"Sample: {n_tracks} tracks")
    
    # Check if effect size > 0.1 for most metrics for MuQ
    positive_effects = 0
    total_metrics = 0
    for col in df.columns:
        if col.startswith("effect_"):
            total_metrics += 1
            print(f"[muq] {col}: {muq_row[col]:.4f}")
            if muq_row[col] > 0.1:
                positive_effects += 1
                
    ratio = positive_effects / total_metrics if total_metrics > 0 else 0
    print(f"Positive effects (>0.1): {positive_effects} of {total_metrics} ({ratio:.2f})")
    
    print(f"\nWARNING: subset verdict n={n_tracks}; thresholds are heuristic. Use the numbers, not only the binary verdict.")
    if ratio > 0.8:
        print("VERDICT: effect stable / persistent topology robust across metrics")
    else:
        print("VERDICT: moderately sensitive / fragile metric dependence")

if __name__ == "__main__":
    main()
