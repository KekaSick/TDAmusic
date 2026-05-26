"""
run_viz_pca20.py
----------------
Эксперимент: так ли полезен takens_pca_dim=20 для отдаления H1-точек от диагонали?
Генерирует только 2 артефакта из 6, но на PCA20:
- genre_panels_pca20.png
- real_shuffle_random_pca20.png

Также выводит max-persistence H1 (baseline vs pca20).
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import embed_spaces
import preprocess
import pointcloud
import persistence
import controls
from viz import trajectories

def get_track_by_genre(meta, genre, seed=42):
    rng = np.random.default_rng(seed)
    sub = meta[meta["genre"] == genre]
    return sub.iloc[rng.integers(len(sub))]

def max_pers(dgm):
    if len(dgm) == 0: return 0.0
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0: return 0.0
    return float((finite[:, 1] - finite[:, 0]).max())

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    
    pc_cfg_base = dict(cfg["pointcloud"])
    pc_cfg_base["takens_pca_dim"] = None
    
    pc_cfg_pca20 = dict(cfg["pointcloud"])
    pc_cfg_pca20["takens_pca_dim"] = 20
    
    pers_cfg = cfg["persistence"]
    meta = pd.read_csv(cfg["paths"]["meta"])
    files = [os.path.join(cfg["paths"]["audio"], f) for f in meta["filename"]]
    
    track_classical = get_track_by_genre(meta, "classical", seed=42)
    track_metal = get_track_by_genre(meta, "metal", seed=42)
    track_jazz = get_track_by_genre(meta, "jazz", seed=42)
    
    idx_classical = meta.index[meta["filename"] == track_classical["filename"]].tolist()[0]
    idx_metal = meta.index[meta["filename"] == track_metal["filename"]].tolist()[0]
    idx_jazz = meta.index[meta["filename"] == track_jazz["filename"]].tolist()[0]
    
    from sklearn.model_selection import train_test_split
    labels = meta[cfg["data"]["stratify_by"]].values
    tr_idx, _ = train_test_split(
        np.arange(len(files)), test_size=cfg["data"]["test_size"],
        stratify=labels, random_state=cfg["seed"])
    
    out_dir = "results/figures"
    
    reducer = preprocess.PCAReducer(dim=cfg["common"]["pca_dim"])
    feats = [embed_spaces.extract(files[i], "muq", cfg) for i in tr_idx]
    reducer.fit(feats)
    
    def get_cloud_res(idx, pc_cfg, do_cocycles=False):
        x = embed_spaces.extract(files[idx], "muq", cfg)
        x = preprocess.prepare_track(x, cfg["spaces"]["muq"], cfg["common"], reducer)
        cloud = pointcloud.build_cloud(x, pc_cfg)
        
        # Ручное применение PCA после Такенса, если задано (логика из pointcloud)
        if pc_cfg.get("takens_pca_dim") is not None and cloud.shape[1] > pc_cfg["takens_pca_dim"]:
            cloud = PCA(n_components=pc_cfg["takens_pca_dim"]).fit_transform(cloud)
            
        res = persistence.compute_diagrams(cloud, pers_cfg["maxdim"], pers_cfg["persistence_threshold"], do_cocycles=do_cocycles)
        return x, cloud, res

    print("=== Max-Persistence Comparison (Baseline vs PCA20) ===")
    
    # 1. Real / Shuffle / Random Panel
    print("\nGenerating Figure 1: real_shuffle_random_pca20.png ...")
    rng = np.random.default_rng(cfg["seed"])
    track_results = []
    
    for name, idx in [("Classical", idx_classical), ("Metal", idx_metal)]:
        # --- Baseline для вывода в консоль ---
        _, _, res_base = get_cloud_res(idx, pc_cfg_base)
        
        # --- PCA20 ---
        x, cloud_pca20, res_real = get_cloud_res(idx, pc_cfg_pca20)
        
        print(f"  {name:10s} H1 MaxPers | baseline: {max_pers(res_base['dgms'][1]):.4f} | pca20: {max_pers(res_real['dgms'][1]):.4f}")
        
        # Shuffle PCA20
        x_shuf = controls.shuffle_frames(x, rng)
        cloud_shuf = pointcloud.build_cloud(x_shuf, pc_cfg_pca20)
        if pc_cfg_pca20["takens_pca_dim"] and cloud_shuf.shape[1] > pc_cfg_pca20["takens_pca_dim"]:
            cloud_shuf = PCA(n_components=pc_cfg_pca20["takens_pca_dim"]).fit_transform(cloud_shuf)
        res_shuf = persistence.compute_diagrams(cloud_shuf, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
        
        # Random PCA20
        x_rand = controls.random_like(x, rng, match_autocorr=True)
        cloud_rand = pointcloud.build_cloud(x_rand, pc_cfg_pca20)
        if pc_cfg_pca20["takens_pca_dim"] and cloud_rand.shape[1] > pc_cfg_pca20["takens_pca_dim"]:
            cloud_rand = PCA(n_components=pc_cfg_pca20["takens_pca_dim"]).fit_transform(cloud_rand)
        res_rand = persistence.compute_diagrams(cloud_rand, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
        
        track_results.append({
            "name": name,
            "real": res_real["dgms"],
            "shuffle": res_shuf["dgms"],
            "random": res_rand["dgms"]
        })
        
    fig1 = trajectories.plot_real_shuffle_random_panel(track_results)
    fig1.savefig(os.path.join(out_dir, "real_shuffle_random_pca20.png"), bbox_inches="tight", dpi=150)
    print("  -> results/figures/real_shuffle_random_pca20.png")
    
    # 2. Genre Panels PCA20
    print("\nGenerating Figure 4: genre_panels_pca20.png ...")
    genre_dgm = {}
    for name, idx in [("classical", idx_classical), ("metal", idx_metal), ("jazz", idx_jazz)]:
        _, _, res_base = get_cloud_res(idx, pc_cfg_base)
        _, _, res_pca20 = get_cloud_res(idx, pc_cfg_pca20)
        genre_dgm[name] = res_pca20["dgms"]
        if name == "jazz":
             print(f"  {name.capitalize():10s} H1 MaxPers | baseline: {max_pers(res_base['dgms'][1]):.4f} | pca20: {max_pers(res_pca20['dgms'][1]):.4f}")
             
    fig4 = trajectories.plot_genre_panels(genre_dgm, title="H1-диаграммы по жанрам (MuQ-10, Takens PCA=20)")
    fig4.savefig(os.path.join(out_dir, "genre_panels_pca20.png"), bbox_inches="tight", dpi=150)
    print("  -> results/figures/genre_panels_pca20.png")
    
    print("\nDone.")

if __name__ == "__main__":
    main()
