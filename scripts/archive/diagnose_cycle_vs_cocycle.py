import os
import sys
import yaml
import glob
import numpy as np
import pandas as pd
import dionysus as d
import ripser as ripser_lib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import preprocess
import pointcloud

CACHE_DIR = "cache/muq_spotify90"
SPOTIFY_DIR = "data/top50musicSpotify"

def get_both_representatives(X):
    # 1. Ripser cocycle
    r = ripser_lib.ripser(X, maxdim=1, do_cocycles=True)
    dgm1 = r["dgms"][1]
    if len(dgm1) == 0:
        return set(), set()
        
    life = dgm1[:, 1] - dgm1[:, 0]
    life[np.isinf(dgm1[:, 1])] = -1
    idx = int(np.argmax(life))
    if life[idx] < 0:
        return set(), set()
        
    r_death = float(dgm1[idx, 1])
    cc = r["cocycles"][1][idx]
    c_verts = set(np.unique(cc[:, :2].astype(int)))
    
    # 2. Dionysus cycle
    max_radius = r_death * 1.1
    f = d.fill_rips(X.astype(np.float64), 2, max_radius)
    m = d.homology_persistence(f, prime=2)
    dgms = d.init_diagrams(m, f)
    
    dgm1 = dgms[1]
    h1_ho = [pt for pt in dgm1 if pt.death < float("inf")]
    if not h1_ho:
        return c_verts, set()
        
    best_ho = max(h1_ho, key=lambda p: p.death - p.birth)
    death_idx = best_ho.data
    birth_idx = m.pair(death_idx)
    chain = m[birth_idx]
    
    cy_verts = set()
    for entry in chain:
        s = f[entry.index]
        if len(list(s)) == 2:
            cy_verts.update(list(s))
            
    return c_verts, cy_verts

def synthetic_test():
    print("=" * 70)
    print("1A. SYNTHETIC: Cocycle vs Cycle on a clean circle")
    print("=" * 70)
    
    print(f"{'n_points':>10} | {'Ripser Cocycle':>20} | {'Dionysus Cycle':>20}")
    print("-" * 55)
    
    for n_pts in [50, 100, 300]:
        t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        X = np.column_stack([np.cos(t), np.sin(t)])
        X += 0.03 * np.random.default_rng(42).standard_normal(X.shape)
        
        c_verts, cy_verts = get_both_representatives(X)
        n_c = len(c_verts)
        n_cy = len(cy_verts)
        
        c_frac = n_c / n_pts
        cy_frac = n_cy / n_pts
        
        print(f"{n_pts:>10} | {n_c:>4} ({c_frac:>5.0%})           | {n_cy:>4} ({cy_frac:>5.0%})")
    print()

def real_tracks_test():
    print("=" * 70)
    print("1B. REAL TRACKS: Cocycle vs Cycle")
    print("=" * 70)
    
    cfg = yaml.safe_load(open("config.yaml"))
    target_fps = cfg["common"]["target_fps"]
    
    all_tracks = []
    for genre in ["pop", "electronic", "hip-hop", "reggae"]:
        wavs = sorted(glob.glob(os.path.join(SPOTIFY_DIR, genre, "*.wav")))
        for w in wavs:
            all_tracks.append({"filepath": w, "genre": genre, "basename": os.path.basename(w)})
            
    # Load PCA
    raw_embs = []
    for t in all_tracks:
        base = os.path.splitext(t["basename"])[0]
        cache_path = os.path.join(CACHE_DIR, f"{base}.npy")
        if os.path.exists(cache_path):
            raw_embs.append(np.load(cache_path))
    reducer = preprocess.PCAReducer(
        dim=cfg["common"]["pca_dim"],
        standardize=cfg["common"].get("standardize_features", True),
        normalize=cfg["common"]["normalize"])
    reducer.fit(raw_embs)
    
    # Pick 5 tracks
    df = pd.read_csv("results/tables/loop_spotify.csv")
    sig_small = df[df["significant"]].nsmallest(1, "n_loop_vertices")
    sig_large = df[df["significant"]].nlargest(1, "n_loop_vertices")
    nonsig = df[~df["significant"] & df["p_value"].notna()].nlargest(1, "max_persistence")
    
    picks = pd.concat([sig_small, sig_large, nonsig])
    for g in ["pop", "hip-hop"]:
        g_tracks = df[(df["genre"] == g) & df["significant"]].head(1)
        picks = pd.concat([picks, g_tracks])
    picks = picks.drop_duplicates("basename").head(5)
    
    real_tracks = []
    for _, row in picks.iterrows():
        for t in all_tracks:
            if t["basename"] == row["basename"]:
                real_tracks.append(t)
                break
                
    pc_cfg = cfg["pointcloud"].copy()
    pc_cfg["takens_pca_dim"] = None
    cloud_size = pc_cfg["n_points"]
    window = pc_cfg["window"]
    
    print(f"{'Track':>45} | {'Cocycle verts (span)':>25} | {'Cycle verts (span)':>25}")
    print("-" * 105)
    
    res_df = []
    
    def get_span(verts, sub_indices, takens_starts):
        if not verts: return 0.0
        secs = []
        for v in verts:
            sec = (takens_starts[sub_indices[int(v)]] + window/2) / target_fps
            secs.append(sec)
        return max(secs) - min(secs) if len(secs) > 1 else 0.0
    
    for i, t in enumerate(real_tracks):
        base = t["basename"]
        cache_base = os.path.splitext(base)[0]
        emb = np.load(os.path.join(CACHE_DIR, f"{cache_base}.npy"))
        x_pca = preprocess.prepare_track(emb, cfg["spaces"]["muq"], cfg["common"], reducer)
        cloud, takens_starts, sub_indices = pointcloud.build_cloud_with_indices(x_pca, pc_cfg)
        
        c_verts, cy_verts = get_both_representatives(cloud)
        
        n_c = len(c_verts)
        n_cy = len(cy_verts)
        c_frac = n_c / cloud_size
        cy_frac = n_cy / cloud_size
        
        c_span = get_span(c_verts, sub_indices, takens_starts)
        cy_span = get_span(cy_verts, sub_indices, takens_starts)
        
        print(f"{base[-45:]:>45} | {n_c:>3} ({c_frac:>4.0%}) {c_span:>5.1f}s | {n_cy:>3} ({cy_frac:>4.0%}) {cy_span:>5.1f}s")
        
        res_df.append({
            "basename": base,
            "genre": t["genre"],
            "cocycle_verts": n_c,
            "cocycle_frac": c_frac,
            "cocycle_span": c_span,
            "cycle_verts": n_cy,
            "cycle_frac": cy_frac,
            "cycle_span": cy_span
        })
        
    df_out = pd.DataFrame(res_df)
    df_out.to_csv("results/tables/diagnose_cycle_vs_cocycle.csv", index=False)
    print(f"\nSaved detailed results to results/tables/diagnose_cycle_vs_cocycle.csv")

if __name__ == "__main__":
    synthetic_test()
    real_tracks_test()
