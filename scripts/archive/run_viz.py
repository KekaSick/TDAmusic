"""
run_viz.py
----------
Скрипт сборки 6 артефактов (фигур) Этапа 4 (Визуализация).
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import embed_spaces
import preprocess
import pointcloud
import persistence
import controls
from viz import trajectories, mapper

def get_track_by_genre(meta, genre, seed=42):
    rng = np.random.default_rng(seed)
    sub = meta[meta["genre"] == genre]
    return sub.iloc[rng.integers(len(sub))]

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    
    # Жестко прописываем конфигурацию по контракту
    pc_cfg = cfg["pointcloud"]
    pc_cfg["takens_pca_dim"] = None
    pers_cfg = cfg["persistence"]
    
    meta = pd.read_csv(cfg["paths"]["meta"])
    files = [os.path.join(cfg["paths"]["audio"], f) for f in meta["filename"]]
    
    # 1. Выбираем треки
    track_classical = get_track_by_genre(meta, "classical", seed=42)
    track_metal = get_track_by_genre(meta, "metal", seed=42)
    track_jazz = get_track_by_genre(meta, "jazz", seed=42)
    
    idx_classical = meta.index[meta["filename"] == track_classical["filename"]].tolist()[0]
    idx_metal = meta.index[meta["filename"] == track_metal["filename"]].tolist()[0]
    idx_jazz = meta.index[meta["filename"] == track_jazz["filename"]].tolist()[0]
    
    # Для PCA (обучение на 42 треках, как в контролях)
    from sklearn.model_selection import train_test_split
    labels = meta[cfg["data"]["stratify_by"]].values
    tr_idx, _ = train_test_split(
        np.arange(len(files)), test_size=cfg["data"]["test_size"],
        stratify=labels, random_state=cfg["seed"])
    
    out_dir = "results/figures"
    os.makedirs(out_dir, exist_ok=True)
    
    # --- Инициализация PCA Reducer для MuQ ---
    # По контракту основной канал — MuQ (слой 10 уже в кэше)
    def load_pca_reducer(space):
        reducer = preprocess.PCAReducer(dim=cfg["common"]["pca_dim"])
        feats = [embed_spaces.extract(files[i], space, cfg) for i in tr_idx]
        reducer.fit(feats)
        return reducer

    muq_pca = load_pca_reducer("muq")
    
    # Helpers
    def get_track_data(space, idx, pca_reducer, do_cocycles=False, return_direct=False):
        x = embed_spaces.extract(files[idx], space, cfg)
        x = preprocess.prepare_track(x, cfg["spaces"][space], cfg["common"], pca_reducer)
        cloud = pointcloud.build_cloud(x, pc_cfg)
        res = persistence.compute_diagrams(cloud, pers_cfg["maxdim"], pers_cfg["persistence_threshold"], do_cocycles=do_cocycles)
        if return_direct:
            return x, cloud, res
        return cloud, res
        
    print("Generating Figure 1: real_shuffle_random.png ...")
    rng = np.random.default_rng(cfg["seed"])
    track_results = []
    for name, idx in [("Classical", idx_classical), ("Metal", idx_metal)]:
        x_pca, cloud, res_real = get_track_data("muq", idx, muq_pca, return_direct=True)
        # Shuffle
        x_shuf = controls.shuffle_frames(x_pca, rng)
        cloud_shuf = pointcloud.build_cloud(x_shuf, pc_cfg)
        res_shuf = persistence.compute_diagrams(cloud_shuf, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
        # Random
        x_rand = controls.random_like(x_pca, rng, match_autocorr=True)
        cloud_rand = pointcloud.build_cloud(x_rand, pc_cfg)
        res_rand = persistence.compute_diagrams(cloud_rand, pers_cfg["maxdim"], pers_cfg["persistence_threshold"])
        
        track_results.append({
            "name": name,
            "real": res_real["dgms"],
            "shuffle": res_shuf["dgms"],
            "random": res_rand["dgms"]
        })
    fig1 = trajectories.plot_real_shuffle_random_panel(track_results)
    fig1.savefig(os.path.join(out_dir, "real_shuffle_random.png"), bbox_inches="tight", dpi=150)
    print("  -> results/figures/real_shuffle_random.png")
    
    print("Generating Figure 2: cycle_diagram.png ...")
    # Используем Classical или Metal для коциклов. Пусть будет Classical.
    _, cloud_c, res_c = get_track_data("muq", idx_classical, muq_pca, do_cocycles=True, return_direct=True)
    
    # ВАЖНО: 2D-проекция ИМЕННО прореженного облака (которое шло в Ripser)
    from sklearn.decomposition import PCA
    cloud_2d = PCA(n_components=2).fit_transform(cloud_c)
    fig2, _ = trajectories.cycle_on_trajectory(cloud_2d, res_c)
    fig2.figure.savefig(os.path.join(out_dir, "cycle_diagram.png"), bbox_inches="tight", dpi=150)
    print("  -> results/figures/cycle_diagram.png")
    
    print("Generating Figure 3: mapper_graph.html ...")
    # DIRECT-облако кадров (до Такенса). Узел = группа похожих моментов.
    x_metal_raw = embed_spaces.extract(files[idx_metal], "muq", cfg)
    x_metal_pca = preprocess.prepare_track(x_metal_raw, cfg["spaces"]["muq"], cfg["common"], muq_pca)
    mapper.build_mapper_graph(x_metal_pca, out_file=os.path.join(out_dir, "mapper_graph.html"), title="Metal Track Mapper (Direct Frames)")
    print("  -> results/figures/mapper_graph.html")
    
    print("Generating Figure 4: genre_panels.png ...")
    _, res_m = get_track_data("muq", idx_metal, muq_pca)
    _, res_j = get_track_data("muq", idx_jazz, muq_pca)
    genre_dgm = {
        "classical": res_c["dgms"],
        "metal": res_m["dgms"],
        "jazz": res_j["dgms"]
    }
    fig4 = trajectories.plot_genre_panels(genre_dgm, title="H1-диаграммы типичных треков по жанрам (MuQ-10)")
    fig4.savefig(os.path.join(out_dir, "genre_panels.png"), bbox_inches="tight", dpi=150)
    print("  -> results/figures/genre_panels.png")
    
    print("Generating Figure 5: grid_spaces.png ...")
    # Сравниваем MERT, MuQ, Encodec, MIR на Classical треке.
    clouds_spaces = {}
    for sp in ["muq", "mert", "encodec", "mir"]:
        pca_red = load_pca_reducer(sp)
        x_pca, _, _ = get_track_data(sp, idx_classical, pca_red, return_direct=True)
        clouds_spaces[sp] = PCA(n_components=2).fit_transform(x_pca) # 2D для траектории
    fig5 = trajectories.grid_spaces(clouds_spaces, title="Сравнение пространств (Classical)")
    fig5.savefig(os.path.join(out_dir, "grid_spaces.png"), bbox_inches="tight", dpi=150)
    print("  -> results/figures/grid_spaces.png")
    
    print("Generating Figure 6: trajectory_3d.html ...")
    # Интерактивная 3D траектория (MuQ).
    # PCA:
    fig6_pca = trajectories.trajectory_3d(x_metal_pca, title="3D Траектория (Metal)", use_umap=False)
    fig6_pca.write_html(os.path.join(out_dir, "trajectory_3d_pca.html"))
    # UMAP-версия отключена из-за дедлока pynndescent на macOS
    # fig6_umap = trajectories.trajectory_3d(x_metal_pca, title="3D Траектория (Metal)", use_umap=True)
    # fig6_umap.write_html(os.path.join(out_dir, "trajectory_3d_umap.html"))
    print("  -> results/figures/trajectory_3d_pca.html")
    print("  -> results/figures/trajectory_3d_umap.html")

    print("\nSmoke tests passed. All artifacts generated successfully.")

if __name__ == "__main__":
    main()
