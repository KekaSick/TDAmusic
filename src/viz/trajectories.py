"""
viz/trajectories.py
-------------------
Визуализация. ЖЕЛЕЗНОЕ ПРАВИЛО: гомологии считаются на PCA, UMAP — ТОЛЬКО
для картинки и НИКОГДА не подаётся в Ripser.

  * trajectory_2d / _3d  : траектория, цвет = время.
  * cycle_on_trajectory  : подсветить representative cycle самого
                           персистентного H1-класса поверх траектории +
                           показать соответствующую точку на диаграмме.
                           (связка «петля в музыке <-> точка на диаграмме»)
  * grid_spaces          : одна песня в MERT/MuQ/Encodec/MIR бок о бок.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def trajectory_2d(cloud_pca, ax=None, title="", time_indices=None):
    """cloud_pca: (N, >=2). Цвет = индекс времени."""
    ax = ax or plt.gca()
    if time_indices is None:
        t = np.arange(len(cloud_pca))
    else:
        t = time_indices
    ax.plot(cloud_pca[:, 0], cloud_pca[:, 1], "-", lw=0.6, alpha=0.4, color="gray")
    sc = ax.scatter(cloud_pca[:, 0], cloud_pca[:, 1], c=t, cmap="viridis", s=12)
    ax.set_title(title)
    return ax


def trajectory_3d(cloud, title="", use_umap=False, time_indices=None):
    """Interactive 3D plot using plotly. 
    cloud: input (N, D).
    If use_umap=True, projects to 3D with UMAP and adds disclaimer.
    Otherwise projects to 3D with PCA."""
    import plotly.graph_objects as go
    
    if use_umap:
        import umap
        proj = umap.UMAP(n_components=3, random_state=42, n_jobs=1).fit_transform(cloud)
        full_title = f"{title}<br><sup>(проекция UMAP, искажающая, для наглядности; гомологии на ней НЕ считались)</sup>"
    else:
        from sklearn.decomposition import PCA
        proj = PCA(n_components=3).fit_transform(cloud) if cloud.shape[1] > 3 else cloud
        full_title = f"{title}<br><sup>(PCA, метрически корректная приближённо)</sup>"

    if time_indices is None:
        t = np.arange(len(proj))
    else:
        t = time_indices
        
    fig = go.Figure(go.Scatter3d(
        x=proj[:, 0], y=proj[:, 1], z=proj[:, 2],
        mode="lines+markers",
        marker=dict(size=3, color=t, colorscale="Viridis"),
        line=dict(width=2, color="gray")))
    fig.update_layout(title=full_title)
    return fig


def cycle_on_trajectory(cloud_pca, ripser_result, ax_traj=None, ax_dgm=None):
    """
    Подсветить рёбра representative cocycle самого долгоживущего H1-класса.
    cloud_pca: 2D-проекция ИМЕННО ТОГО прореженного облака, что подавалось в Ripser.
               Это гарантирует совпадение индексов рёбер и точек.
    ripser_result: требует compute_diagrams(..., do_cocycles=True).
    """
    if ax_traj is None or ax_dgm is None:
        fig, (ax_traj, ax_dgm) = plt.subplots(1, 2, figsize=(10, 5))
        
    trajectory_2d(cloud_pca, ax=ax_traj, title="Takens Cloud (2D PCA) + H1 Cycle")
    
    dgm1 = ripser_result["dgms"][1]
    dgm0 = ripser_result["dgms"][0]
    
    from persim import plot_diagrams
    plot_diagrams([dgm0, dgm1], ax=ax_dgm, show=False)
    ax_dgm.set_title("Persistence Diagram")
    
    if len(dgm1) == 0:
        return ax_traj, ax_dgm
        
    life = dgm1[:, 1] - dgm1[:, 0]
    k = int(np.argmax(life))                 # самый персистентный класс
    
    # Отмечаем точку красной звездой на диаграмме
    b, d = dgm1[k]
    ax_dgm.plot(b, d, 'r*', markersize=15, markeredgecolor='black')
    
    # Рисуем рёбра поверх траектории
    cocycle = ripser_result["cocycles"][1][k]   # рёбра [i, j, val]
    for i, j, val in cocycle:
        ax_traj.plot([cloud_pca[i, 0], cloud_pca[j, 0]],
                     [cloud_pca[i, 1], cloud_pca[j, 1]], "r-", lw=1.5, zorder=5)
                     
    return ax_traj, ax_dgm


def grid_spaces(clouds_by_space: dict, title=""):
    """{'mert': cloud, 'muq': cloud, ...} -> сетка 2D-траекторий."""
    n = len(clouds_by_space)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, cloud) in zip(axes, clouds_by_space.items()):
        trajectory_2d(cloud, ax=ax, title=name)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_real_shuffle_random_panel(track_results):
    """
    Панель H1-диаграмм: строки = треки, столбцы = Real / Shuffle / Random.
    ОДИНАКОВЫЕ ПРЕДЕЛЫ ОСЕЙ для всей строки, чтобы схлопнутость случайных была видна.
    track_results: list of dicts, each with 'name', 'real', 'shuffle', 'random' diagrams.
    """
    from persim import plot_diagrams
    n_tracks = len(track_results)
    fig, axes = plt.subplots(n_tracks, 3, figsize=(12, 4 * n_tracks))
    if n_tracks == 1:
        axes = [axes]
        
    for i, res in enumerate(track_results):
        d_real, d_shuf, d_rand = res["real"], res["shuffle"], res["random"]
        
        # Найти общий масштаб для H1
        all_pts = []
        if len(d_real[1]) > 0: all_pts.append(d_real[1])
        if len(d_shuf[1]) > 0: all_pts.append(d_shuf[1])
        if len(d_rand[1]) > 0: all_pts.append(d_rand[1])
        
        max_val = 1.0
        if all_pts:
            pts = np.vstack(all_pts)
            valid = pts[np.isfinite(pts)]
            if len(valid) > 0:
                max_val = np.max(valid) * 1.1
                
        for j, (dgm, ttl) in enumerate(zip([d_real, d_shuf, d_rand], 
                                           ["Real", "Shuffle", "Random"])):
            ax = axes[i][j]
            plot_diagrams(dgm, ax=ax, show=False)
            if i == 0:
                ax.set_title(ttl, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{res['name']}\nDeath", fontweight="bold")
            # Фиксируем оси!
            ax.set_xlim([0, max_val])
            ax.set_ylim([0, max_val])
            
    fig.tight_layout()
    return fig


def plot_genre_panels(genre_diagrams, title=""):
    """
    По одному треку classical/metal/jazz.
    genre_diagrams: dict genre -> diagrams.
    """
    from persim import plot_diagrams
    n = len(genre_diagrams)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1: axes = [axes]
    
    for ax, (genre, dgm) in zip(axes, genre_diagrams.items()):
        plot_diagrams(dgm, ax=ax, show=False)
        ax.set_title(genre.capitalize())
        
    fig.suptitle(title)
    fig.tight_layout()
    return fig
