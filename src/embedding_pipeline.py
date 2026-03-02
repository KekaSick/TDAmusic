"""
Полный pipeline: от папки с аудиофайлами до UMAP/HDBSCAN визуализации.

Использует interpretable_embeddings для вычисления 52D bar-level
и 523D track-level эмбеддингов.

Этапы:
  1. Вычисление эмбеддингов для каждого трека
  2. StandardScaler нормализация
  3. UMAP проекция в 2D
  4. HDBSCAN кластеризация
  5. Визуализация (жанры + кластеры)
  6. Метрики (ARI, NMI)
  7. Сохранение в HDF5
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from .interpretable_embeddings import (
    compute_bar_embeddings,
    full_track_embedding,
)

logger = logging.getLogger(__name__)


def full_pipeline(
    data_dir: str,
    genre_labels: Dict[str, str],
    *,
    sr: int = 22050,
    beats_per_bar: int = 4,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    random_state: int = 42,
    output_plot: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Полный pipeline от папки с аудиофайлами до UMAP + HDBSCAN.

    Parameters
    ----------
    data_dir : str
        Путь к папке с аудиофайлами (WAV, MP3, FLAC).
    genre_labels : dict
        {filename: genre_string} — жанровые метки.
    sr : int
        Sample rate для загрузки.
    beats_per_bar : int
        Долей в такте.
    n_neighbors : int
        Параметр UMAP (локальность).
    min_dist : float
        Параметр UMAP (плотность кластеров).
    min_cluster_size : int
        Параметр HDBSCAN.
    min_samples : int
        Параметр HDBSCAN.
    random_state : int
        Seed для воспроизводимости.
    output_plot : str, optional
        Путь для сохранения графика (PNG).
    verbose : bool
        Печатать ли прогресс.

    Returns
    -------
    dict с ключами:
        'X'              : np.ndarray (n_tracks, 523)
        'X_scaled'       : np.ndarray (n_tracks, 523)
        'X_2d'           : np.ndarray (n_tracks, 2)
        'cluster_labels' : np.ndarray (n_tracks,)
        'genres'         : np.ndarray (n_tracks,)
        'filenames'      : list[str]
        'bar_data'       : dict {filename: (n_bars, 52)}
        'ari'            : float  (если есть валидные кластеры)
        'nmi'            : float  (если есть валидные кластеры)
    """
    import umap
    import hdbscan

    # ═══ ЭТАП 1: Вычисление эмбеддингов ═══
    if verbose:
        print("Computing embeddings...")

    all_tracks = []
    all_genres = []
    all_filenames = []
    all_bar_data = {}

    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    audio_files = sorted(
        p for p in Path(data_dir).iterdir()
        if p.suffix.lower() in audio_extensions
    )

    for wav_path in audio_files:
        filename = wav_path.name
        if filename not in genre_labels:
            continue

        try:
            bar_emb = compute_bar_embeddings(
                str(wav_path), sr=sr, beats_per_bar=beats_per_bar,
            )
            if len(bar_emb) < 4:
                logger.warning(f"Skipping '{filename}': too few bars ({len(bar_emb)})")
                continue

            track_vec = full_track_embedding(bar_emb)

            all_tracks.append(track_vec)
            all_genres.append(genre_labels[filename])
            all_filenames.append(filename)
            all_bar_data[filename] = bar_emb

        except Exception as e:
            logger.exception(f"Error processing {filename}: {e}")
            continue

    if len(all_tracks) == 0:
        raise ValueError(
            f"No tracks processed from '{data_dir}'. "
            "Check that audio files match genre_labels keys."
        )

    X = np.array(all_tracks, dtype=np.float32)
    genres = np.array(all_genres)

    if verbose:
        print(f"Computed embeddings for {len(X)} tracks, dim={X.shape[1]}")

    # ═══ ЭТАП 2: Нормализация ═══
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # ═══ ЭТАП 3: UMAP ═══
    if verbose:
        print("Running UMAP...")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='euclidean',
        random_state=random_state,
    )
    X_2d = reducer.fit_transform(X_scaled)

    # ═══ ЭТАП 4: HDBSCAN ═══
    if verbose:
        print("Running HDBSCAN...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
    )
    cluster_labels = clusterer.fit_predict(X_scaled)

    # ═══ ЭТАП 5: Визуализация ═══
    if output_plot is not None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        unique_genres = np.unique(genres)
        colors_genre = plt.cm.tab20(np.linspace(0, 1, len(unique_genres)))
        genre_to_color = {
            g: colors_genre[i] for i, g in enumerate(unique_genres)
        }

        # Жанры (ground truth)
        ax = axes[0]
        for genre in unique_genres:
            mask = genres == genre
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=[genre_to_color[genre]],
                label=genre, s=15, alpha=0.7,
            )
        ax.set_title('UMAP colored by Genre (ground truth)')
        ax.legend(fontsize=8, markerscale=2)

        # Кластеры (HDBSCAN)
        ax = axes[1]
        scatter = ax.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=cluster_labels, cmap='tab20', s=15, alpha=0.7,
        )
        ax.set_title('UMAP colored by HDBSCAN clusters')
        plt.colorbar(scatter, ax=ax)

        plt.tight_layout()
        plt.savefig(output_plot, dpi=150, bbox_inches='tight')
        if verbose:
            print(f"Plot saved to {output_plot}")
        plt.close(fig)

    # ═══ ЭТАП 6: Метрики ═══
    result: Dict[str, Any] = {
        'X': X,
        'X_scaled': X_scaled,
        'X_2d': X_2d,
        'cluster_labels': cluster_labels,
        'genres': genres,
        'filenames': all_filenames,
        'bar_data': all_bar_data,
    }

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    valid = cluster_labels >= 0
    if valid.sum() > 0:
        ari = adjusted_rand_score(genres[valid], cluster_labels[valid])
        nmi = normalized_mutual_info_score(genres[valid], cluster_labels[valid])
        result['ari'] = ari
        result['nmi'] = nmi
        if verbose:
            print(f"ARI: {ari:.3f}, NMI: {nmi:.3f}")
    else:
        if verbose:
            print("No valid clusters found (all noise).")

    return result


def save_all(
    output_path: str,
    tracks_data: Dict[str, Dict[str, Any]],
) -> None:
    """
    Сохраняет все эмбеддинги в HDF5.

    Parameters
    ----------
    output_path : str
        Путь к выходному HDF5 файлу.
    tracks_data : dict
        {filename: {
            'bar_embeddings': np.ndarray (n_bars, 52),
            'track_embedding': np.ndarray (523,),
            'genre': str,
            'popularity': float (optional),
        }}
    """
    import h5py

    with h5py.File(output_path, 'w') as f:
        for filename, data in tracks_data.items():
            grp = f.create_group(filename)

            grp.create_dataset(
                'bar_embeddings', data=data['bar_embeddings'],
                compression='gzip',
            )
            grp.create_dataset(
                'track_embedding', data=data['track_embedding'],
                compression='gzip',
            )

            if 'genre' in data:
                grp.attrs['genre'] = data['genre']
            if 'popularity' in data:
                grp.attrs['popularity'] = data['popularity']
            grp.attrs['n_bars'] = data['bar_embeddings'].shape[0]

    logger.info(f"Saved {len(tracks_data)} tracks to '{output_path}'")
