# Interpretable Music Embeddings for Genre Analysis

52-dimensional interpretable audio embedding grounded in European music theory for genre classification and embedding space analysis.

## Key Results

- **71.7% accuracy** on GTZAN (10 genres, 999 tracks) with Random Forest — surpassing the foundational result of Tzanetakis & Cook (2002) at 61.0%
- **Full interpretability**: every component has a precise music-theoretic meaning (diatonicity, Tonnetz position, sensory dissonance, tonal stability, etc.)
- **Topologically aware**: respects circular topology of pitch classes ($S^1$), toroidal structure of Tonnetz ($T^3$), simplex structure of chroma ($\Delta^{11}$)

## Feature Vector Architecture (52D, 9 blocks)

| Block | Features | Dim | Musical Domain |
|-------|----------|-----|----------------|
| 1 | Normalized chromagram | 12 | Pitch |
| 2 | Tonal DFT magnitudes | 5 | Pitch (transposition-invariant) |
| 3 | Tonal DFT phases | 5 | Pitch (key) |
| 4 | Tonnetz coordinates | 6 | Harmonic proximity |
| 5 | Harmonic tension | 3 | Dissonance, tonal stability |
| 6 | MFCC | 13 | Timbre |
| 7 | Spectral descriptors | 3 | Timbre |
| 8 | Dynamics | 2 | Loudness |
| 9 | Rhythmic features | 3 | Rhythm |

## Project Structure

```
3rdCourseWork/
├── src/                          # Python modules
│   ├── interpretable_embeddings.py  # 52D bar-level feature vector
│   ├── embedding_pipeline.py        # Track-level aggregation pipeline
│   ├── topology_methods.py          # CQT-chroma bar embeddings
│   ├── mir_bar_features.py          # MIR bar-level features
│   └── chaos_methods.py             # Chaos analysis methods
│
├── notebooks_tda/                # Jupyter notebooks
│   ├── vector_validation_gtzan.ipynb  # GTZAN validation (classification, importance, ablation)
│   ├── vector_validation.ipynb        # Feature sanity checks
│   ├── umap_analysis.ipynb            # UMAP + HDBSCAN embedding space analysis
│   ├── topology_mir_umap_visualization.ipynb
│   ├── topology_mir_visualization.ipynb
│   ├── topology_tda_pipeline.ipynb
│   ├── topology_visualization.ipynb
│   └── topology_popularity_analysis.ipynb
│
├── plots/                        # Generated figures (PNG)
│   ├── gtzan_confusion_matrix.png
│   ├── gtzan_feature_importance.png
│   ├── gtzan_block_importance.png
│   ├── gtzan_ablation_study.png
│   ├── gtzan_feature_distributions.png
│   ├── gtzan_correlation_matrix.png
│   ├── umap_visualization.png
│   ├── umap_feature_coloring.png
│   ├── umap_block_projection.png
│   └── genre_distance_matrix.png
│
├── latex/                        # LaTeX paper
│   ├── main.tex
│   ├── refs.bib
│   └── graphics/
│
├── data/                         # Audio data (not in git)
├── scripts/                      # Utility scripts
├── setup.py
└── requirements_scraper.txt
```

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Core dependencies:
```bash
pip install numpy scipy librosa soundfile matplotlib scikit-learn umap-learn hdbscan tqdm
```

## Usage

### Computing embeddings

```python
from src.interpretable_embeddings import compute_bar_embeddings
from src.embedding_pipeline import compute_track_embedding

# Bar-level: matrix (N_bars × 52)
bar_matrix = compute_bar_embeddings("path/to/audio.wav")

# Track-level: vector (52,) — mean across bars
track_vector = compute_track_embedding("path/to/audio.wav", strategy="mean")
```

### Running validation notebooks

```bash
jupyter notebook notebooks_tda/
```

Key notebooks:
- **`vector_validation_gtzan.ipynb`** — genre classification, feature/block importance, ablation study
- **`umap_analysis.ipynb`** — UMAP projection, HDBSCAN clustering, feature-colored UMAP, block projections, inter-genre distances

## References

- Amiot, E. (2016). *Music Through Fourier Space*. Springer.
- Fujishima, T. (1999). Realtime chord recognition. ICMC.
- Harte, C. & Sandler, M. (2006). Detecting harmonic change in musical audio. ACM MM.
- Krumhansl, C. (1990). *Cognitive Foundations of Musical Pitch*. Oxford.
- Plomp, R. & Levelt, W. (1965). Tonal consonance and critical bandwidth. JASA.
- Tzanetakis, G. & Cook, P. (2002). Musical genre classification of audio signals. IEEE TSAP.
