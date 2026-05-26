# Music-TDA

Topological analysis of frame-level music audio embeddings.

The project tests whether trajectories of audio embeddings contain robust
topological structure, and where that structure is useful or misleading for
music analysis.

Pipeline:

```text
audio
-> frame-level embeddings (MERT / MuQ / EnCodec / MIR)
-> preprocessing (resample, StandardScaler, L2 normalization, PCA)
-> Takens point cloud
-> persistent homology H0/H1
-> controls, distances, classification, cycle interpretation
```

## Main Findings

- Real embedding trajectories contain robust H1 loops: real tracks exceed
  shuffle, phase-randomized, and IAAFT surrogate controls in all four
  representation spaces.
- Topology is only weakly genre-specific. Within-genre diagram distances are
  smaller than between-genre distances, but the effect is small.
- Topological features are not practically useful for genre classification:
  best topology macro-F1 is near chance, while mean-pooled embeddings are much
  stronger.
- Cross-model topology is not universal. MERT/MuQ have weak agreement; EnCodec
  is largely different.
- H-loop interpretation is partial: after correcting cocycle/cycle and
  statistical testing issues, about 27% of tested Spotify90 tracks show
  chromatic recurrence along the representative cycle.

## Repository Layout

```text
src/                         Core reusable modules
scripts/                     Final experiment entrypoints
scripts/ablations/           Robustness checks
scripts/archive/             Old diagnostics and exploratory scripts
latex/                       Coursework source and compiled PDF
results/tables/main_results/ Final tables used by the paper figures
results/tables/ablations/    Final robustness tables
results/figures/paper/       Final paper figures
config.yaml                  Main experiment configuration
```

Large local inputs and generated caches are intentionally ignored by git:
`data/`, `cache/`, `.venv/`, and archived old result tables.

## Data Expected

The configuration expects:

- GTZAN audio under `data/GTZAN`
- labels under `data/meta/labels.csv`
- optional Spotify90 audio/metadata for the H-loop and popularity branches

Embeddings are cached under `cache/`. The cache is not committed because it is
large and machine/generated.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some embedding extraction steps require model downloads and are much faster on
GPU. After embeddings are cached, most analysis scripts run from cached arrays.

## Reproducing The Main Tables

Run from the repository root.

Extract frame-level embeddings first:

```bash
python src/embed_spaces.py
```

Then run the main analysis scripts:

```bash
python scripts/run_controls.py
python scripts/run_classify_betti.py
python scripts/run_loop_spotify.py
python scripts/run_loop_spotify_mir.py
python scripts/run_cycle_analysis.py
python scripts/analyze_popularity_topology.py
python scripts/collect_pvalues.py
```

Run robustness checks:

```bash
python scripts/ablations/ablation_seed_stability.py
python scripts/ablations/ablation_pca_dims.py
python scripts/ablations/ablation_normalization.py
python scripts/ablations/ablation_stride.py
python scripts/ablations/ablation_window_scales.py
python scripts/ablations/ablation_resampling.py
python scripts/ablations/ablation_distance_concentration.py
python scripts/ablations/ablation_persistence_metrics.py
```

Regenerate paper figures:

```bash
MPLBACKEND=Agg python scripts/generate_paper_figures.py
```

Compile the report:

```bash
cd latex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

## Important Methodological Rules

- Persistent homology is computed on PCA-reduced frame trajectories, not on
  UMAP or plotting coordinates.
- Shuffle controls are applied to the frame sequence before Takens embedding.
  Shuffling an already-built point cloud is not a valid temporal-order control.
- Phase and IAAFT controls are needed because shuffle only tests frame order;
  it does not rule out smooth linear-spectral explanations.
- Preprocessing parameters are fitted on the training split and then reused, to
  avoid data leakage.
- For H-loop interpretation, cocycles are not treated as geometric loop
  contours. The final analysis uses Dionysus cycle representatives.
- Report effect sizes and confidence intervals alongside p-values; large sample
  sizes can make tiny effects statistically significant.

## Tests And Sanity Checks

```bash
python smoke_test.py
python test_iaaft.py
python scripts/verify_determinism.py
```

## Notes

`run_pipeline.py` is not the current paper pipeline. The final reproducible
entrypoints are the scripts listed above. Older exploratory code was moved to
`scripts/archive/` to keep the active workflow readable.
