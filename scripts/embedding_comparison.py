#!/usr/bin/env python3
"""
Сравнение 5 вариантов track-level эмбеддингов на GTZAN.

Варианты:
  1. Full 523D          — full_track_embedding
  2. Compact 211D       — compact_track_embedding
  3. Compact+5MFCC 179D — compact, но MFCC[0:4] вместо [0:12]
  4. SelectKBest 150D   — mutual information feature selection на 523D
  5. Mean-only 52D      — bar_embeddings.mean(axis=0)

Для каждого: 5-fold CV accuracy.
Для лучшего: ablation study + confusion matrix.

Usage:
    python scripts/embedding_comparison.py
"""

import sys
import os
import time
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import confusion_matrix, classification_report

# ── project imports ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.interpretable_embeddings import (
    compute_bar_embeddings,
    full_track_embedding,
    compact_track_embedding,
    feature_names_full,
    feature_names_compact,
    feature_names,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

GTZAN_DIR = Path(__file__).resolve().parent.parent / "data" / "genres_30sec"
RANDOM_STATE = 42
N_ESTIMATORS = 200
N_SPLITS = 5


# ═══════════════════════════════════════════════════════════════
# 1. Compute bar embeddings for all tracks
# ═══════════════════════════════════════════════════════════════

def load_gtzan_bar_embeddings(data_dir: Path, max_per_genre: int = 100):
    """Load all GTZAN tracks and compute bar-level embeddings."""
    all_bar_data = []   # list of (n_bars, 52) arrays
    labels = []
    filenames = []

    genres = sorted([d.name for d in data_dir.iterdir()
                     if d.is_dir() and not d.name.startswith('.')])
    print(f"Found {len(genres)} genres: {genres}")

    for genre in genres:
        genre_dir = data_dir / genre
        audio_files = sorted(genre_dir.glob("*.*"))[:max_per_genre]
        count = 0
        for af in audio_files:
            if af.suffix.lower() not in ('.wav', '.mp3', '.flac', '.au', '.ogg'):
                continue
            try:
                bar_emb = compute_bar_embeddings(str(af))
                if bar_emb.shape[0] == 0:
                    continue
                all_bar_data.append(bar_emb)
                labels.append(genre)
                filenames.append(af.name)
                count += 1
            except Exception as e:
                print(f"  ⚠ Error processing {af.name}: {e}")
        print(f"  {genre}: {count} tracks")

    print(f"\nTotal: {len(all_bar_data)} tracks")
    return all_bar_data, labels, filenames


# ═══════════════════════════════════════════════════════════════
# 2. Build embedding variants
# ═══════════════════════════════════════════════════════════════

# MFCC indices in the 52D bar vector: [31:44] (13 MFCCs)
# We keep first 5 (indices 31:36), drop 8 (indices 36:44)
MFCC_BAR_SLICE = slice(31, 44)      # all 13 MFCC in bar vector
MFCC_KEEP = slice(31, 36)           # first 5 MFCC
MFCC_DROP = list(range(36, 44))     # indices of MFCC[5:12] to drop

# Indices of all 52 features minus MFCC[5:12] → 44 features
KEEP_44_INDICES = [i for i in range(52) if i not in MFCC_DROP]


def compact_5mfcc_embedding(bar_embeddings: np.ndarray) -> np.ndarray:
    """
    Compact embedding but with only 5 MFCC instead of 13.
    (n_bars, 52) → slice to 44D bar-level → compact → 179D
    """
    bar_44 = bar_embeddings[:, KEEP_44_INDICES]  # (n_bars, 44)
    n_bars = len(bar_44)
    parts = []

    parts.append(bar_44.mean(axis=0))                                   # 44
    parts.append(bar_44.std(axis=0))                                    # 44
    parts.append(
        np.percentile(bar_44, 75, axis=0)
        - np.percentile(bar_44, 25, axis=0)
    )                                                                    # 44

    if n_bars > 1:
        deltas = np.diff(bar_44, axis=0)
        parts.append(deltas.std(axis=0))                                 # 44
    else:
        parts.append(np.zeros(44, dtype=np.float32))

    # Meta (3D) — use original bar_embeddings for indices
    parts.append(np.array([
        float(n_bars),
        float(bar_embeddings[:, 47].mean()),
        float(bar_embeddings[:, 15].std()),
    ], dtype=np.float32))

    result = np.concatenate(parts).astype(np.float32)
    assert result.shape == (179,), f"Expected (179,), got {result.shape}"
    return result


def feature_names_compact_5mfcc():
    """Feature names for compact+5MFCC variant (179D)."""
    base_52 = feature_names()
    base_44 = [base_52[i] for i in KEEP_44_INDICES]
    names = []
    for prefix in ('mean', 'std', 'iqr'):
        names.extend(f'{prefix}_{n}' for n in base_44)
    names.extend(f'delta_std_{n}' for n in base_44)
    names.extend(['n_bars', 'mean_energy', 'diatonicity_variability'])
    assert len(names) == 179, f"Expected 179, got {len(names)}"
    return names


def build_all_variants(all_bar_data):
    """Build 5 embedding variants from bar-level data."""
    n = len(all_bar_data)
    variants = {}

    # 1. Full 523D
    print("Building Full 523D...")
    X_full = np.array([full_track_embedding(b) for b in all_bar_data])
    variants['Full 523D'] = X_full

    # 2. Compact 211D
    print("Building Compact 211D...")
    X_compact = np.array([compact_track_embedding(b) for b in all_bar_data])
    variants['Compact 211D'] = X_compact

    # 3. Compact + 5 MFCC 179D
    print("Building Compact+5MFCC 179D...")
    X_5mfcc = np.array([compact_5mfcc_embedding(b) for b in all_bar_data])
    variants['Compact+5MFCC 179D'] = X_5mfcc

    # 4. Feature selection (computed later, needs labels)
    # Placeholder — filled in evaluate_all()

    # 5. Mean-only 52D
    print("Building Mean-only 52D...")
    X_mean = np.array([b.mean(axis=0) for b in all_bar_data])
    variants['Mean-only 52D'] = X_mean

    for name, X in variants.items():
        print(f"  {name}: shape {X.shape}")

    return variants


# ═══════════════════════════════════════════════════════════════
# 3. Evaluate all variants
# ═══════════════════════════════════════════════════════════════

def evaluate_all(variants, labels):
    """Run 5-fold CV on all variants."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    results = {}

    for name, X in variants.items():
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        scores = cross_val_score(
            RandomForestClassifier(N_ESTIMATORS, random_state=RANDOM_STATE),
            X_scaled, y,
            cv=StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
            scoring='accuracy',
        )
        mean_acc = scores.mean()
        std_acc = scores.std()
        results[name] = (mean_acc, std_acc)
        print(f"  {name:30s}  {mean_acc:.3f} ± {std_acc:.3f}")

    # 4. Feature Selection 150D — applied to Full 523D
    print("\nRunning feature selection (mutual information, top-150)...")
    X_full = variants['Full 523D']
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)

    selector = SelectKBest(mutual_info_classif, k=150)
    X_selected = selector.fit_transform(X_full_scaled, y)

    scores = cross_val_score(
        RandomForestClassifier(N_ESTIMATORS, random_state=RANDOM_STATE),
        X_selected, y,
        cv=StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        scoring='accuracy',
    )
    mean_acc = scores.mean()
    std_acc = scores.std()
    results['SelectKBest 150D'] = (mean_acc, std_acc)
    print(f"  {'SelectKBest 150D':30s}  {mean_acc:.3f} ± {std_acc:.3f}")

    # Print selected features by block
    full_names = feature_names_full()
    selected_mask = selector.get_support()
    selected_names = [n for n, s in zip(full_names, selected_mask) if s]
    print(f"\n  Selected feature distribution by block:")
    for block in ['chroma', 'dft_mag', 'dft_phase', 'tonnetz',
                  'mfcc', 'spectral', 'onset', 'beat',
                  'dissonance', 'tonal_stability', 'log_rms']:
        count = sum(1 for n in selected_names if block in n)
        if count > 0:
            print(f"    {block:25s}: {count}")

    return results, y, le, selector


# ═══════════════════════════════════════════════════════════════
# 4. Ablation study on best variant
# ═══════════════════════════════════════════════════════════════

# Block definitions for 52D bar-level features
BAR_BLOCKS = {
    'Chroma':       list(range(0, 12)),
    'DFT mag':      list(range(12, 17)),
    'DFT phase':    list(range(17, 22)),
    'Tonnetz':      list(range(22, 28)),
    'Harm.Tension': list(range(28, 31)),
    'MFCC':         list(range(31, 44)),
    'Spectral':     list(range(44, 47)),
    'Dynamics':     list(range(47, 49)),
    'Rhythm':       list(range(49, 52)),
}


def get_track_indices_for_bar_block(bar_indices, aggregation='compact'):
    """
    Map bar-level feature indices to track-level indices
    for a given aggregation scheme.
    """
    if aggregation == 'compact':
        # compact: mean(52) + std(52) + iqr(52) + delta_std(52) + meta(3)
        track_indices = []
        for agg_offset in range(4):  # 4 aggregation groups
            for bi in bar_indices:
                track_indices.append(agg_offset * 52 + bi)
        return track_indices
    elif aggregation == 'full':
        # full: mean(52)+std(52)+median(52)+iqr(52) + sec1-4(52*4) + delta_mean(52)+delta_std(52) + meta(3)
        track_indices = []
        for agg_offset in range(10):  # 10 aggregation groups
            for bi in bar_indices:
                track_indices.append(agg_offset * 52 + bi)
        return track_indices
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def ablation_study(X, y, aggregation='compact', n_splits=5):
    """Remove each block and measure accuracy change."""
    n_features = X.shape[1]

    # Baseline
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    baseline_scores = cross_val_score(
        RandomForestClassifier(N_ESTIMATORS, random_state=RANDOM_STATE),
        X_scaled, y,
        cv=StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE),
        scoring='accuracy',
    )
    baseline = baseline_scores.mean()
    print(f"\n  Baseline ({n_features}D): {baseline:.3f}")

    results = {}
    for block_name, bar_indices in BAR_BLOCKS.items():
        track_indices = get_track_indices_for_bar_block(bar_indices, aggregation)
        # Keep meta features (last 3)
        keep = [i for i in range(n_features)
                if i not in track_indices]

        X_ablated = X[:, keep]
        scaler_ab = StandardScaler()
        X_ab_scaled = scaler_ab.fit_transform(X_ablated)

        scores = cross_val_score(
            RandomForestClassifier(N_ESTIMATORS, random_state=RANDOM_STATE),
            X_ab_scaled, y,
            cv=StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE),
            scoring='accuracy',
        )
        acc = scores.mean()
        drop = baseline - acc  # positive = block helps, negative = block hurts
        results[block_name] = (acc, drop)

    # Sort by drop (most helpful first)
    print(f"\n  {'Block':20s} {'w/o block':>10s} {'Drop':>8s}  Interpretation")
    print("  " + "─" * 60)
    for name, (acc, drop) in sorted(results.items(), key=lambda x: x[1][1], reverse=True):
        if drop > 0.005:
            interp = "✅ Helps"
        elif drop < -0.005:
            interp = "⚠️  HURTS (remove it!)"
        else:
            interp = "~  Neutral"
        print(f"  {name:20s} {acc:10.3f} {drop:+8.3f}  {interp}")

    return baseline, results


# ═══════════════════════════════════════════════════════════════
# 5. Confusion matrix
# ═══════════════════════════════════════════════════════════════

def print_confusion_matrix(X, y, le):
    """Train/test split confusion matrix."""
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y,
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(N_ESTIMATORS, random_state=RANDOM_STATE)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    acc = (y_pred == y_test).mean()
    print(f"\n  Test accuracy: {acc:.3f}")

    labels_sorted = le.classes_
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    print(f"  {'':12s}", end='')
    for g in labels_sorted:
        print(f"{g[:5]:>6s}", end='')
    print()
    for i, g in enumerate(labels_sorted):
        print(f"  {g:12s}", end='')
        for j in range(len(labels_sorted)):
            val = cm[i, j]
            if i == j:
                print(f"{val:6d}", end='')
            elif val > 0:
                print(f"  [{val:2d}]", end='')
            else:
                print(f"{'·':>6s}", end='')
        print()

    # Top confusions
    print(f"\n  Top confused pairs:")
    confusions = []
    for i in range(len(labels_sorted)):
        for j in range(len(labels_sorted)):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], labels_sorted[i], labels_sorted[j]))
    confusions.sort(reverse=True)
    for count, true_g, pred_g in confusions[:10]:
        print(f"    {true_g:10s} → {pred_g:10s} : {count}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 65)
    print("  EMBEDDING COMPARISON: GTZAN Genre Classification")
    print("=" * 65)

    # 1. Compute bar embeddings
    print("\n[1/5] Computing bar-level embeddings...")
    all_bar_data, labels, filenames = load_gtzan_bar_embeddings(GTZAN_DIR)

    # 2. Build variants
    print("\n[2/5] Building embedding variants...")
    variants = build_all_variants(all_bar_data)

    # 3. Cross-validation
    print("\n[3/5] Cross-validation comparison:")
    results, y, le, selector = evaluate_all(variants, labels)

    # Summary table
    print("\n" + "=" * 50)
    print("  RESULTS SUMMARY")
    print("=" * 50)
    for name, (acc, std) in sorted(results.items(), key=lambda x: -x[1][0]):
        marker = " ← BEST" if acc == max(v[0] for v in results.values()) else ""
        print(f"  {name:30s}  {acc:.3f} ± {std:.3f}{marker}")

    # Find best variant
    best_name = max(results, key=lambda k: results[k][0])
    print(f"\n  Best variant: {best_name}")

    # 4. Ablation on best variant
    print(f"\n[4/5] Ablation study on {best_name}:")
    if best_name == 'Full 523D':
        X_best = variants['Full 523D']
        agg = 'full'
    elif best_name == 'Compact 211D':
        X_best = variants['Compact 211D']
        agg = 'compact'
    elif best_name == 'Compact+5MFCC 179D':
        # ablation doesn't map cleanly to 44D, use compact instead
        X_best = variants['Compact 211D']
        agg = 'compact'
        print("  (Using Compact 211D for ablation since block mapping is cleaner)")
    elif best_name == 'Mean-only 52D':
        X_best = variants['Mean-only 52D']
        agg = 'compact'  # single aggregation, indices map 1:1
        print("  (Using simple index mapping for 52D)")
    else:
        # Feature selection — use compact for ablation
        X_best = variants['Compact 211D']
        agg = 'compact'
        print("  (Using Compact 211D for ablation analysis)")

    ablation_study(X_best, y, aggregation=agg)

    # 5. Confusion matrix on best
    print(f"\n[5/5] Confusion matrix ({best_name}):")
    print_confusion_matrix(X_best, y, le)

    elapsed = time.time() - t0
    print(f"\n{'=' * 50}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"{'=' * 50}")


if __name__ == '__main__':
    main()
