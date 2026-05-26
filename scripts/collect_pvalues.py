"""
collect_pvalues.py
------------------
Сбор ВСЕХ p-value из выходных таблиц results/tables/ и коррекция
на множественность (Benjamini–Hochberg FDR).

Семейства FDR (обоснование выбора):
===================================
Каждый тест относится к одному из логически связанных «вопросов» в статье.
Применять одну глобальную FDR ко всему семейству из ~400 p-values
слишком консервативно: тесты из разных частей статьи отвечают на независимые
гипотезы (контроли H1, Mantel cross-space).
Выбираем компромисс: семейства по ТИПУ теста (scope).

Семейства:
  1. controls_shuffle     — Wilcoxon shuffle > within (по каналам)
  2. controls_persistence — Wilcoxon max-pers: real vs random/shuffle/iaaft (каналы)
  3. mantel               — Mantel-тесты между парами пространств

FDR применяется ВНУТРИ каждого семейства.

Выход: results/tables/pvalues_master.csv
Колонки: family, test, scope, p_raw, p_fdr, significant_fdr

Также выводит список тестов, значимых по сырому p, но не выживающих FDR.

Примечание (косметика): p_raw = 0.0 для тестов shuffle означает машинный ноль
(т.е. p < 1e-300). При переносе результатов в статью следует указывать p < 1e-300.
"""
from __future__ import annotations

import os
import sys
import json
import re
import warnings

import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.multitest import multipletests

# ---- paths ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "tables")
TABLES_RESULTS = os.path.join(TABLES, "results")

# ---- load config for FDR alpha ----
cfg_path = os.path.join(ROOT, "config.yaml")
cfg = yaml.safe_load(open(cfg_path))
FDR_ALPHA = cfg.get("fdr_alpha", 0.05)

# ---- helpers ----


def _parse_sci(val) -> float | None:
    """Parse p-value from various formats: float, '1.23e-04', etc."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        v = float(val)
        return v if np.isfinite(v) else None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "", "inf", "-inf"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def collect_all() -> pd.DataFrame:
    """Собирает все p-value из всех таблиц. Возвращает DataFrame."""
    rows: list[dict] = []

    # ==================================================================
    # 1. controls_full.json  — основные контроли по 4 каналам
    # ==================================================================
    json_path = os.path.join(TABLES_RESULTS, "controls_full.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            cj = json.load(f)
        for space in ("mert", "muq", "encodec", "mir"):
            r = cj.get(space, {})
            # shuffle p-value (Wilcoxon paired, shuffle > within)
            p = _parse_sci(r.get("shuffle_p_value"))
            if p is not None:
                rows.append({
                    "family": "controls_shuffle",
                    "test": "wilcoxon_shuffle_gt_within",
                    "scope": space,
                    "p_raw": p,
                })
            # max_pers real vs random
            p = _parse_sci(r.get("max_pers_real_vs_random_p"))
            if p is not None:
                rows.append({
                    "family": "controls_persistence",
                    "test": "wilcoxon_maxpers_real_vs_random",
                    "scope": space,
                    "p_raw": p,
                })
            # max_pers real vs shuffle
            p = _parse_sci(r.get("max_pers_real_vs_shuffle_p"))
            if p is not None:
                rows.append({
                    "family": "controls_persistence",
                    "test": "wilcoxon_maxpers_real_vs_shuffle",
                    "scope": space,
                    "p_raw": p,
                })
            # max_pers real vs iaaft
            p = _parse_sci(r.get("max_pers_real_vs_iaaft_p"))
            if p is not None:
                rows.append({
                    "family": "controls_persistence",
                    "test": "wilcoxon_maxpers_real_vs_iaaft",
                    "scope": space,
                    "p_raw": p,
                })
    else:
        print(f"  WARNING: {json_path} not found, skipping controls_full")

    # ==================================================================
    # 2. controls_summary.csv  — formatted p-values (fallback if JSON absent)
    # ==================================================================
    summary_path = os.path.join(TABLES_RESULTS, "controls_summary.csv")
    # Already covered by JSON above, skip to avoid duplicates.

    # ==================================================================
    # 3. mantel_matrix.csv  — Mantel permutation test p-values
    # ==================================================================
    mantel_path = os.path.join(TABLES, "mantel_matrix.csv")
    if os.path.exists(mantel_path):
        mantel_df = pd.read_csv(mantel_path)
        if "p" in mantel_df.columns:
            for _, row in mantel_df.iterrows():
                p = _parse_sci(row.get("p"))
                pair = row.get("pair", "")
                if p is not None:
                    rows.append({
                        "family": "mantel",
                        "test": "mantel_permutation",
                        "scope": pair,
                        "p_raw": p,
                    })
        else:
            print(f"  INFO: {mantel_path} has no 'p' column, "
                  f"Mantel R-only — no p-values to collect")
    else:
        print(f"  WARNING: {mantel_path} not found")

    # ==================================================================
    # 4. cycle_vs_cocycle_full.csv  — chroma tests for cycles
    # ==================================================================
    cvc_path = os.path.join(TABLES, "cycle_vs_cocycle_full.csv")
    if os.path.exists(cvc_path):
        cvc_df = pd.read_csv(cvc_path)
        # Дедупликация: группируем по уникальным трекам, чтобы избежать run1/run2/nested.
        # В файле full.csv предполагается 199 уникальных треков.
        unique_tracks = cvc_df.drop_duplicates(subset=["track"])
        
        for _, row in unique_tracks.iterrows():
            track = row.get("track", "")
            
            # Основной тест: цикл Dionysus. Идёт в основную множественность
            p_cy = _parse_sci(row.get("p_dio_cy"))
            if p_cy is not None:
                rows.append({
                    "family": "chroma_loop_cycle",
                    "test": "p_dio_cy",
                    "scope": track,
                    "p_raw": p_cy,
                })
                
            # Дополнительные (коциклы, старые тесты) - НЕ идут в основную множественность,
            # выносим в отдельное deprecated семейство.
            deprecated_cols = ["p_ripser_co", "p_dio_co", "prev_p"]
            for col in deprecated_cols:
                if col in unique_tracks.columns:
                    p_dep = _parse_sci(row.get(col))
                    if p_dep is not None:
                        rows.append({
                            "family": "chroma_loop_deprecated",
                            "test": col,
                            "scope": track,
                            "p_raw": p_dep,
                        })
    else:
        print(f"  INFO: {cvc_path} not found")

    return pd.DataFrame(rows)


def apply_fdr(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Apply Benjamini-Hochberg FDR correction WITHIN each family.
    Returns augmented DataFrame with p_fdr and significant_fdr columns.
    """
    df = df.copy()
    df["p_fdr"] = np.nan
    df["significant_fdr"] = False
    df["significant_raw"] = df["p_raw"] < alpha

    for family, group in df.groupby("family"):
        idx = group.index
        pvals = group["p_raw"].values

        if len(pvals) == 0:
            continue

        # Benjamini-Hochberg
        reject, p_corrected, _, _ = multipletests(
            pvals, alpha=alpha, method="fdr_bh"
        )

        df.loc[idx, "p_fdr"] = p_corrected
        df.loc[idx, "significant_fdr"] = reject

    return df


def main():
    print("=" * 70)
    print("  collect_pvalues.py — сбор p-values и FDR-коррекция")
    print("=" * 70)
    print(f"  FDR alpha = {FDR_ALPHA}")
    print(f"  Tables dir: {TABLES}")
    print()

    # 1. Collect
    df = collect_all()
    print(f"\nTotal p-values collected: {len(df)}")
    print(f"By family:")
    for fam, cnt in df["family"].value_counts().items():
        print(f"  {fam:30s}  {cnt:4d}")

    # 2. Apply FDR
    df = apply_fdr(df, alpha=FDR_ALPHA)

    # 3. Save
    out_path = os.path.join(TABLES, "pvalues_master.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Columns: {list(df.columns)}")

    # 4. Summary
    n_sig_raw = df["significant_raw"].sum()
    n_sig_fdr = df["significant_fdr"].sum()
    print(f"\nSignificant tests (raw p < {FDR_ALPHA}):  {n_sig_raw}")
    print(f"Significant tests (FDR q < {FDR_ALPHA}):  {n_sig_fdr}")
    print(f"Lost after FDR correction:             {n_sig_raw - n_sig_fdr}")

    # 5. Report: tests significant raw but NOT surviving FDR
    lost = df[df["significant_raw"] & ~df["significant_fdr"]]
    if len(lost) > 0:
        print(f"\n{'='*70}")
        print(f"  ТЕСТЫ, ЗНАЧИМЫЕ ПО СЫРОМУ p, НО НЕ ВЫЖИВАЮЩИЕ FDR:")
        print(f"{'='*70}")
        for _, row in lost.iterrows():
            print(f"  [{row['family']}] {row['test']} | {row['scope']}")
            print(f"    p_raw = {row['p_raw']:.6e}  →  p_fdr = {row['p_fdr']:.6e}")
        print(f"\nВсего: {len(lost)} тестов потеряли значимость после FDR")
    else:
        print("\nВсе тесты, значимые по сырому p, также значимы после FDR.")

    # 6. Per-family summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY BY FAMILY")
    print(f"{'='*70}")
    print(f"{'Family':30s}  {'Total':>6s}  {'Sig(raw)':>8s}  {'Sig(FDR)':>8s}  {'Lost':>6s}")
    for fam in sorted(df["family"].unique()):
        sub = df[df["family"] == fam]
        n_total = len(sub)
        n_raw = sub["significant_raw"].sum()
        n_fdr = sub["significant_fdr"].sum()
        n_lost = n_raw - n_fdr
        print(f"{fam:30s}  {n_total:6d}  {n_raw:8d}  {n_fdr:8d}  {n_lost:6d}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
