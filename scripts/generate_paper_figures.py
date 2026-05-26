"""
generate_paper_figures.py
-------------------------
Publication figures for the coursework PDF.
Reads from results/tables/{main_results,ablations}/ and cache/.
Outputs PNG (200 dpi) to results/figures/paper/.

Unified monochromatic palette: navy / steel / pale-blue + warm-rust accent + grays.

Figures:
  fig01_pipeline.png             — methodology overview with thumbnail visualisations
  fig02_h1_max_persistence.png   — H1: real vs shuffle/phase/IAAFT, 4 channels
  fig03_h2_within_between.png    — H2: within vs between Wasserstein + CI
  fig04_h3_phase_effect.png      — H3: real vs phase rank-biserial
  fig05_h4_mantel.png            — H4: Mantel r with 95% CI, 3 pairs
  fig06_h5_classification.png    — H5: macro-F1 by feature set × channel × classifier
  fig07_hloop_progression.png    — H-loop methodology correction waterfall
  fig08_popularity_correlations.png — popularity vs topology Spearman
  fig09_cycle_vs_cocycle.png     — cocycle vs cycle vertex count + significance
  fig10_robustness_panel.png     — ablation grid (6 small multiples)
  fig11_persistence_metrics.png  — effect size by persistence summary
  fig12_persistence_diagrams.png — example diagrams: real vs surrogate ladder
  fig13_etalon_cycle_vs_cocycle.png — 60-point circle: 3 representative methods
  fig14_chromagram_example.png   — chromagram + self-similarity, loop intuition
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TBL = ROOT / "results" / "tables"
ABL = TBL / "ablations"
MAIN = TBL / "main_results"
OUT = ROOT / "results" / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

CHANNELS = ["muq", "mert", "encodec", "mir"]
CHANNEL_LABELS = {"muq": "MuQ", "mert": "MERT", "encodec": "EnCodec", "mir": "MIR"}

# Unified monochromatic palette
# Blues family for signal, warm rust accent for highlights.
#   + grays (surrogate ladder, light → dark = weaker → stricter null)
C_NAVY      = "#1F3A5F"   # primary — real signal
C_STEEL     = "#4F7CAC"   # secondary blue
C_PALE      = "#A4C2E5"   # tertiary blue
C_PALE2     = "#D6E4F2"   # very light blue
C_ACCENT    = "#C45A2A"   # warm rust — highlight / key finding
C_ACCENT_L  = "#E8A87C"   # warm light
C_GRAY_D    = "#3D3D3D"   # IAAFT (strictest control)
C_GRAY_M    = "#7A7A7A"   # phase
C_GRAY_L    = "#BFBFBF"   # shuffle

# Channel palette (when distinguishing channels): monochrome blues + rust
CHANNEL_COLORS = {
    "muq":     C_NAVY,
    "mert":    C_STEEL,
    "encodec": C_PALE,
    "mir":     C_ACCENT,
}

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#CCCCCC",
    "grid.linewidth": 0.5,
})


def save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


# fig01: Pipeline
def fig01_pipeline():
    rng = np.random.default_rng(7)

    fig = plt.figure(figsize=(16, 6.2))
    fig.patch.set_facecolor("white")

    # 6 stages, each with title row + thumbnail axes + symbol row + arrow
    n_stages = 6
    left_margin = 0.02
    right_margin = 0.02
    width_total = 1 - left_margin - right_margin
    panel_w = 0.115
    gap_w = (width_total - panel_w * n_stages) / (n_stages - 1)

    xs = [left_margin + i * (panel_w + gap_w) for i in range(n_stages)]
    y_panel = 0.42
    panel_h = 0.32

    # ---- Stage 1: Audio waveform ----
    ax = fig.add_axes([xs[0], y_panel, panel_w, panel_h])
    t = np.linspace(0, 1, 1500)
    sig = (np.sin(2*np.pi*4*t) * (0.6 + 0.4*np.sin(2*np.pi*0.7*t))
           + 0.35*np.sin(2*np.pi*11*t)
           + 0.10*rng.standard_normal(1500))
    sig /= np.abs(sig).max() * 1.1
    ax.fill_between(t, sig, 0, color=C_STEEL, alpha=0.30, linewidth=0)
    ax.plot(t, sig, color=C_NAVY, lw=0.55)
    ax.set_ylim(-1.2, 1.2); ax.set_xlim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    # ---- Stage 2: Embedding heatmap (frames × features) ----
    ax = fig.add_axes([xs[1], y_panel, panel_w, panel_h])
    H = rng.standard_normal((24, 80))
    H = np.cumsum(H, axis=1) * 0.15
    H = (H - H.min()) / (H.max() - H.min())
    ax.imshow(H, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor("#888"); s.set_linewidth(0.6)

    # ---- Stage 3: Preprocessed PCA traces ----
    ax = fig.add_axes([xs[2], y_panel, panel_w, panel_h])
    n = 240
    for k, c, a in [(0, C_NAVY, 1.0), (1, C_STEEL, 0.9), (2, C_ACCENT, 0.85)]:
        y = np.cumsum(rng.standard_normal(n)) * 0.07
        y -= y.mean()
        ax.plot(y + (1.4 - k*1.4), color=c, lw=1.1, alpha=a)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    # ---- Stage 4: Takens point cloud (2D projection of a loop) ----
    ax = fig.add_axes([xs[3], y_panel, panel_w, panel_h])
    n_pts = 120
    theta = np.linspace(0, 4 * np.pi, n_pts)
    r = 1.0 + 0.08 * np.sin(3 * theta) + rng.normal(0, 0.05, n_pts)
    px = r * np.cos(theta) + rng.normal(0, 0.03, n_pts)
    py = r * np.sin(theta) + rng.normal(0, 0.03, n_pts)
    ax.scatter(px, py, s=10, c=C_NAVY, alpha=0.55, edgecolor="none")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    # ---- Stage 5: Persistence diagram ----
    ax = fig.add_axes([xs[4], y_panel, panel_w, panel_h])
    n_pd = 40
    births = rng.uniform(0.0, 0.55, n_pd)
    deaths = births + np.abs(rng.normal(0.04, 0.03, n_pd))
    # One prominent off-diagonal feature (the loop)
    births[0], deaths[0] = 0.18, 0.62
    lim = 0.85
    ax.plot([0, lim], [0, lim], color="#999", lw=0.9, linestyle="--")
    ax.scatter(births[1:], deaths[1:], s=14, c=C_STEEL, alpha=0.55, edgecolor="none")
    ax.scatter([births[0]], [deaths[0]], s=140, marker="*",
               c=C_ACCENT, edgecolor="black", linewidth=0.5, zorder=5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values(): s.set_edgecolor("#888"); s.set_linewidth(0.6)
    ax.text(0.05, lim - 0.05, "birth", fontsize=7, color="#666", ha="left", va="top")
    ax.text(lim - 0.05, 0.05, "death", fontsize=7, color="#666", ha="right", va="bottom")

    # ---- Stage 6: Outputs (small bar chart icon) ----
    ax = fig.add_axes([xs[5], y_panel, panel_w, panel_h])
    bars = [0.82, 0.35, 0.62, 0.49]
    colors = [C_NAVY, C_GRAY_L, C_GRAY_M, C_GRAY_D]
    ax.bar(range(4), bars, color=colors, edgecolor="black", linewidth=0.4, width=0.7)
    ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    # ---- Math symbols above each panel ----
    sym_y = y_panel + panel_h + 0.07
    symbols = [r"$x(t)$",
               r"$E \in \mathbb{R}^{T\times d}$",
               r"$X \in \mathbb{R}^{T'\times 32}$",
               r"$Z \subset \mathbb{R}^{wm}$",
               r"$D_1=\{(b_i,d_i)\}$",
               r"max-pers, $W_2$, $\beta_D$"]
    for x, s in zip(xs, symbols):
        fig.text(x + panel_w / 2, sym_y, s, ha="center", va="bottom",
                 fontsize=10, color=C_NAVY, style="italic")

    # ---- Stage titles below ----
    title_y = y_panel - 0.04
    titles = [
        ("Audio", "30s GTZAN · 90s Spotify"),
        ("Embedding", "MERT · MuQ · EnCodec · MIR"),
        ("Preprocess", "resample → scale → ℓ₂ → PCA(32)"),
        ("Takens cloud", "window 16, stride 4"),
        ("Persistent H₁", "Vietoris–Rips (Ripser)"),
        ("Outputs", "scalar · vector · cycle"),
    ]
    for x, (t1, t2) in zip(xs, titles):
        fig.text(x + panel_w / 2, title_y, t1, ha="center", va="top",
                 fontsize=10.5, weight="bold")
        fig.text(x + panel_w / 2, title_y - 0.045, t2, ha="center", va="top",
                 fontsize=8.5, color="#555")

    # ---- Arrows between consecutive stages ----
    arr_y = y_panel + panel_h / 2
    for i in range(n_stages - 1):
        x0 = xs[i] + panel_w + 0.005
        x1 = xs[i + 1] - 0.005
        arr = FancyArrowPatch((x0, arr_y), (x1, arr_y),
                              arrowstyle="-|>", mutation_scale=16,
                              color="#555", lw=1.4,
                              transform=fig.transFigure)
        fig.add_artist(arr)

    # ---- Surrogate branch: the null models are generated from X, then use the same Takens/Ripser path ----
    surr_left = xs[2] - 0.015
    surr_right = xs[3] + panel_w + 0.015
    surr_bottom = 0.035
    surr_h = 0.145
    surr_box = FancyBboxPatch(
        (surr_left, surr_bottom), surr_right - surr_left, surr_h,
        boxstyle="round,pad=0.012",
        facecolor="#FBEFE6", edgecolor=C_ACCENT, lw=0.9,
        transform=fig.transFigure)
    fig.add_artist(surr_box)

    surr_mid = (surr_left + surr_right) / 2
    fig.text(surr_mid, surr_bottom + surr_h - 0.030,
             "Surrogate controls from preprocessed X (K=20)",
             ha="center", va="center", fontsize=9.5, weight="bold", color=C_ACCENT)
    fig.text(surr_mid, surr_bottom + surr_h - 0.061,
             "same Takens + Ripser pipeline, compared against real diagrams",
             ha="center", va="center", fontsize=8.2, color="#555")

    # ---- Downstream branch: analyses start after persistence diagrams have been computed ----
    down_left = xs[4] - 0.020
    down_right = xs[5] + panel_w + 0.020
    down_bottom = 0.035
    down_h = 0.145
    down_box = FancyBboxPatch(
        (down_left, down_bottom), down_right - down_left, down_h,
        boxstyle="round,pad=0.012",
        facecolor="#EAF0F5", edgecolor=C_NAVY, lw=0.9,
        transform=fig.transFigure)
    fig.add_artist(down_box)
    down_mid = (down_left + down_right) / 2
    fig.text(down_mid, down_bottom + down_h - 0.030,
             "Downstream analyses of persistence diagrams",
             ha="center", va="center", fontsize=9.5, weight="bold", color=C_NAVY)
    fig.text(down_mid, down_bottom + down_h - 0.061,
             "hypothesis tests, distances, classifiers, and representative cycles",
             ha="center", va="center", fontsize=8.2, color="#555")

    fig.suptitle("Methodology pipeline — from raw audio to persistent homology and statistical tests",
                 fontsize=12.5, weight="bold", y=0.99)
    save(fig, "fig01_pipeline.png")


# fig02: H1 max-persistence
def fig02_h1_max_persistence():
    df = pd.read_csv(MAIN / "controls_summary.csv").set_index("space")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(CHANNELS))
    width = 0.2

    series = [
        ("Real",    "max_pers_real_mean",    "max_pers_real_sem",    C_NAVY),
        ("Shuffle", "max_pers_shuffle_mean", "max_pers_shuffle_sem", C_GRAY_L),
        ("Phase",   "max_pers_random_mean",  "max_pers_random_sem",  C_GRAY_M),
        ("IAAFT",   "max_pers_iaaft_mean",   "max_pers_iaaft_sem",   C_GRAY_D),
    ]
    for i, (label, col, sem_col, color) in enumerate(series):
        means = [df.loc[c, col] for c in CHANNELS]
        sems  = [df.loc[c, sem_col] for c in CHANNELS]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=sems, capsize=2.5,
               label=label, color=color, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([CHANNEL_LABELS[c] for c in CHANNELS])
    ax.set_ylabel(r"Mean max $H_1$ persistence")
    ax.set_title("H1 — Real loops exceed all surrogate controls in every channel\n"
                 r"(rank-biserial $r \approx 0.9$–$1.0$, $p<10^{-130}$ on all pairs)",
                 fontsize=10.5)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    save(fig, "fig02_h1_max_persistence.png")


# fig03: Within vs between Wasserstein
def fig03_h2_within_between():
    df = pd.read_csv(MAIN / "controls_summary.csv").set_index("space")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4),
                                    gridspec_kw={"width_ratios": [1.3, 1]})

    x = np.arange(len(CHANNELS))
    w = 0.38
    within  = [df.loc[c, "within_mean"]  for c in CHANNELS]
    between = [df.loc[c, "between_mean"] for c in CHANNELS]
    ax1.bar(x - w/2, within,  w, label="Within-genre",  color=C_NAVY,
            edgecolor="black", linewidth=0.4)
    ax1.bar(x + w/2, between, w, label="Between-genre", color=C_PALE,
            edgecolor="black", linewidth=0.4)
    ax1.set_xticks(x); ax1.set_xticklabels([CHANNEL_LABELS[c] for c in CHANNELS])
    ax1.set_ylabel(r"Mean Wasserstein $W_2$ between $H_1$ diagrams")
    ax1.set_title("Within- vs between-genre distances")
    ax1.legend(frameon=False)
    ax1.grid(axis="y", alpha=0.3, linestyle=":")

    gap = [df.loc[c, "gap_obs_track"] for c in CHANNELS]
    lo  = [df.loc[c, "gap_ci95_track_lo"] for c in CHANNELS]
    hi  = [df.loc[c, "gap_ci95_track_hi"] for c in CHANNELS]
    err_lo = np.array(gap) - np.array(lo)
    err_hi = np.array(hi) - np.array(gap)
    ax2.barh(x, gap, xerr=[err_lo, err_hi], capsize=3.5, color=C_ACCENT,
             edgecolor="black", linewidth=0.4)
    ax2.set_yticks(x); ax2.set_yticklabels([CHANNEL_LABELS[c] for c in CHANNELS])
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel(r"$\Delta = W_{between} - W_{within}$  (95% bootstrap CI by track)")
    ax2.set_title("H2 — genre gap (all CIs exclude 0,\nbut small vs within-std ≈ 3–5)")
    ax2.grid(axis="x", alpha=0.3, linestyle=":")
    save(fig, "fig03_h2_within_between.png")


# fig04: Rank-biserial effect sizes
def fig04_h3_phase_effect():
    df = pd.read_csv(MAIN / "controls_summary.csv").set_index("space")
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    x = np.arange(len(CHANNELS))
    width = 0.27

    eff_shuf  = [df.loc[c, "real_vs_shuffle_effect_r"] for c in CHANNELS]
    eff_rand  = [df.loc[c, "real_vs_random_effect_r"]  for c in CHANNELS]
    eff_iaaft = [df.loc[c, "real_vs_iaaft_effect_r"]   for c in CHANNELS]

    ax.bar(x - width, eff_shuf,  width, label="vs Shuffle", color=C_GRAY_L,
           edgecolor="black", linewidth=0.4)
    ax.bar(x,         eff_rand,  width, label="vs Phase",   color=C_GRAY_M,
           edgecolor="black", linewidth=0.4)
    ax.bar(x + width, eff_iaaft, width, label="vs IAAFT",   color=C_GRAY_D,
           edgecolor="black", linewidth=0.4)

    for thresh, lbl in [(0.5, "medium"), (0.8, "large")]:
        ax.axhline(thresh, color=C_ACCENT, lw=0.7, linestyle="--", alpha=0.6)
        ax.text(len(CHANNELS) - 0.55, thresh + 0.012, lbl, fontsize=8,
                color=C_ACCENT)

    ax.set_xticks(x); ax.set_xticklabels([CHANNEL_LABELS[c] for c in CHANNELS])
    ax.set_ylabel("Rank-biserial $r$ (paired Wilcoxon)")
    ax.set_ylim(0, 1.05)
    ax.set_title("H3 — Real exceeds every surrogate by a large effect;\n"
                 "phase / IAAFT slightly smaller than shuffle "
                 "(stricter spectrum-preserving surrogates leave less margin)",
                 fontsize=10.5)
    ax.legend(frameon=False, ncol=3, loc="lower center")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    save(fig, "fig04_h3_phase_effect.png")


# fig05: Mantel cross-representation
def fig05_h4_mantel():
    df = pd.read_csv(MAIN / "mantel_matrix.csv")
    # Build symmetric 3×3 matrix
    spaces = ["mert", "muq", "encodec"]
    labels = ["MERT", "MuQ", "EnCodec"]
    n = len(spaces)
    R = np.full((n, n), np.nan)
    P = np.full((n, n), np.nan)
    for _, row in df.iterrows():
        a, _, b = row["pair"].partition("_vs_")
        if a in spaces and b in spaces:
            i, j = spaces.index(a), spaces.index(b)
            R[i, j] = R[j, i] = row["r"]
            P[i, j] = P[j, i] = row["p"]
    np.fill_diagonal(R, 1.0)
    np.fill_diagonal(P, 0.0)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    cmap = plt.get_cmap("RdBu_r")
    im = ax.imshow(R, cmap=cmap, vmin=-1, vmax=1)

    for i in range(n):
        for j in range(n):
            if np.isnan(R[i, j]):
                continue
            txt_color = "white" if abs(R[i, j]) > 0.55 else "black"
            if i == j:
                label = "1.000"
            else:
                star = "***" if P[i, j] < 0.001 else ("*" if P[i, j] < 0.05 else "ns")
                label = f"r = {R[i, j]:.3f}\np = {P[i, j]:.3g}\n{star}"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=9.5, color=txt_color)

    ax.set_xticks(range(n)); ax.set_xticklabels(labels)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Mantel correlation $r$ (Wasserstein distance matrices)", fontsize=9)
    ax.set_title("H4 — Cross-representation topological consistency\n"
                 "strongest pair r=0.28 (SSL–SSL); codec pair ≈ 0\n"
                 "* p<0.05, *** p<0.001, ns = not significant",
                 fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig05_h4_mantel.png")


# fig06: Classification
def fig06_h5_classification():
    df = pd.read_csv(MAIN / "classification_full.csv")
    df["Space"] = df["Space"].str.lower()

    def parse_ci(s):
        s = s.strip("[] ")
        a, b = s.split(",")
        return float(a), float(b)

    cis = df["F1_CI"].apply(parse_ci)
    df["f1_lo"] = [c[0] for c in cis]
    df["f1_hi"] = [c[1] for c in cis]

    feature_order = ["persistence", "persistence@shuffle", "persistence@phase",
                     "persistence@iaaft", "mean_pool", "concat(pers+mean)"]
    feature_lbl = {
        "persistence": "topology (real)", "persistence@shuffle": "topology@shuffle",
        "persistence@phase": "topology@phase", "persistence@iaaft": "topology@IAAFT",
        "mean_pool": "mean_pool", "concat(pers+mean)": "concat",
    }
    # monochrome ladder: topology = navy, surrogates = gray ladder, mean_pool = accent
    f_colors = [C_NAVY, C_GRAY_L, C_GRAY_M, C_GRAY_D, C_ACCENT, C_ACCENT_L]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, clf in zip(axes, ["logreg", "hgb"]):
        sub = df[df["classifier"] == clf]
        x = np.arange(len(CHANNELS))
        n_f = len(feature_order)
        bw = 0.13
        for i, feat in enumerate(feature_order):
            vals, errs = [], []
            for ch in CHANNELS:
                r = sub[(sub["Space"] == ch) & (sub["Features"] == feat)]
                if len(r):
                    f1 = float(r["Macro_F1"].iloc[0])
                    vals.append(f1)
                    errs.append([f1 - float(r["f1_lo"].iloc[0]),
                                 float(r["f1_hi"].iloc[0]) - f1])
                else:
                    vals.append(np.nan); errs.append([0, 0])
            errs = np.array(errs).T
            ax.bar(x + (i - (n_f - 1) / 2) * bw, vals, bw,
                   yerr=errs if clf == "logreg" else None, capsize=1.8,
                   label=feature_lbl[feat], color=f_colors[i],
                   edgecolor="black", linewidth=0.3)
        # Random-guess baseline for 10-genre classification
        ax.axhline(0.10, color=C_ACCENT, lw=1.0, linestyle="--", alpha=0.9)
        ax.text(0.05, 0.115, "random-guess baseline (1/10 genres = 0.10)",
                fontsize=8.5, color=C_ACCENT, ha="left", weight="bold")
        ax.set_xticks(x); ax.set_xticklabels([CHANNEL_LABELS[c] for c in CHANNELS])
        ax.set_title(f"{clf.upper()}", fontsize=11, weight="bold")
        ax.set_ylim(0, 0.95)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
    axes[0].set_ylabel("Macro-F1 (GTZAN test split)")
    axes[1].legend(ncol=2, frameon=False, loc="upper right", fontsize=8.5)
    fig.suptitle("H5 — Topology features ≈ chance; classification is carried by mean_pool (timbre / position)",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    save(fig, "fig06_h5_classification.png")


# fig07: H-loop methodology progression
def fig07_hloop_progression():
    methods = ["Cocycle +\nMann–Whitney\n(uncorrected)",
               "Cocycle +\npermutation test",
               "Cocycle +\nperm + FDR",
               "Dionysus cycle +\nperm + FDR"]
    pct = [55.0, 43.0, 35.0, 27.0]
    n   = ["~100/192", "82/192", "67/192", "52/192"]
    # monotonic blue gradient ending in accent for "final correct"
    colors = [C_PALE, C_STEEL, C_NAVY, C_ACCENT]

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    bars = ax.bar(methods, pct, color=colors, edgecolor="black",
                  linewidth=0.5, width=0.6)
    for b, p, lbl in zip(bars, pct, n):
        ax.text(b.get_x() + b.get_width() / 2, p + 1.2, f"{p:.0f}%\n({lbl})",
                ha="center", va="bottom", fontsize=9)
    for i in range(len(pct) - 1):
        d = pct[i] - pct[i + 1]
        ax.annotate(f"−{d:.0f}%", xy=(i + 0.5, (pct[i] + pct[i + 1]) / 2),
                    ha="center", fontsize=9, color="#555",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#aaa", lw=0.5))
    ax.set_ylim(0, 70)
    ax.set_ylabel("Tracks with significant chromatic recurrence (%)")
    ax.set_title("H-loop — methodological corrections almost halve the rate of \"meaningful\" loops\n"
                 "(cocycle support is not a loop contour; dependent comparisons inflate p-values)",
                 fontsize=10.5)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    save(fig, "fig07_hloop_progression.png")


# fig08: Popularity correlations
def fig08_popularity_correlations():
    df = pd.read_csv(MAIN / "popularity_topology_correlations.csv")
    metrics_keep = ["max_persistence", "n_loop_vertices", "loop_span_sec", "chroma_excess"]
    pretty = {"max_persistence": "max H₁ persistence",
              "n_loop_vertices": "n loop vertices",
              "loop_span_sec":   "loop temporal span",
              "chroma_excess":   "chroma excess"}
    spaces = ["MuQ", "MIR"]
    # Columns: [MuQ raw, MuQ centered, MIR raw, MIR centered]
    col_labels = ["MuQ\nraw ρ", "MuQ\ngenre-centered", "MIR\nraw ρ", "MIR\ngenre-centered"]
    R = np.full((len(metrics_keep), 4), np.nan)
    P = np.full((len(metrics_keep), 4), np.nan)
    for i, m in enumerate(metrics_keep):
        for j, space in enumerate(spaces):
            row = df[(df["space"] == space) & (df["metric"] == m)]
            if len(row):
                R[i, 2*j]     = row["spearman_rho"].iloc[0]
                R[i, 2*j + 1] = row["genre_centered_rho"].iloc[0]
                P[i, 2*j]     = row["p_value"].iloc[0]
                P[i, 2*j + 1] = row["genre_centered_p"].iloc[0]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    cmap = plt.get_cmap("RdBu_r")
    vmax = 0.20
    im = ax.imshow(R, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if np.isnan(R[i, j]): continue
            txt_color = "white" if abs(R[i, j]) > 0.13 else "black"
            star = "*" if P[i, j] < 0.05 else "ns"
            ax.text(j, i, f"ρ = {R[i, j]:+.3f}\np = {P[i, j]:.2f}  {star}",
                    ha="center", va="center", fontsize=9, color=txt_color)

    ax.set_xticks(range(R.shape[1])); ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(R.shape[0]))
    ax.set_yticklabels([pretty[m] for m in metrics_keep])
    ax.set_xticks(np.arange(-.5, R.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, R.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Spearman ρ vs Spotify popularity", fontsize=9)
    ax.set_title("Popularity vs topological metrics  (n=198 top-Spotify tracks, 4 genres)\n"
                 "no cell survives correction; the largest |ρ| ≈ 0.12 \n"
                 "* p<0.05 raw, ns = not significant",
                 fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig08_popularity_correlations.png")


# fig09: Cycle vs cocycle
def fig09_cycle_vs_cocycle():
    df = pd.read_csv(MAIN / "cycle_vs_cocycle_full.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                    gridspec_kw={"width_ratios": [1.1, 1]})

    parts = [df["n_ripser_co"].values, df["n_dio_co"].values, df["n_dio_cy"].values]
    labels = ["Ripser\ncocycle", "Dionysus\ncocycle", "Dionysus\ncycle"]
    colors = [C_STEEL, C_GRAY_M, C_ACCENT]
    bp = ax1.boxplot(parts, tick_labels=labels, patch_artist=True, widths=0.55,
                     medianprops=dict(color="black", lw=1.2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor("black")
    medians = [int(np.median(p)) for p in parts]
    for i, m in enumerate(medians, start=1):
        ax1.text(i, m + 6, f"median={m}", ha="center", fontsize=9, weight="bold")
    ax1.set_ylabel("Vertices in the loop representative")
    ax1.set_title(f"Geometric compactness (N={len(df)} Spotify90 tracks)\n"
                  "Homological cycle ≈ 2× more compact than cocycle support")
    ax1.grid(axis="y", alpha=0.3, linestyle=":")

    methods = ["Ripser cocycle", "Dionysus cocycle", "Dionysus cycle"]
    n_sig_raw = [int((df[p] < 0.05).sum()) for p in ["p_ripser_co", "p_dio_co", "p_dio_cy"]]
    n_sig_fdr = [int(df[s].sum()) for s in ["sig_ripser_co", "sig_dio_co", "sig_dio_cy"]]
    x = np.arange(len(methods)); w = 0.36
    b1 = ax2.bar(x - w/2, n_sig_raw, w, color=C_PALE, edgecolor="black",
                 linewidth=0.4, label="raw p<0.05")
    b2 = ax2.bar(x + w/2, n_sig_fdr, w, color=C_NAVY, edgecolor="black",
                 linewidth=0.4, label="FDR p<0.05")
    for bars in (b1, b2):
        for b in bars:
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                     f"{int(b.get_height())}", ha="center", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(methods, fontsize=9)
    ax2.set_ylabel(f"Tracks with chromatic recurrence (of N={len(df)})")
    ax2.set_title("Significance after FDR correction\n"
                  "Strict pipeline: cycle yields 52/192 (≈27%)")
    ax2.legend(frameon=False, loc="upper right")
    ax2.grid(axis="y", alpha=0.3, linestyle=":")
    save(fig, "fig09_cycle_vs_cocycle.png")


# fig10: Robustness ablation panel
def fig10_robustness_panel():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    markers = {"muq": "o", "mert": "s", "encodec": "^", "mir": "D"}

    def _plot_channels(ax, d, xcol, ycol):
        for sp in d["space"].unique():
            sub = d[d["space"] == sp].sort_values(xcol)
            ax.plot(sub[xcol], sub[ycol], marker=markers.get(sp, "o"),
                    markersize=5, lw=1.4,
                    color=CHANNEL_COLORS.get(sp, C_NAVY),
                    label=CHANNEL_LABELS.get(sp, sp))

    # (a) PCA dim
    ax = axes[0, 0]
    d = pd.read_csv(ABL / "ablation_pca_dims.csv")
    _plot_channels(ax, d, "pca_dim", "gap")
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 32, 64, 128]); ax.set_xticklabels([16, 32, 64, 128])
    ax.set_xlabel("PCA dimension"); ax.set_ylabel(r"Gap $\Delta$")
    ax.set_title("(a) PCA dimension")
    ax.axvline(32, color=C_ACCENT, lw=0.7, linestyle="--", alpha=0.7)
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3, linestyle=":")

    # (b) Window time-scale
    ax = axes[0, 1]
    d = pd.read_csv(ABL / "ablation_window_scales.csv")
    d = d[d["variant"] == "standard"]
    _plot_channels(ax, d, "timescale_sec", "gap")
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.32, 0.64, 1.28, 2.56])
    ax.set_xticklabels(["0.32", "0.64", "1.28", "2.56"])
    ax.set_xlabel("Takens time-scale (s)"); ax.set_ylabel(r"Gap $\Delta$")
    ax.set_title("(b) Window scale")
    ax.axvline(0.64, color=C_ACCENT, lw=0.7, linestyle="--", alpha=0.7)
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3, linestyle=":")

    # (c) Stride
    ax = axes[0, 2]
    d = pd.read_csv(ABL / "ablation_stride.csv")
    _plot_channels(ax, d, "stride", "gap")
    ax.set_xticks(sorted(d["stride"].unique()))
    ax.invert_xaxis()
    ax.set_xlabel("Stride (smaller = denser cloud)")
    ax.set_ylabel(r"Gap $\Delta$"); ax.set_title("(c) Stride")
    ax.axvline(4, color=C_ACCENT, lw=0.7, linestyle="--", alpha=0.7)
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3, linestyle=":")

    # (d) Normalization variants
    ax = axes[1, 0]
    d = pd.read_csv(ABL / "ablation_normalization.csv")
    spaces_in = list(d["space"].unique())
    variants = list(d["variant"].unique())
    short = {"A_StandardScaler_Only": "Std", "B_StandardScaler_L2": "Std+ℓ₂",
             "C_RobustScaler_L2": "Robust+ℓ₂", "D_No_Normalization": "None"}
    variant_lbls = [short.get(v, v) for v in variants]
    x = np.arange(len(variants)); w = 0.8 / max(1, len(spaces_in))
    for j, sp in enumerate(spaces_in):
        sub = d[d["space"] == sp].set_index("variant").reindex(variants)
        ax.bar(x + (j - (len(spaces_in)-1)/2) * w, sub["gap"], w,
               color=CHANNEL_COLORS.get(sp, C_NAVY),
               edgecolor="black", linewidth=0.4,
               label=CHANNEL_LABELS.get(sp, sp))
    ax.set_xticks(x); ax.set_xticklabels(variant_lbls, fontsize=9)
    ax.set_ylabel(r"Gap $\Delta$"); ax.set_title("(d) Normalization")
    ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", alpha=0.3, linestyle=":")

    # (e) Seed stability
    ax = axes[1, 1]
    d = pd.read_csv(ABL / "ablation_seed_stability.csv")
    seeds_sorted = sorted(d["seed"].unique())
    seed_pos = {s: i for i, s in enumerate(seeds_sorted)}
    for sp in d["space"].unique():
        sub = d[d["space"] == sp].sort_values("seed")
        xpos = [seed_pos[s] for s in sub["seed"]]
        ax.plot(xpos, sub["gap"], marker=markers.get(sp, "o"),
                markersize=6, lw=1.4,
                color=CHANNEL_COLORS.get(sp, C_NAVY),
                label=CHANNEL_LABELS.get(sp, sp))
    ax.set_xticks(range(len(seeds_sorted)))
    ax.set_xticklabels([str(s) for s in seeds_sorted])
    ax.set_xlabel("Random seed (5 splits)"); ax.set_ylabel(r"Gap $\Delta$")
    ax.set_title("(e) Split-seed stability")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3, linestyle=":")

    # (f) Distance concentration
    ax = axes[1, 2]
    d = pd.read_csv(ABL / "ablation_distance_concentration.csv")
    spaces_in = list(d["space"].unique())
    x = np.arange(len(spaces_in)); w = 0.27
    raw = [d[d["space"] == s]["raw_concentration"].iloc[0]    for s in spaces_in]
    pca = [d[d["space"] == s]["pca_concentration"].iloc[0]    for s in spaces_in]
    tak = [d[d["space"] == s]["takens_concentration"].iloc[0] for s in spaces_in]
    ax.bar(x - w, raw, w, label="raw frames",   color=C_PALE,  edgecolor="black", linewidth=0.4)
    ax.bar(x,     pca, w, label="PCA frames",   color=C_STEEL, edgecolor="black", linewidth=0.4)
    ax.bar(x + w, tak, w, label="Takens cloud", color=C_NAVY,  edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([CHANNEL_LABELS.get(s, s) for s in spaces_in])
    ax.set_ylabel(r"Concentration $\kappa$")
    ax.set_title("(f) Pairwise-distance concentration")
    ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", alpha=0.3, linestyle=":")

    fig.suptitle("Robustness analysis — central conclusion (real > controls) is stable; "
                 "absolute metric values are parameter-sensitive",
                 fontsize=12, weight="bold", y=1.00)
    fig.tight_layout()
    save(fig, "fig10_robustness_panel.png")


# fig11: Persistence-summary metric ablation
def fig11_persistence_metrics():
    """Question: does the conclusion 'real > shuffle' depend on which scalar
    summary of the persistence diagram we pick? Effect = +1 means real always
    beats shuffle on this metric; −1 means shuffle always beats real."""
    d = pd.read_csv(ABL / "ablation_persistence_metrics.csv")
    # Group: STABLE (always +1 on real data) vs UNSTABLE (sign flips by channel)
    metrics = [
        ("effect_max",        "max persistence",        "stable"),
        ("effect_top3_mean",  "mean of top-3 lifetimes","stable"),
        ("effect_count_05",   "# loops (life > 0.05)",  "unstable"),
        ("effect_total",      "total H₁ lifetime",      "unstable"),
        ("effect_betti_area", "Betti-curve area",       "unstable"),
        ("effect_entropy",    "persistence entropy",    "unstable"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    spaces_in = list(d["space"].unique())
    x = np.arange(len(metrics))
    w = 0.8 / max(1, len(spaces_in))
    for j, sp in enumerate(spaces_in):
        row = d[d["space"] == sp].iloc[0]
        vals = [row[k] for k, _, _ in metrics]
        ax.bar(x + (j - (len(spaces_in)-1)/2) * w, vals, w,
               label=CHANNEL_LABELS.get(sp, sp),
               color=CHANNEL_COLORS.get(sp, C_NAVY),
               edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.8,  color=C_ACCENT, lw=0.6, linestyle="--", alpha=0.6)
    ax.axhline(-0.8, color=C_ACCENT, lw=0.6, linestyle="--", alpha=0.6)
    ax.text(len(metrics) - 0.4, 0.83, "large effect (real ≫ shuffle)",
            fontsize=8, color=C_ACCENT, ha="right")
    ax.text(len(metrics) - 0.4, -0.92, "large effect (shuffle ≫ real)",
            fontsize=8, color=C_ACCENT, ha="right")

    # Shaded background to separate stable / unstable metrics
    n_stable = sum(1 for _, _, k in metrics if k == "stable")
    ax.axvspan(-0.5, n_stable - 0.5, color=C_PALE2, alpha=0.35, zorder=0)
    ax.axvspan(n_stable - 0.5, len(metrics) - 0.5, color="#F5E8E2", alpha=0.6, zorder=0)
    ax.text((n_stable - 1) / 2, 1.10, "ROBUST summaries\n(use these)",
            ha="center", fontsize=9.5, weight="bold", color=C_NAVY)
    ax.text((n_stable + len(metrics) - 1) / 2, 1.10,
            "FRAGILE summaries — can give shuffle > real\n"
            "(shuffle generates many noisy short-lived loops)",
            ha="center", fontsize=9.5, weight="bold", color=C_ACCENT)

    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in metrics], rotation=15, ha="right")
    ax.set_ylabel("Rank-biserial $r$  (real vs shuffle)")
    ax.set_ylim(-1.20, 1.30)
    ax.set_title("Which scalar summary of the persistence diagram should we use?\n"
                 "Max-persistence and top-3 lifetimes always say «real > shuffle»;\n"
                 "aggregate counts and entropy can flip sign across channels — pick a robust summary",
                 fontsize=10.5)
    ax.legend(frameon=False, ncol=4, loc="lower center")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    save(fig, "fig11_persistence_metrics.png")


# fig12: Persistence diagrams
def fig12_persistence_diagrams():
    """One demo track, compute four persistence diagrams (real / shuffle / phase / IAAFT)
    and plot side-by-side. Demonstrates visually what the H1 statistics summarise."""
    import preprocess, pointcloud, controls, persistence as pers_mod
    import ripser as ripser_lib

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    track_npy = ROOT / "cache" / "muq_spotify90" / "pop_29_Something_Just_Like_This_The_Chainsmokers.npy"
    if not track_npy.exists():
        # fall back to first available cache file
        track_npy = next((ROOT / "cache" / "muq_spotify90").glob("*.npy"))
    reducer = preprocess.PCAReducer.load(ROOT / "cache" / "pca_reducer_muq_spotify90.joblib")

    emb = np.load(track_npy)
    x = preprocess.prepare_track(emb, cfg["spaces"]["muq"], cfg["common"], reducer)

    pc_cfg = dict(cfg["pointcloud"]); pc_cfg["takens_pca_dim"] = None
    pers_cfg = cfg["persistence"]

    def diagrams(arr):
        cloud = pointcloud.build_cloud(arr, pc_cfg)
        res = pers_mod.compute_diagrams(cloud, pers_cfg["maxdim"], 0.0)
        dgms = res["dgms"]
        h0 = dgms[0] if len(dgms) > 0 else np.empty((0, 2))
        h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
        return h0, h1

    rng_s = np.random.default_rng(42)
    rng_p = np.random.default_rng(43)
    rng_i = np.random.default_rng(44)

    real_h0, real_h1 = diagrams(x)
    shuf_h0, shuf_h1 = diagrams(controls.shuffle_frames(x, rng_s))
    phase_h0, phase_h1 = diagrams(controls.random_like(x, rng_p, match_autocorr=True))
    iaaft_h0, iaaft_h1 = diagrams(controls.iaaft_surrogate(x, rng_i, n_iter=200, tol=1e-8))

    panels = [
        ("Real trajectory",                          real_h0,  real_h1),
        ("Shuffle (order destroyed)",                shuf_h0,  shuf_h1),
        ("Phase-randomization (linear spectrum kept)", phase_h0, phase_h1),
        ("IAAFT (spectrum + marginals kept)",        iaaft_h0, iaaft_h1),
    ]

    # Common limit from finite H1 deaths (H0 deaths are also finite except 1 infinite point)
    finite_all = []
    for _, h0, h1 in panels:
        for d in (h0, h1):
            if len(d):
                fin = d[np.isfinite(d[:, 1])]
                if len(fin): finite_all.append(fin)
    if finite_all:
        lim = max(d[:, 1].max() for d in finite_all) * 1.08
    else:
        lim = 1.0

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharey=True, sharex=True)

    H0_COLOR = C_STEEL
    H1_COLOR = C_NAVY

    for ax_i, (ax, (title, h0, h1)) in enumerate(zip(axes, panels)):
        ax.plot([0, lim], [0, lim], color="#999", lw=0.9, linestyle="--")
        ax.fill_between([0, lim], [0, lim], 0, color="#EEEEEE", alpha=0.4, zorder=0)

        # ---- H0 (connected components) — circle markers, blue ----
        if len(h0) > 0:
            h0f = h0[np.isfinite(h0[:, 1])]
            if len(h0f) > 0:
                ax.scatter(h0f[:, 0], h0f[:, 1], s=16, marker="o",
                           c=H0_COLOR, alpha=0.55, edgecolor="black",
                           linewidth=0.25,
                           label=f"$H_0$  (n={len(h0)})" if ax_i == 0 else None)
            # Show the infinite H0 point as an arrow at top
            n_inf = int(np.sum(~np.isfinite(h0[:, 1])))
            if n_inf > 0:
                ax.annotate(f"$H_0$: {n_inf} class → ∞", xy=(0.04, lim - 0.05),
                            fontsize=8, color=H0_COLOR, ha="left", va="top")

        # ---- H1 (loops) — square markers, navy ----
        if len(h1) > 0:
            h1f = h1[np.isfinite(h1[:, 1])]
            if len(h1f) > 0:
                lifetimes = h1f[:, 1] - h1f[:, 0]
                sizes = 22 + 110 * (lifetimes / max(lifetimes.max(), 1e-6))
                ax.scatter(h1f[:, 0], h1f[:, 1], s=sizes, marker="s",
                           c=H1_COLOR, alpha=0.7, edgecolor="black",
                           linewidth=0.3,
                           label=f"$H_1$  (n={len(h1)})" if ax_i == 0 else None)
                k = int(np.argmax(lifetimes))
                ax.scatter([h1f[k, 0]], [h1f[k, 1]],
                           s=240, marker="*", c=C_ACCENT,
                           edgecolor="black", linewidth=0.7, zorder=10,
                           label="max-persistence $H_1$" if ax_i == 0 else None)
                ax.text(h1f[k, 0] + 0.02, h1f[k, 1] + 0.03,
                        f"max $H_1$ pers = {lifetimes.max():.3f}",
                        fontsize=8.5, color=C_ACCENT, weight="bold")

        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("Birth $\\varepsilon$")
        ax.set_title(title, fontsize=10.5)
        ax.grid(alpha=0.3, linestyle=":")
        ax.text(lim * 0.55, lim * 0.20, "(below diagonal:\nnoise / short-lived)",
                fontsize=7.5, color="#888", ha="center")

    axes[0].set_ylabel("Death $\\varepsilon$")
    axes[0].legend(loc="lower right", frameon=True, fontsize=8.5,
                   facecolor="white", framealpha=0.95)

    fig.suptitle(f"Example persistence diagrams ($H_0$ components + $H_1$ loops) — "
                 f"track «{track_npy.stem}» (MuQ, 90s)\n"
                 "real trajectory exhibits a salient off-diagonal $H_1$ point; "
                 "surrogates compress $H_1$ toward the diagonal while $H_0$ stays comparable",
                 fontsize=11.5, y=1.05)
    fig.tight_layout()
    save(fig, "fig12_persistence_diagrams.png")


# fig13: Etalon cycle vs cocycle
def fig13_etalon_cycle_vs_cocycle():
    import ripser as ripser_lib
    import dionysus as dio

    rng = np.random.default_rng(0)
    n = 60
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cloud = np.column_stack([np.cos(theta), np.sin(theta)])

    # ---- Ripser cocycle ----
    r = ripser_lib.ripser(cloud, maxdim=1, do_cocycles=True)
    rdgm = r["dgms"][1]
    life = rdgm[:, 1] - rdgm[:, 0]
    rk = int(np.argmax(life))
    rcc = r["cocycles"][1][rk]
    ripser_edges = [(int(e[0]), int(e[1])) for e in rcc[:, :2]]
    ripser_verts = sorted(set(int(v) for e in ripser_edges for v in e))

    # ---- Dionysus filtration ----
    max_r = float(rdgm[rk, 1]) * 1.05
    f = dio.fill_rips(cloud, 2, max_r); f.sort()

    cp = dio.cohomology_persistence(f, prime=2, keep_cocycles=True)
    dgms_co = dio.init_diagrams(cp, f)
    h1_co = [pt for pt in dgms_co[1] if pt.death < float("inf")]
    best_co = max(h1_co, key=lambda p: p.death - p.birth)
    cc = cp.cocycle(best_co.data)
    dio_co_edges, dio_co_verts = [], set()
    for entry in cc:
        s = f[entry.index]; vs = list(s)
        if len(vs) == 2:
            dio_co_edges.append(vs); dio_co_verts.update(vs)
    dio_co_verts = sorted(dio_co_verts)

    m = dio.homology_persistence(f, prime=2, progress=False)
    dgms_ho = dio.init_diagrams(m, f)
    h1_ho = [pt for pt in dgms_ho[1] if pt.death < float("inf")]
    best_ho = max(h1_ho, key=lambda p: p.death - p.birth)
    death_idx = best_ho.data
    birth_idx = m.pair(death_idx)
    chain = m[birth_idx]
    dio_cy_edges, dio_cy_verts = [], set()
    for entry in chain:
        s = f[entry.index]; vs = list(s)
        if len(vs) == 2:
            dio_cy_edges.append(vs); dio_cy_verts.update(vs)
    dio_cy_verts = sorted(dio_cy_verts)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7))
    panels = [
        (axes[0], "Ripser cocycle",    ripser_edges, ripser_verts, C_STEEL,
         "Chords across the disc — cocycle support is NOT a geometric loop"),
        (axes[1], "Dionysus cocycle",  dio_co_edges, dio_co_verts, C_GRAY_M,
         "Different representative, same diagram — still chords"),
        (axes[2], "Dionysus cycle",    dio_cy_edges, dio_cy_verts, C_ACCENT,
         "Clean boundary loop — the geometric repetition contour"),
    ]
    for ax, title, edges, verts, color, subtitle in panels:
        # All cloud points (faded)
        ax.scatter(cloud[:, 0], cloud[:, 1], s=22, c="#DDDDDD",
                   edgecolor="#999", linewidth=0.4, zorder=1)
        # Edges of the representative
        for e in edges:
            ax.plot(cloud[e, 0], cloud[e, 1], color=color, lw=0.7,
                    alpha=0.45, zorder=2)
        # Supporting vertices
        if verts:
            ax.scatter(cloud[verts, 0], cloud[verts, 1], s=42, c=color,
                       edgecolor="black", linewidth=0.5, zorder=3)
        ax.set_aspect("equal")
        ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}  ({len(verts)} verts)", fontsize=11, weight="bold")
        ax.text(0, -1.35, subtitle, ha="center", va="top", fontsize=9, style="italic", color="#444")
        for s in ax.spines.values(): s.set_edgecolor("#999")

    fig.suptitle("Cocycle vs cycle on a 60-point reference circle — same persistence diagram, different geometric supports",
                 fontsize=11.5, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig13_etalon_cycle_vs_cocycle.png")


# fig14: Chromagram and self-similarity
def fig14_chromagram_example():
    import librosa
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    sr = cfg["data"]["sample_rate"]

    candidates = [
        ROOT / "data" / "top50musicSpotify" / "pop" / "pop_29_Something_Just_Like_This_The_Chainsmokers.wav",
        ROOT / "data" / "top50musicSpotify" / "pop" / "pop_01_BIRDS_OF_A_FEATHER_Billie_Eilish.wav",
    ]
    wav_path = next((p for p in candidates if p.exists()), None)
    if wav_path is None:
        for genre in ("electronic", "hip-hop", "pop", "reggae"):
            d = ROOT / "data" / "top50musicSpotify" / genre
            if d.exists():
                cand = sorted(d.glob("*.wav"))
                if cand:
                    wav_path = cand[0]; break
    if wav_path is None:
        print("  skip fig14: no demo .wav available")
        return

    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    total = len(y) / sr
    if total > 90:
        s0 = int((total - 90) / 2 * sr)
        y = y[s0:s0 + int(90 * sr)]
    dur = len(y) / sr

    hop = 512
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop)
    times = np.arange(chroma.shape[1]) * hop / sr

    # Down-sample chroma for SSM to keep it tractable
    target_n = 200
    step = max(1, chroma.shape[1] // target_n)
    chr_ds = chroma[:, ::step]
    t_ds = times[::step]
    # cosine similarity
    cn = chr_ds / (np.linalg.norm(chr_ds, axis=0, keepdims=True) + 1e-9)
    ssm = cn.T @ cn

    # ---- Detect matched repeated sections from the SSM ----
    # Mask diagonal band (within ±5s), then pick top off-diagonal pairs that are
    # well-separated in time so we have visually distinct examples.
    n_ds = ssm.shape[0]
    min_lag = 5.0  # seconds
    sec_per_step = t_ds[1] - t_ds[0] if n_ds > 1 else 1.0
    min_lag_steps = int(np.ceil(min_lag / sec_per_step))

    masked = ssm.copy()
    for i in range(n_ds):
        lo = max(0, i - min_lag_steps); hi = min(n_ds, i + min_lag_steps + 1)
        masked[i, lo:hi] = -np.inf

    # Pick 3 best off-diagonal matches with a minimum separation between picks
    picks = []
    used = np.zeros(n_ds, dtype=bool)
    flat_order = np.argsort(masked, axis=None)[::-1]
    for idx in flat_order:
        i, j = np.unravel_index(idx, masked.shape)
        if j < i:                       # upper triangle only
            continue
        if not np.isfinite(masked[i, j]):
            continue
        # require both endpoints to be in fresh regions
        if used[max(0, i - 3): i + 4].any() or used[max(0, j - 3): j + 4].any():
            continue
        picks.append((i, j))
        used[max(0, i - 3): i + 4] = True
        used[max(0, j - 3): j + 4] = True
        if len(picks) >= 3:
            break
    pick_colors = [C_ACCENT, "#BB8F4C", "#7A4B9B"]

    fig = plt.figure(figsize=(14, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[2, 1.25],
                          hspace=0.45, wspace=0.20)

    # Panel A: waveform with shaded matched pairs
    axw = fig.add_subplot(gs[0, 0])
    tw = np.arange(len(y)) / sr
    axw.fill_between(tw, y, 0, color=C_STEEL, alpha=0.35, linewidth=0)
    axw.plot(tw, y, color=C_NAVY, lw=0.4)
    half = sec_per_step * 1.5    # ±1.5 chroma steps as a visible window
    for k, (i, j) in enumerate(picks):
        ti, tj = t_ds[i], t_ds[j]
        c = pick_colors[k]
        for tc in (ti, tj):
            axw.axvspan(tc - half, tc + half, color=c, alpha=1, lw=0)
        axw.plot([], [], color=c, lw=4, alpha=1, label=f"match #{k+1}: {ti:.1f}s ↔ {tj:.1f}s")
    axw.set_xlim(0, dur)
    axw.set_ylabel("Waveform")
    axw.set_title(f"«{wav_path.stem}» — 90 s center crop  (coloured spans = matched repeated sections)",
                  fontsize=10, loc="left")
    axw.set_xticklabels([])
    axw.grid(alpha=1, linestyle=":")
    axw.legend(frameon=False, loc="lower right", fontsize=8, ncol=3)

    # Panel B: chromagram with same shading
    axc = fig.add_subplot(gs[1, 0], sharex=axw)
    axc.imshow(chroma, aspect="auto", origin="lower",
               extent=[times[0], times[-1], 0, 12],
               cmap="Blues", interpolation="nearest")
    for k, (i, j) in enumerate(picks):
        ti, tj = t_ds[i], t_ds[j]
        c = pick_colors[k]
        for tc in (ti, tj):
            axc.axvspan(tc - half, tc + half, color=c, alpha=0.30, lw=0)
    axc.set_yticks(np.arange(0.5, 12.5))
    axc.set_yticklabels(["C", "C♯", "D", "D♯", "E", "F", "F♯", "G",
                         "G♯", "A", "A♯", "B"], fontsize=7)
    axc.set_xlabel("Time (s)")
    axc.set_ylabel("Pitch class")
    axc.set_title("Chroma (CENS) — note that the matched spans share similar pitch-class patterns",
                  fontsize=10, loc="left")

    # Panel C: self-similarity matrix with circles on the matched pairs
    axs = fig.add_subplot(gs[:, 1])
    im2 = axs.imshow(ssm, origin="lower",
                     extent=[t_ds[0], t_ds[-1], t_ds[0], t_ds[-1]],
                     cmap="Blues", interpolation="nearest",
                     vmin=np.percentile(ssm, 5), vmax=1.0)
    for k, (i, j) in enumerate(picks):
        ti, tj = t_ds[i], t_ds[j]
        c = pick_colors[k]
        # mark both (i,j) and (j,i)
        axs.plot([tj, ti], [ti, tj], "o", color=c, markersize=14,
                 markerfacecolor="none", markeredgewidth=2.0)
        axs.annotate(f"#{k+1}", xy=(tj, ti), xytext=(tj + 2, ti + 2),
                     fontsize=9, color=c, weight="bold")
    axs.set_xlabel("Time (s)"); axs.set_ylabel("Time (s)")
    axs.set_title("Chroma self-similarity matrix\n"
                  "circles = brightest off-diagonal pairs\n"
                  "(these are the repeated sections shown on the left)",
                  fontsize=9.5)
    cb = fig.colorbar(im2, ax=axs, fraction=0.046, pad=0.04)
    cb.set_label("cosine similarity", fontsize=8)

    fig.suptitle("Chromatic recurrence — the audio quantity the H-loop test correlates against persistent $H_1$ cycles",
                 fontsize=11.5, weight="bold", y=1.00)
    save(fig, "fig14_chromagram_example.png")


# Main
def main():
    print(f"Output directory: {OUT.relative_to(ROOT)}")
    fig01_pipeline()
    fig02_h1_max_persistence()
    fig03_h2_within_between()
    fig04_h3_phase_effect()
    fig05_h4_mantel()
    fig06_h5_classification()
    fig07_hloop_progression()
    fig08_popularity_correlations()
    fig09_cycle_vs_cocycle()
    fig10_robustness_panel()
    fig11_persistence_metrics()
    fig12_persistence_diagrams()
    fig13_etalon_cycle_vs_cocycle()
    fig14_chromagram_example()
    print("\nDone.")


if __name__ == "__main__":
    main()
