from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "tables" / "main_results"

METADATA_PATH = ROOT / "data" / "top50_tracks.csv"
LOOP_TABLES = {
    "MuQ": ROOT / "results" / "tables" / "loop_spotify.csv",
    "MIR": ROOT / "results" / "tables" / "loop_spotify_mir.csv",
}

METRICS = [
    "max_persistence",
    "n_loop_vertices",
    "loop_span_sec",
    "distant_loop_sim_mean",
    "distant_random_sim_mean",
    "chroma_excess",
    "repeat_frac",
]


def parse_track_basename(value: str) -> tuple[str, int, str]:
    stem = Path(str(value)).stem
    match = re.match(r"^(.+?)_(\d{2})_(.+)$", stem)
    if not match:
        raise ValueError(f"Cannot parse track basename: {value}")
    return match.group(1), int(match.group(2)), stem


def benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    values = p_values.fillna(1.0).to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    previous = 1.0
    for reverse_position, index in enumerate(order[::-1], start=1):
        rank = len(values) - reverse_position + 1
        current = min(previous, values[index] * len(values) / rank)
        adjusted[index] = current
        previous = current
    return np.minimum(adjusted, 1.0)


def load_matched_tracks() -> pd.DataFrame:
    metadata = pd.read_csv(METADATA_PATH)
    metadata["rank"] = metadata.groupby("genre").cumcount() + 1
    metadata = metadata[
        ["genre", "rank", "name", "artist", "popularity", "duration_ms", "spotify_url", "id"]
    ]

    matched_tables = []
    for space, table_path in LOOP_TABLES.items():
        loop_table = pd.read_csv(table_path)
        parsed = loop_table["basename"].apply(parse_track_basename)
        loop_table["genre"] = [item[0] for item in parsed]
        loop_table["rank"] = [item[1] for item in parsed]
        loop_table["track_key"] = [item[2] for item in parsed]
        loop_table = loop_table.merge(
            metadata,
            on=["genre", "rank"],
            how="left",
            validate="many_to_one",
        )
        loop_table = loop_table[loop_table["popularity"].notna()].copy()
        loop_table["space"] = space
        loop_table["chroma_excess"] = (
            loop_table["distant_loop_sim_mean"] - loop_table["distant_random_sim_mean"]
        )
        matched_tables.append(loop_table)

    return pd.concat(matched_tables, ignore_index=True)


def compute_correlations(tracks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for space, space_table in tracks.groupby("space"):
        for metric in METRICS:
            mask = np.isfinite(space_table["popularity"]) & np.isfinite(space_table[metric])
            if mask.sum() < 3:
                rho, p_value = np.nan, np.nan
                centered_rho, centered_p_value = np.nan, np.nan
            else:
                rho, p_value = spearmanr(
                    space_table.loc[mask, "popularity"],
                    space_table.loc[mask, metric],
                )
                centered_popularity = (
                    space_table.loc[mask, "popularity"]
                    - space_table.loc[mask].groupby("genre")["popularity"].transform("mean")
                )
                centered_metric = (
                    space_table.loc[mask, metric]
                    - space_table.loc[mask].groupby("genre")[metric].transform("mean")
                )
                centered_rho, centered_p_value = spearmanr(centered_popularity, centered_metric)

            rows.append(
                {
                    "space": space,
                    "metric": metric,
                    "n": int(mask.sum()),
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "genre_centered_rho": centered_rho,
                    "genre_centered_p": centered_p_value,
                }
            )

    correlations = pd.DataFrame(rows)
    correlations["p_fdr"] = benjamini_hochberg(correlations["p_value"])
    correlations["genre_centered_p_fdr"] = benjamini_hochberg(
        correlations["genre_centered_p"]
    )
    return correlations


def compute_high_low_tests(tracks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for space, space_table in tracks.groupby("space"):
        space_table = space_table.copy()
        space_table["pop_centered"] = (
            space_table["popularity"]
            - space_table.groupby("genre")["popularity"].transform("mean")
        )
        low_threshold, high_threshold = space_table["pop_centered"].quantile([1 / 3, 2 / 3])
        low_table = space_table[space_table["pop_centered"] <= low_threshold]
        high_table = space_table[space_table["pop_centered"] >= high_threshold]

        for metric in METRICS:
            low_values = low_table[metric].dropna().to_numpy()
            high_values = high_table[metric].dropna().to_numpy()
            if len(low_values) == 0 or len(high_values) == 0:
                p_value = np.nan
            else:
                _, p_value = mannwhitneyu(high_values, low_values, alternative="two-sided")
            rows.append(
                {
                    "space": space,
                    "metric": metric,
                    "low_n": len(low_values),
                    "high_n": len(high_values),
                    "low_mean": np.nanmean(low_values),
                    "high_mean": np.nanmean(high_values),
                    "high_minus_low": np.nanmean(high_values) - np.nanmean(low_values),
                    "mw_p": p_value,
                }
            )

    high_low = pd.DataFrame(rows)
    high_low["mw_p_fdr"] = benjamini_hochberg(high_low["mw_p"])
    return high_low


def compute_per_genre_correlations(tracks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected_metrics = ["max_persistence", "chroma_excess", "n_loop_vertices"]
    for (space, genre), group_table in tracks.groupby(["space", "genre"]):
        for metric in selected_metrics:
            mask = np.isfinite(group_table["popularity"]) & np.isfinite(group_table[metric])
            if mask.sum() < 8:
                rho, p_value = np.nan, np.nan
            else:
                rho, p_value = spearmanr(
                    group_table.loc[mask, "popularity"],
                    group_table.loc[mask, metric],
                )
            rows.append(
                {
                    "space": space,
                    "genre": genre,
                    "metric": metric,
                    "n": int(mask.sum()),
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def compute_summary(tracks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for space, space_table in tracks.groupby("space"):
        rows.append(
            {
                "space": space,
                "n_tracks": len(space_table),
                "genres": ", ".join(sorted(space_table["genre"].unique())),
                "popularity_min": space_table["popularity"].min(),
                "popularity_median": space_table["popularity"].median(),
                "popularity_max": space_table["popularity"].max(),
                "significant_loop_rate": space_table["significant"].mean(),
                "mean_max_persistence": space_table["max_persistence"].mean(),
                "mean_chroma_excess": space_table["chroma_excess"].mean(),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
    high_low: pd.DataFrame,
) -> None:
    report_path = OUT_DIR / "popularity_topology_report.md"
    selected_metrics = ["max_persistence", "chroma_excess", "n_loop_vertices", "loop_span_sec"]
    with report_path.open("w") as report:
        report.write("# Popularity vs topological findings\n\n")
        report.write(
            "Matched tracks: 198 cached Spotify90 tracks with popularity metadata; "
            "genres: electronic, hip-hop, pop, reggae. Pop rank 50 is missing in "
            "metadata for pop and was excluded.\n\n"
        )
        report.write("## Summary\n\n")
        report.write(summary.to_markdown(index=False, floatfmt=".3f"))
        report.write("\n\n## Spearman correlations\n\n")
        report.write(
            correlations[correlations["metric"].isin(selected_metrics)].to_markdown(
                index=False, floatfmt=".3g"
            )
        )
        report.write("\n\n## High-low popularity tertiles\n\n")
        report.write(
            high_low[high_low["metric"].isin(selected_metrics)].to_markdown(
                index=False, floatfmt=".3g"
            )
        )
        report.write("\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tracks = load_matched_tracks()
    summary = compute_summary(tracks)
    correlations = compute_correlations(tracks)
    per_genre = compute_per_genre_correlations(tracks)
    high_low = compute_high_low_tests(tracks)

    tracks.to_csv(OUT_DIR / "popularity_topology_tracks.csv", index=False)
    summary.to_csv(OUT_DIR / "popularity_topology_summary.csv", index=False)
    correlations.to_csv(OUT_DIR / "popularity_topology_correlations.csv", index=False)
    per_genre.to_csv(OUT_DIR / "popularity_topology_by_genre.csv", index=False)
    high_low.to_csv(OUT_DIR / "popularity_topology_high_low.csv", index=False)
    write_report(summary, correlations, high_low)

    print(summary.to_string(index=False))
    print(f"Wrote popularity analysis tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
