"""
run_radius_robustness.py
------------------------
Two-pronged robustness check for the max_radius multiplier in cycle extraction.

APPROACH 1 (Ripser — seconds): verify that the persistence DIAGRAM (birth, death)
of the most persistent H1 feature is invariant to max_radius cutoff.
→ cycle_radius_robustness_diagram.csv

APPROACH 2A (Dionysus on subsampled cloud — minutes): verify that the cycle
REPRESENTATIVE (vertex set, temporal span) is stable at multipliers 1.1 and 1.3
on a maxmin-subsampled cloud (robustness.max_points from config.yaml).
→ cycle_radius_robustness_representative.csv

    .venv/bin/python scripts/run_radius_robustness.py
"""
import os
import sys
sys.path.insert(0, "src")
import yaml
import csv
import time
import numpy as np
import ripser as ripser_lib
import dionysus as d

import preprocess
import pointcloud


# ======================================================================
# Config
# ======================================================================

CACHE_DIR = "cache/muq_spotify90"


def load_cfg():
    return yaml.safe_load(open("config.yaml"))


def load_tracks_from_csv(csv_path):
    """Load ALL track basenames + persistence from loop_spotify.csv."""
    tracks = []
    with open(csv_path, "r") as fh:
        for row in csv.DictReader(fh):
            basename = row["basename"]
            track_base = basename.replace(".wav", "")
            pers = float(row.get("max_persistence", 0))
            if pers > 0:
                tracks.append((track_base, pers))
    return tracks


def vertices_to_seconds(vertex_ids, sub_indices, takens_starts,
                        window, target_fps):
    """Map cloud vertex IDs to time (center of Takens window)."""
    seconds = []
    for v in vertex_ids:
        takens_row = sub_indices[v]
        start_frame = takens_starts[takens_row]
        sec = (start_frame + window / 2) / target_fps
        seconds.append(sec)
    return np.sort(seconds)


def jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 1.0


# ======================================================================
# APPROACH 1: Ripser diagram stability (fast)
# ======================================================================

def approach1_ripser_diagram(test_tracks, clouds, cfg):
    """Check if (birth, death) of most persistent H1 is invariant to thresh."""
    rob_cfg = cfg["robustness"]
    multipliers = rob_cfg["diagram_multipliers"] + [np.inf]  # add no-cutoff

    print(f"\n{'='*70}")
    print("APPROACH 1: Ripser diagram stability")
    print(f"  Multipliers: {multipliers}")
    print(f"{'='*70}")

    rows = []

    for track_base, cloud in zip(test_tracks, clouds):
        print(f"\n  Track: {track_base} ({cloud.shape[0]} points)")

        # Baseline: no threshold (full diagram)
        r_full = ripser_lib.ripser(cloud, maxdim=1, do_cocycles=False)
        rdgm = r_full["dgms"][1]
        if len(rdgm) == 0:
            print(f"    SKIP: no H1")
            continue
        rlife = rdgm[:, 1] - rdgm[:, 0]
        rk = int(np.argmax(rlife))
        baseline_birth = float(rdgm[rk, 0])
        baseline_death = float(rdgm[rk, 1])
        print(f"    Baseline (no thresh): birth={baseline_birth:.6f}, "
              f"death={baseline_death:.6f}, "
              f"pers={baseline_death - baseline_birth:.6f}")

        for mult in multipliers:
            thresh = baseline_death * mult if mult != np.inf else np.inf
            t0 = time.time()
            r = ripser_lib.ripser(cloud, maxdim=1, do_cocycles=False,
                                  thresh=thresh)
            elapsed = time.time() - t0
            dgm = r["dgms"][1]

            if len(dgm) == 0:
                print(f"    mult={mult}: NO H1 ({elapsed:.2f}s)")
                rows.append({
                    "track": track_base,
                    "multiplier": mult,
                    "thresh": thresh,
                    "birth": np.nan,
                    "death": np.nan,
                    "persistence": np.nan,
                    "matches_baseline": False,
                    "time_sec": elapsed,
                })
                continue

            # Filter for finite-death features (thresh truncation creates
            # spurious inf-death features that mask the real most-persistent)
            finite_mask = np.isfinite(dgm[:, 1])
            dgm_finite = dgm[finite_mask]

            if len(dgm_finite) == 0:
                # All features are truncated at this thresh — too small
                print(f"    mult={mult}: all {len(dgm)} features truncated "
                      f"to inf-death ({elapsed:.2f}s)")
                rows.append({
                    "track": track_base,
                    "multiplier": mult,
                    "thresh": thresh if thresh != np.inf else "inf",
                    "birth": np.nan,
                    "death": np.nan,
                    "persistence": np.nan,
                    "matches_baseline": False,
                    "n_total_h1": len(dgm),
                    "n_finite_h1": 0,
                    "time_sec": elapsed,
                })
                continue

            life = dgm_finite[:, 1] - dgm_finite[:, 0]
            k = int(np.argmax(life))
            b, dd = float(dgm_finite[k, 0]), float(dgm_finite[k, 1])
            match = (abs(b - baseline_birth) < 1e-6
                     and abs(dd - baseline_death) < 1e-6)

            print(f"    mult={mult}: birth={b:.6f}, death={dd:.6f}, "
                  f"pers={dd-b:.6f}, match={'✓' if match else '✗'} "
                  f"(finite={finite_mask.sum()}/{len(dgm)}, {elapsed:.2f}s)")

            rows.append({
                "track": track_base,
                "multiplier": mult,
                "thresh": thresh if thresh != np.inf else "inf",
                "birth": b,
                "death": dd,
                "persistence": dd - b,
                "matches_baseline": match,
                "n_total_h1": len(dgm),
                "n_finite_h1": int(finite_mask.sum()),
                "time_sec": elapsed,
            })

    return rows


# ======================================================================
# APPROACH 2A: Dionysus representative on subsampled cloud
# ======================================================================

def approach2a_dionysus_representative(test_tracks, clouds_full,
                                       takens_starts_list,
                                       sub_indices_full_list,
                                       cfg):
    """Check cycle representative stability on maxmin-subsampled cloud."""
    rob_cfg = cfg["robustness"]
    max_points = rob_cfg["max_points"]
    representative_multipliers = rob_cfg["representative_multipliers"]
    span_tolerance = rob_cfg["span_tolerance"]
    target_fps = cfg["common"]["target_fps"]
    window = cfg["pointcloud"]["window"]
    seed = cfg["seed"]

    print(f"\n{'='*70}")
    print("APPROACH 2A: Dionysus representative stability (subsampled)")
    print(f"  Max points: {max_points}")
    print(f"  Multipliers: {representative_multipliers}")
    print(f"  Span tolerance (CV): {span_tolerance}")
    print(f"{'='*70}")

    rows = []

    for i, track_base in enumerate(test_tracks):
        cloud_full = clouds_full[i]
        takens_starts = takens_starts_list[i]
        sub_indices_full = sub_indices_full_list[i]

        print(f"\n  Track: {track_base} (full: {cloud_full.shape[0]} points)")

        # Maxmin subsample
        if cloud_full.shape[0] > max_points:
            cloud, sub_idx_local = pointcloud._maxmin_with_indices(
                cloud_full, max_points, seed=seed)
            sub_indices = sub_indices_full[sub_idx_local]
            print(f"    Subsampled to {cloud.shape[0]} points (maxmin)")
        else:
            cloud = cloud_full
            sub_indices = sub_indices_full

        # Ripser on subsampled cloud for r_death
        r = ripser_lib.ripser(cloud, maxdim=1, do_cocycles=False)
        rdgm = r["dgms"][1]
        if len(rdgm) == 0:
            print(f"    SKIP: no H1")
            continue
        rlife = rdgm[:, 1] - rdgm[:, 0]
        rk = int(np.argmax(rlife))
        r_death = float(rdgm[rk, 1])
        print(f"    Ripser r_death={r_death:.4f}")

        track_results = {}
        for mult in representative_multipliers:
            max_radius = r_death * mult
            print(f"\n    Multiplier={mult}, max_radius={max_radius:.4f}")

            t0 = time.time()
            print(f"      Building fill_rips(n={cloud.shape[0]}, dim=2, "
                  f"r={max_radius:.4f})...", flush=True)
            f = d.fill_rips(cloud, 2, max_radius)
            f.sort()
            filt_time = time.time() - t0
            print(f"      Filtration: {len(f)} simplices in {filt_time:.1f}s",
                  flush=True)

            t0 = time.time()
            m = d.homology_persistence(f, prime=2, progress=False)
            dgms = d.init_diagrams(m, f)
            pers_time = time.time() - t0
            print(f"      Persistence done in {pers_time:.1f}s", flush=True)

            h1 = [pt for pt in dgms[1] if pt.death < float("inf")]
            if not h1:
                print(f"      SKIP: no H1")
                track_results[mult] = None
                continue

            best = max(h1, key=lambda p: p.death - p.birth)
            death_idx = best.data
            birth_idx = m.pair(death_idx)
            chain = m[birth_idx]

            verts = set()
            for entry in chain:
                s = f[entry.index]
                vs = list(s)
                if len(vs) == 2:
                    verts.update(vs)

            secs = vertices_to_seconds(
                sorted(verts), sub_indices, takens_starts, window, target_fps)
            span = float(secs.max() - secs.min()) if len(secs) > 1 else 0.0

            track_results[mult] = {
                "verts": sorted(verts),
                "n_verts": len(verts),
                "span_sec": span,
                "persistence": float(best.death - best.birth),
                "filtration_size": len(f),
            }
            elapsed = filt_time + pers_time
            print(f"      n_verts={len(verts)}, span={span:.2f}s, "
                  f"total={elapsed:.1f}s")

        # Compare
        valid = {m: r for m, r in track_results.items() if r is not None}
        if len(valid) < 2:
            for mult, res in valid.items():
                rows.append({
                    "track": track_base,
                    "multiplier": mult,
                    "n_verts": res["n_verts"],
                    "span_sec": res["span_sec"],
                    "persistence": res["persistence"],
                    "filtration_size": res["filtration_size"],
                    "jaccard_vs_first": 1.0,
                    "span_cv": np.nan,
                    "span_stable": "N/A",
                })
            continue

        spans = [r["span_sec"] for r in valid.values()]
        span_mean = np.mean(spans)
        span_std = np.std(spans)
        span_cv = span_std / span_mean if span_mean > 0 else 0.0
        span_stable = span_cv < span_tolerance

        first_mult = representative_multipliers[0]
        first_verts = (set(valid[first_mult]["verts"])
                       if first_mult in valid else set())

        print(f"\n    --- Comparison ---")
        print(f"    Spans: {[f'{s:.2f}' for s in spans]}")
        print(f"    CV={span_cv:.4f}, STABLE={'YES' if span_stable else 'NO'}")

        for mult in representative_multipliers:
            res = track_results.get(mult)
            if res is None:
                rows.append({
                    "track": track_base,
                    "multiplier": mult,
                    "n_verts": 0,
                    "span_sec": np.nan,
                    "persistence": np.nan,
                    "filtration_size": 0,
                    "jaccard_vs_first": np.nan,
                    "span_cv": span_cv,
                    "span_stable": "N/A",
                })
                continue

            j = (jaccard(first_verts, set(res["verts"]))
                 if first_verts else np.nan)
            print(f"    mult={mult}: Jaccard={j:.4f}, "
                  f"n_verts={res['n_verts']}, span={res['span_sec']:.2f}s")

            rows.append({
                "track": track_base,
                "multiplier": mult,
                "n_verts": res["n_verts"],
                "span_sec": res["span_sec"],
                "persistence": res["persistence"],
                "filtration_size": res["filtration_size"],
                "jaccard_vs_first": j,
                "span_cv": span_cv,
                "span_stable": "YES" if span_stable else "NO",
            })

    return rows


# ======================================================================
# Main
# ======================================================================

def main():
    cfg = load_cfg()
    target_fps = cfg["common"]["target_fps"]
    pc_cfg = cfg["pointcloud"].copy()
    pc_cfg["takens_pca_dim"] = None

    rob_cfg = cfg["robustness"]
    n_test_tracks = rob_cfg["n_test_tracks"]

    # Load PCA reducer
    REDUCER_PATH = "cache/pca_reducer_muq_spotify90.joblib"
    if not os.path.exists(REDUCER_PATH):
        print(f"ERROR: {REDUCER_PATH} not found.")
        sys.exit(1)
    reducer = preprocess.PCAReducer.load(REDUCER_PATH)
    print(f"PCA dim={reducer.dim}, explained={reducer.explained:.4f}")

    # Load tracks sorted by persistence → pick top N
    csv_path = os.path.join(cfg["paths"]["results"], "tables", "loop_spotify.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)
    all_tracks = load_tracks_from_csv(csv_path)
    all_tracks.sort(key=lambda x: x[1], reverse=True)

    test_tracks = []
    clouds = []
    takens_starts_list = []
    sub_indices_list = []

    for track_base, pers in all_tracks:
        cache_path = os.path.join(CACHE_DIR, f"{track_base}.npy")
        if not os.path.exists(cache_path):
            continue
        emb = np.load(cache_path)
        x_pca = preprocess.prepare_track(
            emb, cfg["spaces"]["muq"], cfg["common"], reducer)
        cloud, takens_starts, sub_indices = \
            pointcloud.build_cloud_with_indices(x_pca, pc_cfg)

        test_tracks.append(track_base)
        clouds.append(cloud)
        takens_starts_list.append(takens_starts)
        sub_indices_list.append(sub_indices)
        print(f"  Loaded {track_base}: cloud {cloud.shape}")

        if len(test_tracks) >= n_test_tracks:
            break

    print(f"\nSelected {len(test_tracks)} test tracks")

    out_tables = os.path.join(cfg["paths"]["results"], "tables")
    os.makedirs(out_tables, exist_ok=True)

    # ---- APPROACH 1: Ripser diagram stability ----
    rows1 = approach1_ripser_diagram(test_tracks, clouds, cfg)
    csv1 = os.path.join(out_tables, "cycle_radius_robustness_diagram.csv")
    if rows1:
        with open(csv1, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows1[0].keys()))
            w.writeheader()
            w.writerows(rows1)
        print(f"\n→ {csv1}")

    # ---- APPROACH 2A: Dionysus representative on subsampled cloud ----
    rows2 = approach2a_dionysus_representative(
        test_tracks, clouds, takens_starts_list, sub_indices_list, cfg)
    csv2 = os.path.join(out_tables, "cycle_radius_robustness_representative.csv")
    if rows2:
        with open(csv2, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows2[0].keys()))
            w.writeheader()
            w.writerows(rows2)
        print(f"\n→ {csv2}")

    # ---- COMBINED VERDICT ----
    print(f"\n{'='*70}")
    print("COMBINED VERDICT")
    print(f"{'='*70}")

    # Approach 1 verdict
    if rows1:
        all_match = all(r["matches_baseline"] for r in rows1)
        n_match = sum(1 for r in rows1 if r["matches_baseline"])
        print(f"\n  Approach 1 (Ripser diagram):")
        print(f"    {n_match}/{len(rows1)} tests: diagram matches baseline")
        if all_match:
            print(f"    ✓ The persistence invariant (birth, death) is COMPLETELY")
            print(f"      invariant to the max_radius cutoff.")
        else:
            mismatches = [r for r in rows1 if not r["matches_baseline"]]
            print(f"    ✗ {len(mismatches)} mismatches found")

    # Approach 2A verdict
    if rows2:
        stable_rows = [r for r in rows2 if r["span_stable"] not in ("N/A",)]
        if stable_rows:
            all_stable = all(r["span_stable"] == "YES" for r in stable_rows)
            print(f"\n  Approach 2A (Dionysus representative, subsampled):")
            if all_stable:
                print(f"    ✓ Cycle representative span is STABLE across "
                      f"tested multipliers")
            else:
                unstable = [r for r in stable_rows
                            if r["span_stable"] != "YES"]
                print(f"    ✗ {len(unstable)} unstable cases")

    # Manuscript paragraph
    if rows1 and all(r["matches_baseline"] for r in rows1):
        print(f"\n  SUGGESTED MANUSCRIPT TEXT:")
        print(f"  ─────────────────────────")
        mults = sorted(set(r["multiplier"] for r in rows1))
        print(f"  \"The persistence invariant (birth, death) of the most")
        print(f"  persistent H1 feature was verified to be completely invariant")
        print(f"  to the Rips filtration radius cutoff, tested at multipliers")
        print(f"  {mults} of the death value (N={len(test_tracks)} tracks).")
        print(f"  The representative cycle was extracted at the standard")
        print(f"  1.1× cutoff. This confirms that the topological signal is")
        print(f"  intrinsic to the data and not an artifact of the radius")
        print(f"  parameter choice.\"")

    print("\nDone!")


if __name__ == "__main__":
    main()
