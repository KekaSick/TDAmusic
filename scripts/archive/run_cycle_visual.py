import os
import sys
sys.path.insert(0, "src")
import yaml
import csv
import time
import json
import numpy as np
import librosa
import soundfile as sf
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import false_discovery_control
import ripser as ripser_lib
import dionysus as d

import preprocess
import pointcloud

CLIP_SEC = 90
CACHE_DIR = "cache/muq_spotify90"
PLAYER_OUT_DIR = "results/figures/player"

def load_cfg():
    return yaml.safe_load(open("config.yaml"))

def load_tracks_from_csv(csv_path):
    tracks = []
    with open(csv_path, "r") as fh:
        for row in csv.DictReader(fh):
            basename = row["basename"]
            track_base = basename.replace(".wav", "")
            pers = float(row.get("max_persistence", 0))
            if pers > 0:
                tracks.append(track_base)
    return tracks

def extract_topk_representatives(cloud, k=5, max_radius=None):
    n = cloud.shape[0]
    res = {
        "n_points": n,
        "ripser_co": [],
        "dio_co": [],
        "dio_cy": []
    }

    # 1. Ripser cocycle
    r = ripser_lib.ripser(cloud, maxdim=1, do_cocycles=True)
    rdgm = r["dgms"][1]
    
    r_death_max = 0.0
    if len(rdgm) > 0:
        rlife = rdgm[:, 1] - rdgm[:, 0]
        # Sort descending by persistence
        order = np.argsort(rlife)[::-1][:k]
        for rank, rk in enumerate(order):
            birth, death = float(rdgm[rk, 0]), float(rdgm[rk, 1])
            r_death_max = max(r_death_max, death)
            rcc = r["cocycles"][1][rk]
            verts = sorted(set(np.unique(rcc[:, :2].astype(int))))
            edges = [(int(e[0]), int(e[1])) for e in rcc[:, :2]]
            res["ripser_co"].append({
                "rank": rank,
                "birth": birth,
                "death": death,
                "persistence": death - birth,
                "verts": verts,
                "edges": edges,
                "n": len(verts)
            })

    # Dionysus filtration
    if max_radius is None:
        max_radius = r_death_max * 1.1 if r_death_max > 0 else 1.0
    
    t0 = time.time()
    f = d.fill_rips(cloud, 2, max_radius)
    f.sort()
    filt_time = time.time() - t0
    res["max_radius"] = max_radius
    res["filtration_size"] = len(f)
    res["filt_time"] = filt_time

    # 2. Dionysus cocycle
    cp = d.cohomology_persistence(f, prime=2, keep_cocycles=True)
    dgms_co = d.init_diagrams(cp, f)
    h1_co = [pt for pt in dgms_co[1] if pt.death < float("inf")]
    h1_co.sort(key=lambda p: p.death - p.birth, reverse=True)
    for rank, pt in enumerate(h1_co[:k]):
        cc = cp.cocycle(pt.data)
        edges = []
        verts = set()
        for entry in cc:
            s = f[entry.index]
            vs = list(s)
            if len(vs) == 2:
                edges.append(vs)
                verts.update(vs)
        res["dio_co"].append({
            "rank": rank,
            "birth": float(pt.birth),
            "death": float(pt.death),
            "persistence": float(pt.death - pt.birth),
            "verts": sorted(verts),
            "edges": edges,
            "n": len(verts)
        })

    # 3. Dionysus cycle
    m = d.homology_persistence(f, prime=2, progress=False)
    dgms_ho = d.init_diagrams(m, f)
    h1_ho = [pt for pt in dgms_ho[1] if pt.death < float("inf")]
    h1_ho.sort(key=lambda p: p.death - p.birth, reverse=True)
    for rank, pt in enumerate(h1_ho[:k]):
        death_idx = pt.data
        birth_idx = m.pair(death_idx)
        chain = m[birth_idx]
        edges = []
        verts = set()
        for entry in chain:
            s = f[entry.index]
            vs = list(s)
            if len(vs) == 2:
                edges.append(vs)
                verts.update(vs)
        res["dio_cy"].append({
            "rank": rank,
            "birth": float(pt.birth),
            "death": float(pt.death),
            "persistence": float(pt.death - pt.birth),
            "verts": sorted(verts),
            "edges": edges,
            "n": len(verts)
        })

    return res

def vertices_to_seconds(vertex_ids, sub_indices, takens_starts, window, target_fps):
    seconds = []
    for v in vertex_ids:
        takens_row = sub_indices[v]
        start_frame = takens_starts[takens_row]
        sec = (start_frame + window / 2) / target_fps
        seconds.append(sec)
    return np.sort(seconds)

def _mean_distant_sim(chroma_vecs, times, distant_thresh):
    sims = []
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if abs(times[i] - times[j]) >= distant_thresh:
                sims.append(1.0 - cosine_dist(chroma_vecs[i], chroma_vecs[j]))
    return float(np.mean(sims)) if sims else np.nan, len(sims)

def chroma_test(wav, sr, seconds, window_sec, hop_length=512, distant_thresh=5.0, n_perm=2000, seed=42):
    chroma_full = librosa.feature.chroma_cens(y=wav, sr=sr, hop_length=hop_length)
    chroma_fps = sr / hop_length
    n_chroma = chroma_full.shape[1]
    clip_dur = len(wav) / sr
    half_win = window_sec / 2

    def get_chroma(sec):
        sf = max(0, int((sec - half_win) * chroma_fps))
        ef = min(n_chroma, int((sec + half_win) * chroma_fps) + 1)
        if ef <= sf:
            ef = sf + 1
        return chroma_full[:, sf:ef].mean(axis=1)

    k = len(seconds)
    if k == 0:
        return {"p_value": np.nan}
        
    observed_stat, n_distant = _mean_distant_sim(
        np.array([get_chroma(s) for s in seconds]), seconds, distant_thresh)

    if np.isnan(observed_stat) or n_distant < 1:
        return {"p_value": np.nan}

    all_times = np.linspace(window_sec / 2, clip_dur - window_sec / 2, max(400, k + 10))
    all_chroma = np.array([get_chroma(s) for s in all_times])

    rng = np.random.default_rng(seed)
    null_stats = []
    for _ in range(n_perm):
        idx = rng.choice(len(all_times), size=k, replace=False)
        rand_times = all_times[idx]
        rand_chroma = all_chroma[idx]
        stat, cnt = _mean_distant_sim(rand_chroma, rand_times, distant_thresh)
        if not np.isnan(stat) and cnt >= 1:
            null_stats.append(stat)

    null_stats = np.array(null_stats)
    if len(null_stats) == 0:
        return {"p_value": np.nan}

    p_value = float((np.sum(null_stats >= observed_stat) + 1) / (len(null_stats) + 1))
    return {"p_value": p_value}

def main():
    cfg = load_cfg()
    sr = cfg["data"]["sample_rate"]
    target_fps = cfg["common"]["target_fps"]
    hop_length = cfg["spaces"]["mir"]["hop_length"]
    pc_cfg = cfg["pointcloud"].copy()
    pc_cfg["takens_pca_dim"] = None
    window = pc_cfg["window"]
    window_sec = window / target_fps
    k_loops = 5

    os.makedirs(PLAYER_OUT_DIR, exist_ok=True)

    REDUCER_PATH = "cache/pca_reducer_muq_spotify90.joblib"
    if not os.path.exists(REDUCER_PATH):
        print(f"ERROR: {REDUCER_PATH} not found.")
        sys.exit(1)
    reducer = preprocess.PCAReducer.load(REDUCER_PATH)

    csv_path = os.path.join(cfg["paths"]["results"], "tables", "loop_spotify.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)
    all_tracks = load_tracks_from_csv(csv_path)

    # Filter to only interesting tracks for quick visualization
    target_tracks = [
        "electronic_16_Midas_Maribou_State", 
        "pop_47_Love_Me_Lil_Wayne",
        "hip-hop_13_Love_Me_Lil_Wayne",
        "pop_40_Closer_The_Chainsmokers",
        "hip-hop_21_Passionfruit_Drake",
        "pop_01_BIRDS_OF_A_FEATHER_Billie_Eilish",
        "hip-hop_07_HUMBLE__Kendrick_Lamar"
    ]
    all_tracks = [t for t in all_tracks if t in target_tracks]

    # Hardcode n_perm to 1000 for the visual script to save time, since precision is less critical for the player
    n_perm = 1000

    results_data = [] # To hold data for FDR calculation and JSON saving

    # First pass: computation and data gathering
    for idx, track_base in enumerate(all_tracks):
        print(f"\n[{idx+1}/{len(all_tracks)}] {track_base}")

        cache_path = os.path.join(CACHE_DIR, f"{track_base}.npy")
        if not os.path.exists(cache_path):
            print(f"  SKIP: cache not found")
            continue
        emb = np.load(cache_path)
        x_pca = preprocess.prepare_track(emb, cfg["spaces"]["muq"], cfg["common"], reducer)
        cloud, takens_starts, sub_indices = pointcloud.build_cloud_with_indices(x_pca, pc_cfg)

        t0 = time.time()
        rep = extract_topk_representatives(cloud, k=k_loops)
        elapsed = time.time() - t0
        print(f"  Loops: ripser={len(rep['ripser_co'])} dio_co={len(rep['dio_co'])} dio_cy={len(rep['dio_cy'])} in {elapsed:.1f}s")

        # Load audio for chroma
        wav_path = None
        track_genre = track_base.split("_")[0]
        candidate = os.path.join("data/top50musicSpotify", track_genre, track_base + ".wav")
        if os.path.exists(candidate):
            wav_path = candidate
        else:
            for genre_dir in sorted(os.listdir("data/top50musicSpotify")):
                candidate = os.path.join("data/top50musicSpotify", genre_dir, track_base + ".wav")
                if os.path.exists(candidate):
                    wav_path = candidate
                    break
        if wav_path is None:
            print(f"  SKIP: wav not found")
            continue

        wav_full, _ = librosa.load(wav_path, sr=sr, mono=True)
        total_sec = len(wav_full) / sr
        if total_sec > CLIP_SEC:
            start = int((total_sec - CLIP_SEC) / 2 * sr)
            wav = wav_full[start:start + int(CLIP_SEC * sr)]
        else:
            wav = wav_full
        
        cloud_c = cloud - cloud.mean(axis=0)
        _, _, Vt = np.linalg.svd(cloud_c, full_matrices=False)
        proj = cloud_c @ Vt[:2].T
        cloud2d = [[round(float(x), 4), round(float(y), 4)] for x, y in proj]

        track_data = {
            "track": track_base,
            "clip_seconds": CLIP_SEC,
            "window_sec": window_sec,
            "methods": {
                "ripser_co": {"loops": []},
                "dio_co": {"loops": []},
                "dio_cy": {"loops": []}
            },
            "cloud2d": cloud2d,
            "wav": wav # Temporary store, will be removed before json dump
        }

        for method in ["ripser_co", "dio_co", "dio_cy"]:
            for loop in rep[method]:
                verts_sorted = sorted(int(v) for v in loop["verts"])
                secs_unsorted = []
                for v in verts_sorted:
                    takens_row = sub_indices[v]
                    start_frame = takens_starts[takens_row]
                    sec = (start_frame + window / 2) / target_fps
                    secs_unsorted.append(float(sec))
                
                span = max(secs_unsorted) - min(secs_unsorted) if len(secs_unsorted) > 1 else 0.0
                ct = chroma_test(wav, sr, secs_unsorted, window_sec, hop_length, n_perm=n_perm)
                
                p_val = ct.get("p_value", None)
                p_val_raw = float(p_val) if p_val is not None and not np.isnan(p_val) else None
                
                track_data["methods"][method]["loops"].append({
                    "rank": int(loop["rank"]),
                    "persistence": float(loop["persistence"]),
                    "n": int(loop["n"]),
                    "span_sec": float(span),
                    "p_value_raw": p_val_raw,
                    "vertices": verts_sorted,
                    "vertex_seconds": secs_unsorted
                })
        
        results_data.append(track_data)
        
    # Second pass: FDR correction
    print("\n--- Applying FDR Correction ---")
    raw_p_list = [] # List of tuples: (track_idx, method, loop_idx, p_raw)
    
    for t_idx, track_data in enumerate(results_data):
        for method in ["ripser_co", "dio_co", "dio_cy"]:
            for l_idx, loop in enumerate(track_data["methods"][method]["loops"]):
                p_raw = loop["p_value_raw"]
                if p_raw is not None:
                    raw_p_list.append((t_idx, method, l_idx, p_raw))
                    
    if raw_p_list:
        raw_p_arr = np.array([x[3] for x in raw_p_list])
        fdr_p_arr = false_discovery_control(raw_p_arr, method='bh')
        
        for (t_idx, method, l_idx, _), fdr_p in zip(raw_p_list, fdr_p_arr):
            results_data[t_idx]["methods"][method]["loops"][l_idx]["p_value"] = float(fdr_p)
            
    # For loops with NaN p_value, assign None
    for track_data in results_data:
        for method in ["ripser_co", "dio_co", "dio_cy"]:
            for loop in track_data["methods"][method]["loops"]:
                pval = loop.get("p_value", None)
                if pval is None or np.isnan(pval):
                    loop["p_value"] = None
                else:
                    loop["p_value"] = float(pval)
                if "p_value_raw" in loop:
                    del loop["p_value_raw"] # Clean up temporary key

    # Calculate statistics for Metric B
    sig_count_any = {"ripser_co": 0, "dio_co": 0, "dio_cy": 0}
    total_valid = {"ripser_co": 0, "dio_co": 0, "dio_cy": 0}
    
    for track_data in results_data:
        for method in ["ripser_co", "dio_co", "dio_cy"]:
            loops = track_data["methods"][method]["loops"]
            if len(loops) > 0:
                total_valid[method] += 1
                has_sig = any(l["p_value"] is not None and l["p_value"] < 0.05 for l in loops)
                if has_sig:
                    sig_count_any[method] += 1

    print("\nMetric B: Fraction of tracks with AT LEAST ONE significant loop in Top-5")
    for method in ["ripser_co", "dio_co", "dio_cy"]:
        if total_valid[method] > 0:
            pct = 100 * sig_count_any[method] / total_valid[method]
            print(f"  {method}: {sig_count_any[method]}/{total_valid[method]} ({pct:.1f}%)")

    # Save JSON and WAV
    print("\n--- Saving Player Files ---")
    for track_data in results_data:
        wav = track_data.pop("wav")
        track_base = track_data["track"]
        
        # Save JSON
        json_path = os.path.join(PLAYER_OUT_DIR, f"{track_base}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(track_data, f, indent=2)
            
        # Save WAV
        wav_path = os.path.join(PLAYER_OUT_DIR, f"{track_base}_crop.wav")
        sf.write(wav_path, wav, sr)
        
        print(f"Saved: {json_path}")
        print(f"Saved: {wav_path}")

    print("\nDone!")

if __name__ == "__main__":
    main()
