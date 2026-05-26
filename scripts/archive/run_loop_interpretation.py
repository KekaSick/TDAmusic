"""
run_loop_interpretation.py
--------------------------
Семантическая интерпретация H1-петель: «петля = музыкальный повтор».

Гипотеза: вершины самой персистентной H1-петли (representative cocycle)
соответствуют моментам трека, содержащим ПОВТОРЯЮЩИЙСЯ музыкальный материал.
Проверка: chroma-сходство далёких пар петли vs далёких случайных пар того же
трека (Mann-Whitney).

Канал: MuQ-10 (единственный с жанро-информативной топологией).

    .venv/bin/python scripts/run_loop_interpretation.py
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import embed_spaces
import preprocess
import pointcloud
import persistence


# ======================================================================
# 0. Helpers
# ======================================================================

def load_cfg():
    cfg = yaml.safe_load(open("config.yaml"))
    return cfg


def get_pca_reducer(cfg, space, files, tr_idx):
    """Обучить PCAReducer на train-сплите (как в основном пайплайне)."""
    reducer = preprocess.PCAReducer(
        dim=cfg["common"]["pca_dim"],
        standardize=cfg["common"].get("standardize_features", True),
        normalize=cfg["common"]["normalize"])
    feats = [embed_spaces.extract(files[i], space, cfg) for i in tr_idx]
    reducer.fit(feats)
    return reducer


def cocycle_vertices_to_seconds(cocycle, sub_indices, takens_starts,
                                 window, target_fps):
    """Cocycle вершины → центры окон (секунды).

    cocycle: ndarray (n_edges, 3) — [i, j, val] из ripser.
    sub_indices: индексы maxmin-подвыборки в полном Такенс-облаке.
    takens_starts: стартовые кадры каждого окна Такенса в PCA-ряду.
    window: длина окна Такенса (кадров).
    target_fps: частота PCA-ряда (Гц).

    Возвращает: sorted unique seconds (np.ndarray).
    """
    vertex_ids = np.unique(cocycle[:, :2].astype(int).ravel())
    seconds = []
    for v in vertex_ids:
        takens_row = sub_indices[v]          # строка в полном Такенс-облаке
        start_frame = takens_starts[takens_row]  # стартовый кадр окна
        center_sec = (start_frame + window / 2) / target_fps
        seconds.append(center_sec)
    return np.sort(seconds), vertex_ids


def extract_chroma_at_windows(wav, sr, seconds, window_sec, hop_length=512):
    """Извлечь усреднённый chroma вектор для каждого момента времени.

    Для каждой секунды берём окно [sec - window_sec/2, sec + window_sec/2]
    и усредняем chroma_cens по нему.

    Возвращает: (n_moments, 12) — chroma-вектора.
    """
    # Полный chromagram
    chroma_full = librosa.feature.chroma_cens(y=wav, sr=sr, hop_length=hop_length)
    # (12, T_chroma)
    chroma_fps = sr / hop_length
    n_chroma = chroma_full.shape[1]

    result = []
    half_win = window_sec / 2.0
    for sec in seconds:
        start_frame = max(0, int((sec - half_win) * chroma_fps))
        end_frame = min(n_chroma, int((sec + half_win) * chroma_fps) + 1)
        if end_frame <= start_frame:
            end_frame = start_frame + 1
        chunk = chroma_full[:, start_frame:end_frame]
        result.append(chunk.mean(axis=1))
    return np.array(result)


def pairwise_cosine_sim(vecs):
    """Попарное косинусное сходство. Возвращает (n, n) матрицу."""
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(vecs)


def similarity_pairs(sim_matrix, indices_a, indices_b=None):
    """Извлечь сходства для указанных пар.

    Если indices_b=None, все уникальные пары из indices_a.
    Иначе — все пары (a, b) из декартова произведения.
    """
    sims = []
    if indices_b is None:
        for i in range(len(indices_a)):
            for j in range(i + 1, len(indices_a)):
                sims.append(sim_matrix[indices_a[i], indices_a[j]])
    else:
        for a in indices_a:
            for b in indices_b:
                if a != b:
                    sims.append(sim_matrix[a, b])
    return np.array(sims)


# ======================================================================
# 1. Выбор трека с максимальной H1-persistence
# ======================================================================

def find_best_track(cfg, files, meta, reducer, genres=None, max_tracks=30):
    """Перебрать треки из указанных жанров, найти max H1 persistence."""
    if genres is None:
        genres = ["classical", "disco", "pop"]
    pc_cfg = cfg["pointcloud"].copy()
    pc_cfg["takens_pca_dim"] = None

    best_pers = -1
    best_info = None

    candidates = meta[meta["genre"].isin(genres)]
    # Берём до max_tracks из каждого жанра
    rng = np.random.default_rng(cfg["seed"])
    sample_idx = []
    for g in genres:
        g_idx = candidates[candidates["genre"] == g].index.tolist()
        if len(g_idx) > max_tracks:
            g_idx = rng.choice(g_idx, max_tracks, replace=False).tolist()
        sample_idx.extend(g_idx)

    print(f"Scanning {len(sample_idx)} tracks from {genres} ...")
    for idx in sample_idx:
        filepath = files[idx]
        x_raw = embed_spaces.extract(filepath, "muq", cfg)
        x = preprocess.prepare_track(x_raw, cfg["spaces"]["muq"],
                                      cfg["common"], reducer)
        cloud, takens_starts, sub_indices = \
            pointcloud.build_cloud_with_indices(x, pc_cfg)
        res = persistence.compute_diagrams(cloud, maxdim=1, do_cocycles=True)
        dgm1 = res["dgms"][1]
        if len(dgm1) == 0:
            continue
        life = dgm1[:, 1] - dgm1[:, 0]
        mp = float(life.max())
        if mp > best_pers:
            best_pers = mp
            best_info = {
                "idx": idx,
                "filepath": filepath,
                "genre": meta.iloc[idx]["genre"] if "genre" in meta.columns else "?",
                "filename": meta.iloc[idx]["filename"],
                "max_persistence": mp,
                "cloud": cloud,
                "takens_starts": takens_starts,
                "sub_indices": sub_indices,
                "ripser_result": res,
                "x_pca": x,
            }
            print(f"  New best: {best_info['filename']} "
                  f"(genre={best_info['genre']}, max_pers={mp:.4f})")

    return best_info


# ======================================================================
# 2. Синтетический тест маппинга
# ======================================================================

def verify_mapping_synthetic():
    """Проверка цепочки индексов на синтетике (круговая траектория)."""
    print("\n=== Синтетическая проверка маппинга ===")
    T, d = 400, 8
    target_fps = 25.0
    total_sec = T / target_fps  # 16 сек

    # Круговая траектория: 4 полных оборота за 16 секунд (период = 4 сек)
    t = np.linspace(0, 8 * np.pi, T)
    x = np.zeros((T, d))
    x[:, 0], x[:, 1] = np.cos(t), np.sin(t)
    x[:, 2:] = 0.02 * np.random.default_rng(0).standard_normal((T, d - 2))

    pc_cfg = {"method": "takens", "window": 16, "stride": 4,
              "n_points": 150, "subsample": "maxmin"}

    cloud, takens_starts, sub_indices = pointcloud.build_cloud_with_indices(
        x, pc_cfg)
    res = persistence.compute_diagrams(cloud, maxdim=1, do_cocycles=True)
    dgm1 = res["dgms"][1]

    if len(dgm1) == 0:
        print("  WARNING: нет H1 в синтетике!")
        return False

    life = dgm1[:, 1] - dgm1[:, 0]
    k = int(np.argmax(life))
    cocycle = res["cocycles"][1][k]

    seconds, vertices = cocycle_vertices_to_seconds(
        cocycle, sub_indices, takens_starts, pc_cfg["window"], target_fps)

    print(f"  Cocycle вершин: {len(vertices)}, рёбер: {len(cocycle)}")
    print(f"  Секунды: {seconds}")
    print(f"  Диапазон: [{seconds.min():.1f}, {seconds.max():.1f}] "
          f"из [0, {total_sec:.1f}]")

    # Все секунды должны быть в пределах [0, total_sec]
    assert seconds.min() >= 0, f"отрицательная секунда: {seconds.min()}"
    assert seconds.max() <= total_sec + 1e-6, \
        f"секунда {seconds.max():.2f} > длительность {total_sec:.1f}"

    # Round-trip: cloud строки совпадают
    cloud_full = pointcloud.takens(x, pc_cfg["window"], pc_cfg["stride"])
    for v in vertices[:5]:
        np.testing.assert_array_equal(
            cloud[v], cloud_full[sub_indices[v]],
            err_msg=f"round-trip провалился для вершины {v}")

    print("  ✓ Синтетическая проверка пройдена!")
    return True


# ======================================================================
# 3. Основной анализ
# ======================================================================

def run_interpretation(best, cfg):
    """Полный анализ: chroma similarity, close/distant pairs, визуализации."""
    pc_cfg = cfg["pointcloud"].copy()
    pc_cfg["takens_pca_dim"] = None
    window = pc_cfg["window"]
    target_fps = cfg["common"]["target_fps"]
    sr = cfg["data"]["sample_rate"]
    hop_length = cfg["spaces"]["mir"]["hop_length"]
    window_sec = window / target_fps  # длительность окна в секундах

    # Cocycle → секунды
    dgm1 = best["ripser_result"]["dgms"][1]
    life = dgm1[:, 1] - dgm1[:, 0]
    k = int(np.argmax(life))
    cocycle = best["ripser_result"]["cocycles"][1][k]
    birth, death = dgm1[k]
    pers = death - birth

    seconds, vertices = cocycle_vertices_to_seconds(
        cocycle, best["sub_indices"], best["takens_starts"],
        window, target_fps)

    print(f"\n=== Трек: {best['filename']} ({best['genre']}) ===")
    print(f"Max H1 persistence: {pers:.4f} (birth={birth:.4f}, death={death:.4f})")
    print(f"Cocycle: {len(cocycle)} рёбер, {len(vertices)} уникальных вершин")
    print(f"Секунды петли: {np.round(seconds, 2)}")

    # Загрузить wav для chroma
    filepath = best["filepath"]
    wav, _ = librosa.load(filepath, sr=sr, mono=True)
    clip_sec = cfg["data"]["clip_seconds"]

    # Chroma для моментов петли (окно-усреднённые)
    loop_chroma = extract_chroma_at_windows(wav, sr, seconds, window_sec,
                                             hop_length)

    # Полный chromagram для визуализации
    chroma_full = librosa.feature.chroma_cens(y=wav, sr=sr, hop_length=hop_length)
    chroma_fps = sr / hop_length

    # Self-similarity matrix (полный трек)
    chroma_T = chroma_full.T  # (T_chroma, 12)
    ssm = pairwise_cosine_sim(chroma_T)

    # ---- Пары петли: close vs distant ----
    CLOSE_THRESH = 2.0   # секунды
    DISTANT_THRESH = 5.0  # секунды

    loop_pairs_close = []
    loop_pairs_distant = []
    loop_sims_close = []
    loop_sims_distant = []

    for i in range(len(seconds)):
        for j in range(i + 1, len(seconds)):
            dt = abs(seconds[i] - seconds[j])
            sim = 1.0 - cosine_dist(loop_chroma[i], loop_chroma[j])
            if dt < CLOSE_THRESH:
                loop_pairs_close.append((seconds[i], seconds[j], sim, dt))
                loop_sims_close.append(sim)
            elif dt >= DISTANT_THRESH:
                loop_pairs_distant.append((seconds[i], seconds[j], sim, dt))
                loop_sims_distant.append(sim)

    loop_sims_close = np.array(loop_sims_close)
    loop_sims_distant = np.array(loop_sims_distant)

    print(f"\nПары петли: {len(loop_pairs_close)} близких (<{CLOSE_THRESH}s), "
          f"{len(loop_pairs_distant)} далёких (>{DISTANT_THRESH}s)")

    # ---- Случайные пары (контроль) — те же временны́е дистанции ----
    rng = np.random.default_rng(cfg["seed"])
    n_random = max(1000, len(loop_pairs_distant) * 10)

    # Сетка равномерных моментов для случайных пар
    all_times = np.linspace(window_sec / 2, clip_sec - window_sec / 2, 200)
    all_chroma = extract_chroma_at_windows(wav, sr, all_times, window_sec,
                                            hop_length)

    # Случайные далёкие пары (dt > DISTANT_THRESH)
    random_sims_distant = []
    attempts = 0
    while len(random_sims_distant) < n_random and attempts < n_random * 20:
        a, b = rng.choice(len(all_times), 2, replace=False)
        dt = abs(all_times[a] - all_times[b])
        if dt >= DISTANT_THRESH:
            sim = 1.0 - cosine_dist(all_chroma[a], all_chroma[b])
            random_sims_distant.append(sim)
        attempts += 1
    random_sims_distant = np.array(random_sims_distant)

    # ---- Статистика ----
    print(f"\n=== Chroma-сходство (cosine) ===")
    if len(loop_sims_close) > 0:
        print(f"  Близкие пары петли (<{CLOSE_THRESH}s): "
              f"mean={loop_sims_close.mean():.4f} ± {loop_sims_close.std():.4f} "
              f"(n={len(loop_sims_close)})")
    if len(loop_sims_distant) > 0:
        print(f"  Далёкие пары петли (>{DISTANT_THRESH}s): "
              f"mean={loop_sims_distant.mean():.4f} ± {loop_sims_distant.std():.4f} "
              f"(n={len(loop_sims_distant)})")
    print(f"  Далёкие случайные пары (>{DISTANT_THRESH}s): "
          f"mean={random_sims_distant.mean():.4f} ± {random_sims_distant.std():.4f} "
          f"(n={len(random_sims_distant)})")

    verdict = "INCONCLUSIVE"
    p_value = None

    if len(loop_sims_distant) >= 3:
        stat, p_value = mannwhitneyu(loop_sims_distant, random_sims_distant,
                                      alternative="greater")
        print(f"\n  Mann-Whitney (distant loop > distant random): "
              f"U={stat:.1f}, p={p_value:.6f}")
        if p_value < 0.05:
            verdict = "ПОДТВЕРЖДЕНА"
            print(f"  → Гипотеза ПОДТВЕРЖДЕНА (p < 0.05): далёкие пары петли "
                  f"похожи сильнее случайных.")
        else:
            verdict = "НЕ ПОДТВЕРЖДЕНА"
            print(f"  → Гипотеза НЕ ПОДТВЕРЖДЕНА (p = {p_value:.4f}): "
                  f"далёкие пары петли не похожи сильнее случайных.")
    else:
        print(f"\n  Недостаточно далёких пар петли ({len(loop_sims_distant)}) "
              f"для статистического теста.")
        verdict = "INCONCLUSIVE (мало далёких пар)"

    # ---- Визуализации ----
    out_dir = os.path.join(cfg["paths"]["results"], "figures")
    os.makedirs(out_dir, exist_ok=True)

    # === Фигура 1: Timeline + Chromagram ===
    fig1, (ax_chroma, ax_timeline) = plt.subplots(
        2, 1, figsize=(14, 6), height_ratios=[3, 1],
        sharex=True, gridspec_kw={"hspace": 0.08})

    # Chromagram
    times_chroma = np.arange(chroma_full.shape[1]) / chroma_fps
    ax_chroma.pcolormesh(times_chroma,
                          np.arange(12),
                          chroma_full, cmap="magma", shading="auto")
    ax_chroma.set_ylabel("Chroma pitch class")
    ax_chroma.set_yticks(np.arange(12))
    ax_chroma.set_yticklabels(["C", "C#", "D", "D#", "E", "F",
                                "F#", "G", "G#", "A", "A#", "B"])
    ax_chroma.set_title(f"{best['filename']} ({best['genre']}) — "
                         f"Chromagram + H1 loop moments "
                         f"(persistence={pers:.3f})")

    # Timeline с моментами петли
    ax_timeline.axhline(0, color="gray", lw=0.5)
    ax_timeline.set_xlim(0, clip_sec)
    ax_timeline.set_ylim(-0.5, 0.5)
    ax_timeline.set_yticks([])
    ax_timeline.set_xlabel("Time (seconds)")

    # Отметить каждый момент вертикальной линией на обоих axes
    for sec in seconds:
        ax_chroma.axvline(sec, color="cyan", alpha=0.7, lw=1.2, ls="--")
        ax_timeline.axvline(sec, color="red", alpha=0.8, lw=2)

    # Дуги для рёбер cocycle
    for edge_i, edge_j, val in cocycle:
        ei, ej = int(edge_i), int(edge_j)
        sec_i = (best["takens_starts"][best["sub_indices"][ei]]
                 + window / 2) / target_fps
        sec_j = (best["takens_starts"][best["sub_indices"][ej]]
                 + window / 2) / target_fps
        mid = (sec_i + sec_j) / 2
        height = abs(sec_j - sec_i) / clip_sec * 0.4
        ax_timeline.annotate("", xy=(sec_j, 0), xytext=(sec_i, 0),
                              arrowprops=dict(arrowstyle="-",
                                              connectionstyle=f"arc3,rad={0.3}",
                                              color="blue", alpha=0.3, lw=0.8))

    ax_timeline.set_ylabel("Loop")
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "loop_on_timeline.png"),
                 bbox_inches="tight", dpi=150)
    print(f"\n  → {os.path.join(out_dir, 'loop_on_timeline.png')}")
    plt.close(fig1)

    # === Фигура 2: Self-Similarity Matrix с отмеченными парами ===
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    times_chroma_sec = np.arange(ssm.shape[0]) / chroma_fps
    im = ax2.imshow(ssm, origin="lower", cmap="magma", aspect="auto",
                     extent=[0, clip_sec, 0, clip_sec])
    plt.colorbar(im, ax=ax2, label="Cosine similarity")

    # Подсветить моменты петли
    for sec in seconds:
        ax2.axvline(sec, color="cyan", alpha=0.5, lw=0.8)
        ax2.axhline(sec, color="cyan", alpha=0.5, lw=0.8)

    # Отметить пересечения далёких пар петли
    for s1, s2, sim, dt in loop_pairs_distant:
        ax2.plot(s1, s2, "o", color="lime", markersize=4, alpha=0.8)
        ax2.plot(s2, s1, "o", color="lime", markersize=4, alpha=0.8)

    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title(f"Self-Similarity Matrix (chroma) — {best['filename']}\n"
                   f"Cyan: loop moments, Green: distant loop pairs")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "loop_similarity_matrix.png"),
                 bbox_inches="tight", dpi=150)
    print(f"  → {os.path.join(out_dir, 'loop_similarity_matrix.png')}")
    plt.close(fig2)

    # === Фигура 3: Boxplot ===
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    data_for_box = []
    labels_for_box = []
    if len(loop_sims_close) > 0:
        data_for_box.append(loop_sims_close)
        labels_for_box.append(f"Loop close\n(<{CLOSE_THRESH}s, n={len(loop_sims_close)})")
    if len(loop_sims_distant) > 0:
        data_for_box.append(loop_sims_distant)
        labels_for_box.append(f"Loop distant\n(>{DISTANT_THRESH}s, n={len(loop_sims_distant)})")
    data_for_box.append(random_sims_distant)
    labels_for_box.append(f"Random distant\n(>{DISTANT_THRESH}s, n={len(random_sims_distant)})")

    bp = ax3.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                      widths=0.5)
    colors = ["#4CAF50", "#2196F3", "#9E9E9E"][:len(data_for_box)]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax3.set_ylabel("Cosine similarity (chroma)")
    title_p = f" | Mann-Whitney p={p_value:.4f}" if p_value is not None else ""
    ax3.set_title(f"Chroma similarity: loop pairs vs random — {best['filename']}"
                   f"{title_p}")
    ax3.grid(axis="y", alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "loop_stats.png"),
                 bbox_inches="tight", dpi=150)
    print(f"  → {os.path.join(out_dir, 'loop_stats.png')}")
    plt.close(fig3)

    # ---- Текстовый отчёт ----
    report_dir = os.path.join(cfg["paths"]["results"], "tables")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "loop_interpretation.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("H1 LOOP SEMANTIC INTERPRETATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Track: {best['filename']}\n")
        f.write(f"Genre: {best['genre']}\n")
        f.write(f"Channel: MuQ-10\n")
        f.write(f"Max H1 persistence: {pers:.4f} "
                f"(birth={birth:.4f}, death={death:.4f})\n")
        f.write(f"Cocycle: {len(cocycle)} edges, "
                f"{len(vertices)} unique vertices\n")
        f.write(f"Loop seconds: {np.round(seconds, 2).tolist()}\n\n")

        f.write("-" * 70 + "\n")
        f.write("CHROMA SIMILARITY (cosine)\n")
        f.write("-" * 70 + "\n")
        if len(loop_sims_close) > 0:
            f.write(f"Close loop pairs (<{CLOSE_THRESH}s): "
                    f"mean={loop_sims_close.mean():.4f} "
                    f"± {loop_sims_close.std():.4f} (n={len(loop_sims_close)})\n")
        if len(loop_sims_distant) > 0:
            f.write(f"Distant loop pairs (>{DISTANT_THRESH}s): "
                    f"mean={loop_sims_distant.mean():.4f} "
                    f"± {loop_sims_distant.std():.4f} "
                    f"(n={len(loop_sims_distant)})\n")
        f.write(f"Distant random pairs (>{DISTANT_THRESH}s): "
                f"mean={random_sims_distant.mean():.4f} "
                f"± {random_sims_distant.std():.4f} "
                f"(n={len(random_sims_distant)})\n\n")

        if p_value is not None:
            f.write(f"Mann-Whitney U (distant loop > distant random): "
                    f"p={p_value:.6f}\n")
        f.write(f"\nVERDICT: {verdict}\n")
        f.write("=" * 70 + "\n")

    print(f"  → {report_path}")
    print(f"\n=== ВЕРДИКТ: {verdict} ===")

    return {
        "seconds": seconds,
        "loop_sims_distant": loop_sims_distant,
        "random_sims_distant": random_sims_distant,
        "p_value": p_value,
        "verdict": verdict,
    }


# ======================================================================
# main
# ======================================================================

def main():
    cfg = load_cfg()

    meta = pd.read_csv(cfg["paths"]["meta"])
    files = [os.path.join(cfg["paths"]["audio"], f) for f in meta["filename"]]

    # Train/test split (как в основном пайплайне)
    from sklearn.model_selection import train_test_split
    labels = meta[cfg["data"]["stratify_by"]].values
    tr_idx, _ = train_test_split(
        np.arange(len(files)), test_size=cfg["data"]["test_size"],
        stratify=labels, random_state=cfg["seed"])

    print("Training PCA reducer on train split (MuQ) ...")
    reducer = get_pca_reducer(cfg, "muq", files, tr_idx)

    # Синтетическая проверка маппинга
    verify_mapping_synthetic()

    # Поиск лучшего трека
    print("\n=== Поиск трека с максимальной H1-persistence ===")
    best = find_best_track(cfg, files, meta, reducer,
                            genres=["classical", "disco", "pop"],
                            max_tracks=30)

    if best is None:
        print("ERROR: не найдено ни одного трека с H1!")
        sys.exit(1)

    # Основной анализ
    results = run_interpretation(best, cfg)


if __name__ == "__main__":
    main()
