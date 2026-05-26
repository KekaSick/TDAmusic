"""
smoke_test.py
-------------
Быстрая проверка ВСЕГО потока (кроме извлечения моделей) на синтетических
рядах за секунды. Гоняй ДО ночного извлечения GTZAN, чтобы поймать ошибки
в облаке / гомологиях / векторизации / расстояниях / контролях.

Идея теста: генерируем два «класса» синтетических траекторий —
  * периодические (должны давать долгоживущую петлю H1),
  * шумовые (петли быть не должно),
и проверяем, что within < between и что shuffle ломает H1 периодических.
Если это работает на синтетике — пайплайн собран правильно.

    python smoke_test.py
"""
import sys; sys.path.insert(0, "src")
import numpy as np
import pointcloud, persistence, controls, distances


CFG_PC = {"method": "takens", "window": 12, "stride": 2,
          "n_points": 150, "subsample": "maxmin"}
CFG_PERS = {"vectorization": "betti", "image_pixels": 20,
            "image_bandwidth": 0.1}


def make_periodic(T=400, d=8, seed=0):
    """Квазипериодическая траектория: круг в первых 2 коорд + шум в остальных."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8 * np.pi, T)
    x = np.zeros((T, d))
    x[:, 0], x[:, 1] = np.cos(t), np.sin(t)
    x[:, 2:] = 0.05 * rng.standard_normal((T, d - 2))
    return x


def make_noise(T=400, d=8, seed=0):
    """Гладкое случайное блуждание — петли быть не должно."""
    rng = np.random.default_rng(seed)
    return controls.random_like(np.zeros((T, d)), rng, match_autocorr=False).cumsum(0) * 0.1


def h1_diagram(x):
    cloud = pointcloud.build_cloud(x, CFG_PC)
    res = persistence.compute_diagrams(cloud, maxdim=1)
    return res["dgms"][1]


def max_persistence(dgm):
    if len(dgm) == 0:
        return 0.0
    return float((dgm[:, 1] - dgm[:, 0]).max())


def main():
    rng = np.random.default_rng(0)
    print("1) генерация синтетических классов ...")
    periodic = [make_periodic(seed=i) for i in range(6)]
    noise = [make_noise(seed=i) for i in range(6)]

    print("2) H1 периодических должно быть > H1 шумовых ...")
    p_pers = np.mean([max_persistence(h1_diagram(x)) for x in periodic])
    n_pers = np.mean([max_persistence(h1_diagram(x)) for x in noise])
    print(f"   max-persistence H1:  periodic={p_pers:.3f}  noise={n_pers:.3f}")
    assert p_pers > n_pers, "петля периодических должна жить дольше!"

    print("3) shuffle должен ломать петлю периодических ...")
    x = periodic[0]
    before = max_persistence(h1_diagram(x))
    xs = controls.shuffle_frames(x, rng)        # перемешиваем ДО окна Такенса
    after = max_persistence(h1_diagram(xs))
    print(f"   H1 до shuffle={before:.3f}  после={after:.3f}")
    assert before > after, "shuffle обязан ослаблять временну́ю петлю!"

    print("4) within < between (Вассерштейн) ...")
    diags = [h1_diagram(x) for x in periodic + noise]
    labels = ["per"] * 6 + ["noi"] * 6
    D = distances.pairwise_wasserstein(diags)
    w, b, gap = distances.within_between(D, labels)
    boot_gap, ci = distances.bootstrap_gap(D, labels, n_boot=500, rng=rng)
    print(f"   within={w:.3f} between={b:.3f} gap={gap:.3f} "
          f"boot_gap={boot_gap:.3f} CI={ci}")
    assert gap > 0, "between должно быть больше within!"

    print("5) Mantel самой с собой = 1.0 ...")
    assert abs(distances.mantel(D, D) - 1.0) < 1e-9

    print("6) build_cloud_with_indices: маппинг индексов ...")
    x_per = periodic[0]
    cloud_idx, takens_starts, sub_idx = pointcloud.build_cloud_with_indices(
        x_per, CFG_PC)
    # Проверить: subsampled cloud совпадает с полным Такенс-облаком по индексам
    cloud_full = pointcloud.takens(x_per, CFG_PC["window"], CFG_PC["stride"])
    np.testing.assert_array_equal(cloud_idx, cloud_full[sub_idx],
                                  err_msg="round-trip: cloud != takens_full[sub_idx]")
    # Проверить: стартовые кадры в пределах длины ряда
    assert takens_starts.max() + CFG_PC["window"] <= x_per.shape[0], \
        "start + window выходит за длину ряда"
    # Проверить: sub_idx в пределах полного Такенс-облака
    assert sub_idx.max() < len(takens_starts), \
        "sub_idx выходит за пределы полного Такенс-облака"
    # Проверить: маппинг к секундам осмыслён
    target_fps = 25.0
    window = CFG_PC["window"]
    for vi in range(min(5, len(sub_idx))):
        start_frame = takens_starts[sub_idx[vi]]
        center_sec = (start_frame + window / 2) / target_fps
        assert 0 <= center_sec <= x_per.shape[0] / target_fps, \
            f"секунда {center_sec} вне диапазона"

    print("\nВСЕ ПРОВЕРКИ ПРОШЛИ. Пайплайн собран корректно.")
    print("Теперь можно запускать реальное извлечение: python src/embed_spaces.py")


if __name__ == "__main__":
    main()
