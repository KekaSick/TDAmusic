"""
verify_subset.py
----------------
Проверяет, что для каждого трека из labels.csv
есть .npy в cache/{mert,muq,encodec,mir}/.
Выводит статистику и формы первых файлов.
"""
import os
import csv
import yaml
import numpy as np
from collections import Counter

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    labels_csv = cfg["paths"]["meta"]
    spaces = list(cfg["spaces"].keys())

    with open(labels_csv) as f:
        rows = list(csv.DictReader(f))

    print(f"=== labels.csv: {len(rows)} треков ===")
    genre_counts = Counter(r["genre"] for r in rows)
    for g in sorted(genre_counts):
        print(f"  {g}: {genre_counts[g]}")

    print(f"\n=== Проверка кэша ===")
    for space in spaces:
        cache_dir = os.path.join(cfg["paths"]["cache"], space)
        present = 0
        missing = []
        shapes = []

        for row in rows:
            base = os.path.splitext(row["filename"])[0]
            npy = os.path.join(cache_dir, f"{base}.npy")
            if os.path.exists(npy):
                present += 1
                if len(shapes) < 3:
                    arr = np.load(npy)
                    shapes.append((row["filename"], arr.shape, arr.dtype,
                                   arr.min(), arr.max()))
            else:
                missing.append(row["filename"])

        status = "✅" if present == len(rows) else "❌"
        print(f"\n  {space}: {status} {present}/{len(rows)} .npy")
        if missing:
            print(f"    missing: {missing[:5]}{'...' if len(missing)>5 else ''}")
        for fname, shape, dtype, vmin, vmax in shapes:
            print(f"    {fname}: shape={shape} dtype={dtype} "
                  f"range=[{vmin:.4f}, {vmax:.4f}]")

    # Общее число .npy на канал (включая test.npy и т.д.)
    print(f"\n=== Всего .npy в кэше ===")
    for space in spaces:
        cache_dir = os.path.join(cfg["paths"]["cache"], space)
        npys = [f for f in os.listdir(cache_dir) if f.endswith(".npy")]
        print(f"  {space}: {len(npys)} файлов")

if __name__ == "__main__":
    main()
