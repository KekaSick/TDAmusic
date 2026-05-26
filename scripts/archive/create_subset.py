"""
create_subset.py
----------------
Создаёт размеченное подмножество GTZAN для этапа 3.
- 3 контрастных жанра: classical, metal, jazz
- 20 треков на жанр (итого 60), стратифицированно, seed=42
- Сохраняет data/meta/labels.csv
- Создаёт симлинки в data/raw/ для каждого выбранного трека
"""
import os
import random
import csv
from pathlib import Path

SEED = 42
GENRES = ["classical", "metal", "jazz"]
N_PER_GENRE = 20
GTZAN_DIR = "data/GTZAN"
RAW_DIR = "data/raw"
META_DIR = "data/meta"
LABELS_CSV = os.path.join(META_DIR, "labels.csv")

def main():
    random.seed(SEED)
    os.makedirs(META_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    rows = []
    for genre in GENRES:
        genre_dir = os.path.join(GTZAN_DIR, genre)
        wavs = sorted([f for f in os.listdir(genre_dir) if f.endswith(".wav")])
        if len(wavs) < N_PER_GENRE:
            raise ValueError(f"{genre}: only {len(wavs)} wavs, need {N_PER_GENRE}")
        
        selected = random.sample(wavs, N_PER_GENRE)
        selected.sort()  # для воспроизводимости порядка
        
        for fname in selected:
            src = os.path.abspath(os.path.join(genre_dir, fname))
            dst = os.path.join(RAW_DIR, fname)
            
            # Создаём симлинк, если файла/линка ещё нет
            if not os.path.exists(dst):
                os.symlink(src, dst)
                print(f"  symlink: {fname}")
            else:
                print(f"  exists:  {fname}")
            
            rows.append({"filename": fname, "genre": genre})

    # Сохраняем labels.csv
    with open(LABELS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "genre"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== labels.csv: {len(rows)} треков ===")
    for genre in GENRES:
        count = sum(1 for r in rows if r["genre"] == genre)
        print(f"  {genre}: {count}")
    print(f"Saved to {LABELS_CSV}")

if __name__ == "__main__":
    main()
