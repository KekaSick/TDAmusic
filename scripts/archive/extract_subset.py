"""
extract_subset.py
-----------------
Прогоняет извлечение эмбеддингов ТОЛЬКО для треков из labels.csv
по всем 4 каналам. Пропускает битые файлы, логирует ошибки.
После извлечения обновляет labels.csv, убирая провалившиеся треки.
"""
import os
import sys
import csv
import yaml
import time
import numpy as np

# Добавляем src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from embed_spaces import extract


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    labels_csv = cfg["paths"]["meta"]
    audio_dir = cfg["paths"]["audio"]
    spaces = list(cfg["spaces"].keys())  # mert, muq, encodec, mir

    # Читаем labels.csv
    with open(labels_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"=== Извлечение для {len(rows)} треков по {len(spaces)} каналам ===")
    print(f"Каналы: {spaces}")
    print()

    failed = {}  # {filename: error_msg}
    success_files = set()

    for space in spaces:
        t0 = time.time()
        space_fail = 0
        space_skip = 0
        space_done = 0

        print(f"--- {space} ---")
        for i, row in enumerate(rows):
            fname = row["filename"]
            filepath = os.path.join(audio_dir, fname)

            # Проверяем кэш (через ту же логику, что extract())
            cache_dir = os.path.join(cfg["paths"]["cache"], space)
            base = os.path.splitext(fname)[0]
            cache_path = os.path.join(cache_dir, f"{base}.npy")

            if os.path.exists(cache_path):
                space_skip += 1
                if space == spaces[0]:
                    success_files.add(fname)
                continue

            try:
                arr = extract(filepath, space, cfg)
                space_done += 1
                if space == spaces[0]:
                    success_files.add(fname)
                print(f"  [{i+1}/{len(rows)}] {fname}: {arr.shape}")
            except Exception as e:
                space_fail += 1
                failed[fname] = str(e)
                print(f"  [{i+1}/{len(rows)}] FAIL {fname}: {e}")

        elapsed = time.time() - t0
        print(f"  {space}: done={space_done} skipped={space_skip} "
              f"failed={space_fail} time={elapsed:.1f}s")
        print()

    # Собираем список файлов, успешных по ВСЕМ каналам
    all_ok = set()
    for row in rows:
        fname = row["filename"]
        base = os.path.splitext(fname)[0]
        if all(
            os.path.exists(os.path.join(cfg["paths"]["cache"], sp, f"{base}.npy"))
            for sp in spaces
        ):
            all_ok.add(fname)

    # Обновляем labels.csv: оставляем только успешные
    final_rows = [r for r in rows if r["filename"] in all_ok]
    if len(final_rows) < len(rows):
        removed = [r["filename"] for r in rows if r["filename"] not in all_ok]
        # Safety: если >50% провалилось — скорее всего проблема окружения
        if len(final_rows) < len(rows) * 0.5:
            print(f"\n⚠️  >50% треков не извлечены ({len(rows) - len(final_rows)}/{len(rows)}).")
            print("    Скорее всего проблема окружения (неправильный Python/torch).")
            print("    labels.csv НЕ изменён. Исправь окружение и перезапусти.")
            final_rows = rows  # оставляем оригинал
        else:
            print(f"\n!!! Убрано из labels.csv (нет .npy по всем каналам): {removed}")
            with open(labels_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["filename", "genre"])
                writer.writeheader()
                writer.writerows(final_rows)
            print(f"labels.csv обновлён: {len(final_rows)} треков")
    else:
        print(f"\nВсе {len(rows)} треков извлечены успешно по всем каналам!")

    # Итоговая статистика
    print(f"\n=== ИТОГО labels.csv ===")
    from collections import Counter
    genre_counts = Counter(r["genre"] for r in final_rows)
    for g in sorted(genre_counts):
        print(f"  {g}: {genre_counts[g]}")
    print(f"  всего: {len(final_rows)}")

    if failed:
        print(f"\n=== ОШИБКИ ({len(failed)}) ===")
        for fn, err in failed.items():
            print(f"  {fn}: {err}")


if __name__ == "__main__":
    main()
