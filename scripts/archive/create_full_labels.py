"""
create_full_labels.py
---------------------
Скрипт для сборки полного файла data/meta/labels.csv для всего датасета GTZAN (999 треков).
Один битый трек (jazz.00054.wav) будет пропущен автоматически (или вручную, если библиотека
не упадет, но мы знаем, что он битый).
"""
import os
import glob
import pandas as pd
import soundfile as sf
import yaml

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    raw_dir = cfg["paths"]["audio"]
    out_path = cfg["paths"]["meta"]
    
    # Резервное копирование старого файла (если он есть)
    if os.path.exists(out_path):
        backup_path = out_path.replace(".csv", "_subset_backup.csv")
        if not os.path.exists(backup_path):
            os.rename(out_path, backup_path)
            print(f"Backed up old labels to {backup_path}")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    data = []
    # GTZAN structure: data/genres_30sec/<genre>/<genre>.<id>.wav
    wav_files = glob.glob(os.path.join(raw_dir, "*", "*.wav"))
    
    for wav_path in wav_files:
        # Проверим, читается ли трек (отсеиваем битый)
        try:
            info = sf.info(wav_path)
            if info.duration < 1.0:
                print(f"Skipping {wav_path} (too short)")
                continue
        except Exception as e:
            print(f"Skipping {wav_path} due to error: {e}")
            continue
            
        genre = os.path.basename(os.path.dirname(wav_path))
        # Сохраняем относительный путь для config
        rel_path = os.path.relpath(wav_path, raw_dir)
        data.append({"filename": rel_path, "genre": genre})
        
    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False)
    print(f"Created {out_path} with {len(df)} tracks.")
    
    # Статистика по жанрам
    print("\nTracks per genre:")
    print(df["genre"].value_counts())

if __name__ == "__main__":
    main()
