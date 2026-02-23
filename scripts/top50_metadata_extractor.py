#!/usr/bin/env python3
"""
Скрипт для извлечения метаданных существующих треков из папки top50musicSpotify.
Парсит имена файлов, ищет треки в Spotify API и сохраняет метаданные в CSV.

Использование:
    python scripts/top50_metadata_extractor.py --input data/top50musicSpotify --output data/top50_tracks.csv
"""
import os
import sys
import csv
import re
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


def parse_filename(filename: str) -> Tuple[str, str, str]:
    """
    Парсит имя файла вида: genre_NN_Song_Name_Artist.wav
    Возвращает: (genre, song_name, artist)
    """
    # Убираем расширение
    name = Path(filename).stem
    
    # Формат: genre_NN_rest
    # Пример: blues_01_Seven_Nation_Army_The_White_Stripes
    
    # Разбиваем по первым двум underscore
    parts = name.split('_', 2)
    if len(parts) < 3:
        return '', '', ''
    
    genre = parts[0]
    # parts[1] — номер (01, 02, ...)
    rest = parts[2]  # Song_Name_Artist
    
    # Пробуем найти разделитель между названием и артистом
    # Обычно артист в конце, а название может содержать feat, -, etc
    
    # Заменяем _ на пробелы для поиска
    rest_spaced = rest.replace('_', ' ')
    
    # Попробуем несколько эвристик:
    # 1. Если есть " - " в оригинале, разделяем по нему
    # 2. Иначе ищем известные паттерны артистов
    # 3. Иначе берём последнюю часть как артиста
    
    # Смотрим на оригинальную структуру с underscore
    # Артист обычно в конце и может содержать & или feat
    
    # Простая эвристика: последние слова после последнего заглавного слова
    # Но это ненадёжно
    
    # Лучше: вернуть всё как запрос для поиска
    return genre, rest_spaced, ''


def smart_parse_filename(filename: str) -> Tuple[str, str, str]:
    """
    Улучшенный парсер имён файлов.
    Пытается разделить название и артиста.
    """
    name = Path(filename).stem
    
    # Формат: genre_NN_rest
    parts = name.split('_', 2)
    if len(parts) < 3:
        return '', name, ''
    
    genre = parts[0]
    rest = parts[2]
    
    # Заменяем _ на пробелы
    rest = rest.replace('_', ' ')
    
    # Убираем лишние пробелы
    rest = ' '.join(rest.split())
    
    # Известные паттерны разделения:
    # - "Song Name Artist Name" — сложно
    # - "Song Name - Artist Name" — если есть дефис
    # - "Song Name (feat. X) Artist" — если есть feat
    
    # Пробуем найти артиста по известным паттернам
    # Часто артист — это последние 1-3 слова
    
    # Для top50 проще всего: использовать весь rest как поисковый запрос
    return genre, rest, ''


class Top50MetadataExtractor:
    """Извлекает метаданные для существующих треков через Spotify API"""

    def __init__(self, client_id: str, client_secret: str):
        credentials = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        )
        self.sp = spotipy.Spotify(client_credentials_manager=credentials)
        logger.info("Spotify клиент инициализирован")

    def _api_call_with_retry(self, func, *args, max_retries: int = 5, **kwargs):
        """API вызов с retry при rate limit"""
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except SpotifyException as e:
                if e.http_status == 429:
                    retry_after = int(e.headers.get('Retry-After', 5)) if e.headers else 5
                    logger.warning(f"Rate limit. Ждём {retry_after}с...")
                    time.sleep(retry_after + 1)
                elif e.http_status >= 500:
                    wait = 2 ** attempt
                    logger.warning(f"Server error. Retry через {wait}с...")
                    time.sleep(wait)
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        return None

    def search_track(self, query: str, genre: str = '') -> Optional[Dict]:
        """
        Ищет трек по запросу и возвращает метаданные.
        """
        try:
            results = self._api_call_with_retry(
                self.sp.search,
                q=query,
                type='track',
                limit=5,
                market='US'
            )
            
            if not results or not results.get('tracks', {}).get('items'):
                return None
            
            # Берём первый результат
            track = results['tracks']['items'][0]
            
            # Получаем полную информацию для ISRC
            track_full = self._api_call_with_retry(self.sp.track, track['id'])
            
            return {
                'id': track['id'],
                'name': track['name'],
                'artist': ', '.join([a['name'] for a in track['artists']]),
                'popularity': track['popularity'],
                'isrc': track_full.get('external_ids', {}).get('isrc') if track_full else None,
                'duration_ms': track.get('duration_ms', 0),
                'preview_url': track.get('preview_url') or (track_full.get('preview_url') if track_full else None),
                'spotify_url': track.get('external_urls', {}).get('spotify', ''),
                'genre': genre,
            }
            
        except Exception as e:
            logger.error(f"Ошибка поиска '{query}': {e}")
            return None

    def process_directory(self, input_dir: Path) -> List[Dict]:
        """
        Обрабатывает директорию с треками.
        """
        all_tracks = []
        
        # Получаем все жанровые папки
        genre_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        
        for genre_dir in sorted(genre_dirs):
            genre = genre_dir.name
            logger.info(f"\n{'='*50}")
            logger.info(f"Жанр: {genre.upper()}")
            logger.info(f"{'='*50}")
            
            # Получаем все wav файлы
            wav_files = sorted(genre_dir.glob("*.wav"))
            
            genre_tracks = []
            for i, wav_file in enumerate(wav_files, 1):
                # Парсим имя файла
                parsed_genre, search_query, _ = smart_parse_filename(wav_file.name)
                
                logger.info(f"[{i}/{len(wav_files)}] {wav_file.name[:50]}...")
                logger.info(f"  Поиск: '{search_query[:50]}'")
                
                # Ищем в Spotify
                track_info = self.search_track(search_query, genre)
                
                if track_info:
                    genre_tracks.append(track_info)
                    logger.info(f"  ✓ Найдено: {track_info['name']} - {track_info['artist']}")
                else:
                    logger.warning(f"  ✗ Не найдено")
                    # Добавляем placeholder
                    genre_tracks.append({
                        'id': '',
                        'name': search_query,
                        'artist': '',
                        'popularity': 0,
                        'isrc': '',
                        'duration_ms': 0,
                        'preview_url': '',
                        'spotify_url': '',
                        'genre': genre,
                    })
                
                time.sleep(0.2)  # Rate limiting
            
            all_tracks.extend(genre_tracks)
            found = sum(1 for t in genre_tracks if t.get('id'))
            logger.info(f"✓ {genre}: найдено {found}/{len(genre_tracks)}")
        
        return all_tracks


def save_to_csv(tracks: List[Dict], output_path: Path):
    """Сохраняет треки в CSV"""
    fieldnames = [
        'genre', 'name', 'artist', 'popularity',
        'isrc', 'duration_ms', 'preview_url', 'spotify_url', 'id'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(tracks)

    logger.info(f"✓ Сохранено {len(tracks)} треков в {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Извлечение метаданных треков из top50musicSpotify"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/top50musicSpotify",
        help="Папка с треками"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/top50_tracks.csv",
        help="Выходной CSV файл"
    )
    args = parser.parse_args()

    # API ключи
    api_key = os.getenv('api', '').strip().strip("'\"")
    api_secret = os.getenv('api_secret') or os.getenv('SPOTIFY_CLIENT_SECRET', '')

    if ':' in api_key and not api_secret:
        parts = api_key.split(':', 1)
        client_id, client_secret = parts[0], parts[1]
    else:
        client_id = api_key
        client_secret = api_secret

    if not client_id or not client_secret:
        logger.error("Не найдены Spotify API ключи в .env")
        sys.exit(1)

    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Папка не найдена: {input_dir}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = Top50MetadataExtractor(client_id, client_secret)
    
    logger.info(f"Обработка треков из {input_dir}")
    tracks = extractor.process_directory(input_dir)

    save_to_csv(tracks, output_path)

    # Статистика
    logger.info(f"\n{'='*50}")
    logger.info("СТАТИСТИКА")
    logger.info(f"{'='*50}")
    logger.info(f"Всего треков: {len(tracks)}")
    
    found = sum(1 for t in tracks if t.get('id'))
    logger.info(f"Найдено в Spotify: {found} ({100*found/len(tracks):.1f}%)")
    
    with_preview = sum(1 for t in tracks if t.get('preview_url'))
    logger.info(f"С preview_url: {with_preview} ({100*with_preview/len(tracks):.1f}%)")


if __name__ == '__main__':
    main()

