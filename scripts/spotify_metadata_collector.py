#!/usr/bin/env python3
"""
Скрипт для сбора метаданных треков из Spotify API.
Сохраняет CSV с 1000 треками (по 100 на жанр) с распределением популярности.

Особенности:
- Фильтрация по жанрам АРТИСТОВ (не треков — у треков нет жанра)
- Батчевое обогащение sp.tracks() / sp.artists() для скорости
- Retry с backoff при rate limit (429)
- Глобальный used_ids для избежания дубликатов между жанрами
- Добор из соседних сегментов популярности для равномерности

Использование:
    python scripts/spotify_metadata_collector.py --output data/spotify_tracks.csv
    python scripts/spotify_metadata_collector.py --output data/spotify_tracks.csv --per-genre 50
"""
import os
import sys
import csv
import time
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional

from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_GENRES = [
    'blues', 'classical', 'country', 'electronic',
    'hip-hop', 'jazz', 'metal', 'pop', 'reggae', 'rock'
]

# Синонимы/поджанры для фильтрации артистов
GENRE_SYNONYMS = {
    'blues': ['blues', 'chicago blues', 'delta blues', 'electric blues', 'soul blues'],
    'classical': ['classical', 'baroque', 'romantic', 'orchestra', 'symphony', 'piano', 'opera', 'chamber'],
    'country': ['country', 'americana', 'country rock', 'outlaw country', 'nashville', 'bluegrass'],
    'electronic': ['electronic', 'electro', 'edm', 'house', 'techno', 'trance', 'dubstep', 'drum and bass', 'ambient'],
    'hip-hop': ['hip hop', 'hip-hop', 'rap', 'trap', 'gangsta rap', 'southern hip hop', 'conscious hip hop'],
    'jazz': ['jazz', 'smooth jazz', 'bebop', 'fusion', 'swing', 'cool jazz', 'latin jazz'],
    'metal': ['metal', 'heavy metal', 'thrash metal', 'death metal', 'black metal', 'power metal', 'metalcore', 'nu metal'],
    'pop': ['pop', 'dance pop', 'synth-pop', 'electropop', 'teen pop', 'indie pop'],
    'reggae': ['reggae', 'dancehall', 'ska', 'dub', 'roots reggae', 'reggaeton'],
    'rock': ['rock', 'hard rock', 'classic rock', 'alternative rock', 'indie rock', 'punk rock', 'progressive rock'],
}

DEFAULT_PER_GENRE = 100


class SpotifyMetadataCollector:
    """Сборщик метаданных треков из Spotify"""

    def __init__(self, client_id: str, client_secret: str):
        credentials = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        )
        self.sp = spotipy.Spotify(client_credentials_manager=credentials)
        logger.info("Spotify клиент инициализирован")

    def _api_call_with_retry(self, func, *args, max_retries: int = 5, **kwargs):
        """Выполняет API вызов с retry при 429 и exponential backoff"""
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except SpotifyException as e:
                if e.http_status == 429:
                    # Rate limit — читаем Retry-After
                    retry_after = int(e.headers.get('Retry-After', 5)) if e.headers else 5
                    logger.warning(f"Rate limit (429). Ждём {retry_after}с...")
                    time.sleep(retry_after + 1)
                elif e.http_status >= 500:
                    # Server error — exponential backoff
                    wait = 2 ** attempt
                    logger.warning(f"Server error {e.http_status}. Retry через {wait}с...")
                    time.sleep(wait)
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Ошибка: {e}. Retry через {wait}с...")
                    time.sleep(wait)
                else:
                    raise
        return None

    def _get_artists_genres_batch(self, artist_ids: List[str]) -> Dict[str, List[str]]:
        """Получает жанры артистов батчем (до 50 за раз)"""
        result = {}
        
        # Убираем дубликаты и None
        unique_ids = list(set(aid for aid in artist_ids if aid))
        
        for i in range(0, len(unique_ids), 50):
            chunk = unique_ids[i:i+50]
            try:
                response = self._api_call_with_retry(self.sp.artists, chunk)
                if response and response.get('artists'):
                    for artist in response['artists']:
                        if artist:
                            result[artist['id']] = [g.lower() for g in artist.get('genres', [])]
                time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Ошибка получения артистов: {e}")
        
        return result

    def _artist_matches_genre(self, artist_genres: List[str], target_genre: str) -> bool:
        """Проверяет, соответствует ли артист целевому жанру"""
        synonyms = GENRE_SYNONYMS.get(target_genre.lower(), [target_genre.lower()])
        
        for ag in artist_genres:
            ag_lower = ag.lower()
            for syn in synonyms:
                if syn in ag_lower or ag_lower in syn:
                    return True
        
        return False

    def _enrich_tracks_batch(self, tracks: List[Dict]) -> List[Dict]:
        """Обогащает треки ISRC и preview_url батчем (до 50 за раз)"""
        if not tracks:
            return tracks
        
        logger.info(f"Обогащение {len(tracks)} треков батчами...")
        
        track_ids = [t['id'] for t in tracks]
        track_map = {t['id']: t for t in tracks}
        
        for i in range(0, len(track_ids), 50):
            chunk = track_ids[i:i+50]
            try:
                response = self._api_call_with_retry(self.sp.tracks, chunk)
                if response and response.get('tracks'):
                    for full_track in response['tracks']:
                        if full_track and full_track['id'] in track_map:
                            t = track_map[full_track['id']]
                            t['isrc'] = full_track.get('external_ids', {}).get('isrc')
                            t['duration_ms'] = full_track.get('duration_ms', t.get('duration_ms', 0))
                            if not t.get('preview_url'):
                                t['preview_url'] = full_track.get('preview_url')
                
                time.sleep(0.1)
                
                if (i + 50) % 200 == 0 or i + 50 >= len(track_ids):
                    logger.info(f"  Обогащено {min(i+50, len(track_ids))}/{len(track_ids)}")
                    
            except Exception as e:
                logger.warning(f"Ошибка обогащения батча: {e}")
        
        return tracks

    def search_tracks_raw(
        self,
        query: str,
        limit: int = 1000,
        used_ids: Set[str] = None
    ) -> List[Dict]:
        """Поиск треков (без фильтрации по жанрам артистов)"""
        if used_ids is None:
            used_ids = set()
        
        tracks = []
        offset = 0
        max_offset = 1000
        batch_size = 50
        
        while len(tracks) < limit and offset < max_offset:
            try:
                results = self._api_call_with_retry(
                    self.sp.search,
                    q=query,
                    type='track',
                    limit=batch_size,
                    offset=offset,
                    market='US'
                )
                
                if not results:
                    break
                
                items = results.get('tracks', {}).get('items', [])
                if not items:
                    break
                
                for track in items:
                    if track['id'] in used_ids:
                        continue
                    
                    # Собираем ID первого артиста для последующей фильтрации
                    artist_id = track['artists'][0]['id'] if track['artists'] else None
                    
                    used_ids.add(track['id'])
                    tracks.append({
                        'id': track['id'],
                        'name': track['name'],
                        'artist': ', '.join([a['name'] for a in track['artists']]),
                        'artist_id': artist_id,
                        'popularity': track['popularity'],
                        'preview_url': track.get('preview_url'),
                        'spotify_url': track.get('external_urls', {}).get('spotify', ''),
                        'duration_ms': track.get('duration_ms', 0),
                        'isrc': None,
                    })
                
                if len(items) < batch_size:
                    break
                
                offset += batch_size
                time.sleep(0.15)
                
            except Exception as e:
                logger.error(f"Ошибка поиска '{query}': {e}")
                break
        
        return tracks

    def collect_genre_pool(
        self,
        genre: str,
        target_size: int = 1000,
        global_used_ids: Set[str] = None
    ) -> List[Dict]:
        """
        Собирает пул треков для жанра.
        Фильтрует по жанрам артистов для точности.
        """
        if global_used_ids is None:
            global_used_ids = set()
        
        all_raw_tracks: List[Dict] = []
        
        # Разные варианты поисковых запросов
        queries = [
            f'genre:"{genre}"',
            f'genre:{genre}',
            f'{genre}',
        ]
        
        # Запросы с годами для разнообразия популярности
        years = ['2024', '2023', '2022', '2021', '2020', '2015', '2010', '2005', '2000', '1995', '1990', '1985', '1980']
        for year in years:
            queries.append(f'genre:"{genre}" year:{year}')
        
        logger.info(f"Сбор сырого пула треков для жанра '{genre}'...")
        
        for query in queries:
            # Собираем с запасом, т.к. после фильтрации часть отсеется
            tracks = self.search_tracks_raw(
                query=query,
                limit=200,
                used_ids=global_used_ids
            )
            all_raw_tracks.extend(tracks)
            
            if tracks:
                logger.info(f"  '{query[:35]}...': +{len(tracks)} (всего сырых: {len(all_raw_tracks)})")
            
            if len(all_raw_tracks) >= target_size * 3:
                break
        
        if not all_raw_tracks:
            logger.warning(f"Пул пуст для жанра '{genre}'")
            return []
        
        # Получаем жанры артистов батчем
        logger.info(f"Получение жанров артистов для фильтрации...")
        artist_ids = list(set(t['artist_id'] for t in all_raw_tracks if t.get('artist_id')))
        artist_genres = self._get_artists_genres_batch(artist_ids)
        
        # Фильтруем по жанрам артистов
        filtered_tracks = []
        for track in all_raw_tracks:
            artist_id = track.get('artist_id')
            if artist_id and artist_id in artist_genres:
                if self._artist_matches_genre(artist_genres[artist_id], genre):
                    filtered_tracks.append(track)
        
        logger.info(f"После фильтрации по жанрам артистов: {len(filtered_tracks)}/{len(all_raw_tracks)}")
        
        # Если после фильтрации мало — берём и нефильтрованные (с предупреждением)
        if len(filtered_tracks) < target_size // 2:
            logger.warning(f"Мало треков после фильтрации, добавляем нефильтрованные")
            existing_ids = {t['id'] for t in filtered_tracks}
            for t in all_raw_tracks:
                if t['id'] not in existing_ids:
                    filtered_tracks.append(t)
                    existing_ids.add(t['id'])
        
        return filtered_tracks

    def select_with_popularity_distribution(
        self,
        pool: List[Dict],
        count: int,
        segment_count: int = 10
    ) -> List[Dict]:
        """
        Выбирает треки с равномерным распределением популярности.
        Добирает из соседних сегментов (±1, ±2...) для равномерности.
        """
        if not pool:
            return []
        
        seg_size = 100.0 / segment_count
        segments: List[List[Dict]] = [[] for _ in range(segment_count)]
        
        for track in pool:
            pop = track['popularity']
            idx = min(segment_count - 1, max(0, int(pop // seg_size)))
            segments[idx].append(track)
        
        for seg in segments:
            random.shuffle(seg)
        
        # Логируем распределение в пуле
        logger.info("Распределение в пуле:")
        for i, seg in enumerate(segments):
            pop_min = int(i * seg_size)
            pop_max = int((i + 1) * seg_size)
            logger.info(f"  [{pop_min:2d}-{pop_max:2d}]: {len(seg):4d} треков")
        
        # Выбираем с квотами и добором из соседних сегментов
        per_segment = count // segment_count
        extra = count % segment_count
        
        selected = []
        taken_per_segment = [0] * segment_count
        
        for i in range(segment_count):
            quota = per_segment + (1 if i < extra else 0)
            
            # Сначала берём из своего сегмента
            available = segments[i][:quota]
            selected.extend(available)
            taken_per_segment[i] = len(available)
            segments[i] = segments[i][len(available):]
            
            # Если не хватило — добираем из соседних (±1, ±2, ...)
            shortfall = quota - len(available)
            if shortfall > 0:
                for distance in range(1, segment_count):
                    if shortfall <= 0:
                        break
                    
                    # Проверяем соседа слева
                    left_idx = i - distance
                    if 0 <= left_idx < segment_count and segments[left_idx]:
                        take = min(shortfall, len(segments[left_idx]))
                        selected.extend(segments[left_idx][:take])
                        segments[left_idx] = segments[left_idx][take:]
                        shortfall -= take
                    
                    # Проверяем соседа справа
                    right_idx = i + distance
                    if shortfall > 0 and 0 <= right_idx < segment_count and segments[right_idx]:
                        take = min(shortfall, len(segments[right_idx]))
                        selected.extend(segments[right_idx][:take])
                        segments[right_idx] = segments[right_idx][take:]
                        shortfall -= take
        
        selected = selected[:count]
        selected.sort(key=lambda x: x['popularity'])
        
        # Логируем ИТОГОВОЕ распределение выбранных
        logger.info("Итоговое распределение выбранных:")
        final_segments = [0] * segment_count
        for t in selected:
            idx = min(segment_count - 1, max(0, int(t['popularity'] // seg_size)))
            final_segments[idx] += 1
        
        for i in range(segment_count):
            pop_min = int(i * seg_size)
            pop_max = int((i + 1) * seg_size)
            bar = '█' * final_segments[i] if final_segments[i] > 0 else ''
            logger.info(f"  [{pop_min:2d}-{pop_max:2d}]: {final_segments[i]:3d} {bar}")
        
        return selected

    def collect_genre(
        self,
        genre: str,
        count: int,
        segment_count: int = 10,
        global_used_ids: Set[str] = None
    ) -> List[Dict]:
        """Собирает треки для одного жанра"""
        if global_used_ids is None:
            global_used_ids = set()
        
        # 1. Собираем пул с фильтрацией по жанрам артистов
        pool = self.collect_genre_pool(genre, target_size=max(1000, count * 10), global_used_ids=global_used_ids)
        
        if not pool:
            return []
        
        # 2. Выбираем с учётом популярности
        selected = self.select_with_popularity_distribution(pool, count, segment_count)
        
        # 3. Обогащаем батчем
        selected = self._enrich_tracks_batch(selected)
        
        # 4. Добавляем жанр и убираем служебные поля
        for t in selected:
            t['genre'] = genre
            t.pop('artist_id', None)
        
        return selected

    def collect_all_genres(
        self,
        genres: List[str],
        per_genre: int,
        segment_count: int = 10
    ) -> List[Dict]:
        """Собирает треки по всем жанрам (глобальный used_ids для уникальности)"""
        all_tracks = []
        global_used_ids: Set[str] = set()

        for genre in genres:
            logger.info(f"\n{'='*60}")
            logger.info(f"Жанр: {genre.upper()}")
            logger.info(f"{'='*60}")

            tracks = self.collect_genre(genre, per_genre, segment_count, global_used_ids)
            all_tracks.extend(tracks)
            
            if tracks:
                pops = [t['popularity'] for t in tracks]
                with_preview = sum(1 for t in tracks if t.get('preview_url'))
                logger.info(
                    f"✓ {genre}: {len(tracks)} треков, "
                    f"pop: {min(pops)}-{max(pops)} (avg={sum(pops)/len(pops):.1f}), "
                    f"preview: {with_preview}/{len(tracks)}"
                )

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
        description="Сбор метаданных треков из Spotify → CSV"
    )
    parser.add_argument("--output", "-o", default="data/spotify_tracks.csv")
    parser.add_argument("--per-genre", type=int, default=DEFAULT_PER_GENRE)
    parser.add_argument("--genres", nargs="*", default=None)
    parser.add_argument("--segments", type=int, default=10)
    args = parser.parse_args()

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

    genres = args.genres or DEFAULT_GENRES
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    collector = SpotifyMetadataCollector(client_id, client_secret)

    logger.info(f"Сбор: {len(genres)} жанров × {args.per_genre} треков = {len(genres) * args.per_genre}")
    tracks = collector.collect_all_genres(genres, args.per_genre, args.segments)

    save_to_csv(tracks, output_path)

    # Статистика
    logger.info(f"\n{'='*60}")
    logger.info("ИТОГОВАЯ СТАТИСТИКА")
    logger.info(f"{'='*60}")
    logger.info(f"Всего уникальных треков: {len(tracks)}")

    if tracks:
        with_preview = sum(1 for t in tracks if t.get('preview_url'))
        with_isrc = sum(1 for t in tracks if t.get('isrc'))
        all_pops = [t['popularity'] for t in tracks]
        
        logger.info(f"С preview_url: {with_preview} ({100*with_preview/len(tracks):.1f}%)")
        logger.info(f"С ISRC: {with_isrc} ({100*with_isrc/len(tracks):.1f}%)")
        logger.info(f"Популярность: min={min(all_pops)}, max={max(all_pops)}, avg={sum(all_pops)/len(all_pops):.1f}")
        
        logger.info("\nРаспределение популярности (все жанры):")
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for i in range(len(bins) - 1):
            cnt = sum(1 for p in all_pops if bins[i] <= p < bins[i+1])
            bar = '█' * (cnt // 10) if cnt > 0 else ''
            logger.info(f"  {bins[i]:2d}-{bins[i+1]:2d}: {cnt:4d} {bar}")

        logger.info("\nПо жанрам:")
        for genre in genres:
            genre_tracks = [t for t in tracks if t['genre'] == genre]
            if genre_tracks:
                pops = [t['popularity'] for t in genre_tracks]
                preview_cnt = sum(1 for t in genre_tracks if t.get('preview_url'))
                logger.info(f"  {genre}: {len(genre_tracks)} (pop: {min(pops)}-{max(pops)}, preview: {preview_cnt})")


if __name__ == '__main__':
    main()
