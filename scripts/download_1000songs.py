#!/usr/bin/env python3
"""
Download 1000 original WAV music files from spotify_tracks1000.csv.

Uses yt-dlp to search YouTube by ISRC (guarantees original recording, not cover),
with fallback to "{name} {artist} official audio" query.

Usage:
    python scripts/download_1000songs.py
    python scripts/download_1000songs.py --workers 8 --delay 0.5
    python scripts/download_1000songs.py --cookies-browser chrome
"""
import os
import sys
import csv
import time
import shutil
import subprocess
import argparse
import logging
import threading
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "spotify_tracks1000.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "1000songstry1"

stats_lock = threading.Lock()


# ── Helpers ──────────────────────────────────────────────────────────
def clean(text: str, max_len: int = 80) -> str:
    """Sanitise text for use in a filename."""
    for ch in "'\":/\\|?*<>()[]{}%&\r\n":
        text = text.replace(ch, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_ ")[:max_len]


def make_filename(track: Dict, index: int) -> str:
    genre = clean(track.get("genre", "unknown"), 20)
    artist = clean(track.get("artist", "unknown"), 40)
    name = clean(track.get("name", "unknown"), 50)
    return f"{genre}_{index:04d}_{artist}_{name}.wav"


# ── Core download logic ─────────────────────────────────────────────
def download_track(
    track: Dict,
    filepath: Path,
    cookies_file: str = None,
    cookies_browser: str = None,
    verbose: bool = False,
) -> Tuple[bool, str]:
    """
    Download a single track as WAV.

    Search order:
      1. YouTube by ISRC  (exact original recording)
      2. YouTube by "{name} {artist} official audio"
      3. YouTube by "{name} {artist}"

    Returns (success, method_used).
    """
    # Skip if already downloaded
    if filepath.exists() and filepath.stat().st_size > 10_000:
        return True, "cached"

    try:
        import yt_dlp
    except ImportError:
        logger.error("yt-dlp is not installed. Run: pip install yt-dlp")
        return False, "missing_dep"

    name = track.get("name", "")
    artist = track.get("artist", "")
    isrc = track.get("isrc", "").strip()

    # Build search queries in priority order
    queries = []
    if isrc:
        queries.append(f"ytsearch1:{isrc}")
    queries.append(f'ytsearch1:{name} {artist} official audio')
    queries.append(f"ytsearch1:{name} {artist}")

    output_template = str(filepath.with_suffix("")) + ".%(ext)s"

    base_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": not verbose,
        "no_warnings": not verbose,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 3,
        "js_runtimes": {"node": {"path": "/usr/local/bin/node"}},
        "remote_components": "ejs:github",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
    }

    if cookies_file:
        base_opts["cookiefile"] = cookies_file
    elif cookies_browser:
        base_opts["cookiesfrombrowser"] = (cookies_browser,)

    for query in queries:
        method = "isrc" if "isrc" not in query.lower() and queries.index(query) == 0 and isrc else "search"
        if query == queries[0] and isrc:
            method = "isrc"
        else:
            method = "search"

        try:
            # Clean any partial files from previous attempts
            for ext in [".wav", ".webm", ".m4a", ".mp3", ".part", ".temp"]:
                partial = filepath.with_suffix(ext)
                if partial.exists() and partial != filepath:
                    partial.unlink(missing_ok=True)

            with yt_dlp.YoutubeDL(base_opts) as ydl:
                ydl.download([query])

            # Check if the file was created
            time.sleep(0.3)
            if filepath.exists() and filepath.stat().st_size > 10_000:
                return True, method

            # Sometimes the extension differs; look for any output
            for ext in [".wav", ".mp3", ".m4a", ".webm"]:
                alt = filepath.with_suffix(ext)
                if alt.exists() and alt.stat().st_size > 10_000:
                    if alt != filepath:
                        alt.rename(filepath)
                    return True, method

        except Exception as e:
            err = str(e)
            if verbose:
                logger.debug(f"  query failed ({query[:50]}): {err[:120]}")
            continue

    return False, "all_failed"


def download_task(
    track: Dict,
    filepath: Path,
    index: int,
    total: int,
    delay: float,
    cookies_file: str = None,
    cookies_browser: str = None,
    verbose: bool = False,
) -> Tuple[bool, str, int]:
    """Worker task for a single track download."""
    label = f"[{index}/{total}]"
    track_label = f"{track.get('artist', '?')} – {track.get('name', '?')}"

    try:
        success, method = download_track(
            track, filepath,
            cookies_file=cookies_file,
            cookies_browser=cookies_browser,
            verbose=verbose,
        )

        if success:
            if method != "cached":
                logger.info(f"  ✓ {label} {track_label[:60]} ({method})")
        else:
            logger.warning(f"  ✗ {label} {track_label[:60]} — FAILED")

        if delay > 0 and method != "cached":
            time.sleep(delay)

        return success, method, index

    except Exception as e:
        logger.error(f"  ✗ {label} {track_label[:60]} — ERROR: {e}")
        return False, "error", index


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download 1000 original WAV tracks from CSV"
    )
    parser.add_argument("--csv", default=str(CSV_PATH), help="Path to CSV")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Parallel workers")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between downloads (seconds)")
    parser.add_argument("--cookies", default=None, help="Path to cookies.txt for YouTube")
    parser.add_argument("--cookies-browser", default=None, help="Browser for cookies (chrome, firefox, etc.)")
    parser.add_argument("--limit", type=int, default=None, help="Download only first N tracks")
    parser.add_argument("--start", type=int, default=0, help="Start from track index (0-based)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose yt-dlp output")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output)

    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        sys.exit(1)

    # Check ffmpeg
    if not shutil.which("ffmpeg"):
        logger.error("ffmpeg is required but not found. Install it first.")
        sys.exit(1)

    # Load tracks
    tracks: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tracks.append(row)

    logger.info(f"Loaded {len(tracks)} tracks from {csv_path.name}")

    # Apply start/limit
    if args.start > 0:
        tracks = tracks[args.start:]
        logger.info(f"Starting from index {args.start}")
    if args.limit:
        tracks = tracks[: args.limit]
        logger.info(f"Limited to {len(tracks)} tracks")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stats
    stats = {"total": 0, "success": 0, "failed": 0, "cached": 0}

    logger.info(f"")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Downloading {len(tracks)} tracks → {output_dir}")
    logger.info(f"  Workers: {args.workers}  |  Delay: {args.delay}s")
    logger.info(f"{'=' * 60}")
    logger.info(f"")

    # Build tasks — each genre goes into its own subfolder
    task_items = []
    for i, track in enumerate(tracks, 1):
        genre = clean(track.get("genre", "unknown"), 20)
        genre_dir = output_dir / genre
        genre_dir.mkdir(parents=True, exist_ok=True)
        filename = make_filename(track, i + args.start)
        filepath = genre_dir / filename
        task_items.append((track, filepath, i + args.start))

    total = len(task_items)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_task,
                track, filepath, idx, total, args.delay,
                args.cookies, args.cookies_browser, args.verbose,
            ): idx
            for track, filepath, idx in task_items
        }

        for future in as_completed(futures):
            try:
                success, method, idx = future.result()
                with stats_lock:
                    stats["total"] += 1
                    if success:
                        if method == "cached":
                            stats["cached"] += 1
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1
            except Exception as e:
                with stats_lock:
                    stats["total"] += 1
                    stats["failed"] += 1
                logger.error(f"  Worker exception: {e}")

    # Final report
    logger.info(f"")
    logger.info(f"{'=' * 60}")
    logger.info(f"  DONE")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Total tracks:  {stats['total']}")
    logger.info(f"  Downloaded:    {stats['success'] - stats['cached']}")
    logger.info(f"  Cached/skip:   {stats['cached']}")
    logger.info(f"  Failed:        {stats['failed']}")
    pct = 100 * stats["success"] / max(1, stats["total"])
    logger.info(f"  Success rate:  {pct:.1f}%")
    logger.info(f"  Output dir:    {output_dir}")
    logger.info(f"{'=' * 60}")

    if stats["failed"] > 0:
        logger.info(f"\nTip: Re-run the script to retry failed tracks (already-downloaded files are skipped).")


if __name__ == "__main__":
    main()
