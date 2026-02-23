import os
import logging
from typing import Literal, Tuple

import numpy as np
import librosa
import soundfile as sf
from scipy.io import wavfile


logger = logging.getLogger(__name__)


def _load_mono_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Локальная копия загрузчика аудио (как в topology_methods._load_mono_audio),
    чтобы избежать циклических импортов.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".wav":
        sr, data = wavfile.read(file_path)
        if data.ndim == 2:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if data.max() > 1.0:  # int16 → float
            data /= np.abs(data).max() + 1e-9
        logger.info(f"WAV '{file_path}' loaded, sr={sr}")
    else:  # mp3, flac, …
        data, sr = sf.read(file_path, always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        logger.info(f"Audio '{file_path}' loaded via soundfile, sr={sr}")
    return data, sr


def _compute_bar_boundaries(
    x: np.ndarray,
    sr: int,
    beats_per_bar: int = 4,
    use_auto_tempo: bool = True,
    bpm: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Повторяет логику разбиения на такты из topology_methods.cqt_chroma_bar_embeddings.
    Возвращает:
        bar_boundaries : массив временных меток начала тактов (секунды), shape (n_bars+1,)
        tempo          : оценённый темп в BPM
    """
    duration = len(x) / sr

    # Определение темпа и долей
    if bpm is not None:
        tempo = float(bpm)
        logger.info(f"Using provided BPM: {tempo:.2f} BPM")
        _, beat_frames = librosa.beat.beat_track(y=x, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        first_beat_time = beat_times[0] if len(beat_times) > 0 else 0.0
    elif use_auto_tempo:
        tempo, beat_frames = librosa.beat.beat_track(y=x, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = tempo[0]
        tempo = float(tempo)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        first_beat_time = beat_times[0] if len(beat_times) > 0 else 0.0
        logger.info(f"Detected tempo: {tempo:.2f} BPM, first beat at {first_beat_time:.3f}s")
    else:
        tempo, beat_frames = librosa.beat.beat_track(y=x, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = tempo[0]
        tempo = float(tempo)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        first_beat_time = beat_times[0] if len(beat_times) > 0 else 0.0
        logger.info(f"Using librosa beat tracking, tempo: {tempo:.2f} BPM")

    bar_duration = (beats_per_bar * 60.0) / tempo

    bar_boundaries: list[float] = []

    if len(beat_times) > 0:
        beat_phase = first_beat_time % bar_duration
        start_time = first_beat_time - beat_phase
        current_time = max(0.0, start_time)
    else:
        current_time = 0.0

    while current_time < duration:
        bar_boundaries.append(current_time)
        current_time += bar_duration

    if len(bar_boundaries) == 0:
        bar_boundaries = [0.0, duration]
    elif bar_boundaries[-1] < duration:
        bar_boundaries.append(duration)

    bar_boundaries_arr = np.array(bar_boundaries, dtype=float)
    logger.info(f"Detected {len(bar_boundaries_arr) - 1} bars, bar duration: {bar_duration:.3f}s")
    return bar_boundaries_arr, tempo


def _aggregate_feature_by_bars(
    feat: np.ndarray,
    times: np.ndarray,
    bar_boundaries: np.ndarray,
    aggregation: Literal["mean", "sum"] = "mean",
) -> np.ndarray:
    """
    Агрегирует временной ряд признаков feat (shape=(n_feat, n_frames))
    по тактам, определённым bar_boundaries (shape=(n_bars+1,)).

    Возвращает массив shape (n_bars, n_feat).
    """
    n_feat, n_frames = feat.shape
    n_bars = len(bar_boundaries) - 1
    bar_feats = np.zeros((n_bars, n_feat), dtype=np.float32)

    for i in range(n_bars):
        bar_start = bar_boundaries[i]
        bar_end = bar_boundaries[i + 1]
        mask = (times >= bar_start) & (times < bar_end)

        if np.sum(mask) == 0:
            # Если нет кадров — ищем ближайший кадр (как fallback)
            if n_frames == 0:
                continue
            # ближайший индекс по времени к центру такта
            center = 0.5 * (bar_start + bar_end)
            idx = int(np.clip(np.argmin(np.abs(times - center)), 0, n_frames - 1))
            bar_frame_vals = feat[:, idx:idx + 1]
        else:
            bar_frame_vals = feat[:, mask]

        if aggregation == "mean":
            bar_vec = np.mean(bar_frame_vals, axis=1)
        elif aggregation == "sum":
            bar_vec = np.sum(bar_frame_vals, axis=1)
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

        bar_feats[i] = bar_vec.astype(np.float32)

    return bar_feats


def mir_bar_embeddings(
    file_path: str,
    *,
    beats_per_bar: int = 4,
    use_auto_tempo: bool = True,
    bpm: float | None = None,
    n_mfcc: int = 20,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin_rolloff: float | None = None,
    aggregation: Literal["mean", "sum"] = "mean",
) -> np.ndarray:
    """
    MIR + топологический баровый вектор признаков.

    Для каждого такта считаются:
      - MFCC (n_mfcc)
      - Δ MFCC (n_mfcc)
      - ΔΔ MFCC (n_mfcc)
      - spectral centroid (1)
      - spectral bandwidth (1)
      - spectral contrast (~7 полос)
      - spectral flatness (1)
      - spectral rolloff (1)
      - zero-crossing rate (1)
      - tonnetz (6)
      - плюс 12-мерный хрома-вектор из CQT-метрики (как в topology_methods),
        чтобы сохранить «авторскую» частотную метрику (в приоритете).

    На выходе: массив shape (n_bars, D).
    """
    # Загрузка аудио и разбиение на такты
    x, sr = _load_mono_audio(file_path)
    bar_boundaries, _ = _compute_bar_boundaries(
        x, sr,
        beats_per_bar=beats_per_bar,
        use_auto_tempo=use_auto_tempo,
        bpm=bpm,
    )

    # Базовые STFT/фичи
    S = np.abs(librosa.stft(y=x, n_fft=n_fft, hop_length=hop_length)) ** 2
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)

    # MFCC + дельты
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), sr=sr, n_mfcc=n_mfcc)
    d_mfcc = librosa.feature.delta(mfcc, order=1)
    dd_mfcc = librosa.feature.delta(mfcc, order=2)

    # Спектральные признаки
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    flatness = librosa.feature.spectral_flatness(S=S)
    rolloff = librosa.feature.spectral_rolloff(
        S=S,
        sr=sr,
        roll_percent=0.85,
    )

    # ZCR считается по временной области, но кадрируем тем же hop_length
    zcr = librosa.feature.zero_crossing_rate(
        y=x,
        frame_length=n_fft,
        hop_length=hop_length,
    )

    # Tonnetz — через хрома
    chroma_cens = librosa.feature.chroma_cens(y=x, sr=sr, hop_length=hop_length)
    tonnetz = librosa.feature.tonnetz(chroma=chroma_cens, sr=sr)

    # Агрегация по тактам
    mfcc_bars = _aggregate_feature_by_bars(mfcc, times, bar_boundaries, aggregation)
    d_mfcc_bars = _aggregate_feature_by_bars(d_mfcc, times, bar_boundaries, aggregation)
    dd_mfcc_bars = _aggregate_feature_by_bars(dd_mfcc, times, bar_boundaries, aggregation)
    centroid_bars = _aggregate_feature_by_bars(centroid, times, bar_boundaries, aggregation)
    bandwidth_bars = _aggregate_feature_by_bars(bandwidth, times, bar_boundaries, aggregation)
    contrast_bars = _aggregate_feature_by_bars(contrast, times, bar_boundaries, aggregation)
    flatness_bars = _aggregate_feature_by_bars(flatness, times, bar_boundaries, aggregation)
    rolloff_bars = _aggregate_feature_by_bars(rolloff, times, bar_boundaries, aggregation)
    zcr_bars = _aggregate_feature_by_bars(zcr, times, bar_boundaries, aggregation)
    tonnetz_bars = _aggregate_feature_by_bars(tonnetz, times, bar_boundaries, aggregation)

    # Подключаем «авторскую» тактовую хрому из topology_methods
    try:
        from .topology_methods import cqt_chroma_bar_embeddings

        chroma_bars = cqt_chroma_bar_embeddings(
            file_path=file_path,
            beats_per_bar=beats_per_bar,
            aggregation="mean",
            use_auto_tempo=use_auto_tempo,
            bpm=bpm,
        )
    except Exception as e:
        logger.exception(f"Failed to compute CQT bar chroma from topology_methods: {e}")
        # В крайнем случае заполняем нулями подходящей длины
        n_bars = len(bar_boundaries) - 1
        chroma_bars = np.zeros((n_bars, 12), dtype=np.float32)

    # Выравниваем число тактов, если вдруг источники разошлись
    n_bars = min(
        mfcc_bars.shape[0],
        d_mfcc_bars.shape[0],
        dd_mfcc_bars.shape[0],
        centroid_bars.shape[0],
        bandwidth_bars.shape[0],
        contrast_bars.shape[0],
        flatness_bars.shape[0],
        rolloff_bars.shape[0],
        zcr_bars.shape[0],
        tonnetz_bars.shape[0],
        chroma_bars.shape[0],
    )

    parts = [
        mfcc_bars[:n_bars],
        d_mfcc_bars[:n_bars],
        dd_mfcc_bars[:n_bars],
        centroid_bars[:n_bars],
        bandwidth_bars[:n_bars],
        contrast_bars[:n_bars],
        flatness_bars[:n_bars],
        rolloff_bars[:n_bars],
        zcr_bars[:n_bars],
        tonnetz_bars[:n_bars],
        chroma_bars[:n_bars],
    ]

    bar_feat = np.concatenate(parts, axis=1).astype(np.float32)
    return bar_feat


