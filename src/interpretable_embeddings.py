"""
52-компонентный интерпретируемый вектор эмбеддингов для музыкального анализа.

Вектор строится по четырём фундаментальным измерениям музыки:
  ВЫСОТА (Pitch)   — блоки 1–5:  хромаграмма, DFT, Tonnetz, Harmonic Tension
  ТЕМБР  (Timbre)  — блоки 6–7:  MFCC, спектральные дескрипторы
  ДИНАМИКА         — блок 8:     log RMS, dynamic range
  РИТМ             — блок 9:     onset strength, beat strength

Bar-level:   compute_bar_embeddings(file) → (n_bars, 52)
Track-level: full_track_embedding(bar_emb) → (523,)

Источники:
  Fujishima 1999, Amiot 2016, Yust 2015, Harte & Sandler 2006,
  Plomp & Levelt 1965, Krumhansl 1990, Müller 2015,
  Tzanetakis & Cook 2002, Cohn 1997.
"""

import os
import logging
from typing import Literal, Tuple, List, Optional

import numpy as np
import librosa
import soundfile as sf
from scipy.io import wavfile

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Константы: Krumhansl key profiles (24 ключа)
# ═══════════════════════════════════════════════════════════════

# Krumhansl & Kessler (1990), нормализованы в сумму 1
KRUMHANSL_MAJOR = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
], dtype=np.float64)
KRUMHANSL_MAJOR = KRUMHANSL_MAJOR / KRUMHANSL_MAJOR.sum()

KRUMHANSL_MINOR = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
], dtype=np.float64)
KRUMHANSL_MINOR = KRUMHANSL_MINOR / KRUMHANSL_MINOR.sum()

# Все 24 ключевых профиля: 12 мажорных + 12 минорных транспозиций
ALL_KEY_PROFILES = np.zeros((24, 12), dtype=np.float64)
for _i in range(12):
    ALL_KEY_PROFILES[_i] = np.roll(KRUMHANSL_MAJOR, _i)
    ALL_KEY_PROFILES[12 + _i] = np.roll(KRUMHANSL_MINOR, _i)


# ═══════════════════════════════════════════════════════════════
# FEATURE_NAMES: имена 52 компонент
# ═══════════════════════════════════════════════════════════════

_PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']

_BASE_FEATURE_NAMES: List[str] = (
    # Блок 1: Нормализованная хромаграмма (12)
    [f'chroma_{pc}' for pc in _PITCH_CLASSES]
    # Блок 2: DFT magnitudes (5)
    + ['dft_mag_chromaticity', 'dft_mag_dyadic', 'dft_mag_triadic',
       'dft_mag_diatonicity', 'dft_mag_fifthness']
    # Блок 3: DFT phases (5)
    + ['dft_phase_chromaticity', 'dft_phase_dyadic', 'dft_phase_triadic',
       'dft_phase_diatonicity', 'dft_phase_fifthness']
    # Блок 4: Tonnetz (6)
    + ['tonnetz_fifth_x', 'tonnetz_fifth_y',
       'tonnetz_minor3_x', 'tonnetz_minor3_y',
       'tonnetz_major3_x', 'tonnetz_major3_y']
    # Блок 5: Harmonic Tension (3)
    + ['sensory_dissonance', 'harmonic_change_rate', 'tonal_stability']
    # Блок 6: MFCC (13)
    + [f'mfcc_{i}' for i in range(13)]
    # Блок 7: Spectral descriptors (3)
    + ['spectral_centroid', 'spectral_flatness', 'spectral_bandwidth']
    # Блок 8: Dynamics (2)
    + ['log_rms', 'dynamic_range']
    # Блок 9: Rhythm (3)
    + ['onset_strength_mean', 'onset_strength_std', 'beat_strength']
)

assert len(_BASE_FEATURE_NAMES) == 52, (
    f"Expected 52 feature names, got {len(_BASE_FEATURE_NAMES)}"
)


def feature_names() -> List[str]:
    """Имена 52 компонент bar-level вектора."""
    return list(_BASE_FEATURE_NAMES)


def feature_names_full() -> List[str]:
    """
    Имена всех 523 компонент track-level вектора
    (full_track_embedding).
    """
    base = _BASE_FEATURE_NAMES
    names: List[str] = []

    # Глобальные статистики (208)
    for prefix in ('mean', 'std', 'median', 'iqr'):
        names.extend(f'{prefix}_{n}' for n in base)

    # Секционные средние (4 × 52 = 208)
    for sec in range(1, 5):
        names.extend(f'section{sec}_{n}' for n in base)

    # Траекторные статистики (2 × 52 = 104)
    for prefix in ('delta_mean', 'delta_std'):
        names.extend(f'{prefix}_{n}' for n in base)

    # Мета (3)
    names.extend(['n_bars', 'mean_energy', 'diatonicity_variability'])

    assert len(names) == 523, f"Expected 523, got {len(names)}"
    return names


# ═══════════════════════════════════════════════════════════════
# Вспомогательные функции
# ═══════════════════════════════════════════════════════════════

def _load_mono_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Загрузка аудио в моно float32.
    (копия из mir_bar_features, чтобы модуль был автономным)
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".wav":
        sr, data = wavfile.read(file_path)
        if data.ndim == 2:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if data.max() > 1.0:
            data /= np.abs(data).max() + 1e-9
        logger.info(f"WAV '{file_path}' loaded, sr={sr}")
    else:
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
    bpm: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Определяет границы тактов (секунды).
    Возвращает (bar_boundaries, tempo).
    """
    duration = len(x) / sr

    if bpm is not None:
        tempo = float(bpm)
        _, beat_frames = librosa.beat.beat_track(y=x, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        first_beat_time = beat_times[0] if len(beat_times) > 0 else 0.0
    else:
        tempo_est, beat_frames = librosa.beat.beat_track(y=x, sr=sr)
        if isinstance(tempo_est, np.ndarray):
            tempo_est = tempo_est[0]
        tempo = float(tempo_est)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        first_beat_time = beat_times[0] if len(beat_times) > 0 else 0.0
        logger.info(
            f"Detected tempo: {tempo:.2f} BPM, first beat at {first_beat_time:.3f}s"
        )

    # Guard against tempo=0 (e.g. static sine waves with no beats)
    if tempo < 1.0:
        tempo = 120.0
        logger.warning("Tempo detection returned ~0, falling back to 120 BPM")

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

    arr = np.array(bar_boundaries, dtype=float)
    logger.info(
        f"Detected {len(arr) - 1} bars, bar duration: {bar_duration:.3f}s"
    )
    return arr, tempo


def _sensory_dissonance(spectrum: np.ndarray, freqs: np.ndarray) -> float:
    """
    Упрощённая модель сенсорного диссонанса Plomp & Levelt (1965).

    Для каждой пары спектральных пиков вычисляет "шероховатость"
    на основе разности частот относительно critical bandwidth,
    взвешенную амплитудами.

    Parameters
    ----------
    spectrum : np.ndarray, shape (n_bins,)
        Усреднённый спектр мощности такта.
    freqs : np.ndarray, shape (n_bins,)
        Частоты, соответствующие бинам спектра.

    Returns
    -------
    float
        Мера сенсорного диссонанса (≥0).
    """
    # Берём top-K пиков для вычислительной эффективности
    n_peaks = 20
    amps = np.sqrt(spectrum + 1e-12)
    peak_idx = np.argsort(amps)[-n_peaks:]
    peak_freqs = freqs[peak_idx]
    peak_amps = amps[peak_idx]

    # Фильтруем нулевые частоты
    valid = peak_freqs > 0
    peak_freqs = peak_freqs[valid]
    peak_amps = peak_amps[valid]

    if len(peak_freqs) < 2:
        return 0.0

    dissonance = 0.0
    for i in range(len(peak_freqs)):
        for j in range(i + 1, len(peak_freqs)):
            f_low = min(peak_freqs[i], peak_freqs[j])
            f_high = max(peak_freqs[i], peak_freqs[j])
            a_prod = peak_amps[i] * peak_amps[j]

            # Critical bandwidth (Plomp & Levelt approximation)
            cb = 1.72 * (f_low ** 0.65)
            if cb < 1e-6:
                continue

            s = abs(f_high - f_low) / cb
            # Plomp-Levelt dissonance curve: peak at s ≈ 0.25
            if s < 1.2:
                d = s * np.exp(1 - s / 0.25) / 0.25
                dissonance += a_prod * d

    # Нормализация по общей амплитуде
    total_amp = peak_amps.sum()
    if total_amp > 1e-9:
        dissonance /= (total_amp ** 2)

    return float(dissonance)


def _tonal_stability(chroma_norm: np.ndarray) -> float:
    """
    Максимальная корреляция нормализованной хромы
    со всеми 24 ключевыми профилями Krumhansl.

    Parameters
    ----------
    chroma_norm : np.ndarray, shape (12,)
        L1-нормализованный хрома-вектор.

    Returns
    -------
    float
        Корреляция ∈ [-1, 1], обычно [0.3, 0.95].
    """
    max_corr = -1.0
    for profile in ALL_KEY_PROFILES:
        c = np.corrcoef(chroma_norm, profile)[0, 1]
        if not np.isnan(c) and c > max_corr:
            max_corr = c
    return float(max_corr)


# ═══════════════════════════════════════════════════════════════
# BAR-LEVEL: 52D интерпретируемый вектор
# ═══════════════════════════════════════════════════════════════

def compute_bar_embeddings(
    file_path: str,
    *,
    sr: int = 22050,
    beats_per_bar: int = 4,
    use_auto_tempo: bool = True,
    bpm: Optional[float] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Вычисляет 52-компонентный интерпретируемый вектор для каждого такта.

    Блоки вектора:
      [0:12]  — нормализованная хромаграмма (12D)
      [12:17] — DFT magnitudes на Z/12Z (5D)
      [17:22] — DFT phases на Z/12Z (5D)
      [22:28] — Tonnetz координаты (6D)
      [28:31] — Harmonic Tension (3D)
      [31:44] — MFCC (13D)
      [44:47] — Spectral descriptors (3D)
      [47:49] — Dynamics (2D)
      [49:52] — Rhythm (3D)

    Parameters
    ----------
    file_path : str
        Путь к аудиофайлу (WAV, MP3, FLAC, ...).
    sr : int
        Sample rate для загрузки (по умолчанию 22050).
    beats_per_bar : int
        Долей в такте (4 для 4/4).
    use_auto_tempo : bool
        Использовать автоматическое определение темпа.
    bpm : float, optional
        Явное указание BPM.
    n_fft : int
        Размер окна FFT.
    hop_length : int
        Шаг между фреймами.

    Returns
    -------
    np.ndarray, shape (n_bars, 52)
        Матрица эмбеддингов тактов.
    """
    # ── Загрузка аудио ──
    y, orig_sr = _load_mono_audio(file_path)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    actual_sr = sr

    # ── Границы тактов ──
    bar_boundaries, tempo = _compute_bar_boundaries(
        y, actual_sr,
        beats_per_bar=beats_per_bar,
        use_auto_tempo=use_auto_tempo,
        bpm=bpm,
    )
    n_bars = len(bar_boundaries) - 1
    if n_bars == 0:
        logger.warning(f"No bars detected in '{file_path}'")
        return np.zeros((0, 52), dtype=np.float32)

    # ── Вычисление фич по фреймам ──

    # Хромаграмма (CQT): (12, n_frames_cqt)
    chroma = librosa.feature.chroma_cqt(
        y=y, sr=actual_sr, hop_length=hop_length
    )

    # Tonnetz: (6, n_frames)
    tonnetz = librosa.feature.tonnetz(y=y, sr=actual_sr, hop_length=hop_length)

    # MFCC: (13, n_frames)
    mfcc = librosa.feature.mfcc(
        y=y, sr=actual_sr, n_mfcc=13,
        hop_length=hop_length, n_fft=n_fft,
    )

    # Спектрограмма мощности: (n_fft/2+1, n_frames)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    freqs = librosa.fft_frequencies(sr=actual_sr, n_fft=n_fft)

    centroid = librosa.feature.spectral_centroid(S=S, sr=actual_sr)  # (1, n)
    flatness = librosa.feature.spectral_flatness(S=S)                # (1, n)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=actual_sr)  # (1, n)

    # RMS: (1, n_frames)
    rms = librosa.feature.rms(
        y=y, frame_length=n_fft, hop_length=hop_length
    )

    # Onset strength: (n_frames,)
    onset_env = librosa.onset.onset_strength(
        y=y, sr=actual_sr, hop_length=hop_length
    )

    # Временные метки фреймов
    n_stft_frames = S.shape[1]
    frame_times = librosa.frames_to_time(
        np.arange(n_stft_frames), sr=actual_sr, hop_length=hop_length
    )

    # Хрома может иметь другое число фреймов (CQT), считаем отдельные метки
    chroma_times = librosa.frames_to_time(
        np.arange(chroma.shape[1]), sr=actual_sr, hop_length=hop_length
    )
    tonnetz_times = librosa.frames_to_time(
        np.arange(tonnetz.shape[1]), sr=actual_sr, hop_length=hop_length
    )
    mfcc_times = librosa.frames_to_time(
        np.arange(mfcc.shape[1]), sr=actual_sr, hop_length=hop_length
    )
    onset_times = librosa.frames_to_time(
        np.arange(len(onset_env)), sr=actual_sr, hop_length=hop_length
    )

    # ── Агрегация по тактам ──
    bar_embeddings = np.zeros((n_bars, 52), dtype=np.float32)

    for bar_idx in range(n_bars):
        t_start = bar_boundaries[bar_idx]
        t_end = bar_boundaries[bar_idx + 1]

        # --- Блок 1: Нормализованная хромаграмма (12D) [0:12] ---
        cmask = (chroma_times >= t_start) & (chroma_times < t_end)
        if cmask.sum() > 0:
            bar_chroma = chroma[:, cmask].mean(axis=1)
        else:
            bar_chroma = np.zeros(12, dtype=np.float32)

        chroma_sum = bar_chroma.sum()
        bar_chroma_norm = bar_chroma / (chroma_sum + 1e-9)
        bar_embeddings[bar_idx, 0:12] = bar_chroma_norm

        # --- Блок 2: DFT magnitudes на Z/12Z (5D) [12:17] ---
        # --- Блок 3: DFT phases на Z/12Z (5D) [17:22] ---
        dft = np.fft.fft(bar_chroma_norm)
        for k in range(1, 6):
            bar_embeddings[bar_idx, 12 + (k - 1)] = np.abs(dft[k])
            bar_embeddings[bar_idx, 17 + (k - 1)] = np.angle(dft[k])

        # --- Блок 4: Tonnetz (6D) [22:28] ---
        tmask = (tonnetz_times >= t_start) & (tonnetz_times < t_end)
        if tmask.sum() > 0:
            bar_embeddings[bar_idx, 22:28] = tonnetz[:, tmask].mean(axis=1)

        # --- Блок 5: Harmonic Tension (3D) [28:31] ---
        # 5a: Sensory dissonance (Plomp-Levelt)
        smask = (frame_times >= t_start) & (frame_times < t_end)
        if smask.sum() > 0:
            bar_S = S[:, smask].mean(axis=1)
            bar_embeddings[bar_idx, 28] = _sensory_dissonance(bar_S, freqs)

        # 5b: Harmonic change rate (скорость движения в Tonnetz)
        if tmask.sum() > 1:
            bar_tonnetz = tonnetz[:, tmask]
            diffs = np.diff(bar_tonnetz, axis=1)
            bar_embeddings[bar_idx, 29] = float(
                np.mean(np.linalg.norm(diffs, axis=0))
            )

        # 5c: Tonal stability (корреляция с Krumhansl profiles)
        bar_embeddings[bar_idx, 30] = _tonal_stability(bar_chroma_norm)

        # --- Блок 6: MFCC (13D) [31:44] ---
        mmask = (mfcc_times >= t_start) & (mfcc_times < t_end)
        if mmask.sum() > 0:
            bar_embeddings[bar_idx, 31:44] = mfcc[:, mmask].mean(axis=1)

        # --- Блок 7: Spectral descriptors (3D) [44:47] ---
        if smask.sum() > 0:
            bar_embeddings[bar_idx, 44] = (
                centroid[0, smask].mean() / (actual_sr / 2)
            )
            bar_embeddings[bar_idx, 45] = flatness[0, smask].mean()
            bar_embeddings[bar_idx, 46] = (
                bandwidth[0, smask].mean() / (actual_sr / 2)
            )

        # --- Блок 8: Dynamics (2D) [47:49] ---
        if smask.sum() > 0:
            bar_rms = rms[0, smask]
            bar_embeddings[bar_idx, 47] = np.log1p(bar_rms.mean())
            bar_embeddings[bar_idx, 48] = (
                np.log1p(bar_rms.max()) - np.log1p(bar_rms.min() + 1e-9)
            )

        # --- Блок 9: Rhythm (3D) [49:52] ---
        omask = (onset_times >= t_start) & (onset_times < t_end)
        if omask.sum() > 0:
            bar_onset = onset_env[omask]
            bar_embeddings[bar_idx, 49] = bar_onset.mean()
            bar_embeddings[bar_idx, 50] = bar_onset.std()

            # Beat strength via autocorrelation of onset envelope
            # (plp() needs longer segments; autocorrelation works on bar-level)
            if len(bar_onset) > 10:
                autocorr = np.correlate(bar_onset, bar_onset, mode='full')
                autocorr = autocorr[len(autocorr) // 2:]
                autocorr /= (autocorr[0] + 1e-9)
                # Peak autocorrelation after lag=1 → rhythmic regularity
                if len(autocorr) > 3:
                    bar_embeddings[bar_idx, 51] = float(
                        np.max(autocorr[2:min(len(autocorr), 50)])
                    )
                else:
                    bar_embeddings[bar_idx, 51] = 0.0
            else:
                bar_embeddings[bar_idx, 51] = 0.0

    logger.info(
        f"Computed 52D bar embeddings for '{file_path}': "
        f"shape={bar_embeddings.shape}"
    )
    return bar_embeddings


# ═══════════════════════════════════════════════════════════════
# TRACK-LEVEL: агрегация bar → track
# ═══════════════════════════════════════════════════════════════

def track_statistics(bar_embeddings: np.ndarray) -> np.ndarray:
    """
    Стратегия 2: статистики по тактам.

    (n_bars, 52) → (208,)

    Для каждой из 52 компонент: mean, std, median, IQR.
    Сохраняет информацию о вариативности внутри трека.
    """
    stats = np.concatenate([
        bar_embeddings.mean(axis=0),                           # 52
        bar_embeddings.std(axis=0),                            # 52
        np.median(bar_embeddings, axis=0),                     # 52
        (np.percentile(bar_embeddings, 75, axis=0)
         - np.percentile(bar_embeddings, 25, axis=0)),         # 52
    ])
    return stats.astype(np.float32)  # (208,)


def sectional_embedding(
    bar_embeddings: np.ndarray,
    n_sections: int = 4,
) -> np.ndarray:
    """
    Стратегия 3: секционная агрегация.

    (n_bars, 52) → (n_sections * 52,)

    Делит трек на n_sections равных частей, для каждой считает mean.
    Сохраняет хронологию: начало, развитие, кульминация, конец.
    """
    n_bars = len(bar_embeddings)
    section_size = max(1, n_bars // n_sections)
    sections: list[np.ndarray] = []

    for i in range(n_sections):
        start = i * section_size
        if i == n_sections - 1:
            end = n_bars
        else:
            end = min((i + 1) * section_size, n_bars)
        sections.append(bar_embeddings[start:end].mean(axis=0))

    return np.concatenate(sections).astype(np.float32)


def full_track_embedding(bar_embeddings: np.ndarray) -> np.ndarray:
    """
    Стратегия 4 (рекомендуемая): максимальное сохранение информации.

    (n_bars, 52) → (523,)

    Структура:
      ├── global_stats  (52×4 = 208):  mean, std, median, IQR
      ├── sections      (4×52  = 208):  4 хронологических секции
      ├── trajectory    (52×2  = 104):  mean и std первой разности
      └── meta          (3):           n_bars, mean energy,
                                       diatonicity variability
    """
    n_bars = len(bar_embeddings)
    parts: list[np.ndarray] = []

    # ═══ 1. Глобальные статистики (208D) ═══
    parts.append(bar_embeddings.mean(axis=0))                        # 52
    parts.append(bar_embeddings.std(axis=0))                         # 52
    parts.append(np.median(bar_embeddings, axis=0))                  # 52
    parts.append(
        np.percentile(bar_embeddings, 75, axis=0)
        - np.percentile(bar_embeddings, 25, axis=0)
    )                                                                 # 52

    # ═══ 2. Секционные средние (208D) ═══
    n_sections = 4
    section_size = max(1, n_bars // n_sections)
    for i in range(n_sections):
        start = i * section_size
        if i < n_sections - 1:
            end = min((i + 1) * section_size, n_bars)
        else:
            end = n_bars
        parts.append(bar_embeddings[start:end].mean(axis=0))          # 52

    # ═══ 3. Траекторные статистики (104D) ═══
    if n_bars > 1:
        deltas = np.diff(bar_embeddings, axis=0)  # (n_bars-1, 52)
        parts.append(deltas.mean(axis=0))    # 52: средний тренд
        parts.append(deltas.std(axis=0))     # 52: волатильность
    else:
        parts.append(np.zeros(52, dtype=np.float32))
        parts.append(np.zeros(52, dtype=np.float32))

    # ═══ 4. Мета (3D) ═══
    parts.append(np.array([
        float(n_bars),                            # длина трека
        float(bar_embeddings[:, 47].mean()),       # средняя энергия (log_rms)
        float(bar_embeddings[:, 15].std()),         # вариативность diatonicity
    ], dtype=np.float32))

    result = np.concatenate(parts).astype(np.float32)
    assert result.shape == (523,), f"Expected (523,), got {result.shape}"
    return result


def compact_track_embedding(bar_embeddings: np.ndarray) -> np.ndarray:
    """
    Компактная агрегация bar → track (рекомендуемая).

    (n_bars, 52) → (211,)

    Убраны:
    - median (r>0.95 с mean)
    - sections (4×52=208D — слишком много для 999 сэмплов)
    - delta_mean (оставлен только delta_std = волатильность)

    Структура:
      ├── mean       (52D):  что в среднем происходит
      ├── std        (52D):  насколько вариативно
      ├── iqr        (52D):  робастная вариативность
      ├── delta_std  (52D):  траекторная волатильность
      └── meta       (3D):  n_bars, mean_energy, diatonicity_var
    """
    n_bars = len(bar_embeddings)
    parts: list[np.ndarray] = []

    # Глобальные статистики (52 × 3 = 156D)
    parts.append(bar_embeddings.mean(axis=0))                         # 52
    parts.append(bar_embeddings.std(axis=0))                          # 52
    parts.append(
        np.percentile(bar_embeddings, 75, axis=0)
        - np.percentile(bar_embeddings, 25, axis=0)
    )                                                                  # 52

    # Траекторная волатильность (52D)
    if n_bars > 1:
        deltas = np.diff(bar_embeddings, axis=0)
        parts.append(deltas.std(axis=0))                               # 52
    else:
        parts.append(np.zeros(52, dtype=np.float32))

    # Мета (3D)
    parts.append(np.array([
        float(n_bars),
        float(bar_embeddings[:, 47].mean()),       # mean energy (log_rms)
        float(bar_embeddings[:, 15].std()),         # diatonicity variability
    ], dtype=np.float32))

    result = np.concatenate(parts).astype(np.float32)
    assert result.shape == (211,), f"Expected (211,), got {result.shape}"
    return result


def feature_names_compact() -> List[str]:
    """
    Имена всех 211 компонент compact track-level вектора
    (compact_track_embedding).
    """
    base = _BASE_FEATURE_NAMES
    names: List[str] = []

    # Глобальные статистики (3 × 52 = 156)
    for prefix in ('mean', 'std', 'iqr'):
        names.extend(f'{prefix}_{n}' for n in base)

    # Траекторная волатильность (52)
    names.extend(f'delta_std_{n}' for n in base)

    # Мета (3)
    names.extend(['n_bars', 'mean_energy', 'diatonicity_variability'])

    assert len(names) == 211, f"Expected 211, got {len(names)}"
    return names
