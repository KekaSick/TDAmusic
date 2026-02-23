import os
import logging
import numpy as np
from scipy.io import wavfile
import soundfile as sf
import librosa
import math


logger = logging.getLogger(__name__)


def _load_mono_audio(file_path):
    """
    Считываем WAV (scipy) или любой другой формат (soundfile/librosa backend)
    и нормализуем до float32 [-1, 1].
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".wav":
        sr, data = wavfile.read(file_path)
        if data.ndim == 2:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if data.max() > 1.0:          # int16 → float
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


def cqt_chroma_bar_embeddings(file_path,
                              bins_per_octave: int = 36,
                              hop_len: int = 2048,
                              n_octaves: int = 7,
                              fmin: float | None = None,
                              beats_per_bar: int = 4,
                              aggregation: str = 'mean',
                              use_auto_tempo: bool = True,
                              bpm: float | None = None):
    """
    CQT → хрома (сжатая в одну октаву) → эмбеддинги тактов.
    
    Выполняет CQT преобразование, сжимает все октавы в одну октаву (хрома векторы),
    определяет такты через BPM/доли и возвращает временной ряд эмбеддингов тактов.
    Каждый эмбеддинг такта представляет собой 12-мерный хрома вектор.
    
    Параметры
    ---------
    file_path : str
        Путь к аудиофайлу
    bins_per_octave : int, default=36
        Количество бинов на октаву для CQT (≥ 12, кратно 12)
    hop_len : int, default=2048
        Размер hop для CQT в сэмплах
    n_octaves : int, default=7
        Количество октав для CQT (C1–C8 ≈ 7 октав)
    fmin : float, optional
        Нижняя частота для CQT. По умолчанию C1 (≈ 32.7 Гц)
    beats_per_bar : int, default=4
        Количество долей в такте (обычно 4 для 4/4)
    aggregation : str, default='mean'
        Метод агрегации хрома векторов внутри такта:
        - 'mean': среднее значение
        - 'sum': сумма
        - 'root': индекс ноты с максимальным средним значением
        - 'top3': индексы 0..11 трёх максимумов по убыванию
        - 'top4': индексы 0..11 четырёх максимумов по убыванию
    use_auto_tempo : bool, default=True
        Если True, автоматически определяет темп через librosa.beat.beat_track
    bpm : float, optional
        Явное указание BPM. Если задано, используется вместо автоматического определения
        
    Returns
    -------
    np.ndarray
        Массив эмбеддингов тактов, shape (n_bars, 12)
        Каждая строка - это 12-мерный хрома вектор для одного такта
    """
    
    # Загрузка аудио
    x, sr = _load_mono_audio(file_path)
    duration = len(x) / sr
    
    # Определение темпа
    if bpm is not None:
        tempo = float(bpm)
        logger.info(f"Using provided BPM: {tempo:.2f} BPM")
        # Используем первую долю для выравнивания границ тактов
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
        # Используем стандартный подход для определения долей
        tempo, beat_frames = librosa.beat.beat_track(y=x, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = tempo[0]
        tempo = float(tempo)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        first_beat_time = beat_times[0] if len(beat_times) > 0 else 0.0
        logger.info(f"Using librosa beat tracking, tempo: {tempo:.2f} BPM")
    
    # Вычисление длительности такта в секундах
    bar_duration = (beats_per_bar * 60.0) / tempo
    
    # Вычисление границ тактов на основе BPM
    # Выравниваем первый такт по первой доле
    bar_boundaries = []
    
    if len(beat_times) > 0:
        # Определяем позицию первой доли относительно начала такта
        # Выравниваем границы тактов так, чтобы первая доля была на границе такта
        # или близко к ней. Считаем, что первая доля - это начало такта или близка к нему
        beat_phase = first_beat_time % bar_duration
        
        # Начинаем с начала такта, выровненного по первой доле
        # Если первая доля не на границе такта, считаем её началом такта
        start_time = first_beat_time - beat_phase
        
        # Начинаем с начала первого такта (но не раньше 0)
        current_time = max(0.0, start_time)
    else:
        # Если не удалось определить доли, начинаем с начала аудио
        current_time = 0.0
    
    # Генерируем границы тактов
    while current_time < duration:
        bar_boundaries.append(current_time)
        current_time += bar_duration
    
    # Добавляем последнюю границу, если необходимо
    if len(bar_boundaries) == 0:
        bar_boundaries = [0.0, duration]
    elif bar_boundaries[-1] < duration:
        bar_boundaries.append(duration)
    
    bar_boundaries = np.array(bar_boundaries)
    n_bars = len(bar_boundaries) - 1
    
    logger.info(f"Detected {n_bars} bars, bar duration: {bar_duration:.3f}s")
    
    # CQT + хрома
    if fmin is None:
        fmin = librosa.note_to_hz('C1')
    
    n_bins = bins_per_octave * n_octaves
    C_mag = np.abs(librosa.cqt(x, sr=sr, hop_length=hop_len,
                               fmin=fmin, n_bins=n_bins,
                               bins_per_octave=bins_per_octave))**2
    
    # Получение временных меток для каждого кадра CQT
    times = librosa.frames_to_time(np.arange(C_mag.shape[1]), sr=sr, hop_length=hop_len)
    
    # Суммирование всех октав → 12-мерная хрома
    chroma = np.zeros((12, C_mag.shape[1]), dtype=np.float32)
    for pc in range(12):
        chroma[pc] = C_mag[pc::bins_per_octave].sum(axis=0)
    
    # Агрегация хрома векторов по тактам
    bar_embeddings = []
    
    for i in range(n_bars):
        bar_start = bar_boundaries[i]
        bar_end = bar_boundaries[i + 1]
        
        # Находим индексы кадров, попадающих в этот такт
        mask = (times >= bar_start) & (times < bar_end)
        
        if np.sum(mask) == 0:
            # Если нет кадров в этом такте, используем ближайший кадр
            if i > 0:
                # Берём последний кадр предыдущего такта или первый доступный
                prev_mask = (times >= bar_boundaries[max(0, i-1)]) & (times < bar_start)
                if np.sum(prev_mask) > 0:
                    bar_chroma = chroma[:, prev_mask][:, -1:]
                elif chroma.shape[1] > 0:
                    # Используем последний доступный кадр
                    last_idx = np.searchsorted(times, bar_start) - 1
                    if last_idx >= 0 and last_idx < chroma.shape[1]:
                        bar_chroma = chroma[:, last_idx:last_idx+1]
                    else:
                        bar_chroma = np.zeros((12, 1))
                else:
                    bar_chroma = np.zeros((12, 1))
            else:
                # Для первого такта берём первый доступный кадр
                first_idx = np.searchsorted(times, bar_end)
                if first_idx < chroma.shape[1]:
                    bar_chroma = chroma[:, first_idx:first_idx+1]
                else:
                    bar_chroma = np.zeros((12, 1))
        else:
            bar_chroma = chroma[:, mask]
        
        # Агрегация внутри такта
        if aggregation == 'mean':
            bar_vector = np.mean(bar_chroma, axis=1)
            bar_embeddings.append(bar_vector)
        elif aggregation == 'sum':
            bar_vector = np.sum(bar_chroma, axis=1)
            bar_embeddings.append(bar_vector)
        elif aggregation == 'root':
            mean_vec = np.mean(bar_chroma, axis=1)
            root_idx = int(np.argmax(mean_vec))
            masked = np.zeros_like(mean_vec)
            masked[root_idx] = mean_vec[root_idx]
            bar_embeddings.append(masked)
        elif aggregation == 'top3':
            mean_vec = np.mean(bar_chroma, axis=1)
            top_idx = np.argsort(mean_vec)[-3:][::-1].astype(int)
            masked = np.zeros_like(mean_vec)
            masked[top_idx] = mean_vec[top_idx]
            bar_embeddings.append(masked)
        elif aggregation == 'top4':
            mean_vec = np.mean(bar_chroma, axis=1)
            top_idx = np.argsort(mean_vec)[-4:][::-1].astype(int)
            masked = np.zeros_like(mean_vec)
            masked[top_idx] = mean_vec[top_idx]
            bar_embeddings.append(masked)
        else:
            raise ValueError(
                f"Unknown aggregation method: {aggregation}. "
                f"Use 'mean','sum','max','root','top3','top4'"
            )
    
    # Всегда возвращаем (n_bars, 12) для согласованности размерности
    return np.array(bar_embeddings, dtype=np.float32)

