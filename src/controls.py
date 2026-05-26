"""
controls.py
-----------
ЯДРО РАБОТЫ. Три контроля, каждый убивает свою альтернативную гипотезу:

  shuffle    -> «топология от РАСПРЕДЕЛЕНИЯ точек, время ни при чём»
  random     -> «топология от ГЛАДКОСТИ ряда, музыка ни при чём»
  mean_pool  -> baseline полезности (не контроль времени)

КРИТИЧНО про shuffle:
  Перемешивать ИСХОДНЫЙ временно́й ряд кадров ДО построения окна Такенса,
  затем строить окно ЗАНОВО. Нельзя перемешивать готовые точки облака —
  Vietoris-Rips инвариантен к их порядку и проигнорирует это.
  *Block Shuffle* перемешивает трек блоками (например, по 3 секунды),
  чтобы сохранить локальную гладкость, но разрушить глобальную структуру.

КРИТИЧНО про random:
  Используется Phase Randomization. Делает Фурье-преобразование трека, 
  рандомизирует фазу (с сохранением симметрии) и делает обратное преобразование.
  Это математически гарантирует идеальное совпадение амплитудного спектра
  и автокорреляции с оригиналом, но полностью уничтожает смысл и структуру.
"""
from __future__ import annotations
import numpy as np


def shuffle_frames(x: np.ndarray, rng) -> np.ndarray:
    """Перемешать ВРЕМЕННО́Й порядок ВСЕХ кадров (T, d)."""
    perm = rng.permutation(x.shape[0])
    return x[perm]


def block_shuffle_frames(x: np.ndarray, rng, block_size: int = 75) -> np.ndarray:
    """Перемешать ряд блоками. Сохраняет локальную структуру внутри блока."""
    T, d = x.shape
    n_blocks = T // block_size
    remainder = T % block_size
    
    if n_blocks <= 1:
        return x.copy()
        
    blocks = [x[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    perm = rng.permutation(n_blocks)
    shuffled_blocks = [blocks[i] for i in perm]
    
    if remainder > 0:
        shuffled_blocks.append(x[n_blocks*block_size:])
        
    return np.concatenate(shuffled_blocks, axis=0)


def random_like(x: np.ndarray, rng, match_autocorr: bool = True) -> np.ndarray:
    """Случайный ряд. Если match_autocorr=True, использует Phase Randomization."""
    T, d = x.shape
    if not match_autocorr:
        return rng.standard_normal((T, d))
        
    # Phase Randomization
    from numpy.fft import rfft, irfft
    
    # Считаем спектр. rfft возвращает только положительные частоты, 
    # что гарантирует, что при обратном irfft сигнал останется вещественным.
    X_f = rfft(x, axis=0)
    n_freqs = X_f.shape[0]
    
    # Генерируем случайные фазы равномерно от 0 до 2pi
    phases = rng.uniform(0, 2 * np.pi, (n_freqs, 1))
    
    # Нулевая частота (DC) не должна иметь фазы
    phases[0] = 0.0
    # Частота Найквиста (если T четное) тоже должна быть вещественной
    if T % 2 == 0:
        phases[-1] = 0.0
        
    # Применяем фазовый сдвиг
    X_f_randomized = X_f * np.exp(1j * phases)
    
    # Обратное преобразование Фурье
    x_rand = irfft(X_f_randomized, n=T, axis=0)
    
    return x_rand


def iaaft_surrogate(x: np.ndarray, rng, n_iter: int = 200,
                    tol: float = 1e-8) -> np.ndarray:
    """IAAFT-суррогат (Iterated Amplitude Adjusted Fourier Transform).

    Schreiber & Schmitz, Phys. Rev. Lett. 77, 635 (1996).

    Третья ступень лестницы контролей:
      1. shuffle       — убивает временну́ю структуру целиком
      2. phase random  — сохраняет амплитудный спектр (≡ автокорреляцию),
                         но навязывает гауссово распределение
      3. IAAFT         — сохраняет И амплитудный спектр, И эмпирическое
                         распределение значений одновременно

    Отличие реальных данных от IAAFT-суррогата изолирует
    НЕЛИНЕЙНУЮ динамическую структуру.

    ВАЖНО: классический IAAFT одномерный и применяется здесь ПОКОМПОНЕНТНО
    (по каждому столбцу отдельно). Поэтому, в отличие от phase-суррогата
    с общими фазами (random_like), он НЕ сохраняет кросс-корреляции между
    PCA-компонентами. Для теста на нелинейность этого достаточно, но это
    необходимо оговорить в статье.

    Parameters
    ----------
    x : ndarray, shape (T, d)
        Многомерный временно́й ряд.
    rng : numpy Generator
        Источник случайности.
    n_iter : int
        Максимальное число итераций IAAFT.
    tol : float
        Порог сходимости по относительной спектральной ошибке.

    Returns
    -------
    surr : ndarray, shape (T, d)
        IAAFT-суррогат.
    """
    from numpy.fft import rfft, irfft

    T, d = x.shape
    surr = np.empty_like(x)

    for c in range(d):
        col = x[:, c].copy()

        # Целевые инварианты
        amp_target = np.abs(rfft(col))
        sorted_target = np.sort(col)

        # Старт: случайная перестановка
        current = rng.permutation(col)
        prev_err = np.inf

        for _ in range(n_iter):
            # (a) Навязать амплитудный спектр, сохранив текущие фазы
            F = rfft(current)
            phases = F / np.maximum(np.abs(F), 1e-12)
            current = irfft(amp_target * phases, n=T)

            # (b) Навязать эмпирическое распределение через rank-matching
            ranks = np.argsort(np.argsort(current))
            current = sorted_target[ranks]

            # Проверка сходимости: относительная спектральная ошибка
            amp_current = np.abs(rfft(current))
            err = np.sqrt(np.sum((amp_current - amp_target) ** 2)
                          / np.sum(amp_target ** 2 + 1e-30))

            if prev_err > 1e-12 and abs(prev_err - err) < tol:
                break
            prev_err = err

        surr[:, c] = current

    return surr


def mean_pool(x: np.ndarray) -> np.ndarray:
    """Усреднение по кадрам. (T, d) -> (d,)"""
    return x.mean(axis=0)
