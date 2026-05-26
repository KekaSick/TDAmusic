"""
controls.py
-----------
Surrogate controls for frame-level trajectories.

Shuffle must be applied before Takens embedding. Shuffling an already-built
point cloud is not a temporal-order control because Vietoris-Rips is invariant
to point order.
"""
from __future__ import annotations
import numpy as np


def shuffle_frames(x: np.ndarray, rng) -> np.ndarray:
    """Permute the temporal order of all frames."""
    perm = rng.permutation(x.shape[0])
    return x[perm]


def block_shuffle_frames(x: np.ndarray, rng, block_size: int = 75) -> np.ndarray:
    """Shuffle blocks while preserving within-block local smoothness."""
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
    """Generate a random control, optionally via phase randomization."""
    T, d = x.shape
    if not match_autocorr:
        return rng.standard_normal((T, d))
        
    from numpy.fft import rfft, irfft
    
    X_f = rfft(x, axis=0)
    n_freqs = X_f.shape[0]
    
    phases = rng.uniform(0, 2 * np.pi, (n_freqs, 1))
    
    phases[0] = 0.0
    if T % 2 == 0:
        phases[-1] = 0.0
        
    X_f_randomized = X_f * np.exp(1j * phases)
    x_rand = irfft(X_f_randomized, n=T, axis=0)
    
    return x_rand


def iaaft_surrogate(x: np.ndarray, rng, n_iter: int = 200,
                    tol: float = 1e-8) -> np.ndarray:
    """IAAFT surrogate (Iterated Amplitude Adjusted Fourier Transform).

    Schreiber & Schmitz, Phys. Rev. Lett. 77, 635 (1996).

    Applied componentwise: it preserves each component's amplitude spectrum
    and empirical distribution, but not cross-component correlations.
    """
    from numpy.fft import rfft, irfft

    T, d = x.shape
    surr = np.empty_like(x)

    for c in range(d):
        col = x[:, c].copy()

        amp_target = np.abs(rfft(col))
        sorted_target = np.sort(col)

        current = rng.permutation(col)
        prev_err = np.inf

        for _ in range(n_iter):
            F = rfft(current)
            phases = F / np.maximum(np.abs(F), 1e-12)
            current = irfft(amp_target * phases, n=T)

            ranks = np.argsort(np.argsort(current))
            current = sorted_target[ranks]

            amp_current = np.abs(rfft(current))
            err = np.sqrt(np.sum((amp_current - amp_target) ** 2)
                          / np.sum(amp_target ** 2 + 1e-30))

            if prev_err > 1e-12 and abs(prev_err - err) < tol:
                break
            prev_err = err

        surr[:, c] = current

    return surr


def mean_pool(x: np.ndarray) -> np.ndarray:
    """Average frames into one vector."""
    return x.mean(axis=0)
