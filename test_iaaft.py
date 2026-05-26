"""Verify the IAAFT surrogate on synthetic signals.

Checks:
  (1) sort(surr[:, c]) == sort(x[:, c]) for each component.
  (2) relative spectral error < 2%.
"""
import sys; sys.path.insert(0, "src")
import numpy as np
import controls

T = 512
d = 3
rng = np.random.default_rng(42)

x = np.zeros((T, d))

t = np.linspace(0, 8 * np.pi, T)
x[:, 0] = np.sin(t) + 0.3 * np.sin(3 * t)

logistic = np.empty(T)
logistic[0] = 0.1
for i in range(1, T):
    logistic[i] = 3.9 * logistic[i-1] * (1 - logistic[i-1])
x[:, 1] = logistic

x[:, 2] = rng.standard_normal(T).cumsum()

surr = controls.iaaft_surrogate(x, rng, n_iter=200, tol=1e-8)

print(f"Input:  shape={x.shape}")
print(f"Output: shape={surr.shape}")
print()

print("=== Check 1: value distribution (sort(surr) == sort(x)) ===")
names = ["sine", "logistic map", "random walk"]
all_dist_ok = True
for c in range(d):
    match = np.allclose(np.sort(surr[:, c]), np.sort(x[:, c]), atol=1e-12)
    status = "OK" if match else "FAIL"
    if not match:
        all_dist_ok = False
    print(f"  [{c}] {names[c]:>25s}: {status}")

assert all_dist_ok, "Value distribution was not preserved"
print("  All components preserve the value distribution exactly\n")

print("=== Check 2: relative spectral error ===")
from numpy.fft import rfft
all_spec_ok = True
for c in range(d):
    amp_orig = np.abs(rfft(x[:, c]))
    amp_surr = np.abs(rfft(surr[:, c]))
    rel_err = np.sqrt(np.sum((amp_surr - amp_orig)**2) / np.sum(amp_orig**2 + 1e-30))
    pct = rel_err * 100
    status = "OK" if pct < 2.0 else "FAIL"
    if pct >= 2.0:
        all_spec_ok = False
    print(f"  [{c}] {names[c]:>25s}: {pct:.4f}%  {status}")

assert all_spec_ok, "Spectral error >= 2%"
print("  All components have spectral error < 2%\n")

print("=== ALL IAAFT CHECKS PASSED ===")
