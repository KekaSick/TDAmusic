"""
test_iaaft.py — верификация IAAFT-суррогата на синтетике.

Три компоненты:
  1. Синусоида (линейная динамика)
  2. Логистическое отображение (нелинейная динамика)
  3. Случайное блуждание (стохастический процесс)

Проверки:
  (1) sort(surr[:,c]) == sort(x[:,c]) для каждого c  — распределение сохранено ТОЧНО
  (2) относительная спектральная ошибка < 2%
"""
import sys; sys.path.insert(0, "src")
import numpy as np
import controls

T = 512
d = 3
rng = np.random.default_rng(42)

# --- Синтетика ---
x = np.zeros((T, d))

# 1. Синусоида
t = np.linspace(0, 8 * np.pi, T)
x[:, 0] = np.sin(t) + 0.3 * np.sin(3 * t)

# 2. Логистическое отображение (r=3.9, хаос)
logistic = np.empty(T)
logistic[0] = 0.1
for i in range(1, T):
    logistic[i] = 3.9 * logistic[i-1] * (1 - logistic[i-1])
x[:, 1] = logistic

# 3. Случайное блуждание
x[:, 2] = rng.standard_normal(T).cumsum()

# --- IAAFT ---
surr = controls.iaaft_surrogate(x, rng, n_iter=200, tol=1e-8)

print(f"Вход:  shape={x.shape}")
print(f"Выход: shape={surr.shape}")
print()

# --- Проверка 1: распределение значений ---
print("=== Проверка 1: распределение значений (sort(surr) == sort(x)) ===")
names = ["синус", "логист. отображение", "случ. блуждание"]
all_dist_ok = True
for c in range(d):
    match = np.allclose(np.sort(surr[:, c]), np.sort(x[:, c]), atol=1e-12)
    status = "✓ ТОЧНО" if match else "✗ ОТЛИЧАЕТСЯ"
    if not match:
        all_dist_ok = False
    print(f"  [{c}] {names[c]:>25s}: {status}")

assert all_dist_ok, "Распределение значений НЕ сохранено!"
print("  → Все компоненты: распределение сохранено ТОЧНО\n")

# --- Проверка 2: спектральная ошибка ---
print("=== Проверка 2: относительная спектральная ошибка ===")
from numpy.fft import rfft
all_spec_ok = True
for c in range(d):
    amp_orig = np.abs(rfft(x[:, c]))
    amp_surr = np.abs(rfft(surr[:, c]))
    rel_err = np.sqrt(np.sum((amp_surr - amp_orig)**2) / np.sum(amp_orig**2 + 1e-30))
    pct = rel_err * 100
    status = "✓" if pct < 2.0 else "✗"
    if pct >= 2.0:
        all_spec_ok = False
    print(f"  [{c}] {names[c]:>25s}: {pct:.4f}%  {status}")

assert all_spec_ok, "Спектральная ошибка >= 2%!"
print("  → Все компоненты: спектральная ошибка < 2%\n")

print("=== ВСЕ ПРОВЕРКИ IAAFT ПРОЙДЕНЫ ===")
