"""
classify.py
-----------
DEPRECATED / МЁРТВЫЙ КОД.
Этот модуль НЕ находится на критическом пути генерации результатов статьи.
Он импортируется только из run_pipeline.py (скелет-оркестратор), который
не производит финальных CSV/таблиц.

Вся downstream-классификация и подсчёт метрик (accuracy, F1, confusion matrix)
реализованы непосредственно в телах скриптов:
  - scripts/run_classify_betti.py
  - scripts/run_full_scale.py

Числа в results/tables/ получены НЕ через этот файл.
Оставлен для обратной совместимости с run_pipeline.py.
См. PROVENANCE.md §3 «Мёртвый код».

---
Оригинальное описание:
Downstream-проверка ПОЛЕЗНОСТИ. Намеренно простой классификатор (logreg) —
меряем качество ФИЧЕЙ, а не модели.

Сравниваем на одном сплите для КАЖДОГО пространства:
  persistence (метод)  vs  mean_pool (baseline)  vs
  persistence@shuffle (контроль)  vs  persistence@random (контроль)

Вывод курсовой — НЕ «побил SOTA», а «топология несёт сигнал выше random и
выше shuffle, и вот насколько; добавляет/не добавляет к mean-pool».
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def evaluate(X_tr, y_tr, X_te, y_te, n_boot=500, rng=None):
    """Обучить logreg, вернуть acc/f1 с bootstrap-CI по тесту."""
    rng = rng or np.random.default_rng(0)
    clf = LogisticRegression(max_iter=2000, multi_class="auto")
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc, f1 = accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro")
    accs, f1s = [], []
    n = len(y_te)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        accs.append(accuracy_score(y_te[idx], pred[idx]))
        f1s.append(f1_score(y_te[idx], pred[idx], average="macro"))
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"acc": acc, "acc_ci": ci(accs), "f1": f1, "f1_ci": ci(f1s)}


def compare_feature_sets(feature_sets: dict, y_tr, y_te, tr_idx, te_idx):
    """feature_sets: {'persistence': X, 'mean_pool': X, 'shuffle': X, ...}.
    Возвращает таблицу метрик по каждому набору."""
    out = {}
    for name, X in feature_sets.items():
        out[name] = evaluate(X[tr_idx], y_tr, X[te_idx], y_te)
    return out
