"""
classify.py
-----------
Deprecated helper kept for compatibility with run_pipeline.py.

Final paper classification is implemented in scripts/run_classify_betti.py.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def evaluate(X_tr, y_tr, X_te, y_te, n_boot=500, rng=None):
    """Train logistic regression and return bootstrap CIs on the test set."""
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
    Return metrics for each feature set."""
    out = {}
    for name, X in feature_sets.items():
        out[name] = evaluate(X[tr_idx], y_tr, X[te_idx], y_te)
    return out
