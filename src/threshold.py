from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve

TARGET_PRECISION = 0.98
MIN_ALERTS = 20     # a threshold backed by only a handful of validation alerts is noise


def pick_strict_threshold(y_val, proba_val, target: float = TARGET_PRECISION, min_alerts: int = MIN_ALERTS):
    """Lowest score whose validation precision is >= target and that raises enough alerts.

    Chosen on the validation set only. Returns (threshold, found). If no cut-off reaches the
    target, falls back to 0.5 and found=False so the report can say so.
    """
    y_val, proba_val = np.asarray(y_val), np.asarray(proba_val)
    prec, _, thr = precision_recall_curve(y_val, proba_val)
    n_alerts = np.array([(proba_val >= t).sum() for t in thr])
    ok = np.where((prec[:-1] >= target) & (n_alerts >= min_alerts))[0]
    return (float(thr[ok[0]]), True) if len(ok) else (0.5, False)
