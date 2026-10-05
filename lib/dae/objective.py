"""Funzioni pure per objective AUC-PR, filtro non-learning e geometria DAE."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score


def seed_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average Precision per seme; NaN se manca una delle classi."""
    y_true = np.asarray(y_true)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0 or n_pos == len(y_true):
        return float("nan")
    return float(average_precision_score(y_true, np.asarray(y_score)))


def is_non_learning(history_ap: Sequence[float], prevalence: float,
                    min_epochs: int, margin: float) -> bool:
    """True se, dopo `min_epochs`, il miglior AUC-PR non supera la baseline con margine."""
    if len(history_ap) < int(min_epochs):
        return False
    valid = [float(v) for v in history_ap if v is not None and math.isfinite(float(v))]
    threshold = float(prevalence) * (1.0 + float(margin))
    if not valid:
        return True
    return max(valid) <= threshold


def aggregate_seeds(per_seed_ap: Sequence[float], k_std: float) -> float:
    """Mean(AUC-PR) - k_std * std sui semi validi; NaN se non ce ne sono."""
    genuine = [float(v) for v in per_seed_ap if v is not None and math.isfinite(float(v))]
    if not genuine:
        return float("nan")
    arr = np.asarray(genuine, dtype=np.float64)
    return float(arr.mean() - float(k_std) * arr.std(ddof=0))


def is_valid_geometry(exp: int, mid: int, btl: int) -> bool:
    """True se btl < mid < exp."""
    return bool(int(btl) < int(mid) < int(exp))
