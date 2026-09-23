"""lib/dae/objective.py — core threshold-free della ricerca DAE (parti pure).

Sostituisce la logica MCC@p99 di v5. Niente soglia, niente top-3: il segnale di
selezione e stop e' l'Average Precision (AUC-PR), adatta allo sbilanciamento.

  - seed_ap            : AUC-PR per-seme (NaN sui casi degeneri a una sola classe);
  - is_non_learning    : filtro non-apprendimento ancorato alla PREVALENZA misurata
                         sul val (l'AUC-PR di un classificatore casuale ≈ prevalenza, NON 0);
  - aggregate_seeds    : objective = mean(AUC-PR) − K·std(AUC-PR) sui semi genuini (la penalita'
                         −std e' varianza inter-seme = robustezza, non parsimonia);
  - is_valid_geometry  : vincolo architetturale btl < mid < exp (no config degeneri).

Funzioni pure (NumPy + sklearn), deterministiche, verificabili bit-exact.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score


def seed_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average Precision (AUC-PR) per un seme. NaN se una sola classe e' presente
    (AUC-PR non definita): cosi' i semi degeneri sono filtrabili a valle."""
    y_true = np.asarray(y_true)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0 or n_pos == len(y_true):
        return float("nan")
    return float(average_precision_score(y_true, np.asarray(y_score)))


def is_non_learning(history_ap: Sequence[float], prevalence: float,
                    min_epochs: int, margin: float) -> bool:
    """True se, dopo almeno `min_epochs` epoche, il miglior AUC-PR osservato non supera
    la baseline casuale `prevalence * (1 + margin)`: il modello non ha imparato nulla
    oltre il rumore. I valori non-finiti nella history sono ignorati."""
    if len(history_ap) < int(min_epochs):
        return False
    valid = [float(v) for v in history_ap if v is not None and math.isfinite(float(v))]
    threshold = float(prevalence) * (1.0 + float(margin))
    if not valid:
        return True  # nessun AUC-PR valido dopo min_epochs epoche = non apprende
    return max(valid) <= threshold


def aggregate_seeds(per_seed_ap: Sequence[float], k_std: float) -> float:
    """Objective Optuna: mean(AUC-PR) − k_std·std(AUC-PR) sui semi genuini (AUC-PR finiti).
    std popolazionale (ddof=0). NaN se nessun seme genuino."""
    genuine = [float(v) for v in per_seed_ap if v is not None and math.isfinite(float(v))]
    if not genuine:
        return float("nan")
    arr = np.asarray(genuine, dtype=np.float64)
    return float(arr.mean() - float(k_std) * arr.std(ddof=0))


def is_valid_geometry(exp: int, mid: int, btl: int) -> bool:
    """Vincolo di sensatezza architetturale: compressione stretta btl < mid < exp."""
    return bool(int(btl) < int(mid) < int(exp))
