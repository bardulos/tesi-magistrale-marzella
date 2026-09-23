"""lib/dae/threshold.py — soglia di anomalia tau (repov6), sorgente unica.

tau = percentile (1 - FPR_TARGET) degli score sui benigni di validazione (p99 per FPR=1%).
È l'unico punto che calcola la soglia: il LOAO (lib.dae.loao) la chiama qui, e la
riusano le metriche del secondo stadio. Calcolo deterministico, NumPy puro (nessun TF).
"""
from __future__ import annotations

import numpy as np

from lib.dae.constants import FPR_TARGET


def compute_threshold(err_val_benign, fpr_target: float = FPR_TARGET) -> float:
    """Percentile (1 - fpr_target) degli score benigni di validazione (FPR target = fpr_target)."""
    return float(np.percentile(np.asarray(err_val_benign, dtype=float), 100 * (1 - fpr_target)))
