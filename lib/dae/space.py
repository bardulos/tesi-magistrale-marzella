"""lib/dae/space.py — spazio di ricerca degli iperparametri DAE (repov6).

btl CERCATO [6,16] e exp [40,256] (allargati dove premevano i bordi). Il range di `mid` e'
condizionato a (exp, btl) cosi' da garantire SEMPRE la compressione stretta btl < mid < exp,
senza config degeneri da scartare. mid_range e' pura; sample_dae_space (no TF, no stato globale).
"""
from __future__ import annotations

from lib.dae.constants import (BTL_HIGH, BTL_LOW, EXP_HIGH, EXP_LOW, LR_HIGH,
                               LR_LOW, MID_HIGH, MID_LOW, SIGMA_HIGH, SIGMA_LOW)


def mid_range(exp: int, btl: int) -> tuple[int, int]:
    """Estremi ammessi per `mid` dato (exp, btl): [max(btl+1, MID_LOW), min(MID_HIGH, exp-1)].
    Per exp in [EXP_LOW=40, EXP_HIGH] e btl in [BTL_LOW, BTL_HIGH=16] il range e' non vuoto e
    ogni mid ammesso soddisfa btl < mid < exp."""
    lo = max(int(btl) + 1, MID_LOW)
    hi = min(MID_HIGH, int(exp) - 1)
    return lo, hi


def sample_dae_space(trial) -> dict:
    """Campiona gli iperparametri cercati da un trial Optuna (o compatibile).
    Ritorna {exp, btl, mid, sigma, lr}. mid e' condizionato a (exp, btl) -> geometria valida."""
    exp = trial.suggest_int("exp", EXP_LOW, EXP_HIGH)
    btl = trial.suggest_int("btl", BTL_LOW, BTL_HIGH)
    lo, hi = mid_range(exp, btl)
    mid = trial.suggest_int("mid", lo, hi)
    sigma = trial.suggest_float("sigma", SIGMA_LOW, SIGMA_HIGH)
    lr = trial.suggest_float("lr", LR_LOW, LR_HIGH, log=True)
    return {"exp": exp, "btl": btl, "mid": mid, "sigma": sigma, "lr": lr}
