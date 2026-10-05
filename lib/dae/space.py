"""Spazio di ricerca DAE con `mid` condizionato a `exp` e `btl`."""
from __future__ import annotations

from lib.dae.constants import (BTL_HIGH, BTL_LOW, EXP_HIGH, EXP_LOW, LR_HIGH,
                               LR_LOW, MID_HIGH, MID_LOW, SIGMA_HIGH, SIGMA_LOW)


def mid_range(exp: int, btl: int) -> tuple[int, int]:
    """Intervallo di `mid` che garantisce btl < mid < exp."""
    lo = max(int(btl) + 1, MID_LOW)
    hi = min(MID_HIGH, int(exp) - 1)
    return lo, hi


def sample_dae_space(trial) -> dict:
    """Campiona exp, btl, mid, sigma e lr con geometria valida."""
    exp = trial.suggest_int("exp", EXP_LOW, EXP_HIGH)
    btl = trial.suggest_int("btl", BTL_LOW, BTL_HIGH)
    lo, hi = mid_range(exp, btl)
    mid = trial.suggest_int("mid", lo, hi)
    sigma = trial.suggest_float("sigma", SIGMA_LOW, SIGMA_HIGH)
    lr = trial.suggest_float("lr", LR_LOW, LR_HIGH, log=True)
    return {"exp": exp, "btl": btl, "mid": mid, "sigma": sigma, "lr": lr}
