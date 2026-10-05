"""Helper puri per ordinamento dei semi e selezione del vincitore."""
from __future__ import annotations

from typing import Any

import numpy as np


def seed_order_from_trial_id(trial_id_str: str, seeds: list[int]) -> tuple[int, ...]:
    """Permuta deterministicamente i semi usando l'ID del trial Ray."""
    trial_seed_int = int(trial_id_str.split("_")[0][:8], 16) % (2 ** 31)
    rng = np.random.default_rng(trial_seed_int)
    return tuple(int(s) for s in rng.permutation(list(seeds)))


def select_winner(analysis: Any, metric_name: str, max_t: int,
                  mode: str = "max") -> dict:
    """Seleziona il miglior trial completo (`seeds_completed == max_t`)."""
    trials = list(getattr(analysis, "trials", []) or [])
    eligible = []
    for t in trials:
        lr = getattr(t, "last_result", None) or {}
        sc = lr.get("seeds_completed")
        if sc is None or int(sc) != int(max_t):
            continue
        val = lr.get(metric_name)
        if val is None:
            continue
        eligible.append((t, float(val)))

    n_total = len(trials)
    if not eligible:
        return {"trial_id": None, "config": None, "objective": None,
                "seeds_completed": None, "n_complete": 0, "n_total": n_total}

    keyfn = (lambda kv: kv[1]) if mode == "max" else (lambda kv: -kv[1])
    best_t, best_val = max(eligible, key=keyfn)
    return {
        "trial_id": getattr(best_t, "trial_id", None),
        "config": getattr(best_t, "config", None),
        "objective": best_val,
        "seeds_completed": int(max_t),
        "n_complete": len(eligible),
        "n_total": n_total,
    }
