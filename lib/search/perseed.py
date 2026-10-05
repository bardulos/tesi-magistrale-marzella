"""Ricostruisce metriche per-seme da medie cumulative nei result.json.

Associa l'ordine di esecuzione ai semi fisici usando il trial ID. La lista `seeds`
deve mantenere l'ordine usato dal motore.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lib.search.selection import seed_order_from_trial_id


def reconstruct_seed_values(records, mean_key: str, n_key: str = "n_genuine") -> tuple[list, bool]:
    """Ricostruisce i valori per-seme da una media cumulativa ordinata per completamento."""
    values, prev_m, prev_n, clean = [], 0.0, 0, True
    for r in records:
        n = int(r[n_key]); m = float(r[mean_key])
        if n != prev_n + 1:
            clean = False
        values.append(n * m - prev_n * prev_m)
        prev_m, prev_n = m, n
    return values, clean


def reconstruct_seed_aps(records) -> tuple[list, bool]:
    """Ricostruisce l'AUC-PR per seme in ordine di esecuzione.

    clean=False se un seme e' stato filtrato come non-apprendente (n_genuine non cresce di 1).
    """
    return reconstruct_seed_values(records, mean_key="ap_mean")


def hex_from_trial_dir(name: str) -> str:
    """hex8 del trial-id dal nome cartella Ray '_run_trainable_<hex8>_<idx>_...'."""
    tail = name.split("_run_trainable_", 1)[-1]
    return tail.split("_")[0][:8]


def map_to_physical_seeds(aps_run_order, trial_hex: str, seeds_engine_order) -> dict:
    """Mappa le metriche ai semi fisici; `seeds_engine_order` deve seguire l'ordine motore."""
    order = seed_order_from_trial_id(trial_hex, list(seeds_engine_order))
    return {int(order[i]): float(ap) for i, ap in enumerate(aps_run_order)}


def load_result_records(result_json) -> list:
    """Righe (dict) del result.json (JSONL), in ordine; [] se assente/illeggibile."""
    try:
        return [json.loads(l) for l in Path(result_json).read_text().splitlines() if l.strip()]
    except Exception:  # noqa: BLE001
        return []


def build_matrix(trial_dirs, seeds_engine_order):
    """Costruisce la matrice per-seme dei trial COMPLETE e puliti."""
    seeds_engine_order = [int(s) for s in seeds_engine_order]
    seeds_sorted = sorted(seeds_engine_order)
    rows, kept, skipped = [], [], 0
    for d in trial_dirs:
        d = Path(d)
        recs = load_result_records(d / "result.json")
        if not recs or int(recs[-1].get("seeds_completed", 0)) != len(seeds_sorted):
            skipped += 1; continue
        aps, clean = reconstruct_seed_aps(recs)
        if not clean or len(aps) != len(seeds_sorted):
            skipped += 1; continue
        phys = map_to_physical_seeds(aps, hex_from_trial_dir(d.name), seeds_engine_order)
        if set(phys) != set(seeds_sorted):
            skipped += 1; continue
        rows.append([phys[s] for s in seeds_sorted])
        kept.append(d.name)
    return np.array(rows, dtype=np.float64), seeds_sorted, kept, skipped
