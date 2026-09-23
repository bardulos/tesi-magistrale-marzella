"""lib/search/perseed.py — ricostruzione degli AUC-PR (e altre metriche) per-seme dai result.json.

Gli AUC-PR per-seme NON sono salvati esplicitamente, ma il result.json (JSONL, una riga per seme)
riporta la media cumulativa `ap_mean`: il singolo AUC-PR si ricostruisce per differenza
    AP_k = k·ap_mean_k − (k−1)·ap_mean_{k−1}
(valido finché tutti i semi sono genuini: `n_genuine` cresce di 1 a ogni passo).
`reconstruct_seed_values` generalizza la stessa telescopia a QUALUNQUE media cumulativa loggata
a ogni seme (es. mcc_fpr1_mean del secondo stadio): usato da tools/monitor_search3.py per la
σ per-seme di metriche che il componente riporta solo come media (mai materializzate per-seme).

L'ordine ricostruito è quello di ESECUZIONE; il mapping al seme FISICO è deterministico dal
trial-id via `seed_order_from_trial_id` — la STESSA funzione che il motore usa (engine.py:129),
validata 9/9 contro gli `seed_order.json` reali. ATTENZIONE: l'ordine dipende dall'ordine della
lista `seeds` passata (rng.permutation): va passata SEMPRE in ordine-motore (C.SEEDS), non
ordinata, altrimenti il mapping è silenziosamente sbagliato.

Condiviso, fra gli altri, da tools/extract_perseed_ap.py, shortlist_sizing.py, monitor_search.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lib.search.selection import seed_order_from_trial_id


def reconstruct_seed_values(records, mean_key: str, n_key: str = "n_genuine") -> tuple[list, bool]:
    """Generalizza reconstruct_seed_aps a QUALUNQUE media cumulativa loggata a ogni seme
    (es. 'mcc_fpr1_mean', 'mcc_bal_mean', non solo 'ap_mean'): value_k = n_k·mean_k − n_{k-1}·mean_{k-1}
    dai report (dict con `n_key` e `mean_key`, ordinati per seeds_completed). Ritorna
    (values, clean): clean=False se `n_key` non cresce di 1 a ogni passo (seme filtrato
    non-apprendente in mezzo) — in quel passo il valore ricostruito è 0 (nessuna crescita del
    genuino), da SCARTARE lato consumatore se serve la sola sequenza dei semi genuini."""
    values, prev_m, prev_n, clean = [], 0.0, 0, True
    for r in records:
        n = int(r[n_key]); m = float(r[mean_key])
        if n != prev_n + 1:
            clean = False
        values.append(n * m - prev_n * prev_m)
        prev_m, prev_n = m, n
    return values, clean


def reconstruct_seed_aps(records) -> tuple[list, bool]:
    """AUC-PR per-seme in ordine di ESECUZIONE dai report cumulativi (dict con 'n_genuine' e
    'ap_mean', ordinati per seeds_completed). Ritorna (aps, clean): clean=False se un seme è
    stato filtrato non-apprendente (n_genuine non cresce di 1 a ogni passo)."""
    return reconstruct_seed_values(records, mean_key="ap_mean")


def hex_from_trial_dir(name: str) -> str:
    """hex8 del trial-id dal nome cartella Ray '_run_trainable_<hex8>_<idx>_...'."""
    tail = name.split("_run_trainable_", 1)[-1]
    return tail.split("_")[0][:8]


def map_to_physical_seeds(aps_run_order, trial_hex: str, seeds_engine_order) -> dict:
    """{seme_fisico: ap} mappando l'ordine di esecuzione via seed_order_from_trial_id.
    `seeds_engine_order` DEVE essere C.SEEDS nell'ordine del motore (non ordinato).
    len(aps) può essere < len(seeds) per i PRUNED (solo i primi semi del rung)."""
    order = seed_order_from_trial_id(trial_hex, list(seeds_engine_order))
    return {int(order[i]): float(ap) for i, ap in enumerate(aps_run_order)}


def load_result_records(result_json) -> list:
    """Righe (dict) del result.json (JSONL), in ordine; [] se assente/illeggibile."""
    try:
        return [json.loads(l) for l in Path(result_json).read_text().splitlines() if l.strip()]
    except Exception:  # noqa: BLE001
        return []


def build_matrix(trial_dirs, seeds_engine_order):
    """Matrice (M configs × n_seed) degli AUC-PR per-seme dei trial COMPLETE e puliti.
    Colonne = semi fisici in ordine CRESCENTE; righe = trial con tutti i semi genuini.
    Ritorna (matrix — shape (0,) se zero trial superstiti —, seeds_sorted, kept_names, skipped)."""
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
