"""Costruisce enqueue batch di warm-start e importa opzionalmente la storia Optuna.

Le config traslate dai trial top e quelle casuali coprono la regione allargata.
Ogni config mantiene la geometria `btl < mid < exp`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from lib.dae.constants import (BTL_HIGH, BTL_LOW, EXP_HIGH, EXP_LOW, LR_HIGH,
                               LR_LOW, SIGMA_HIGH, SIGMA_LOW)
from lib.dae.space import mid_range

BTL_NEW = [13, 14, 15, 16]
EXP_NEW = [232, 240, 248, 256]


def _clip_mid(exp: int, btl: int, mid: int) -> int:
    """Clippa mid all'intervallo valido per exp e btl."""
    lo, hi = mid_range(int(exp), int(btl))
    return int(max(lo, min(hi, int(mid))))


def _cfg(exp, mid, btl, sigma, lr) -> dict:
    return {"exp": int(exp), "mid": _clip_mid(exp, btl, mid), "btl": int(btl),
            "sigma": float(sigma), "lr": float(lr)}


def translate_from_top(completed: list[dict], n_btl=16, n_exp=10, n_both=8, n_leader=6,
                       rng_seed=0) -> list[dict]:
    """Trasla le config pregresse verso la regione allargata.

    Dai top per objective ricava config con btl 13..16, con btl 13..16 ed exp 232..256 e,
    per i due migliori, con btl 12..14; dai trial con exp piu' alto ricava config con exp
    232..256. sigma e lr restano quelli d'origine, mid viene clippato.
    """
    top = sorted(completed, key=lambda c: -float(c.get("value", 0.0)))
    if not top:
        return []
    out: list[dict] = []

    # Sposta i top sui valori btl estesi.
    for i, c in enumerate(top[:n_btl]):
        btl = BTL_NEW[i % len(BTL_NEW)]
        out.append(_cfg(c["exp"], c["mid"], btl, c["sigma"], c["lr"]))

    # Sposta i top per exp sui valori estesi.
    by_exp = sorted(completed, key=lambda c: -int(c["exp"]))
    for i, c in enumerate(by_exp[:n_exp]):
        exp = EXP_NEW[i % len(EXP_NEW)]
        out.append(_cfg(exp, c["mid"], c["btl"], c["sigma"], c["lr"]))

    # Sposta i top su entrambi i bordi.
    for i, c in enumerate(top[:n_both]):
        out.append(_cfg(EXP_NEW[i % len(EXP_NEW)], c["mid"], BTL_NEW[i % len(BTL_NEW)],
                        c["sigma"], c["lr"]))

    # Aggiunge passi piccoli di btl per i top assoluti.
    lead_btls = [12, 13, 14]
    n_cycle = min(2, len(top))
    for i in range(n_leader):
        c = top[i % n_cycle]
        btl = lead_btls[i % len(lead_btls)]
        out.append(_cfg(c["exp"], c["mid"], btl, c["sigma"], c["lr"]))

    return out


def random_region(n: int, rng_seed=0) -> list[dict]:
    """Genera config nella regione nuova, alternando btl alto ed exp alto."""
    rng = np.random.default_rng(rng_seed)
    out: list[dict] = []
    for k in range(int(n)):
        if k % 2 == 0:
            btl = int(rng.integers(13, BTL_HIGH + 1))
            exp = int(rng.integers(EXP_LOW, EXP_HIGH + 1))
        else:
            exp = int(rng.integers(225, EXP_HIGH + 1))
            btl = int(rng.integers(BTL_LOW, BTL_HIGH + 1))
        lo, hi = mid_range(exp, btl)
        mid = int(rng.integers(lo, hi + 1))
        sigma = float(rng.uniform(SIGMA_LOW, SIGMA_HIGH))
        lr = float(10.0 ** rng.uniform(np.log10(LR_LOW), np.log10(LR_HIGH)))
        out.append({"exp": exp, "mid": mid, "btl": btl, "sigma": sigma, "lr": lr})
    return out


def _dedup(cfgs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in cfgs:
        key = (c["exp"], c["mid"], c["btl"], round(c["sigma"], 9), round(c["lr"], 9))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def build_enqueue(completed: list[dict], n_random=80, rng_seed=0) -> list[dict]:
    """Unisce config traslate e casuali, rimuovendo i duplicati."""
    cfgs = translate_from_top(completed, rng_seed=rng_seed) + random_region(n_random, rng_seed + 1)
    return _dedup(cfgs)


def load_completed(db_path: str | Path, study_name: str) -> list[dict]:
    """Legge i trial COMPLETE del DB Optuna pregresso (read-only) -> [{exp,mid,btl,sigma,lr,value}]."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    abspath = Path(db_path).resolve()
    study = optuna.load_study(study_name=study_name,
                              storage=f"sqlite:///file:{abspath}?mode=ro&uri=true")
    out = []
    for t in study.trials:
        if t.state.name != "COMPLETE" or not t.params or t.value is None:
            continue
        p = t.params
        try:
            out.append({"exp": int(p["exp"]), "mid": int(p["mid"]), "btl": int(p["btl"]),
                        "sigma": float(p["sigma"]), "lr": float(p["lr"]), "value": float(t.value)})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def load_all_trials(db_path: str | Path, study_name: str) -> list[dict]:
    """Legge trial COMPLETE e PRUNED per importarli come storia Optuna."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    abspath = Path(db_path).resolve()
    study = optuna.load_study(study_name=study_name,
                              storage=f"sqlite:///file:{abspath}?mode=ro&uri=true")
    out = []
    for t in study.trials:
        if t.state.name not in ("COMPLETE", "PRUNED") or not t.params or t.value is None:
            continue
        p = t.params
        try:
            out.append({"params": {"exp": int(p["exp"]), "btl": int(p["btl"]), "mid": int(p["mid"]),
                                   "sigma": float(p["sigma"]), "lr": float(p["lr"])},
                        "value": float(t.value), "state": t.state.name})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def build_import_trials(records: list[dict]) -> list:
    """Crea FrozenTrial con le distribuzioni correnti, senza rieseguire i trial."""
    from optuna.distributions import FloatDistribution, IntDistribution
    from optuna.trial import TrialState, create_trial
    state_map = {"COMPLETE": TrialState.COMPLETE, "PRUNED": TrialState.PRUNED}
    out = []
    for r in records:
        p = r["params"]
        exp, btl, mid = int(p["exp"]), int(p["btl"]), int(p["mid"])
        lo, hi = mid_range(exp, btl)
        dists = {
            "exp": IntDistribution(EXP_LOW, EXP_HIGH),
            "btl": IntDistribution(BTL_LOW, BTL_HIGH),
            "mid": IntDistribution(lo, hi),
            "sigma": FloatDistribution(SIGMA_LOW, SIGMA_HIGH),
            "lr": FloatDistribution(LR_LOW, LR_HIGH, log=True),
        }
        params = {"exp": exp, "btl": btl, "mid": mid,
                  "sigma": float(p["sigma"]), "lr": float(p["lr"])}
        out.append(create_trial(params=params, distributions=dists, value=float(r["value"]),
                                state=state_map.get(r["state"], TrialState.COMPLETE)))
    return out


def import_history(study, db_path: str | Path, study_name: str) -> int:
    """Importa i trial pregressi e restituisce quanti ne sono stati aggiunti."""
    fts = build_import_trials(load_all_trials(db_path, study_name))
    for ft in fts:
        study.add_trial(ft)
    return len(fts)
