"""lib/search/warmstart.py — config di warm-start per la search allargata (enqueue batch).

Strategia (decisione di piano, opzione "enqueue batch"): un nuovo studio TPE viene pre-caricato
con una coda di config da eseguire come primi trial; l'import della storia come prior (add_trial
via import_history) e' OPZIONALE, attivato dal config (es. search_extended: import_history: true).

Due famiglie:
  - TRASLATE: config forti (top per objective) spostate verso i bordi allargati (btl 12->13..16,
    exp ->232..256), tenendo mid/sigma/lr ottimizzati dalla search — diversita' (molte basi diverse)
    piu' che densita'. Portano la conoscenza pregressa nella regione nuova.
  - RANDOM: campionamento uniforme che cade SEMPRE nella regione nuova (btl>12 OPPURE exp>224),
    a coprire alla cieca cio' che le traslate non toccano.

Funzioni pure (translate_from_top/random_region/build_enqueue): nessun I/O, nessun Optuna, testabili.
load_completed/load_all_trials leggono il DB pregresso; import_history scrive i FrozenTrial nel
nuovo studio. Ogni config prodotta ha geometria valida (mid clippato -> btl < mid < exp).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from lib.dae.constants import (BTL_HIGH, BTL_LOW, EXP_HIGH, EXP_LOW, LR_HIGH,
                               LR_LOW, SIGMA_HIGH, SIGMA_LOW)
from lib.dae.space import mid_range

BTL_NEW = [13, 14, 15, 16]          # livelli di bottleneck oltre il vecchio tetto (12)
EXP_NEW = [232, 240, 248, 256]      # livelli di exp oltre il vecchio tetto (224)


def _clip_mid(exp: int, btl: int, mid: int) -> int:
    """mid clippato nel range condizionato [mid_range] -> garantisce btl < mid < exp."""
    lo, hi = mid_range(int(exp), int(btl))
    return int(max(lo, min(hi, int(mid))))


def _cfg(exp, mid, btl, sigma, lr) -> dict:
    return {"exp": int(exp), "mid": _clip_mid(exp, btl, mid), "btl": int(btl),
            "sigma": float(sigma), "lr": float(lr)}


def translate_from_top(completed: list[dict], n_btl=16, n_exp=10, n_both=8, n_leader=6,
                       rng_seed=0) -> list[dict]:
    """Config TRASLATE verso la regione allargata, da `completed` (con chiave 'value'). Sposta i
    bordi tenendo (mid clippato, sigma, lr); rng_seed inerte. Top per value desc (gruppo exp: exp desc)."""
    top = sorted(completed, key=lambda c: -float(c.get("value", 0.0)))
    if not top:
        return []
    out: list[dict] = []

    # btl-traslate: i top spostati a btl 13..16 (ciclico), exp/sigma/lr di origine
    for i, c in enumerate(top[:n_btl]):
        btl = BTL_NEW[i % len(BTL_NEW)]
        out.append(_cfg(c["exp"], c["mid"], btl, c["sigma"], c["lr"]))

    # exp-traslate: i top con exp piu' alto spostati a 232..256 (ciclico), btl/sigma/lr di origine
    by_exp = sorted(completed, key=lambda c: -int(c["exp"]))
    for i, c in enumerate(by_exp[:n_exp]):
        exp = EXP_NEW[i % len(EXP_NEW)]
        out.append(_cfg(exp, c["mid"], c["btl"], c["sigma"], c["lr"]))

    # both: top spostati su ENTRAMBI i bordi
    for i, c in enumerate(top[:n_both]):
        out.append(_cfg(EXP_NEW[i % len(EXP_NEW)], c["mid"], BTL_NEW[i % len(BTL_NEW)],
                        c["sigma"], c["lr"]))

    # leader-traslate: i top assoluti, btl piccolo-passo (12,13,14) per il ponte dal vincitore
    lead_btls = [12, 13, 14]
    n_cycle = min(2, len(top))   # >= 1: il guard iniziale esclude top vuoto
    for i in range(n_leader):
        c = top[i % n_cycle]
        btl = lead_btls[i % len(lead_btls)]
        out.append(_cfg(c["exp"], c["mid"], btl, c["sigma"], c["lr"]))

    return out


def random_region(n: int, rng_seed=0) -> list[dict]:
    """`n` config RANDOM che cadono SEMPRE nella regione nuova (btl>12 OPPURE exp>224).
    Metà forza il ramo btl alto, metà il ramo exp alto; resto uniforme nello spazio valido."""
    rng = np.random.default_rng(rng_seed)
    out: list[dict] = []
    for k in range(int(n)):
        if k % 2 == 0:                       # ramo btl alto
            btl = int(rng.integers(13, BTL_HIGH + 1))
            exp = int(rng.integers(EXP_LOW, EXP_HIGH + 1))
        else:                                # ramo exp alto
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
    """Coda completa di warm-start: traslate (dai top) + random (regione nuova), deduplicata.
    Deterministica a `rng_seed` fisso."""
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
    """COMPLETE+PRUNED del DB pregresso con (params, value, state) — per importarli come STORIA
    (osservazioni del TPE, non ri-eseguiti). PRUNED inclusi: hanno il valore-al-rung."""
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
    """FrozenTrial Optuna (COMPLETE/PRUNED col loro value) pronti per study.add_trial. Le distribuzioni
    sono RI-DICHIARATE coi NUOVI bound (exp[EXP_LOW,EXP_HIGH], btl[BTL_LOW,BTL_HIGH], mid condizionale,
    sigma, lr) — così il TPE le tratta come lo STESSO parametro che campiona nei nuovi trial. Importa
    la storia come osservazioni: NON ri-esegue i trial."""
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
    """Importa la storia (COMPLETE+PRUNED del DB pregresso) nello studio via add_trial. Ritorna il
    numero di trial importati."""
    fts = build_import_trials(load_all_trials(db_path, study_name))
    for ft in fts:
        study.add_trial(ft)
    return len(fts)
