"""lib/secondo_stadio/loao.py — LOAO a RIADDESTRAMENTO del secondo stadio (P2 sup, P5 pa).

Fold leak-free per classe di attacco esclusa (a differenza del LOAO del DAE — cap2,
forward-only — qui il modello impara dagli attacchi o li usa in selezione: si RIADDESTRA):
  - ceiling (sup): la classe esce da TRAINING (att_train), metrica per-epoca (att_es) e
    selezione/soglia finale (att_select); benigni e soglia p99 (benigni-select) sono
    classe-invarianti. DR zero-day della classe misurata su VAL a tau=p99 benigni-select
    (GATE A D9: stesso set e stessa soglia del LOAO DAE cap2 -> confronto a parita'; il
    test resta vergine per fp-mining/fusione).
  - PA (P5): il training (benigni + pseudo-anomalie) e' classe-invariante; la classe esce
    dalla metrica per-epoca (att_es) e dallo sweep standalone della soglia (att_pos).
    DR su VAL alla tau_clf del fold.

Parti pure (costruzione del fold, aggregazione, tabelle, lettura record DAE) verificate; il
riaddestramento riusa i core dei componenti (_fit_mlp / _fit_pa).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lib.dae.constants import FPR_TARGET
from lib.dae.threshold import compute_threshold
from lib.secondo_stadio.data import predict_phi
from lib.secondo_stadio.metrics import best_mcc_bal_standalone


def loao_shared_sup(shared_state: dict, exclude_class: str) -> dict:
    """Shared-state del fold ceiling: la classe esclusa esce da att_train/att_es/att_select;
    train_idx/train_y ricostruiti; benigni invarianti. Ritorna una COPIA (side-effect-free)."""
    labs = shared_state["val_attack"]
    if exclude_class == "Benign" or not (labs == exclude_class).any():
        raise ValueError(f"classe da escludere assente o non valida: {exclude_class!r}")

    def keep(idx):
        idx = np.asarray(idx)
        return idx[labs[idx] != exclude_class]

    att_train = keep(shared_state["att_train"])
    att_select = keep(shared_state["att_select"])
    att_es = keep(shared_state["att_es"])
    ben_train_abs = shared_state["train_idx"][shared_state["train_y"] == 0]
    train_idx = np.concatenate([ben_train_abs, att_train])
    train_y = np.concatenate([np.zeros(len(ben_train_abs), dtype=np.float32),
                              np.ones(len(att_train), dtype=np.float32)])
    ben_sel = shared_state["ben_sel_abs"]
    prevalence_sel = len(att_es) / (len(ben_sel) + len(att_es))
    return {**shared_state, "att_train": att_train, "att_select": att_select,
            "att_es": att_es, "train_idx": train_idx, "train_y": train_y,
            "prevalence_sel": float(prevalence_sel), "exclude_class": exclude_class}


def train_loao_sup(config: dict, seed: int, shared_state: dict, parent_args: dict,
                   exclude_class: str) -> dict:
    """Un fold (classe, seme) del LOAO supervisionato: riaddestra da zero senza la classe,
    poi DR zero-day della classe su TUTTE le sue righe del val, a tau=p99 benigni-select."""
    from lib.secondo_stadio.sup_search import _fit_mlp

    sh = loao_shared_sup(shared_state, exclude_class)
    model, cb = _fit_mlp(config, seed, sh, parent_args)

    Zs, R = sh["Zs"], sh["R"]
    clf_ben = predict_phi(model, Zs, R, sh["ben_sel_abs"])
    tau = compute_threshold(clf_ben, FPR_TARGET)          # p99 benigni-select (classe-invarianti)
    excl_pos = np.where(sh["val_attack"] == exclude_class)[0]
    dr = float((predict_phi(model, Zs, R, excl_pos) >= tau).mean())
    return {
        "exclude_class": exclude_class, "seed": int(seed), "dr": dr,
        "tau": float(tau), "fpr_sel": float((clf_ben >= tau).mean()),
        "n_class_rows": int(len(excl_pos)),
        "best_ap": float(cb.best_ap), "best_epoch": int(cb.best_epoch),
        "n_epochs": len(cb.history_ap), "filter_reason": cb.filter_reason,
    }



def loao_shared_pa(shared_state: dict, exclude_class: str) -> dict:
    """Shared-state del fold PA: il training (benigni+pseudo) e' invariante; la classe esce
    da att_es (metrica per-epoca) e att_pos (sweep soglia standalone)."""
    labs = shared_state["val_attack"]
    if exclude_class == "Benign" or not (labs == exclude_class).any():
        raise ValueError(f"classe da escludere assente o non valida: {exclude_class!r}")

    def keep(idx):
        idx = np.asarray(idx)
        return idx[labs[idx] != exclude_class]

    att_es = keep(shared_state["att_es"])
    att_pos = keep(shared_state["att_pos"])
    ben_sel = shared_state["ben_sel_abs"]
    prevalence_sel = len(att_es) / (len(ben_sel) + len(att_es))
    return {**shared_state, "att_es": att_es, "att_pos": att_pos,
            "prevalence_sel": float(prevalence_sel), "exclude_class": exclude_class}


def train_loao_pa(config: dict, seed: int, shared_state: dict, parent_args: dict,
                  exclude_class: str) -> dict:
    """Un fold (classe, seme) del LOAO PA: riaddestra (benigni+pseudo, invariante), sceglie
    la soglia standalone SENZA la classe, misura la DR zero-day della classe su val."""
    from lib.secondo_stadio.pa_search import _fit_pa

    sh = loao_shared_pa(shared_state, exclude_class)
    model, cb = _fit_pa(config, seed, sh, parent_args)

    Zs, R = sh["Zs"], sh["R"]
    clf_ben = predict_phi(model, Zs, R, sh["ben_sel_abs"])
    clf_att = predict_phi(model, Zs, R, sh["att_pos"])     # attacchi SENZA la classe
    best = best_mcc_bal_standalone(clf_ben, clf_att)
    excl_pos = np.where(sh["val_attack"] == exclude_class)[0]
    dr = float((predict_phi(model, Zs, R, excl_pos) >= best["tau"]).mean())
    return {
        "exclude_class": exclude_class, "seed": int(seed), "dr": dr,
        "tau_clf": float(best["tau"]), "p_star": best["p_star"],
        "mcc_bal_fold": float(best["mcc_bal"]), "fpr_sel": float(best["fpr"]),
        "n_class_rows": int(len(excl_pos)),
        "best_ap": float(cb.best_ap), "best_epoch": int(cb.best_epoch),
        "n_epochs": len(cb.history_ap), "filter_reason": cb.filter_reason,
    }



def aggregate_loao(rows: list[dict]) -> dict:
    """Per classe: media/std (ddof=0)/mediana della DR sui semi. rows = output dei fold."""
    by_class: dict[str, list[float]] = {}
    for r in rows:
        by_class.setdefault(r["exclude_class"], []).append(float(r["dr"]))
    out = {}
    for cls, drs in sorted(by_class.items()):
        arr = np.asarray(drs, dtype=float)
        out[cls] = {"dr_mean": float(arr.mean()), "dr_std": float(arr.std(ddof=0)),
                    "dr_median": float(np.median(arr)), "n_seeds": int(len(arr))}
    return out


def build_dr_table(dae_by_class: dict, sup_by_class: dict,
                   pa_by_class: dict | None = None) -> list[dict]:
    """Tabella di confronto per classe (P2: DAE vs ceiling; P5: + PA), pura e ordinata per
    nome classe. Delta = metodo − DAE. Classi presenti in almeno una sorgente; NaN sui buchi."""
    classes = sorted(set(dae_by_class) | set(sup_by_class) | set(pa_by_class or {}))
    rows = []
    for cls in classes:
        d = dae_by_class.get(cls, {})
        s = sup_by_class.get(cls, {})
        row = {
            "classe": cls,
            "dr_dae": d.get("dr_mean", float("nan")),
            "dr_dae_std": d.get("dr_std", float("nan")),
            "dr_ceiling": s.get("dr_mean", float("nan")),
            "dr_ceiling_std": s.get("dr_std", float("nan")),
        }
        row["delta_ceiling_dae"] = row["dr_ceiling"] - row["dr_dae"]
        if pa_by_class is not None:
            p = pa_by_class.get(cls, {})
            row["dr_pa"] = p.get("dr_mean", float("nan"))
            row["dr_pa_std"] = p.get("dr_std", float("nan"))
            row["delta_pa_dae"] = row["dr_pa"] - row["dr_dae"]
            row["delta_pa_ceiling"] = row["dr_pa"] - row["dr_ceiling"]
        rows.append(row)
    return rows


def dae_dr_from_records(jsonl_path, trial_num: int = 569) -> dict:
    """DR per classe del DAE congelato dal LOAO cap2 (runs/dae/loao_fase3/loao_records.jsonl):
    aggrega dr_by_class sui semi del trial richiesto (mean/std ddof=0)."""
    by_class: dict[str, list[float]] = {}
    n_seeds = 0
    for line in Path(jsonl_path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if int(rec.get("trial_num", -1)) != int(trial_num):
            continue
        n_seeds += 1
        for cls, dr in rec["dr_by_class"].items():
            by_class.setdefault(cls, []).append(float(dr))
    if not by_class:
        raise ValueError(f"nessun record per trial_num={trial_num} in {jsonl_path}")
    return {cls: {"dr_mean": float(np.mean(v)), "dr_std": float(np.std(v)),
                  "n_seeds": len(v)}
            for cls, v in sorted(by_class.items())}
