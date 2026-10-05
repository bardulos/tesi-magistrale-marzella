"""LOAO del secondo stadio con riaddestramento per classe esclusa.

Nel ceiling la classe esce da training, early stopping e selezione; nella PA
esce da early stopping e calibrazione. Il test resta escluso.
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
    """Esclude la classe dagli split del ceiling e ricostruisce le label di training."""
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
    tau = compute_threshold(clf_ben, FPR_TARGET)
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
    """Esclude la classe da early stopping e calibrazione PA; training invariato."""
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
    """Riaddestra la PA e misura la DR della classe esclusa su validation."""
    from lib.secondo_stadio.pa_search import _fit_pa

    sh = loao_shared_pa(shared_state, exclude_class)
    model, cb = _fit_pa(config, seed, sh, parent_args)

    Zs, R = sh["Zs"], sh["R"]
    clf_ben = predict_phi(model, Zs, R, sh["ben_sel_abs"])
    clf_att = predict_phi(model, Zs, R, sh["att_pos"])
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
    """Tabella per classe di DAE, ceiling e PA; delta = metodo - DAE."""
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
    """Aggrega per classe i record LOAO DAE del trial richiesto."""
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
