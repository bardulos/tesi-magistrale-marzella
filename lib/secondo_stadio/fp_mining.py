"""lib/secondo_stadio/fp_mining.py — hard-negative mining (bootstrap offline per round) del PA.

Hardening a tempo di addestramento (P6, cap3; port da repo_v5 fp_mining.py, adattato a
phi-107 e ai moduli repov6): a ogni round si minano i falsi positivi del modello precedente
(benigni-train con punteggio >= tau_clf), si accumulano in un pool cumulativo, si riaddestra
DA ZERO su ben_train + pool*K (K copie extra, molteplicita' x(K+1)) e si tiene il best-round
(argmax pAUC_MCC sul selection set). Il TEST e' interrogato UNA volta sola, sul best-round
(confirm_on_test), e solo se richiesto esplicitamente.

Tesi della sezione (da misurare): il guadagno standalone raramente sopravvive alla fusione
AND — per questo ogni round riporta ENTRAMBE le colonne (mcc_bal standalone e and_fpr/and_dr
al punto operativo), prima e dopo l'hardening. Default canonici K=4, k_min=0.002, n_max=5
(dal config legacy, non dai default del codice v5: divergenza dichiarata nel findings §2.6).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from lib.dae.constants import FPR_TARGET
from lib.secondo_stadio import constants as C
from lib.secondo_stadio.data import load_tau_dae, predict_phi
from lib.secondo_stadio.metrics import (and_at_oppoint, best_mcc_bal_standalone,
                                        mcc_at_fpr, mcc_bal, mcc_fpr_curve,
                                        partial_auc_mcc)

log = logging.getLogger(__name__)

# Griglia FPR per la curva grezza salvata (plotting a valle).
_PLOT_GRID = np.linspace(C.FPM_PLOT_FPR_LO, C.FPM_PLOT_FPR_HI, C.FPM_PLOT_N)


def _curve_metrics(clf_ben, clf_att) -> dict:
    """pAUC_MCC (band, scala MCC) + curva MCC(FPR) estesa (dati grezzi) dai punteggi del clf."""
    pauc, _, _ = partial_auc_mcc(clf_ben, clf_att)
    return {"pauc_mcc": float(pauc),
            "curve_fpr": _PLOT_GRID.tolist(),
            "curve_mcc": mcc_fpr_curve(clf_ben, clf_att, _PLOT_GRID).tolist()}


# Helper puri (testabili senza TF) — port verbatim v5 :29-51

def grow_pool(pool: np.ndarray, new_fp: np.ndarray) -> np.ndarray:
    """Unione cumulativa del pool con i nuovi FP (mai droppare, indici unici)."""
    if len(pool) == 0:
        return np.unique(np.asarray(new_fp)).astype(np.int64)
    return np.union1d(np.asarray(pool), np.asarray(new_fp)).astype(np.int64)


def augment_train(idx_train: np.ndarray, pool: np.ndarray, K: int) -> np.ndarray:
    """idx_train + K copie EXTRA di ciascun FP (molteplicita' effettiva x(K+1))."""
    idx_train = np.asarray(idx_train)
    if len(pool) == 0 or K <= 0:
        return idx_train
    return np.concatenate([idx_train, np.repeat(np.asarray(pool), int(K))])


def should_stop(curr_mcc: float, prev_mcc: float, k_min: float) -> bool:
    """Auto-stop a soglia singola: copre degrado (delta<0) e gain trascurabile (0<=delta<k_min)."""
    return (curr_mcc - prev_mcc) < k_min


def assert_pool_subset(pool: np.ndarray, idx_train: np.ndarray) -> None:
    if len(pool) and not np.isin(pool, idx_train).all():
        raise AssertionError("leak-free violato: pool non subset dei benigni-train")


# Forward / mining

def _load_model(weights_path, config: dict, mlp_fixed: dict):
    """Ricostruisce il modello IDENTICO a pa_search._fit_pa (config-first: `h1_from_config` gestisce
    `exp` o `h1` legacy; ratio/dropout dal config; l2 = L2_FIXED_S2). Deve combaciare con
    l'architettura con cui i pesi sono stati addestrati. `mlp_fixed` inerte (KEEP-and-flag: firma
    lasciata per i call-site)."""
    from lib.secondo_stadio.model import build_mlp, h1_from_config
    params = {"h1": h1_from_config(config), "n_layers": int(config["n_layers"]),
              "ratio": float(config["ratio"]), "dropout": float(config["dropout"]),
              "l2_reg": C.L2_FIXED_S2}
    model = build_mlp(params)
    model.load_weights(str(weights_path))
    return model


def mine_fps(weights_path, config, mlp_fixed, shared, tau: float) -> np.ndarray:
    """Falsi positivi (benigni-train con score >= tau) col modello del round precedente.
    Indici ASSOLUTI (ben_train_abs)."""
    model = _load_model(weights_path, config, mlp_fixed)
    ben_train = shared["ben_train_abs"]
    s = predict_phi(model, shared["Zs"], shared["R"], ben_train)
    return ben_train[s >= tau]


def eval_weights(weights_path, config, mlp_fixed, shared, dae_neg, dae_pos,
                 tau_dae: float) -> dict:
    """Round 0 (baseline): standalone MCC_bal-ottimo + AND al punto operativo (no training)."""
    model = _load_model(weights_path, config, mlp_fixed)
    clf_ben = predict_phi(model, shared["Zs"], shared["R"], shared["ben_sel_abs"])
    clf_att = predict_phi(model, shared["Zs"], shared["R"], shared["att_pos"])
    mo = best_mcc_bal_standalone(clf_ben, clf_att)
    op = mcc_at_fpr(clf_ben, clf_att, FPR_TARGET)   # punto operativo a FPR=1% (valutazione finale)
    and_fpr, and_dr = and_at_oppoint(clf_ben, clf_att, dae_neg, dae_pos, mo["tau"], tau_dae)
    return {"round": 0, "mcc_bal": mo["mcc_bal"], "mcc_fpr1": float(op["mcc"]),
            **_curve_metrics(clf_ben, clf_att),
            "fpr": mo["fpr"], "recall": mo["recall"],
            "tau_clf": mo["tau"], "and_fpr": and_fpr, "and_dr": and_dr,
            "pool_size": 0, "n_new_fps": 0, "best_epoch": -1}


def confirm_on_test(weights_path, config, mlp_fixed, tau_clf: float, tctx: dict,
                    tau_dae: float) -> dict:
    """Conferma su TEST (UNA volta, sul best-round): soglia presa su val, applicata al test."""
    model = _load_model(weights_path, config, mlp_fixed)
    clf_neg = predict_phi(model, tctx["Zs"], tctx["R"], tctx["ben_pos"])
    clf_pos = predict_phi(model, tctx["Zs"], tctx["R"], tctx["att_pos"])
    fpr = float((clf_neg >= tau_clf).mean())
    recall = float((clf_pos >= tau_clf).mean())
    dae_neg = tctx["dae_score"][tctx["ben_pos"]]
    dae_pos = tctx["dae_score"][tctx["att_pos"]]
    and_fpr, and_dr = and_at_oppoint(clf_neg, clf_pos, dae_neg, dae_pos, tau_clf, tau_dae)
    return {"test_mcc_bal": mcc_bal(fpr, recall), "test_fpr": fpr, "test_recall": recall,
            "test_and_fpr": and_fpr, "test_and_dr": and_dr}


# Un round + traiettoria

def fp_mining_round(round_idx: int, prev_weights_path, prev_tau: float, shared: dict,
                    config: dict, parent_args: dict, fp_pool: np.ndarray, K: int,
                    seed: int, dae_neg, dae_pos, tau_dae: float, weights_dir: Path):
    """Un round: mina col modello precedente, cresce il pool, riaddestra DA ZERO sul train
    aumentato (pa_search.train_one_seed con ben_train_abs aumentato), misura standalone+AND."""
    from lib.secondo_stadio.pa_search import train_one_seed

    mlp_fixed = parent_args["mlp_fixed"]
    new_fp = mine_fps(prev_weights_path, config, mlp_fixed, shared, prev_tau)
    fp_pool = grow_pool(fp_pool, new_fp)
    assert_pool_subset(fp_pool, shared["ben_train_abs"])
    shared_round = {**shared,
                    "ben_train_abs": augment_train(shared["ben_train_abs"], fp_pool, K)}
    pa_round = {**parent_args, "save_weights": True}
    res = train_one_seed(config, seed, shared_round, pa_round, weights_dir,
                         return_scores=True)
    and_fpr, and_dr = and_at_oppoint(res["clf_ben_sel"], res["clf_att"], dae_neg, dae_pos,
                                     res["tau_clf"], tau_dae)
    metrics = {"round": round_idx, "mcc_bal": res["mcc_bal"], "mcc_fpr1": res["mcc_fpr1"],
               **_curve_metrics(res["clf_ben_sel"], res["clf_att"]),
               "fpr": res["fpr"], "recall": res["recall"], "tau_clf": res["tau_clf"],
               "and_fpr": and_fpr, "and_dr": and_dr,
               "pool_size": int(len(fp_pool)), "n_new_fps": int(len(new_fp)),
               "best_epoch": res["best_epoch"]}
    return res["weights_path"], fp_pool, metrics


def run_trajectory(seed: int, config: dict, shared: dict, dae_neg, dae_pos,
                   tau_dae: float, parent_args: dict, K: int, stop_threshold: float, n_max: int,
                   traj_dir: Path, tctx: dict | None, base_weights) -> dict:
    """Round 0 baseline -> loop con auto-stop su **pAUC_MCC** (Delta < stop_threshold = c*sigma_m
    per-modello) -> best-round = **argmax pAUC_MCC** (metrica di selezione; NON MCC_bal, che era il
    protocollo legacy) -> conferma su test UNA volta (solo se tctx)."""
    mlp_fixed = parent_args["mlp_fixed"]
    m0 = eval_weights(base_weights, config, mlp_fixed, shared, dae_neg, dae_pos, tau_dae)
    rounds = [m0]
    pool = np.array([], dtype=np.int64)
    prev_w, prev_tau = base_weights, m0["tau_clf"]
    weights_by_round = {0: str(base_weights)}
    for r in range(1, n_max + 1):
        wdir = Path(traj_dir) / f"round_{r}"
        new_w, pool, m_r = fp_mining_round(r, prev_w, prev_tau, shared, config,
                                           parent_args, pool, K, seed, dae_neg, dae_pos,
                                           tau_dae, wdir)
        rounds.append(m_r)
        weights_by_round[r] = str(new_w)
        if should_stop(m_r["pauc_mcc"], rounds[r - 1]["pauc_mcc"], stop_threshold):
            break
        prev_w, prev_tau = new_w, m_r["tau_clf"]
    best_i = int(np.argmax([m["pauc_mcc"] for m in rounds]))
    test_m = (confirm_on_test(weights_by_round[best_i], config, mlp_fixed,
                              rounds[best_i]["tau_clf"], tctx, tau_dae) if tctx else {})
    return {"seed": int(seed), "rounds": rounds, "best_round": best_i,
            "weights_by_round": weights_by_round, "test": test_m}


# Stage

def run_fp_mining_stage(cfg: dict) -> None:
    """Stage `fp_mining`: hardening del/dei finalisti selezionati (P5).

    cfg: cache_dir, labels_val, l1l2_csv (tau_dae, sorgente unica), base_weights (pesi del
    round 0 = finalista), model_config (h1/n_layers/ratio/dropout/delta/alpha_lo/alpha_hi),
    mlp_fixed (fissi P1), out_dir; K (default 4), stop_threshold (default C.FPM_K_MIN),
    n_max (default 5),
    seed; confirm_test: false di default — se true richiede labels_test e interroga il test
    UNA volta sul best-round.
    """
    from lib.secondo_stadio import constants as C
    from lib.secondo_stadio.data import load_test_context
    from lib.secondo_stadio.pa_search import build_parent_args, setup

    out_dir = Path(cfg["out_dir"])
    if not out_dir.exists():
        raise FileNotFoundError(f"out_dir non esiste (shell-first): {out_dir}")
    tau_dae = load_tau_dae(cfg["l1l2_csv"])
    config = dict(cfg["model_config"])

    parent_args = build_parent_args(cfg)
    shared = setup(parent_args)
    dae_neg = shared["dae_score"][shared["ben_sel_abs"]]
    dae_pos = shared["dae_score"][shared["att_pos"]]

    tctx = None
    if cfg.get("confirm_test"):
        tctx = load_test_context(cfg["cache_dir"], cfg["labels_test"])

    t0 = time.perf_counter()
    out = run_trajectory(int(cfg.get("seed", C.SEEDS_PA[0])), config, shared,
                         dae_neg, dae_pos, tau_dae, parent_args,
                         int(cfg.get("K", C.FPM_K)),
                         float(cfg.get("stop_threshold", C.FPM_K_MIN)),
                         int(cfg.get("n_max", C.FPM_N_MAX)),
                         out_dir, tctx=tctx, base_weights=str(cfg["base_weights"]))
    (out_dir / "trajectory.json").write_text(json.dumps(out, indent=2))
    log.info("fp_mining seed%d: best_round=%d (mcc_bal %f -> %f) in %.0fs",
             out["seed"], out["best_round"], out["rounds"][0]["mcc_bal"],
             out["rounds"][out["best_round"]]["mcc_bal"], time.perf_counter() - t0)
