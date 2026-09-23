"""lib/secondo_stadio/sup_search.py — componente P1: ceiling supervisionato (mlp_supervisionato).

MLP binario su phi=[z_s, r_w] con etichette REALI (attacchi del val): il tetto informativo del
secondo stadio, cercato col motore condiviso (lib.search.engine, SearchComponent). Protocollo
congelato al GATE A:

- SPLIT FISSO seed 0 (D2, non per-seme come il legacy v3): benigni 80/20 + attacchi 80/20;
  training su 80%benigni ∪ 80%attacchi al rapporto naturale ~63:37, BCE SENZA class weight
  (D11: l'asimmetria di bilanciamento col PA — che addestra su 1:1 sintetico — e' assorbita
  dallo sweep standalone della soglia e dalla valutazione calibrata a pi). Il TEST non e'
  MAI toccato in ricerca (diverge dal legacy v3, che lo interrogava per ogni trial).
- ES per-epoca su AUC-PR del selection set (D1, riuso PrAucCallback: restore-best, patience,
  filtro non-apprendimento) — MAI MCC per-epoca (patologia legacy documentata nel cap2).
  COERENZA ES<->OBIETTIVO: l'AUC-PR per-epoca su set FISSO e' legittima perche' il ranking e'
  intra-traiettoria a prevalenza costante; la dipendenza dalla prevalenza [brabec2020imbalance]
  riguarda confronti fra prevalenze diverse, che qui non avvengono.
- Selezione del trial su AUC-PR (objective = mean − K·std, ddof=0, semi genuini). Colonne
  INFORMATIVE per-seme (mai selettive): MCC@FPR1% (tau=p99 benigni-select, comparabile col
  primo stadio) e MCC_bal alla prevalenza di RIFERIMENTO pi (D5 — mai "di deployment").
- Il gap ceiling<->PA misurato in search e' APPROSSIMATO (selection set asimmetrici, D8):
  il numero definitivo si misura sul test, una volta, nella valutazione finale dichiarata.
- Trial potati = PRUNED nel DB (PruningAwareOptunaSearch); il TPE 4.8 li include nel fit col
  valore-al-rung: scelta accettata e dichiarata (v. lib/search/engine.py).

setup/train_one_seed sono lo STOCASTICO (TF): validati dal dry-run, non da unit test; le
parti pure (spazio, aggregate, parent_args) sono verificate.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

from lib.dae.constants import FPR_TARGET, K_STD, NON_LEARNING_MARGIN, NON_LEARNING_MIN_EPOCHS
from lib.dae.objective import aggregate_seeds
from lib.search.engine import SearchComponent
from lib.secondo_stadio import constants as C
from lib.secondo_stadio.data import assemble_phi, load_phi_cache, predict_phi
from lib.secondo_stadio.metrics import mcc_at_fpr, mcc_bal

log = logging.getLogger(__name__)


def define_space(trial) -> dict:
    """Spazio P1 a 5 dimensioni, forma a rapporto fisso (niente larghezze libere per-layer,
    niente condizionali). batch e L2 sono FISSI (D3): il regolarizzatore cercato e' il dropout."""
    return {
        "n_layers": trial.suggest_categorical("n_layers", list(C.SUP_N_LAYERS)),
        "h1": trial.suggest_int("h1", C.SUP_H1_LOW, C.SUP_H1_HIGH, log=True),
        "ratio": trial.suggest_float("ratio", C.SUP_RATIO_LOW, C.SUP_RATIO_HIGH),
        "lr": trial.suggest_float("lr", C.SUP_LR_LOW, C.SUP_LR_HIGH, log=True),
        "dropout": trial.suggest_float("dropout", C.SUP_DROPOUT_LOW, C.SUP_DROPOUT_HIGH),
    }


def build_parent_args(cfg: dict) -> dict:
    """Path e iperparametri fissi spediti ai worker (serializzati)."""
    return {
        "cache_dir": cfg["cache_dir"],
        "labels_val": cfg["labels_val"],
        "max_epochs": int(cfg.get("max_epochs", C.MLP_MAX_EPOCHS)),
        "patience": int(cfg.get("patience", C.MLP_PATIENCE)),
        "batch": int(cfg.get("batch", C.BATCH_FIXED)),
        "l2_reg": float(cfg.get("l2_reg", C.L2_FIXED_S2)),
        "es_n_attacks": int(cfg.get("es_n_attacks", C.ES_N_ATTACKS)),
        "split_seed": int(cfg.get("split_seed", C.SPLIT_SEED)),
        "val_frac": float(cfg.get("val_frac", C.VAL_FRAC)),
        "non_learning_margin": float(cfg.get("non_learning_margin", NON_LEARNING_MARGIN)),
        "non_learning_min_epochs": int(cfg.get("non_learning_min_epochs",
                                               NON_LEARNING_MIN_EPOCHS)),
        "save_weights": bool(cfg.get("save_weights", False)),
        # Radice DUREVOLE dei pesi (HOME del worker, mai /tmp): v. engine._run_trainable.
        "weights_home": cfg.get("weights_home"),
    }


def setup(parent_args: dict) -> dict:
    """Caricato UNA VOLTA per trainable: cache phi + split leak-free del ceiling
    (attack_split=True, D2). Prevalenza del selection set per-epoca (benigni-select ∪
    att_es) misurata per il filtro non-apprendimento."""
    data = load_phi_cache(parent_args["cache_dir"], parent_args["labels_val"],
                          val_frac=parent_args["val_frac"],
                          split_seed=parent_args["split_seed"],
                          es_n_attacks=parent_args["es_n_attacks"],
                          attack_split=True)
    ben_train_abs = data["ben_pos"][data["idx_train"]]
    ben_sel_abs = data["ben_pos"][data["idx_select"]]
    train_idx = np.concatenate([ben_train_abs, data["att_train"]])
    train_y = np.concatenate([np.zeros(len(ben_train_abs), dtype=np.float32),
                              np.ones(len(data["att_train"]), dtype=np.float32)])
    prevalence_sel = len(data["att_es"]) / (len(ben_sel_abs) + len(data["att_es"]))
    log.info("setup mlp_sup: train=%d (ben=%d att=%d, rapporto naturale), "
             "select: ben=%d att_full=%d att_es=%d (prevalenza per-epoca=%.4f)",
             len(train_idx), len(ben_train_abs), len(data["att_train"]),
             len(ben_sel_abs), len(data["att_select"]), len(data["att_es"]), prevalence_sel)
    return {**data, "ben_sel_abs": ben_sel_abs, "train_idx": train_idx,
            "train_y": train_y, "prevalence_sel": float(prevalence_sel)}


def _make_train_ds(Zs, R, train_idx, train_y, batch: int, seed: int):
    """tf.data dal generatore su blocchi di indici: esempi permutati deterministicamente
    dal seme UNA volta (stesso ordine a ogni epoca, pattern del DAE cap2), phi assemblato
    per-batch (r_w resta mmap: page-cache condivisa, niente materializzazione)."""
    import tensorflow as tf

    n = (len(train_idx) // batch) * batch
    order = np.random.default_rng(seed).permutation(len(train_idx))[:n]
    idx_perm = train_idx[order]
    y_perm = train_y[order]
    steps = n // batch

    def gen():
        for b in range(steps):
            sl = slice(b * batch, (b + 1) * batch)
            yield assemble_phi(Zs, R, idx_perm[sl]), y_perm[sl]

    sig = (tf.TensorSpec((batch, C.PHI_DIM), tf.float32),
           tf.TensorSpec((batch,), tf.float32))
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig).repeat().prefetch(
        tf.data.AUTOTUNE)
    return ds, steps


def _fit_mlp(config: dict, seed: int, shared_state: dict, parent_args: dict):
    """Core di addestramento del ceiling (condiviso con lib.secondo_stadio.loao): seeding
    deterministico, build+BCE, ES su AUC-PR del selection set (ben_sel_abs ∪ att_es del
    shared_state passato — il LOAO passa uno shared_state gia' filtrato dalla classe),
    restore-best. Ritorna (model, callback)."""
    import tensorflow as tf

    from lib.dae.callback import PrAucCallback
    from lib.dae.objective import seed_ap
    from lib.secondo_stadio.model import build_mlp

    tf.keras.utils.set_random_seed(int(seed))
    tf.config.experimental.enable_op_determinism()
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except RuntimeError:
        pass  # contesto TF gia' inizializzato (test in-process); sui worker Ray il vincolo
        #       a 1 thread e' comunque garantito dalle env TF_NUM_*_THREADS (engine runtime_env)

    Zs, R = shared_state["Zs"], shared_state["R"]
    ben_sel = shared_state["ben_sel_abs"]
    att_es = shared_state["att_es"]

    params = {"h1": int(config["h1"]), "n_layers": int(config["n_layers"]),
              "ratio": float(config["ratio"]), "dropout": float(config["dropout"]),
              "l2_reg": float(parent_args["l2_reg"])}
    model = build_mlp(params)
    # D11: BCE semplice, nessun class weight (rapporto naturale nel training).
    model.compile(optimizer=tf.keras.optimizers.Adam(float(config["lr"])),
                  loss="binary_crossentropy")

    y_es = np.concatenate([np.zeros(len(ben_sel)), np.ones(len(att_es))])

    def ap_fn():
        proba = np.concatenate([predict_phi(model, Zs, R, ben_sel),
                                predict_phi(model, Zs, R, att_es)])
        return seed_ap(y_es, proba)

    cb = PrAucCallback(ap_fn=ap_fn, patience=int(parent_args["patience"]),
                       prevalence=float(shared_state["prevalence_sel"]),
                       min_epochs=int(parent_args["non_learning_min_epochs"]),
                       margin=float(parent_args["non_learning_margin"]))
    train_ds, steps = _make_train_ds(Zs, R, shared_state["train_idx"],
                                     shared_state["train_y"],
                                     int(parent_args["batch"]), int(seed))
    model.fit(train_ds, epochs=int(parent_args["max_epochs"]), steps_per_epoch=steps,
              callbacks=[cb], verbose=0)
    return model, cb


def train_one_seed(config: dict, seed: int, shared_state: dict,
                   parent_args: dict, seed_dir: Path) -> dict:
    """Addestra il ceiling per un seme; ritorna best_ap (AUC-PR al restore-best) + le colonne
    informative MCC@FPR1% e MCC_bal.

    Le colonne informative sono calcolate a fine seme coi pesi gia' ripristinati al best dal
    restore-best di PrAucCallback, sul punto operativo benigni-select ∪ att_select COMPLETI
    (att_es e' solo il subsample per-epoca dell'early-stopping, D7)."""
    Zs, R = shared_state["Zs"], shared_state["R"]
    ben_sel = shared_state["ben_sel_abs"]
    att_select = shared_state["att_select"]

    model, cb = _fit_mlp(config, seed, shared_state, parent_args)

    clf_ben = predict_phi(model, Zs, R, ben_sel)
    clf_att = predict_phi(model, Zs, R, att_select)
    op = mcc_at_fpr(clf_ben, clf_att, FPR_TARGET)
    mcc_bal_pi_ref = mcc_bal(op["fpr"], op["recall"])

    weights_path = None
    if parent_args.get("save_weights"):
        seed_dir.mkdir(parents=True, exist_ok=True)
        weights_path = str(seed_dir / "best.weights.h5")
        model.save_weights(weights_path)
        (seed_dir / "history_ap.json").write_text(json.dumps(
            [float(x) if x is not None else None for x in cb.history_ap]))

    return {
        "seed": int(seed),
        "best_ap": float(cb.best_ap) if cb.best_ap > -1e18 else float("nan"),
        "best_epoch": int(cb.best_epoch),
        "filter_reason": cb.filter_reason,
        "n_epochs": len(cb.history_ap),
        "mcc_fpr1": float(op["mcc"]),
        "mcc_bal": float(mcc_bal_pi_ref),
        "weights_path": weights_path,
    }


def aggregate(per_seed_results: list, config: dict) -> dict:
    """objective = mean(AUC-PR) − K_STD·std(AUC-PR) sui semi GENUINI (ddof=0); sentinella -1 se
    nessun genuino. Le colonne MCC sono INFORMATIVE (medie sui genuini + scalari dell'ultimo
    seme, `*_last`, per la lettura per-seme da result.json senza ricostruzione)."""
    genuine = [r for r in per_seed_results if r.get("filter_reason") != "non_learning"]
    aps = [r["best_ap"] for r in genuine
           if r["best_ap"] is not None and math.isfinite(float(r["best_ap"]))]
    obj = aggregate_seeds([r["best_ap"] for r in genuine], K_STD)
    if math.isnan(obj):
        obj = -1.0
    last = per_seed_results[-1]
    return {
        "objective": float(obj),
        "n_genuine": len(aps),
        "ap_mean": float(np.mean(aps)) if aps else float("nan"),
        "ap_std": float(np.std(aps)) if aps else float("nan"),
        "mcc_fpr1_mean": float(np.mean([r["mcc_fpr1"] for r in genuine])) if genuine else float("nan"),
        "mcc_bal_mean": float(np.mean([r["mcc_bal"] for r in genuine])) if genuine else float("nan"),
        "seed_last": int(last["seed"]),
        "ap_last": float(last["best_ap"]),
        "mcc_fpr1_last": float(last["mcc_fpr1"]),
        "mcc_bal_last": float(last["mcc_bal"]),
    }


REPORT_EXTRA_KEYS = ("n_genuine", "ap_mean", "ap_std", "mcc_fpr1_mean", "mcc_bal_mean",
                     "seed_last", "ap_last", "mcc_fpr1_last", "mcc_bal_last")


def make_sup_component(seeds=None) -> SearchComponent:
    seeds_list = list(seeds) if seeds else list(C.SEEDS)
    return SearchComponent(
        name="mlp_sup",
        seeds=seeds_list,
        metric_name="objective",
        define_space=define_space,
        build_parent_args=build_parent_args,
        setup=setup,
        train_one_seed=train_one_seed,
        aggregate=aggregate,
        report_extra_keys=REPORT_EXTRA_KEYS,
        max_t=None,   # = len(seeds): 6 semi base; rung unico a 4 (grace/rf dal config YAML)
    )


def run_sup_search_stage(cfg: dict) -> dict:
    """Entry dello stage 'sup_search': componente ceiling + motore condiviso."""
    from lib.search.engine import run_search
    component = make_sup_component(seeds=cfg.get("seeds"))
    return run_search(component, cfg)
