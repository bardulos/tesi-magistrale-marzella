"""lib/secondo_stadio/pa_search.py — componente P3: classificatore di pseudo-anomalie (PA).

MLP binario su phi=[z_s, r_w] addestrato SENZA etichette d'attacco: benigni reali (label 0)
contro pseudo-anomalie generate online (label 1, rapporto 1:1) — swap [z_i, r_j] e radial
[z_i, alpha*r_i] mescolate dalla frazione radiale delta (lib.secondo_stadio.generators).
Protocollo congelato al GATE A:

- OBIETTIVO del trial = mean(MCC_bal) − std (ddof=0) sui semi genuini, con soglia scelta
  STANDALONE sul selection set (sweep percentile, best_mcc_bal_standalone). MCC_bal e'
  rate-scaled alla prevalenza di RIFERIMENTO pi (nativa del val, D5 — mai "di deployment");
  la motivazione in letteratura (brabec2020imbalance, siblini2020calibration, chicco2020mcc)
  e la riconciliazione col cap2 sono nel findings, §4.
- ES per-epoca su AUC-PR del selection set (D1, riuso PrAucCallback) — MAI MCC per-epoca:
  il legacy fermava su MCC_bal smussato (k=3), la patologia documentata nel cap2.
  COERENZA ES<->OBIETTIVO: l'AUC-PR per-epoca su set FISSO e' legittima perche' il ranking
  e' intra-traiettoria a prevalenza costante; la dipendenza dalla prevalenza
  [brabec2020imbalance] riguarda confronti fra prevalenze diverse, che qui non avvengono.
- AUC-ROC = colonna INFORMATIVA (sentinella di coerenza: divergenze marcate fra
  l'ordinamento MCC_bal e quello ROC nel vertice vanno segnalate), mai selettiva; idem
  MCC@FPR1% (comparabilita' col primo stadio).
- 8 semi (SEEDS_PA), rung ASHA a 4 e 6 semi (grace=4, rf=1.5, max_t=8: engine rf-float).
- REDESIGN 2026-07-03 (direttiva maintainer): spazio a 8 dim, la PA CERCA l'ARCHITETTURA
  (exp/n_layers/ratio/dropout, intervalli di P1) + lr RISTRETTO [1e-4,1e-3] + generazione
  (delta/alpha_lo/alpha_hi). Motivo: compito diverso da P1 (pseudo-anomalie, geometria diversa) ->
  la fANOVA di P1 non trasferisce, l'architettura resta cercabile. `mlp_fixed` NON piu' obbligatorio
  (il config del trial porta ratio/lr/dropout; l2=L2_FIXED_S2); resta come fallback retro-compat
  per loao/fp_mining. WARM-START dalle 10 architetture piu' diverse fra le rosse del ceiling P1.
- Trial potati = PRUNED nel DB; il TPE 4.8 li include nel fit col valore-al-rung (dichiarato).

_fit_pa/train_one_seed sono lo STOCASTICO (TF), validati dal dry-run; setup e' caricamento
deterministico e, come le parti pure, e' verificato.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

from lib.dae.constants import (FPR_TARGET, K_STD, NON_LEARNING_MARGIN,
                               NON_LEARNING_MIN_EPOCHS)
from lib.dae.objective import aggregate_seeds
from lib.search.engine import SearchComponent
from lib.secondo_stadio import constants as C
from lib.secondo_stadio.data import assemble_phi, load_phi_cache, predict_phi
from lib.secondo_stadio.generators import generate_fakes
from lib.secondo_stadio.metrics import best_mcc_bal_standalone, mcc_at_fpr

log = logging.getLogger(__name__)


def define_space(trial) -> dict:
    """Spazio P3 a 8 dimensioni (REDESIGN 2026-07-03): la PA CERCA l'architettura + la generazione.
    - architettura (intervalli di P1, INTATTI): exp = fattore di espansione [1.5,4] (h1 derivato =
      round(exp*N_in)), n_layers, ratio, dropout; l2 resta fisso (L2_FIXED_S2).
    - lr: UNICO restringimento, in [1e-4,1e-3] (ceiling: lr dominante, ottimo al floor 2e-4).
    - generazione pseudo: delta (frazione radiale), alpha_lo>=1 (no pseudo-benigni), alpha_hi<=5.
    Niente mlp_fixed: il config del trial porta ratio/lr/dropout. exp campionato UNIFORME (prior
    piatto sul fattore di espansione, l'unita' interpretabile)."""
    return {
        "exp": trial.suggest_float("exp", C.SUP_EXPANSION_LOW, C.SUP_EXPANSION_HIGH),
        "n_layers": trial.suggest_categorical("n_layers", list(C.SUP_N_LAYERS)),
        "ratio": trial.suggest_float("ratio", C.SUP_RATIO_LOW, C.SUP_RATIO_HIGH),
        "dropout": trial.suggest_float("dropout", C.SUP_DROPOUT_LOW, C.SUP_DROPOUT_HIGH),
        "lr": trial.suggest_float("lr", C.PA_LR_LOW, C.PA_LR_HIGH, log=True),
        "delta": trial.suggest_float("delta", C.DELTA_LOW, C.DELTA_HIGH),
        "alpha_lo": trial.suggest_float("alpha_lo", C.ALPHA_LO_LOW, C.ALPHA_LO_HIGH),
        "alpha_hi": trial.suggest_float("alpha_hi", C.ALPHA_HI_LOW, C.ALPHA_HI_HIGH),
    }


def build_parent_args(cfg: dict) -> dict:
    """Path e iperparametri per i worker. REDESIGN 2026-07-03: la PA CERCA ratio/lr/dropout (nel
    config del trial), quindi `mlp_fixed` NON e' piu' obbligatorio. RETRO-COMPAT: se `mlp_fixed`
    e' presente (path che riusano _fit_pa/train_one_seed con config parziali — loao/fp_mining/
    pa_compress), fa da FALLBACK per ratio/lr/dropout/l2. L2 di default = L2_FIXED_S2."""
    pa = {
        "cache_dir": cfg["cache_dir"],
        "labels_val": cfg["labels_val"],
        "max_epochs": int(cfg.get("max_epochs", C.MLP_MAX_EPOCHS)),
        "patience": int(cfg.get("patience", C.MLP_PATIENCE)),
        "batch": int(cfg.get("batch", C.BATCH_FIXED)),
        "es_n_attacks": int(cfg.get("es_n_attacks", C.ES_N_ATTACKS)),
        "split_seed": int(cfg.get("split_seed", C.SPLIT_SEED)),
        "val_frac": float(cfg.get("val_frac", C.VAL_FRAC)),
        "non_learning_margin": float(cfg.get("non_learning_margin", NON_LEARNING_MARGIN)),
        "non_learning_min_epochs": int(cfg.get("non_learning_min_epochs",
                                               NON_LEARNING_MIN_EPOCHS)),
        "save_weights": bool(cfg.get("save_weights", False)),
        "weights_home": cfg.get("weights_home"),
    }
    mlp_fixed = cfg.get("mlp_fixed")
    if mlp_fixed:                                   # fallback opzionale (retro-compat)
        pa["mlp_fixed"] = {k: float(v) for k, v in mlp_fixed.items()}
    return pa


def setup(parent_args: dict) -> dict:
    """Caricato UNA VOLTA per trainable: cache phi + split leak-free del PA
    (attack_split=False: gli attacchi NON entrano nel training; att_es dallo stream legacy)."""
    data = load_phi_cache(parent_args["cache_dir"], parent_args["labels_val"],
                          val_frac=parent_args["val_frac"],
                          split_seed=parent_args["split_seed"],
                          es_n_attacks=parent_args["es_n_attacks"],
                          attack_split=False)
    ben_train_abs = data["ben_pos"][data["idx_train"]]
    ben_sel_abs = data["ben_pos"][data["idx_select"]]
    prevalence_sel = len(data["att_es"]) / (len(ben_sel_abs) + len(data["att_es"]))
    log.info("setup pa: ben_train=%d ben_sel=%d att_full=%d att_es=%d "
             "(prevalenza per-epoca=%.4f)", len(ben_train_abs), len(ben_sel_abs),
             len(data["att_pos"]), len(data["att_es"]), prevalence_sel)
    return {**data, "ben_train_abs": ben_train_abs, "ben_sel_abs": ben_sel_abs,
            "prevalence_sel": float(prevalence_sel)}


def _make_pa_train_ds(Zs, R, ben_train_abs, batch: int, seed: int,
                      delta: float, alpha_lo: float, alpha_hi: float):
    """tf.data per il PA: per ogni batch di `batch` benigni (ordine permutato UNA volta dal
    seme, pattern DAE/ceiling) genera `batch` pseudo-anomalie ONLINE (1:1) e mescola. Il rng
    numpy vive nella CLOSURE: lo stato persiste fra le epoche (pseudo diverse a ogni epoca,
    come il legacy), deterministico dato il seme perche' from_generator consuma il generatore
    in modo sequenziale (nessun parallelismo sul generator)."""
    import tensorflow as tf

    rng = np.random.default_rng(seed)
    n = (len(ben_train_abs) // batch) * batch
    order = rng.permutation(len(ben_train_abs))[:n]
    idx = ben_train_abs[order]
    steps = n // batch

    def gen():
        for b in range(steps):
            rows = idx[b * batch:(b + 1) * batch]
            phi_real = assemble_phi(Zs, R, rows)
            Zb, Rb = phi_real[:, :C.Z_DIM], phi_real[:, C.Z_DIM:]
            phi_fake = generate_fakes(Zb, Rb, rng, delta, alpha_lo, alpha_hi)
            X = np.vstack([phi_real, phi_fake])
            y = np.concatenate([np.zeros(batch, np.float32), np.ones(batch, np.float32)])
            sh = rng.permutation(len(y))
            yield X[sh], y[sh]

    sig = (tf.TensorSpec((2 * batch, C.PHI_DIM), tf.float32),
           tf.TensorSpec((2 * batch,), tf.float32))
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig).repeat().prefetch(
        tf.data.AUTOTUNE)
    return ds, steps


def _fit_pa(config: dict, seed: int, shared_state: dict, parent_args: dict):
    """Core di addestramento del PA (condiviso con loao/fp_mining): seeding deterministico,
    build (h1/n_layers cercati + mlp_fixed), BCE su benigni-vs-pseudo, ES su AUC-PR del
    selection set (ben_sel_abs ∪ att_es dello shared_state passato), restore-best.
    Ritorna (model, callback)."""
    import tensorflow as tf

    from lib.dae.callback import PrAucCallback
    from lib.dae.objective import seed_ap
    from lib.secondo_stadio.model import build_mlp, h1_from_config

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
    # CONFIG-FIRST (la PA cerca ratio/lr/dropout); mlp_fixed = fallback per i path legacy che
    # riusano _fit_pa con config parziali (loao/fp_mining). h1 da exp (o h1 legacy); l2 fisso.
    fixed = parent_args.get("mlp_fixed") or {}
    params = {"h1": h1_from_config(config), "n_layers": int(config["n_layers"]),
              "ratio": float(config.get("ratio", fixed.get("ratio"))),
              "dropout": float(config.get("dropout", fixed.get("dropout"))),
              "l2_reg": float(config.get("l2_reg", fixed.get("l2_reg", C.L2_FIXED_S2)))}
    model = build_mlp(params)
    model.compile(optimizer=tf.keras.optimizers.Adam(float(config.get("lr", fixed.get("lr")))),
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
    train_ds, steps = _make_pa_train_ds(Zs, R, shared_state["ben_train_abs"],
                                        int(parent_args["batch"]), int(seed),
                                        float(config["delta"]),
                                        float(config["alpha_lo"]),
                                        float(config["alpha_hi"]))
    model.fit(train_ds, epochs=int(parent_args["max_epochs"]), steps_per_epoch=steps,
              callbacks=[cb], verbose=0)
    return model, cb


def train_one_seed(config: dict, seed: int, shared_state: dict,
                   parent_args: dict, seed_dir: Path,
                   return_scores: bool = False) -> dict:
    """Addestra il PA per un seme. Metrica del trial: MCC_bal con soglia standalone (sweep
    percentile) su benigni-select ∪ ATTACCHI COMPLETI; informative: AUC-ROC (sentinella) e
    MCC@FPR1% (comparabilita' col primo stadio). `return_scores=True` aggiunge gli score
    (clf_ben_sel/clf_att) al risultato: serve al fp-mining (regola AND al punto operativo)."""
    from sklearn.metrics import roc_auc_score

    Zs, R = shared_state["Zs"], shared_state["R"]
    ben_sel = shared_state["ben_sel_abs"]
    att_pos = shared_state["att_pos"]

    model, cb = _fit_pa(config, seed, shared_state, parent_args)

    clf_ben = predict_phi(model, Zs, R, ben_sel)
    clf_att = predict_phi(model, Zs, R, att_pos)
    best = best_mcc_bal_standalone(clf_ben, clf_att)
    y_full = np.concatenate([np.zeros(len(clf_ben)), np.ones(len(clf_att))])
    auc_roc = float(roc_auc_score(y_full, np.concatenate([clf_ben, clf_att])))
    op = mcc_at_fpr(clf_ben, clf_att, FPR_TARGET)

    weights_path = None
    if parent_args.get("save_weights"):
        seed_dir.mkdir(parents=True, exist_ok=True)
        weights_path = str(seed_dir / "best.weights.h5")
        model.save_weights(weights_path)
        (seed_dir / "history_ap.json").write_text(json.dumps(
            [float(x) if x is not None else None for x in cb.history_ap]))

    out = {
        "seed": int(seed),
        "mcc_bal": float(best["mcc_bal"]),
        "p_star": float(best["p_star"]),
        "tau_clf": float(best["tau"]),
        "fpr": float(best["fpr"]),
        "recall": float(best["recall"]),
        "auc_roc": auc_roc,
        "mcc_fpr1": float(op["mcc"]),
        "best_ap": float(cb.best_ap) if cb.best_ap > -1e18 else float("nan"),
        "best_epoch": int(cb.best_epoch),
        "n_epochs": len(cb.history_ap),
        "filter_reason": cb.filter_reason,
        "weights_path": weights_path,
    }
    if return_scores:
        out["clf_ben_sel"], out["clf_att"] = clf_ben, clf_att
    return out


def aggregate(per_seed_results: list, config: dict) -> dict:
    """objective = mean(MCC_bal) − K_STD·std (ddof=0) sui semi GENUINI; sentinella -1 se
    nessun genuino. AUC-ROC/MCC@FPR1%/AUC-PR informative (medie sui genuini + scalari `*_last`
    per la lettura per-seme diretta da result.json)."""
    genuine = [r for r in per_seed_results if r.get("filter_reason") != "non_learning"]
    mccs = [r["mcc_bal"] for r in genuine
            if r["mcc_bal"] is not None and math.isfinite(float(r["mcc_bal"]))]
    obj = aggregate_seeds([r["mcc_bal"] for r in genuine], K_STD)
    if math.isnan(obj):
        obj = -1.0
    last = per_seed_results[-1]

    def _mean(key):
        return float(np.mean([r[key] for r in genuine])) if genuine else float("nan")

    return {
        "objective": float(obj),
        "n_genuine": len(mccs),
        "mccbal_mean": float(np.mean(mccs)) if mccs else float("nan"),
        "mccbal_std": float(np.std(mccs)) if mccs else float("nan"),
        "aucroc_mean": _mean("auc_roc"),
        "mcc_fpr1_mean": _mean("mcc_fpr1"),
        "ap_mean": _mean("best_ap"),
        "seed_last": int(last["seed"]),
        "mccbal_last": float(last["mcc_bal"]),
        "aucroc_last": float(last["auc_roc"]),
        "mcc_fpr1_last": float(last["mcc_fpr1"]),
        "ap_last": float(last["best_ap"]),
    }


REPORT_EXTRA_KEYS = ("n_genuine", "mccbal_mean", "mccbal_std", "aucroc_mean",
                     "mcc_fpr1_mean", "ap_mean", "seed_last", "mccbal_last",
                     "aucroc_last", "mcc_fpr1_last", "ap_last")


def make_pa_component(seeds=None) -> SearchComponent:
    seeds_list = list(seeds) if seeds else list(C.SEEDS_PA)
    return SearchComponent(
        name="pa",
        seeds=seeds_list,
        metric_name="objective",
        define_space=define_space,
        build_parent_args=build_parent_args,
        setup=setup,
        train_one_seed=train_one_seed,
        aggregate=aggregate,
        report_extra_keys=REPORT_EXTRA_KEYS,
        max_t=None,   # = len(seeds): 8 semi base; rung a 4 e 6 (grace=4, rf=1.5 dal config YAML)
    )


def run_pa_search_stage(cfg: dict) -> dict:
    """Entry dello stage 'pa_search': componente PA + motore condiviso."""
    from lib.search.engine import run_search
    component = make_pa_component(seeds=cfg.get("seeds"))
    return run_search(component, cfg)
