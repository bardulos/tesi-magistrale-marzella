"""lib/dae/loao.py — LOAO per-classe del DAE one-class (repov6, FASE 3).

Per un DAE one-class il leave-one-attack-out si riduce a una misura per-classe a soglia fissa:
escludere una classe d'attacco NON cambia né il training (il DAE vede solo benigni) né la soglia
(p99 sui benigni). Quindi:
  - threshold = p99 sugli score benigni di validazione (lib.dae.threshold);
  - detection-rate per classe d'attacco = frazione di flussi della classe con score > threshold;
  - Z-DR = media dei detection-rate sulle classi (copertura aggregata);
  - Cohen's d per classe = separazione score-classe vs score-benigni.

`loao_metrics` è PURA (NumPy): dato score+etichette → metriche, testabile senza TF. `run_loao`
fa il forward del modello (config + pesi salvati) e poi chiama loao_metrics: inferenza deterministica
(il GaussianNoise è inerte a training=False), validata sui dati reali.
"""
from __future__ import annotations

import math

import numpy as np

from lib.dae import constants as C
from lib.dae.evaluate import cohens_d
from lib.dae.threshold import compute_threshold


def loao_metrics(scores, attack, fpr_target: float = C.FPR_TARGET,
                 benign_name: str = "Benign") -> dict:
    """Metriche LOAO per-classe da score di anomalia + etichette di classe.
    Ritorna {threshold, dr_by_class, z_dr, overall_dr, cohens_d_by_class, n_by_class}.

    - z_dr       = media MACRO dei detection-rate per-classe (ogni classe pesa uguale);
    - overall_dr = detection-rate MICRO = frazione di TUTTI gli attacchi uniti (pool di tutte
      le classi non-benigne) con score > soglia. È il "detection-rate complessivo" che unifica
      le classi: dominato dalle classi numerose, a differenza dello Z-DR macro.
    """
    scores = np.asarray(scores, dtype=float)
    attack = np.asarray(attack).astype(str)
    benign = scores[attack == benign_name]
    threshold = compute_threshold(benign, fpr_target)
    classes = sorted(c for c in np.unique(attack).tolist() if c != benign_name)
    dr_by_class, n_by_class, cohens_d_by_class = {}, {}, {}
    for c in classes:
        sc = scores[attack == c]
        n_by_class[c] = int(len(sc))
        dr_by_class[c] = float((sc > threshold).mean()) if len(sc) else float("nan")
        cohens_d_by_class[c] = cohens_d(sc, benign)
    drs = [dr_by_class[c] for c in classes if not math.isnan(dr_by_class[c])]
    z_dr = float(np.mean(drs)) if drs else float("nan")
    attack_scores = scores[attack != benign_name]
    overall_dr = float((attack_scores > threshold).mean()) if len(attack_scores) else float("nan")
    return {"threshold": float(threshold), "dr_by_class": dr_by_class, "z_dr": z_dr,
            "overall_dr": overall_dr, "cohens_d_by_class": cohens_d_by_class,
            "n_by_class": n_by_class}


def run_loao(weights_path, config: dict, val_X, val_attack,
             fpr_target: float = C.FPR_TARGET, benign_name: str = "Benign") -> dict:
    """Forward del DAE (config + pesi salvati) su val_X → score → loao_metrics. Inferenza pura
    (TF a training=False: GaussianNoise inerte → deterministica). Validata sui dati reali."""
    import tensorflow as tf

    from lib.dae.evaluate import compute_scores_chunked
    from lib.dae.model import build_dae
    from lib.utils import N_BINARY, N_CONTINUOUS, N_FEATURES

    cont_idx = list(range(N_CONTINUOUS))
    bin_idx = list(range(N_CONTINUOUS, N_FEATURES))
    model, _enc = build_dae(
        n_features=N_FEATURES, btl=int(config["btl"]), exp=int(config["exp"]),
        n_continuous=N_CONTINUOUS, n_binary=N_BINARY,
        continuous_idx=cont_idx, binary_idx=bin_idx,
        l2_reg=C.L2_FIXED, noise_std=float(config["sigma"]), noise_type=C.NOISE_TYPE,
        loss_continuous=C.LOSS_CONTINUOUS, mid=int(config["mid"]))
    model.load_weights(str(weights_path))

    @tf.function(reduce_retracing=True,
                 input_signature=[tf.TensorSpec(shape=(None, N_FEATURES), dtype=tf.float32)])
    def predict_fn(x):
        return model(x, training=False)

    scores = compute_scores_chunked(predict_fn, np.asarray(val_X, dtype=np.float32))
    return loao_metrics(scores, val_attack, fpr_target, benign_name)
