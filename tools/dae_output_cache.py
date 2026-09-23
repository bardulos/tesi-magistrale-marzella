#!/usr/bin/env python
# TAG: DRIVER-PIPELINE | 2026-07-13 | cache phi canonica cap2 (config output_cache*.yaml) | vedi docs/refactoring_censimento.md
"""tools/dae_output_cache.py — CHIUSURA del cap2: caratterizzazione dell'uscita del DAE #569 (seme 2034).

Forward del DAE congelato su val+test → attivazioni latenti z(16) standardizzate + residui r(91) sbiancati
(Ledoit-Wolf sui benigni val). Salva la cache (whitening_params, z-scaler, z/r, score), la tabella L1/L2 +
score-DAE su TEST (MCC/precision/recall/F1/FPR realizzato, soglia p99 su val benigni), e le figure di
chiusura (box AUC-PR dei 25 modelli LOAO, curve PR/ROC + confusione #569, distribuzioni L1/L2), che si
producono solo se e' disponibile lib/plotting, non incluso nel repository.

NON è il secondo stadio: qui si caratterizza SOLO l'uscita del DAE. z e r restano separati (nessuna
concatenazione φ). #569/2034 congelato (solo forward, nessun training).

Uso:  python tools/dae_output_cache.py configs/dae/output_cache.yaml [--override chiave=valore ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lib.cli import apply_overrides, resolve_paths  # noqa: E402
from lib.utils import (N_BINARY, N_CONTINUOUS, N_FEATURES,  # noqa: E402
                       setup_blas_env)

# Config congelata del #569 (DB600). NON sovrascrivibile da override: identifica il modello.
CFG_569 = dict(btl=16, exp=112, mid=80, sigma=0.15834)
PATH_KEYS = {"weights", "val_dir", "val_attack", "test_dir", "out_dir", "apply_from"}


def load_ap_per_model_ordered(repo: Path):
    """AP per-seme (semi di compressione) per i 25 finalisti, NELL'ORDINE di finalists_ordered.csv.
    Join finalists_ordered.trial_id ↔ compress_extra.trial_id. Da disco, nessun modello rieseguito."""
    fo = repo / "runs/dae/loao_fase3/finalists_ordered.csv"
    ce = repo / "runs/dae/compress_fase2/compress_extra.csv"
    ap_by_tid: dict[str, list] = {}
    with open(ce) as f:
        for row in csv.DictReader(f):
            try:
                ap_by_tid.setdefault(row["trial_id"], []).append(float(row["ap"]))
            except (ValueError, KeyError):
                pass
    ordered = []
    with open(fo) as f:
        for row in csv.DictReader(f):
            ordered.append((row["trial_num"], ap_by_tid.get(row["trial_id"], [])))
    return ordered


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    if not a.config.exists():
        raise SystemExit(f"config non trovato: {a.config}")
    cfg = yaml.safe_load(a.config.read_text())
    cfg = apply_overrides(cfg, a.override)
    cfg = resolve_paths(cfg, REPO, PATH_KEYS)
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    figs = out / "figs"; figs.mkdir(exist_ok=True)

    setup_blas_env(int(cfg.get("threads", 8)), deterministic=True)
    import numpy as np
    import tensorflow as tf

    from lib.dae import constants as C
    from lib.dae.evaluate import (binary_metrics_at_threshold,
                                  compute_scores_and_residuals, derive_whitening,
                                  pr_and_roc_curves, score_l1, score_l2,
                                  whiten_residuals)
    from lib.dae.model import build_dae
    from lib.dae.threshold import compute_threshold

    cont_idx = list(range(N_CONTINUOUS)); bin_idx = list(range(N_CONTINUOUS, N_FEATURES))
    model, encoder = build_dae(n_features=N_FEATURES, btl=CFG_569["btl"], exp=CFG_569["exp"],
                               n_continuous=N_CONTINUOUS, n_binary=N_BINARY,
                               continuous_idx=cont_idx, binary_idx=bin_idx, l2_reg=C.L2_FIXED,
                               noise_std=CFG_569["sigma"], noise_type=C.NOISE_TYPE,
                               loss_continuous=C.LOSS_CONTINUOUS, mid=CFG_569["mid"])
    model.load_weights(cfg["weights"])
    print(f"[output_cache] #569 seme 2034 caricato da {cfg['weights']}", flush=True)

    @tf.function(reduce_retracing=True,
                 input_signature=[tf.TensorSpec(shape=(None, N_FEATURES), dtype=tf.float32)])
    def predict_fn(x):
        return model(x, training=False)

    def encode_chunked(X, chunk=50000):
        Z = np.empty((len(X), CFG_569["btl"]), dtype=np.float32)
        for i in range(0, len(X), chunk):
            j = min(i + chunk, len(X))
            xb = np.ascontiguousarray(X[i:j], dtype=np.float32)
            Z[i:j] = np.asarray(encoder(tf.constant(xb), training=False))
        return Z

    def load_set(prefix, x, y, attack):
        X = np.load(x, mmap_mode="r")
        Y = np.asarray(np.load(y)).astype(np.int8)
        # allow_pickle: le label Attack sono object-array di stringhe ("Benign"/…) prodotte dal NOSTRO
        # preprocessing (sorgente fidata, non input esterno) — necessario per gli object array, sicuro.
        # attack=None (regime apply_from / UNSW senza label multiclasse): A=None, niente np.load.
        A = np.load(attack, allow_pickle=True) if attack is not None else None
        print(f"[output_cache] {prefix}: X{X.shape} benigni={int((Y==0).sum())} attacchi={int((Y==1).sum())}",
              flush=True)
        return X, Y, A

    # ---------- VAL: forward, poi sbiancamento + z-scaler (FIT su CSE oppure APPLY da apply_from) ----------
    # apply_from (regime DEPLOY leak-free, cross-dataset): APPLICA whitening + z-scaler CONGELATI di un
    # altro dataset (CSE, runs/dae/output_569) SENZA rifittarli qui. Serve a portare phi=[z_s, r_w] di UNSW
    # nello STESSO spazio atteso dal 2° stadio addestrato su CSE. Se apply_from è assente: FIT locale (path
    # CSE, semantica invariata — ramo else identico all'originale, verificabile per equivalenza statica).
    apply_from = cfg.get("apply_from")
    vd = Path(cfg["val_dir"])
    # in apply_from non serve la label multiclasse Attack (nessun fit Ledoit-Wolf): attack=None.
    val_attack_path = None if apply_from else cfg["val_attack"]
    val_X, val_y, val_attack = load_set("val", vd / "val_X.npy", vd / "val_y.npy", val_attack_path)
    z_val = encode_chunked(val_X)
    dae_score_val, r_val = compute_scores_and_residuals(predict_fn, val_X)
    ben = (val_y == 0)
    if apply_from:
        wp = np.load(Path(apply_from) / "whitening_params.npz")
        zs = np.load(Path(apply_from) / "z_scaler.npz")
        mu, L, sigma_R = wp["mu"], wp["L"], wp["sigma_R"]
        mu_z, sd_z = zs["mu_z"], zs["sd_z"]
        shrink = -1.0                                        # sentinella: nessun Ledoit-Wolf fittato qui
        chol_source = f"applied_from:{apply_from}"
        print(f"[output_cache] APPLY (no fit): whitening + z-scaler CONGELATI da {apply_from}", flush=True)
    else:
        mu, L, sigma_R, shrink, chol_source = derive_whitening(r_val, val_attack)
        print(f"[output_cache] Ledoit-Wolf: shrinkage={shrink:.6f} chol={chol_source}", flush=True)
        mu_z = z_val[ben].mean(0); sd_z = z_val[ben].std(0); sd_z[sd_z < 1e-8] = 1.0
    r_val_w = whiten_residuals(r_val, mu, L, out=r_val)      # in-place (uguale per entrambi i rami)
    z_s_val = (z_val - mu_z) / sd_z                          # (uguale per entrambi i rami)

    # ---------- TEST: forward + applica le STESSE mu,L (no leakage) + z-scaler ----------
    td = Path(cfg["test_dir"])
    # test_Attack.npy opzionale (UNSW non ha la label multiclasse): se manca → None.
    test_attack_path = td / "test_Attack.npy"
    if not test_attack_path.exists():
        test_attack_path = None
    test_X, test_y, test_attack = load_set("test", td / "test_X.npy", td / "test_y.npy", test_attack_path)
    z_test = encode_chunked(test_X)
    dae_score_test, r_test = compute_scores_and_residuals(predict_fn, test_X)
    r_test_w = whiten_residuals(r_test, mu, L, out=r_test)
    z_s_test = (z_test - mu_z) / sd_z

    # ---------- salvataggio cache ----------
    np.savez(out / "whitening_params.npz", mu=mu, L=L, sigma_R=sigma_R)
    np.savez(out / "z_scaler.npz", mu_z=mu_z, sd_z=sd_z)
    np.save(out / "z_s_val.npy", z_s_val); np.save(out / "z_s_test.npy", z_s_test)
    np.save(out / "r_val_w.npy", r_val_w); np.save(out / "r_test_w.npy", r_test_w)
    np.save(out / "dae_score_val.npy", dae_score_val); np.save(out / "dae_score_test.npy", dae_score_test)
    build_info = dict(trial=569, seed=2034, **CFG_569, shrinkage=shrink, chol_source=chol_source,
                      z_dim=int(z_val.shape[1]), r_dim=int(r_val_w.shape[1]),
                      n_val=int(len(val_y)), n_test=int(len(test_y)), n_benign_val=int(ben.sum()),
                      apply_from=(str(apply_from) if apply_from else None),
                      dataset=("unsw" if apply_from else "cse"))
    (out / "build_info.json").write_text(json.dumps(build_info, indent=2))

    # ---------- tabella L1/L2 + score-DAE su TEST (soglia p99 su val benigni) ----------
    l1_val, l2_val = score_l1(r_val_w), score_l2(r_val_w)
    l1_test, l2_test = score_l1(r_test_w), score_l2(r_test_w)
    rows = []
    for name, sval, stest in (("L1", l1_val, l1_test), ("L2", l2_val, l2_test),
                              ("score_DAE", dae_score_val, dae_score_test)):
        tau = float(compute_threshold(np.asarray(sval)[ben]))
        m = binary_metrics_at_threshold(test_y, stest, tau)
        rows.append(dict(metric=name, tau=tau, mcc=m["mcc"], precision=m["precision"],
                         recall=m["recall"], f1=m["f1"], fpr_test=m["fpr"]))
    with open(out / "l1l2_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "tau", "mcc", "precision", "recall", "f1", "fpr_test"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\n=== L1/L2 + score-DAE su TEST (soglia p99 su val benigni) ===", flush=True)
    print(f"{'metric':>10} {'tau':>10} {'MCC':>8} {'prec':>7} {'recall':>7} {'F1':>7} {'FPR_test':>9}")
    for r in rows:
        print(f"{r['metric']:>10} {r['tau']:>10.5g} {r['mcc']:>8.4f} {r['precision']:>7.4f} "
              f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['fpr_test']:>9.4f}")

    # ---------- figure (guarded: il calcolo/cache sopra è già salvato senza plotting) ----------
    try:
        # in apply_from (UNSW) le figure sono CSE-specifiche (load_ap_per_model_ordered legge i file LOAO
        # CSE): si saltano subito, il guard esistente stampa "plotting saltato" e la cache resta salvata.
        if apply_from:
            raise RuntimeError("figure saltate in apply_from (CSE-specifiche: loao/AP per-modello)")
        from lib.plotting.dae_output_plots import (plot_auc_pr_boxplot_models,
                                                   plot_confusion_matrix, plot_pr_curve,
                                                   plot_roc_curve, plot_whitened_residual_norms)
        from sklearn.metrics import average_precision_score, roc_auc_score
        tau_dae = float(compute_threshold(dae_score_val[ben]))
        prec, rec, fpr, tpr = pr_and_roc_curves(test_y, dae_score_test)
        ap_score = float(average_precision_score(test_y, dae_score_test))
        auc_roc = float(roc_auc_score(test_y, dae_score_test))
        plot_pr_curve(prec, rec, ap_score, figs / "pr_curve_569_test.png")
        plot_roc_curve(fpr, tpr, auc_roc, figs / "roc_curve_569_test.png")
        md = binary_metrics_at_threshold(test_y, dae_score_test, tau_dae)
        plot_confusion_matrix(md["tp"], md["fp"], md["tn"], md["fn"],
                              figs / "confusion_569_test.png", metrics=md)
        tb = (test_y == 0)
        tau_l1 = float(compute_threshold(l1_val[ben])); tau_l2 = float(compute_threshold(l2_val[ben]))
        plot_whitened_residual_norms(l1_test[tb], l1_test[~tb], l2_test[tb], l2_test[~tb],
                                     tau_l1, tau_l2, figs / "whitened_norms_569_test.png")
        ap_ordered = load_ap_per_model_ordered(REPO)
        plot_auc_pr_boxplot_models(ap_ordered, figs / "auc_pr_box_25models.png",
                                   title="AUC-PR per modello (25 finalisti LOAO, semi di compressione)",
                                   highlight=569)
        print(f"\n[output_cache] figure in {figs}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[output_cache] plotting saltato (cache già salvata): {e}", flush=True)

    print(f"\n[output_cache] cache in {out}  (z_dim={build_info['z_dim']}, r_dim={build_info['r_dim']})",
          flush=True)


if __name__ == "__main__":
    main()
