#!/usr/bin/env python
# TAG: ONE-SHOT | 2026-07-13 | statistiche latenti 17 semi WP-3a (testato) | vedi docs/refactoring_censimento.md
"""tools/dae_latent_history.py — Cap4 WP-3a: re-pass di UN seme del #569 con statistiche LATENTI per-epoca.

Mirror di tools/train_569_history.py (stessa config #569, stessa ricetta di training) con due differenze:
  1. NESSUN early-stop: si addestra fino a --max-epochs (default 60, decisione chiusa n.5) e gli stop
     si simulano OFFLINE (tools/dae_latent_analysis.py), come per il PA (pa_es_history.py).
  2. Un callback aggiuntivo logga per-epoca statistiche label-free sullo SPAZIO LATENTE z (bottleneck),
     calcolate su probe-set FISSI e seedati (~10k, generati una volta prima del training):
       (a) separabilità benigni-vs-perturbati: proiezione sulla direzione dei centroidi →
           Fisher ratio + AUC Mann-Whitney. Perturbazioni = SOLO rumore gaussiano sugli input
           (3 scale: 0.5/1.0/2.0 × sigma di training; MAI pseudo del PA — anti-circolarità, C4).
       (b) dimensionalità effettiva del latente sui benigni: participation ratio + PCA95.
       (c) probe di open-space (benigni con colonne permutate indipendentemente: marginali preservate,
           struttura congiunta distrutta): distanza dal centroide benigno (ratio) + Fisher/AUC.
     Solo statistiche aggregate (pochi float/epoca): NESSUN dump di z.

Il criterio candidato (letteratura: AE-SAD 2305.10464, adversarial AE 1901.06355, Steinwart 2005):
la separabilità nel latente dovrebbe reggere oltre il picco di AUC-PR mentre la val_loss di
ricostruzione continua a scendere (criterio naïf smentito su 17/17 semi).

Uso:  python tools/dae_latent_history.py --seed 42 --data-dir <HOME>/dae_data_91 \\
          --out runs/dae/latent_history/seed_42.json --threads 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.utils import setup_blas_env  # noqa: E402

# #569 (config autorevole DB600), identica a train_569_history.py
CFG = {"btl": 16, "exp": 112, "mid": 80, "sigma": 0.15834, "lr": 0.002071}

PROBE_SEED = 20260708          # seme FISSO dei probe-set (indipendente da MON_SEED/PSEUDO_SEED del PA)
NOISE_SCALES = (0.5, 1.0, 2.0)  # multipli della sigma di training per le perturbazioni gaussiane


# ---------------------------------------------------------------- statistiche pure (testabili, no TF)
def fisher_and_auc(z_ben: np.ndarray, z_pert: np.ndarray) -> tuple[float, float]:
    """Separabilità di due nuvole in R^d sulla proiezione centroide→centroide.

    Fisher = (m1−m0)² / (v0+v1) sulla proiezione; AUC = Mann-Whitney (ties gestite con rank medi).
    AUC ~0.5 = indistinguibili; →1 = perturbati separati dai benigni.
    """
    mu0 = z_ben.mean(axis=0)
    mu1 = z_pert.mean(axis=0)
    w = mu1 - mu0
    norm = float(np.linalg.norm(w))
    if norm == 0.0:
        return 0.0, 0.5
    w = w / norm
    p0 = z_ben @ w
    p1 = z_pert @ w
    v0 = float(p0.var())
    v1 = float(p1.var())
    fisher = float((p0.mean() - p1.mean()) ** 2 / (v0 + v1 + 1e-12))
    # Mann-Whitney con rank medi (ties): AUC = (R1 − n1(n1+1)/2) / (n0·n1)
    from scipy.stats import rankdata
    n0, n1 = len(p0), len(p1)
    ranks = rankdata(np.concatenate([p0, p1]))
    r1 = float(ranks[n0:].sum())
    auc = (r1 - n1 * (n1 + 1) / 2.0) / (n0 * n1)
    return fisher, float(auc)


def participation_ratio_and_pca95(z: np.ndarray) -> tuple[float, int]:
    """Dimensionalità effettiva del latente: PR = (Σλ)²/Σλ² e n. componenti al 95% di varianza."""
    zc = z - z.mean(axis=0, keepdims=True)
    cov = (zc.T @ zc) / max(1, len(zc) - 1)
    lam = np.linalg.eigvalsh(cov.astype(np.float64))
    lam = np.clip(lam, 0.0, None)
    tot = float(lam.sum())
    if tot <= 0.0:
        return 0.0, 0
    pr = float(tot ** 2 / (np.square(lam).sum() + 1e-24))
    frac = np.cumsum(np.sort(lam)[::-1]) / tot
    k95 = int(np.searchsorted(frac, 0.95) + 1)
    return pr, k95


def dist_ratio(z_ben: np.ndarray, z_pert: np.ndarray) -> float:
    """Distanza media dal centroide benigno: ratio perturbati/benigni (>1 = restano fuori)."""
    mu = z_ben.mean(axis=0)
    d_ben = float(np.linalg.norm(z_ben - mu, axis=1).mean())
    d_pert = float(np.linalg.norm(z_pert - mu, axis=1).mean())
    return d_pert / (d_ben + 1e-12)


def column_permute(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Open-space marker: ogni colonna permutata indipendentemente (marginali intatte,
    struttura congiunta distrutta)."""
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = x[rng.permutation(len(x)), j]
    return out


def build_probes(val_x, val_y, probe_size: int, sigma: float) -> dict[str, np.ndarray]:
    """Probe-set fissi e seedati: benigni + 3 famiglie gaussiane + open-space (colonne permutate)."""
    rng = np.random.default_rng(PROBE_SEED)
    ben_idx = np.flatnonzero(np.asarray(val_y) == 0)
    pick = rng.choice(ben_idx, size=int(probe_size), replace=False)
    pick.sort()
    ben = np.ascontiguousarray(val_x[pick], dtype=np.float32)
    probes = {"ben": ben}
    for scale in NOISE_SCALES:
        noise = rng.normal(0.0, scale * sigma, size=ben.shape).astype(np.float32)
        probes[f"n{int(scale * 10):02d}"] = ben + noise
    probes["open"] = column_permute(ben, rng)
    return probes


# ---------------------------------------------------------------- driver (TF solo qui)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--probe-size", type=int, default=10000)
    a = ap.parse_args()

    setup_blas_env(a.threads, deterministic=True)
    import tensorflow as tf

    from lib.dae import constants as C
    from lib.dae.callback import PrAucCallback
    from lib.dae.evaluate import compute_scores_chunked
    from lib.dae.model import build_dae, make_dae_loss
    from lib.dae.objective import seed_ap
    from lib.dae.search import _make_train_ds
    from lib.utils import N_BINARY, N_CONTINUOUS, N_FEATURES

    t0 = time.time()
    tf.keras.utils.set_random_seed(int(a.seed))
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(int(a.threads))

    d = Path(a.data_dir)
    train_mm = [np.load(p, mmap_mode="r") for p in sorted(d.glob("train_shard_*.npy"))]
    val_X = np.load(d / "val_X.npy", mmap_mode="r")
    val_y = np.asarray(np.load(d / "val_y.npy")).astype(np.int8)
    prevalence = float(val_y.mean())
    cont_idx = list(range(N_CONTINUOUS))
    bin_idx = list(range(N_CONTINUOUS, N_FEATURES))

    model, encoder = build_dae(n_features=N_FEATURES, btl=CFG["btl"], exp=CFG["exp"],
                               n_continuous=N_CONTINUOUS, n_binary=N_BINARY, continuous_idx=cont_idx,
                               binary_idx=bin_idx, l2_reg=C.L2_FIXED, noise_std=CFG["sigma"],
                               noise_type=C.NOISE_TYPE, loss_continuous=C.LOSS_CONTINUOUS,
                               mid=CFG["mid"])
    loss_fn = make_dae_loss(cont_idx, bin_idx, C.LOSS_CONTINUOUS)
    model.compile(optimizer=tf.keras.optimizers.Adam(CFG["lr"]), loss=loss_fn)

    @tf.function(reduce_retracing=True,
                 input_signature=[tf.TensorSpec(shape=(None, N_FEATURES), dtype=tf.float32)])
    def predict_fn(x):
        return model(x, training=False)

    def ap_fn():
        return seed_ap(val_y, compute_scores_chunked(predict_fn, val_X))

    # AUC-PR loggata per-epoca ma SENZA early-stop (patience oltre il cap): gli stop si simulano offline
    cb_ap = PrAucCallback(ap_fn=ap_fn, patience=int(a.max_epochs) + 1, prevalence=prevalence,
                          min_epochs=int(C.NON_LEARNING_MIN_EPOCHS), margin=float(C.NON_LEARNING_MARGIN))

    # val_loss per-epoca sui benigni (criterio naïf, per confronto): stesso subset fisso del mirror
    _bidx = np.flatnonzero(np.asarray(val_y) == 0)[:262144]
    val_benign_sub = np.ascontiguousarray(val_X[_bidx], dtype=np.float32)

    class ValLossCb(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.history = []

        def on_epoch_end(self, epoch, logs=None):
            yh = predict_fn(val_benign_sub)
            l = float(tf.reduce_mean(loss_fn(tf.convert_to_tensor(val_benign_sub), yh)).numpy())
            self.history.append(l)

    # probe fissi (generati UNA volta: le curve per-epoca riflettono solo l'evoluzione del modello)
    probes = build_probes(val_X, val_y, a.probe_size, CFG["sigma"])
    pert_keys = []
    for scale in NOISE_SCALES:
        pert_keys.append(f"n{int(scale * 10):02d}")
    pert_keys.append("open")

    class LatentStatsCb(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.history = {"participation_ratio": [], "pca95": []}
            for k in pert_keys:
                self.history[f"fisher_{k}"] = []
                self.history[f"auc_{k}"] = []
                self.history[f"dist_{k}"] = []

        def on_epoch_end(self, epoch, logs=None):
            z = {}
            for name, x in probes.items():
                z[name] = np.asarray(encoder(x, training=False))
            pr, k95 = participation_ratio_and_pca95(z["ben"])
            self.history["participation_ratio"].append(pr)
            self.history["pca95"].append(k95)
            for k in pert_keys:
                fisher, auc = fisher_and_auc(z["ben"], z[k])
                self.history[f"fisher_{k}"].append(fisher)
                self.history[f"auc_{k}"].append(auc)
                self.history[f"dist_{k}"].append(dist_ratio(z["ben"], z[k]))

    cb_vl = ValLossCb()
    cb_lat = LatentStatsCb()
    train_ds, steps = _make_train_ds(train_mm, int(a.batch), int(a.seed))
    hist = model.fit(train_ds, epochs=int(a.max_epochs), steps_per_epoch=steps,
                     callbacks=[cb_ap, cb_vl, cb_lat], verbose=0)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": int(a.seed),
        "history_ap": [float(x) if x is not None else None for x in cb_ap.history_ap],
        "history_val_loss": [float(x) for x in cb_vl.history],
        "history_train_loss": [float(x) for x in hist.history.get("loss", [])],
        "best_epoch": int(cb_ap.best_epoch), "best_ap": float(cb_ap.best_ap),
        "n_epochs": len(cb_ap.history_ap),
        "epochs_cap": int(a.max_epochs),
        "probe_seed": PROBE_SEED, "probe_size": int(a.probe_size),
        "noise_scales": list(NOISE_SCALES),
        "latent": cb_lat.history,
        "sec_total": round(time.time() - t0, 1),
    }
    out.write_text(json.dumps(payload))
    print(f"seed {a.seed}: {len(cb_ap.history_ap)} epoche in {payload['sec_total']:.0f}s, "
          f"best_ap={cb_ap.best_ap:.5f} @epoch {cb_ap.best_epoch} -> {out}")


if __name__ == "__main__":
    main()
