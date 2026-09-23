#!/usr/bin/env python
# TAG: ONE-SHOT | 2026-07-13 | trainer locale UNSW (target degli shim) | vedi docs/refactoring_censimento.md
"""tools/deploy_unsw_train.py — addestramento LOCALE (workstation) di un DAE su UNSW, TF puro, NO Ray.

Un (config, seme) per processo. Stessi parametri della search/compressione (max_epochs=100,
patience=18, batch=512, margin=0.05, L2/noise/loss da constants). Calcola AUC-PR (val) e
MCC@FPR1% (τ = p99 benigni-val, misura su test).

OTTIMIZZAZIONI MEMORIA (per non saturare workstation e NON uccidere i driver Ray):
  - train_X/val_X/test_X via MEMMAP (mmap_mode='r') → page-cache CONDIVISA tra i processi
    paralleli (una sola copia in RAM dei file, non N);
  - scoring a chunk (compute_scores_chunked) → forward a memoria limitata, mai y_pred intero;
  - tf.data prefetch piccolo (2), batch copiati uno alla volta dal memmap;
  - thread BLAS/TF limitati (default 4) → 4 processi × 4 thread = 16.

Output: una riga CSV per (tag, seed) in --out (file dedicato per job → niente race di append).

Uso (un job): python tools/deploy_unsw_train.py --exp 112 --mid 80 --btl 16 --sigma 0.1583 \\
                 --lr 0.00207 --seed 42 --tag 569 --data-dir runs/unsw_local/data \\
                 --out runs/unsw_local/results/569_seed42.csv --threads 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, required=True)
    ap.add_argument("--mid", type=int, required=True)
    ap.add_argument("--btl", type=int, required=True)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=18)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--margin", type=float, default=0.05)
    args = ap.parse_args()

    # Env BLAS/TF PRIMA di importare numpy/TF (thread limitati + op deterministe best-effort)
    from lib.utils import setup_blas_env
    setup_blas_env(args.threads, deterministic=True)

    import numpy as np
    import tensorflow as tf

    from lib.dae import constants as C
    from lib.dae.callback import PrAucCallback
    from lib.dae.evaluate import binary_metrics_at_threshold, compute_scores_chunked
    from lib.dae.model import build_dae, make_dae_loss
    from lib.dae.objective import seed_ap
    from lib.dae.threshold import compute_threshold
    from lib.utils import N_BINARY, N_CONTINUOUS, N_FEATURES

    tf.keras.utils.set_random_seed(int(args.seed))
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(int(args.threads))

    d = Path(args.data_dir)
    # MEMMAP sugli array grandi (page-cache condivisa); y minuscoli in RAM
    train_X = np.load(d / "train_X.npy", mmap_mode="r")
    val_X   = np.load(d / "val_X.npy", mmap_mode="r")
    test_X  = np.load(d / "test_X.npy", mmap_mode="r")
    val_y   = np.asarray(np.load(d / "val_y.npy")).astype(np.int8)
    test_y  = np.asarray(np.load(d / "test_y.npy")).astype(np.int8)
    prevalence = float(val_y.mean())
    batch = int(args.batch)
    cont_idx, bin_idx = list(range(N_CONTINUOUS)), list(range(N_CONTINUOUS, N_FEATURES))

    model, encoder = build_dae(
        n_features=N_FEATURES, btl=int(args.btl), exp=int(args.exp),
        n_continuous=N_CONTINUOUS, n_binary=N_BINARY,
        continuous_idx=cont_idx, binary_idx=bin_idx,
        l2_reg=C.L2_FIXED, noise_std=float(args.sigma), noise_type=C.NOISE_TYPE,
        loss_continuous=C.LOSS_CONTINUOUS, mid=int(args.mid))
    model.compile(optimizer=tf.keras.optimizers.Adam(float(args.lr)),
                  loss=make_dae_loss(cont_idx, bin_idx, C.LOSS_CONTINUOUS))

    @tf.function(reduce_retracing=True,
                 input_signature=[tf.TensorSpec(shape=(None, N_FEATURES), dtype=tf.float32)])
    def predict_fn(x):
        return model(x, training=False)

    def ap_fn():
        return seed_ap(val_y, compute_scores_chunked(predict_fn, val_X))

    cb = PrAucCallback(ap_fn=ap_fn, patience=int(args.patience), prevalence=prevalence,
                       min_epochs=int(C.NON_LEARNING_MIN_EPOCHS), margin=float(args.margin))

    # train_ds: batch dal MEMMAP, ordine permutato dal seme; prefetch piccolo (memoria contenuta)
    n_batches = len(train_X) // batch
    order = np.random.default_rng(int(args.seed)).permutation(n_batches)

    def gen():
        for k in order:
            s = int(k) * batch
            yield (np.ascontiguousarray(train_X[s:s + batch], dtype=np.float32),) * 2

    sig = (tf.TensorSpec((batch, N_FEATURES), tf.float32),) * 2
    ds = (tf.data.Dataset.from_generator(gen, output_signature=sig)
          .repeat().prefetch(2))
    model.fit(ds, epochs=int(args.max_epochs), steps_per_epoch=n_batches,
              callbacks=[cb], verbose=0)

    # AUC-PR (val) + MCC@FPR1% (τ su benigni-val, misura su test)
    score_val = compute_scores_chunked(predict_fn, val_X)
    ap = seed_ap(val_y, score_val)
    tau = compute_threshold(score_val[val_y == 0])
    del score_val
    score_test = compute_scores_chunked(predict_fn, test_X)
    mcc = binary_metrics_at_threshold(test_y, score_test, tau)["mcc"]

    # ---- ANALISI LATENTE (collasso-per-dati vs spegnimento-per-regolarizzazione) ----
    # z = output GREZZO dell'encoder (post-ReLU del layer 'bottleneck'), sui benigni UNSW val.
    # training=False → GaussianNoise inerte → z deterministico.
    benign = np.ascontiguousarray(val_X[val_y == 0], dtype=np.float32)
    zchunks = []
    for i in range(0, len(benign), 50000):
        zchunks.append(np.asarray(encoder(benign[i:i + 50000], training=False)))
    z = np.concatenate(zchunks, axis=0)
    del benign, zchunks
    var_k = z.var(axis=0).astype(np.float64)               # varianza attivazione per componente
    mean_k = z.mean(axis=0).astype(np.float64)
    frac_active_samples = (z > 0).mean(axis=0).astype(np.float64)   # quota campioni con ReLU acceso
    n_active_var = int(np.sum(var_k / var_k.max() >= 0.01))         # attive: var_k/var_max ≥ 1%
    # PCA inline: dimensionalità effettiva (autovalori covarianza di z centrato)
    zc = (z - z.mean(axis=0, keepdims=True)).astype(np.float64)
    cov = (zc.T @ zc) / zc.shape[0]
    ev = np.clip(np.linalg.eigvalsh(cov)[::-1], 0, None)
    cum = np.cumsum(ev) / ev.sum()
    pca = {c: int(np.searchsorted(cum, c) + 1) for c in (0.90, 0.95, 0.99)}
    pca_spectrum = (ev / ev.sum()).astype(np.float64)
    del z, zc, cov
    # Norme dei pesi: encoder→z_k (colonna k del 'bottleneck') e z_k→decoder (riga k del 1° dec)
    W_enc = model.get_layer("bottleneck").get_weights()[0]         # (in_dim, btl)
    enc_norm = np.linalg.norm(W_enc, axis=0).astype(np.float64)    # per componente k
    dec_layer = "dec_mid" if int(args.mid) > 0 else "dec_exp"
    W_dec = model.get_layer(dec_layer).get_weights()[0]            # (btl, out_dim)
    dec_norm = np.linalg.norm(W_dec, axis=1).astype(np.float64)    # per componente k

    out = Path(args.out)
    np.savez(out.with_suffix(".npz"), var=var_k, mean=mean_k, frac_active=frac_active_samples,
             enc_norm=enc_norm, dec_norm=dec_norm, btl=int(args.btl),
             n_active_var=n_active_var, pca90=pca[0.90], pca95=pca[0.95], pca99=pca[0.99],
             pca_spectrum=pca_spectrum, ap=float(ap), mcc=float(mcc),
             tag=str(args.tag), seed=int(args.seed))
    model.save_weights(str(out.with_suffix("")) + ".weights.h5")
    print(f"  [latent] btl={args.btl} attive(var)={n_active_var} "
          f"PCA90={pca[0.90]} PCA95={pca[0.95]} PCA99={pca[0.99]}", flush=True)
    out.write_text(
        "tag,seed,exp,mid,btl,sigma,lr,ap,mcc,n_epochs,filter_reason\n"
        f"{args.tag},{args.seed},{args.exp},{args.mid},{args.btl},{args.sigma},{args.lr},"
        f"{ap:.8f},{mcc:.8f},{len(cb.history_ap)},{cb.filter_reason}\n")
    print(f"[{args.tag} seed {args.seed}] AP={ap:.5f} MCC@1%={mcc:.5f} "
          f"epochs={len(cb.history_ap)} filt={cb.filter_reason}", flush=True)


if __name__ == "__main__":
    main()
