#!/usr/bin/env python
# TAG: ONE-SHOT | EntropyStop su dati contaminati
"""Train DAE #569 on contaminated data and analyze EntropyStop offline.

Training remains label-blind (X==y). Real attacks come only from validation and are split into
disjoint training-contamination, D_eval, and oracle pools. The protected test set is never read.
The fixed split is shared across rates; lower-rate attack sets are prefixes of higher-rate sets.
Histories include epoch 0 before training, except history_train_loss, which comes from Keras
and starts after epoch 1. Analysis uses saved curves and does not load TensorFlow.

Usage: python tools/dae_entropy_contam.py train --seed 42 --rate 0.01 --data-dir DIR --out FILE
       python tools/dae_entropy_contam.py analyze --dir DIR --rate 0.01 --pilot
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.utils import setup_blas_env  # noqa: E402

CFG = {"btl": 16, "exp": 112, "mid": 80, "sigma": 0.15834, "lr": 0.002071}
SPLIT_SEED = 0
EVAL_SEED = 0
N_POOL_MAX = 700_000
N_EVAL_OUT = 25_000
N_EVAL_DEFAULT = 262_144

K_GRID = (5, 10, 15, 18)
R_DOWN_GRID = (0.01, 0.03, 0.05, 0.1)
FIXED_EPOCH = 13
PILOT_TV_THRESHOLD = 5.0

AP_YLIM = (0.0, 1.0)
H_L_YLIM = (0.0, 13.0)


# ==================================================================== funzioni senza TF (pure, tranne l'IO .npz dei pesi)
def n_attacks_for_rate(n_benign: int, r: float) -> int:
    """Return the attack count for REPLACE contamination at rate r."""
    if not (0.0 <= r < 1.0):
        raise ValueError(f"rate fuori da [0,1): {r}")
    return int(round(r * n_benign))


def split_attack_positions(att_pos: np.ndarray, n_pool: int, n_eval_out: int,
                           split_seed: int) -> dict:
    """Split attack positions into fixed, disjoint contamination, evaluation, and oracle pools."""
    att_pos = np.asarray(att_pos)
    n_att = len(att_pos)
    need = n_pool + n_eval_out
    if need > n_att:
        raise ValueError(f"att_pos troppo piccolo ({n_att}) per pool+eval_out richiesti ({need})")
    perm = np.random.default_rng(split_seed).permutation(att_pos)
    contam_pool = perm[:n_pool]
    eval_out = perm[n_pool:n_pool + n_eval_out]
    oracle_att = perm[n_pool + n_eval_out:]
    return {"contam_pool": contam_pool, "eval_out": eval_out, "oracle_att": oracle_att}


def build_injection_schedule(n_blocks: int, batch: int, n_attacks: int, seed: int,
                             epoch: int) -> dict[int, list[tuple[int, int]]]:
    """Distribute attacks uniformly across one epoch and reshuffle them by epoch."""
    if n_attacks < 0:
        raise ValueError(f"n_attacks negativo: {n_attacks}")
    total_slots = n_blocks * batch
    if n_attacks > total_slots:
        raise ValueError(f"n_attacks ({n_attacks}) > slot disponibili ({total_slots})")
    schedule: dict[int, list[tuple[int, int]]] = {}
    if n_attacks == 0:
        return schedule
    # Evenly spaced slots avoid clustering attacks in a few batches.
    slot_positions = (np.arange(n_attacks, dtype=np.int64) * total_slots) // n_attacks
    rng = np.random.default_rng((int(seed) * 1_000_003 + int(epoch)) & 0xFFFFFFFF)
    attack_order = rng.permutation(n_attacks)
    for k in range(n_attacks):
        slot = int(slot_positions[k])
        block_ord = slot // batch
        pos_in_batch = slot % batch
        attack_idx = int(attack_order[k])
        schedule.setdefault(block_ord, []).append((pos_in_batch, attack_idx))
    return schedule


def compute_loss_entropy(v: np.ndarray) -> tuple[float, float]:
    """Compute loss entropy and total loss, treating zero terms as 0 log 0 = 0."""
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 0:
        return 0.0, 0.0
    if np.any(v < 0):
        raise ValueError("loss per-campione negativa: v deve essere >= 0")
    S = float(v.sum())
    if S <= 0.0:
        return 0.0, S
    u = v / S
    mask = u > 0.0
    H = float(-(u[mask] * np.log(u[mask])).sum())
    return H, S


def entropystop_select(entropy_hist: list[float], k: int, r_down: float) -> int:
    """Select an EntropyStop epoch using patience k and downtrend threshold r_down.

    Accept a new minimum only when its drop divided by accumulated total variation exceeds
    r_down. Otherwise patience advances; return the last accepted epoch on stop or end of history.
    """
    if k <= 0:
        raise ValueError(f"patience k deve essere positivo: {k}")
    if not (0.0 < r_down < 1.0):
        raise ValueError(f"r_down fuori da (0,1): {r_down}")
    if not entropy_hist:
        raise ValueError("entropy_hist vuota")
    e_min = float(entropy_hist[0])
    best_ep = 0
    G = 0.0
    patience = 0
    for j in range(1, len(entropy_hist)):
        e_j = float(entropy_hist[j])
        e_prev = float(entropy_hist[j - 1])
        G += abs(e_j - e_prev)
        improved = e_j < e_min and G > 0.0 and (e_min - e_j) / G > r_down
        if improved:
            e_min = e_j
            best_ep = j
            G = 0.0
            patience = 0
        else:
            patience += 1
            if patience >= k:
                return best_ep
    return best_ep


def build_deval_indices(eval_out: np.ndarray, benign_val_pos: np.ndarray, n_eval: int,
                        r: float, eval_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample D_eval from held-out benign and attack pools with a fixed, rate-nested seed."""
    n_out = int(round(r * n_eval))
    n_in = n_eval - n_out
    if n_out > len(eval_out):
        raise ValueError(f"n_out richiesti ({n_out}) > eval_out disponibili ({len(eval_out)})")
    if n_in > len(benign_val_pos):
        raise ValueError(f"n_in richiesti ({n_in}) > benigni val disponibili ({len(benign_val_pos)})")
    rng_out = np.random.default_rng(eval_seed)
    perm_out = rng_out.permutation(eval_out)
    out_idx = perm_out[:n_out]
    rng_in = np.random.default_rng(eval_seed + 1)
    perm_in = rng_in.permutation(benign_val_pos)
    in_idx = perm_in[:n_in]
    return in_idx, out_idx


def save_weights_npz(weights: list, path) -> None:
    """Save model weights to NPZ using numbered keys in list order."""
    np.savez(str(path), **{f"w{i}": np.asarray(w) for i, w in enumerate(weights)})


def load_weights_npz(path) -> list:
    """Load NPZ weights in their original numbered-key order."""
    with np.load(str(path)) as d:
        return [d[f"w{i}"] for i in range(len(d.files))]


# ==================================================================== funzioni pure (analisi offline)
def pearson(x: list[float], y: list[float]) -> float:
    """Correlazione di Pearson pura numpy (nessuna dipendenza scipy per l'analisi offline)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def tv_over_range(curve: list[float]) -> float:
    """Return total variation divided by range; constant curves return zero."""
    c = np.asarray(curve, dtype=float)
    if len(c) < 2:
        return 0.0
    rng = float(c.max() - c.min())
    if rng <= 0.0:
        return 0.0
    tv = float(np.abs(np.diff(c)).sum())
    return tv / rng


def mean_std_padded(curves) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pointwise mean and standard deviation after NaN-padding curves."""
    ml = max(len(c) for c in curves)
    pad = np.full((len(curves), ml), np.nan)
    for j, c in enumerate(curves):
        pad[j, :len(c)] = c
    return np.arange(ml), np.nanmean(pad, axis=0), np.nanstd(pad, axis=0)


def evaluate_run(run: dict, entropy_key: str) -> list[dict]:
    """Applica la griglia K_GRID x R_DOWN_GRID a UNA run, per UNA variante di entropia."""
    ap = run["history_ap"]
    entropy = run[entropy_key]
    best_ap = max(v for v in ap if v is not None)
    rows = []
    for k in K_GRID:
        for r_down in R_DOWN_GRID:
            if k >= len(entropy):
                continue
            stop_ep = entropystop_select(entropy, k, r_down)
            ap_at_stop = ap[stop_ep]
            delta = best_ap - ap_at_stop if ap_at_stop is not None else float("nan")
            rows.append({"k": k, "r_down": r_down, "stop_epoch": stop_ep,
                        "ap_at_stop": ap_at_stop, "delta": delta})
    return rows


def analyze_group(runs: list[dict]) -> dict:
    """Summarize both entropy variants across seeds for one rate/learning-rate group."""
    per_variant = {}
    for entropy_key in ("history_entropy", "history_entropy_score"):
        config_deltas: dict[tuple, list[float]] = {}
        pearsons = []
        tv_ratios = []
        cap_fixed_deltas = []
        for run in runs:
            ap = run["history_ap"]
            best_ap = max(v for v in ap if v is not None)
            entropy = run[entropy_key]
            n = min(len(ap), len(entropy))
            pearsons.append(pearson(ap[:n], entropy[:n]))
            tv_ratios.append(tv_over_range(entropy))
            fixed_idx = min(FIXED_EPOCH, len(ap) - 1)
            cap_fixed_deltas.append(best_ap - ap[fixed_idx])
            for row in evaluate_run(run, entropy_key):
                key = (row["k"], row["r_down"])
                config_deltas.setdefault(key, []).append(row["delta"])

        config_medians = {k: statistics.median(v) for k, v in config_deltas.items()}
        # Short test histories may be too short for every patience value.
        if config_medians:
            best_config = min(config_medians, key=lambda k: abs(config_medians[k]))
            best_config_out = {"k": best_config[0], "r_down": best_config[1]}
            best_config_delta = config_medians[best_config]
        else:
            best_config_out = None
            best_config_delta = float("nan")
        finite_pearsons = [p for p in pearsons if not np.isnan(p)]
        per_variant[entropy_key] = {
            "n_seeds": len(runs),
            "best_config": best_config_out,
            "best_config_median_delta": best_config_delta,
            "pearson_per_seed": pearsons,
            "pearson_median": statistics.median(finite_pearsons) if finite_pearsons else float("nan"),
            "n_seeds_pearson_lt_neg05": sum(1 for p in finite_pearsons if p < -0.5),
            "tv_over_range_per_seed": tv_ratios,
            "tv_over_range_median": statistics.median(tv_ratios) if tv_ratios else float("nan"),
            "cap_fixed_median_delta": statistics.median(cap_fixed_deltas),
            "all_configs_median": {f"k{k}_r{rd}": v for (k, rd), v in config_medians.items()},
        }
    return per_variant


def decide_pilot(groups: dict[float, dict]) -> dict:
    """Choose between exactly two learning rates at one contamination rate.

    For each learning rate, use the entropy variant with the most negative median Pearson. The
    sole lr passing Pearson and smoothness wins; if both pass Pearson, the smaller absolute median
    AP delta wins; if only one passes Pearson but not smoothness, it still wins with a warning.
    If neither passes Pearson, return STOP and keep the fixed epoch cap.
    """
    if len(groups) != 2:
        raise ValueError(f"decide_pilot richiede esattamente 2 gruppi lr, ricevuti {len(groups)}")
    lrs = sorted(groups)

    def lr_stats(per_variant: dict) -> dict:
        best_key = min(per_variant, key=lambda k: (per_variant[k]["pearson_median"]
                                                    if not np.isnan(per_variant[k]["pearson_median"])
                                                    else 0.0))
        res = per_variant[best_key]
        pearson_ok = res["n_seeds_pearson_lt_neg05"] >= 1
        tv_ok = res["tv_over_range_median"] < PILOT_TV_THRESHOLD
        return {"variant": best_key, "pearson_ok": pearson_ok, "tv_ok": tv_ok,
               "full_pass": pearson_ok and tv_ok, **res}

    stats = {lr: lr_stats(groups[lr]) for lr in lrs}
    full_pass = [lr for lr in lrs if stats[lr]["full_pass"]]
    pearson_pass = [lr for lr in lrs if stats[lr]["pearson_ok"]]

    if len(full_pass) == 1:
        w = full_pass[0]
        return {"verdict": "GO", "winner_lr": w, "stats": stats,
               "reason": f"lr={w} e' l'unico con Pearson<-0.5 e TV/range<{PILOT_TV_THRESHOLD:.0f}"}
    if len(pearson_pass) == 2:
        w = min(lrs, key=lambda lr: abs(stats[lr]["best_config_median_delta"]))
        return {"verdict": "GO", "winner_lr": w, "stats": stats,
               "reason": "entrambi i lr hanno Pearson<-0.5; vince per Delta mediano minore "
                        f"({stats[w]['best_config_median_delta']:+.5f})"}
    if len(pearson_pass) == 1:
        w = pearson_pass[0]
        return {"verdict": "GO", "winner_lr": w, "stats": stats,
               "reason": f"lr={w} e' l'unico con Pearson<-0.5 ma TV/range="
                        f"{stats[w]['tv_over_range_median']:.2f} (soglia "
                        f"{PILOT_TV_THRESHOLD:.0f} non soddisfatta: verificare la curva)"}
    return {"verdict": "STOP", "winner_lr": None, "stats": stats,
           "reason": "nessun lr ha Pearson<-0.5 su almeno 1 seme: il metodo non funziona, "
                    "si chiude con il cap-epoche fisso"}


# ==================================================================== driver TF (training)
def _make_contaminated_train_ds(train_mm, A: np.ndarray, batch: int, seed: int, n_features: int):
    """Variante di lib.dae.search._make_train_ds con iniezione REPLACE-in-block: stessa
    lettura a blocchi contigui dal mmap (permutazione dei blocchi fissa per la run, come
    l'originale), ma alcuni slot dello stream, a passo uniforme sull'intera epoca (ai tassi bassi
    non in ogni blocco), sono sostituiti con righe da A (attacchi in RAM) secondo uno schedule
    ri-permutato per-epoca (build_injection_schedule). steps_per_epoch resta quello
    dell'originale (= numero di blocchi benigni)."""
    import tensorflow as tf

    desc = [(si, b * batch) for si, s in enumerate(train_mm) for b in range(len(s) // batch)]
    order = np.random.default_rng(seed).permutation(len(desc))
    n_blocks = len(desc)
    n_attacks = len(A)

    def gen():
        epoch = 0
        while True:
            schedule = build_injection_schedule(n_blocks, batch, n_attacks, seed, epoch)
            for rank, k in enumerate(order):
                si, start = desc[int(k)]
                chunk = np.ascontiguousarray(train_mm[si][start:start + batch], dtype=np.float32)
                if not chunk.flags.writeable:
                    chunk = chunk.copy()
                for pos, a_idx in schedule.get(rank, []):
                    chunk[pos] = A[a_idx]
                yield chunk, chunk
            epoch += 1

    sig = (tf.TensorSpec((batch, n_features), tf.float32),) * 2
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(tf.data.AUTOTUNE)
    return ds, n_blocks


class _MmapView:
    """View leggero di un mmap tramite indici ordinati. __getitem__ legge on-demand dal mmap,
    restituendo un chunk contiguo in RAM. Usa zero memoria residente finché non si accede.
    Sostituisce np.ascontiguousarray(val_X[indices]) che copia tutto in RAM (~728MB per
    val_X_oracle). compute_scores_chunked usa X[i:j] → chiama __getitem__ con slice."""
    def __init__(self, mmap_arr, indices_sorted):
        self._mmap = mmap_arr
        self._idx = indices_sorted
    def __len__(self):
        return len(self._idx)
    def __getitem__(self, sl):
        return np.ascontiguousarray(self._mmap[self._idx[sl]], dtype=np.float32)


def cmd_train(a) -> None:
    setup_blas_env(a.threads, deterministic=True)
    import tensorflow as tf

    from lib.dae import constants as C
    from lib.dae.callback import PrAucCallback
    from lib.dae.evaluate import compute_scores_chunked, mse_binary_score
    from lib.dae.model import build_dae, make_dae_loss
    from lib.dae.objective import seed_ap
    from lib.utils import N_BINARY, N_CONTINUOUS, N_FEATURES

    t0 = time.time()
    tf.keras.utils.set_random_seed(int(a.seed))
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(int(a.threads))

    d = Path(a.data_dir)
    train_mm = [np.load(p, mmap_mode="r") for p in sorted(d.glob("train_shard_*.npy"))]
    n_benign = sum(len(s) for s in train_mm)
    val_X = np.load(d / "val_X.npy", mmap_mode="r")
    val_y = np.asarray(np.load(d / "val_y.npy")).astype(np.int8)
    cont_idx = list(range(N_CONTINUOUS))
    bin_idx = list(range(N_CONTINUOUS, N_FEATURES))

    # --- split anti-leak (fisso, indipendente dal seme del modello) ---
    att_pos = np.where(val_y == 1)[0]
    benign_val_pos = np.where(val_y == 0)[0]
    split = split_attack_positions(att_pos, N_POOL_MAX, N_EVAL_OUT, a.split_seed)
    n_attacks = n_attacks_for_rate(n_benign, a.rate)
    contam_idx = split["contam_pool"][:n_attacks]
    A = np.ascontiguousarray(val_X[np.sort(contam_idx)], dtype=np.float32)

    # --- oracolo AUC-PR: val-benigni UNION oracle_att (mai contaminante) ---
    # _MmapView: zero RAM residente, legge dal mmap on-demand in compute_scores_chunked
    oracle_idx_sorted = np.sort(np.concatenate([benign_val_pos, split["oracle_att"]]))
    val_X_oracle = _MmapView(val_X, oracle_idx_sorted)
    val_y_oracle = val_y[oracle_idx_sorted]
    prevalence = float(val_y_oracle.mean())

    # --- D_eval per H_L (mix nella proporzione r, mai contaminante) ---
    # indici ordinati nel mmap (zero RAM: l'entropia è calcolata in chunk in measure_entropy)
    in_idx, out_idx = build_deval_indices(split["eval_out"], benign_val_pos, int(a.n_eval),
                                          a.rate, a.eval_seed)
    deval_idx = np.sort(np.concatenate([in_idx, out_idx]))

    model, _ = build_dae(n_features=N_FEATURES, btl=CFG["btl"], exp=CFG["exp"],
                         n_continuous=N_CONTINUOUS, n_binary=N_BINARY, continuous_idx=cont_idx,
                         binary_idx=bin_idx, l2_reg=C.L2_FIXED, noise_std=CFG["sigma"],
                         noise_type=C.NOISE_TYPE, loss_continuous=C.LOSS_CONTINUOUS, mid=CFG["mid"])
    loss_fn = make_dae_loss(cont_idx, bin_idx, C.LOSS_CONTINUOUS)
    model.compile(optimizer=tf.keras.optimizers.Adam(float(a.lr)), loss=loss_fn)

    @tf.function(reduce_retracing=True,
                input_signature=[tf.TensorSpec(shape=(None, N_FEATURES), dtype=tf.float32)])
    def predict_fn(x):
        return model(x, training=False)

    def ap_fn():
        return seed_ap(val_y_oracle, compute_scores_chunked(predict_fn, val_X_oracle))

    def measure_entropy() -> tuple[float, float, float]:
        """Compute both loss entropies on D_eval in chunks."""
        chunk = 50000
        all_v_dae = []
        all_v_score = []
        for i in range(0, len(deval_idx), chunk):
            block = np.ascontiguousarray(val_X[deval_idx[i:i + chunk]], dtype=np.float32)
            yh = predict_fn(block)
            all_v_dae.append(loss_fn(tf.convert_to_tensor(block), yh).numpy())
            all_v_score.append(mse_binary_score(yh.numpy(), block, N_CONTINUOUS, N_BINARY))
            del block, yh
        v_dae = np.concatenate(all_v_dae)
        v_score = np.concatenate(all_v_score)
        H_dae, S_dae = compute_loss_entropy(v_dae)
        H_score, _ = compute_loss_entropy(v_score)
        del v_dae, v_score
        return H_dae, H_score, S_dae

    # --- baseline e0 (Algorithm 1 righe 2-5): modello APPENA INIZIALIZZATO, prima di
    # qualunque gradiente. Prependuta agli array storici a fine training (sotto): le
    # callback restano nella convenzione nativa di Keras on_epoch_end, invariata. ---
    H_dae0, H_score0, S_dae0 = measure_entropy()
    ap0 = ap_fn()

    # --- salvataggio pesi per-epoca (opzionale, --weights-dir): e0 = modello pre-gradiente ---
    weights_dir = Path(a.weights_dir) if getattr(a, "weights_dir", None) else None
    if weights_dir is not None:
        weights_dir.mkdir(parents=True, exist_ok=True)
        save_weights_npz(model.get_weights(), weights_dir / "weights_epoch_000.npz")

    # AUC-PR oracolo loggata per-epoca ma SENZA early-stop: gli stop si simulano offline
    cb_ap = PrAucCallback(ap_fn=ap_fn, patience=int(a.max_epochs) + 1, prevalence=prevalence,
                          min_epochs=int(C.NON_LEARNING_MIN_EPOCHS), margin=float(C.NON_LEARNING_MARGIN))

    class EntropyCb(tf.keras.callbacks.Callback):
        """Record D_eval entropy using DAE loss and binary-score variants."""
        def __init__(self):
            super().__init__()
            self.history_entropy: list[float] = []
            self.history_entropy_score: list[float] = []
            self.history_S: list[float] = []

        def on_epoch_end(self, epoch, logs=None):
            H_dae, H_score, S_dae = measure_entropy()
            self.history_entropy.append(H_dae)
            self.history_entropy_score.append(H_score)
            self.history_S.append(S_dae)
            # progresso nel log (sicuro: print con PYTHONUNBUFFERED=1 va a file subito)
            cur_ap = cb_ap.history_ap[-1] if cb_ap.history_ap else None
            print(f"EPOCH {epoch+1}/{a.max_epochs} best_ap={cb_ap.best_ap:.5f} @ep{cb_ap.best_epoch}"
                  f" H_L={H_score:.4f}", flush=True)

    cb_ent = EntropyCb()

    class WeightSaverCb(tf.keras.callbacks.Callback):
        """Save per-epoch weights for offline curve verification when requested."""
        def __init__(self, wdir):
            super().__init__()
            self.wdir = wdir

        def on_epoch_end(self, epoch, logs=None):
            save_weights_npz(self.model.get_weights(), self.wdir / f"weights_epoch_{epoch + 1:03d}.npz")

    callbacks = [cb_ap, cb_ent]
    if weights_dir is not None:
        callbacks.append(WeightSaverCb(weights_dir))

    train_ds, steps = _make_contaminated_train_ds(train_mm, A, int(a.batch), int(a.seed), N_FEATURES)
    hist = model.fit(train_ds, epochs=int(a.max_epochs), steps_per_epoch=steps,
                     callbacks=callbacks, verbose=0)

    history_ap_full = [ap0] + [float(x) if x is not None else None for x in cb_ap.history_ap]
    history_entropy_full = [H_dae0] + [float(x) for x in cb_ent.history_entropy]
    history_entropy_score_full = [H_score0] + [float(x) for x in cb_ent.history_entropy_score]
    history_S_full = [S_dae0] + [float(x) for x in cb_ent.history_S]
    n_deval = len(deval_idx)
    history_val_loss_full = [float(s) / n_deval for s in history_S_full]

    valid = [(i, v) for i, v in enumerate(history_ap_full) if v is not None and np.isfinite(v)]
    best_idx, best_ap_val = max(valid, key=lambda p: p[1])

    cfg_used = dict(CFG)
    cfg_used["lr"] = float(a.lr)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": int(a.seed),
        "rate": float(a.rate),
        "lr": float(a.lr),
        "n_attacks_contam": int(n_attacks),
        "n_benign": int(n_benign),
        "split_seed": int(a.split_seed),
        "eval_seed": int(a.eval_seed),
        "n_eval": int(a.n_eval),
        "n_eval_out": int(len(out_idx)),
        "prevalence_oracle": prevalence,
        "history_ap": history_ap_full,
        "history_entropy": history_entropy_full,
        "history_entropy_score": history_entropy_score_full,
        "history_S": history_S_full,
        "history_val_loss": history_val_loss_full,
        "history_train_loss": [float(x) for x in hist.history.get("loss", [])],
        "best_epoch": int(best_idx), "best_ap": float(best_ap_val),
        "n_epochs": len(cb_ap.history_ap),
        "epochs_cap": int(a.max_epochs),
        "att_split_sizes": {"pool": len(split["contam_pool"]), "oracle": len(split["oracle_att"]),
                            "eval_out": len(split["eval_out"])},
        "cfg": cfg_used,
        "sec_total": round(time.time() - t0, 1),
    }
    out.write_text(json.dumps(payload))
    print(f"seed {a.seed} rate {a.rate} lr {a.lr}: {len(cb_ap.history_ap)} epoche (+e0) in "
         f"{payload['sec_total']:.0f}s, best_ap={best_ap_val:.5f} @idx {best_idx} -> {out}")

    # cleanup immediato (strategia di contenimento della memoria)
    del model, train_ds, hist, cb_ap, cb_ent
    del val_X_oracle, A, train_mm, val_X, val_y, val_y_oracle
    del oracle_idx_sorted, deval_idx, split
    gc.collect()
    tf.keras.backend.clear_session()


# ==================================================================== driver analisi offline
def load_runs(art_dir: Path, rate: str, lr: str) -> list[dict]:
    """Load runs using exact directory strings, avoiding collisions from float reformatting."""
    pattern = str(art_dir / f"rate_{rate}" / f"lr_{lr}" / "seed_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"nessun JSON trovato con pattern {pattern}")
    return [json.load(open(f)) for f in files]


def discover_groups(art_dir: Path) -> list[tuple[str, str]]:
    """Find rate/lr directory pairs and sort them by numeric value."""
    groups = set()
    for p in art_dir.glob("rate_*/lr_*/seed_*.json"):
        rate = p.parent.parent.name.replace("rate_", "")
        lr = p.parent.name.replace("lr_", "")
        groups.add((rate, lr))
    return sorted(groups, key=lambda rl: (float(rl[0]), float(rl[1]), rl[0], rl[1]))


def discover_grid_cells(art_dir: Path, exclude_lr_dirs: tuple = ()) -> dict:
    """Group run paths by exact directory names, optionally excluding learning-rate directories."""
    cells: dict = {}
    for p in sorted(art_dir.glob("rate_*/lr_*/seed_*.json")):
        lr_dir = p.parent.name
        if lr_dir in exclude_lr_dirs:
            continue
        rate_dir = p.parent.parent.name
        cells.setdefault((rate_dir, lr_dir), []).append(str(p))
    return cells


def print_group_report(label: str, result: dict, n_seeds: int) -> None:
    print(f"\n=== {label} (n={n_seeds} semi) ===")
    for entropy_key, res in result.items():
        variant_label = "dae_loss (primaria)" if entropy_key == "history_entropy" else "mse_binary_score (secondaria)"
        print(f"  --- {variant_label} ---")
        print(f"  Pearson per seme: {[round(p, 3) for p in res['pearson_per_seed']]}  "
             f"mediano: {res['pearson_median']:+.3f}  (<-0.5: {res['n_seeds_pearson_lt_neg05']}/{n_seeds})")
        print(f"  TV/range per seme: {[round(t, 2) for t in res['tv_over_range_per_seed']]}  "
             f"mediano: {res['tv_over_range_median']:.2f}")
        bc = res["best_config"]
        if bc is None:
            print(f"  Griglia K_GRID non valutabile: curva piu' corta del piu' piccolo k ({K_GRID[0]})")
        else:
            print(f"  Migliore k={bc['k']}, R_down={bc['r_down']}: Delta mediano = "
                 f"{res['best_config_median_delta']:+.5f}")
        print(f"  Cap fisso idx {FIXED_EPOCH} (ricomputato su questo gruppo): Delta mediano = "
             f"{res['cap_fixed_median_delta']:+.5f}")


def print_pilot_verdict(verdict: dict) -> None:
    print(f"\n=== VERDETTO PILOTA: {verdict['verdict']} ===")
    print(f"  motivazione: {verdict['reason']}")
    if verdict["winner_lr"] is not None:
        print(f"  lr vincente: {verdict['winner_lr']}")


def cmd_analyze(a) -> None:
    art_dir = Path(a.dir)

    if a.pilot:
        if a.rate is None:
            raise SystemExit("--pilot richiede --rate (i 2 gruppi lr allo stesso tasso)")
        lrs = sorted({lr for r, lr in discover_groups(art_dir) if r == a.rate}, key=float)
        if len(lrs) != 2:
            raise SystemExit(f"--pilot richiede esattamente 2 gruppi lr a rate={a.rate}, "
                            f"trovati {len(lrs)}: {lrs}")
        groups = {}
        for lr in lrs:
            runs = load_runs(art_dir, a.rate, lr)
            groups[lr] = analyze_group(runs)
            print_group_report(f"rate={a.rate} lr={lr}", groups[lr], len(runs))
        print_pilot_verdict(decide_pilot(groups))
        return

    if a.rate is not None and a.lr is not None:
        runs = load_runs(art_dir, a.rate, a.lr)
        result = analyze_group(runs)
        print_group_report(f"rate={a.rate} lr={a.lr}", result, len(runs))
        return

    groups_found = discover_groups(art_dir)
    if a.rate is not None:
        groups_found = [(r, lr) for r, lr in groups_found if r == a.rate]
    if a.lr is not None:
        groups_found = [(r, lr) for r, lr in groups_found if lr == a.lr]
    if not groups_found:
        raise SystemExit(f"nessun gruppo trovato sotto {art_dir} (rate={a.rate}, lr={a.lr})")

    print(f"=== {len(groups_found)} gruppi (rate,lr) trovati ===")
    reliable_rates = []
    for rate, lr in groups_found:
        runs = load_runs(art_dir, rate, lr)
        result = analyze_group(runs)
        print_group_report(f"rate={rate} lr={lr}", result, len(runs))
        for res in result.values():
            if res["n_seeds_pearson_lt_neg05"] >= max(1, int(0.75 * res["n_seeds"])):
                reliable_rates.append(rate)
                break
    if reliable_rates:
        print(f"\n=== tasso minimo affidabile (Pearson<-0.5 su >=3/4 semi): {min(reliable_rates, key=float)} ===")
    else:
        print("\n=== nessun tasso raggiunge l'affidabilita' (Pearson<-0.5 su >=3/4 semi) ===")


# ==================================================================== driver verifica pesi<->curve (C3)
def pick_verification_target(runs: list[dict], seed: int,
                             entropy_key: str = "history_entropy_score") -> tuple[int, float, dict]:
    """Get the selected stop epoch and recorded AP for one seed."""
    run = next((r for r in runs if r["seed"] == seed), None)
    if run is None:
        raise ValueError(f"seed {seed} non trovato nel gruppo (semi disponibili: "
                         f"{[r['seed'] for r in runs]})")
    bc = analyze_group(runs)[entropy_key]["best_config"]
    k = bc["k"] if bc is not None else 3
    r_down = bc["r_down"] if bc is not None else 0.05
    stop_ep = entropystop_select(run[entropy_key], k, r_down)
    recorded_ap = run["history_ap"][stop_ep]
    return stop_ep, recorded_ap, {"k": k, "r_down": r_down}


def cmd_verify(a) -> None:
    """Recompute the selected epoch's oracle AP from saved weights and compare with history."""
    setup_blas_env(a.threads, deterministic=True)
    import tensorflow as tf

    from lib.dae import constants as C
    from lib.dae.evaluate import compute_scores_chunked
    from lib.dae.model import build_dae
    from lib.dae.objective import seed_ap
    from lib.utils import N_BINARY, N_CONTINUOUS, N_FEATURES

    art_dir = Path(a.dir)
    runs = load_runs(art_dir, a.rate, a.lr)
    stop_ep, recorded_ap, bc = pick_verification_target(runs, a.seed)

    d = Path(a.data_dir)
    val_X = np.load(d / "val_X.npy", mmap_mode="r")
    val_y = np.asarray(np.load(d / "val_y.npy")).astype(np.int8)
    att_pos = np.where(val_y == 1)[0]
    benign_val_pos = np.where(val_y == 0)[0]
    run = next(r for r in runs if r["seed"] == a.seed)
    split = split_attack_positions(att_pos, N_POOL_MAX, N_EVAL_OUT, run["split_seed"])
    oracle_idx_sorted = np.sort(np.concatenate([benign_val_pos, split["oracle_att"]]))
    val_X_oracle = _MmapView(val_X, oracle_idx_sorted)
    val_y_oracle = val_y[oracle_idx_sorted]

    cont_idx = list(range(N_CONTINUOUS))
    bin_idx = list(range(N_CONTINUOUS, N_FEATURES))
    model, _ = build_dae(n_features=N_FEATURES, btl=CFG["btl"], exp=CFG["exp"],
                         n_continuous=N_CONTINUOUS, n_binary=N_BINARY, continuous_idx=cont_idx,
                         binary_idx=bin_idx, l2_reg=C.L2_FIXED, noise_std=CFG["sigma"],
                         noise_type=C.NOISE_TYPE, loss_continuous=C.LOSS_CONTINUOUS, mid=CFG["mid"])

    wdir = Path(a.weights_dir) if a.weights_dir else (
        art_dir / f"rate_{a.rate}" / f"lr_{a.lr}" / f"weights_seed_{a.seed}")
    weights = load_weights_npz(wdir / f"weights_epoch_{stop_ep:03d}.npz")
    model.set_weights(weights)

    @tf.function(reduce_retracing=True,
                input_signature=[tf.TensorSpec(shape=(None, N_FEATURES), dtype=tf.float32)])
    def predict_fn(x):
        return model(x, training=False)

    recomputed_ap = float(seed_ap(val_y_oracle, compute_scores_chunked(predict_fn, val_X_oracle)))
    delta = recomputed_ap - recorded_ap
    result = {
        "rate": a.rate, "lr": a.lr, "seed": a.seed, "stop_epoch": stop_ep,
        "k": bc["k"], "r_down": bc["r_down"],
        "recorded_ap": recorded_ap, "recomputed_ap": recomputed_ap, "delta": delta,
    }
    print(json.dumps(result))
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2))

    del model, val_X, val_y, val_X_oracle, val_y_oracle, split
    gc.collect()
    tf.keras.backend.clear_session()


# ==================================================================== driver plotting (opzionale)
def cmd_plot(a) -> None:
    """Plot paired AP and entropy curves with the EntropyStop epoch marked."""
    from lib.plotting.base import DPI_FINAL, savefig_close
    import matplotlib.pyplot as plt

    art_dir = Path(a.dir)
    runs = sorted(load_runs(art_dir, a.rate, a.lr), key=lambda r: r["seed"])
    entropy_key = "history_entropy_score" if a.variant == "score" else "history_entropy"
    variant_label = "mse_binary_score (secondaria)" if a.variant == "score" else "dae_loss (primaria)"

    bc = analyze_group(runs)[entropy_key]["best_config"]
    if bc is None and (a.k is None or a.r_down is None):
        print(f"ATTENZIONE: griglia K_GRID non valutabile su questo gruppo (curva piu' corta "
             f"del piu' piccolo k={K_GRID[0]}) — uso k=3, R_down=0.05 di fallback per il marcatore")
    k = a.k if a.k is not None else (bc["k"] if bc is not None else 3)
    r_down = a.r_down if a.r_down is not None else (bc["r_down"] if bc is not None else 0.05)

    n = len(runs)
    ncols = min(n, 2)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.8 * nrows), squeeze=False)

    for i, run in enumerate(runs):
        ax = axes[i // ncols][i % ncols]
        ap = run["history_ap"]
        entropy = run[entropy_key]
        x = list(range(len(ap)))

        l1, = ax.plot(x, ap, color="#1f77b4", label="AUC-PR")
        ax.set_xlabel("indice (0=e0 pre-training, i>=1=i epoche)")
        ax.set_ylabel("AUC-PR", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax.set_ylim(*AP_YLIM)

        ax2 = ax.twinx()
        l2, = ax2.plot(x, entropy, color="#ff7f0e", label="H_L")
        ax2.set_ylabel(f"H_L ({variant_label})", color="#ff7f0e")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")
        ax2.set_ylim(*H_L_YLIM)

        stop_ep = entropystop_select(entropy, k, r_down)
        ap_stop, ent_stop = ap[stop_ep], entropy[stop_ep]
        ax.axvline(stop_ep, color="#ff7f0e", linestyle=":", alpha=0.8)
        ax.plot([stop_ep], [ap_stop], marker="o", ms=5, color="#1f77b4", zorder=5)
        ax2.plot([stop_ep], [ent_stop], marker="o", ms=5, color="#ff7f0e", zorder=5)
        ax.annotate(f"AUC-PR={ap_stop:.3f}", (stop_ep, ap_stop), textcoords="offset points",
                   xytext=(6, 6), fontsize=7, color="#1f77b4")
        ax2.annotate(f"H_L={ent_stop:.3f}", (stop_ep, ent_stop), textcoords="offset points",
                    xytext=(6, -12), fontsize=7, color="#ff7f0e")
        ax.set_title(f"seed={run['seed']}  lr={run.get('lr', '?')}  "
                    f"stop_epoch={stop_ep} (k={k},R_down={r_down})",
                    fontsize=9)
        ax.legend(handles=[l1, l2], loc="upper center", bbox_to_anchor=(0.5, -0.18),
                 ncol=2, fontsize=8, frameon=False)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"AUC-PR / H_L — rate={a.rate} lr={a.lr} ({variant_label})")
    fig.tight_layout()
    out = Path(a.out) if a.out else art_dir / "figs" / f"curves_rate{a.rate}_lr{a.lr}_{a.variant}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    savefig_close(fig, out, dpi=DPI_FINAL)
    print(f"figura salvata: {out} ({n} semi, k={k}, R_down={r_down})")


def cmd_grid(a) -> None:
    """Plot multi-seed AP and entropy means with ±1σ bands and mean stop epoch."""
    from lib.plotting.base import DPI_FINAL, savefig_close
    import matplotlib.pyplot as plt

    art_dir = Path(a.dir)
    exclude = tuple(a.exclude_lr) if a.exclude_lr else ()
    cells = discover_grid_cells(art_dir, exclude_lr_dirs=exclude)
    if not cells:
        raise SystemExit(f"nessuna cella trovata sotto {art_dir} (exclude={exclude})")

    rate_dirs = sorted({rd for rd, _ in cells}, key=lambda s: float(s.replace("rate_", "")))
    lr_dirs = sorted({ld for _, ld in cells}, key=lambda s: float(s.replace("lr_", "")))
    nrows, ncols = len(rate_dirs), len(lr_dirs)

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.2 * nrows),
                             squeeze=False, sharex=True)
    for i, rd in enumerate(rate_dirs):
        for j, ld in enumerate(lr_dirs):
            ax = axes[i][j]
            paths = cells.get((rd, ld))
            if not paths:
                ax.axis("off")
                continue
            runs = [json.load(open(p)) for p in paths]
            bc = analyze_group(runs)["history_entropy_score"]["best_config"]
            ap_curves = [r["history_ap"] for r in runs]
            ent_curves = [r["history_entropy_score"] for r in runs]

            x_ap, ap_mean, ap_std = mean_std_padded(ap_curves)
            x_ent, ent_mean, ent_std = mean_std_padded(ent_curves)

            ax2 = ax.twinx()
            ax.plot(x_ap, ap_mean, color="#2ca02c", lw=1.0)
            ax.fill_between(x_ap, ap_mean - ap_std, ap_mean + ap_std,
                           color="#2ca02c", alpha=0.2, lw=0)
            ax2.plot(x_ent, ent_mean, color="#1f77b4", lw=1.0)
            ax2.fill_between(x_ent, ent_mean - ent_std, ent_mean + ent_std,
                            color="#1f77b4", alpha=0.2, lw=0)
            if bc is not None:
                stops = [entropystop_select(c, bc["k"], bc["r_down"]) for c in ent_curves]
                ax.axvline(float(np.mean(stops)), color="#ff7f0e", ls="--", lw=0.9, alpha=0.8)

            ax.set_ylim(*AP_YLIM)
            ax2.set_ylim(*H_L_YLIM)
            ax2.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(ld.replace("lr_", "lr="), fontsize=8)
            if j == 0:
                ax.set_ylabel(rd.replace("rate_", "r="), fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle("Griglia multiseed — media±1σ AUC-PR reale, scala 0-1 (verde, asse sx) vs H_L "
                 "reale, scala 0-13 (blu, asse dx), stop ES medio (tratteggio)", fontsize=10)
    fig.tight_layout()
    out = Path(a.out) if a.out else art_dir / "figs" / "multiseed_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    savefig_close(fig, out, dpi=DPI_FINAL)
    print(f"griglia salvata: {out} ({nrows} tassi x {ncols} lr; esclusi {exclude})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="riaddestra il DAE su training contaminato")
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--rate", type=float, required=True, help="tasso di contaminazione r")
    p_train.add_argument("--lr", type=float, default=CFG["lr"], help="learning rate Adam")
    p_train.add_argument("--data-dir", required=True)
    p_train.add_argument("--out", required=True)
    p_train.add_argument("--threads", type=int, default=1)
    p_train.add_argument("--max-epochs", type=int, default=60)
    p_train.add_argument("--batch", type=int, default=512)
    p_train.add_argument("--n-eval", type=int, default=N_EVAL_DEFAULT)
    p_train.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    p_train.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    p_train.add_argument("--weights-dir", default=None, dest="weights_dir",
                         help="se presente, salva i pesi a ogni epoca in DIR/weights_epoch_NNN.npz (e0=000)")

    p_analyze = sub.add_parser("analyze", help="analisi offline (griglia EntropyStop, nessun TF)")
    p_analyze.add_argument("--dir", default="runs/dae/entropy_contam")
    p_analyze.add_argument("--rate", type=str, default=None,
                          help="stringa esatta di directory, es. 0.005 (mai un float riformattato)")
    p_analyze.add_argument("--lr", type=str, default=None,
                          help="stringa esatta di directory, es. 1e-3 (mai un float riformattato)")
    p_analyze.add_argument("--pilot", action="store_true",
                          help="applica la regola di decisione del pilota (2 gruppi lr, stesso rate)")

    p_plot = sub.add_parser("plot", help="curve AUC-PR/H_L appaiate a doppio asse, un gruppo (rate,lr)")
    p_plot.add_argument("--dir", default="runs/dae/entropy_contam")
    p_plot.add_argument("--rate", type=str, required=True,
                        help="stringa esatta di directory, es. 0.005 (mai un float riformattato)")
    p_plot.add_argument("--lr", type=str, required=True,
                        help="stringa esatta di directory, es. 1e-3 (mai un float riformattato)")
    p_plot.add_argument("--variant", choices=["dae", "score"], default="dae")
    p_plot.add_argument("--k", type=int, default=None, help="default: best_config del gruppo")
    p_plot.add_argument("--r-down", type=float, default=None, dest="r_down")
    p_plot.add_argument("--out", default=None)

    p_grid = sub.add_parser("grid", help="griglia multiseed rate x lr (ricostruisce multiseed_grid.png)")
    p_grid.add_argument("--dir", default="runs/dae/entropy_contam")
    p_grid.add_argument("--out", default=None)
    p_grid.add_argument("--exclude-lr", nargs="*", dest="exclude_lr",
                        default=["lr_0.002071", "lr_0.0002"],
                        help="dir lr da escludere (default: pilota workstation, non bit-exact con le VPS)")

    p_verify = sub.add_parser("verify", help="C3: verifica end-to-end pesi<->curve (DEVE girare su VPS)")
    p_verify.add_argument("--dir", default="runs/dae/entropy_contam")
    p_verify.add_argument("--rate", type=str, required=True)
    p_verify.add_argument("--lr", type=str, required=True)
    p_verify.add_argument("--seed", type=int, required=True)
    p_verify.add_argument("--data-dir", required=True)
    p_verify.add_argument("--weights-dir", default=None, dest="weights_dir",
                          help="default: DIR/rate_{rate}/lr_{lr}/weights_seed_{seed}")
    p_verify.add_argument("--threads", type=int, default=1)
    p_verify.add_argument("--out", default=None)

    a = ap.parse_args()
    if a.cmd == "train":
        cmd_train(a)
    elif a.cmd == "analyze":
        cmd_analyze(a)
    elif a.cmd == "grid":
        cmd_grid(a)
    elif a.cmd == "verify":
        cmd_verify(a)
    else:
        cmd_plot(a)


if __name__ == "__main__":
    main()
