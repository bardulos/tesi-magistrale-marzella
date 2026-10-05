#!/usr/bin/env python3
"""train.py — retrain label-free del NIDS a due stadi sul traffico locale.

Nel train_venv esegue sette passi: partizione in pool gradient/monitor/select; DAE con EntropyStop;
whitening e standardizzazione; generazione pseudo-monitor; training PA con arresto BCE;
FP-mining; calibrazione della fusione sui benigni-select.

Ogni invocazione allena un seme. main.py orchestra i semi e seleziona quello con FPR monitor minimo
in modalita' A (B come spareggio). La regola di fusione e' condivisa con infer.py.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import infer  # noqa: E402  (stesso pacchetto: fusione + encoding condivisi, mai divergenti)

# ============================================================================
# COSTANTI canoniche
# ============================================================================

# Architetture congelate (identiche al forward del pacchetto)
DAE_EXP, DAE_MID, DAE_BTL = 112, 80, 16
DAE_SIGMA, DAE_L2, DAE_LR = 0.15834, 1e-6, 0.002071
PA_EXP, PA_N_LAYERS, PA_RATIO = 3.928824460722608, 2, 0.6159888054811458
PA_DROPOUT, PA_L2, PA_LR = 0.27477804731454736, 1e-6, 0.0008975201332140938

# TRAIN addestra il PA; MONITOR fornisce il dataset per l'arresto BCE.
PA_GEN_TRAIN = {"delta": 0.09261080547245101, "alpha_lo": 1.9481223527402847, "alpha_hi": 3.079948793054766}
PA_GEN_MONITOR = {"delta": 0.19786238154311037, "alpha_lo": 1.3269568638840954, "alpha_hi": 3.21621}

# Arresto DAE: EntropyStop sulla contaminazione naturale, patience k=5, downtrend r_down=0.01.
ES_K, ES_R_DOWN = 5, 0.01
ES_N_EVAL = 262_144           # cardinalita' di D_eval (campione per H_L)

# Arresto PA: BCE-monitor in modalita' min, patience 15 e massimo 100 epoche; ripristina la BCE minima.
PA_PATIENCE, PA_CAP = 15, 100

# Partizione in tre pool: frazioni limitate a [min, max]. I massimi corrispondono ai valori CSE
# (200k/355k su 1,755M); i min garantiscono percentili stabili fino al p99,8 del select.
MONITOR_FRAC, MONITOR_CLAMP = 0.11, (30_000, 200_000)
SELECT_FRAC, SELECT_CLAMP = 0.20, (50_000, 355_000)

# Soglie per la taglia del dataset: avviso sotto 500k righe, blocco sotto 100k (forzabile con --force).
SIZE_WARN, SIZE_BLOCK, SIZE_REF = 500_000, 100_000, 1_755_000
INGEST_CHUNK = 100_000        # righe/blocco dell'ingest CSV a chunk (contiene il picco RAM del parsing)
FWD_CHUNK = 100_000           # Righe per blocco: limita l'arena TensorFlow.

# Hardening FP-mining.
FPM_K, FPM_K_MIN, FPM_N_MAX = 4, 0.002, 5

BIN_SLICE = slice(34, 91)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Funzioni pure portate inline (EntropyStop, pseudo, BCE, whitening, fp-mining)
# ============================================================================


def compute_loss_entropy(v):
    """Loss Entropy H_L (Huang 2024): u=v/sum(v), H_L=-sum(u log u). v = loss per-campione >=0."""
    v = np.asarray(v, np.float64)
    S = float(v.sum())
    if S <= 0.0 or len(v) == 0:
        return 0.0
    u = v / S
    m = u > 0.0
    return float(-(u[m] * np.log(u[m])).sum())


def entropystop_select(entropy_hist, k=ES_K, r_down=ES_R_DOWN):
    """EntropyStop Algorithm 1: patience k + downtrend smooth r_down. Ritorna l'epoca di stop."""
    if not entropy_hist:
        return 0
    e_min, best_ep, G, patience = float(entropy_hist[0]), 0, 0.0, 0
    for j in range(1, len(entropy_hist)):
        G += abs(float(entropy_hist[j]) - float(entropy_hist[j - 1]))
        if float(entropy_hist[j]) < e_min and G > 0 and (e_min - entropy_hist[j]) / G > r_down:
            e_min, best_ep, G, patience = float(entropy_hist[j]), j, 0.0, 0
        else:
            patience += 1
            if patience >= k:
                return best_ep
    return best_ep


def mse_binary_score(y_pred, x_true):
    d = y_pred[:, BIN_SLICE] - x_true[:, BIN_SLICE]
    return np.mean(d * d, axis=1).astype(np.float32)


def bce(y, p, eps=1e-7):
    p = np.clip(np.asarray(p, np.float64), eps, 1.0 - eps)
    y = np.asarray(y, np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def generate_fakes_typed(Zs, R, rng, delta, alpha_lo, alpha_hi):
    """Genera una pseudo per riga: residuo radiale o residuo scambiato da un'altra riga."""
    n = len(Zs)
    out = np.empty((n, Zs.shape[1] + R.shape[1]), dtype=np.float32)
    is_radial = rng.random(n) < delta
    for i in range(n):
        if is_radial[i]:
            a = float(rng.uniform(alpha_lo, alpha_hi))
            out[i] = np.concatenate([Zs[i], a * R[i]])
        else:
            j = int(rng.integers(n))
            while j == i:
                j = int(rng.integers(n))
            out[i] = np.concatenate([Zs[i], R[j]])
    return out


def derive_whitening(R_benign):
    """Ledoit-Wolf sui residui benigni -> (mu_R, L, sigma_R). L = Cholesky di Sigma^-1 (whitening);
    PD-fallback per binarie quasi-costanti. sigma_R = sqrt(diag(Sigma)) per gli alert."""
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(np.asarray(R_benign, np.float64))
    mu = lw.location_.astype(np.float64)
    try:
        L = np.linalg.cholesky(lw.precision_).astype(np.float64)
    except np.linalg.LinAlgError:
        C = np.linalg.cholesky(lw.covariance_)
        L = np.linalg.inv(C).T.astype(np.float64)
    sigma_R = np.sqrt(np.diag(lw.covariance_)).astype(np.float64)
    return mu, L, sigma_R


def should_stop_mining(curr, prev, k_min=FPM_K_MIN):
    return (curr - prev) < k_min


# ============================================================================
# Modelli TF (architetture #569 / #589)
# ============================================================================


def build_dae_tf():
    from tensorflow.keras import Model, regularizers
    from tensorflow.keras.layers import Concatenate, Dense, GaussianNoise, Input
    reg = regularizers.l2(DAE_L2)
    inp = Input(shape=(91,), name="input")
    x = GaussianNoise(DAE_SIGMA, name="corruption")(inp)
    x = Dense(DAE_EXP, activation="relu", kernel_regularizer=reg, name="enc_exp")(x)
    x = Dense(DAE_MID, activation="relu", kernel_regularizer=reg, name="enc_mid")(x)
    btl = Dense(DAE_BTL, activation="relu", kernel_regularizer=reg, name="bottleneck")(x)
    x = Dense(DAE_MID, activation="relu", kernel_regularizer=reg, name="dec_mid")(btl)
    x = Dense(DAE_EXP, activation="relu", kernel_regularizer=reg, name="dec_exp")(x)
    cont = Dense(34, activation="linear", name="out_continuous")(x)
    binr = Dense(57, activation="sigmoid", name="out_binary")(x)
    out = Concatenate(name="output")([cont, binr])
    return Model(inp, out, name="dae"), Model(inp, btl, name="encoder")


def dae_loss_tf():
    import tensorflow as tf
    eps = 1e-7

    def loss(y_true, y_pred):
        yc, pc = y_true[:, :34], y_pred[:, :34]
        yb, pb = y_true[:, 34:], tf.clip_by_value(y_pred[:, 34:], eps, 1 - eps)
        cont = tf.reduce_sum(tf.square(yc - pc), axis=-1) / 34.0
        binr = tf.reduce_sum(-(yb * tf.math.log(pb) + (1 - yb) * tf.math.log(1 - pb)), axis=-1) / 57.0
        return cont + binr
    return loss


def build_pa_tf():
    from tensorflow.keras import Sequential, layers, regularizers
    h1 = int(round(PA_EXP * 107))
    widths = [max(1, int(h1 * PA_RATIO ** k)) for k in range(PA_N_LAYERS)]
    m = Sequential(name="pa")
    m.add(layers.Input(shape=(107,)))
    for k, w in enumerate(widths):
        m.add(layers.Dense(w, activation="relu", kernel_regularizer=regularizers.l2(PA_L2), name=f"dense_{k}"))
        if PA_DROPOUT > 0:
            m.add(layers.Dropout(PA_DROPOUT, name=f"drop_{k}"))
    m.add(layers.Dense(1, activation="sigmoid", name="out"))
    return m


# ============================================================================
# Catena a 7 passi
# ============================================================================


def partition_pools(n, rng):
    """Partiziona in pool disgiunti; sui dataset piccoli riduce i pool per proteggere il gradiente."""
    n_mon = int(np.clip(int(MONITOR_FRAC * n), *MONITOR_CLAMP))
    n_sel = int(np.clip(int(SELECT_FRAC * n), *SELECT_CLAMP))
    n_mon = min(n_mon, n)
    n_sel = min(n_sel, n - n_mon)
    min_grad = max(1000, int(0.10 * n))
    if n - n_mon - n_sel < min_grad:
        deficit = min_grad - (n - n_mon - n_sel)
        take = min(deficit, max(0, n_sel - 1))
        n_sel -= take
        deficit -= take
        if deficit > 0:
            n_mon = max(1, n_mon - deficit)
        _log(f"WARNING (N4) — gradiente degenere coi clamp minimi: monitor/select ridotti per "
             f"garantire gradiente >= {min_grad} (dataset molto piccolo, modello di bassa qualita')")
    perm = rng.permutation(n)
    idx_sel = perm[:n_sel]
    idx_mon = perm[n_sel:n_sel + n_mon]
    idx_grad = perm[n_sel + n_mon:]
    _log(f"passo 1 — pool: gradiente={len(idx_grad)} ({len(idx_grad)/n*100:.1f}%) "
         f"monitor={len(idx_mon)} ({len(idx_mon)/n*100:.1f}%) select={len(idx_sel)} "
         f"({len(idx_sel)/n*100:.1f}%) [frazioni monitor {MONITOR_FRAC:.0%} clamp {MONITOR_CLAMP}, "
         f"select {SELECT_FRAC:.0%} clamp {SELECT_CLAMP}]")
    return idx_grad, idx_mon, idx_sel


def train_dae(X_grad, X_eval, seed, max_epochs, batch=512):
    """Passo 2 — DAE su gradiente + EntropyStop su X_eval (contaminazione naturale), senza ripiego."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(int(seed))
    model, encoder = build_dae_tf()
    model.compile(optimizer=tf.keras.optimizers.Adam(DAE_LR), loss=dae_loss_tf())

    H_hist, weights_hist = [], []

    def measure_H():
        yh = model(X_eval, training=False).numpy()
        return compute_loss_entropy(mse_binary_score(yh, X_eval))

    H_hist.append(measure_H())
    weights_hist.append([w.copy() for w in model.get_weights()])
    # Convertiamo una volta sola e manteniamo fit di un'epoca per preservare RNG e EntropyStop.
    x_grad_t = tf.constant(X_grad)
    for _ in range(max_epochs):
        model.fit(x_grad_t, x_grad_t, epochs=1, batch_size=batch, verbose=0)
        H_hist.append(measure_H())
        weights_hist.append([w.copy() for w in model.get_weights()])

    stop_ep = entropystop_select(H_hist)
    _log(f"passo 2 — DAE fermato a epoca {stop_ep}/{max_epochs} via EntropyStop (k={ES_K}, r_down={ES_R_DOWN})")
    if stop_ep == 0:
        # Nessun ripiego: l'epoca 0 sono i pesi casuali, quindi il seme si ferma qui senza scrivere il modello
        # (main.py lo conta fra i semi falliti). Il messaggio riassume la curva H_L per la diagnosi.
        h = np.asarray(H_hist, float)
        j = int(np.argmin(h))
        raise SystemExit(
            "STOP (passo 2, EntropyStop): selezionata epoca 0 = pesi casuali pre-training. "
            f"Nessun minimo di H_L accettato dopo l'inizio (H_L: epoca 0 {h[0]:.4f}, minimo {h[j]:.4f} "
            f"all'epoca {j}, ultima {h[-1]:.4f}, {len(h) - 1} epoche): il dataset e' troppo piccolo, "
            "gia' ottimale, o contiene solo rumore. Nessun modello scritto per questo seme."
        )
    model.set_weights(weights_hist[stop_ep])
    return model, encoder, {"stop_epoch": stop_ep, "ramo": "EntropyStop", "H_hist": H_hist}


def _forward_chunked(model, X, chunk=FWD_CHUNK):
    """Esegue il forward a blocchi per limitare l'arena TensorFlow.

    Dense e' indipendente fra righe, ma kernel dipendenti da M possono produrre piccole differenze
    numeriche rispetto al forward intero: l'equivalenza e' numerica, non bit a bit.
    """
    n = len(X)
    if n <= chunk:
        return model(X, training=False).numpy()
    return np.concatenate(
        [model(X[s:min(s + chunk, n)], training=False).numpy() for s in range(0, n, chunk)], axis=0)


def sdae_score_chunked(model, X, chunk=FWD_CHUNK):
    """Calcola lo score DAE per blocchi senza materializzare Yhat intero."""
    n = len(X)
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = mse_binary_score(model(X[s:e], training=False).numpy(), X[s:e])
    return out


def compute_phi(model, encoder, X, mu_R, L, mu_z, sigma_z, chunk=FWD_CHUNK):
    """Costruisce phi=[z_s, r_w] a blocchi, senza materializzare Yhat o R per l'intero dataset.

    Riusa il buffer dei residui e lo scarta a ogni blocco. Il risultato coincide col calcolo
    sull'intero array a meno di piccole differenze numeriche.
    """
    n = len(X)
    w_z = mu_z.shape[0]
    phi = np.empty((n, w_z + L.shape[1]), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        Xc = X[s:e]
        z = encoder(Xc, training=False).numpy().astype(np.float64)
        R = model(Xc, training=False).numpy().astype(np.float64)
        R -= Xc
        phi[s:e, :w_z] = ((z - mu_z) / np.maximum(sigma_z, 1e-8)).astype(np.float32)
        del z
        R -= mu_R
        phi[s:e, w_z:] = (R @ L).astype(np.float32)
        del R
    return phi


def gather_permuted(A, B, p, a_index=None, chunk=262_144):
    """Materializza concat(A_eff, B)[p] senza creare l'array concatenato intermedio.
    A_eff = A[a_index] se a_index e' dato, altrimenti A.

    Il chunk limita il temporaneo; l'array di uscita resta unico. Il percorso attivo non la usa
    (train_pa e harden_fp_mining lavorano per batch con _pa_take_batch): resta come riferimento.
    """
    n_a = len(A) if a_index is None else len(a_index)
    out = np.empty((len(p), A.shape[1]), dtype=A.dtype)
    for i in range(0, len(p), chunk):
        j = min(i + chunk, len(p))
        pj = p[i:j]
        m = pj < n_a
        blk = out[i:j]
        blk[m] = A[pj[m] if a_index is None else a_index[pj[m]]]
        blk[~m] = B[pj[~m] - n_a]
    return out


def labels_permuted(p, n_a):
    """Crea le etichette permutate senza materializzare l'array concatenato."""
    return (p >= n_a).astype(np.float64)


def _pa_take_batch(phi_grad, pseudo, bi, n_a, a_index=None):
    """Costruisce un batch PA dagli array residenti, senza creare l'array concatenato (2N,107)."""
    m = bi < n_a
    Xb = np.empty((len(bi), phi_grad.shape[1]), dtype=phi_grad.dtype)
    Xb[m] = phi_grad[bi[m] if a_index is None else a_index[bi[m]]]
    Xb[~m] = pseudo[bi[~m] - n_a]
    return Xb, (bi >= n_a).astype(np.float64)


def train_pa(phi_grad, phi_mon, pseudo_mon, seed, batch=512, cap=PA_CAP):
    """Allena il PA e ritorna i pesi con BCE-monitor minima, anche al cap."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(int(seed) + 7)
    model = build_pa_tf()
    model.compile(optimizer=tf.keras.optimizers.Adam(PA_LR), loss="binary_crossentropy")
    rng = np.random.default_rng(int(seed) + 11)

    X_mon = np.concatenate([phi_mon, pseudo_mon], axis=0)
    y_mon = np.concatenate([np.zeros(len(phi_mon)), np.ones(len(pseudo_mon))])
    Zs_g, R_g = phi_grad[:, :16], phi_grad[:, 16:]

    best_bce, best_w, best_ep, patience = float("inf"), None, 0, 0
    bce_hist = []
    n_a = len(phi_grad)
    for ep in range(cap):
        pseudo_tr = generate_fakes_typed(Zs_g, R_g, rng, **PA_GEN_TRAIN)
        idx = rng.permutation(n_a + len(pseudo_tr))
        # I batch indicizzano gli array residenti: niente array concatenato (2N,107) ne' copia Keras.
        for i in range(0, len(idx), batch):
            Xb, yb = _pa_take_batch(phi_grad, pseudo_tr, idx[i:i + batch], n_a)
            model.train_on_batch(Xb, yb)
        del pseudo_tr
        bce_ep = bce(y_mon, _forward_chunked(model, X_mon).ravel())
        bce_hist.append(bce_ep)
        if bce_ep < best_bce - 1e-6:
            best_bce, best_w, best_ep, patience = bce_ep, [w.copy() for w in model.get_weights()], ep, 0
        else:
            patience += 1
            if patience >= PA_PATIENCE:
                ramo = "patience-15"
                break
    else:
        ramo = "censura (cap, ripristino best-BCE)"
    if best_w is None:
        raise SystemExit(
            "STOP (passo 5, PA): best_w e' None — il loop non ha completato nemmeno un'epoca "
            "(--max-epochs-pa 0, o BCE=NaN alla prima valutazione). "
            "Controlla il dataset e i parametri: --max-epochs-pa deve essere >= 1."
        )
    model.set_weights(best_w)
    _log(f"passo 5 — PA fermato a epoca {best_ep} via {ramo} (BCE-monitor min={best_bce:.5f})")
    return model, {"stop_epoch": best_ep, "ramo": ramo, "bce_min": best_bce}


def harden_fp_mining(model_pa, phi_grad, sel_scores, params_A, seed, n_max=FPM_N_MAX):
    """Mina i FP label-free del sistema fuso in modalita' A sul pool gradiente e riaddestra.

    Si ferma quando la proxy 1 - FP/benigni-gradiente guadagna meno di k_min. La proxy si usa
    in ogni retrain al posto della pAUC_MCC del laboratorio, che richiede etichette.
    """
    rng = np.random.default_rng(int(seed) + 23)
    Zs_g, R_g = phi_grad[:, :16], phi_grad[:, 16:]
    pool = np.array([], dtype=np.int64)
    prev_metric = -1.0
    rounds_run = 0
    stopped_early = False
    for rnd in range(n_max):
        logit_g = infer.fus_logit(_forward_chunked(model_pa, phi_grad).ravel())
        s_dae_g = sel_scores["sdae_grad"]
        fp = np.where(infer.apply_fusion(params_A, s_dae_g, logit_g))[0]
        pool = np.union1d(pool, fp).astype(np.int64)
        metric = 1.0 - len(fp) / max(len(phi_grad), 1)
        if rnd > 0 and should_stop_mining(metric, prev_metric):
            _log(f"passo 6 — hardening auto-stop al round {rnd} (Delta<{FPM_K_MIN})")
            stopped_early = True
            break
        prev_metric = metric
        # augment: FP ripetuti x(K+1) come benigni "difficili"
        aug_idx = np.concatenate([np.arange(len(phi_grad)), np.repeat(pool, FPM_K)]) if len(pool) else np.arange(len(phi_grad))
        pseudo = generate_fakes_typed(Zs_g, R_g, rng, **PA_GEN_TRAIN)
        pmt = rng.permutation(len(aug_idx) + len(pseudo))
        n_a = len(aug_idx)
        # a_index porta i FP ripetuti senza materializzare l'array concatenato.
        for i in range(0, len(pmt), 512):
            Xb, yb = _pa_take_batch(phi_grad, pseudo, pmt[i:i + 512], n_a, a_index=aug_idx)
            model_pa.train_on_batch(Xb, yb)
        del pseudo
        rounds_run += 1
    info = {"rounds_run": rounds_run, "auto_stop": stopped_early, "fp_mined_cumulative": int(len(pool))}
    _log(f"passo 6 — hardening completato ({len(pool)} FP minati cumulativi, {rounds_run} round)")
    return model_pa, info


def _keras_to_dae_npz(model):
    d = {}
    name_map = {"enc_exp": "enc_exp", "enc_mid": "enc_mid", "bottleneck": "btl",
                "dec_mid": "dec_mid", "dec_exp": "dec_exp",
                "out_continuous": "out_cont", "out_binary": "out_bin"}
    for lyr in model.layers:
        if lyr.name in name_map:
            W, b = lyr.get_weights()
            d[f"W_{name_map[lyr.name]}"] = W.astype(np.float32)
            d[f"b_{name_map[lyr.name]}"] = b.astype(np.float32)
    return d


def dae_npz_to_keras(npz):
    """Ricostruisce model ed encoder Keras dai pesi DAE, per --fixed-dae-dir."""
    model, encoder = build_dae_tf()
    inv = {"enc_exp": "enc_exp", "enc_mid": "enc_mid", "bottleneck": "btl", "dec_mid": "dec_mid",
           "dec_exp": "dec_exp", "out_continuous": "out_cont", "out_binary": "out_bin"}
    for lyr in model.layers:
        if lyr.name in inv:
            k = inv[lyr.name]
            lyr.set_weights([npz[f"W_{k}"], npz[f"b_{k}"]])
    return model, encoder


def _keras_to_pa_npz(model, mu_R, L, sigma_R, mu_z, sigma_z):
    d = {"pa_n_layers": np.int64(PA_N_LAYERS),
         "mu_R": mu_R.astype(np.float32), "W": L.astype(np.float32), "sigma_R": sigma_R.astype(np.float32),
         "mu_z": mu_z.astype(np.float32), "sigma_z": sigma_z.astype(np.float32)}
    denses = [lyr for lyr in model.layers if lyr.get_weights() and len(lyr.get_weights()[0].shape) == 2]
    for i, lyr in enumerate(denses):
        W, b = lyr.get_weights()
        key = "pa_W_out" if i == len(denses) - 1 else f"pa_W{i}"
        bkey = "pa_b_out" if i == len(denses) - 1 else f"pa_b{i}"
        d[key], d[bkey] = W.astype(np.float32), b.astype(np.float32)
    return d


def subsample_rows(X, max_rows):
    """Limita le righe con un subsample deterministico rng(0), comune a tutti i semi."""
    n = len(X)
    if not max_rows or n <= max_rows:
        return X
    idx = np.sort(np.random.default_rng(0).choice(n, size=max_rows, replace=False))
    return np.ascontiguousarray(X[idx])


def main(argv=None):
    p = argparse.ArgumentParser(description="Retrain label-free del NIDS a 2 stadi (catena a 7 passi).")
    p.add_argument("--benign-npy", help="feature gia' preprocessate (N,91) float32 [smoke/E2E]")
    p.add_argument("--dataset", help="CSV nProbe grezzo del traffico locale, gia' contaminato al "
                                     "naturale [deploy reale] (encoding via infer). NB: EntropyStop "
                                     "valuta sul prefisso temporale della cattura (~22%%): un fenomeno "
                                     "assente dal prefisso non influenza l'arresto (limite noto)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", required=True, help="cartella modello di uscita (deve esistere)")
    p.add_argument("--weights-src", default=str(Path(__file__).resolve().parent / "modelli" / "produzione" / "cse"),
                   help="sorgente dei metadati di preprocessing (feature_order/scaler/transform_meta), "
                        "fittati sui benigni CSE di laboratorio: il retrain riaddestra i PESI sul "
                        "traffico locale, il preprocessing resta quello ereditato")
    p.add_argument("--max-epochs-dae", type=int, default=60)
    p.add_argument("--max-epochs-pa", type=int, default=PA_CAP, help="cap epoche PA (default 100)")
    p.add_argument("--fpm-rounds", type=int, default=FPM_N_MAX, help="round massimi di FP-mining (default 5)")
    p.add_argument("--force", action="store_true", help="prosegue anche sotto la soglia minima di righe (N4)")
    p.add_argument("--max-rows", type=int, default=0,
                   help="cap righe benigne in input (default 0 = NESSUN cap, usa l'intero traffico "
                        f"catturato). Con un valore >0 (es. SIZE_REF={SIZE_REF:_}) sub-sample deterministico rng(0), "
                        f"fisso fra i semi (replica benign_1755k.npy).")
    p.add_argument("--dae-only", action="store_true",
                   help="diagnostico: allena solo il DAE (passo 2) + whitening (passo 3), salva dae.npz + "
                        "whitening.npz + metadati, ed esce PRIMA del passo 5 (PA). Modello parziale, NON di "
                        "deploy (niente pa.npz/soglie.json). Usato per misurare il DAE-solo a piena scala "
                        "senza l'OOM del PA.")
    p.add_argument("--fixed-dae-dir",
                   help="retrain del solo 2o stadio su un DAE FISSO: carica dae.npz da questa cartella "
                        "invece di allenare il DAE (passo 2). --seed governa il PA; --partition-seed i pool.")
    p.add_argument("--partition-seed", type=int, default=None,
                   help="seed della partizione dei pool (default = --seed). Con --fixed-dae-dir fissarlo "
                        "al seed del DAE per avere phi identico fra i semi PA.")
    args = p.parse_args(argv)

    # Applica la stessa soglia RAM del menu anche ai lanci diretti.
    try:
        with open("/proc/meminfo") as f:
            mem_gb = int(f.readline().split()[1]) / 1024 / 1024
        if mem_gb < 14.5:
            raise SystemExit(f"BLOCCATO: il riaddestramento richiede una macchina da almeno 16 GB "
                             f"di RAM (rilevati {mem_gb:.1f} GiB). Nessuna opzione lo aggira.")
    except OSError:
        pass

    out = Path(args.out_dir)
    if not out.exists():
        raise FileNotFoundError(f"out-dir non esiste (shell-first): {out}")
    src = Path(args.weights_src)
    fo = json.loads((src / "feature_order.json").read_text())["order"]
    scaler = json.loads((src / "scaler_params.json").read_text())
    logf = json.loads((src / "transform_meta.json").read_text())["log_features"]

    # --- ingest del dataset (traffico locale, gia' contaminato al naturale) ---
    if args.benign_npy:
        # Il memmap evita di copiare in RAM le .npy canoniche; la guardia converte dtype inattesi.
        X = np.load(args.benign_npy, mmap_mode="r")
    elif args.dataset:
        import csv
        # Converte e scarta ogni blocco di dict: il picco non include la lista completa del CSV.
        parti = []
        with infer._open_csv(args.dataset) as f:
            blocco = []
            # Usa lo stesso lettore dell'inferenza, incluse le righe nProbe riallineate.
            for row in infer._flussi(f):
                blocco.append(row)
                if len(blocco) >= INGEST_CHUNK:
                    parti.append(infer.csv_to_features(blocco, fo, scaler, logf))
                    blocco = []
            if blocco:
                parti.append(infer.csv_to_features(blocco, fo, scaler, logf))
        X = np.concatenate(parti) if parti else np.zeros((0, len(fo)), dtype=np.float32)
        del parti
    else:
        raise SystemExit("serve --dataset o --benign-npy")
    if X.dtype != np.float32:
        _log(f"WARNING: input {X.dtype} != float32 -> conversione (copia in RAM, "
             f"{X.nbytes // 2**20:,} MB): le .npy canoniche sono float32.")
        X = X.astype(np.float32)
    n0 = len(X)
    X = subsample_rows(X, args.max_rows)
    n = len(X)
    if n < n0:
        _log(f"sub-sample deterministico (rng 0, fisso fra i semi): {n0:,} -> {n:,} righe "
             f"(--max-rows; una PMI opera a scala SIZE_REF, non sul capture intero)")

    # --- Soglie sulla taglia del dataset ---
    _log(f"benigni: {n:,} righe (riferimento canonico CSE {SIZE_REF:,})")
    if n < SIZE_BLOCK and not args.force:
        raise SystemExit(f"STOP (N4): {n:,} < {SIZE_BLOCK:,} righe: dataset troppo piccolo per un "
                         f"retrain affidabile. Usare --force per procedere comunque (sconsigliato).")
    if n < SIZE_WARN:
        _log(f"WARNING (N4): {n:,} < {SIZE_WARN:,} righe — le frazioni scalano ma la qualita' attesa "
             f"degrada. {'PROSEGUO per --force.' if n < SIZE_BLOCK else ''}")

    part_seed = args.partition_seed if args.partition_seed is not None else args.seed
    idx_g, idx_m, idx_s = partition_pools(n, np.random.default_rng(part_seed))
    Xg, Xm, Xs = np.ascontiguousarray(X[np.sort(idx_g)]), np.ascontiguousarray(X[np.sort(idx_m)]), \
        np.ascontiguousarray(X[np.sort(idx_s)])
    del X

    # --- passo 2: DAE + EntropyStop (o DAE FISSO caricato, --fixed-dae-dir: si vara solo il 2o stadio) ---
    if args.fixed_dae_dir:
        fd = Path(args.fixed_dae_dir)
        model, encoder = dae_npz_to_keras(dict(np.load(fd / "dae.npz")))
        _rep = fd / "train_report.json"
        if not _rep.exists():
            _rep = fd / "dae_report.json"
        dae_info = {**json.loads(_rep.read_text()).get("dae", {}), "fixed_from": str(fd)}
        _log(f"passo 2 — DAE FISSO caricato da {fd} (nessun training; ramo {dae_info.get('ramo','?')}, "
             f"ep {dae_info.get('stop_epoch','?')}; partizione seed {part_seed}, PA seed {args.seed})")
    else:
        # H_L usa il prefisso temporale del pool gradiente e la contaminazione naturale del traffico.
        # Un fenomeno assente dal prefisso non influenza l'arresto.
        X_eval = Xg[:min(ES_N_EVAL, len(Xg))]
        _log("passo 2 — DAE + EntropyStop sulla contaminazione naturale del dataset")
        model, encoder, dae_info = train_dae(Xg, X_eval, args.seed, args.max_epochs_dae)

    # --- passo 3: whitening + z-scaler sulla phi del DAE fermato (fit su select) ---
    z_s_dummy = _forward_chunked(encoder, Xs).astype(np.float64)
    Yhat_s = _forward_chunked(model, Xs).astype(np.float64)
    # Riusa Yhat_s per i residui, evitando la copia float64 dell'ingresso e un array separato.
    Yhat_s -= Xs
    R_s = Yhat_s
    mu_R, L, sigma_R = derive_whitening(R_s)
    mu_z, sigma_z = z_s_dummy.mean(0), z_s_dummy.std(0)
    sigma_z[sigma_z < 1e-8] = 1.0
    _log(f"passo 3 — whitening Ledoit-Wolf + z-scaler fittati sulla phi del DAE (select, {len(Xs)} righe)")

    # --- --dae-only: salva DAE + whitening ed esci PRIMA del passo 5 (PA), evitando l'OOM a piena scala ---
    if args.dae_only:
        np.savez(out / "dae.npz", **_keras_to_dae_npz(model))
        np.savez(out / "whitening.npz", mu_R=mu_R, L=L, sigma_R=sigma_R, mu_z=mu_z, sigma_z=sigma_z)
        for fn in ("feature_order.json", "scaler_params.json", "transform_meta.json"):
            (out / fn).write_text((src / fn).read_text())
        (out / "dae_report.json").write_text(json.dumps(
            {"seed": args.seed, "n_benign": n, "dae_only": True,
             "dae": {k: v for k, v in dae_info.items() if k != "H_hist"},
             "pools": {"grad": len(idx_g), "monitor": len(idx_m), "select": len(idx_s)}}, indent=2))
        _log(f"FATTO (--dae-only) — DAE + whitening scritti in {out} (dae.npz, whitening.npz, metadati); "
             f"passo 5 (PA) saltato (modello parziale diagnostico, non di deploy)")
        return

    phi_g = compute_phi(model, encoder, Xg, mu_R, L, mu_z, sigma_z)
    phi_m = compute_phi(model, encoder, Xm, mu_R, L, mu_z, sigma_z)
    phi_s = compute_phi(model, encoder, Xs, mu_R, L, mu_z, sigma_z)

    # Calcola gli score DAE prima di liberare modello, pool e arena TF per contenere il picco del PA.
    # Il DAE resta congelato; il PA mantiene l'equivalenza statistica della configurazione validata.
    import tensorflow as tf
    sdae_g = sdae_score_chunked(model, Xg)
    sdae_m = sdae_score_chunked(model, Xm)
    sdae_s = sdae_score_chunked(model, Xs)
    dae_npz = _keras_to_dae_npz(model)
    del model, encoder, Xg, Xm, Xs
    gc.collect()
    tf.keras.backend.clear_session()

    # --- passo 4: pseudo-monitor (dataset del criterio d'arresto) ---
    rng_m = np.random.default_rng(args.seed + 99)
    pseudo_mon = generate_fakes_typed(phi_m[:, :16], phi_m[:, 16:], rng_m, **PA_GEN_MONITOR)
    _log(f"passo 4 — {len(pseudo_mon)} pseudo-monitor (PA_GEN_MONITOR) dal pool monitor")

    # --- passo 5: PA + BCE-monitor ---
    model_pa, pa_info = train_pa(phi_g, phi_m, pseudo_mon, args.seed, cap=args.max_epochs_pa)

    lg_g = infer.fus_logit(_forward_chunked(model_pa, phi_g).ravel())
    lg_s = infer.fus_logit(_forward_chunked(model_pa, phi_s).ravel())

    # --- passo 6: hardening FP-mining Mod A ---
    tau_dae_A = float(np.percentile(sdae_s, infer.FUSION_MODES["A"]["tau_dae_pct"]))
    tau_hi_A = float(np.percentile(sdae_s, infer.FUSION_MODES["A"]["tau_hi_pct"]))
    params_A0 = infer.build_fusion(sdae_g, lg_g, sdae_s, lg_s, tau_dae_A,
                                   infer.FUSION_MODES["A"]["fpr_target"],
                                   infer.FUSION_MODES["A"]["alpha_and"], tau_hi=tau_hi_A)
    model_pa, harden_info = harden_fp_mining(model_pa, phi_g, {"sdae_grad": sdae_g}, params_A0,
                                             args.seed, n_max=args.fpm_rounds)

    # --- passo 7: calibrazione fusione (ricetta a percentili sui select) -> soglie.json ---
    lg_g = infer.fus_logit(_forward_chunked(model_pa, phi_g).ravel())
    lg_s = infer.fus_logit(_forward_chunked(model_pa, phi_s).ravel())
    modes = infer.build_fusion_modes(sdae_g, lg_g, sdae_s, lg_s)
    soglie = {name: {"mu": [float(x) for x in pr["mu"]], "sd": [float(x) for x in pr["sd"]],
                     "tau_z": pr["tau_z"], "theta_z": pr["theta_z"],
                     "theta_union": (None if not np.isfinite(pr["theta_union"]) else float(pr["theta_union"])),
                     "tau_hi_z": (None if not np.isfinite(pr["tau_hi_z"]) else float(pr["tau_hi_z"])),
                     "ricetta": infer.FUSION_MODES[name]}
              for name, pr in modes.items()}
    _log("passo 7 — fusione calibrata (ricetta a percentili sui benigni-select) -> soglie.json")

    # --- FPR sul pool monitor, separato dalla calibrazione, per entrambe le modalita'. E' il criterio
    #     label-free usato per scegliere il seme: il select realizza
    #     per costruzione l'FPR nominale (cieco), il monitor no. La scelta fra i semi (minimo Mod A,
    #     tie-break Mod B) e' in main.py; qui si misura e si salva per ciascun seme. ---
    lg_m = infer.fus_logit(_forward_chunked(model_pa, phi_m).ravel())
    fpr_monitor = {name: infer.fpr_of(infer.apply_fusion(modes[name], sdae_m, lg_m)) for name in modes}
    _log(f"N2 — FPR realizzato sul monitor (holdout, {len(phi_m)} righe): "
         f"Mod A {fpr_monitor['A'] * 100:.4f}%  Mod B {fpr_monitor['B'] * 100:.4f}%  "
         f"[selezione seme a deploy: minimo FPR Mod A, tie-break Mod B]")

    # --- salvataggio artefatti (formato npz del pacchetto) ---
    np.savez(out / "dae.npz", **dae_npz)
    np.savez(out / "pa.npz", **_keras_to_pa_npz(model_pa, mu_R, L, sigma_R, mu_z, sigma_z))
    (out / "soglie.json").write_text(json.dumps(soglie, indent=2, ensure_ascii=False))
    for fn in ("feature_order.json", "scaler_params.json", "transform_meta.json"):
        (out / fn).write_text((src / fn).read_text())
    report = {"seed": args.seed, "n_benign": n, "dae": {k: v for k, v in dae_info.items() if k != "H_hist"},
              "pa": pa_info, "harden": harden_info,
              "pools": {"grad": len(idx_g), "monitor": len(idx_m), "select": len(idx_s)},
              "fpr_monitor": fpr_monitor}
    (out / "train_report.json").write_text(json.dumps(report, indent=2))
    _log(f"FATTO — modello scritto in {out} (dae.npz, pa.npz, soglie.json, metadati, train_report.json)")


if __name__ == "__main__":
    main()
