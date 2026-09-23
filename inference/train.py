#!/usr/bin/env python3
"""train.py — retrain LABEL-FREE del NIDS a 2 stadi sul traffico locale (deliverable WP-6, cap4).

Gira nel venv di training (train_venv, Python 3.12, TensorFlow + scikit-learn). Implementa la
catena di deploy label-free a 7 passi, ogni passo con artefatti propri e un log timestampato:

  1. partizione del dataset locale in 3 pool disgiunti (gradiente / monitor / select), taglie a
     FRAZIONI con clamp (non valori assoluti: su domini piccoli il grosso resta al training);
  2. DAE #569 sul gradiente, arresto EntropyStop (Huang KDD 2024) sulla contaminazione naturale del
     dataset (nessun contaminante esterno), diagnostica TV/range su H_L: <5 -> EntropyStop, >=5 -> cap-13;
  3. sbiancamento Ledoit-Wolf + standardizzazione del latente sulla phi del DAE appena fermato;
  4. pseudo-monitor (PA_GEN_MONITOR) generate dal pool monitor: dataset del criterio d'arresto;
  5. PA #589 sul gradiente (pseudo PA_GEN_TRAIN), arresto BCE per-epoca sulle pseudo-monitor
     (patience-15, min); se la patience non scatta entro il cap si ripristinano comunque i
     pesi best-BCE (ramo "censura"), mai quelli dell'ultima epoca;
  6. hardening FP-mining label-free sul sistema fuso Mod A (K=4, k_min=0,002, n_max=5);
  7. calibrazione della fusione: soglie a percentili + budget dai benigni-select -> soglie.json.

Singolo seme (--seed): il PARALLELISMO multi-seme e' orchestrazione shell (N invocazioni, mai
multiprocessing Python) — vedi main.py. A deploy il seme di produzione e' il MIGLIORE per
FPR (non il mediano di laboratorio): l'FPR realizzato al punto operativo Mod A e' misurato QUI sul
pool MONITOR (holdout disgiunto dalla calibrazione, esaurito il suo ruolo dopo lo stop del PA) e
salvato in train_report.json["fpr_monitor"]; la selezione fra i semi paralleli (minimo FPR Mod A,
tie-break Mod B) e' in main.py::_select_best_seed. E' una scelta OPERATIVA label-free: il select,
essendo la base della calibrazione, realizza per costruzione l'FPR nominale (confronto cieco); il
monitor no.

La fusione (regola a tre rami) e' importata da infer.py (stessa copia inline: nessuna divergenza).
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

# Due set di pseudo-anomalie nominati (la distinzione e' un contributo della tesi):
#   TRAIN addestra il confine del PA; MONITOR (piu' vicine al benigno, piu' radiali, range piu'
#   ampio) sono il dataset del criterio d'arresto BCE. alpha_hi come costante diretta.
PA_GEN_TRAIN = {"delta": 0.09261080547245101, "alpha_lo": 1.9481223527402847, "alpha_hi": 3.079948793054766}
PA_GEN_MONITOR = {"delta": 0.19786238154311037, "alpha_lo": 1.3269568638840954, "alpha_hi": 3.21621}

# EntropyStop (WP-3a): contaminazione naturale, patience k=5, downtrend r_down=0.01, fallback cap-13.
ES_K, ES_R_DOWN = 5, 0.01
ES_FIXED_EPOCH = 13           # fallback cap-epoche (mediana best_epoch su CSE)
ES_TV_THRESHOLD = 5.0         # TV/range(H_L) < 5 -> EntropyStop affidabile; >= 5 -> fallback
ES_N_EVAL = 262_144           # cardinalita' di D_eval (campione per H_L)

# PA BCE-monitor (WP-4_v2): patience-15 mode=min, cap 100; al cap si ripristina il best-BCE.
PA_PATIENCE, PA_CAP = 15, 100

# Partizione a 3 pool (N1): FRAZIONI con clamp [min, max]. I max sono i valori canonici CSE
# (200k/355k su 1,755M); i min garantiscono percentili stabili fino al p99,8 del select.
MONITOR_FRAC, MONITOR_CLAMP = 0.11, (30_000, 200_000)
SELECT_FRAC, SELECT_CLAMP = 0.20, (50_000, 355_000)

# Guardie sulla taglia del dataset (N4): sotto 500k warning; sotto 100k blocco (superabile --force).
SIZE_WARN, SIZE_BLOCK, SIZE_REF = 500_000, 100_000, 1_755_000
INGEST_CHUNK = 100_000        # righe/blocco dell'ingest CSV a chunk (contiene il picco RAM del parsing)
FWD_CHUNK = 100_000           # righe/blocco dei forward TF batchati (Step B): contiene l'arena TF, che
                              # sul forward intero (2,41M righe) esplodeva a molti GiB (VmHWM 13,6 GiB)

# Hardening FP-mining (N.B. protocollo cap3)
FPM_K, FPM_K_MIN, FPM_N_MAX = 4, 0.002, 5

BIN_SLICE = slice(34, 91)     # 57 feature binarie (score DAE)


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


def tv_over_range(curve):
    c = np.asarray(curve, float)
    if len(c) < 2:
        return 0.0
    rng = float(c.max() - c.min())
    return float(np.abs(np.diff(c)).sum()) / rng if rng > 0 else 0.0


def mse_binary_score(y_pred, x_true):
    d = y_pred[:, BIN_SLICE] - x_true[:, BIN_SLICE]
    return np.mean(d * d, axis=1).astype(np.float32)


def bce(y, p, eps=1e-7):
    p = np.clip(np.asarray(p, np.float64), eps, 1.0 - eps)
    y = np.asarray(y, np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def generate_fakes_typed(Zs, R, rng, delta, alpha_lo, alpha_hi):
    """Una pseudo per riga: radial (prob delta) phi=[Zs[i], alpha*R[i]], alpha~U[lo,hi] (alpha_lo>=1
    -> mai interne al supporto); swap (prob 1-delta) phi=[Zs[i], R[j]], j!=i."""
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
    """Passo 1 — 3 pool disgiunti a frazioni con clamp. Ritorna (idx_grad, idx_mon, idx_sel).

    Guard-rail per n piccolo (deploy sotto-soglia con --force): se i clamp minimi lasciano un
    gradiente degenere, riduce select e poi monitor per garantire al gradiente almeno
    max(1000, 10% del totale), con un WARNING esplicito. Al caso atteso (100k righe) il
    gradiente naturale e' sopra soglia e la correzione non interviene."""
    n_mon = int(np.clip(int(MONITOR_FRAC * n), *MONITOR_CLAMP))
    n_sel = int(np.clip(int(SELECT_FRAC * n), *SELECT_CLAMP))
    n_mon = min(n_mon, n)
    n_sel = min(n_sel, n - n_mon)      # dopo il clamp di n_mon: mai negativo (n<30k con --force)
    # Guard-rail sul gradiente degenere (contratto in docstring).
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
    """Passo 2 — DAE su gradiente + EntropyStop su X_eval (contaminazione naturale), TV/range fallback."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(int(seed))
    model, encoder = build_dae_tf()
    model.compile(optimizer=tf.keras.optimizers.Adam(DAE_LR), loss=dae_loss_tf())

    H_hist, weights_hist = [], []

    def measure_H():
        yh = model(X_eval, training=False).numpy()
        return compute_loss_entropy(mse_binary_score(yh, X_eval))

    H_hist.append(measure_H())                            # e0 (pre-training)
    weights_hist.append([w.copy() for w in model.get_weights()])
    # F5 (mandato memoria 2026-07-16): conversione del pool gradiente in tensore UNA volta, non a
    # ogni epoca (il loop fit(epochs=1) resta invariato: consolidare le epoche cambierebbe lo stream
    # RNG dello shuffle e il percorso EntropyStop). Bit-exact verificato: H_hist, epoca di stop,
    # TV/range e pesi finali identici al percorso numpy per-epoca.
    x_grad_t = tf.constant(X_grad)
    for _ in range(max_epochs):
        model.fit(x_grad_t, x_grad_t, epochs=1, batch_size=batch, verbose=0)
        H_hist.append(measure_H())
        weights_hist.append([w.copy() for w in model.get_weights()])

    tv = tv_over_range(H_hist)
    if tv < ES_TV_THRESHOLD:
        stop_ep, ramo = entropystop_select(H_hist), "EntropyStop"
    else:
        stop_ep, ramo = min(ES_FIXED_EPOCH, len(H_hist) - 1), "fallback cap-13"
    _log(f"passo 2 — DAE fermato a epoca {stop_ep}/{max_epochs} via {ramo} "
         f"(TV/range(H_L)={tv:.2f}, soglia {ES_TV_THRESHOLD})")
    if stop_ep == 0 and ramo == "EntropyStop":
        raise SystemExit(
            "STOP (passo 2, EntropyStop): selezionata epoca 0 = pesi casuali pre-training. "
            "La curva H_L non mostra un minimo successivo all'inizio: il dataset e' troppo piccolo, "
            "gia' ottimale, o contiene solo rumore. Usare --dae-only per diagnosticare la curva H_L."
        )
    model.set_weights(weights_hist[stop_ep])
    return model, encoder, {"stop_epoch": stop_ep, "ramo": ramo, "tv_range": tv, "H_hist": H_hist}


def _forward_chunked(model, X, chunk=FWD_CHUNK):
    """Forward TF a blocchi di FWD_CHUNK righe -> output concatenato (Step B, RAM).

    Il forward Dense e' row-independent (nessun BatchNorm/op batch-dipendente nel DAE/PA), ma su
    questa TF/BLAS multi-thread NON e' bit-identico al forward intero: oneDNN seleziona kernel
    M-dipendenti e TF 2.21 ignora gli env di threading -> Delta~1e-7 fra chunk e whole (verificato).
    EQUIVALENZA NUMERICA (non bit-exact), validata a valle da Gate 1. Contiene
    l'arena TF, che sul forward intero (2,41M righe) saturava la RAM (VmHWM 13,6 GiB)."""
    n = len(X)
    if n <= chunk:
        return model(X, training=False).numpy()
    return np.concatenate(
        [model(X[s:min(s + chunk, n)], training=False).numpy() for s in range(0, n, chunk)], axis=0)


def sdae_score_chunked(model, X, chunk=FWD_CHUNK):
    """Score DAE (MSE f32 sulle 57 binarie) a blocchi: forward+MSE per chunk, mai Yhat intero in RAM.
    Per-riga per costruzione; equivalenza numerica al forward intero (v. _forward_chunked)."""
    n = len(X)
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = mse_binary_score(model(X[s:e], training=False).numpy(), X[s:e])
    return out


def compute_phi(model, encoder, X, mu_R, L, mu_z, sigma_z, chunk=FWD_CHUNK):
    """phi = [z_s, r_w] dal DAE fermato: z_s=(z-mu_z)/sigma_z, r_w=(Yhat-X-mu_R)@L.

    Step B (RAM) — forward e GEMM R@L a BLOCCHI di righe: ne' Yhat/R (2,41M righe f64 = 1,76 GiB a
    piena scala) ne' l'arena TF del forward intero vengono mai materializzati; phi si costruisce in un
    array preallocato, un blocco alla volta. Aritmetica IN-PLACE sul buffer dei residui per blocco
    (F3): Rc=Yhat_c e le sottrazioni non materializzano copie f64 dell'intero blocco.

    NON bit-identico al percorso whole-array su questa TF/BLAS multi-thread: il forward Dense (kernel
    oneDNN M-dipendenti) e la GEMM R@L multi-thread cambiano bit al variare di M (Delta~1e-6 su phi,
    verificato). EQUIVALENZA NUMERICA, validata a valle da Gate 1 (MCC_bal +-0,01, FPR +-0,1pp;
    Delta ~10^4 sotto le tolleranze operative). R viene consumato per blocco e non ritornato.
    """
    n = len(X)
    w_z = mu_z.shape[0]
    phi = np.empty((n, w_z + L.shape[1]), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        Xc = X[s:e]
        z = encoder(Xc, training=False).numpy().astype(np.float64)
        R = model(Xc, training=False).numpy().astype(np.float64)   # R = Yhat_c (stesso buffer)
        R -= Xc                                                    # R = Yhat_c - Xc
        phi[s:e, :w_z] = ((z - mu_z) / np.maximum(sigma_z, 1e-8)).astype(np.float32)
        del z
        R -= mu_R                                                  # R = (Yhat_c - Xc) - mu_R
        phi[s:e, w_z:] = (R @ L).astype(np.float32)
        del R
    return phi


def gather_permuted(A, B, p, a_index=None, chunk=262_144):
    """Materializza concat(A_eff, B)[p] SENZA costruire il concat intermedio, dove A_eff = A[a_index]
    se a_index e' dato, altrimenti A (F4', mandato memoria 2026-07-16).

    KEEP-and-flag: dopo A3 (Step A RAM) il path vivo di train_pa/harden_fp_mining non la usa piu' — lavora
    per batch (_pa_take_batch) senza mai materializzare l'array (2N,107). Resta come oracolo dell'invariante
    F4' e per l'eventuale Step B; non cancellare per estetica.

    Solo movimento di dati (nessuna aritmetica): bit-identico per costruzione al percorso
    concat(...)[p] che sostituisce. Il blocco (chunk) limita solo la taglia del temporaneo del
    gather, non l'array di uscita, che resta unico e viene consegnato a .fit() identico a prima
    (Keras rimescola comunque per conto suo: shuffle=True di default).
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
    """Etichette di concat(zeros(n_a), ones(n_b))[p] senza costruire il concat (F4').
    float64 come il percorso precedente (dtype invariato per decisione del maintainer)."""
    return (p >= n_a).astype(np.float64)


def _pa_take_batch(phi_grad, pseudo, bi, n_a, a_index=None):
    """A3 — costruisce UN batch (X, y) del training PA indicizzando gli array RESIDENTI phi_grad/pseudo
    con gli indici globali `bi` di una fetta della permutazione: bi<n_a -> benigno (label 0) da phi_grad
    (via a_index se dato, come gather_permuted), bi>=n_a -> pseudo (label 1). Riga-per-riga equivale a
    (concat(phi_grad[a_index], pseudo)[bi], labels_permuted(bi, n_a)) ma NON materializza mai l'array
    (2N,107) ne' la sua copia-tensore Keras -> la RAM del PA scende alla taglia del batch."""
    m = bi < n_a
    Xb = np.empty((len(bi), phi_grad.shape[1]), dtype=phi_grad.dtype)
    Xb[m] = phi_grad[bi[m] if a_index is None else a_index[bi[m]]]
    Xb[~m] = pseudo[bi[~m] - n_a]
    return Xb, (bi >= n_a).astype(np.float64)


def train_pa(phi_grad, phi_mon, pseudo_mon, seed, batch=512, cap=PA_CAP):
    """Passo 5 — PA su gradiente (pseudo TRAIN per-epoca) + BCE-monitor (pseudo MONITOR fisse),
    patience-15 min. I pesi restituiti sono SEMPRE quelli a BCE-monitor minima (best_w): sia
    quando scatta la patience sia quando si esaurisce il cap (ramo "censura"), mai quelli
    dell'ultima epoca. Ritorna (model, stop_info)."""
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
        idx = rng.permutation(n_a + len(pseudo_tr))            # stesso consumo di rng di prima (era `p`)
        # A3: i batch si prendono dagli array RESIDENTI, senza materializzare Xtr_p (2N,107) ne'
        # la copia-tensore di Keras: il picco di RAM resta dell'ordine del batch.
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
    """Passo 6 — FP-mining label-free sul sistema fuso Mod A: minano i FP dei benigni-gradiente,
    li si ripete x(K+1) nel training, si ri-addestra; auto-stop se il guadagno < k_min. Smoke:
    logica portata, misura semplificata (frazione di FP catturati)."""
    # (versione operativa del protocollo; per lo smoke gira a n_max ridotto — la QUALITA' non e'
    #  richiesta dal gate, solo che il path percorra il mining cumulativo leak-free)
    rng = np.random.default_rng(int(seed) + 23)
    Zs_g, R_g = phi_grad[:, :16], phi_grad[:, 16:]
    pool = np.array([], dtype=np.int64)
    prev_metric = -1.0
    rounds_run = 0            # round di training effettivamente eseguiti (fit)
    stopped_early = False
    for rnd in range(n_max):
        logit_g = infer.fus_logit(_forward_chunked(model_pa, phi_grad).ravel())
        s_dae_g = sel_scores["sdae_grad"]
        fp = np.where(infer.apply_fusion(params_A, s_dae_g, logit_g))[0]   # FP del sistema fuso Mod A
        pool = np.union1d(pool, fp).astype(np.int64)
        metric = 1.0 - len(fp) / max(len(phi_grad), 1)                     # proxy: meno FP = meglio
        if rnd > 0 and should_stop_mining(metric, prev_metric):
            _log(f"passo 6 — hardening auto-stop al round {rnd} (Delta<{FPM_K_MIN})")
            stopped_early = True
            break
        prev_metric = metric
        # augment: FP ripetuti x(K+1) come benigni "difficili"
        aug_idx = np.concatenate([np.arange(len(phi_grad)), np.repeat(pool, FPM_K)]) if len(pool) else np.arange(len(phi_grad))
        pseudo = generate_fakes_typed(Zs_g, R_g, rng, **PA_GEN_TRAIN)
        pmt = rng.permutation(len(aug_idx) + len(pseudo))      # stesso consumo di rng di prima
        n_a = len(aug_idx)
        # A3: come in train_pa, batch dagli array residenti; a_index=aug_idx porta i FP ripetuti
        # x(K+1) senza materializzare Xtr_p (2N,107).
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
    """Inverso di _keras_to_dae_npz: ricostruisce (model, encoder) Keras dai pesi npz del DAE, per il
    retrain a DAE FISSO (--fixed-dae-dir). I layer di build_dae_tf hanno nomi noti; il GaussianNoise
    non ha pesi (no-op a training=False). Verificato bit-vicino a infer.forward_dae al call-site."""
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
    """Cap deterministico delle righe benigne a max_rows (0/None = nessun cap). Sub-sample con rng(0)
    FISSO (indipendente dal --seed) -> stesso sottoinsieme benigno per ogni seme. Replica esattamente
    la generazione di benign_1755k.npy: sort(default_rng(0).choice(n, max_rows, replace=False))."""
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

    # Guardia RAM (stessa soglia del menu, main.py): lanciato diretto, senza questa il lavoro
    # partirebbe su macchine piccole e verrebbe ucciso a meta' strada dopo ore.
    try:
        with open("/proc/meminfo") as f:
            mem_gb = int(f.readline().split()[1]) / 1024 / 1024   # MemTotal in GiB
        if mem_gb < 14.5:
            raise SystemExit(f"BLOCCATO: il riaddestramento richiede una macchina da almeno 16 GB "
                             f"di RAM (rilevati {mem_gb:.1f} GiB). Nessuna opzione lo aggira.")
    except OSError:
        pass                                            # /proc assente (non-Linux): nessun blocco

    out = Path(args.out_dir)
    if not out.exists():
        raise FileNotFoundError(f"out-dir non esiste (shell-first): {out}")
    src = Path(args.weights_src)
    fo = json.loads((src / "feature_order.json").read_text())["order"]
    scaler = json.loads((src / "scaler_params.json").read_text())
    logf = json.loads((src / "transform_meta.json").read_text())["log_features"]

    # --- ingest del dataset (traffico locale, gia' contaminato al naturale) ---
    if args.benign_npy:
        # F1 (mandato memoria 2026-07-16): il memmap NON si copia in RAM (le .npy canoniche sono gia'
        # float32): la page-cache resta condivisa fra i processi del fan-out multi-seme. La guardia
        # sotto converte solo se il dtype non e' quello atteso.
        X = np.load(args.benign_npy, mmap_mode="r")
    elif args.dataset:
        import csv
        # Ingest a CHUNK: si costruisce l'array di feature un blocco alla volta e si scartano SUBITO i
        # dict del blocco. Cosi' il picco RAM e' ~l'array di feature (91 float32/riga = 0,36 KB) + UN
        # blocco di dict, NON l'intera lista di dict (~2,7 KB/riga -> a 3M righe ~15 GB, la causa dell'OOM).
        # Nessun cap: si addestra su TUTTE le righe del traffico catturato (il mmap vale solo per --benign-npy).
        parti = []
        with infer._open_csv(args.dataset) as f:
            blocco = []
            # _flussi (non csv.DictReader): ripara le righe sfalsate dalle serie per-secondo
            # nProbe non quotate. Deve essere lo STESSO lettore dell'inferenza, altrimenti il
            # modello si addestra su feature diverse da quelle che vedra' in esercizio.
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
    if X.dtype != np.float32:                                  # guardia F1: conversione esplicita
        _log(f"WARNING: input {X.dtype} != float32 -> conversione (copia in RAM, "
             f"{X.nbytes // 2**20:,} MB): le .npy canoniche sono float32.")
        X = X.astype(np.float32)
    n0 = len(X)
    X = subsample_rows(X, args.max_rows)
    n = len(X)
    if n < n0:
        _log(f"sub-sample deterministico (rng 0, fisso fra i semi): {n0:,} -> {n:,} righe "
             f"(--max-rows; una PMI opera a scala SIZE_REF, non sul capture intero)")

    # --- N4: guardie sulla taglia ---
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
    del X                      # F2: i pool partizionano X, che da qui in poi non si rilegge piu'

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
        # D_eval per H_L = il PREFISSO del pool gradiente (indici np.sort-ordinati -> prefisso
        # TEMPORALE, ~22% della cattura a piena scala). Il dataset di ingresso e' assunto gia'
        # contaminato (anomalie naturali del traffico locale): EntropyStop lavora su quella
        # contaminazione, senza sorgenti esterne di contaminante. LIMITE NOTO (B2-#29, deciso
        # 2026-07-23): un fenomeno assente dal prefisso non influenza l'arresto; il campionamento
        # a seme fisso e' sviluppo futuro, NON introdotto per non disallineare il codice consegnato
        # dagli studi congelati che usarono il prefisso.
        X_eval = Xg[:min(ES_N_EVAL, len(Xg))]
        _log("passo 2 — DAE + EntropyStop sulla contaminazione naturale del dataset")
        model, encoder, dae_info = train_dae(Xg, X_eval, args.seed, args.max_epochs_dae)

    # --- passo 3: whitening + z-scaler sulla phi del DAE fermato (fit su select) ---
    z_s_dummy = _forward_chunked(encoder, Xs).astype(np.float64)
    Yhat_s = _forward_chunked(model, Xs).astype(np.float64)
    # Residui IN POSTO: `Yhat_s - Xs.astype(np.float64)` materializzava DUE array interi in piu' (la
    # copia float64 dell'ingresso e l'array dei residui separato), ~500 MB sul pool select a piena
    # scala. La sottrazione in posto e' BIT-IDENTICA: numpy promuove Xs a float64 elemento per
    # elemento, e un float32 e' esattamente rappresentabile in float64 (nessun arrotondamento nella
    # conversione). Yhat_s non viene piu' letto dopo questa riga, quindi il riuso e' sicuro.
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

    # A1 (M4) — score DAE (MSE binarie del DAE fermato) PRE-calcolati qui: sono gli ULTIMI consumatori
    # dei pool Xg/Xm/Xs (~1 GiB). Forward deterministico -> valori bit-identici al percorso precedente
    # (erano alle righe 625/626/655). A2 — poi si estraggono i pesi DAE (npz, pochi KB) e si LIBERA il
    # modello DAE + i pool + l'arena TF PRIMA del PA: il DAE e' gia' congelato (resta bit-exact; i pesi
    # si salvano a fine catena da dae_npz), l'encoder non serve piu'. Solo il PA scende a equivalenza
    # statistica (stesso gate del generatore A3, non abbassa ulteriormente l'asticella).
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

    # --- N2: FPR realizzato sul pool MONITOR (holdout disgiunto dalla calibrazione) per le due
    #     modalita'. E' il criterio LABEL-FREE di selezione del seme a deploy: il select realizza
    #     per costruzione l'FPR nominale (cieco), il monitor no. La scelta fra i semi (minimo Mod A,
    #     tie-break Mod B) e' in main.py; qui si misura e si salva per ciascun seme. ---
    lg_m = infer.fus_logit(_forward_chunked(model_pa, phi_m).ravel())
    fpr_monitor = {name: infer.fpr_of(infer.apply_fusion(modes[name], sdae_m, lg_m)) for name in modes}
    _log(f"N2 — FPR realizzato sul monitor (holdout, {len(phi_m)} righe): "
         f"Mod A {fpr_monitor['A'] * 100:.4f}%  Mod B {fpr_monitor['B'] * 100:.4f}%  "
         f"[selezione seme a deploy: minimo FPR Mod A, tie-break Mod B]")

    # --- salvataggio artefatti (formato npz del pacchetto) ---
    np.savez(out / "dae.npz", **dae_npz)                        # A2: pesi estratti prima del clear_session
    np.savez(out / "pa.npz", **_keras_to_pa_npz(model_pa, mu_R, L, sigma_R, mu_z, sigma_z))
    (out / "soglie.json").write_text(json.dumps(soglie, indent=2, ensure_ascii=False))
    for fn in ("feature_order.json", "scaler_params.json", "transform_meta.json"):
        (out / fn).write_text((src / fn).read_text())
    report = {"seed": args.seed, "n_benign": n, "dae": {k: v for k, v in dae_info.items() if k != "H_hist"},
              "pa": pa_info, "harden": harden_info,
              "pools": {"grad": len(idx_g), "monitor": len(idx_m), "select": len(idx_s)},
              "fpr_monitor": fpr_monitor}   # N2: criterio di selezione del seme a deploy
    (out / "train_report.json").write_text(json.dumps(report, indent=2))
    _log(f"FATTO — modello scritto in {out} (dae.npz, pa.npz, soglie.json, metadati, train_report.json)")


if __name__ == "__main__":
    main()
