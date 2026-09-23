"""lib/dae/evaluate.py — scoring di anomalia del DAE (repov6, parti per la ricerca).

Punteggio = MSE float32 sulle SOLE binarie [N_CONTINUOUS:N_FEATURES] (lib.utils.SCORE_DEFINITION):
nessuna BCE (la BCE e' solo loss di training). Le funzioni della ricerca AUC-PR convivono
qui con le metriche offline e la caratterizzazione cap2 (whitening/L1-L2/Cohen-d).
"""
from __future__ import annotations

import math

import numpy as np

from lib.utils import N_BINARY, N_CONTINUOUS, N_FEATURES


def mse_binary_score(y_pred: np.ndarray, x_true: np.ndarray,
                     n_cont: int = N_CONTINUOUS, n_bin: int = N_BINARY) -> np.ndarray:
    """Punteggio di anomalia: mean((Yhat_bin - X_bin)^2, axis=1) float32, sole binarie
    [n_cont:n_cont+n_bin]. Funzione PURA NumPy: definizione unica condivisa col deploy."""
    bin_slice = slice(n_cont, n_cont + n_bin)
    diff_bin = y_pred[:, bin_slice] - x_true[:, bin_slice]
    return np.mean(diff_bin * diff_bin, axis=1).astype(np.float32)


def compute_n_params(exp: int, mid: int, btl: int, n_features: int = N_FEATURES) -> int:
    """Numero parametri totale del DAE 5-layer (formula chiusa).
    Architettura: n_features -> exp -> mid -> btl -> mid -> exp -> (n_cont lin + n_bin sig)."""
    return ((n_features + 1) * exp
            + (exp + 1) * mid
            + (mid + 1) * btl
            + (btl + 1) * mid
            + (mid + 1) * exp
            + (exp + 1) * n_features)


def compute_scores_chunked(predict_fn, X: np.ndarray, n_cont: int = N_CONTINUOUS,
                           n_bin: int = N_BINARY, chunk_size: int = 50000) -> np.ndarray:
    """Predice X in chunk via tf.function compilata e ritorna SOLO gli score MSE binari (1D).
    Mai materializza y_pred completo (peak transient contenuto). TF import lazy."""
    import tensorflow as tf
    n = int(len(X))
    if n == 0:
        return np.empty(0, dtype=np.float32)
    scores = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk_size):
        j = min(i + chunk_size, n)
        chunk_np = np.ascontiguousarray(X[i:j], dtype=np.float32)
        y_pred = predict_fn(tf.constant(chunk_np)).numpy()
        scores[i:j] = mse_binary_score(y_pred, chunk_np, n_cont, n_bin)
        del y_pred, chunk_np
    return scores


def compute_scores_and_residuals(predict_fn, X: np.ndarray, n_cont: int = N_CONTINUOUS,
                                 n_bin: int = N_BINARY, chunk_size: int = 50000):
    """Come compute_scores_chunked ma ritorna ANCHE i residui R = Yhat - X su TUTTE le
    n_cont+n_bin (=91) feature, in chunk. Ritorna (scores (N,) float32, residuals (N,91) float32).
    Usato dalla caratterizzazione dell'uscita del DAE (residui + sbiancamento, chiusura cap2)."""
    import tensorflow as tf
    n_total = int(len(X))
    n_feat = n_cont + n_bin
    scores = np.empty(n_total, dtype=np.float32)
    residuals = np.empty((n_total, n_feat), dtype=np.float32)
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        x_chunk = np.ascontiguousarray(X[start:end], dtype=np.float32)
        y_pred = predict_fn(tf.constant(x_chunk)).numpy()
        residuals[start:end] = y_pred - x_chunk
        scores[start:end] = mse_binary_score(y_pred, x_chunk, n_cont, n_bin)
        del y_pred, x_chunk
    return scores, residuals


# ---- metriche di valutazione (LOAO / analisi offline) ----

def binary_metrics_at_threshold(y_true, y_score, threshold: float) -> dict:
    """Metriche binarie a soglia singola: y_pred = (y_score > threshold). Restituisce conteggi
    di confusione, precision/recall/f1, MCC, FPR effettivo, AUC-ROC/PR. MCC/AUC = NaN se una
    classe è assente; le AUC anche se lo score è costante. sklearn import LAZY (il solo
    import del modulo non trascina sklearn)."""
    from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                                 f1_score, matthews_corrcoef, precision_score,
                                 recall_score, roc_curve)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score > threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_pos, n_neg = tp + fn, tn + fp
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if (n_pos > 0 and n_neg > 0) else float("nan")
    fpr_value = fp / n_neg if n_neg > 0 else float("nan")
    if n_pos == 0 or n_neg == 0 or len(np.unique(y_score)) < 2:
        auc_roc = auc_pr = float("nan")
    else:
        fpr_c, tpr_c, _ = roc_curve(y_true, y_score)
        auc_roc = auc(fpr_c, tpr_c)
        auc_pr = average_precision_score(y_true, y_score)
    return dict(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                n_pos=int(n_pos), n_neg=int(n_neg),
                precision=float(prec), recall=float(rec), f1=float(f1),
                mcc=float(mcc), fpr=float(fpr_value),
                auc_roc=float(auc_roc), auc_pr=float(auc_pr))


def cohens_d(a, b) -> float:
    """Cohen's d = (mean(a) - mean(b)) / pooled_std (varianze ddof=1). NaN se pooled_std == 0
    o n < 2. Misura la separazione tra gli score di una classe e quelli benigni (per-classe LOAO)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    n1, n2 = len(a), len(b)
    pooled = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if pooled == 0.0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


# ---- caratterizzazione dell'uscita del DAE: sbiancamento residui + norme L1/L2 (chiusura cap2) ----
# Ricetta de-etichettata dall'oracolo repo_v5 (lib/dae/evaluate.py + secondo_stadio/build_cache.py),
# adattata a 91 feature. NON è il secondo stadio: caratterizza l'uscita del DAE (residui sbiancati).

def derive_whitening(R_val: np.ndarray, val_attack: np.ndarray):
    """Deriva (mu, L, sigma_R, shrinkage, chol_source) da UNA sola LedoitWolf sui residui BENIGNI val.
    mu=location_; L=Cholesky inferiore di Sigma^-1 (whitening, L L^T=Sigma^-1); sigma_R=sqrt(diag(Sigma)).
    PD-fallback (91 feat, binarie quasi-costanti): se cholesky(precision_) fallisce → L=inv(chol(Sigma))^T.
    sklearn import LAZY (analisi offline)."""
    from sklearn.covariance import LedoitWolf
    R_val = np.asarray(R_val)
    mask_ben = (np.asarray(val_attack) == "Benign")
    lw = LedoitWolf().fit(R_val[mask_ben])
    mu = lw.location_.astype(np.float64)
    try:
        L = np.linalg.cholesky(lw.precision_).astype(np.float64)
        chol_source = "precision"
    except np.linalg.LinAlgError:
        C = np.linalg.cholesky(lw.covariance_)           # C C^T = Sigma  -> L = C^-T
        L = np.linalg.inv(C).T.astype(np.float64)
        chol_source = "covariance_fallback"
    sigma_R = np.sqrt(np.diag(lw.covariance_)).astype(np.float64)
    return mu, L, sigma_R, float(lw.shrinkage_), chol_source


def whiten_residuals(R: np.ndarray, mu: np.ndarray, L: np.ndarray,
                     chunk_size: int = 50_000, out: np.ndarray | None = None) -> np.ndarray:
    """Sbiancamento (R - mu) @ L chunkato. L = Cholesky inferiore di Sigma^-1 → cov(R_w|benigni) ≈ I.
    `out` (es. R stesso) → scrittura in-place a blocchi (cap RAM). Bit-exact vs out-of-place: la RHS
    di ogni blocco è valutata prima dell'assegnazione e i blocchi non si sovrappongono."""
    R = np.asarray(R)
    N, D = R.shape
    if out is None:
        out = np.empty((N, D), dtype=np.float32)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        out[start:end] = (R[start:end].astype(np.float64) - mu) @ L
    return out


def score_l1(R_w: np.ndarray) -> np.ndarray:
    """Proiezione scalare L1 dei residui sbiancati: (1/D) * sum_i |R_w,i| per flusso, float32."""
    return np.abs(R_w).mean(axis=1).astype(np.float32)


def score_l2(R_w: np.ndarray) -> np.ndarray:
    """Proiezione scalare L2 dei residui sbiancati: (1/D) * sum_i R_w,i^2 per flusso (norma
    quadratica media, non radice). Accumulo float64 per stabilità numerica."""
    return (np.asarray(R_w).astype(np.float64) ** 2).mean(axis=1)


def pr_and_roc_curves(y_true, y_score):
    """Curve complete precision-recall e ROC (array, non scalari). Ritorna
    (precision, recall, fpr, tpr). sklearn import LAZY."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return precision, recall, fpr, tpr
