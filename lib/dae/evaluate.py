"""Scoring DAE, metriche offline e caratterizzazione dei residui.

Lo score e' MSE float32 sulle sole binarie; la BCE e' solo loss di training.
"""
from __future__ import annotations

import math

import numpy as np

from lib.utils import N_BINARY, N_CONTINUOUS, N_FEATURES


def mse_binary_score(y_pred: np.ndarray, x_true: np.ndarray,
                     n_cont: int = N_CONTINUOUS, n_bin: int = N_BINARY) -> np.ndarray:
    """MSE float32 sulle feature binarie, per riga."""
    bin_slice = slice(n_cont, n_cont + n_bin)
    diff_bin = y_pred[:, bin_slice] - x_true[:, bin_slice]
    return np.mean(diff_bin * diff_bin, axis=1).astype(np.float32)


def compute_n_params(exp: int, mid: int, btl: int, n_features: int = N_FEATURES) -> int:
    """Numero di parametri del DAE a cinque layer."""
    return ((n_features + 1) * exp
            + (exp + 1) * mid
            + (mid + 1) * btl
            + (btl + 1) * mid
            + (mid + 1) * exp
            + (exp + 1) * n_features)


def compute_scores_chunked(predict_fn, X: np.ndarray, n_cont: int = N_CONTINUOUS,
                           n_bin: int = N_BINARY, chunk_size: int = 50000) -> np.ndarray:
    """Calcola gli score MSE binari a blocchi senza materializzare tutte le predizioni."""
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
    """Calcola score e residui R=Yhat-X su tutte le feature, a blocchi."""
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
    """Metriche binarie a soglia singola (`y_score > threshold`).

    MCC e AUC sono NaN se manca una classe (le AUC anche con score costante), FPR se
    mancano i negativi; precision, recall e F1 non definite valgono 0.
    """
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
    """Cohen's d con varianze campionarie; NaN se la deviazione pooled e' zero o n < 2."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    n1, n2 = len(a), len(b)
    pooled = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if pooled == 0.0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


# ---- Whitening dei residui e score L1/L2 ----

def derive_whitening(R_val: np.ndarray, val_attack: np.ndarray):
    """Stima whitening e scale dei residui benigni val con Ledoit-Wolf.

    Se il Cholesky della precisione fallisce, ricava L dalla covarianza.
    """
    from sklearn.covariance import LedoitWolf
    R_val = np.asarray(R_val)
    mask_ben = (np.asarray(val_attack) == "Benign")
    lw = LedoitWolf().fit(R_val[mask_ben])
    mu = lw.location_.astype(np.float64)
    try:
        L = np.linalg.cholesky(lw.precision_).astype(np.float64)
        chol_source = "precision"
    except np.linalg.LinAlgError:
        C = np.linalg.cholesky(lw.covariance_)
        L = np.linalg.inv(C).T.astype(np.float64)
        chol_source = "covariance_fallback"
    sigma_R = np.sqrt(np.diag(lw.covariance_)).astype(np.float64)
    return mu, L, sigma_R, float(lw.shrinkage_), chol_source


def whiten_residuals(R: np.ndarray, mu: np.ndarray, L: np.ndarray,
                     chunk_size: int = 50_000, out: np.ndarray | None = None) -> np.ndarray:
    """Applica (R-mu)@L a blocchi; `out` consente la scrittura in-place."""
    R = np.asarray(R)
    N, D = R.shape
    if out is None:
        out = np.empty((N, D), dtype=np.float32)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        out[start:end] = (R[start:end].astype(np.float64) - mu) @ L
    return out


def score_l1(R_w: np.ndarray) -> np.ndarray:
    """Media dei valori assoluti dei residui sbiancati per riga."""
    return np.abs(R_w).mean(axis=1).astype(np.float32)


def score_l2(R_w: np.ndarray) -> np.ndarray:
    """Media dei residui quadratici per riga, senza radice; accumula in float64."""
    return (np.asarray(R_w).astype(np.float64) ** 2).mean(axis=1)


def pr_and_roc_curves(y_true, y_score):
    """Restituisce le curve precision-recall e ROC come array."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return precision, recall, fpr, tpr
