"""Stime NumPy per dimensionare shortlist con probabilita' di selezione corretta."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def pooled_sigma(stds: Sequence[float], n_seed: int) -> tuple[float, float, int]:
    """Restituisce dispersione combinata, errore standard e gradi di liberta'."""
    arr = np.asarray(list(stds), dtype=np.float64)
    if arr.size == 0 or int(n_seed) < 2:
        raise ValueError("pooled_sigma richiede >=1 trial e n_seed>=2")
    var_mle = float(np.mean(arr ** 2))
    sigma = math.sqrt((n_seed / (n_seed - 1)) * var_mle)
    se = sigma / math.sqrt(n_seed)
    nu_pool = int(arr.size) * (int(n_seed) - 1)
    return sigma, se, nu_pool


def pcs_curve(means_sorted: Sequence[float], se: float, t: int, draws: int, rng) -> dict:
    """Stima PCS(K), K=t..N, con medie campionate da normali indipendenti N(media, se^2).

    `means_sorted` deve essere in ordine decrescente: gli indici 0..K-1 sono la top-K osservata.
    """
    m = np.asarray(means_sorted, dtype=np.float64)
    n = m.size
    mu = m[None, :] + float(se) * rng.standard_normal((int(draws), n))
    true_top = np.argsort(-mu, axis=1)[:, :int(t)]
    return {K: float(np.mean((true_top < K).all(axis=1))) for K in range(int(t), n + 1)}


def smallest_K(curve: dict, thr: float) -> int:
    """Più piccolo K con curve[K] >= thr; max(K) se nessuno raggiunge la soglia."""
    for K in sorted(curve):
        if curve[K] >= float(thr):
            return K
    return max(curve)


def pcs_curve_paired(ap_matrix, t, draws, rng, chunk=20000) -> dict:
    """Stima PCS(K) con bootstrap appaiato dei semi, preservandone la correlazione."""
    A = np.asarray(ap_matrix, dtype=np.float64)
    N, ns = A.shape
    A = A[np.argsort(-A.mean(axis=1))]
    m_needed = np.empty(int(draws), dtype=np.int64)
    done = 0
    while done < int(draws):
        b = min(int(chunk), int(draws) - done)
        idx = rng.integers(0, ns, size=(b, ns))
        mb = A[:, idx].mean(axis=2).T
        top = np.argsort(-mb, axis=1)[:, :int(t)]
        m_needed[done:done + b] = top.max(axis=1) + 1
        done += b
    return {K: float(np.mean(m_needed <= K)) for K in range(int(t), N + 1)}


def seeds_for_target_K(means_sorted, sigma, t, conf, K_target, draws, seed, n_max=4000) -> int | None:
    """Stima il minimo numero di semi per ottenere una shortlist <= K_target."""
    import math as _m

    def K_at(n: int) -> int:
        rng = np.random.default_rng(int(seed) + int(n))
        return smallest_K(pcs_curve(means_sorted, sigma / _m.sqrt(n), t, draws, rng), conf)

    if K_at(n_max) > int(K_target):
        return None
    lo, hi = 1, int(n_max)
    while lo < hi:
        mid = (lo + hi) // 2
        if K_at(mid) <= int(K_target):
            hi = mid
        else:
            lo = mid + 1
    return lo


def pcs_curve_bayes(ap_matrix, t, draws, rng, chunk=20000) -> dict:
    """Stima PCS(K) con bootstrap bayesiano a pesi Dirichlet sui semi."""
    A = np.asarray(ap_matrix, dtype=np.float64)
    N, ns = A.shape
    A = A[np.argsort(-A.mean(axis=1))]
    m_needed = np.empty(int(draws), dtype=np.int64)
    done = 0
    while done < int(draws):
        b = min(int(chunk), int(draws) - done)
        W = rng.dirichlet(np.ones(ns), size=b)
        mb = (A @ W.T).T
        top = np.argsort(-mb, axis=1)[:, :int(t)]
        m_needed[done:done + b] = top.max(axis=1) + 1
        done += b
    return {K: float(np.mean(m_needed <= K)) for K in range(int(t), N + 1)}


def variance_homogeneity(ap_matrix) -> tuple[float, float]:
    """(stat, p-value) del test di Levene (robusto alla non-normalità) sull'uguaglianza
    delle varianze inter-seme tra le configurazioni (righe della matrice). p<0.05 → varianze
    NON omogenee (il pooling della dispersione di seme andrebbe rivisto)."""
    from scipy.stats import levene
    A = np.asarray(ap_matrix, dtype=np.float64)
    stat, p = levene(*A)
    return float(stat), float(p)
