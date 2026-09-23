"""lib/search/shortlist.py — dimensionamento della rosa (probabilità di selezione corretta).

Funzioni pure (NumPy, piu' un test scipy) condivise fra gli altri da `tools/shortlist_sizing.py`
(che le importa tutte e 7), `tools/select_shortlist.py` (freeze rosa) e `tools/monitor_search.py`:

  - pooled_sigma : stima NON distorta della dispersione di seme messa in comune sui trial
                   (sigma combinata), errore standard della media, gradi di libertà.
                   sigma^2 = n/(n-1) * media(std^2), con std = ap_std (ddof=0) della ricerca;
  - pcs_curve    : PCS(K) = P(vere top-t ⊆ top-K per media osservata) via Monte Carlo
                   (posterior gaussiana mu_i ~ Normal(media_i, SE^2), indipendenza);
  - smallest_K   : più piccolo K con PCS(K) >= soglia.

Niente TF: testabili bit-exact (pcs_curve è riproducibile a parità di seme RNG passato).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def pooled_sigma(stds: Sequence[float], n_seed: int) -> tuple[float, float, int]:
    """(sigma_comb, se_mean, nu_pool) dai per-trial std (ddof=0, = ap_std della ricerca).

    sigma_comb: sigma^2 = n/(n-1)·media(std^2) è stima NON distorta della VARIANZA di seme
    (il fattore n/(n-1) corregge la distorsione della varianza MLE riportata con ddof=0).
    SE = sigma_comb/sqrt(n_seed); nu_pool = N·(n_seed-1) gradi di libertà della stima combinata.
    """
    arr = np.asarray(list(stds), dtype=np.float64)
    if arr.size == 0 or int(n_seed) < 2:
        raise ValueError("pooled_sigma richiede >=1 trial e n_seed>=2")
    var_mle = float(np.mean(arr ** 2))
    sigma = math.sqrt((n_seed / (n_seed - 1)) * var_mle)
    se = sigma / math.sqrt(n_seed)
    nu_pool = int(arr.size) * (int(n_seed) - 1)
    return sigma, se, nu_pool


def pcs_curve(means_sorted: Sequence[float], se: float, t: int, draws: int, rng) -> dict:
    """PCS(K) per K=t..N: P(le vere top-t ⊆ top-K per media osservata).

    means_sorted è ordinato per media decrescente (così l'indice 0..K-1 = top-K). Per ogni
    estrazione si campiona mu ~ Normal(media, se^2) (indipendenza) e si verifica se le t medie
    vere più grandi cadono tutte nei primi K. Ritorna {K: PCS}.
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
    """PCS(K) col BOOTSTRAP APPAIATO sui semi: cattura la correlazione dei semi condivisi
    dai dati e NON assume normalità (sostituisce il Monte Carlo gaussiano indipendente).

    ap_matrix (N, n_seed): AUC-PR per-seme allineati per SEME FISICO (stessa colonna = stesso seme
    per tutte le config). A ogni replica si ricampionano i semi con reinserimento — gli STESSI
    semi per tutte le config — si ricalcolano le medie, si ordina e si guarda se le top-t cadono
    nelle prime K (per media piena osservata). Ritorna {K: PCS} per K=t..N.
    """
    A = np.asarray(ap_matrix, dtype=np.float64)
    N, ns = A.shape
    A = A[np.argsort(-A.mean(axis=1))]              # ordina per media piena: indice 0..K-1 = top-K
    m_needed = np.empty(int(draws), dtype=np.int64)
    done = 0
    while done < int(draws):
        b = min(int(chunk), int(draws) - done)
        idx = rng.integers(0, ns, size=(b, ns))     # ricampiona ns semi (gli stessi per ogni config)
        mb = A[:, idx].mean(axis=2).T               # (b, N) medie bootstrap
        top = np.argsort(-mb, axis=1)[:, :int(t)]   # top-t per replica
        m_needed[done:done + b] = top.max(axis=1) + 1   # più piccolo K che le contiene
        done += b
    return {K: float(np.mean(m_needed <= K)) for K in range(int(t), N + 1)}


def seeds_for_target_K(means_sorted, sigma, t, conf, K_target, draws, seed, n_max=4000) -> int | None:
    """Minimo numero di semi $n$ per cui la shortlist al livello `conf` per le vere top-t scende a
    $\\le$ K_target, assumendo le medie OSSERVATE come vere e SE(n)=sigma/sqrt(n) (power analysis
    forward). K(n) è non-crescente in attesa → bisezione (draws assorbe il rumore MC). None se irraggiungibile entro n_max.
    Deterministica dato `seed` (un RNG per ogni n provato)."""
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
    """PCS(K) col BOOTSTRAP BAYESIANO (pesi Dirichlet(1) sui semi): liscio, senza le
    molteplicità intere del multinomiale (niente repliche degeneri a pochi semi distinti).
    NON è il modello operativo — la sua varianza della media è s²/(n+1), SOTTO-dispersa di
    (n+1)/(n-1) rispetto alla varianza frequentista s²/(n-1) (è una posteriori). Serve come
    CONTROPROVA: se la coda del multinomiale sparisce qui, era degenerazione a n piccolo."""
    A = np.asarray(ap_matrix, dtype=np.float64)
    N, ns = A.shape
    A = A[np.argsort(-A.mean(axis=1))]
    m_needed = np.empty(int(draws), dtype=np.int64)
    done = 0
    while done < int(draws):
        b = min(int(chunk), int(draws) - done)
        W = rng.dirichlet(np.ones(ns), size=b)          # (b, ns) pesi continui > 0
        mb = (A @ W.T).T                                # (b, N)
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
