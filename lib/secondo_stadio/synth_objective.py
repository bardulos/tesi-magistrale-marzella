"""lib/secondo_stadio/synth_objective.py — Cap4 WP-4: dataset sintetico di validazione.

Logica pura e testabile (numpy; sklearn solo lazy dentro ap_metric) per la ricerca sul
DATASET SINTETICO (mai sul modello — anti-circolarità): generazione parametrica delle
pseudo-anomalie con località dello swap, metriche di copertura dell'open-space attorno
al manifold benigno, potere risolutivo sui 25 candidati congelati.

Obiettivo (decisione chiusa n.1 del piano cap4): copertura geometrica PRIMARIA
(coverage_objective) + potere risolutivo come vincolo/diagnostica (resolving_power,
filtro a valle RP_MIN — non entra nell'obiettivo).

Generazione (generalizza lib/secondo_stadio/generators.py, che resta la referenza
canonica per il training del PA — qui serve taglia arbitraria e località dello swap;
semantica invariata: swap = [z_i, r_j] con j != i, radial = [z_i, alpha*r_i]):
  - k_swap = 0  → partner GLOBALE uniforme (come make_swap);
  - k_swap = k  → partner fra i k vicini più prossimi di i nello spazio latente z
                  (nn_idx precalcolato, il vicino 0 = self è già escluso).

Le metriche di copertura sono calcolate in phi = [z_s(16), r_w(91)] ∈ R^107 rispetto a
un riferimento benigno FISSO; la scala è ancorata alle distanze self benigne (quantili
q50/q99 precalcolati): "dentro il supporto" = d_min < q50_self (pseudo-benigni, rumore
di etichetta), "shell difficile" = q50_self <= d_min < SHELL_UPPER_MULT*q99_self,
"lontane" = oltre (facili, inutili). La copertura direzionale è il participation ratio
delle direzioni (pseudo - centroide benigno) normalizzate.
"""
from __future__ import annotations

import numpy as np

from lib.secondo_stadio.constants import PHI_DIM

# costanti di design WP-4 (pesi dell'obiettivo + bordo shell + soglia rp), documentate nel piano
LAMBDA_INSIDE = 1.0      # penalità per pseudo dentro il supporto benigno (rumore di etichetta)
LAMBDA_DIR = 0.5         # premio per la copertura direzionale (participation ratio / PHI_DIM)
SHELL_UPPER_MULT = 4.0   # bordo esterno della shell difficile, in multipli di q99_self
RP_MIN = 3.0             # vincolo di potere risolutivo (filtro a valle, mai nell'obiettivo)


# generazione
def generate_synth(Z_pool: np.ndarray, R_pool: np.ndarray, nn_idx: np.ndarray | None,
                   rng: np.random.Generator, delta: float, alpha_lo: float, alpha_hi: float,
                   k_swap: int, size: int) -> np.ndarray:
    """Genera `size` pseudo-anomalie phi float32 dal pool benigno (righe iid).

    radial con probabilità delta: phi = [z_i, alpha*r_i], alpha ~ U(alpha_lo, alpha_hi);
    swap con probabilità 1-delta: phi = [z_i, r_j] con j != i (globale se k_swap=0,
    altrimenti j fra i primi k_swap vicini di i in nn_idx).
    """
    n_pool = len(Z_pool)
    if n_pool < 2:
        raise ValueError(f"pool di generazione troppo piccolo (n={n_pool})")
    if not (0.0 <= delta <= 1.0):
        raise ValueError(f"delta fuori da [0,1]: {delta}")
    if alpha_hi < alpha_lo:
        raise ValueError(f"alpha_hi < alpha_lo: {alpha_hi} < {alpha_lo}")
    if k_swap < 0:
        raise ValueError(f"k_swap negativo: {k_swap}")
    if k_swap > 0:
        if nn_idx is None:
            raise ValueError("k_swap > 0 richiede nn_idx precalcolato")
        if k_swap > nn_idx.shape[1]:
            raise ValueError(f"k_swap={k_swap} > vicini disponibili ({nn_idx.shape[1]})")

    src = rng.integers(n_pool, size=size)
    is_radial = rng.random(size) < delta
    alpha = rng.uniform(alpha_lo, alpha_hi, size=size).astype(np.float32)

    if k_swap == 0:
        # partner globale j != i: un ricampionamento + shift deterministico dei residui collisi
        partner = rng.integers(n_pool, size=size)
        collide = partner == src
        partner[collide] = (partner[collide] + 1) % n_pool
    else:
        col = rng.integers(k_swap, size=size)
        partner = nn_idx[src, col]

    z_part = np.asarray(Z_pool[src], dtype=np.float32)
    r_swap = np.asarray(R_pool[partner], dtype=np.float32)
    r_radial = np.asarray(R_pool[src], dtype=np.float32) * alpha[:, None]
    r_part = np.where(is_radial[:, None], r_radial, r_swap)

    phi = np.hstack([z_part, r_part]).astype(np.float32)
    if phi.shape[1] != PHI_DIM:
        raise ValueError(f"phi ha dimensione {phi.shape[1]} != PHI_DIM={PHI_DIM}")
    return phi


# distanze
def min_dist_chunked(x: np.ndarray, ref: np.ndarray, chunk: int = 1024,
                     exclude_self: bool = False) -> np.ndarray:
    """Distanza euclidea di ogni riga di x dal punto più vicino di ref (GEMM a blocchi).

    Con exclude_self=True x e ref DEVONO essere lo stesso array: la coppia (i, i) è
    esclusa (distanza self benigna leave-one-out).
    """
    x = np.asarray(x, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    ref_sq = np.square(ref).sum(axis=1)
    out = np.empty(len(x), dtype=np.float32)
    for a in range(0, len(x), chunk):
        b = min(a + chunk, len(x))
        blk = x[a:b]
        d2 = np.square(blk).sum(axis=1)[:, None] - 2.0 * (blk @ ref.T) + ref_sq[None, :]
        if exclude_self:
            for k in range(b - a):
                d2[k, a + k] = np.inf
        np.clip(d2, 0.0, None, out=d2)
        out[a:b] = np.sqrt(d2.min(axis=1))
    return out


def benign_self_quantiles(phi_ref: np.ndarray, chunk: int = 1024) -> tuple[float, float]:
    """Quantili (q50, q99) della distanza self benigna (nearest neighbour leave-one-out):
    la scala di riferimento per 'dentro il supporto' / 'shell difficile'."""
    d_self = min_dist_chunked(phi_ref, phi_ref, chunk=chunk, exclude_self=True)
    return float(np.quantile(d_self, 0.50)), float(np.quantile(d_self, 0.99))


# copertura
def coverage_metrics(phi_pseudo: np.ndarray, phi_ref: np.ndarray,
                     q50_self: float, q99_self: float, chunk: int = 1024) -> dict:
    """Metriche di copertura dell'open-space per un campione di pseudo, vs riferimento benigno.

    frac_inside/frac_shell/frac_far partizionano il campione (somma 1); dir_pr è il
    participation ratio delle direzioni normalizzate (pseudo - centroide benigno).
    """
    d = min_dist_chunked(phi_pseudo, phi_ref, chunk=chunk)
    upper = SHELL_UPPER_MULT * q99_self
    frac_inside = float((d < q50_self).mean())
    frac_far = float((d >= upper).mean())
    frac_shell = float(1.0 - frac_inside - frac_far)

    centroid = np.asarray(phi_ref, dtype=np.float64).mean(axis=0)
    dirs = np.asarray(phi_pseudo, dtype=np.float64) - centroid
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    dirs = dirs / norms
    cov = (dirs.T @ dirs) / max(1, len(dirs) - 1)
    lam = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    tot = float(lam.sum())
    dir_pr = float(tot ** 2 / (np.square(lam).sum() + 1e-24)) if tot > 0.0 else 0.0

    return {"frac_inside": frac_inside, "frac_shell": frac_shell,
            "frac_far": frac_far, "dir_pr": dir_pr,
            "d_min_med": float(np.median(d)), "q50_self": q50_self, "q99_self": q99_self}


def coverage_objective(m: dict) -> float:
    """Obiettivo di copertura (da MASSIMIZZARE), pesi dichiarati in testa al modulo."""
    return m["frac_shell"] - LAMBDA_INSIDE * m["frac_inside"] + LAMBDA_DIR * (m["dir_pr"] / PHI_DIM)


# modelli (NumPy puro)
def mlp_logits(x: np.ndarray, layers: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Forward NumPy dell'MLP del secondo stadio: Dense-ReLU sugli hidden, ultimo layer
    LINEARE (= logit pre-sigmoide: per AUC-PR/ordinamenti la sigmoide è irrilevante)."""
    h = np.asarray(x, dtype=np.float32)
    for w, b in layers[:-1]:
        h = np.maximum(h @ w + b, 0.0)
    w, b = layers[-1]
    return (h @ w + b).ravel()


def ap_metric(logit_ben: np.ndarray, logit_pseudo: np.ndarray) -> float:
    """Average Precision benigni(0) vs pseudo(1) sui logit (metrica sintetica per-modello,
    convenzione AUC-PR del repo)."""
    from sklearn.metrics import average_precision_score
    y = np.concatenate([np.zeros(len(logit_ben)), np.ones(len(logit_pseudo))])
    s = np.concatenate([logit_ben, logit_pseudo])
    return float(average_precision_score(y, s))


def resolving_power(ap_by_model: dict, sigma_by_model: dict) -> float:
    """Potere risolutivo del dataset: dispersione inter-modello della metrica sintetica /
    rumore intra-modello mediano (sigma inter-seme precalcolata sul dataset di riferimento).
    Vincolo a valle: un dataset con rp < RP_MIN non separa i candidati e viene scartato."""
    aps = np.array([ap_by_model[k] for k in sorted(ap_by_model)])
    sigmas = np.array([sigma_by_model[k] for k in sorted(sigma_by_model)])
    sigma_med = float(np.median(sigmas))
    if sigma_med <= 0.0:
        return float("inf") if aps.std() > 0 else 0.0
    return float(aps.std() / sigma_med)
