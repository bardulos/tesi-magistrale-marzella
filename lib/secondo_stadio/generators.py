"""lib/secondo_stadio/generators.py — generatori di pseudo-anomalie nello spazio phi.

Due classi di pseudo-anomalie (port da repo_v5 pseudo_anomalie.py:94-147, adattato a
PHI_DIM=107; logica invariata):
  - swap   : phi = [z_s_i, r_w_j] con j != i (latente di un benigno, residuo di un ALTRO);
  - radial : phi = [z_s_i, alpha * r_w_i] con alpha ~ U(alpha_lo, alpha_hi) (residuo
             gonfiato radialmente; alpha_lo >= 1 evita pseudo-benigni).
Miscela: radial con probabilita' delta, swap con (1 - delta). Una pseudo per riga benigna
(rapporto benigni:PA = 1:1 nel batch). Determinismo: tutto il campionamento passa dal
`rng` del chiamante (np.random.default_rng(seed) per-seme); nessuno stato globale.
Proprieta' verificate: self-swap escluso, radial scala solo la parte residua, frazione
radiale ~ delta, dtype float32, determinismo a parita' di rng.
"""
from __future__ import annotations

import numpy as np

from lib.secondo_stadio.constants import PHI_DIM


def make_swap(i: int, Zs: np.ndarray, R_w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Swap globale: phi = [Zs[i], R_w[j]], j != i campionato uniforme (rejection sampling).
    Senza il vincolo j != i un self-swap produrrebbe phi = [Zs[i], R_w[i]] = un benigno
    reale etichettato pseudo-anomalia. PRECONDIZIONE n >= 2: con n = 1 il rejection
    sampling non terminerebbe (il guard converte l'hang in errore esplicito ed e' inerte
    per n >= 2: non consuma RNG, non cambia l'output)."""
    n = len(Zs)
    if n < 2:
        raise ValueError(f"make_swap richiede un pool di >=2 benigni (n={n}): con n=1 il "
                         f"rejection sampling j != i non terminerebbe.")
    j = int(rng.integers(n))
    while j == i:
        j = int(rng.integers(n))
    return np.concatenate([Zs[i], R_w[j]], axis=0).astype(np.float32)


def make_radial(i: int, Zs: np.ndarray, R_w: np.ndarray, rng: np.random.Generator,
                alpha_lo: float, alpha_hi: float) -> np.ndarray:
    """Radial globale: phi = [Zs[i], alpha * R_w[i]], alpha ~ U(alpha_lo, alpha_hi)."""
    alpha = float(rng.uniform(alpha_lo, alpha_hi))
    return np.concatenate([Zs[i], alpha * R_w[i]], axis=0).astype(np.float32)


def generate_fakes_typed(Zs_b: np.ndarray, R_b: np.ndarray, rng: np.random.Generator,
                         delta: float, alpha_lo: float, alpha_hi: float):
    """Una pseudo per riga: radial con prob delta, swap con prob (1-delta).
    Ritorna (out, is_radial). PRECONDIZIONE n >= 2 all'ingresso: da meno di 2 benigni non
    si costruisce un set di pseudo-anomalie sensato, qualunque sia il ramo (guard inerte
    per n >= 2: non consuma RNG, output invariato)."""
    n = len(Zs_b)
    if n < 2:
        raise ValueError(f"generate_fakes richiede un pool di >=2 benigni (n={n}).")
    out = np.empty((n, PHI_DIM), dtype=np.float32)
    is_radial = rng.random(n) < delta
    for k in range(n):
        if is_radial[k]:
            out[k] = make_radial(k, Zs_b, R_b, rng, alpha_lo, alpha_hi)
        else:
            out[k] = make_swap(k, Zs_b, R_b, rng)
    return out, is_radial


def generate_fakes(Zs_b: np.ndarray, R_b: np.ndarray, rng: np.random.Generator,
                   delta: float, alpha_lo: float, alpha_hi: float) -> np.ndarray:
    """Una pseudo-anomalia per riga benigna (radial prob delta, swap prob 1-delta)."""
    return generate_fakes_typed(Zs_b, R_b, rng, delta, alpha_lo, alpha_hi)[0]


def pseudo_seed_for_trial(trial_number: int, base_seed: int) -> int:
    """Seed deterministico per le pseudo-monitor di un trial Optuna (WP-4_v2, ricerca dei
    parametri delta/alpha_lo/alpha_hi su tools/pa_wp4v2_search.py): combina `trial_number`
    (univoco per-studio, assegnato una volta sola sul JournalStorage condiviso — nessuna
    collisione fra trial concorrenti) con `base_seed` (costante fissa, indipendente dal seme
    del modello: lib.secondo_stadio.constants.PSEUDO_MONITOR_SEED). Stesso trial_number ->
    stesse pseudo, riproducibile. Forma aritmetica ESPLICITA (non hash() builtin): l'hash di
    una tupla di interi e' stabile fra processi in CPython (PYTHONHASHSEED randomizza solo
    str/bytes), ma non e' un contratto documentato dal linguaggio — questa derivazione e'
    portabile per costruzione, indipendente dall'implementazione dell'hash."""
    return (int(base_seed) * 1_000_003 + int(trial_number)) % (2**31 - 1)
