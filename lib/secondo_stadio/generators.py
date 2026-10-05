"""Generatori di pseudo-anomalie swap e radial nello spazio phi."""
from __future__ import annotations

import numpy as np

from lib.secondo_stadio.constants import PHI_DIM


def make_swap(i: int, Zs: np.ndarray, R_w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Combina il latente `i` con un residuo casuale di indice diverso; richiede n>=2."""
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
    """Genera una pseudo per riga e restituisce anche il flag radial; richiede n>=2."""
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
    """Seed deterministico dalle chiavi trial e base, senza usare `hash()`."""
    return (int(base_seed) * 1_000_003 + int(trial_number)) % (2**31 - 1)
