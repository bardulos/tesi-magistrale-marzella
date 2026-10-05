"""I/O NumPy dei pesi e forward MLP senza TensorFlow.

I pesi sono coppie kernel/bias ordinate per layer; l'ordine e' numerico, non
lessicografico. `mlp_forward` replica il forward Keras in inference (scarto <=1e-5).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_weights_npz(weights: list, path) -> None:
    """Salva i pesi in NPZ con chiavi numerate `w0`, `w1`, ..."""
    np.savez(str(path), **{f"w{i}": np.asarray(w) for i, w in enumerate(weights)})


def load_weights_npz(path) -> list:
    """Carica i pesi NPZ nell'ordine numerico originale."""
    with np.load(str(path)) as d:
        return [d[f"w{i}"] for i in range(len(d.files))]


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """Sigmoid stabile, calcolata in float64."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def mlp_forward(weights: list, X: np.ndarray) -> np.ndarray:
    """Forward NumPy di Dense-ReLU e Dense-sigmoid, compatibile con i pesi Keras."""
    if len(weights) % 2 != 0:
        raise ValueError(f"weights deve essere una sequenza di coppie [kernel,bias]: "
                         f"{len(weights)} elementi (dispari)")
    h = np.asarray(X, dtype=np.float32)
    n_pairs = len(weights) // 2
    for k in range(n_pairs):
        W, b = np.asarray(weights[2 * k], dtype=np.float32), np.asarray(weights[2 * k + 1], dtype=np.float32)
        h = h @ W + b
        if k < n_pairs - 1:
            h = np.maximum(h, np.float32(0.0))
        else:
            h = _sigmoid_stable(h).astype(np.float32)
    return h.ravel()


def load_weight_trajectory(weights_dir, n_epochs: int) -> list:
    """Carica i pesi salvati da e0 a e{n_epochs}."""
    weights_dir = Path(weights_dir)
    return [load_weights_npz(weights_dir / f"weights_epoch_{e:03d}.npz") for e in range(n_epochs + 1)]


def consolidate_weight_trajectory(weights_dir, out_dir, n_epochs: int) -> int:
    """Consolida ogni array di peso in un file NPY impilato e memmap-abile."""
    traj = load_weight_trajectory(weights_dir, n_epochs)
    n_arrays = len(traj[0])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_arrays):
        stacked = np.stack([epoch_w[i] for epoch_w in traj], axis=0)
        np.save(out_dir / f"w{i}.npy", stacked)
    return n_arrays // 2


def load_consolidated_trajectory(out_dir, mmap: bool = True) -> list:
    """Carica gli array consolidati in ordine numerico, opzionalmente via mmap."""
    out_dir = Path(out_dir)
    paths = sorted(out_dir.glob("w*.npy"), key=lambda p: int(p.stem[1:]))
    mode = "r" if mmap else None
    return [np.load(p, mmap_mode=mode) for p in paths]


def weights_at_epoch(consolidated: list, epoch: int) -> list:
    """Estrae in RAM i pesi di una singola epoca dalla traiettoria."""
    return [np.asarray(w[epoch]) for w in consolidated]
