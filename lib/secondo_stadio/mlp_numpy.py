"""lib/secondo_stadio/mlp_numpy.py — I/O pesi per-epoca + forward NumPy-puro del PA (WP-4_v2).

Due responsabilita' pure e testabili, usate dal loop Optuna di Fase 2 (`tools/pa_wp4v2_search.py`)
per evitare TensorFlow nel path caldo (costo + non-thread-safety su 16 processi concorrenti):

  - `save_weights_npz`/`load_weights_npz`: round-trip di `model.get_weights()` <-> `.npz`, port
    IDENTICO di `tools/dae_entropy_contam.py:218-229` (stesso schema chiavi `w0,w1,...`, stesso
    accesso esplicito per chiave — non per ordine lessicografico dei file interni all'archivio,
    che romperebbe l'ordine con >=10 array: "w10" < "w2" lessicograficamente). Non importato da
    li' per la separazione libreria/driver del progetto: `dae_entropy_contam.py` e' un driver
    in tools/ (oggi senza side-effect all'import: parse_args() vive dentro main()), e questa
    logica serve come libreria. Duplicazione minima (~8 righe) accettata deliberatamente
    (grandfathered, l'originale resta) piuttosto che rendere importabile un driver CLI.

  - `mlp_forward`: forward puro di un MLP costruito da `lib.secondo_stadio.model.build_mlp`
    (Dense-ReLU per ogni hidden, Dense-sigmoid in uscita). I layer Dropout non hanno pesi propri
    (`get_weights()` non li include), quindi sono no-op sia in Keras a `training=False` sia qui:
    la lista pesi e' semplicemente una sequenza di coppie [kernel_k, bias_k] consecutive, una per
    ogni Dense (hidden + uscita), indipendentemente da `n_layers`. Equivalenza numerica con
    `model.predict(X, training=False)` verificata: scarto <=1e-5 rispetto al forward Keras. E' una
    proprieta' vincolante, non un dettaglio — tutta la Fase 2 (migliaia di forward per trial) gira su
    questa funzione, mai su TF.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_weights_npz(weights: list, path) -> None:
    """Salva una lista di array (l'output di model.get_weights()) in un .npz con chiavi
    w0,w1,... (ordine preservato). NumPy-puro, indipendente da TF: il round-trip con
    load_weights_npz ricostruisce la lista identica, pronta per model.set_weights().
    Port identico di tools/dae_entropy_contam.py:218-222."""
    np.savez(str(path), **{f"w{i}": np.asarray(w) for i, w in enumerate(weights)})


def load_weights_npz(path) -> list:
    """Inverso di save_weights_npz: ritorna [w0, w1, ...] nell'ordine originale (accesso per
    chiave esplicita, non per ordinamento lessicografico dei file interni all'archivio).
    Port identico di tools/dae_entropy_contam.py:225-229."""
    with np.load(str(path)) as d:
        return [d[f"w{i}"] for i in range(len(d.files))]


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """Sigmoid numericamente stabile (nessun overflow di exp per |x| grande), calcolo in
    float64 poi cast a float32 dal chiamante — coerente con la clip in _bce di pa_es_history.py."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def mlp_forward(weights: list, X: np.ndarray) -> np.ndarray:
    """Forward NumPy-puro di un MLP costruito da build_mlp: Dense+ReLU per ogni hidden layer,
    Dense+sigmoid in uscita, dropout no-op (coerente con model.predict, sempre a training=False).
    `weights` e' la lista get_weights() (coppie [kernel,bias] consecutive, una per ogni Dense;
    i Dropout non hanno pesi propri e non compaiono). Ritorna un vettore (N,) di score sigmoid,
    stesso output di model.predict(X, batch_size=..., verbose=0).ravel() a meno di errore
    floating-point BLAS-vs-TF (scarto verificato <=1e-5)."""
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
    """Carica la traiettoria di pesi e0..e{n_epochs} da un callback salva-pesi (WeightSaverCb nei
    driver entropy, RepassCb per il PA; weights_epoch_{NNN:03d}.npz, e0 = epoca 000 pre-gradiente).
    Ritorna una lista di lunghezza n_epochs+1, indice e -> load_weights_npz(epoch e)."""
    weights_dir = Path(weights_dir)
    return [load_weights_npz(weights_dir / f"weights_epoch_{e:03d}.npz") for e in range(n_epochs + 1)]


def consolidate_weight_trajectory(weights_dir, out_dir, n_epochs: int) -> int:
    """Consolida la traiettoria di pesi e0..e{n_epochs} (tanti .npz quante le epoche+1) in tensori
    impilati per-indice-di-array, UN file .npy per indice (w0.npy, w1.npy, ...), shape
    (n_epochs+1, *forma_originale). Formato memmap-abile (np.load(..., mmap_mode='r')), a
    differenza del .npz (contenitore zip): evita di riaprire migliaia di piccoli .npz nel loop di
    ricerca di Fase 2 (WP-4_v2), dove la stessa traiettoria e' letta ripetutamente da molti
    processi concorrenti. Ritorna n_pairs (meta' del numero di array per epoca: kernel+bias
    per ogni Dense)."""
    traj = load_weight_trajectory(weights_dir, n_epochs)          # [[w0,w1,...], ...] len n_epochs+1
    n_arrays = len(traj[0])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_arrays):
        stacked = np.stack([epoch_w[i] for epoch_w in traj], axis=0)
        np.save(out_dir / f"w{i}.npy", stacked)
    return n_arrays // 2


def load_consolidated_trajectory(out_dir, mmap: bool = True) -> list:
    """Inverso di consolidate_weight_trajectory: ritorna [w0_stack, w1_stack, ...], ciascuno di
    shape (n_epochs+1, *forma). mmap=True apre in sola lettura via mmap_mode='r' (pagine condivise
    dalla page-cache del SO fra i processi concorrenti del loop Optuna). Ordinamento per indice
    NUMERICO (w2 < w10), non lessicografico sul nome file."""
    out_dir = Path(out_dir)
    paths = sorted(out_dir.glob("w*.npy"), key=lambda p: int(p.stem[1:]))
    mode = "r" if mmap else None
    return [np.load(p, mmap_mode=mode) for p in paths]


def weights_at_epoch(consolidated: list, epoch: int) -> list:
    """Estrae la lista pesi (pronta per mlp_forward) all'epoca `epoch` da una traiettoria
    consolidata (v. load_consolidated_trajectory): una fetta per array, materializzata in RAM
    (piccola: ~0,6 MB totali per il PA #589, 154.237 parametri)."""
    return [np.asarray(w[epoch]) for w in consolidated]
