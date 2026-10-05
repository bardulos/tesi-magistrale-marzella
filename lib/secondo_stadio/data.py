"""Accesso condiviso alla cache phi e costruzione degli split leak-free.

I valori `z_s` e `r_w` sono gia' normalizzati: non ristandardizzarli. I consumer
devono caricare le cache tramite questo modulo. Indici benigni relativi a `ben_pos`;
indici degli attacchi assoluti nella validazione.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from lib.secondo_stadio import constants as C


def load_tau_dae(l1l2_csv, metric: str = "score_DAE") -> float:
    """Legge la soglia DAE dalla riga `metric` di `l1l2_table.csv`."""
    with open(l1l2_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("metric") == metric:
                return float(row["tau"])
    raise ValueError(f"riga '{metric}' assente in {l1l2_csv}")


def assemble_phi(Zs: np.ndarray, R: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Assembla il blocco float32 [z_s, r_w] per gli indici assoluti richiesti."""
    Zb = np.asarray(Zs[idx], dtype=np.float32)
    Rb = np.asarray(R[idx], dtype=np.float32)
    return np.hstack([Zb, Rb])


def predict_phi(model, Zs: np.ndarray, R: np.ndarray, indices: np.ndarray,
                batch: int = 8192) -> np.ndarray:
    """Predice i punteggi sigmoid a blocchi, assemblando phi on demand."""
    indices = np.asarray(indices)
    out = np.empty(len(indices), dtype=np.float32)
    for s in range(0, len(indices), batch):
        idx = indices[s:s + batch]
        phi = assemble_phi(Zs, R, idx)
        out[s:s + len(idx)] = model.predict(phi, verbose=0, batch_size=len(phi)).ravel()
    return out


def load_phi_cache(cache_dir, labels_val, val_frac: float = C.VAL_FRAC,
                   split_seed: int = C.SPLIT_SEED, es_n_attacks: int = C.ES_N_ATTACKS,
                   attack_split: bool = False) -> dict:
    """Carica la cache phi di VALIDAZIONE e costruisce lo split leak-free.

    Ritorna dict con:
      Zs (N,16) float32 in RAM, R (N,91) float32 mmap, val_attack (N,) str;
      dae_score (N,): punteggio del DAE per la regola AND;
      ben_pos/att_pos: posizioni ASSOLUTE di benigni/attacchi;
      idx_train/idx_select: split 80/20 dei benigni, RELATIVI a ben_pos, disgiunti;
      att_es: subsample assoluto usato dalla metrica per-epoca;
      [attack_split=True] att_train/att_select: split 80/20 degli attacchi, con indici assoluti.
    """
    cache_dir = Path(cache_dir)
    # allow_pickle: array-oggetto di stringhe (classi d'attacco) prodotto dalla nostra pipeline
    # (runs/preprocessing per CSE, runs/dae/output_569_unsw* per UNSW), mai input esterno.
    val_attack = np.load(labels_val, allow_pickle=True).astype(str)
    Zs = np.load(cache_dir / "z_s_val.npy")                    # RAM (~180 MB), GIA' standardizzato
    R = np.load(cache_dir / "r_val_w.npy", mmap_mode="r")      # page-cache condivisa tra processi
    if Zs.shape[1] != C.Z_DIM or R.shape[1] != C.R_DIM:
        raise ValueError(f"cache incoerente: z_s {Zs.shape} / r_w {R.shape}, "
                         f"attesi (*,{C.Z_DIM}) e (*,{C.R_DIM})")
    if not (len(val_attack) == len(Zs) == len(R)):
        raise ValueError(f"cardinalita' incoerenti: labels={len(val_attack)} "
                         f"z_s={len(Zs)} r_w={len(R)}")

    mask_b = (val_attack == "Benign")
    ben_pos = np.where(mask_b)[0]
    att_pos = np.where(~mask_b)[0]
    dae_score = np.load(cache_dir / "dae_score_val.npy")   # per la regola AND

    n_ben = len(ben_pos)
    rng_sp = np.random.default_rng(split_seed)
    perm = rng_sp.permutation(n_ben)
    n_train = int(n_ben * (1.0 - val_frac))
    idx_train, idx_select = perm[:n_train], perm[n_train:]
    assert len(np.intersect1d(idx_train, idx_select)) == 0, "split benigni non disgiunto"

    out = {"Zs": Zs, "R": R, "val_attack": val_attack, "dae_score": dae_score,
           "ben_pos": ben_pos, "att_pos": att_pos,
           "idx_train": idx_train, "idx_select": idx_select}

    att_es_pool = att_pos
    if attack_split:
        n_att = len(att_pos)
        perm_a = rng_sp.permutation(n_att)
        n_att_train = int(n_att * (1.0 - val_frac))
        att_train = np.sort(att_pos[perm_a[:n_att_train]])
        att_select = np.sort(att_pos[perm_a[n_att_train:]])
        assert len(np.intersect1d(att_train, att_select)) == 0, "split attacchi non disgiunto"
        out["att_train"], out["att_select"] = att_train, att_select
        att_es_pool = att_select

    if es_n_attacks and es_n_attacks < len(att_es_pool):
        att_es = np.sort(rng_sp.choice(att_es_pool, size=int(es_n_attacks), replace=False))
    else:
        att_es = att_es_pool
    out["att_es"] = att_es
    return out


def partition_gradient_monitor(ben_train: np.ndarray, mon_n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Partiziona `ben_train` in pool gradient e monitor disgiunti e ordinati.

    A parita' di input e seed la partizione e' deterministica.
    """
    rng_mon = np.random.default_rng(seed)
    mon_pick = rng_mon.choice(len(ben_train), int(mon_n), replace=False)
    ben_monitor = np.sort(ben_train[mon_pick])
    ben_gradient = np.sort(np.setdiff1d(ben_train, ben_monitor))
    assert len(np.intersect1d(ben_gradient, ben_monitor)) == 0, "monitor/gradient non disgiunti"
    assert len(ben_gradient) + len(ben_monitor) == len(ben_train), "partizione non esaustiva"
    return ben_gradient, ben_monitor


def load_test_context(cache_dir, labels_test) -> dict:
    """Carica la cache phi di test per la valutazione finale, mai per la ricerca."""
    cache_dir = Path(cache_dir)
    # allow_pickle: v. load_phi_cache (etichette di prima parte, sorgente fidata).
    test_attack = np.load(labels_test, allow_pickle=True).astype(str)
    Zs_t = np.load(cache_dir / "z_s_test.npy")
    R_t = np.load(cache_dir / "r_test_w.npy", mmap_mode="r")
    dae_score_t = np.load(cache_dir / "dae_score_test.npy")
    if not (len(test_attack) == len(Zs_t) == len(R_t) == len(dae_score_t)):
        raise ValueError("cardinalita' test incoerenti fra labels/z_s/r_w/dae_score")
    mask_b = (test_attack == "Benign")
    return {"Zs": Zs_t, "R": R_t, "test_attack": test_attack,
            "dae_score": dae_score_t,
            "ben_pos": np.where(mask_b)[0], "att_pos": np.where(~mask_b)[0]}
