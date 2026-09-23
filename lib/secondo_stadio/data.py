"""lib/secondo_stadio/data.py — UNICO punto d'ingresso alla cache phi (GATE A).

phi = [z_s (16, GIA' standardizzato), r_w (91, sbiancato Ledoit-Wolf)] in R^107, dalla cache
del primo stadio congelato (runs/dae/output_569, #569 seme 2034).

INVARIANTE anti-doppia-standardizzazione: z_s e r_w sono consumati AS-IS dalla cache. Il
legacy (repo_v5 load_pa_data:174-179) standardizzava Z per-consumer perche' la sua cache
salvava Z grezzo; qui z_s e' gia' standardizzato (z_scaler.npz applicato dal cap2) e
ristandardizzare sarebbe un bug. REGOLA STRUTTURALE: ogni consumatore del pilastro (search,
loao, fp-mining, fusione, placement) passa da load_phi_cache/load_test_context — nessun
np.load autonomo di z_s_*/r_*_w fuori da questo modulo.

Split leak-free (D2/D7/D8, GATE A): benigni 80/20 con default_rng(SPLIT_SEED); per il PA
(attack_split=False) lo stream RNG replica ESATTAMENTE il legacy (permutation -> choice di
att_es); per il ceiling (attack_split=True) si inserisce la permutation degli attacchi fra
le due (stream nuovo, dichiarato). idx_train/idx_select sono RELATIVI a ben_pos (contratto
legacy); att_* sono posizioni ASSOLUTE nel val.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from lib.secondo_stadio import constants as C


def load_tau_dae(l1l2_csv, metric: str = "score_DAE") -> float:
    """Soglia del DAE congelato dalla sorgente unica runs/dae/output_569/l1l2_table.csv
    (riga `metric`, colonna `tau` = p99 sui benigni di validazione). Mai hardcodarla."""
    with open(l1l2_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("metric") == metric:
                return float(row["tau"])
    raise ValueError(f"riga '{metric}' assente in {l1l2_csv}")


def assemble_phi(Zs: np.ndarray, R: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Definizione unica di phi: hstack float32 [z_s, r_w] per le righe `idx` (assolute
    rispetto agli array passati). Materializza SOLO il blocco richiesto (R resta mmap)."""
    Zb = np.asarray(Zs[idx], dtype=np.float32)
    Rb = np.asarray(R[idx], dtype=np.float32)
    return np.hstack([Zb, Rb])


def predict_phi(model, Zs: np.ndarray, R: np.ndarray, indices: np.ndarray,
                batch: int = 8192) -> np.ndarray:
    """Inferenza a blocchi su phi (unifica predict_pa v5:79-87 e _batched_predict v3:142-157):
    assembla phi per blocco di indici (mmap-friendly) e concatena i punteggi sigmoid."""
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
      dae_score (N,): punteggio del DAE per la regola AND (P6/P7);
      ben_pos/att_pos: posizioni ASSOLUTE di benigni/attacchi;
      idx_train/idx_select: split 80/20 dei benigni, RELATIVI a ben_pos, disgiunti;
      att_es: subsample ASSOLUTO per la metrica per-epoca (D7);
      [attack_split=True] att_train/att_select: split 80/20 degli attacchi, ASSOLUTI (D2).
    """
    cache_dir = Path(cache_dir)
    # allow_pickle: array-oggetto di stringhe (classi d'attacco) prodotto dal NOSTRO
    # preprocessing (runs/preprocessing, sorgente fidata di prima parte) — mai input esterno.
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
    dae_score = np.load(cache_dir / "dae_score_val.npy")   # per la regola AND (P6/P7)

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
    """Ritaglia un pool-monitor di `mon_n` benigni da DENTRO `ben_train` (posizioni ASSOLUTE),
    seme FISSO indipendente dal seme del modello. Ritorna (ben_gradient, ben_monitor), disgiunti
    e sorted, ben_gradient = setdiff1d(ben_train, ben_monitor) (il resto). Contratto di
    WP-3b(i)/WP-4_v2 (tools/pa_es_history.py:115-119, tools/pa_wp4v2_repass.py): la stessa
    funzione, chiamata con lo stesso (ben_train, mon_n, seed), deve riprodurre ESATTAMENTE la
    stessa partizione in ogni script che ne ha bisogno (repass, precalc, search) senza dover
    salvare/ricaricare gli indici — determinismo puro via RNG."""
    rng_mon = np.random.default_rng(seed)
    mon_pick = rng_mon.choice(len(ben_train), int(mon_n), replace=False)
    ben_monitor = np.sort(ben_train[mon_pick])
    ben_gradient = np.sort(np.setdiff1d(ben_train, ben_monitor))
    assert len(np.intersect1d(ben_gradient, ben_monitor)) == 0, "monitor/gradient non disgiunti"
    assert len(ben_gradient) + len(ben_monitor) == len(ben_train), "partizione non esaustiva"
    return ben_gradient, ben_monitor


def load_test_context(cache_dir, labels_test) -> dict:
    """Carica la cache phi di TEST (valutazioni finali dichiarate: P6 confirm_on_test,
    P7 fusione). Il test NON entra mai nella ricerca ne' nelle selezioni."""
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
