"""lib/secondo_stadio/pseudo_placement.py — stage `pseudo_placement`: dove cadono le
pseudo-anomalie nel piano grezzo dello score (log10 s_DAE, logit_PA).

Per un pool di pseudo generate dai soli benigni-train (swap + radial, ricetta del finalista)
calcola le coordinate da collocare accanto a benigni e attacchi reali (figura del piano 2D):
  - s_DAE(pseudo) = UN-WHITENING ESATTO della componente residuo r_w di phi:
    R = R_w @ L^-1 + mu (il whitening e' R_w = (R-mu) @ L), poi MSE delle 57 binarie
    BIN_SLICE=[34:91] (identica a lib.dae.evaluate / SCORE_DEFINITION). Per lo swap coincide
    con lo s_DAE del donatore benigno; per il radial e' esatto.
  - logit(pseudo) = logit GREZZO pre-sigmoide del PA (pa_raw_logit): evita la saturazione
    float32 della sigmoide, che schiaccerebbe le pseudo confidenti al tetto del clip
    (~27,6 = -ln EPS) perdendo lo spread reale; benigni/attacchi non saturano, quindi
    F.logit(sigmoide) == logit grezzo e gli assi coincidono — coincidenza valida SOTTO il
    tetto di saturazione float32 della sigmoide: a score esattamente 1,0 il logit ricavato
    dalla sigmoide non e' piu' recuperabile e i due assi divergono.

GATE di fedelta' (STOP, non riconciliare): l'un-whitening dei residui benigni deve
recuperare lo s_DAE salvato in cache (max|Delta| <= 1e-4), altrimenti i numeri delle
pseudo non sarebbero affidabili. Port da repo_v5 pseudo_placement.py:33-125 (PHI_Z 6->16,
cache v6 senza matrice benigna dedicata: si usano le posizioni ben_pos).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from lib.secondo_stadio import constants as C
from lib.secondo_stadio.data import assemble_phi, load_phi_cache
from lib.secondo_stadio.generators import generate_fakes_typed
from lib.utils import BIN_SLICE

log = logging.getLogger(__name__)


def pa_raw_logit(model, phi, batch: int = 8192) -> np.ndarray:
    """Logit GREZZO pre-sigmoide: forward fino alla penultima layer, poi @ W_out + b_out
    (dropout = identita' a inference)."""
    W, b = model.get_layer("out").get_weights()
    out = np.empty(len(phi), dtype=np.float64)
    for s in range(0, len(phi), batch):
        x = np.asarray(phi[s:s + batch], np.float32)
        for layer in model.layers[:-1]:               # esclude 'out' (Dense+sigmoid)
            x = layer(x, training=False).numpy()
        out[s:s + len(x)] = (x.astype(np.float64) @ W + b).ravel()
    return out


def load_unwhitening(whiten_npz):
    """mu, L^-1 dal whitening Ledoit-Wolf (R_w = (R-mu) @ L). Inversione offline NumPy."""
    d = np.load(whiten_npz)
    mu = np.asarray(d["mu"], np.float64)          # (91,)
    L = np.asarray(d["L"], np.float64)            # (91,91)
    return mu, np.linalg.inv(L)


def unwhiten_sdae(R_w, mu, Linv) -> np.ndarray:
    """R_w sbiancato (N x 91) -> s_DAE = MSE delle binarie BIN_SLICE del residuo un-whitened."""
    R = np.asarray(R_w, np.float64) @ Linv + mu
    return np.mean(R[:, BIN_SLICE] ** 2, axis=1)


def gate_unwhiten(R_w_benign, sdae_benign, mu, Linv) -> float:
    """max|s_DAE ricostruito - s_DAE salvato| sui benigni passati (atteso <= 1e-4)."""
    s_hat = unwhiten_sdae(np.asarray(R_w_benign, np.float64), mu, Linv)
    return float(np.max(np.abs(s_hat - np.asarray(sdae_benign, np.float64))))


def compute_pseudo_placement(cache_dir, labels_val, pa_weights, model_config: dict,
                             mlp_fixed: dict, seed: int, n_pool: int,
                             gate_n: int = 20_000) -> dict:
    """Genera pseudo (swap+radial, ricetta del finalista) da un sottocampione dei
    benigni-train, ne calcola (s_DAE via un-whitening esatto, logit grezzo del PA), con
    gate di fedelta' preliminare. Ritorna dict: sdae_ps/logit_ps/is_radial + gate_unwhiten/n_pseudo/frac_radial."""
    from lib.secondo_stadio.fp_mining import _load_model

    cache_dir = Path(cache_dir)
    mu, Linv = load_unwhitening(cache_dir / "whitening_params.npz")
    data = load_phi_cache(cache_dir, labels_val)
    ben_train_abs = data["ben_pos"][data["idx_train"]]

    n = min(int(n_pool), len(ben_train_abs))
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(len(ben_train_abs), size=n, replace=False))
    pool_abs = ben_train_abs[sub]
    phi_pool = assemble_phi(data["Zs"], data["R"], pool_abs)
    Zs_pool, R_pool = phi_pool[:, :C.Z_DIM], phi_pool[:, C.Z_DIM:]

    # GATE di fedelta' un-whitening PRIMA di fidarsi dei numeri delle pseudo.
    g_idx = pool_abs[:min(gate_n, len(pool_abs))]
    g = gate_unwhiten(np.asarray(data["R"][g_idx], np.float64),
                      data["dae_score"][g_idx], mu, Linv)
    log.info("[gate] un-whitening benigni: max|s_DAE_ricostruito - salvato| = %.2e "
             "(atteso <= 1e-4)", g)
    if g > 1e-4:
        raise SystemExit(f"GATE un-whitening fallito ({g:.2e} > 1e-4): s_DAE pseudo non "
                         f"affidabile. STOP.")

    phi, is_radial = generate_fakes_typed(Zs_pool, R_pool, np.random.default_rng(seed),
                                          delta=float(model_config["delta"]),
                                          alpha_lo=float(model_config["alpha_lo"]),
                                          alpha_hi=float(model_config["alpha_hi"]))
    sdae_ps = unwhiten_sdae(phi[:, C.Z_DIM:], mu, Linv)

    model = _load_model(pa_weights, model_config, mlp_fixed)
    logit_ps = pa_raw_logit(model, phi)
    log.info("pseudo placement: n=%d frac_radial=%.3f | log10 s_DAE in [%.2f,%.2f] "
             "logit in [%.2f,%.2f]", len(phi), float(is_radial.mean()),
             float(np.log10(sdae_ps).min()), float(np.log10(sdae_ps).max()),
             float(logit_ps.min()), float(logit_ps.max()))
    return {"sdae_ps": sdae_ps.astype(np.float64), "logit_ps": logit_ps.astype(np.float64),
            "is_radial": is_radial, "gate_unwhiten": g, "n_pseudo": int(len(phi)),
            "frac_radial": float(is_radial.mean())}


def run_pseudo_placement_stage(cfg: dict) -> None:
    """Stage `pseudo_placement`: calcola e persiste il placement delle pseudo-anomalie.

    cfg: cache_dir, labels_val, pa_weights, model_config, mlp_fixed, out_dir;
    seed (default 42), n_pool (default 50000). Scrive pseudo_placement.npz
    (sdae_ps, logit_ps, is_radial) + results.json (gate/n/frac_radial/seed)."""
    out_dir = Path(cfg["out_dir"])
    if not out_dir.exists():
        raise FileNotFoundError(f"out_dir non esiste (shell-first): {out_dir}")
    res = compute_pseudo_placement(cfg["cache_dir"], cfg["labels_val"], cfg["pa_weights"],
                                   cfg["model_config"], cfg["mlp_fixed"],
                                   int(cfg.get("seed", 42)), int(cfg.get("n_pool", 50_000)))
    np.savez_compressed(out_dir / "pseudo_placement.npz", sdae_ps=res["sdae_ps"],
                        logit_ps=res["logit_ps"], is_radial=res["is_radial"])
    (out_dir / "results.json").write_text(json.dumps(
        {"gate_unwhiten": res["gate_unwhiten"], "n_pseudo": res["n_pseudo"],
         "frac_radial": res["frac_radial"], "seed": int(cfg.get("seed", 42))}, indent=2))
    log.info("scritti: %s/{pseudo_placement.npz, results.json}", out_dir)
