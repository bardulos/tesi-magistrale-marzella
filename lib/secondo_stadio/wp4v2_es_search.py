"""Calibra l'early stopping label-free della PA su pseudo di monitoraggio.

Forward-only sui pesi congelati: per ogni terna di parametri misura Delta_AP al
patience stop sui 19 semi e massimizza la mediana. La curva AUC-PR di riferimento
`history_ap` proviene dai risultati precomputati; `_bce_curve_fast` mantiene
l'allineamento epoche/pesi e il break anticipato. La ricerca usa Ray, Optuna e ASHA
sui semi.
"""
from __future__ import annotations

import logging
import struct
from pathlib import Path

import numpy as np

from lib.search.engine import SearchComponent
from lib.secondo_stadio import constants as C
from lib.secondo_stadio.data import load_phi_cache, partition_gradient_monitor
from lib.secondo_stadio.generators import generate_fakes
from lib.secondo_stadio.metrics import pearson_or_worst
from lib.secondo_stadio.mlp_numpy import load_consolidated_trajectory

log = logging.getLogger(__name__)

_BCE_EPS = 1e-7  # Stesso clipping di metrics.bce.


# ==================================================================== calcolo puro (riusato) ====

def _realistic_es_min_censored(curve, patience: int):
    """Applica early stopping min-mode e indica se la patience e' scattata."""
    best, best_ep, cnt = float("inf"), 0, 0
    for i, v in enumerate(curve):
        v = float(v)
        if v < best:
            best, best_ep, cnt = v, i, 0
        else:
            cnt += 1
            if cnt >= patience:
                return best_ep, True
    return best_ep, False


def precompute_bce_ben(bscores) -> np.ndarray:
    """Precalcola la BCE benigna per epoca senza materializzare l'intera matrice."""
    out = np.empty(bscores.shape[0], dtype=np.float64)
    for e in range(bscores.shape[0]):
        s = np.clip(np.asarray(bscores[e], np.float64), _BCE_EPS, 1.0 - _BCE_EPS)
        out[e] = -np.log(1.0 - s).mean()
    return out


def _bce_curve_fast(pseudo_phi, traj, bce_ben, n_b, n_e, patience=None):
    """Calcola la curva BCE e lo stop realistico sui pesi congelati.

    Usa il benigno precalcolato e si ferma quando scatta la patience. Restituisce
    (curva, epoca migliore, scattato); senza patience calcola tutta la curva.
    """
    assert traj[0].shape[0] == n_e + 1 == bce_ben.shape[0], (
        f"traiettoria pesi/precalc disallineati con n_e={n_e}: pesi={traj[0].shape[0]}, "
        f"bce_ben={bce_ben.shape[0]} (attesi entrambi n_e+1={n_e + 1})")
    X = np.ascontiguousarray(pseudo_phi, dtype=np.float32)
    n_pairs = len(traj) // 2
    bufs = [np.empty((X.shape[0], traj[2 * k].shape[2]), dtype=np.float32) for k in range(n_pairs)]
    n_ps = float(X.shape[0])
    n_tot = float(n_b) + n_ps

    curve = np.empty(n_e, dtype=np.float64)
    best, best_ep, cnt = float("inf"), 0, 0
    for e in range(n_e):
        h = X
        for k in range(n_pairs):
            W = np.ascontiguousarray(traj[2 * k][e + 1])
            b = np.ascontiguousarray(traj[2 * k + 1][e + 1])
            np.matmul(h, W, out=bufs[k])
            bufs[k] += b
            if k < n_pairs - 1:
                np.maximum(bufs[k], np.float32(0.0), out=bufs[k])
            h = bufs[k]
        z = h.ravel().astype(np.float64)
        s = np.empty_like(z)
        pos = z >= 0
        neg = ~pos
        s[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ex = np.exp(z[neg])
        s[neg] = ex / (1.0 + ex)
        s = s.astype(np.float32).astype(np.float64)             # Preserva la parita' float32 con mlp_forward.
        s = np.clip(s, _BCE_EPS, 1.0 - _BCE_EPS)
        bce_ps = -np.log(s).mean()
        v = (n_b * bce_ben[e + 1] + n_ps * bce_ps) / n_tot
        curve[e] = v
        if v < best:
            best, best_ep, cnt = v, e, 0
        else:
            cnt += 1
            if patience is not None and cnt >= patience:
                return curve[:e + 1], best_ep, True
    return curve, best_ep, False


def pseudo_seed_from_params(delta: float, alpha_lo: float, alpha_hi: float) -> int:
    """Deriva un seed deterministico dai tre parametri, indipendente dal trial ID."""
    raw = struct.pack("<ddd", float(delta), float(alpha_lo), float(alpha_hi))
    return int.from_bytes(raw, "little") % (2 ** 31 - 1)


# ==================================================================== contratto componente ====

def define_space(trial) -> dict:
    """Campiona delta, alpha_lo e alpha_gap positivo per costruire alpha_hi."""
    return {
        "alpha_lo": trial.suggest_float("alpha_lo", C.WP4V2_ALPHA_LO_LOW, C.WP4V2_ALPHA_LO_HIGH),
        "alpha_gap": trial.suggest_float(
            "alpha_gap", C.WP4V2_ALPHA_HI_LOW - C.WP4V2_ALPHA_LO_LOW,
            C.WP4V2_ALPHA_HI_HIGH - C.WP4V2_ALPHA_LO_HIGH),
        "delta": trial.suggest_float("delta", C.WP4V2_DELTA_LOW, C.WP4V2_DELTA_HIGH),
    }


def build_parent_args(cfg: dict) -> dict:
    """Carica le curve `history_ap` e precalcola la BCE benigna sul driver per passarle ai worker."""
    repass_dir = Path(cfg["repass_dir"])
    precalc_dir = Path(cfg["precalc_dir"])
    seeds = list(C.SEEDS_PA) + list(C.SEEDS_PA_EXTRA)
    import json

    oracle, bce_ben = {}, {}
    for s in seeds:
        rec = json.loads((repass_dir / f"seed_{s}.json").read_text())
        oracle[s] = np.asarray(rec["history_ap"], dtype=np.float64)
        bscores = np.load(precalc_dir / f"benign_scores_seed_{s}.npy", mmap_mode="r")
        assert bscores.shape[0] == len(oracle[s]) + 1, \
            f"seed {s}: precalc {bscores.shape[0]} righe, atteso {len(oracle[s]) + 1}"
        bce_ben[s] = precompute_bce_ben(bscores)
    log.info("build_parent_args WP-4_v2: oracolo+bce_ben per %d semi precalcolati sul driver", len(seeds))
    return {
        "cache_dir": cfg["cache_dir"],
        "labels_val": cfg["labels_val"],
        "precalc_dir": str(precalc_dir),
        "mon_n": int(cfg.get("mon_n", C.WP4V2_MON_N)),
        "mon_seed_v2": int(cfg.get("mon_seed_v2", C.MON_SEED_V2)),
        "split_seed": int(cfg.get("split_seed", C.SPLIT_SEED)),
        "val_frac": float(cfg.get("val_frac", C.VAL_FRAC)),
        "patience": int(cfg.get("patience", C.MLP_PATIENCE)),
        "oracle": oracle,
        "bce_ben": bce_ben,
    }


def setup(parent_args: dict) -> dict:
    """Carica il pool monitor e apre le traiettorie dei pesi per i semi."""
    mon_n = int(parent_args["mon_n"])
    precalc_dir = Path(parent_args["precalc_dir"])
    seeds = list(C.SEEDS_PA) + list(C.SEEDS_PA_EXTRA)

    sh = load_phi_cache(parent_args["cache_dir"], parent_args["labels_val"],
                        val_frac=parent_args["val_frac"], split_seed=parent_args["split_seed"],
                        es_n_attacks=0, attack_split=False)
    ben_train = sh["ben_pos"][sh["idx_train"]]
    _, mon = partition_gradient_monitor(ben_train, mon_n, int(parent_args["mon_seed_v2"]))
    Zb_mon = np.asarray(sh["Zs"][mon], dtype=np.float32)
    Rb_mon = np.asarray(sh["R"][mon], dtype=np.float32)

    weight_traj = {s: load_consolidated_trajectory(
        precalc_dir / "weights_consolidated" / f"seed_{s}", mmap=True) for s in seeds}
    log.info("setup WP-4_v2: Zb_mon=%s Rb_mon=%s, traiettorie per %d semi",
             Zb_mon.shape, Rb_mon.shape, len(weight_traj))
    return {"Zb_mon": Zb_mon, "Rb_mon": Rb_mon, "weight_traj": weight_traj,
            "mon_n": mon_n, "pseudo": None}


def train_one_seed(config: dict, seed: int, shared_state: dict,
                   parent_args: dict, seed_dir: Path) -> dict:
    """Valuta un seme senza training e misura Delta_AP allo stop realistico."""
    alpha_lo = float(config["alpha_lo"])
    alpha_hi = alpha_lo + float(config["alpha_gap"])
    delta = float(config["delta"])

    if shared_state["pseudo"] is None:
        rng = np.random.default_rng(pseudo_seed_from_params(delta, alpha_lo, alpha_hi))
        shared_state["pseudo"] = generate_fakes(shared_state["Zb_mon"], shared_state["Rb_mon"],
                                                rng, delta, alpha_lo, alpha_hi)
    pseudo = shared_state["pseudo"]

    oracle_s = parent_args["oracle"][seed]
    bce_ben_s = parent_args["bce_ben"][seed]
    traj = shared_state["weight_traj"][seed]
    n_e = len(oracle_s)

    bce_prefix, stop_ep, scattato = _bce_curve_fast(
        pseudo, traj, bce_ben_s, shared_state["mon_n"], n_e, patience=int(parent_args["patience"]))
    delta_ap = float(oracle_s[stop_ep] - np.max(oracle_s))
    pearson = pearson_or_worst(bce_prefix, oracle_s[:len(bce_prefix)])
    return {"seed": int(seed), "delta_ap": delta_ap, "stop_epoch": int(stop_ep),
            "censored": (not scattato), "pearson": float(pearson),
            "alpha_lo": alpha_lo, "alpha_hi": alpha_hi, "delta": delta}


def aggregate(per_seed_results: list, config: dict) -> dict:
    """Aggrega Delta_AP e metriche diagnostiche sui semi completati."""
    deltas = np.array([r["delta_ap"] for r in per_seed_results], dtype=np.float64)
    stops = np.array([r["stop_epoch"] for r in per_seed_results], dtype=np.float64)
    pears = np.array([r["pearson"] for r in per_seed_results], dtype=np.float64)
    last = per_seed_results[-1]
    return {
        "objective": float(np.median(deltas)),
        "delta_ap_median": float(np.median(deltas)),
        "delta_ap_min": float(np.min(deltas)),
        "stop_epoch_median": float(np.median(stops)),
        "stop_epoch_std": float(np.std(stops)),
        "n_censored": int(sum(1 for r in per_seed_results if r["censored"])),
        "pearson_median": float(np.median(pears)),
        "alpha_hi": float(last["alpha_hi"]),
    }


REPORT_EXTRA_KEYS = ("delta_ap_median", "delta_ap_min", "stop_epoch_median", "stop_epoch_std",
                     "n_censored", "pearson_median", "alpha_hi")


def make_wp4v2_es_component(seeds=None) -> SearchComponent:
    seeds_list = list(seeds) if seeds else list(C.SEEDS_PA) + list(C.SEEDS_PA_EXTRA)
    return SearchComponent(
        name="wp4v2_es",
        seeds=seeds_list,
        metric_name="objective",
        define_space=define_space,
        build_parent_args=build_parent_args,
        setup=setup,
        train_one_seed=train_one_seed,
        aggregate=aggregate,
        report_extra_keys=REPORT_EXTRA_KEYS,
        max_t=None,
    )


def run_wp4v2_es_search_stage(cfg: dict) -> dict:
    """Cerca un criterio label-free per l'arresto del classificatore PA."""
    from lib.search.engine import run_search
    component = make_wp4v2_es_component(seeds=cfg.get("seeds"))
    return run_search(component, cfg)
