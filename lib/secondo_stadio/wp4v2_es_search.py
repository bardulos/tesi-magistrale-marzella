"""lib/secondo_stadio/wp4v2_es_search.py — componente WP-4_v2: calibrazione del criterio
d'arresto label-free del PA (pseudo-anomalie di MONITORING), sul motore Ray+Optuna+ASHA condiviso
(lib.search.engine). Forward-only: i pesi del PA #589 sono congelati (Fase 0, per-epoca), qui non
si allena nulla.

Obiettivo del trial (correzione 2026-07-10, end-to-end): per una terna (delta, alpha_lo, alpha_hi)
si generano UNA volta le pseudo-monitor da monitor_v2; per ciascuno dei 19 semi si calcola la curva
BCE-monitor epoca per epoca sui pesi congelati (path veloce `_bce_curve_fast`, equivalente
bit-a-bit al riferimento e con break anticipato allo scatto della patience), si applica lo stop
realistico patience-15 e si misura `Delta_AP = history_ap[stop] - max(history_ap)` (<=0 per
costruzione). OBIETTIVO = mediana dei 19 Delta_AP (direction=maximize: il meno negativo = stop piu'
preciso). L'oracolo `history_ap` viene dalla Fase 0 (JSON), mai ricalcolato qui.

Perche' Ray e non piu' Optuna-puro+JournalStorage (ridisegno 2026-07-11): la Fase 2 e' una search
di iperparametri sul cluster, esattamente il compito di `lib/search/engine.py` (come pa_search).
Il motore da' gratis: distribuzione dei trial sui 5 nodi (ray_address=auto), storage sqlite
centralizzato sul head (leggibile da optuna-dashboard), e il PRUNING ASHA sui semi
(time_attr="seeds_completed", rung a 7 e 14) con `PruningAwareOptunaSearch` (PRUNED veritiero) —
gia' implementato e testato nel progetto. Niente orchestrazione ssh manuale, niente studi shardati.

ALLINEAMENTO EPOCHE (bug trovato e corretto 2026-07-10): la traiettoria pesi e' indicizzata 0..100
(0 = stato PRE-training); `history_ap[e]` (e in 0..99) e' il valore DOPO l'epoca e+1. `_bce_curve_fast`
usa i pesi/precalc all'indice e+1 per allinearsi a `history_ap[e]`: allineamento verificato.

Le parti pure (`precompute_bce_ben`, `_bce_curve_fast`, `_realistic_es_min_censored`) vivono QUI
(lib) e sono riusate da tools/pa_wp4v2_search.py (Fase 1 precalc, Fase 3 validate) — che le importa,
non le duplica. Proprieta' verificate: equivalenza col riferimento, allineamento, e il
comportamento del break rispetto allo stop realistico.
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

_BCE_EPS = 1e-7  # stesso eps di lib.secondo_stadio.metrics.bce (clip per stabilita' log)


# ==================================================================== calcolo puro (riusato) ====

def _realistic_es_min_censored(curve, patience: int):
    """Stop realistico su una curva di loss (min-mode, restore-best-in-finestra) che dice anche SE
    la patience e' scattata entro il cap. Duplicazione DELIBERATA di tools/pa_bce_stopper.py:41-58
    (deliverable validato di WP-3b(i), non si tocca per de-duplicare — memoria di progetto)."""
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
    """BCE della SOLA parte benigna (y=0) per ciascuna epoca 0..n_e: -mean(log(1-clip(s))).
    Indipendente dal trial (benigni-monitor e pesi fissi): si precalcola UNA volta e si riusa in
    ogni trial via la decomposizione pesata della media (v. _bce_curve_fast). Loop per-epoca per
    non materializzare (n_e+1, mon_n) in float64 (~160 MB a 200k, ~320 con le temporanee)."""
    out = np.empty(bscores.shape[0], dtype=np.float64)
    for e in range(bscores.shape[0]):
        s = np.clip(np.asarray(bscores[e], np.float64), _BCE_EPS, 1.0 - _BCE_EPS)
        out[e] = -np.log(1.0 - s).mean()
    return out


def _bce_curve_fast(pseudo_phi, traj, bce_ben, n_b, n_e, patience=None):
    """Curva BCE-monitor epoca per epoca (forward NumPy sui pesi congelati) + stop realistico
    fusi, path veloce equivalente-esatto al riferimento _bce_curve_for_pseudo + _realistic_es_min_
    censored (profilo 2026-07-10: ~2/3 del tempo del riferimento erano allocazioni ripetute
    ~336 MB, non matmul). Tre ingredienti, tutti ESATTI:
      1. forward in-place (np.matmul out=, += bias, maximum in-place) — stessi valori float;
      2. BCE benigna precalcolata (bce_ben): media totale = (n_b*bce_ben + n_ps*bce_ps)/(n_b+n_ps);
      3. break anticipato allo scatto della patience (la logica di stop e' sequenziale).
    Allineamento e+1 (v. docstring del modulo). Ritorna (bce_curve_prefix, stop_ep, scattato): col
    break il prefisso arriva all'epoca dello scatto inclusa; con patience=None calcola la curva
    piena senza scattare (stop_ep=argmin, scattato=False — lo stop va calcolato dal chiamante)."""
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
            W = np.ascontiguousarray(traj[2 * k][e + 1])        # ~0,2 MB dal mmap
            b = np.ascontiguousarray(traj[2 * k + 1][e + 1])
            np.matmul(h, W, out=bufs[k])
            bufs[k] += b
            if k < n_pairs - 1:
                np.maximum(bufs[k], np.float32(0.0), out=bufs[k])
            h = bufs[k]
        z = h.ravel().astype(np.float64)
        s = np.empty_like(z)                                    # sigmoid stabile (no overflow exp)
        pos = z >= 0
        neg = ~pos
        s[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ex = np.exp(z[neg])
        s[neg] = ex / (1.0 + ex)
        s = s.astype(np.float32).astype(np.float64)             # round-trip f32: bit-parita' con
        s = np.clip(s, _BCE_EPS, 1.0 - _BCE_EPS)                # mlp_forward (che ritorna f32)
        bce_ps = -np.log(s).mean()
        v = (n_b * bce_ben[e + 1] + n_ps * bce_ps) / n_tot      # decomposizione pesata della media
        curve[e] = v
        if v < best:
            best, best_ep, cnt = v, e, 0
        else:
            cnt += 1
            if patience is not None and cnt >= patience:
                return curve[:e + 1], best_ep, True
    return curve, best_ep, False


def pseudo_seed_from_params(delta: float, alpha_lo: float, alpha_hi: float) -> int:
    """Seme RNG DETERMINISTICO dai parametri della terna (riproducibile in Fase 3 dati gli stessi
    parametri del vincitore, senza dipendere dal trial_id di Ray). Bit dei tre float64 -> intero
    a 31 bit. Config diverse -> pseudo indipendenti; config identica -> pseudo identiche."""
    raw = struct.pack("<ddd", float(delta), float(alpha_lo), float(alpha_hi))
    return int.from_bytes(raw, "little") % (2 ** 31 - 1)


# ==================================================================== contratto componente ====

def define_space(trial) -> dict:
    """Spazio 3-dim. alpha_hi parametrizzato come alpha_lo + alpha_gap (gap>0): il vincolo
    alpha_hi>alpha_lo e' STRUTTURALE (nessun trial sprecato, niente raise-in-define non ammesso
    dal framework). Range coerenti con le costanti WP4V2: alpha_lo in [1,0; 2,5],
    gap in [0,2; 2,5] -> alpha_hi in [1,2; 5,0] agli estremi; delta (frazione radiale) in [0; 1]."""
    return {
        "alpha_lo": trial.suggest_float("alpha_lo", C.WP4V2_ALPHA_LO_LOW, C.WP4V2_ALPHA_LO_HIGH),
        "alpha_gap": trial.suggest_float(
            "alpha_gap", C.WP4V2_ALPHA_HI_LOW - C.WP4V2_ALPHA_LO_LOW,
            C.WP4V2_ALPHA_HI_HIGH - C.WP4V2_ALPHA_LO_HIGH),
        "delta": trial.suggest_float("delta", C.WP4V2_DELTA_LOW, C.WP4V2_DELTA_HIGH),
    }


def build_parent_args(cfg: dict) -> dict:
    """Gira sul DRIVER (workstation, una volta). Precalcola gli array PICCOLI e indipendenti dal trial e li
    spedisce ai worker via pickle (tune.with_parameters): l'oracolo history_ap (19x100) e la BCE
    benigna per-epoca `bce_ben` (19x101, da precompute_bce_ben sui precalc). Cosi' i worker NON
    leggono i benign_scores (grandi): solo i pesi consolidati (mmap, FS locale del nodo) e i
    monitor da load_phi_cache. Tutto ~30 KB, trascurabile nel pickle."""
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
    """Gira sul worker UNA VOLTA per trainable/trial. Carica load_phi_cache (per i benigni-monitor
    Zb_mon/Rb_mon = base della generazione pseudo) e apre le traiettorie pesi consolidate (mmap,
    leggere: solo header) per tutti i 19 semi. `pseudo` e' None: la generazione avviene al primo
    train_one_seed (dipende dalla config del trial) e si cachea qui (stessa config per tutti i semi)."""
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
    """Valuta UN seme (nessun training): genera le pseudo del trial la prima volta (cache in
    shared_state), calcola la curva BCE-monitor coi pesi congelati del seme, ne ricava lo stop
    realistico e il Delta_AP rispetto al tetto dell'oracolo.

    `seed_dir` e' richiesto dall'interfaccia SearchComponent ma resta inutilizzato: questo
    componente e' forward-only e non scrive alcun artefatto per-seme."""
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
    delta_ap = float(oracle_s[stop_ep] - np.max(oracle_s))              # <=0 per costruzione
    pearson = pearson_or_worst(bce_prefix, oracle_s[:len(bce_prefix)])  # diagnostico (sul prefisso)
    return {"seed": int(seed), "delta_ap": delta_ap, "stop_epoch": int(stop_ep),
            "censored": (not scattato), "pearson": float(pearson),
            "alpha_lo": alpha_lo, "alpha_hi": alpha_hi, "delta": delta}


def aggregate(per_seed_results: list, config: dict) -> dict:
    """objective = mediana dei Delta_AP sui semi fatti finora (parziale ai rung ASHA, finale a 19).
    Diagnostica: mediana/std epoche stop, n_censored, Pearson, alpha_hi. config: interfaccia, non usato."""
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
        max_t=None,   # = len(seeds) = 19; rung ASHA a 7 e 14 (grace=7, rf=2 dal config YAML)
    )


def run_wp4v2_es_search_stage(cfg: dict) -> dict:
    """Entry dello stage 'wp4v2_es_search': componente WP-4_v2 + motore Ray condiviso."""
    from lib.search.engine import run_search
    component = make_wp4v2_es_component(seeds=cfg.get("seeds"))
    return run_search(component, cfg)
