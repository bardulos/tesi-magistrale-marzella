#!/usr/bin/env python
# TAG: ONE-SHOT | 2026-07-13 | WP-4_v2 Fasi 1-3 (testato) | vedi docs/refactoring_censimento.md
"""tools/pa_wp4v2_search.py — WP-4_v2 Fasi 1-3: ricerca dei parametri delle pseudo-monitor del PA.

Un solo driver a sottocomandi (KISS, stessa organizzazione del driver della ricerca WP-4 precedente):

  precalc  (Fase 1, una tantum, TF-free) — consolida i pesi per-epoca salvati da
           tools/pa_wp4v2_repass.py (19 semi x 101 epoche, .npz) in tensori .npy impilati
           per-seme (memmap-abili — .npz NON lo e', v. lib.secondo_stadio.mlp_numpy), poi
           forward NumPy-puro dei 200k benigni-monitor su OGNI epoca di OGNI seme -> uno
           score-array (n_epochs+1, mon_n) per seme, salvato .npy. Questo e' l'UNICO costo
           che NON dipende dai parametri delle pseudo (i benigni-monitor sono fissi): si
           calcola una volta, si riusa in ogni trial di Fase 2.

  search   (Fase 2, worker Optuna) — JournalStorage locale al nodo, sequenziale nel processo;
           il fan-out a 16 processi usa un lanciatore di cluster non incluso, descritto in
           tools/README.txt. Per ogni trial (delta, alpha_lo,
           alpha_hi): genera 200k pseudo-monitor UNA volta (seed deterministico dal
           trial.number), poi per ciascuno dei 19 semi (ordine FISSO) calcola la BCE
           benigni-monitor-vs-pseudo epoca per epoca (forward NumPy-puro sui pesi congelati
           precalcolati, path ottimizzato _bce_curve_fast con break anticipato allo stop) e
           applica lo stop realistico patience-15. Obiettivo (correzione 2026-07-10) =
           mediana dei 19 Delta AP al punto di stop, history_ap[stop] - max(history_ap)
           (direction=maximize: il meno negativo e' il migliore); l'oracolo history_ap e' gia'
           nei JSON di Fase 0, mai ricalcolato qui. Pruning SuccessiveHalving sui semi
           (rung a 7 e 14, v. lib.secondo_stadio.constants.WP4V2_RUNG_1/2).

  validate (Fase 3, dopo la ricerca) — con la configurazione vincente: dual-axis per-seme
           (CSV, dati grezzi per il plot), stop patience-15 sulla BCE-monitor (realistic_es
           riusata dalla fase di analisi WP-3b, MAI ri-validata/modificata) confrontato col
           tetto (argmax history_ap) e col cap-epoche fisso.

Anti-circolarita' (stesso invariante della ricerca WP-4 precedente): la ricerca ottimizza i
PARAMETRI delle pseudo, il PA #589 e i suoi pesi (Fase 0, congelati) non cambiano; nessun
training qui, solo forward.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # per `from pa_es_analysis import ...`

from lib.utils import setup_blas_env  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("precalc", help="Fase 1: consolida pesi + forward benigni-monitor")
    pp.add_argument("--repass-dir", default=str(REPO / "runs/secondo_stadio/wp4v2_repass"))
    pp.add_argument("--out-dir", default=str(REPO / "runs/secondo_stadio/wp4v2_search/precalc"))
    pp.add_argument("--cache-dir", default=str(REPO / "runs/dae/output_569"))
    pp.add_argument("--labels-val", default=str(REPO / "runs/preprocessing/val_attack_npy/val_Attack.npy"))
    pp.add_argument("--mon-n", type=int, default=None)
    pp.add_argument("--threads", type=int, default=8)

    ps = sub.add_parser("search", help="Fase 2: worker Optuna (fan-out: pa_wp4v2_run.sh)")
    ps.add_argument("--repass-dir", default=str(REPO / "runs/secondo_stadio/wp4v2_repass"))
    ps.add_argument("--precalc-dir", default=str(REPO / "runs/secondo_stadio/wp4v2_search/precalc"))
    ps.add_argument("--cache-dir", default=str(REPO / "runs/dae/output_569"))
    ps.add_argument("--labels-val", default=str(REPO / "runs/preprocessing/val_attack_npy/val_Attack.npy"))
    ps.add_argument("--storage", required=True, help="path del JournalStorage (file .log)")
    ps.add_argument("--n-trials", type=int, default=125)
    ps.add_argument("--worker-id", type=int, default=0,
                    help="id GLOBALE del worker (unico anche fra nodi: offset per-VPS in "
                         "pa_wp4v2_run.sh) — entra nel seed del TPESampler")
    ps.add_argument("--n-startup", type=int, default=None,
                    help="n_startup_trials del TPE (default: WP4V2_N_STARTUP_TRIALS da "
                         "constants; ridurre per studi per-nodo piccoli)")
    ps.add_argument("--threads", type=int, default=1)

    pv = sub.add_parser("validate", help="Fase 3: dual-axis + patience-15 + confronto cap-epoche")
    pv.add_argument("--repass-dir", default=str(REPO / "runs/secondo_stadio/wp4v2_repass"))
    pv.add_argument("--precalc-dir", default=str(REPO / "runs/secondo_stadio/wp4v2_search/precalc"))
    pv.add_argument("--cache-dir", default=str(REPO / "runs/dae/output_569"))
    pv.add_argument("--labels-val", default=str(REPO / "runs/preprocessing/val_attack_npy/val_Attack.npy"))
    pv.add_argument("--storage", nargs="+", default=None,
                    help="uno o PIU' JournalStorage da cui leggere il best trial (con piu' file "
                         "— studi per-nodo shardati, un journal per VPS — il best e' globale)")
    pv.add_argument("--delta", type=float, default=None)
    pv.add_argument("--alpha-lo", type=float, default=None)
    pv.add_argument("--alpha-hi", type=float, default=None)
    pv.add_argument("--cap-grid", type=int, nargs="+", default=[15, 20, 25])
    pv.add_argument("--out-dir", default=str(REPO / "runs/secondo_stadio/wp4v2_search"))
    pv.add_argument("--threads", type=int, default=4)
    return p


# NB import: niente argparse/setup_blas_env a livello di modulo (a differenza di
# pa_wp4v2_repass.py, un puro driver di training mai importato altrove) — questo file espone
# funzioni riusate e verificate direttamente, mirror dell'organizzazione di
# tools/dae_entropy_contam.py e tools/synth_search.py: `import numpy` a livello di modulo e' safe
# (nessun side-effect), setup_blas_env si chiama in main() PRIMA del lavoro numerico pesante
# (stesso pattern di dae_entropy_contam.py:51,438 — numpy importato in testa, setup_blas_env
# richiamato piu' in basso, dentro le funzioni che fanno il lavoro vero).
import numpy as np  # noqa: E402

from lib.secondo_stadio import constants as C                       # noqa: E402
from lib.secondo_stadio.data import (assemble_phi, load_phi_cache,  # noqa: E402
                                     partition_gradient_monitor)
from lib.secondo_stadio.generators import generate_fakes, pseudo_seed_for_trial  # noqa: E402
from lib.secondo_stadio.metrics import (bce, best_mcc_bal_standalone,  # noqa: E402
                                        pearson_or_worst)
from lib.secondo_stadio.mlp_numpy import (consolidate_weight_trajectory,  # noqa: E402
                                          load_consolidated_trajectory,
                                          mlp_forward, weights_at_epoch)
# path caldo spostato in lib (2026-07-11) perche' ora riusato dal componente Ray
# lib.secondo_stadio.wp4v2_es_search (lib non puo' importare da tools). Qui re-importati per la
# Fase 1 (precalc)/Fase 3 (validate) e per i test che li leggono da questo modulo.
from lib.secondo_stadio.wp4v2_es_search import (_bce_curve_fast,  # noqa: E402,F401
                                                _realistic_es_min_censored,
                                                precompute_bce_ben, pseudo_seed_from_params)


def _load_monitor_pool(a, mon_n: int):
    """Ricarica Zs/R + ben_monitor (v2) con lo STESSO contratto di pa_wp4v2_repass.py: stesso
    load_phi_cache, stessa partition_gradient_monitor(ben_train, mon_n, MON_SEED_V2) -> stessa
    partizione, senza dover salvare/ricaricare gli indici (determinismo puro via RNG). Il dict
    `sh` ritornato contiene anche ben_pos/idx_select/att_pos (dataset PIENO di cap3), usati da
    cmd_validate per delta_mccbal_median."""
    sh = load_phi_cache(a.cache_dir, a.labels_val, val_frac=C.VAL_FRAC,
                        split_seed=C.SPLIT_SEED, es_n_attacks=0, attack_split=False)
    ben_train = sh["ben_pos"][sh["idx_train"]]
    _, ben_monitor = partition_gradient_monitor(ben_train, mon_n, C.MON_SEED_V2)
    return sh, ben_monitor


# _realistic_es_min_censored: importata da lib.secondo_stadio.wp4v2_es_search (sopra).


# ------------------------------------------------------------------ precalc (Fase 1)
def cmd_precalc(a) -> None:
    seeds = list(C.SEEDS_PA) + list(C.SEEDS_PA_EXTRA)
    repass_dir = Path(a.repass_dir)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mon_n = int(a.mon_n) if a.mon_n is not None else int(C.WP4V2_MON_N)

    sh, ben_monitor = _load_monitor_pool(a, mon_n)
    X_monitor = assemble_phi(sh["Zs"], sh["R"], ben_monitor)   # (mon_n, 107) float32, UNA volta

    manifest = {"seeds": seeds, "mon_n": int(mon_n), "mon_seed_v2": int(C.MON_SEED_V2), "per_seed": {}}
    t0 = time.time()
    for s in seeds:
        seed_json_path = repass_dir / f"seed_{s}.json"
        if not seed_json_path.exists():
            print(f"  [SALTA] seed {s}: {seed_json_path} assente (Fase 0 incompleta per questo seme)")
            continue
        seed_rec = json.loads(seed_json_path.read_text())
        n_epochs = int(seed_rec["n_epochs"])
        weights_seed_dir = repass_dir / "weights" / f"seed_{s}"
        consolidated_dir = out_dir / "weights_consolidated" / f"seed_{s}"
        n_pairs = consolidate_weight_trajectory(weights_seed_dir, consolidated_dir, n_epochs)

        traj = load_consolidated_trajectory(consolidated_dir, mmap=True)
        scores = np.empty((n_epochs + 1, len(ben_monitor)), dtype=np.float32)
        for e in range(n_epochs + 1):
            scores[e] = mlp_forward(weights_at_epoch(traj, e), X_monitor)
        np.save(out_dir / f"benign_scores_seed_{s}.npy", scores)
        manifest["per_seed"][str(s)] = {"n_epochs": n_epochs, "n_pairs": n_pairs, "n_monitor": len(ben_monitor)}
        print(f"  seed {s}: {n_epochs + 1} epoche x {n_pairs} Dense -> "
              f"benign_scores_seed_{s}.npy {scores.shape}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    print(f"PRECALC completo: {len(manifest['per_seed'])}/{len(seeds)} semi in "
          f"{time.time() - t0:.1f}s -> {out_dir}")


# ------------------------------------------------------------------ search (Fase 2, worker Optuna)
def _bce_curve_for_pseudo(pseudo_phi, y_mon, traj, bscores, n_e):
    """BCE benigni-monitor-vs-pseudo per ogni epoca, per UN seme. ALLINEAMENTO (bug trovato e
    corretto 2026-07-10): ap_curve[e]/bce_curve[e] (e in range(n_e), n_e=len(history_ap)=100) e'
    il valore DOPO l'epoca e+1 (costruzione del callback Keras on_epoch_end(epoch=e, ...), epoch
    0-based) — i pesi/i logit benigni corrispondenti sono all'indice e+1 della traiettoria
    (indice 0 = stato PRE-training, salvato prima di model.fit). Usare weights_at_epoch(traj, e)
    o bscores[e] qui sarebbe lo sfasamento di un'epoca gia' trovato e corretto: la BCE dell'epoca
    N andrebbe confrontata con l'AUC-PR dell'epoca N-1 su tutta la curva."""
    assert traj[0].shape[0] == n_e + 1 == bscores.shape[0], (
        f"traiettoria pesi/precalc disallineati con n_e={n_e}: pesi={traj[0].shape[0]}, "
        f"precalc={bscores.shape[0]} (attesi entrambi n_e+1={n_e + 1})")
    bce_curve = np.empty(n_e, dtype=np.float64)
    for e in range(n_e):
        s_ps = mlp_forward(weights_at_epoch(traj, e + 1), pseudo_phi)
        bce_curve[e] = bce(y_mon, np.r_[bscores[e + 1], s_ps])
    return bce_curve


# precompute_bce_ben, _bce_curve_fast: importati da lib.secondo_stadio.wp4v2_es_search (sopra).
# _bce_curve_for_pseudo (riferimento per l'equivalenza) resta QUI, usato dai test e da validate.


def build_objective(seeds, oracle, bce_ben, weight_traj, n_b, Zb_mon, Rb_mon):
    """Obiettivo end-to-end (correzione 2026-07-10, sostituisce Pearson(BCE, history_ap)): la
    curva history_ap e' piatta dopo l'epoca ~15 (Delta ceiling->finale mediano -0,0014 su 19
    semi, misurato sul dataset pieno) — 85 epoche di plateau diluiscono qualunque Pearson, il TPE
    non avrebbe gradiente da seguire. L'obiettivo diventa la qualita' dello STOP che la BCE-monitor
    produce nella stessa identica configurazione di deploy (patience-15, restore-best): quanto
    vicino al tetto (argmax AUC-PR) si ferma il training seguendo SOLO la BCE-monitor. Target =
    AUC-PR (non MCC_bal): GATE A del cap3 separa i ruoli, MCC_bal e' la metrica di SELEZIONE fra
    config, AUC-PR e' la metrica di EARLY-STOPPING per-epoca — WP-4_v2 sostituisce label-free il
    secondo ruolo, non il primo."""
    import optuna  # import esplicito (non ereditato dalla closure di cmd_search: objective e'
                   # richiamata da Optuna dentro study.optimize, serve il nome nel suo modulo)

    def objective(trial):
        alpha_lo = trial.suggest_float("alpha_lo", C.WP4V2_ALPHA_LO_LOW, C.WP4V2_ALPHA_LO_HIGH)
        alpha_hi = trial.suggest_float("alpha_hi", C.WP4V2_ALPHA_HI_LOW, C.WP4V2_ALPHA_HI_HIGH)
        delta = trial.suggest_float("delta", C.WP4V2_DELTA_LOW, C.WP4V2_DELTA_HIGH)
        if alpha_hi <= alpha_lo:
            raise optuna.TrialPruned(f"alpha_hi({alpha_hi:.4f}) <= alpha_lo({alpha_lo:.4f})")

        rng = np.random.default_rng(pseudo_seed_for_trial(trial.number, C.PSEUDO_MONITOR_SEED))
        pseudo_phi = generate_fakes(Zb_mon, Rb_mon, rng, delta, alpha_lo, alpha_hi)  # (mon_n,107), UNA volta per trial

        deltas_ap, pearsons, stop_epochs, censored = [], [], [], []
        for k, s in enumerate(seeds):
            ap_curve = oracle[s]
            n_e = len(ap_curve)
            bce_prefix, stop_ep, scattato = _bce_curve_fast(
                pseudo_phi, weight_traj[s], bce_ben[s], n_b, n_e, patience=C.MLP_PATIENCE)

            # Pearson diagnostico sul PREFISSO fino allo scatto (col break anticipato la coda
            # non e' calcolata): confronto retrospettivo col vecchio obiettivo, mai nell'obiettivo.
            pearsons.append(pearson_or_worst(bce_prefix, ap_curve[:len(bce_prefix)]))
            stop_epochs.append(stop_ep)
            censored.append(not scattato)
            deltas_ap.append(float(ap_curve[stop_ep] - np.max(ap_curve)))  # <=0 per costruzione

            n_done = k + 1
            if n_done in (int(C.WP4V2_RUNG_1), int(C.WP4V2_RUNG_2)):
                trial.report(float(np.median(deltas_ap)), step=n_done)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        med_delta = float(np.median(deltas_ap))
        trial.set_user_attr("pearson_bce_ap_median", float(np.median(pearsons)))
        trial.set_user_attr("stop_epoch_median", float(np.median(stop_epochs)))
        trial.set_user_attr("stop_epoch_std", float(np.std(stop_epochs)))
        trial.set_user_attr("n_censored", int(sum(censored)))
        trial.set_user_attr("n_seeds", len(deltas_ap))
        return med_delta  # direction="maximize": il meno negativo (piu' vicino a 0) e' il migliore
    return objective


def cmd_search(a) -> None:
    import optuna
    from optuna.pruners import SuccessiveHalvingPruner
    from optuna.samplers import TPESampler
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    seeds = tuple(list(C.SEEDS_PA) + list(C.SEEDS_PA_EXTRA))
    repass_dir, precalc_dir = Path(a.repass_dir), Path(a.precalc_dir)
    manifest = json.loads((precalc_dir / "manifest.json").read_text())
    mon_n = int(manifest["mon_n"])
    assert int(manifest["mon_seed_v2"]) == int(C.MON_SEED_V2), \
        "precalc con MON_SEED_V2 diverso da quello corrente in constants.py — rilanciare precalc"

    oracle, bce_ben, weight_traj = {}, {}, {}
    t0 = time.time()
    for s in seeds:
        seed_rec = json.loads((repass_dir / f"seed_{s}.json").read_text())
        oracle[s] = np.asarray(seed_rec["history_ap"], dtype=np.float64)
        benign_scores = np.load(precalc_dir / f"benign_scores_seed_{s}.npy", mmap_mode="r")
        weight_traj[s] = load_consolidated_trajectory(precalc_dir / "weights_consolidated" / f"seed_{s}", mmap=True)
        n_e = len(oracle[s])
        assert weight_traj[s][0].shape[0] == n_e + 1 == benign_scores.shape[0], \
            f"seed {s}: n_epochs disallineati fra JSON({n_e})/pesi({weight_traj[s][0].shape[0]-1})/" \
            f"precalc({benign_scores.shape[0]-1})"
        assert benign_scores.shape[1] == mon_n, \
            f"seed {s}: precalc con mon_n={benign_scores.shape[1]} != manifest {mon_n}"
        # BCE benigna per epoca, indipendente dal trial: precalcolata QUI una volta per worker,
        # ogni trial paga solo il forward delle proprie pseudo (v. _bce_curve_fast).
        bce_ben[s] = precompute_bce_ben(benign_scores)
    print(f"[worker {a.worker_id}] startup: oracle+pesi+bce_ben per {len(seeds)} semi "
          f"in {time.time() - t0:.1f}s", flush=True)

    sh, ben_monitor = _load_monitor_pool(a, mon_n)
    Zb_mon = np.asarray(sh["Zs"][ben_monitor], dtype=np.float32)
    Rb_mon = np.asarray(sh["R"][ben_monitor], dtype=np.float32)

    objective = build_objective(seeds, oracle, bce_ben, weight_traj, mon_n, Zb_mon, Rb_mon)

    storage = JournalStorage(JournalFileBackend(a.storage))
    sampler_seed = int(C.WP4V2_OPTUNA_SEED) + int(a.worker_id)  # +worker_id: evita first-sample
    n_startup = int(a.n_startup) if a.n_startup is not None else int(C.WP4V2_N_STARTUP_TRIALS)
    sampler = TPESampler(seed=sampler_seed, multivariate=True, group=True, constant_liar=True,
                         n_startup_trials=n_startup)
    pruner = SuccessiveHalvingPruner(min_resource=int(C.WP4V2_RUNG_1), reduction_factor=2,
                                     min_early_stopping_rate=0)
    study = optuna.create_study(study_name="wp4v2_synth", storage=storage, direction="maximize",
                                sampler=sampler, pruner=pruner, load_if_exists=True)
    study.optimize(objective, n_trials=int(a.n_trials))
    print(f"[worker {a.worker_id}] {a.n_trials} trial completati su {a.storage}")


# ------------------------------------------------------------------ validate (Fase 3)
def cmd_validate(a) -> None:
    import csv

    seeds = tuple(list(C.SEEDS_PA) + list(C.SEEDS_PA_EXTRA))
    repass_dir, precalc_dir = Path(a.repass_dir), Path(a.precalc_dir)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.delta is not None and a.alpha_lo is not None and a.alpha_hi is not None:
        delta, alpha_lo, alpha_hi = float(a.delta), float(a.alpha_lo), float(a.alpha_hi)
        trial_number = None
    else:
        if not a.storage:
            raise SystemExit("validate: servono --delta/--alpha-lo/--alpha-hi OPPURE --storage "
                             "(per leggere il best trial dallo studio Optuna)")
        import optuna
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend
        completed = []                       # (trial, storage_path) da TUTTI gli storage indicati
        for st_path in a.storage:
            study = optuna.load_study(study_name="wp4v2_synth",
                                      storage=JournalStorage(JournalFileBackend(st_path)))
            completed += [(t, st_path) for t in study.trials
                          if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            raise SystemExit(f"validate: nessun trial COMPLETE in {a.storage}")
        best, best_storage = max(completed, key=lambda ts: ts[0].value)  # maximize (mediana Delta AP)
        delta, alpha_lo, alpha_hi = best.params["delta"], best.params["alpha_lo"], best.params["alpha_hi"]
        trial_number = best.number           # per-studio: con piu' storage identifica il trial
        print(f"best GLOBALE su {len(a.storage)} storage ({len(completed)} COMPLETE): "
              f"trial #{trial_number} di {best_storage}")
        print(f"  obj(mediana Delta AP)={best.value:+.5f} "
              f"delta={delta:.4f} alpha_lo={alpha_lo:.4f} alpha_hi={alpha_hi:.4f}")

    manifest = json.loads((precalc_dir / "manifest.json").read_text())
    mon_n = int(manifest["mon_n"])
    sh, ben_monitor = _load_monitor_pool(a, mon_n)
    Zb_mon = np.asarray(sh["Zs"][ben_monitor], dtype=np.float32)
    Rb_mon = np.asarray(sh["R"][ben_monitor], dtype=np.float32)
    # dataset PIENO di cap3 (ben_select + tutti gli attacchi), per delta_mccbal_median — deferito
    # qui (non per-trial in Fase 2, v. piano) perche' costoso: ~6,3s/valutazione misurati.
    ben_select = sh["ben_pos"][sh["idx_select"]]
    att_pos = sh["att_pos"]
    Xb_sel = assemble_phi(sh["Zs"], sh["R"], ben_select)
    Xa_full = assemble_phi(sh["Zs"], sh["R"], att_pos)

    # Fase 3 sul vincitore Ray: stesso seme pseudo del componente wp4v2_es (pseudo_seed_from_params
    # dai parametri), cosi' le pseudo di validate coincidono con quelle del trial vincente. Il ramo
    # legacy (pseudo_seed_for_trial) resta per il vecchio storage Optuna-puro con trial_number.
    pseudo_seed = (pseudo_seed_for_trial(trial_number, C.PSEUDO_MONITOR_SEED)
                  if trial_number is not None else pseudo_seed_from_params(delta, alpha_lo, alpha_hi))
    rng = np.random.default_rng(pseudo_seed)
    pseudo_phi = generate_fakes(Zb_mon, Rb_mon, rng, delta, alpha_lo, alpha_hi)

    rows, bce_curves = [], {}
    for s in seeds:
        seed_rec = json.loads((repass_dir / f"seed_{s}.json").read_text())
        ap_curve = np.asarray(seed_rec["history_ap"], dtype=np.float64)
        bscores = np.load(precalc_dir / f"benign_scores_seed_{s}.npy", mmap_mode="r")
        traj = load_consolidated_trajectory(precalc_dir / "weights_consolidated" / f"seed_{s}", mmap=True)
        n_e = len(ap_curve)
        # curva PIENA (patience=None: qui serve tutta, per dual-axis e Pearson non troncato);
        # path veloce equivalente a _bce_curve_for_pseudo (stesso allineamento e+1, v. docstring)
        bce_curve, _, _ = _bce_curve_fast(pseudo_phi, traj, precompute_bce_ben(bscores),
                                          bscores.shape[1], n_e, patience=None)
        bce_curves[str(s)] = bce_curve.tolist()  # dati grezzi per il plot dual-axis (3a), FUORI dal CSV

        pearson = pearson_or_worst(bce_curve, ap_curve)
        stop_epoch, scattato = _realistic_es_min_censored(bce_curve, C.MLP_PATIENCE)
        best_epoch = int(np.argmax(ap_curve))
        ap_at_stop, best_ap = float(ap_curve[stop_epoch]), float(ap_curve[best_epoch])
        delta_ap = ap_at_stop - best_ap
        cap_ap = {cap: float(ap_curve[min(cap, n_e - 1)]) for cap in a.cap_grid}

        # delta_mccbal_median: forward sul dataset PIENO a stop_epoch e best_epoch (indice pesi
        # +1, stesso allineamento di sopra), best_mcc_bal_standalone su ben_select vs att_pos.
        w_stop = weights_at_epoch(traj, stop_epoch + 1)
        w_ceil = weights_at_epoch(traj, best_epoch + 1)
        mb_stop = best_mcc_bal_standalone(mlp_forward(w_stop, Xb_sel), mlp_forward(w_stop, Xa_full))
        mb_ceil = best_mcc_bal_standalone(mlp_forward(w_ceil, Xb_sel), mlp_forward(w_ceil, Xa_full))
        delta_mccbal = mb_stop["mcc_bal"] - mb_ceil["mcc_bal"]

        rows.append({"seed": s, "n_epochs": n_e, "pearson": round(pearson, 5),
                    "stop_epoch": stop_epoch, "stop_scattato": int(scattato), "best_epoch": best_epoch,
                    "ap_at_stop": round(ap_at_stop, 5), "best_ap": round(best_ap, 5),
                    "delta_ap": round(delta_ap, 5),
                    "mccbal_at_stop": round(mb_stop["mcc_bal"], 5),
                    "mccbal_at_ceiling": round(mb_ceil["mcc_bal"], 5),
                    "delta_mccbal": round(delta_mccbal, 5),
                    **{f"ap_at_cap{cap}": round(v, 5) for cap, v in cap_ap.items()}})

    csv_path = out_dir / "validate.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "validate_bce_curves.json").write_text(json.dumps(
        {"params": {"delta": delta, "alpha_lo": alpha_lo, "alpha_hi": alpha_hi},
         "pseudo_seed": pseudo_seed, "bce_curves": bce_curves}))

    deltas_ap = np.array([r["delta_ap"] for r in rows])
    deltas_mccbal = np.array([r["delta_mccbal"] for r in rows])
    n_censored = sum(1 for r in rows if not r["stop_scattato"])
    pearsons = np.array([r["pearson"] for r in rows])
    print("=" * 84)
    print(f"WP-4_v2 FASE 3 — VALIDAZIONE  delta={delta:.4f} alpha_lo={alpha_lo:.4f} alpha_hi={alpha_hi:.4f}")
    print(f"  Delta AP (stop-tetto, OBIETTIVO Fase 2): mediana {np.median(deltas_ap):+.5f} · IQR "
          f"[{np.percentile(deltas_ap, 25):+.5f}, {np.percentile(deltas_ap, 75):+.5f}]")
    print(f"  Delta MCC_bal (stop-tetto, dataset pieno): mediana {np.median(deltas_mccbal):+.5f} · IQR "
          f"[{np.percentile(deltas_mccbal, 25):+.5f}, {np.percentile(deltas_mccbal, 75):+.5f}]")
    print(f"  Pearson(BCE,AP) diagnostico: mediana {np.median(pearsons):+.4f} · media {pearsons.mean():+.4f} "
          f"· std {pearsons.std():.4f}  (su {len(rows)} semi)")
    print(f"  Stop CENSURATI (patience mai scattata entro il cap): {n_censored}/{len(rows)}")
    try:
        rel = csv_path.relative_to(REPO)
    except ValueError:
        rel = csv_path
    print(f"  CSV: {rel}")
    print("=" * 84)


def main() -> None:
    a = build_parser().parse_args()
    setup_blas_env(int(a.threads), deterministic=False)
    {"precalc": cmd_precalc, "search": cmd_search, "validate": cmd_validate}[a.cmd](a)


if __name__ == "__main__":
    main()
