"""lib/search/engine.py — motore di ricerca Optuna+Ray-Tune (repov6).

Orchestrazione generica e indipendente dal componente (DAE/PA):
- studio Optuna persistente (TPESampler multivariate/group/constant_liar);
- ASHAScheduler sui SEMI (time_attr="seeds_completed"); rung al 4 seme (cfg-driven);
- trainable multiseme generico: setup() UNA VOLTA -> loop semi -> train_one_seed ->
  aggregate -> tune.report({metric, seeds_completed});
- PRUNED VERITIERO: `PruningAwareOptunaSearch` marca PRUNED (study.tell) i trial fermati
  da ASHA prima di max_t semi (riconosciuti da seeds_completed<max_t). Il TPE di Optuna 4.8
  include nativamente i PRUNED nel fit col valore-al-rung (sano a rung-4/AUC-PR); nessun
  sampler custom. Verificato su Ray 2.55.1 / Optuna 4.8.0 (raise TrialPruned NON funziona:
  ASHA pota via Ray, il trial arriverebbe COMPLETE col valore-al-rung).
- GATE select_winner: vincitore SOLO tra seeds_completed==max_t (lib.search.selection).

La parte specifica del componente vive in un `SearchComponent`.
"""
from __future__ import annotations

# === ENV threading PRIMA di numpy/TF (determinismo, coerente con la Fase 0) ===
import os

for _k, _v in {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1", "TF_DETERMINISTIC_OPS": "1", "TF_CPP_MIN_LOG_LEVEL": "3",
}.items():
    os.environ.setdefault(_k, _v)

import json
import logging
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from optuna.samplers import TPESampler
from optuna.trial import TrialState
from ray.tune.result import TRAINING_ITERATION
from ray.tune.search.optuna import OptunaSearch

from lib.search.selection import seed_order_from_trial_id, select_winner

log = logging.getLogger("search_engine")


# ============================================================================
# PRUNED veritiero: marca PRUNED i trial potati da ASHA prima di max_t semi
# ============================================================================


class PruningAwareOptunaSearch(OptunaSearch):
    """OptunaSearch che, su Ray 2.55, marca PRUNED (study.tell) i trial fermati da ASHA
    al rung (seeds_completed < max_t) invece di lasciarli COMPLETE col valore-al-rung.
    Lo stato dello studio resta onesto; il TPE 4.8 include nativamente i PRUNED nel fit.
    Classe di MODULO (picklabile per il resume di Ray Tune)."""

    def __init__(self, *args, max_t: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_t = int(max_t)

    def on_trial_complete(self, trial_id, result=None, error=False):
        if (not error and result is not None
                and trial_id in self._ot_trials
                and trial_id not in self._completed_trials):
            sc = result.get("seeds_completed")
            if sc is not None and int(sc) < self._max_t:
                ot_trial = self._ot_trials[trial_id]
                val = result.get(self.metric) if not isinstance(self.metric, list) else None
                step = result.get(TRAINING_ITERATION)
                if val is not None and step is not None:
                    try:
                        ot_trial.report(float(val), int(step))   # registra il valore-al-rung
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    self._ot_study.tell(ot_trial, state=TrialState.PRUNED)
                except Exception as exc:  # noqa: BLE001
                    log.warning("tell(PRUNED) fallito per %s: %s", trial_id, exc)
                self._completed_trials.add(trial_id)
                return
        return super().on_trial_complete(trial_id, result, error)


# ============================================================================
# Contratto componente
# ============================================================================


@dataclass
class SearchComponent:
    """Tutto cio' che e' SPECIFICO del componente (DAE/PA). I callable devono essere
    funzioni a livello di modulo (picklabili da Ray) — niente closure/lambda."""

    name: str
    seeds: list[int]
    metric_name: str
    define_space: Callable[[Any], dict]
    build_parent_args: Callable[[dict], dict]
    setup: Callable[[dict], Any]
    train_one_seed: Callable[[dict, int, Any, dict, Path], dict]
    aggregate: Callable[[list, dict], dict]
    seed_order: Callable[[str, list], Any] = seed_order_from_trial_id
    report_extra_keys: tuple = ()
    max_t: int | None = None

    def resolved_max_t(self) -> int:
        return int(self.max_t) if self.max_t is not None else len(self.seeds)


# ============================================================================
# Trainable generico (eseguito dai worker Ray)
# ============================================================================


def _run_trainable(config: dict, parent_args: dict, component: SearchComponent) -> None:
    """Trainable Ray multiseme generico. La config TF/threading e' del componente
    (setup/train_one_seed): il motore resta framework-agnostico."""
    project_root = parent_args.get("project_root")
    if project_root and project_root not in sys.path:
        sys.path.insert(0, project_root)
    from ray import tune as _ray_tune

    trial_dir = Path.cwd()
    metric_name = component.metric_name

    trial_id_str = _ray_tune.get_context().get_trial_id()
    seeds_trial = list(component.seed_order(trial_id_str, component.seeds))
    (trial_dir / "seed_order.json").write_text(
        json.dumps({"trial_id": trial_id_str, "seeds_trial": seeds_trial}))

    shared_state = component.setup(parent_args)

    # Radice dei pesi: HOME durevole se `weights_home` è impostata (mai /tmp), altrimenti la
    # working-dir Ray (cwd) come da comportamento storico. Vedi build_parent_args (search.py).
    _wh = parent_args.get("weights_home")
    weights_base = (Path(_wh) / trial_id_str) if _wh else trial_dir

    per_seed_results: list[dict] = []
    for seed_idx, seed in enumerate(seeds_trial):
        seed_dir = weights_base / f"seed_{seed}"
        result = component.train_one_seed(config, int(seed), shared_state, parent_args, seed_dir)
        per_seed_results.append(result)

        agg = component.aggregate(per_seed_results, config)
        if metric_name not in agg:
            raise KeyError(
                f"aggregate() del componente '{component.name}' non ha restituito "
                f"la chiave metrica '{metric_name}'")

        report = {metric_name: float(agg[metric_name]), "seeds_completed": int(seed_idx + 1)}
        for k in component.report_extra_keys:
            if k in agg:
                report[k] = agg[k]

        agg_dump = dict(agg)
        agg_dump["seeds_completed"] = int(seed_idx + 1)
        agg_dump["seeds_trial"] = seeds_trial
        (trial_dir / "metrics.json").write_text(json.dumps(agg_dump, indent=2, default=float))

        _ray_tune.report(report)
        # NB: nessun raise di optuna.TrialPruned (non funziona sotto Ray 2.55). Il PRUNED
        # veritiero e' gestito dal searcher (PruningAwareOptunaSearch.on_trial_complete).


# ============================================================================
# Orchestratore
# ============================================================================


def run_search(component: SearchComponent, cfg: dict) -> dict:
    """Esegue la ricerca Optuna+ASHA per `component` sul cluster Ray. Warm-start opzionale via
    enqueue (cfg['warmstart']: config esplicite o traslate+random in coda; import-storia
    opzionale via ws['import_history']). Ritorna un dict riepilogo (winner, study_path)."""
    from lib import utils as common

    project_root = Path(cfg.get("project_root", Path.cwd()))
    out_dir = Path(cfg["out_dir"])
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = None if cfg.get("no_file_log", False) else (out_dir / "log.txt")
    common.setup_logging(log_file)
    log.info("search '%s' (host=%s)", component.name, socket.gethostname())

    n_trials = int(cfg.get("n_trials", 150))
    max_concurrent = int(cfg.get("max_concurrent_trials", 200))
    grace_period = int(cfg.get("grace_period", 4))          # rung al 4 seme
    # float, non int: il componente PA usa rf=1.5 (rung a 4 e 6 semi con grace=4, max_t=8);
    # un cast a int lo degraderebbe a 1 e l'assert di Ray (reduction_factor > 1) fallirebbe.
    reduction_factor = float(cfg.get("reduction_factor", 2))
    sampler_seed = int(cfg.get("seed", 42))
    n_startup = int(cfg.get("n_startup_trials", 48))         # TPE puro, niente warm-start
    study_name = cfg.get("study_name", f"{component.name}_search")
    ray_address = cfg.get("ray_address", "auto")
    resume_mode = cfg.get("resume", "auto")
    max_failures = int(cfg.get("max_failures", 10))
    max_t = component.resolved_max_t()
    metric_name = component.metric_name

    parent_args = component.build_parent_args(cfg)
    parent_args.setdefault("project_root", str(project_root))

    log.info("config: n_trials=%d max_concurrent=%d asha[grace=%d rf=%s max_t=%d] "
             "tpe[n_startup=%d multivariate constant_liar] metric=%s seeds=%s",
             n_trials, max_concurrent, grace_period, reduction_factor, max_t,
             n_startup, metric_name, component.seeds)

    import optuna

    storage_url = f"sqlite:///{(out_dir / 'optuna_study.db').resolve()}"
    storage = optuna.storages.RDBStorage(url=storage_url)
    log.info("Optuna storage: %s (study='%s')", storage_url, study_name)

    def _make_sampler():
        return TPESampler(seed=sampler_seed, multivariate=True, group=True,
                          constant_liar=True, n_startup_trials=n_startup)

    study = optuna.create_study(study_name=study_name, storage=storage,
                                direction="maximize", load_if_exists=True,
                                sampler=_make_sampler())

    # warm-start (enqueue batch): config traslate (dai top del DB pregresso) + random nella regione
    # allargata, messe in coda come primi trial. Niente import-storia (lo studio resta pulito). Gli
    # enqueued sono consumati da OptunaSearch (stesso storage+study_name) prima del campionamento TPE.
    ws = cfg.get("warmstart")
    if ws:
        # A) ENQUEUE ESPLICITO: lista di config (o path a un JSON) messe in coda come primi trial.
        #    Per gli spazi che build_enqueue non conosce (es. PA: exp/ratio/dropout/lr + delta/alpha).
        #    Config PARZIALI ammesse: Optuna FISSA i parametri dati e CAMPIONA il resto per quel trial
        #    (es. warm-start dell'architettura dalle rosse P1, delta/alpha lasciati al TPE).
        explicit = ws.get("enqueue_configs")
        if isinstance(explicit, str):
            explicit = json.loads(Path(explicit).read_text())
        if explicit:
            n_ok = 0
            for c in explicit:
                try:
                    study.enqueue_trial(c, skip_if_exists=True); n_ok += 1
                except Exception as exc:  # noqa: BLE001
                    log.warning("enqueue esplicito fallito per %s: %s", c, exc)
            log.info("warm-start: %d/%d config ESPLICITE in coda", n_ok, len(explicit))
        # B) BUILD_ENQUEUE dal DB pregresso (DAE: traslate dai top + random regione allargata) +
        #    import-storia opzionale. Solo se `db_path` e' presente.
        if ws.get("db_path"):
            from lib.search.warmstart import build_enqueue, import_history, load_completed
            db_path = ws["db_path"]
            ws_study = ws.get("study_name", study_name)
            if ws.get("import_history"):
                n_imp = import_history(study, db_path, ws_study)
                log.info("warm-start: importate %d trial pregresse come storia (prior del TPE)", n_imp)
            completed = load_completed(db_path, ws_study)
            enq = build_enqueue(completed, n_random=int(ws.get("n_random", 80)),
                                rng_seed=int(ws.get("rng_seed", 0)))
            for c in enq:
                try:
                    study.enqueue_trial(c, skip_if_exists=True)
                except Exception as exc:  # noqa: BLE001
                    log.warning("enqueue fallito per %s: %s", c, exc)
            log.info("warm-start: %d config in coda (da %d COMPLETE pregressi, n_random=%d)",
                     len(enq), len(completed), int(ws.get("n_random", 80)))

    n_already = len(study.trials)
    n_new = cfg.get("n_new_trials")
    num_samples = int(n_new) if n_new is not None else max(0, n_trials - n_already)
    log.info("study size=%d num_samples(NUOVI)=%d", n_already, num_samples)

    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    runtime_env = {"env_vars": {
        "PYTHONPATH": str(project_root),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1", "TF_DETERMINISTIC_OPS": "1", "TF_CPP_MIN_LOG_LEVEL": "3",
    }}
    if ray_address in (None, "local"):
        ray.init(address="local", num_cpus=int(cfg.get("num_cpus_local", 4)),
                 include_dashboard=False, ignore_reinit_error=True,
                 configure_logging=False, runtime_env=runtime_env)
    else:
        ray.init(address=ray_address, ignore_reinit_error=True, runtime_env=runtime_env)
    log.info("Ray cluster: CPU=%s", ray.cluster_resources().get("CPU"))

    search = PruningAwareOptunaSearch(
        space=component.define_space, metric=metric_name, mode="max",
        sampler=_make_sampler(), study_name=study_name, storage=storage, max_t=max_t)
    asha = ASHAScheduler(metric=metric_name, mode="max", max_t=max_t,
                         grace_period=grace_period, reduction_factor=reduction_factor,
                         time_attr="seeds_completed")

    trainable_fn = tune.with_parameters(_run_trainable, parent_args=parent_args, component=component)
    trial_resources = {"cpu": int(cfg.get("cpu_per_trial", 1))}
    ray_storage = str((out_dir / "ray_results").resolve())

    summary = {
        "component": component.name, "study_name": study_name,
        "study_path": str((out_dir / "optuna_study.db").resolve()),
        "metric_name": metric_name, "max_t": max_t,
    }
    if num_samples == 0:
        log.info("num_samples=0: nessun trial da lanciare.")
        ray.shutdown()
        summary["winner"] = None
        return summary

    t0 = time.perf_counter()
    analysis = tune.run(
        tune.with_resources(trainable_fn, trial_resources),
        num_samples=num_samples, scheduler=asha, search_alg=search,
        max_concurrent_trials=max_concurrent, storage_path=ray_storage,
        name=f"{component.name}_search",
        resume="AUTO" if resume_mode == "auto" else None,
        max_failures=max_failures, verbose=int(cfg.get("verbose", 1)))
    log.info("tune.run completato in %.1fs", time.perf_counter() - t0)

    try:
        analysis.dataframe().to_csv(out_dir / "tune_results.csv", index=False)
    except Exception as e:  # noqa: BLE001
        log.warning("dataframe analysis fallito: %s", e)

    hist: dict[int, int] = {}
    for t in list(getattr(analysis, "trials", []) or []):
        sc = (getattr(t, "last_result", None) or {}).get("seeds_completed")
        if sc is not None:
            hist[int(sc)] = hist.get(int(sc), 0) + 1
    summary["seeds_completed_hist"] = hist

    winner = select_winner(analysis, metric_name, max_t, mode="max")
    summary["winner"] = winner
    if winner and winner.get("trial_id"):
        (out_dir / "winner.json").write_text(json.dumps(winner, indent=2, default=float))
        log.info("VINCITORE (seeds_completed==%d): %s %s=%.6f (%d/%d completi)",
                 max_t, winner["trial_id"], metric_name, winner["objective"],
                 winner["n_complete"], winner["n_total"])
    else:
        log.warning("NESSUN trial ha raggiunto seeds_completed==%d (max_t).", max_t)

    ray.shutdown()
    log.info("===== run '%s' completata =====", component.name)
    return summary
