# TAG: DRIVER-PIPELINE | 2026-07-13 | compressione PA K->K' P4 (testato) | vedi docs/refactoring_censimento.md
"""tools/pa_compress.py — compressione multiseme K->K' del PA (P4, cap3). DELIVERABLE (gated).

Per ogni candidato sopravvissuto al pre-cull (tools/pa_shortlist.py precull) riaddestra il PA
sui SEMI EXTRA (SEEDS_PA_EXTRA) e riporta MCC_bal inline (+ informative). Pattern di
tools/refit_compress.py (che NON si rifattorizza): worker Ray a 1 CPU con max_retries=3,
cache dati PER-PROCESSO (pa_search.setup chiamato una volta per worker: page-cache mmap
condivisa fra i job dello stesso nodo), parent_args condivisi via ray.put (i PATH viaggiano
nel payload, mai gli array), CSV con append incrementale + flush (robusto ai crash, i job
gia' in CSV sono saltati al rilancio). Pesi per (trial, seme) in weights_home (HOME, mai /tmp).

ESECUZIONE GATED (GATE C definisce pre-cull e fattori; il lancio richiede cluster + sync gate).

Uso:
  python tools/pa_compress.py --candidates-json <precull>/candidates.json \
      --out-dir <dir esistente> --cache-dir runs/dae/output_569 \
      --labels-val runs/preprocessing/val_attack_npy/val_Attack.npy \
      --mlp-fixed-json <path json {ratio,lr,dropout,l2_reg}> [--seeds 2038 ... ]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.secondo_stadio import constants as C  # noqa: E402

CSV_FIELDS = ["trial_id", "seed", "mcc_bal", "p_star", "tau_clf", "fpr", "recall",
              "auc_roc", "mcc_fpr1", "best_ap", "best_epoch", "n_epochs",
              "filter_reason", "weights_path"]

# Cache di processo del worker: setup() una volta, riusato da tutti i job del processo.
_WORKER_STATE: dict = {}


def _get_shared(parent_args: dict):
    key = (parent_args["cache_dir"], parent_args["labels_val"],
           parent_args["split_seed"], parent_args["val_frac"],
           parent_args["es_n_attacks"])
    if key not in _WORKER_STATE:
        from lib.secondo_stadio.pa_search import setup
        _WORKER_STATE.clear()
        _WORKER_STATE[key] = setup(parent_args)
    return _WORKER_STATE[key]


def _worker_fn(job: dict, parent_args: dict) -> dict:
    """Un job = (candidato, seme extra). Eseguito sul worker Ray (1 CPU)."""
    pr = parent_args.get("project_root")
    if pr and pr not in sys.path:
        sys.path.insert(0, pr)
    from lib.secondo_stadio.pa_search import train_one_seed

    shared = _get_shared(parent_args)
    seed_dir = (Path(parent_args["weights_home"]) / "pa_compress"
                / str(job["trial_id"]) / f"seed_{job['seed']}")
    res = train_one_seed(job["config"], int(job["seed"]), shared, parent_args, seed_dir)
    return {"trial_id": job["trial_id"], **{k: res.get(k) for k in CSV_FIELDS if k != "trial_id"}}


def already_done(csv_path: Path) -> set[tuple[str, int]]:
    """(trial_id, seed) gia' presenti nel CSV (ripartenza incrementale)."""
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {(r["trial_id"], int(r["seed"])) for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates-json", required=True, type=Path,
                    help="[{trial_id, config}] dal pre-cull (pa_shortlist precull)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--labels-val", required=True, type=Path)
    ap.add_argument("--mlp-fixed-json", required=True, type=Path,
                    help="fissi MLP del vincitore P1 (GATE B)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.SEEDS_PA_EXTRA))
    ap.add_argument("--weights-home", type=Path, default=Path.home() / "repov6_weights_s2")
    ap.add_argument("--ray-address", default="auto")
    ap.add_argument("--max-epochs", type=int, default=C.MLP_MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=C.MLP_PATIENCE)
    ap.add_argument("--es-n-attacks", type=int, default=C.ES_N_ATTACKS)
    args = ap.parse_args()
    if not args.out_dir.is_dir():
        raise SystemExit(f"out-dir inesistente (shell-first): {args.out_dir}")

    from lib.secondo_stadio.pa_search import build_parent_args
    parent_args = build_parent_args({
        "cache_dir": str(args.cache_dir), "labels_val": str(args.labels_val),
        "mlp_fixed": json.loads(args.mlp_fixed_json.read_text()),
        "max_epochs": args.max_epochs, "patience": args.patience,
        "es_n_attacks": args.es_n_attacks,
        "save_weights": True, "weights_home": str(args.weights_home),
    })
    parent_args["project_root"] = str(Path(__file__).resolve().parent.parent)

    candidates = json.loads(args.candidates_json.read_text())
    csv_path = args.out_dir / "pa_compress.csv"
    done = already_done(csv_path)
    jobs = [{"trial_id": c["trial_id"], "config": c["config"], "seed": s}
            for c in candidates for s in args.seeds
            if (c["trial_id"], s) not in done]
    print(f"pa_compress: {len(candidates)} candidati x {len(args.seeds)} semi extra = "
          f"{len(candidates) * len(args.seeds)} job ({len(jobs)} da eseguire)")
    if not jobs:
        return

    import ray
    ray.init(address=args.ray_address, ignore_reinit_error=True,
             runtime_env={"env_vars": {"PYTHONPATH": parent_args["project_root"],
                                       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                       "OPENBLAS_NUM_THREADS": "1",
                                       "TF_NUM_INTEROP_THREADS": "1",
                                       "TF_NUM_INTRAOP_THREADS": "1",
                                       "TF_DETERMINISTIC_OPS": "1",
                                       "TF_CPP_MIN_LOG_LEVEL": "3"}})
    remote_fn = ray.remote(num_cpus=1, max_retries=3)(_worker_fn)
    pa_ref = ray.put(parent_args)
    futures = {remote_fn.remote(job, pa_ref): job for job in jobs}

    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
            f.flush()
        pending = list(futures)
        while pending:
            done_refs, pending = ray.wait(pending, num_returns=1)
            job = futures[done_refs[0]]
            try:
                row = ray.get(done_refs[0])
                w.writerow(row)
                f.flush()
            except Exception as exc:  # noqa: BLE001 - un job perso non ferma la campagna
                print(f"[FAIL] trial={job['trial_id']} seed={job['seed']}: {exc}")
    ray.shutdown()
    print(f"pa_compress: CSV in {csv_path}")


if __name__ == "__main__":
    main()
