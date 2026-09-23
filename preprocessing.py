#!/usr/bin/env python3
"""preprocessing.py — entry-point unico per la pipeline preprocessing NF-v3 (spine).

Uso:
    python preprocessing.py configs/preprocessing/ingest.yaml
    python preprocessing.py configs/preprocessing/encode.yaml \\
            --override primary=data/parquet/unsw_raw.parquet \\
            --override out_dir=runs/preprocessing/01/encode_unsw

Il YAML deve contenere `stage:` in {ingest, inspect, stats, encode, reduce, audit,
transform, verify}. Tutti gli
altri campi sono parametri di run; i path relativi (PATH_KEYS) sono risolti rispetto a
project_root (questo file). Dispatcher sottile: il glue (override, path, dispatch) e' in
`lib/cli.py`, condiviso con dae.py e secondo_stadio.py.

Convenzione shell-first: la directory `out_dir` va creata in shell PRIMA di invocare
(es. `mkdir -p runs/preprocessing/01/encode_cse`); l'entry-point non crea cartelle e
fallisce con errore esplicito se `out_dir` non esiste.
"""

from __future__ import annotations
from pathlib import Path

from lib.utils import setup_blas_env

# BLAS a 1 thread + determinismo, PRIMA che gli stage importino numpy (convenzione spine).
setup_blas_env(1, deterministic=True)

from lib.cli import run_orchestrator  # noqa: E402  (dopo setup_blas_env, intenzionale)


PROJECT_ROOT = Path(__file__).resolve().parent

# Dispatch table: stage -> (modulo, funzione). Import lazy via importlib (in lib.cli).
STAGES = {
    "ingest":    ("lib.preprocessing.data",      "run_ingest_stage"),
    "inspect":   ("lib.preprocessing.data",      "run_inspect_stage"),
    "stats":     ("lib.preprocessing.data",      "run_stats_stage"),
    "encode":    ("lib.preprocessing.encode",    "run_encode_stage"),
    "reduce":    ("lib.preprocessing.encode",    "run_reduce_stage"),
    "audit":     ("lib.preprocessing.data",      "run_audit_stage"),
    "transform": ("lib.preprocessing.transform", "run_transform_stage"),
    "verify":    ("lib.preprocessing.verify",    "run_verify_stage"),
}

PATH_KEYS = {
    "primary", "out_dir", "schema_from", "apply_from", "compare",
    "scaler_params", "transform_meta", "encode_primary", "raw_primary",
}


if __name__ == "__main__":
    run_orchestrator(STAGES, PATH_KEYS, "preprocessing", PROJECT_ROOT, __doc__)
