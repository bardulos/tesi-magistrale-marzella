#!/usr/bin/env python3
"""Entry-point della pipeline preprocessing NF-v3.

Uso:
    python preprocessing.py configs/preprocessing/ingest.yaml
    python preprocessing.py configs/preprocessing/encode.yaml \\
            --override primary=data/parquet/unsw_raw.parquet \\
            --override out_dir=runs/preprocessing/01/encode_unsw

Il YAML richiede `stage:` (ingest, inspect, stats, encode, reduce, audit, transform,
verify). I path relativi in PATH_KEYS sono risolti rispetto alla root del progetto.
Il dispatch condiviso e' in `lib/cli.py`.

Creare `out_dir` prima dell'invocazione (es. `mkdir -p runs/preprocessing/01/encode_cse`);
l'entry-point non crea directory.
"""

from __future__ import annotations
from pathlib import Path

from lib.utils import setup_blas_env

# BLAS a 1 thread prima degli import degli stage.
setup_blas_env(1, deterministic=True)

from lib.cli import run_orchestrator  # noqa: E402  (dopo setup_blas_env, intenzionale)


PROJECT_ROOT = Path(__file__).resolve().parent

# Stage -> modulo e funzione; import lazy via lib.cli.
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
