"""Entry-point della pipeline DAE.

Stage: `search`, ricerca iperparametri con AUC-PR.

Uso: `python dae.py configs/dae/<stage>.yaml [--override chiave=valore ...]`.
Lo stage e' scelto da `stage:` nel config.
"""
from __future__ import annotations

from pathlib import Path

from lib.utils import setup_blas_env

setup_blas_env(1, deterministic=True)

from lib.cli import run_orchestrator  # noqa: E402 (dopo setup_blas_env, intenzionale)

PROJECT_ROOT = Path(__file__).resolve().parent
STAGES = {
    "search": ("lib.dae.search", "run_search_stage"),
}
PATH_KEYS = {"out_dir", "shard_dir", "val_X", "val_y", "project_root"}

if __name__ == "__main__":
    run_orchestrator(STAGES, PATH_KEYS, "dae", PROJECT_ROOT, __doc__)
