"""Entry-point del secondo stadio.

Stage: `sup_search`, `pa_search`, `wp4v2_es_search` (arresto label-free del PA), `fp_mining`, `fusione`,
`fusione_canonica`, `pseudo_placement`, `ablazione`. Le condizioni di esecuzione sono
documentate nei config.

Uso: `python secondo_stadio.py configs/secondo_stadio/<stage>.yaml [--override chiave=valore ...]`.
Lo stage e' scelto da `stage:` nel config.
"""
from __future__ import annotations

from pathlib import Path

from lib.utils import setup_blas_env

setup_blas_env(1, deterministic=True)

from lib.cli import run_orchestrator  # noqa: E402 (dopo setup_blas_env, intenzionale)

PROJECT_ROOT = Path(__file__).resolve().parent
STAGES = {
    "sup_search": ("lib.secondo_stadio.sup_search", "run_sup_search_stage"),
    "pa_search": ("lib.secondo_stadio.pa_search", "run_pa_search_stage"),
    "wp4v2_es_search": ("lib.secondo_stadio.wp4v2_es_search", "run_wp4v2_es_search_stage"),
    "fp_mining": ("lib.secondo_stadio.fp_mining", "run_fp_mining_stage"),
    "fusione": ("lib.secondo_stadio.fusione_eval", "run_fusione_stage"),
    "fusione_canonica": ("lib.secondo_stadio.fusione_eval", "run_fusione_canonica_stage"),
    "pseudo_placement": ("lib.secondo_stadio.pseudo_placement",
                         "run_pseudo_placement_stage"),
    "ablazione": ("lib.secondo_stadio.ablazione", "run_ablazione_stage"),
}
PATH_KEYS = {"out_dir", "cache_dir", "labels_val", "labels_test", "l1l2_csv",
             "weights_home", "project_root", "pa_weights", "base_weights",
             "scores_npz", "canonico", "dump_scores_npz", "repass_dir", "precalc_dir"}

if __name__ == "__main__":
    run_orchestrator(STAGES, PATH_KEYS, "secondo_stadio", PROJECT_ROOT, __doc__)
