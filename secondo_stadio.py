"""secondo_stadio.py — entry-point del pilastro secondo_stadio (repov6, cap3).

Stage (P1..P7 del piano cap3 + post-cap3; le ESECUZIONI sono gated, v. i commenti nei config):
  sup_search       — ricerca del ceiling supervisionato (P1, motore condiviso AUC-PR)
  pa_search        — ricerca del classificatore di pseudo-anomalie (P3, MCC_bal)
  wp4v2_es_search  — calibrazione del criterio d'arresto label-free del PA (post-cap3)
  fp_mining        — hardening per hard-negative mining del finalista (P6)
  fusione          — valutazione AND∪logit sui preset FPR, test una volta (P7)
  fusione_canonica — regola v2 a tre rami, modalita' operative A/B (P7, canonica)
  pseudo_placement — collocazione delle pseudo nel piano 2D, gate un-whitening (P7)
  ablazione        — confronto leak-free delle 4 regole di decisione (P7)

Uso: python secondo_stadio.py configs/secondo_stadio/<stage>.yaml [--override chiave=valore ...]
Lo stage e' scelto dal campo `stage:` del config.
"""
from __future__ import annotations

from pathlib import Path

from lib.utils import setup_blas_env

setup_blas_env(1, deterministic=True)  # PRIMA di numpy/TensorFlow

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
