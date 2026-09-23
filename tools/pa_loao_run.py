# TAG: DRIVER-PIPELINE | 2026-07-13 | LOAO PA a riaddestramento P5 (testato) | vedi docs/refactoring_censimento.md
"""tools/pa_loao_run.py — LOAO PA a riaddestramento per i K' finalisti (P5, cap3). DELIVERABLE.

Come tools/sup_loao_run.py (finestra mobile di sottoprocessi, riuso di run_window/sanitize)
ma sul prodotto (finalista x classe x seme): per ogni fold il PA si riaddestra da zero
(benigni + pseudo, invariante alla classe) e la classe esclusa esce dalla metrica per-epoca
(att_es) e dallo sweep standalone della soglia (att_pos) — lib.secondo_stadio.loao.
Output per finalista: records jsonl + summary CSV + tabella a TRE colonne DAE/ceiling/PA
(build_dr_table) contro il LOAO del DAE cap2 e il summary del LOAO supervisionato (P2).

Parallelizzazione e semi = GATE D (parametri CLI espliciti). ESECUZIONE GATED.

Uso (orchestratore):
  python tools/pa_loao_run.py --candidates-json <shortlist K'> --mlp-fixed-json <fissi P1> \
      --ceiling-summary <P2>/loao_sup_summary.csv --out-dir <dir esistente> \
      --cache-dir runs/dae/output_569 --labels-val .../val_Attack.npy \
      --seeds 42 123 --workers 8
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.utils import setup_blas_env  # noqa: E402

setup_blas_env(1, deterministic=True)  # PRIMA di numpy/TF (anche nei job --one)

import argparse    # noqa: E402
import csv         # noqa: E402
import json        # noqa: E402

import numpy as np  # noqa: E402

from sup_loao_run import run_window, sanitize  # noqa: E402 (riuso: zero duplicazione)


def summary_from_csv(path) -> dict:
    """Legge un summary LOAO (classe, dr_mean, dr_std, ...) nel formato di build_dr_table."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["classe"]] = {"dr_mean": float(r["dr_mean"]),
                                "dr_std": float(r["dr_std"])}
    return out


def job_path(out_dir: Path, trial_id: str, cls: str, seed: int) -> Path:
    return out_dir / f"job_{sanitize(str(trial_id))}_{sanitize(cls)}_seed{seed}.json"


def run_one(args) -> None:
    from lib.secondo_stadio.loao import train_loao_pa
    from lib.secondo_stadio.pa_search import build_parent_args, setup

    candidates = json.loads(Path(args.candidates_json).read_text())
    by_id = {str(c["trial_id"]): c["config"] for c in candidates}
    config = by_id[args.one_trial]
    parent_args = build_parent_args({
        "cache_dir": str(args.cache_dir), "labels_val": str(args.labels_val),
        "mlp_fixed": json.loads(Path(args.mlp_fixed_json).read_text()),
        "max_epochs": args.max_epochs, "patience": args.patience,
        "es_n_attacks": args.es_n_attacks,
    })
    shared = setup(parent_args)
    res = train_loao_pa(config, args.one_seed, shared, parent_args, args.one_class)
    res["trial_id"] = args.one_trial
    job_path(Path(args.out_dir), args.one_trial, args.one_class, args.one_seed).write_text(
        json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--labels-val", required=True, type=Path)
    ap.add_argument("--candidates-json", required=True, type=Path,
                    help="K' finalisti [{trial_id, config}] (GATE C)")
    ap.add_argument("--mlp-fixed-json", required=True, type=Path)
    ap.add_argument("--ceiling-summary", required=True, type=Path,
                    help="loao_sup_summary.csv del LOAO supervisionato (P2)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", required=True, help="semi (GATE D)")
    ap.add_argument("--classes", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--es-n-attacks", type=int, default=50_000)
    ap.add_argument("--dae-records", type=Path,
                    default=Path("runs/dae/loao_fase3/loao_records.jsonl"))
    ap.add_argument("--dae-trial", type=int, default=569)
    ap.add_argument("--one", nargs=3, metavar=("TRIAL", "CLASSE", "SEME"), default=None)
    args = ap.parse_args()
    if not args.out_dir.is_dir():
        raise SystemExit(f"out-dir inesistente (shell-first): {args.out_dir}")

    if args.one:
        args.one_trial, args.one_class, args.one_seed = (args.one[0], args.one[1],
                                                         int(args.one[2]))
        run_one(args)
        return

    # allow_pickle: etichette di prima parte (v. lib/secondo_stadio/data.py).
    labs = np.load(args.labels_val, allow_pickle=True).astype(str)
    classes = args.classes or sorted(c for c in np.unique(labs) if c != "Benign")
    candidates = json.loads(args.candidates_json.read_text())
    trials = [str(c["trial_id"]) for c in candidates]
    jobs = [(t, c, s) for t in trials for c in classes for s in args.seeds]
    todo = [j for j in jobs if not job_path(args.out_dir, *j).exists()]
    print(f"LOAO pa: {len(trials)} finalisti x {len(classes)} classi x {len(args.seeds)} "
          f"semi = {len(jobs)} fold ({len(todo)} da eseguire)")

    base = [sys.executable, str(Path(__file__).resolve()),
            "--cache-dir", str(args.cache_dir), "--labels-val", str(args.labels_val),
            "--candidates-json", str(args.candidates_json),
            "--mlp-fixed-json", str(args.mlp_fixed_json),
            "--ceiling-summary", str(args.ceiling_summary),
            "--out-dir", str(args.out_dir), "--seeds", *map(str, args.seeds),
            "--max-epochs", str(args.max_epochs), "--patience", str(args.patience),
            "--es-n-attacks", str(args.es_n_attacks)]
    cmds = [base + ["--one", t, c, str(s)] for (t, c, s) in todo]
    failed = run_window(cmds, args.workers)
    if failed:
        raise SystemExit(f"{failed} fold falliti: correggere e rilanciare (i job fatti restano)")

    # ---- merge + summary + tabella a tre colonne, PER FINALISTA ----
    from lib.secondo_stadio.loao import aggregate_loao, build_dr_table, dae_dr_from_records

    dae = dae_dr_from_records(args.dae_records, trial_num=args.dae_trial)
    ceiling = summary_from_csv(args.ceiling_summary)
    for t in trials:
        rows = [json.loads(job_path(args.out_dir, t, c, s).read_text())
                for c in classes for s in args.seeds]
        with open(args.out_dir / f"loao_pa_{sanitize(t)}_records.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        agg = aggregate_loao(rows)
        with open(args.out_dir / f"loao_pa_{sanitize(t)}_summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["classe", "dr_mean", "dr_std", "dr_median", "n_seeds"])
            for cls, a in agg.items():
                w.writerow([cls, a["dr_mean"], a["dr_std"], a["dr_median"], a["n_seeds"]])
        table = build_dr_table(dae, ceiling, agg)
        with open(args.out_dir / f"dae_vs_ceiling_vs_pa_{sanitize(t)}.csv", "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
        print(f"finalista {t}: records/summary/tabella a tre colonne in {args.out_dir}")


if __name__ == "__main__":
    main()
