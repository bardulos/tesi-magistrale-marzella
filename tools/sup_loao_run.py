# TAG: DRIVER-PIPELINE | 2026-07-13 | LOAO supervisionato locale P2 | vedi docs/refactoring_censimento.md
"""tools/sup_loao_run.py — LOAO supervisionato LOCALE a riaddestramento (P2, cap3). DELIVERABLE.

Orchestrazione: prodotto (classi x semi) eseguito in una FINESTRA MOBILE di sottoprocessi
(subprocess.Popen, MAI multiprocessing Python; pattern repo_v3 ceiling pipeline phase_C):
ogni job e' una self-invocazione `--one <classe> <seme>` con BLAS/TF a 1 thread, che scrive
un JSON per-job (niente race). A valle: merge in loao_sup_records.jsonl, aggregato
loao_sup_summary.csv e tabella di confronto ceiling_vs_dae.csv (SOLO tabella, GATE B) contro
il LOAO del DAE cap2 (runs/dae/loao_fase3/loao_records.jsonl, trial 569).

Parametri del GATE B (modello scelto, semi, worker) espliciti da CLI: nessuna scelta
silenziosa. ESECUZIONE GATED: lanciare solo dopo via libera.

Uso (orchestratore):
  python tools/sup_loao_run.py --params-json <winner.json> --out-dir <dir esistente> \
      --cache-dir runs/dae/output_569 --labels-val runs/preprocessing/val_attack_npy/val_Attack.npy \
      --seeds 42 123 2024 --workers 8
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.utils import setup_blas_env  # noqa: E402

setup_blas_env(1, deterministic=True)  # PRIMA di numpy/TF (anche nei job --one)

import argparse    # noqa: E402
import csv         # noqa: E402
import json        # noqa: E402
import subprocess  # noqa: E402
import time        # noqa: E402

import numpy as np  # noqa: E402


def sanitize(name: str) -> str:
    """Nome classe -> filename sicuro (pattern v3: Brute_Force_-Web, spazi, slash)."""
    return name.replace("/", "_").replace(" ", "_")


def run_window(cmds: list[list[str]], workers: int) -> int:
    """Finestra mobile di sottoprocessi: al piu' `workers` vivi; ritorna il numero di falliti."""
    pending = list(cmds)
    running: list[subprocess.Popen] = []
    failed = 0
    while pending or running:
        while pending and len(running) < workers:
            running.append(subprocess.Popen(pending.pop(0)))
            time.sleep(0.5)  # stagger: evita il picco di import TF simultanei
        done = [p for p in running if p.poll() is not None]
        for p in done:
            if p.returncode != 0:
                failed += 1
                print(f"[FAIL rc={p.returncode}] {' '.join(p.args[-6:])}")
            running.remove(p)
        if not done:
            time.sleep(2)
    return failed


def job_path(out_dir: Path, cls: str, seed: int) -> Path:
    return out_dir / f"job_{sanitize(cls)}_seed{seed}.json"


def run_one(args) -> None:
    """Un fold (classe, seme): eseguito nel sottoprocesso."""
    from lib.secondo_stadio.loao import train_loao_sup
    from lib.secondo_stadio.sup_search import build_parent_args, setup

    cfg = json.loads(Path(args.params_json).read_text())
    config = cfg["config"] if "config" in cfg else cfg   # accetta winner.json o dict piatto
    parent_args = build_parent_args({
        "cache_dir": str(args.cache_dir), "labels_val": str(args.labels_val),
        "max_epochs": args.max_epochs, "patience": args.patience,
        "es_n_attacks": args.es_n_attacks,
    })
    shared = setup(parent_args)
    res = train_loao_sup(config, args.one_seed, shared, parent_args, args.one_class)
    job_path(Path(args.out_dir), args.one_class, args.one_seed).write_text(
        json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--labels-val", required=True, type=Path)
    ap.add_argument("--params-json", required=True, type=Path,
                    help="config del modello scelto al GATE B (winner.json o dict piatto)")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="directory ESISTENTE (shell-first)")
    ap.add_argument("--seeds", type=int, nargs="+", required=True,
                    help="semi del LOAO (GATE B)")
    ap.add_argument("--classes", nargs="*", default=None,
                    help="default: tutte le classi != Benign nelle etichette")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--es-n-attacks", type=int, default=50_000)
    ap.add_argument("--dae-records", type=Path,
                    default=Path("runs/dae/loao_fase3/loao_records.jsonl"))
    ap.add_argument("--dae-trial", type=int, default=569)
    ap.add_argument("--one", nargs=2, metavar=("CLASSE", "SEME"), default=None,
                    help="(interno) esegue un singolo fold")
    args = ap.parse_args()
    if not args.out_dir.is_dir():
        raise SystemExit(f"out-dir inesistente (shell-first): {args.out_dir}")

    if args.one:
        args.one_class, args.one_seed = args.one[0], int(args.one[1])
        run_one(args)
        return

    # allow_pickle: etichette di prima parte (v. lib/secondo_stadio/data.py).
    labs = np.load(args.labels_val, allow_pickle=True).astype(str)
    classes = args.classes or sorted(c for c in np.unique(labs) if c != "Benign")
    jobs = [(c, s) for c in classes for s in args.seeds]
    todo = [(c, s) for (c, s) in jobs if not job_path(args.out_dir, c, s).exists()]
    print(f"LOAO sup: {len(classes)} classi x {len(args.seeds)} semi = {len(jobs)} fold "
          f"({len(todo)} da eseguire, {len(jobs) - len(todo)} gia' presenti)")

    base = [sys.executable, str(Path(__file__).resolve()),
            "--cache-dir", str(args.cache_dir), "--labels-val", str(args.labels_val),
            "--params-json", str(args.params_json), "--out-dir", str(args.out_dir),
            "--seeds", *map(str, args.seeds),
            "--max-epochs", str(args.max_epochs), "--patience", str(args.patience),
            "--es-n-attacks", str(args.es_n_attacks)]
    cmds = [base + ["--one", c, str(s)] for (c, s) in todo]
    failed = run_window(cmds, args.workers)
    if failed:
        raise SystemExit(f"{failed} fold falliti: correggere e rilanciare (i job fatti restano)")

    # ---- merge + aggregati + tabella di confronto ----
    from lib.secondo_stadio.loao import aggregate_loao, build_dr_table, dae_dr_from_records

    rows = [json.loads(job_path(args.out_dir, c, s).read_text()) for (c, s) in jobs]
    with open(args.out_dir / "loao_sup_records.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    agg = aggregate_loao(rows)
    with open(args.out_dir / "loao_sup_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["classe", "dr_mean", "dr_std", "dr_median", "n_seeds"])
        for cls, a in agg.items():
            w.writerow([cls, a["dr_mean"], a["dr_std"], a["dr_median"], a["n_seeds"]])

    dae = dae_dr_from_records(args.dae_records, trial_num=args.dae_trial)
    table = build_dr_table(dae, agg)
    with open(args.out_dir / "ceiling_vs_dae.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print(f"scritti: loao_sup_records.jsonl, loao_sup_summary.csv, ceiling_vs_dae.csv "
          f"in {args.out_dir}")


if __name__ == "__main__":
    main()
