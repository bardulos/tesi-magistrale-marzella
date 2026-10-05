#!/usr/bin/env python
# TAG: DRIVER-PIPELINE | LOAO forward-only canonico
"""Run forward-only LOAO for finalists using previously recovered weights.

Aggregates per-class metrics across seeds. Plotting is optional; the metrics run without it.
Pure helpers aggregate results and load trial configuration.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


# ===================================================================== logica pura ====


def aggregate_loao_over_seeds(per_seed_results: list[dict]) -> dict:
    """Aggregate per-class and Z-DR metrics across a finalist's seeds."""
    if not per_seed_results:
        return {"by_class": {}, "z_dr_mean": float("nan"), "z_dr_std": float("nan")}

    classes = list(per_seed_results[0]["dr_by_class"].keys())
    by_class = {}
    for cls in classes:
        drs = [r["dr_by_class"].get(cls, float("nan")) for r in per_seed_results]
        cds = [r["cohens_d_by_class"].get(cls, float("nan")) for r in per_seed_results]
        drs_valid = [d for d in drs if not math.isnan(d)]
        cds_valid = [d for d in cds if not math.isnan(d)]
        fraction_covered = float(sum(1 for d in drs_valid if d > 0) / len(drs_valid)) \
            if drs_valid else float("nan")
        by_class[cls] = {
            "dr_mean":         float(np.mean(drs_valid))  if drs_valid else float("nan"),
            "dr_std":          float(np.std(drs_valid))   if drs_valid else float("nan"),
            "fraction_covered": fraction_covered,
            "cohens_d_mean":   float(np.mean(cds_valid)) if cds_valid else float("nan"),
            "cohens_d_std":    float(np.std(cds_valid))  if cds_valid else float("nan"),
            "n_seeds":         len(drs_valid),
        }

    z_drs = [r["z_dr"] for r in per_seed_results if not math.isnan(r.get("z_dr", float("nan")))]
    return {
        "by_class":   by_class,
        "z_dr_mean":  float(np.mean(z_drs)) if z_drs else float("nan"),
        "z_dr_std":   float(np.std(z_drs))  if z_drs else float("nan"),
    }


def load_trial_config(trial_dir: Path) -> dict:
    """Legge la config del trial da params.json nel trial_dir Ray."""
    trial_dir = Path(trial_dir)
    params_file = trial_dir / "params.json"
    if not params_file.exists():
        raise FileNotFoundError(f"params.json non trovato in {trial_dir}")
    with open(params_file) as f:
        return json.load(f)


# ===================================================================== driver ====

def main(argv=None):
    """Load finalist weights, run LOAO, aggregate metrics and write loao_by_class.csv.

    The overview figure needs lib.plotting, which is not in the repository, so it is
    normally skipped; the per-model figure loop is an empty placeholder.
    Runs one finalist/seed at a time; use a shell wrapper for parallel Ray fan-out.

    Uso:
      python tools/loao_run.py \\
          --finals-csv   runs/dae/compress_fase2/compress_summary.csv \\
          --weights-dir  <dir pesi> \\
          --val-X        <HOME>/dae_data_91/val_X.npy \\
          --val-attack   <transform_cse>/val_labels.npz \\
          --ray-dir      <search>/ray_results/dae_search \\
          [--import-dir  <search_400>/ray_results/dae_search] \\
          --out-dir      runs/dae/loao_fase3

    At least one of --ray-dir and --import-dir is needed: without them no params.json
    is found and every finalist is skipped.
    """
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finals-csv",  required=True, help="CSV con colonna trial_id (shortlist k')")
    ap.add_argument("--weights-dir", required=True,
                    help="dir con pesi: <weights_dir>/<trial_id>/seed_<seme>.weights.h5")
    ap.add_argument("--val-X",       required=True, help="path a val_X.npy")
    ap.add_argument("--val-attack",  required=True,
                    help="path a val_labels.npz (chiave Attack=stringhe classe)")
    ap.add_argument("--ray-dir",     default=None,
                    help="ray_results della search (per leggere params.json dei trial)")
    ap.add_argument("--import-dir",  default=None,
                    help="ray_results search_400 (per trial provenienti dai 400)")
    ap.add_argument("--out-dir",     required=True, help="dir output (deve esistere)")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lib.dae import constants as C
    from lib.dae.loao import run_loao

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        raise SystemExit(f"--out-dir non esiste: {out_dir}")

    val_X = np.load(args.val_X).astype(np.float32)
    # Attack labels are trusted object arrays produced by the project pipeline.
    val_attack = np.load(args.val_attack, allow_pickle=True)["Attack"]

    finalist_ids = []
    with open(args.finals_csv) as f:
        for row in csv.DictReader(f):
            finalist_ids.append(row["trial_id"])
    print(f"Finalisti LOAO: {len(finalist_ids)} trial", flush=True)

    trial_dirs_map = {}
    for d in [args.ray_dir, args.import_dir]:
        if d:
            for tdir in Path(d).glob("_run_trainable_*"):
                tid = tdir.name
                if tid in finalist_ids:
                    trial_dirs_map[tid] = tdir

    weights_dir = Path(args.weights_dir)
    all_results = {}
    per_class_summary = {}

    for tid in finalist_ids:
        print(f"\n--- {tid} ---", flush=True)
        if tid not in trial_dirs_map:
            print(f"  WARN: params.json non trovato per {tid}, salto.")
            continue
        config = load_trial_config(trial_dirs_map[tid])

        tid_weight_dir = weights_dir / tid
        weight_files = sorted(tid_weight_dir.glob("seed_*.weights.h5")) \
            if tid_weight_dir.exists() else []
        if not weight_files:
            print(f"  WARN: nessun file pesi in {tid_weight_dir}, salto.")
            continue

        per_seed = []
        for wf in weight_files:
            seed = int(wf.name.removesuffix(".weights.h5").split("_")[1])
            print(f"  seme {seed}: forward...", end=" ", flush=True)
            try:
                metrics = run_loao(wf, config, val_X, val_attack,
                                   fpr_target=C.FPR_TARGET, benign_name="Benign")
                per_seed.append(metrics)
                print(f"z_dr={metrics['z_dr']:.3f}")
            except Exception as e:
                print(f"ERRORE: {e}")

        if not per_seed:
            continue

        agg = aggregate_loao_over_seeds(per_seed)
        all_results[tid] = agg
        per_class_summary[tid] = {c: info["dr_mean"] for c, info in agg["by_class"].items()}
        print(f"  Z-DR medio: {agg['z_dr_mean']:.3f} ± {agg['z_dr_std']:.3f}")

    if all_results:
        classes_all = sorted({c for agg in all_results.values() for c in agg["by_class"]})
        csv_out = out_dir / "loao_by_class.csv"
        with open(csv_out, "w", newline="") as f:
            fields = ["trial_id", "z_dr_mean", "z_dr_std"] + \
                     [f"dr_{c}" for c in classes_all] + [f"frac_{c}" for c in classes_all]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for tid, agg in all_results.items():
                row = {"trial_id": tid,
                       "z_dr_mean": agg["z_dr_mean"],
                       "z_dr_std":  agg["z_dr_std"]}
                for c in classes_all:
                    row[f"dr_{c}"]   = agg["by_class"].get(c, {}).get("dr_mean",  float("nan"))
                    row[f"frac_{c}"] = agg["by_class"].get(c, {}).get("fraction_covered", float("nan"))
                w.writerow(row)
        print(f"\nLOAO CSV salvato in {csv_out}")

    try:
        from lib.plotting.loao_plots import plot_dr_by_class_for_model, plot_dr_overview_split
        figs_dir = out_dir / "figs"
        figs_dir.mkdir(exist_ok=True)

        for tid, agg in all_results.items():
            class_to_dr = {c: [r["dr_by_class"].get(c, float("nan"))
                                for r in []]
                           for c in agg["by_class"]}
            pass

        plot_dr_overview_split(per_class_summary,
                               figs_dir / "loao_overview.png",
                               per_block=10,
                               title=f"LOAO detection-rate — {len(all_results)} finalisti")
        print(f"Figura overview salvata in {figs_dir / 'loao_overview.png'}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
