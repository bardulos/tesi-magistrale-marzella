#!/usr/bin/env python
# TAG: DRIVER-PIPELINE | 2026-07-13 | LOAO forward-only canonico (testato) | vedi docs/refactoring_censimento.md
"""tools/loao_run.py — FASE 3: LOAO sui k' finalisti (inferenza pura).

Per ogni finalista (trial_id, config) × seme:
  - carica i pesi salvati (HOME dei VPS, raccolti via rsync);
  - esegue il forward DAE su val_X (una passata, inferenza deterministica);
  - calcola loao_metrics (DR per-classe, Z-DR, Cohen's d);
  - aggrega su tutti i semi: mean/std/fraction_covered per classe;
  - genera le figure LOAO solo se e' disponibile lib/plotting, non incluso nel repository (guard
    try/except: senza, il calcolo procede).

Funzioni pure (testabili senza TF/Ray):
  aggregate_loao_over_seeds — aggrega DR/Cohen per classe su più semi
  load_trial_config         — legge config da params.json nel trial_dir Ray

PREREQUISITO: pesi censiti e recuperati (FASE 3 — usa-e-getta). Il driver non ri-addestra:
usa i pesi esistenti punto per punto. Usare run_loao (lib.dae.loao) per il forward TF.
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
    """Aggrega i risultati di loao_metrics su più semi per un finalista.

    Restituisce:
      by_class[classe] = {dr_mean, dr_std, fraction_covered, cohens_d_mean, cohens_d_std}
      z_dr_mean, z_dr_std  — Z-DR (media dei DR per-classe) aggregato sui semi
    """
    if not per_seed_results:
        return {"by_class": {}, "z_dr_mean": float("nan"), "z_dr_std": float("nan")}

    classes = list(per_seed_results[0]["dr_by_class"].keys())
    by_class = {}
    for cls in classes:
        drs = [r["dr_by_class"].get(cls, float("nan")) for r in per_seed_results]
        cds = [r["cohens_d_by_class"].get(cls, float("nan")) for r in per_seed_results]
        drs_valid = [d for d in drs if not math.isnan(d)]
        cds_valid = [d for d in cds if not math.isnan(d)]
        # fraction_covered: frazione di semi in cui il DR > 0 (copre almeno qualcosa)
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
    """Driver FASE 3: carica pesi per-seme dei k' finalisti, esegue forward LOAO,
    aggrega e genera figure. Lancia in sequenza (un finalista × seme per volta);
    per il fan-out parallelo su Ray usare uno script wrapper shell.

    Uso:
      python tools/loao_run.py \\
          --finals-csv   runs/dae/compress_fase2/compress_summary.csv \\
          --weights-dir  runs/dae/weights \\
          --val-X        <HOME>/dae_data_91/val_X.npy \\
          --val-attack   <HOME>/tesi/repo_v5/.../val_labels.npz \\
          --out-dir      runs/dae/loao_fase3
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

    # Carica val_X e val_Attack
    val_X = np.load(args.val_X).astype(np.float32)
    # allow_pickle: Attack è array di stringhe Python (dtype=object), pipeline controllata
    val_attack = np.load(args.val_attack, allow_pickle=True)["Attack"]

    # Carica shortlist k' finalisti
    finalist_ids = []
    with open(args.finals_csv) as f:
        for row in csv.DictReader(f):
            finalist_ids.append(row["trial_id"])
    print(f"Finalisti LOAO: {len(finalist_ids)} trial", flush=True)

    # Costruisce mappa trial_id → params.json
    trial_dirs_map = {}
    for d in [args.ray_dir, args.import_dir]:
        if d:
            for tdir in Path(d).glob("_run_trainable_*"):
                tid = tdir.name
                if tid in finalist_ids:
                    trial_dirs_map[tid] = tdir

    weights_dir = Path(args.weights_dir)
    all_results = {}   # {trial_id: aggregated_loao}
    per_class_summary = {}  # {trial_id: {classe: dr_medio}}

    for tid in finalist_ids:
        print(f"\n--- {tid} ---", flush=True)
        if tid not in trial_dirs_map:
            print(f"  WARN: params.json non trovato per {tid}, salto.")
            continue
        config = load_trial_config(trial_dirs_map[tid])

        # Trova tutti i file di pesi per questo trial
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

    # Salva CSV aggregato per-classe
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

    # Figure (guard: no-op se matplotlib assente)
    try:
        from lib.plotting.loao_plots import plot_dr_by_class_for_model, plot_dr_overview_split
        figs_dir = out_dir / "figs"
        figs_dir.mkdir(exist_ok=True)

        # (1) Per-modello: distribuzione DR sui semi per classe
        for tid, agg in all_results.items():
            class_to_dr = {c: [r["dr_by_class"].get(c, float("nan"))
                                for r in []]   # placeholder: usare per_seed per-trial
                           for c in agg["by_class"]}
            # Usa dr_mean come valore singolo (senza la distribuzione per-seme qui)
            # La figura per-seme si genera in un secondo passaggio con i per_seed raw
            pass

        # (2) Overview splittato
        plot_dr_overview_split(per_class_summary,
                               figs_dir / "loao_overview.png",
                               per_block=10,
                               title=f"LOAO detection-rate — {len(all_results)} finalisti")
        print(f"Figura overview salvata in {figs_dir / 'loao_overview.png'}")
    except Exception:
        pass  # plotting guard: il calcolo continua senza figure


if __name__ == "__main__":
    main()
