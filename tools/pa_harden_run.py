# TAG: DRIVER-PIPELINE | 2026-07-13 | hardening FP-mining P6 su Ray — responsabile P6 (stage fp_mining mai esercitato) | vedi docs/refactoring_censimento.md
"""tools/pa_harden_run.py — hardening (FP-mining) fan-out su Ray, metrica pAUC_MCC. DELIVERABLE (gated).

Protocollo v6 (P6): la metrica di selezione/stop è la **pAUC_MCC parziale** (Orlova et al. 2025,
arXiv 2507.09338 — AUC_MCC; qui ristretta al band operativo FPR [0,5-1,5%] e normalizzata = MCC medio
sul band). Best-round = argmax pAUC_MCC; auto-stop **per-modello** quando Δ(pAUC_MCC) < c·σ_m, con σ_m =
dev-std di pAUC_MCC del modello sui semi base. NIENTE MCC_bal come driver (era il protocollo legacy).

Due fasi (una sola invocazione, Ray accoda):
  FASE 1 (pre-pass σ): eval forward-only dei modelli base (round 0) → pAUC_MCC per (candidato,seme) →
    σ_m per candidato (salvato in sigma.json). Barriera prima della fase 2.
  FASE 2 (hardening): traiettoria fp_mining per (candidato,seme) con stop_threshold = c·σ_m; salva la
    curva MCC(FPR) grezza per ogni round (trajectory.json) per il plotting a valle.

Pattern Ray di pa_compress: worker 1 CPU (max_retries=2), cache dati per-processo, parent_args via
ray.put (solo PATH), CSV incrementale + flush. Pesi-base da `--harden-base-dir` (synced VPS); pesi round
e trajectory.json in weights_home/harden/traj_<num>/seed_<s>/ (HOME), raccolti a valle.

Uso:
  python tools/pa_harden_run.py --candidates-json <dir>/hardening_candidates.json \
      --out-dir <dir esistente> --cache-dir <ABS>/runs/dae/output_569 \
      --labels-val <ABS>/val_Attack.npy --l1l2-csv <ABS>/l1l2_table.csv \
      --mlp-fixed-json <dummy> --harden-base-dir <ABS synced>/harden_base \
      [--seeds 42 ...] [--K 4 --n-max 5 --stop-sigma 1.0]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.secondo_stadio import constants as C  # noqa: E402

SUM_FIELDS = ["trial_id", "num", "seed", "sigma_m", "stop_threshold",
              "pauc_mcc_0", "pauc_mcc_best", "mcc_fpr1_0", "mcc_fpr1_best",
              "mcc_bal_0", "mcc_bal_best", "best_round", "n_rounds",
              "and_fpr_best", "and_dr_best", "fpr_best", "recall_best"]

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


def _dae_negpos(shared):
    return (shared["dae_score"][shared["ben_sel_abs"]],
            shared["dae_score"][shared["att_pos"]])


def _base_eval_fn(job: dict, parent_args: dict, tau_dae: float) -> dict:
    """FASE 1: eval forward-only del modello base (round 0) → pAUC_MCC (per σ_m)."""
    pr = parent_args.get("project_root")
    if pr and pr not in sys.path:
        sys.path.insert(0, pr)
    from lib.secondo_stadio.fp_mining import eval_weights
    shared = _get_shared(parent_args)
    dae_neg, dae_pos = _dae_negpos(shared)
    m0 = eval_weights(job["base_weights"], job["config"], parent_args["mlp_fixed"],
                      shared, dae_neg, dae_pos, tau_dae)
    return {"num": job["num"], "seed": job["seed"], "pauc_mcc": float(m0["pauc_mcc"])}


def _worker_fn(job: dict, parent_args: dict, tau_dae: float) -> dict:
    """FASE 2: una traiettoria = (candidato, seme), stop σ-aware su pAUC_MCC."""
    pr = parent_args.get("project_root")
    if pr and pr not in sys.path:
        sys.path.insert(0, pr)
    from lib.secondo_stadio.fp_mining import run_trajectory
    shared = _get_shared(parent_args)
    dae_neg, dae_pos = _dae_negpos(shared)
    traj_dir = (Path(parent_args["weights_home"]) / "harden"
                / f"traj_{job['num']}" / f"seed_{job['seed']}")
    out = run_trajectory(int(job["seed"]), job["config"], shared, dae_neg, dae_pos, tau_dae,
                         parent_args, parent_args["K"], float(job["stop_threshold"]),
                         parent_args["n_max"], traj_dir, tctx=None,
                         base_weights=job["base_weights"])
    traj_dir.mkdir(parents=True, exist_ok=True)
    (traj_dir / "trajectory.json").write_text(json.dumps(out))  # curve grezze incluse
    r0 = out["rounds"][0]
    rb = out["rounds"][out["best_round"]]
    return {"trial_id": job["trial_id"], "num": job["num"], "seed": job["seed"],
            "sigma_m": float(job["sigma_m"]), "stop_threshold": float(job["stop_threshold"]),
            "pauc_mcc_0": r0["pauc_mcc"], "pauc_mcc_best": rb["pauc_mcc"],
            "mcc_fpr1_0": r0.get("mcc_fpr1"), "mcc_fpr1_best": rb.get("mcc_fpr1"),
            "mcc_bal_0": r0["mcc_bal"], "mcc_bal_best": rb["mcc_bal"],
            "best_round": out["best_round"], "n_rounds": len(out["rounds"]) - 1,
            "and_fpr_best": rb.get("and_fpr"), "and_dr_best": rb.get("and_dr"),
            "fpr_best": rb["fpr"], "recall_best": rb["recall"]}


def already_done(csv_path: Path) -> set[tuple[str, int]]:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {(r["trial_id"], int(r["seed"])) for r in csv.DictReader(f)}


def _bw(base_dir: Path, num, seed) -> str:
    return str(base_dir / str(num) / f"seed_{seed}" / "best.weights.h5")


def compute_sigma(candidates, seeds, parent_args, tau_dae, base_dir, sigma_json, ray, pa_ref):
    """FASE 1: σ_m per candidato (pAUC_MCC dei modelli base sui semi). Cache in sigma_json."""
    if sigma_json.exists():
        return {int(k): v for k, v in json.loads(sigma_json.read_text()).items()}
    jobs = [{"num": c["num"], "seed": s, "config": c["config"],
             "base_weights": _bw(base_dir, c["num"], s)} for c in candidates for s in seeds]
    fn = ray.remote(num_cpus=1, max_retries=2)(_base_eval_fn)
    futures = {fn.remote(j, pa_ref, tau_dae): j for j in jobs}
    pauc = defaultdict(dict)
    pending = list(futures)
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        try:
            r = ray.get(done[0])
            pauc[r["num"]][r["seed"]] = r["pauc_mcc"]
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL σ] {futures[done[0]]['num']} s{futures[done[0]]['seed']}: {exc}")
    sigma = {num: float(stats.pstdev(list(v.values()))) for num, v in pauc.items() if len(v) > 1}
    sigma_json.write_text(json.dumps({"sigma": sigma,
                                      "pauc_base": {str(k): v for k, v in pauc.items()}}, indent=2))
    return sigma


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates-json", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--labels-val", required=True, type=Path)
    ap.add_argument("--l1l2-csv", required=True, type=Path)
    ap.add_argument("--mlp-fixed-json", required=True, type=Path)
    ap.add_argument("--harden-base-dir", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.SEEDS_PA))
    ap.add_argument("--weights-home", type=Path, default=Path.home() / "repov6_weights_s2")
    ap.add_argument("--ray-address", default="auto")
    ap.add_argument("--K", type=int, default=C.FPM_K)
    ap.add_argument("--n-max", type=int, default=C.FPM_N_MAX)
    ap.add_argument("--stop-sigma", type=float, default=C.FPM_STOP_SIGMA,
                    help="c in stop_threshold = c·σ_m (default 1σ)")
    ap.add_argument("--max-epochs", type=int, default=C.MLP_MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=C.MLP_PATIENCE)
    ap.add_argument("--es-n-attacks", type=int, default=C.ES_N_ATTACKS)
    args = ap.parse_args()
    if not args.out_dir.is_dir():
        raise SystemExit(f"out-dir inesistente (shell-first): {args.out_dir}")

    from lib.secondo_stadio.pa_search import build_parent_args
    from lib.secondo_stadio.data import load_tau_dae
    parent_args = build_parent_args({
        "cache_dir": str(args.cache_dir), "labels_val": str(args.labels_val),
        "mlp_fixed": json.loads(args.mlp_fixed_json.read_text()),
        "max_epochs": args.max_epochs, "patience": args.patience,
        "es_n_attacks": args.es_n_attacks,
        "save_weights": True, "weights_home": str(args.weights_home),
    })
    parent_args["project_root"] = str(Path(__file__).resolve().parent.parent)
    parent_args["K"] = args.K
    parent_args["n_max"] = args.n_max
    tau_dae = load_tau_dae(str(args.l1l2_csv))
    candidates = json.loads(args.candidates_json.read_text())  # #146 PRIMO nella lista

    import ray
    ray.init(address=args.ray_address, ignore_reinit_error=True,
             runtime_env={"env_vars": {"PYTHONPATH": parent_args["project_root"],
                                       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                       "OPENBLAS_NUM_THREADS": "1",
                                       "TF_NUM_INTEROP_THREADS": "1",
                                       "TF_NUM_INTRAOP_THREADS": "1",
                                       "TF_DETERMINISTIC_OPS": "1",
                                       "TF_CPP_MIN_LOG_LEVEL": "3"}})
    pa_ref = ray.put(parent_args)

    print(f"pa_harden: {len(candidates)} candidati x {len(args.seeds)} semi; "
          f"n_max={args.n_max} K={args.K} stop={args.stop_sigma}σ")
    print("FASE 1: pre-pass σ (pAUC_MCC modelli base)...")
    sigma = compute_sigma(candidates, args.seeds, parent_args, tau_dae,
                          args.harden_base_dir, args.out_dir / "sigma.json", ray, pa_ref)
    print(f"  σ_m calcolate per {len(sigma)} candidati "
          f"(min {min(sigma.values()):.4f} / max {max(sigma.values()):.4f})")

    csv_path = args.out_dir / "harden_summary.csv"
    done = already_done(csv_path)
    jobs = [{"trial_id": c["trial_id"], "num": c["num"], "config": c["config"], "seed": int(s),
             "base_weights": _bw(args.harden_base_dir, c["num"], s),
             "sigma_m": sigma.get(c["num"], 0.0),
             "stop_threshold": args.stop_sigma * sigma.get(c["num"], 0.0)}
            for c in candidates for s in args.seeds if (c["trial_id"], s) not in done]
    print(f"FASE 2: hardening — {len(jobs)} traiettorie da eseguire")
    if not jobs:
        ray.shutdown()
        return

    worker = ray.remote(num_cpus=1, max_retries=2)(_worker_fn)
    futures = {worker.remote(j, pa_ref, tau_dae): j for j in jobs}
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUM_FIELDS)
        if new_file:
            w.writeheader()
            f.flush()
        pending = list(futures)
        while pending:
            dref, pending = ray.wait(pending, num_returns=1)
            job = futures[dref[0]]
            try:
                row = ray.get(dref[0])
                w.writerow(row)
                f.flush()
                print(f"  #{job['num']} s{job['seed']}: pAUC {row['pauc_mcc_0']:.4f}->"
                      f"{row['pauc_mcc_best']:.4f} | mcc@1% {row['mcc_fpr1_0']:.4f}->"
                      f"{row['mcc_fpr1_best']:.4f} (r{row['best_round']}, σ{row['sigma_m']:.4f})")
            except Exception as exc:  # noqa: BLE001
                print(f"[FAIL] #{job['num']} s{job['seed']} {job['trial_id']}: {exc}")
    ray.shutdown()
    print(f"pa_harden: summary in {csv_path}")


if __name__ == "__main__":
    main()
