# TAG: DRIVER-PIPELINE | 2026-07-13 | pre-cull + PCS rosa PA P4 (testato) | vedi docs/refactoring_censimento.md
"""tools/pa_shortlist.py — pre-cull e dimensionamento PCS della rosa PA (P4, cap3). DELIVERABLE.

Due sotto-comandi:
  precull  — dai risultati della search PA (trial a max_t=8 semi) elimina i trial a MCC_bal
             troppo basso (soglia --min-mccbal, decisa al GATE C sui numeri reali) e scrive
             candidates.json (input di tools/pa_compress.py) + precull.csv.
  pcs      — costruisce la matrice per-seme (8 base da result.json/seed_order.json + extra
             dal CSV di pa_compress), RI-VERIFICA le assunzioni della PCS sul pool MCC_bal
             (GATE C: le assunzioni NON si ereditano dal cap2, si rimisurano — Levene
             sull'omogeneita', correlazione inter-configurazione, allineamento indice
             ricerca/decisione) e calcola sigma pooled, curva PCS (indipendente e appaiata)
             e smallest_K. K' finale (3-5) si congela al GATE C: il tool riporta, non decide.

Riuso: lib/search/shortlist.py (pooled_sigma, pcs_curve, pcs_curve_paired, smallest_K,
variance_homogeneity — metrica-agnostici) e lib/search/perseed (niente ricostruzione: i
per-seme si leggono da mccbal_last riga-per-riga, mappati sui semi fisici via seed_order.json).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from lib.secondo_stadio import constants as C  # noqa: E402

HP_KEYS = ("exp", "n_layers", "ratio", "dropout", "lr", "delta", "alpha_lo", "alpha_hi")


# ------------------------------------------------------------- parti pure ----

def load_pa_records(search_dir: Path) -> list[dict]:
    """Scansiona ray_results/*/: params.json + TUTTE le righe di result.json. Ritorna
    [{trial_id, config, lines}]. NB: `seed_order.json` NON è affidabile (l'engine lo scrive in
    cwd sui worker Ray → assente/vuoto in ray_results); la mappa seme è nelle righe stesse —
    ogni riga porta `seed_last` (seme fisico) e `mccbal_last`, letti da perseed_from_lines."""
    records = []
    for exp_dir in sorted((search_dir / "ray_results").glob("*")):
        for tdir in sorted(exp_dir.glob("*_*")):
            pj, rj = tdir / "params.json", tdir / "result.json"
            if not (pj.exists() and rj.exists()):
                continue
            lines = [json.loads(ln) for ln in rj.read_text().splitlines() if ln.strip()]
            if not lines:
                continue
            records.append({"trial_id": tdir.name,
                            "config": json.loads(pj.read_text()),
                            "lines": lines})
    return records


def precull_rows(records: list[dict], min_mccbal: float, max_t: int = 8) -> list[dict]:
    """Trial a max_t semi con mccbal_mean >= soglia, ordinati per objective decrescente."""
    rows = []
    for rec in records:
        last = rec["lines"][-1]
        if int(last.get("seeds_completed", 0)) != int(max_t):
            continue
        if float(last.get("mccbal_mean", float("nan"))) < float(min_mccbal):
            continue
        rows.append({"trial_id": rec["trial_id"],
                     "config": {k: rec["config"][k] for k in HP_KEYS},
                     "objective": float(last["objective"]),
                     "mccbal_mean": float(last["mccbal_mean"]),
                     "mccbal_std": float(last.get("mccbal_std", float("nan"))),
                     "n_genuine": int(last.get("n_genuine", 0))})
    return sorted(rows, key=lambda r: -r["objective"])


def perseed_from_lines(lines: list[dict]) -> dict[int, float]:
    """MCC_bal per seme FISICO dalle righe di report: ogni riga porta `seed_last` (il seme
    fisico appena valutato) e `mccbal_last` (il suo MCC_bal). Sorgente diretta, senza
    seed_order.json (verificato: le 8 righe coprono i SEEDS_PA permutati)."""
    out = {}
    for ln in lines:
        if "seed_last" in ln and "mccbal_last" in ln:
            out[int(ln["seed_last"])] = float(ln["mccbal_last"])
    return out


def build_matrix(records: list[dict], extra_rows: list[dict],
                 seeds_base: list[int], seeds_extra: list[int],
                 trial_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Matrice (trial x [base+extra]) di MCC_bal per-seme; solo i trial con TUTTI i semi.
    extra_rows: righe del CSV di pa_compress (trial_id, seed, mcc_bal)."""
    by_trial_extra: dict[str, dict[int, float]] = {}
    for r in extra_rows:
        by_trial_extra.setdefault(str(r["trial_id"]), {})[int(r["seed"])] = float(r["mcc_bal"])
    rec_by_id = {r["trial_id"]: r for r in records}
    all_seeds = list(seeds_base) + list(seeds_extra)
    rows, kept = [], []
    for tid in trial_ids:
        rec = rec_by_id.get(tid)
        if rec is None:
            continue
        base = perseed_from_lines(rec["lines"])
        extra = by_trial_extra.get(tid, {})
        vals = [base.get(s, extra.get(s)) for s in all_seeds]
        if any(v is None or not np.isfinite(v) for v in vals):
            continue
        rows.append(vals)
        kept.append(tid)
    return np.asarray(rows, dtype=float), kept


def check_assumptions(matrix: np.ndarray) -> dict:
    """RI-VERIFICA GATE C delle assunzioni PCS sul pool MCC_bal:
    (1) omogeneita' delle varianze (Levene — attenzione all'eteroschedasticita' da metrica
        limitata vicino a saturazione); (2) correlazione inter-configurazione media sui semi
    (l'appaiamento aiuta solo se >0); (3) allineamento indice di ricerca (mean−std) e indice
    di decisione (mean): rho di Spearman sui ranghi dei trial."""
    from scipy.stats import spearmanr

    from lib.search.shortlist import variance_homogeneity
    lev_stat, lev_p = variance_homogeneity(matrix)
    n = len(matrix)
    cors = []
    for i in range(n):
        for j in range(i + 1, n):
            c = np.corrcoef(matrix[i], matrix[j])[0, 1]
            if np.isfinite(c):
                cors.append(c)
    mean_corr = float(np.mean(cors)) if cors else float("nan")
    means = matrix.mean(axis=1)
    search_idx = means - matrix.std(axis=1, ddof=0)
    rho = float(spearmanr(means, search_idx).statistic) if n >= 3 else float("nan")
    return {"levene_stat": float(lev_stat), "levene_p": float(lev_p),
            "mean_interconfig_corr": mean_corr,
            "spearman_mean_vs_meanstd": rho, "n_trials": int(n),
            "n_seeds": int(matrix.shape[1]) if matrix.ndim == 2 else 0}


# ----------------------------------------------------------------- driver ----

def cmd_precull(args) -> None:
    records = load_pa_records(args.search_dir)
    rows = precull_rows(records, args.min_mccbal, max_t=args.max_t)
    print(f"precull: {len(records)} trial letti -> {len(rows)} sopra mccbal_mean >= "
          f"{args.min_mccbal} (a {args.max_t} semi)")
    (args.out_dir / "candidates.json").write_text(json.dumps(rows, indent=2))
    with open(args.out_dir / "precull.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial_id", "objective", "mccbal_mean", "mccbal_std", "n_genuine",
                    *HP_KEYS])
        for r in rows:
            w.writerow([r["trial_id"], r["objective"], r["mccbal_mean"], r["mccbal_std"],
                        r["n_genuine"], *[r["config"][k] for k in HP_KEYS]])


def cmd_pcs(args) -> None:
    from lib.search.shortlist import (pcs_curve, pcs_curve_paired, pooled_sigma,
                                      smallest_K)
    records = load_pa_records(args.search_dir)
    candidates = json.loads(args.candidates_json.read_text())
    with open(args.extra_csv, newline="") as f:
        extra_rows = [r for r in csv.DictReader(f)
                      if not r.get("filter_reason")]  # semi non-apprendenti esclusi
    trial_ids = [c["trial_id"] for c in candidates]
    matrix, kept = build_matrix(records, extra_rows, list(C.SEEDS_PA),
                                list(args.seeds_extra), trial_ids)
    print(f"matrice per-seme: {matrix.shape[0]} trial completi x {matrix.shape[1]} semi "
          f"({len(trial_ids) - len(kept)} scartati per semi mancanti)")
    if matrix.shape[0] < 3:
        raise SystemExit("meno di 3 trial completi: PCS non calcolabile")

    assum = check_assumptions(matrix)
    print("assunzioni (GATE C):", json.dumps(assum, indent=2))

    means = matrix.mean(axis=1)
    stds = matrix.std(axis=1, ddof=0)
    order = np.argsort(-means)
    means_sorted = means[order]
    kept_sorted = [kept[i] for i in order]

    sigma, se, nu = pooled_sigma(stds.tolist(), matrix.shape[1])
    rng = np.random.default_rng(args.rng_seed)
    curve = pcs_curve(means_sorted, se, t=args.top_t, draws=args.draws, rng=rng)
    k_at = smallest_K(curve, args.pcs_threshold)
    rng_p = np.random.default_rng(args.rng_seed)
    curve_p = pcs_curve_paired(matrix[order], t=args.top_t, draws=args.draws, rng=rng_p)
    k_at_p = smallest_K(curve_p, args.pcs_threshold)
    print(f"sigma_pooled={sigma:.5f} SE={se:.5f} | K@{args.pcs_threshold} "
          f"(indip)={k_at} (appaiata)={k_at_p} | K' finale (3-5) al GATE C")

    with open(args.out_dir / "pcs_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "pcs_indep", "pcs_paired"])
        for k in sorted(curve):
            w.writerow([k, curve[k], curve_p.get(k, float("nan"))])
    with open(args.out_dir / "shortlist_rank.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "trial_id", "mccbal_mean_full", "mccbal_std_full"])
        for i, tid in enumerate(kept_sorted, 1):
            w.writerow([i, tid, means_sorted[i - 1], stds[order][i - 1]])
    (args.out_dir / "pcs_summary.json").write_text(json.dumps(
        {"sigma_pooled": sigma, "se": se, "nu_pool": nu,
         "K_at_threshold_indep": k_at, "K_at_threshold_paired": k_at_p,
         "pcs_threshold": args.pcs_threshold, "top_t": args.top_t,
         "draws": args.draws, "rng_seed": args.rng_seed,
         "assunzioni_gate_c": assum,
         "nota": "K' finale (3-5) si congela al GATE C sui numeri reali."}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("precull")
    p1.add_argument("--search-dir", required=True, type=Path)
    p1.add_argument("--out-dir", required=True, type=Path)
    p1.add_argument("--min-mccbal", required=True, type=float,
                    help="soglia di pre-cull (GATE C, sui numeri reali)")
    p1.add_argument("--max-t", type=int, default=8)

    p2 = sub.add_parser("pcs")
    p2.add_argument("--search-dir", required=True, type=Path)
    p2.add_argument("--candidates-json", required=True, type=Path)
    p2.add_argument("--extra-csv", required=True, type=Path)
    p2.add_argument("--out-dir", required=True, type=Path)
    p2.add_argument("--seeds-extra", type=int, nargs="+", default=list(C.SEEDS_PA_EXTRA))
    p2.add_argument("--pcs-threshold", type=float, default=0.80)
    p2.add_argument("--top-t", type=int, default=3)
    p2.add_argument("--draws", type=int, default=50_000)
    p2.add_argument("--rng-seed", type=int, default=99)

    args = ap.parse_args()
    if not args.out_dir.is_dir():
        raise SystemExit(f"out-dir inesistente (shell-first): {args.out_dir}")
    (cmd_precull if args.cmd == "precull" else cmd_pcs)(args)


if __name__ == "__main__":
    main()
