#!/usr/bin/env python
# TAG: ONE-SHOT | selezione a tre assi
"""Build a three-axis comparison table for the 25 PA finalists, without training.

Combines compression MCC_bal, hardened pAUC_MCC, and LOAO overall-DR with ranks and top-three
markers for each axis. Writes a CSV and displays rank-sum and hardened-pAUC orderings.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
S2 = REPO / "runs/secondo_stadio"


def ranks(vals: dict, reverse=True) -> dict:
    """Return 1-based ranks, with high values first by default."""
    order = sorted(vals, key=lambda k: vals[k], reverse=reverse)
    return {k: i + 1 for i, k in enumerate(order)}


def main() -> None:
    fin = sorted((str(x["num"]) for x in
                  json.load(open(S2 / "pa_compress/hardening_candidates.json"))), key=int)

    a1 = {r["trial"]: r for r in csv.DictReader(open(S2 / "pa_compress/top25_mccbal.csv"))}
    mccbal = {n: float(a1[n]["mccbal_mean"]) for n in fin}
    mccbal_sd = {n: float(a1[n]["mccbal_std"]) for n in fin}
    npar = {n: int(a1[n]["n_par"]) for n in fin}
    nL = {n: int(a1[n]["nL"]) for n in fin}

    byn = defaultdict(list)
    for r in csv.DictReader(open(S2 / "harden/harden_summary.csv")):
        byn[r["num"]].append(r)
    pauc = {n: float(np.mean([float(x["pauc_mcc_best"]) for x in byn[n]])) for n in fin}
    pauc_sd = {n: float(np.std([float(x["pauc_mcc_best"]) for x in byn[n]])) for n in fin}
    eng = {n: sum(1 for x in byn[n] if int(x["best_round"]) > 0) for n in fin}
    n_h = {n: len(byn[n]) for n in fin}

    l3 = {r["num"]: r for r in csv.DictReader(open(S2 / "loao_pa/loao_pa_summary.csv"))}
    odr = {n: float(l3[n]["overall_dr_p_mean"]) for n in fin}
    odr_sd = {n: float(l3[n]["overall_dr_p_std"]) for n in fin}
    zdr = {n: float(l3[n]["z_dr_p_mean"]) for n in fin}
    buchi = {n: int(l3[n]["buchi"]) for n in fin}
    cop = {n: int(l3[n]["coperte12"]) for n in fin}

    r1, r2, r3 = ranks(mccbal), ranks(pauc), ranks(odr)
    rows = []
    for n in fin:
        top = [r1[n] <= 3, r2[n] <= 3, r3[n] <= 3]
        rows.append({
            "num": n, "nL": nL[n], "n_par": npar[n],
            "mccbal_mean": mccbal[n], "mccbal_std": mccbal_sd[n], "rank_mccbal": r1[n],
            "pauc_h_mean": pauc[n], "pauc_h_std": pauc_sd[n], "eng": f"{eng[n]}/{n_h[n]}", "rank_pauc": r2[n],
            "overall_dr_mean": odr[n], "overall_dr_std": odr_sd[n], "z_dr": zdr[n],
            "buchi": buchi[n], "coperte12": cop[n], "rank_loao": r3[n],
            "top3_mccbal": int(top[0]), "top3_pauc": int(top[1]), "top3_loao": int(top[2]),
            "n_top3_assi": sum(top), "somma_ranghi": r1[n] + r2[n] + r3[n],
        })

    out = S2 / "loao_pa/tre_assi_selezione.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def render(rs, title):
        print("\n" + "=" * 118)
        print(title)
        print("=" * 118)
        h = (f"{'trial':>6} {'nL':>2} {'n_par':>6} | {'mccbal±σ':>16} {'r1':>3} | "
             f"{'pAUC-h±σ':>16} {'eng':>4} {'r2':>3} | {'ovDR±σ':>15} {'ZDR':>5} {'bk':>2} {'cp':>4} {'r3':>3} | "
             f"{'top3':>5} {'Σr':>3}")
        print(h); print("-" * len(h))
        for r in rs:
            t = "".join(("M" if r["top3_mccbal"] else "·", "P" if r["top3_pauc"] else "·",
                         "L" if r["top3_loao"] else "·"))
            star = " ★" if r["n_top3_assi"] >= 2 else ""
            print(f"#{r['num']:>5} {r['nL']:>2} {r['n_par']:>6} | "
                  f"{r['mccbal_mean']:.4f}±{r['mccbal_std']:.4f} {r['rank_mccbal']:>3} | "
                  f"{r['pauc_h_mean']:.4f}±{r['pauc_h_std']:.4f} {r['eng']:>4} {r['rank_pauc']:>3} | "
                  f"{r['overall_dr_mean']:.3f}±{r['overall_dr_std']:.3f} {r['z_dr']:.3f} "
                  f"{r['buchi']:>2} {r['coperte12']:>2}/12 {r['rank_loao']:>3} | "
                  f"{t:>5} {r['somma_ranghi']:>3}{star}")

    render(sorted(rows, key=lambda r: r["somma_ranghi"]),
           "TRE ASSI — ordinati per SOMMA DEI RANGHI  (top3: M=mccbal P=pAUC-hardened L=LOAO; ★ = top3 su ≥2 assi)")
    render(sorted(rows, key=lambda r: r["rank_pauc"]),
           "TRE ASSI — ordinati per pAUC-HARDENED (asse deploy, il più discriminante)")

    ge2 = [r["num"] for r in rows if r["n_top3_assi"] >= 2]
    ge3 = [r["num"] for r in rows if r["n_top3_assi"] == 3]
    print(f"\ntop3 su ≥2 assi: {ge2 or '—'}   ·   top3 su TUTTI E TRE: {ge3 or '—'}")
    print(f"CSV: {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
