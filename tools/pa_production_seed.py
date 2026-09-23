#!/usr/bin/env python
# TAG: ONE-SHOT | 2026-07-13 | seme di produzione PA (citato nel report) | vedi docs/refactoring_censimento.md
"""tools/pa_production_seed.py — seme di produzione del PA #589 (mediano composito a 19 semi).

Convenzione DAE-2034: min-max per asse, somma dei 3 normalizzati, rango 10/19 (mediano). Assi:
mccbal, mcc@FPR1%, overall-DR LOAO (micro, pesato per n). Dati dai CSV della compressione, hardening
e LOAO già prodotti. ONE-SHOT.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
S2 = REPO / "runs/secondo_stadio"

SEEDS_BASE = [42, 123, 2024, 7, 99, 1337, 2036, 2037]
SEEDS_EXTRA = list(range(2038, 2049))
ALL_SEEDS = SEEDS_BASE + SEEDS_EXTRA


def load_harden_8() -> dict[int, dict]:
    """mccbal + mcc_fpr1 per gli 8 semi base (round-0 = base, hardening best_round=0)."""
    out = {}
    for r in csv.DictReader(open(S2 / "harden/harden_summary.csv")):
        if r["num"] != "589":
            continue
        s = int(r["seed"])
        out[s] = {"mccbal": float(r["mcc_bal_0"]), "mccfpr": float(r["mcc_fpr1_0"])}
    return out


def load_compress_11() -> dict[int, dict]:
    """mccbal + mcc_fpr1 per gli 11 semi extra (pa_compress)."""
    out = {}
    for r in csv.DictReader(open(S2 / "pa_compress/pa_compress.csv")):
        if "_21a2e100_589_" not in r["trial_id"]:
            continue
        s = int(r["seed"])
        out[s] = {"mccbal": float(r["mcc_bal"]), "mccfpr": float(r["mcc_fpr1"])}
    return out


def load_loao_dr(seed: int) -> float:
    """Overall-DR micro (pesato per n) dal CSV LOAO per-seme."""
    p = S2 / f"loao_pa/rows/589_seed{seed}.csv"
    rows = list(csv.DictReader(open(p)))
    tot_n = sum(int(r["n"]) for r in rows)
    return sum(float(r["dr_primary"]) * int(r["n"]) for r in rows) / tot_n if tot_n else float("nan")


def main() -> None:
    h8 = load_harden_8()
    c11 = load_compress_11()
    data = {}
    for s in ALL_SEEDS:
        if s in h8:
            data[s] = h8[s]
        elif s in c11:
            data[s] = c11[s]
        else:
            raise SystemExit(f"seme {s}: nessun dato mccbal/mccfpr")
        data[s]["dr"] = load_loao_dr(s)
    seeds = sorted(data.keys())
    axes = ["mccbal", "mccfpr", "dr"]
    raw = {ax: np.array([data[s][ax] for s in seeds]) for ax in axes}
    normed = {}
    for ax in axes:
        v = raw[ax]
        lo, hi = v.min(), v.max()
        normed[ax] = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    total = sum(normed[ax] for ax in axes)
    order = np.argsort(-total)
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(seeds) + 1)

    print("=" * 100)
    print(f"SEME DI PRODUZIONE PA #589 — mediano composito a {len(seeds)} semi "
          f"(min-max per asse, somma, rango {len(seeds) // 2 + 1}/{len(seeds)})")
    print("=" * 100)
    hdr = f"{'seed':>6} {'mccbal':>9} {'mccfpr':>9} {'DR-LOAO':>9} | {'norm_mb':>7} {'norm_mf':>7} {'norm_dr':>7} | {'somma':>7} {'rango':>5}"
    print(hdr)
    print("-" * len(hdr))
    med_rank = len(seeds) // 2 + 1
    chosen = None
    for i, s in enumerate(seeds):
        r = rank[i]
        star = " ★" if r == med_rank else ""
        if r == med_rank:
            chosen = s
        base = "B" if s in SEEDS_BASE else "E"
        print(f"  {s:>4}{base} {data[s]['mccbal']:.5f} {data[s]['mccfpr']:.5f} {data[s]['dr']:.5f} | "
              f"{normed['mccbal'][i]:.4f}  {normed['mccfpr'][i]:.4f}  {normed['dr'][i]:.4f}  | "
              f"{total[i]:.4f}   {r:>3}{star}")

    print(f"\nSEME DI PRODUZIONE = {chosen} (rango {med_rank}/{len(seeds)})")
    i = seeds.index(chosen)
    print(f"  mccbal={data[chosen]['mccbal']:.5f}  mcc@1%={data[chosen]['mccfpr']:.5f}  DR-LOAO={data[chosen]['dr']:.5f}")
    w_path = S2 / f"weights_final/21a2e100/seed_{chosen}/best.weights.h5"
    if w_path.exists():
        print(f"  pesi DISPONIBILI: {w_path.relative_to(REPO)}")
    else:
        print(f"  pesi MANCANTI (seme extra, nodi Ray smantellati): {w_path}")
        for sb in SEEDS_BASE:
            wp = S2 / f"weights_final/21a2e100/seed_{sb}/best.weights.h5"
            if wp.exists():
                pass
        med_base = sorted([(rank[seeds.index(sb)], sb) for sb in SEEDS_BASE])[len(SEEDS_BASE) // 2]
        print(f"  FALLBACK: mediano fra i soli {len(SEEDS_BASE)} base = seme {med_base[1]} (rango {med_base[0]})")
    print("=" * 100)


if __name__ == "__main__":
    main()
