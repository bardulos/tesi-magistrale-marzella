#!/usr/bin/env python
# TAG: DRIVER-PIPELINE | congelamento rosa K
"""Freeze the top-K shortlist: smallest K whose PCS (fixed seed) covers the top-t at confidence conf.

Combines and deduplicates completed trials from search_600 and search_400, writes the top-K CSV,
and estimates the extra seed count needed to reduce the shortlist to K'.

Usage: python tools/select_shortlist.py --search-600 DIR --search-400 DIR --out FILE
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.dae.constants import MAX_T  # noqa: E402
from lib.search.shortlist import (pcs_curve, pooled_sigma,  # noqa: E402
                                  seeds_for_target_K, smallest_K)

DRAWS = 50_000
RNG_SEED = 99


# ===================================================================== logica pura ====

def compute_shortlist(rows: list[dict], t: int = 3, conf: float = 0.80,
                      draws: int = DRAWS, rng_seed: int = RNG_SEED) -> dict:
    """Compute K at the requested confidence and return finalists ordered by mean AP."""
    if not rows:
        raise ValueError("pool vuoto: nessun trial COMPLETE")
    ordered = sorted(rows, key=lambda r: -r["ap_mean"])
    sigma, se, nu = pooled_sigma([r["ap_std"] for r in ordered], MAX_T)
    means_sorted = np.array([r["ap_mean"] for r in ordered])
    rng = np.random.default_rng(rng_seed)
    curve = pcs_curve(means_sorted, se, t, draws, rng)
    K = smallest_K(curve, conf)
    return {"K": int(K), "sigma": float(sigma), "se": float(se), "nu": int(nu),
            "top": ordered[:K], "pcs_at_K": float(curve.get(K, float("nan")))}


# ===================================================================== reader ====

def _cfg_key(cfg: dict) -> str:
    """Build the monitor-compatible configuration key."""
    return (f"{int(cfg['exp'])}_{int(cfg['mid'])}_{int(cfg['btl'])}_"
            f"{float(cfg['sigma']):.5f}_{float(cfg['lr']):.6f}")


def read_completed_with_dir(out_dir: Path, source: str) -> list[dict]:
    """Read complete trials with their Ray directory and source label."""
    base = Path(out_dir) / "ray_results/dae_search"
    rows = []
    for d in base.glob("_run_trainable_*"):
        rj, pj = d / "result.json", d / "params.json"
        if not (rj.exists() and pj.exists()):
            continue
        try:
            last = json.loads([l for l in rj.read_text().splitlines() if l.strip()][-1])
            if int(last.get("seeds_completed", 0)) != MAX_T:
                continue
            cfg = json.loads(pj.read_text())
            rows.append({
                "ap_mean": float(last.get("ap_mean", float("nan"))),
                "ap_std":  float(last.get("ap_std", float("nan"))),
                "obj":     float(last["objective"]),
                "cfg":     cfg,
                "cfg_key": _cfg_key(cfg),
                "ray_dir": str(d),
                "source":  source,
            })
        except Exception:  # noqa: BLE001
            continue
    return rows


def build_pool(search_600: Path, search_400: Path) -> list[dict]:
    """Merge pools by config key, preferring search_600 trials."""
    new_rows = read_completed_with_dir(search_600, "new")
    new_keys = {r["cfg_key"] for r in new_rows}
    old_rows = [r for r in read_completed_with_dir(search_400, "import")
                if r["cfg_key"] not in new_keys]
    return new_rows + old_rows


# ===================================================================== driver ====

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search-600", required=True)
    ap.add_argument("--search-400", required=True)
    ap.add_argument("--t", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.80)
    ap.add_argument("--kprime", type=int, default=25)
    ap.add_argument("--out", required=True, help="CSV di output (dir deve esistere)")
    args = ap.parse_args(argv)

    pool = build_pool(Path(args.search_600), Path(args.search_400))
    n_new = sum(1 for r in pool if r["source"] == "new")
    res = compute_shortlist(pool, t=args.t, conf=args.conf)
    K = res["K"]
    top = res["top"]
    n_new_top = sum(1 for r in top if r["source"] == "new")

    means_sorted = np.array([r["ap_mean"] for r in sorted(pool, key=lambda r: -r["ap_mean"])])
    nprime = seeds_for_target_K(means_sorted, res["sigma"], args.t, args.conf,
                                args.kprime, DRAWS, RNG_SEED)
    extra = (nprime - MAX_T) if nprime else None
    budget = (K * extra) if extra else None

    out = Path(args.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "trial_id", "source", "ap_mean", "ap_std", "ray_dir"])
        w.writeheader()
        for rank, r in enumerate(top, 1):
            w.writerow({"rank": rank, "trial_id": Path(r["ray_dir"]).name,
                        "source": r["source"], "ap_mean": f"{r['ap_mean']:.6f}",
                        "ap_std": f"{r['ap_std']:.6f}", "ray_dir": r["ray_dir"]})

    print(f"Pool COMPLETE: {len(pool)} ({n_new} nuovi + {len(pool)-n_new} importati)")
    print(f"sigma={res['sigma']:.4f}  SE={res['se']:.4f}  (nu={res['nu']})")
    print(f"K (top-{args.t} @{args.conf:.2f}, {DRAWS} draw, seed {RNG_SEED}) = {K}  "
          f"(PCS={res['pcs_at_K']:.3f})  comp: {n_new_top} nuovi + {K-n_new_top} importati")
    print(f"n' per K'={args.kprime}: {nprime}  → extra={extra}  budget=K×extra={budget} addestramenti")
    print(f"CSV shortlist → {out}")


if __name__ == "__main__":
    main()
