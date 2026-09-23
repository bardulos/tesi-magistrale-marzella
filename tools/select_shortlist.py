#!/usr/bin/env python
# TAG: DRIVER-PIPELINE | 2026-07-13 | congelamento rosa K FASE 2 (testato) | vedi docs/refactoring_censimento.md
"""tools/select_shortlist.py — FASE 2 passo C: congela la rosa K (top-t @conf) sul pool finale.

Metodo CANONICO (coerente col monitor e con le macro LaTeX): pool unito search_600+search_400
(dedup per cfg_key), PCS Monte Carlo top-t a confidenza fissa, draws/seme fissi → K riproducibile.
Emette il CSV dei top-K (con path del ray_dir di ogni finalista, per leggere params.json +
result.json a valle in refit_compress) e calcola n' per comprimere a K' (seeds_for_target_K).

Logica pura testabile: `compute_shortlist`. Il main legge da disco e scrive il CSV.

Uso:
  python tools/select_shortlist.py \\
      --search-600 runs/dae/search_600 --search-400 runs/dae/search_400 \\
      --t 3 --conf 0.80 --kprime 25 --out runs/dae/compress_fase2/shortlist.csv
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

DRAWS = 50_000     # estrazioni Monte Carlo (canonico monitor)
RNG_SEED = 99      # seme RNG fisso → K riproducibile


# ===================================================================== logica pura ====

def compute_shortlist(rows: list[dict], t: int = 3, conf: float = 0.80,
                      draws: int = DRAWS, rng_seed: int = RNG_SEED) -> dict:
    """Calcola K (top-t @conf) e restituisce i top-K ordinati per ap_mean discendente.

    rows: lista di dict con almeno "ap_mean" e "ap_std" (campi extra passano inalterati).
    Ritorna {K, sigma, se, nu, top: rows_ordinati[:K]}.
    """
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
    """Chiave di identità della config (come monitor_search.cfg_key)."""
    return (f"{int(cfg['exp'])}_{int(cfg['mid'])}_{int(cfg['btl'])}_"
            f"{float(cfg['sigma']):.5f}_{float(cfg['lr']):.6f}")


def read_completed_with_dir(out_dir: Path, source: str) -> list[dict]:
    """Come monitor_search.read_completed ma include ray_dir (path) e source.
    Trial COMPLETE = seeds_completed == MAX_T."""
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
    """Pool unito dedup per cfg_key: i NUOVI (search_600) hanno precedenza sugli importati."""
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

    # n' per comprimere a K'
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
            # trial_id = nome del ray_dir (chiave robusta per params.json/result.json a valle)
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
