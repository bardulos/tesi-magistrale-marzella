#!/usr/bin/env python3
# TAG: ONE-SHOT | 2026-07-13 | aggregazione LOAO PA | vedi docs/refactoring_censimento.md
"""tools/pa_loao_aggregate.py — aggregazione LOAO PA (offline, dai CSV parziali; niente TF/forward).

Simmetrica al cap2 (lib/dae/loao). Produce:
  (1) CSV unito loao_pa_rows.csv;
  (2) tabella 25x: overall-DR (micro) e Z-DR (macro) media±dev.std sui 19 semi, + buchi e coperte/12
      (criterio graduato), primaria (p99 benigni-select) e secondaria (deploy-realistica);
  (3) record per-classe con distribuzione per-seme (per le figure a punti);
  (4) istogramma dei d-logit del PA + soglia di separazione DICHIARATA sui dati del PA (NON 0,85 del DAE);
  (5) criterio graduato: buco strutturale = mediana-DR<0,10 E mediana-d<soglia; Infilteration esclusa se
      il massimo sui modelli della sua mediana-DR e' <0,10 (12 trattabili).

NON seleziona il vincitore (solo il dato LOAO). Uso: python tools/pa_loao_aggregate.py [--d-threshold X]
"""
from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BENIGN = "Benign"
DR_HOLE, DR_COVER = 0.10, 0.50            # soglie DR (cap2): buco <0,10, copertura robusta >=0,50
DISPLAY = {"Infilteration": "Infiltration"}   # typo del dataset → grafia corretta nel display (dato invariato)


def disp(c: str) -> str:
    return DISPLAY.get(c, c)


def load_rows(rows_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(rows_dir / "*.csv"))):
        for r in csv.DictReader(open(f)):
            for k in ("dr_primary", "dr_secondary", "d_logit", "tau_primary", "tau_secondary"):
                r[k] = float(r[k])
            r["n"] = int(r["n"])
            r["seed"] = int(r["seed"])
            rows.append(r)
    return rows


def declare_d_threshold(med_d: list[float]) -> float:
    """Soglia di separazione dai dati del PA (distribuzione bimodale: cluster debole vs forte). Gap
    piu' ampio nella regione bassa POSITIVA (0 < d < mediana) della mediana-d per (modello, classe
    trattabile) → punto medio. Il filtro d>0 scarta il rumore mean-based delle classi minuscole
    (es. SQL_Injection, n=171, d negativo pur con DR>0) che falserebbe il gap."""
    v = np.sort(np.asarray(med_d, dtype=float))
    lo = v[(v > 0) & (v < np.median(v))]
    if len(lo) < 2:
        return float(np.median(v))
    i = int(np.argmax(np.diff(lo)))
    return float((lo[i] + lo[i + 1]) / 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(REPO / "runs/secondo_stadio/loao_pa"))
    ap.add_argument("--d-threshold", type=float, default=None, help="soglia d (default: auto dal gap)")
    a = ap.parse_args()
    out = Path(a.out_dir)

    rows = load_rows(out / "rows")
    if not rows:
        raise SystemExit(f"nessun CSV parziale in {out/'rows'}")
    nums = sorted(set(r["num"] for r in rows), key=int)
    seeds = sorted(set(r["seed"] for r in rows))
    classes = sorted(set(r["classe"] for r in rows))
    print(f"celle: {len(rows)}  ·  modelli: {len(nums)}  ·  semi: {len(seeds)}  ·  classi: {len(classes)}")

    # CSV unito
    with open(out / "loao_pa_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # indicizzazione
    by_ns = defaultdict(list)          # (num, seed) -> righe (13 classi)
    by_nc = defaultdict(list)          # (num, classe) -> righe (19 semi)
    for r in rows:
        by_ns[(r["num"], r["seed"])].append(r)
        by_nc[(r["num"], r["classe"])].append(r)

    def overall(rs, key):              # micro: tutti gli attacchi uniti
        tot = sum(x["n"] for x in rs)
        return sum(x[key] * x["n"] for x in rs) / tot if tot else float("nan")

    def zdr(rs, key):                  # macro: media per-classe
        return float(np.mean([x[key] for x in rs]))

    # overall/Z-DR per (num, seed) → aggregati sui semi
    per_model = {}
    for num in nums:
        o_p = [overall(by_ns[(num, s)], "dr_primary") for s in seeds if (num, s) in by_ns]
        z_p = [zdr(by_ns[(num, s)], "dr_primary") for s in seeds if (num, s) in by_ns]
        o_s = [overall(by_ns[(num, s)], "dr_secondary") for s in seeds if (num, s) in by_ns]
        z_s = [zdr(by_ns[(num, s)], "dr_secondary") for s in seeds if (num, s) in by_ns]
        per_model[num] = {"o_p": np.array(o_p), "z_p": np.array(z_p),
                          "o_s": np.array(o_s), "z_s": np.array(z_s), "n_seeds": len(o_p)}

    # mediane per (num, classe)
    med_dr = {(num, c): float(np.median([x["dr_primary"] for x in by_nc[(num, c)]]))
              for num in nums for c in classes if (num, c) in by_nc}
    med_d = {(num, c): float(np.median([x["d_logit"] for x in by_nc[(num, c)]]))
             for num in nums for c in classes if (num, c) in by_nc}

    # Infilteration: esclusa se max sui modelli della mediana-DR < 0,10
    excluded = []
    for c in classes:
        mx = max(med_dr[(num, c)] for num in nums if (num, c) in med_dr)
        if mx < DR_HOLE:
            excluded.append((c, mx))
    excl_names = {c for c, _ in excluded}
    treatable = [c for c in classes if c not in excl_names]

    # soglia d dichiarata sui dati del PA (trattabili)
    med_d_treat = [med_d[(num, c)] for num in nums for c in treatable if (num, c) in med_d]
    d_thr = a.d_threshold if a.d_threshold is not None else declare_d_threshold(med_d_treat)

    # buchi / coperte per modello (fra le trattabili)
    for num in nums:
        buchi = [c for c in treatable if med_dr[(num, c)] < DR_HOLE and med_d[(num, c)] < d_thr]
        cover = [c for c in treatable if med_dr[(num, c)] >= DR_COVER]
        per_model[num].update(buchi=buchi, coperte=len(cover))

    # ---- output tabella 25x (ordinata per overall-DR primaria) ----
    order = sorted(nums, key=lambda n: -per_model[n]["o_p"].mean())
    print("\n" + "=" * 100)
    print(f"LOAO PA — 25 finalisti × {len(seeds)} semi · soglia d DICHIARATA (dati PA) = {d_thr:.3f} "
          f"(DAE era 0,85; scala logit diversa)")
    print(f"esclusa dal criterio: {', '.join(f'{disp(c)} (max mediana-DR {m:.3f})' for c,m in excluded)} "
          f"→ {len(treatable)} classi trattabili")
    print("=" * 100)
    hdr = (f"{'trial':>6} {'overall-DR ±σ':>18} {'Z-DR ±σ':>18} {'buchi':>6} {'cop/12':>6}  ||  "
           f"{'sec overall ±σ':>18} {'sec Z-DR ±σ':>18}")
    print(hdr); print("-" * len(hdr))
    for num in order:
        m = per_model[num]
        b = f"{len(m['buchi'])}" + (f" ({','.join(disp(x)[:6] for x in m['buchi'])})" if m['buchi'] else "")
        print(f"#{num:>5} {m['o_p'].mean():>8.4f} ±{m['o_p'].std():<7.4f} "
              f"{m['z_p'].mean():>8.4f} ±{m['z_p'].std():<7.4f} {b:>6} {m['coperte']:>4}/12  ||  "
              f"{m['o_s'].mean():>8.4f} ±{m['o_s'].std():<7.4f} {m['z_s'].mean():>8.4f} ±{m['z_s'].std():<7.4f}")

    # ---- CSV tabella + record per-classe ----
    with open(out / "loao_pa_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num", "n_seeds", "overall_dr_p_mean", "overall_dr_p_std", "z_dr_p_mean", "z_dr_p_std",
                    "buchi", "buchi_classi", "coperte12",
                    "overall_dr_s_mean", "overall_dr_s_std", "z_dr_s_mean", "z_dr_s_std"])
        for num in order:
            m = per_model[num]
            w.writerow([num, m["n_seeds"], m["o_p"].mean(), m["o_p"].std(), m["z_p"].mean(), m["z_p"].std(),
                        len(m["buchi"]), "|".join(m["buchi"]), m["coperte"],
                        m["o_s"].mean(), m["o_s"].std(), m["z_s"].mean(), m["z_s"].std()])
    with open(out / "loao_pa_perclass.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["classe", "num", "median_dr_primary", "median_d_logit", "median_dr_secondary",
                    "trattabile"])
        for c in classes:
            for num in nums:
                if (num, c) in med_dr:
                    ms = float(np.median([x["dr_secondary"] for x in by_nc[(num, c)]]))
                    w.writerow([c, num, med_dr[(num, c)], med_d[(num, c)], ms, c in treatable])

    # ---- istogramma d-logit ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        allcells = np.array([r["d_logit"] for r in rows])
        medvals = np.array(med_d_treat)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].hist(allcells, bins=60, color="tab:blue", alpha=0.8)
        ax[0].axvline(d_thr, color="tab:red", ls="--", label=f"soglia dichiarata {d_thr:.2f}")
        ax[0].axvline(0.85, color="gray", ls=":", label="0,85 (DAE, riferimento)")
        ax[0].set_title(f"d-logit — tutte le celle (modello×seme×classe, n={len(allcells)})")
        ax[0].set_xlabel("Cohen's d (scala logit)"); ax[0].legend(fontsize=8)
        ax[1].hist(medvals, bins=40, color="tab:green", alpha=0.8)
        ax[1].axvline(d_thr, color="tab:red", ls="--", label=f"soglia {d_thr:.2f}")
        ax[1].set_title(f"d-logit — mediana per (modello, classe trattabile), n={len(medvals)}")
        ax[1].set_xlabel("mediana Cohen's d (logit)"); ax[1].legend(fontsize=8)
        fig.suptitle("Distribuzione empirica del Cohen's d (scala logit) del PA — LOAO 25×19×13")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out / "d_logit_histogram.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"\nistogramma: {(out/'d_logit_histogram.png').relative_to(REPO)}")
    except Exception as e:
        print(f"(istogramma saltato: {e})")

    # ---- riepilogo criterio ----
    tot_buchi = sum(len(per_model[n]["buchi"]) for n in nums)
    print(f"\nCRITERIO GRADUATO (soglia d={d_thr:.3f}): buchi totali sui 25 modelli = {tot_buchi}")
    print(f"artefatti: loao_pa_rows.csv · loao_pa_summary.csv · loao_pa_perclass.csv · d_logit_histogram.png")


if __name__ == "__main__":
    main()
