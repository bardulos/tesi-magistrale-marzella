#!/usr/bin/env python
# TAG: ONE-SHOT | 2026-07-13 | monitor compressione DAE (testato; citato nel report) | vedi docs/refactoring_censimento.md
"""tools/compress_tables.py — tabelle di ranking delle trial della compressione (FASE 2).

Per ogni finalista aggrega i semi disponibili (6 base ricostruiti dai result.json Optuna +
i semi extra già calcolati dalla compressione) e stampa a schermo:
  1. tabella ordinata per AP medio (nuovo) ± std, con Δap_medio e Δstd vs i 6 semi Optuna,
     AFFIANCATA alla tabella Optuna ordinata per AP (6 semi);
  2. tabella ordinata per MCC@FPR1% medio (nuovo) ± std (primo stadio).

Funziona anche a compressione in corso: la colonna n (semi aggregati; 17 = completo) segnala la
completezza di ogni riga. Stampa tutto a schermo, nessun file, nessun flag.

Uso:  python tools/compress_tables.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from refit_compress import merge_ap_and_mcc, _base_aps_for  # noqa: E402
from monitor_search import cfg_key  # noqa: E402
from lib.dae.evaluate import compute_n_params  # noqa: E402

# Path fissi della compressione FASE 2 (run principale + top-up seme 2035)
SHORTLIST = REPO / "runs/dae/compress_fase2/shortlist.csv"
EXTRA_CSVS = [REPO / "runs/dae/compress_fase2/compress_extra.csv",
              REPO / "runs/dae/compress_fase2_s2035/compress_extra.csv"]
BASE_CSV = REPO / "runs/dae/compress_base/compress_extra.csv"   # 6 semi base ri-addestrati (MCC)
DB600 = REPO / "runs/dae/search_600/optuna_study.db"
DB400 = REPO / "runs/dae/search_400/optuna_study.db"


# ===================================================================== logica pura ====

def build_rows(finalists: list[dict], base_aps: dict, extra_rows: list[dict],
               base_mcc: dict | None = None) -> list[dict]:
    """Una riga per finalista. AP su 17 semi = base_aps (6, dai result.json) + extra (11).
    MCC su 17 semi = base_mcc (6, dal ri-addestramento base) + extra (11). Se base_mcc è None,
    l'MCC resta sui soli extra. Più i valori Optuna (6 semi) e i delta nuovo−Optuna."""
    from collections import defaultdict
    base_mcc = base_mcc or {}
    extra_by: dict[str, list] = defaultdict(list)
    for r in extra_rows:
        extra_by[r["trial_id"]].append(r)

    rows = []
    for fin in finalists:
        tid = fin["trial_id"]
        ap_all  = list(base_aps.get(tid, [])) + [r["ap"] for r in extra_by.get(tid, [])]
        mcc_all = list(base_mcc.get(tid, [])) + [r["mcc"] for r in extra_by.get(tid, [])]
        ap_arr, mcc_arr = np.asarray(ap_all, float), np.asarray(mcc_all, float)
        new_ap_mean = float(np.mean(ap_arr)) if len(ap_arr) else float("nan")
        new_ap_std  = float(np.std(ap_arr))  if len(ap_arr) else float("nan")
        opt_ap_mean, opt_ap_std = fin["opt_ap_mean"], fin["opt_ap_std"]
        rows.append({
            "trial_id": tid, "trial_num": fin["trial_num"], "source": fin.get("source", ""),
            "opt_ap_mean": opt_ap_mean, "opt_ap_std": opt_ap_std,
            "new_ap_mean": new_ap_mean, "new_ap_std": new_ap_std,
            "d_ap_mean": (new_ap_mean - opt_ap_mean) if math.isfinite(new_ap_mean) else float("nan"),
            "d_ap_std":  (new_ap_std - opt_ap_std)   if math.isfinite(new_ap_std)  else float("nan"),
            "mcc_mean": float(np.mean(mcc_arr)) if len(mcc_arr) else float("nan"),
            "mcc_std":  float(np.std(mcc_arr))  if len(mcc_arr) else float("nan"),
            "n_ap": len(ap_all), "n_mcc": len(mcc_all),
        })
    return rows


# ===================================================================== stampa ====

def _trial_num(name: str) -> str:
    """Numero Optuna dal nome del ray_dir '_run_trainable_<hex>_<num>_btl=...'."""
    tail = name.split("_run_trainable_", 1)[-1]
    parts = tail.split("_")
    return parts[1] if len(parts) > 1 else "?"


def _read_extra(paths) -> list[dict]:
    rows = []
    for p in paths:
        if not Path(p).exists():
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append({"trial_id": r["trial_id"], "seed": int(r["seed"]),
                             "ap": float(r["ap"]), "mcc": float(r["mcc"])})
    return rows


def _fmt(x, nd=5):
    return f"{x:.{nd}f}" if isinstance(x, float) and math.isfinite(x) else "  —  "


# Colore solo su terminale (niente codici ANSI se l'output è rediretto su file).
# Evidenziano l'intersezione dei ranking AP nuovo e MCC, in ENTRAMBE le tabelle:
_COLOR  = sys.stdout.isatty()
ROSSO   = "\033[1;38;5;196m"  # top-10 SIA per AP SIA per MCC (precedenza)
ARANCIO = "\033[38;5;208m"    # top-15 SIA per AP SIA per MCC
GIALLO  = "\033[38;5;226m"    # top-20 SIA per AP SIA per MCC
RESET   = "\033[0m"


def _eta(done: int, total: int):
    """Stima ritmo e fine della raccolta: avvio = primo timestamp nei driver.log dei 3 run.
    Ritorna (riga di testo) o '' se non calcolabile."""
    import datetime as dt
    import re
    logs = [REPO / "runs/dae/compress_fase2/driver.log",
            REPO / "runs/dae/compress_fase2_s2035/driver.log",
            REPO / "runs/dae/compress_base/driver.log"]
    starts = []
    for lg in logs:
        if lg.exists():
            m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", lg.read_text()[:8000])
            if m:
                starts.append(dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
    if not starts or done == 0:
        return ""
    elapsed_h = (dt.datetime.now() - min(starts)).total_seconds() / 3600
    if elapsed_h <= 0:
        return ""
    rate = done / elapsed_h
    rem = max(0, total - done)
    if rem == 0:
        return f"raccolta COMPLETA ({done}/{total} semi · {rate:.0f}/h · {elapsed_h:.1f} h)"
    eta_h = rem / rate
    fine = dt.datetime.now() + dt.timedelta(hours=eta_h)
    return (f"raccolta {done}/{total} semi · {rate:.0f}/h · residui {rem} · "
            f"ETA ~{eta_h:.1f} h → fine ~{fine:%a %d/%m %H:%M}")


def _optuna_numbers():
    """({cfg_key: n}, {cfg_key: n}) da search_600 e search_400 (new→600, impo→400)."""
    import optuna

    def load(db):
        try:
            st = optuna.load_study(study_name="dae_search", storage=f"sqlite:///{db}")
            return {cfg_key(t.params): t.number for t in st.trials if t.params}
        except Exception:  # noqa: BLE001
            return {}
    return load(DB600), load(DB400)


def main():
    n600, n400 = _optuna_numbers()
    # Finalisti + valori Optuna (6 semi) dalla shortlist; base AP dai result.json;
    # trial = NUMERO OPTUNA (come il monitor), non l'indice interno del ray_dir.
    finalists, base_aps, meta = [], {}, {}
    with open(SHORTLIST) as f:
        for r in csv.DictReader(f):
            tid = r["trial_id"]
            src = r.get("source", "")
            # ray_dir nella shortlist è relativo alla radice del repo → risolvi contro REPO
            rdir = Path(r["ray_dir"])
            rdir = rdir if rdir.is_absolute() else REPO / rdir
            cfg = json.loads((rdir / "params.json").read_text())
            num = (n600 if src == "new" else n400).get(cfg_key(cfg))
            exp, mid, btl = int(cfg["exp"]), int(cfg["mid"]), int(cfg["btl"])
            meta[tid] = {"btl": btl, "n_par": compute_n_params(exp, mid, btl)}
            finalists.append({"trial_id": tid,
                              "trial_num": str(num) if num is not None else _trial_num(tid),
                              "opt_ap_mean": float(r["ap_mean"]), "opt_ap_std": float(r["ap_std"])})
            base_aps[tid] = _base_aps_for(rdir)

    extra_rows = _read_extra(EXTRA_CSVS)
    # MCC dei 6 semi base dal ri-addestramento (per l'MCC su 17 semi); {} se non ancora pronto
    base_mcc = {}
    if BASE_CSV.exists():
        from collections import defaultdict
        bm = defaultdict(list)
        for r in csv.DictReader(open(BASE_CSV)):
            bm[r["trial_id"]].append(float(r["mcc"]))
        base_mcc = dict(bm)
    all_rows = build_rows(finalists, base_aps, extra_rows, base_mcc)
    for r in all_rows:
        r.update(meta[r["trial_id"]])

    # Solo le trial COMPLETE: tutti e 17 i semi (6 base + 11 extra).
    FULL = 17
    rows = [r for r in all_rows if r["n_ap"] == FULL]

    by_ap  = sorted(rows, key=lambda r: -(r["new_ap_mean"] if math.isfinite(r["new_ap_mean"]) else -9))
    by_mcc = sorted(rows, key=lambda r: -(r["mcc_mean"] if math.isfinite(r["mcc_mean"]) else -9))
    by_opt = sorted(rows, key=lambda r: -r["opt_ap_mean"])

    # Rank (1..N) sul set COMPLETO, per lo spostamento di posizione
    ap_rank  = {r["trial_id"]: i for i, r in enumerate(by_ap, 1)}
    opt_rank = {r["trial_id"]: i for i, r in enumerate(by_opt, 1)}

    def arrow(delta: int) -> str:
        """↑N salito di N posti · ↓N sceso · = invariato."""
        return f"↑{delta}" if delta > 0 else (f"↓{-delta}" if delta < 0 else "=")

    TOP = 25
    both_top10 = {r["trial_id"] for r in by_ap[:10]} & {r["trial_id"] for r in by_mcc[:10]}  # rosso
    both_top15 = {r["trial_id"] for r in by_ap[:15]} & {r["trial_id"] for r in by_mcc[:15]}  # arancio
    both_top20 = {r["trial_id"] for r in by_ap[:20]} & {r["trial_id"] for r in by_mcc[:20]}  # giallo

    def colorize(line: str, tid: str) -> str:
        """Intersezione dei ranking AP∩MCC: rosso top-10 ⊃ arancio top-15 ⊃ giallo top-20."""
        if not _COLOR:
            return line
        col = (ROSSO if tid in both_top10 else ARANCIO if tid in both_top15
               else GIALLO if tid in both_top20 else "")
        return f"{col}{line}{RESET}" if col else line

    n_fin = len(finalists)
    n_extra, tot_extra = len(extra_rows), n_fin * 11      # 11 semi extra per finalista
    n_base,  tot_base = sum(len(v) for v in base_mcc.values()), n_fin * 6   # 6 semi base
    total_seeds = tot_extra + tot_base                    # 71 × 17 = 1207 (extra + base)
    done_seeds = n_extra + n_base
    print(f"\nTrial COMPLETE (17 semi): {len(rows)}/{len(all_rows)}  ·  top {TOP}  "
          f"(trial = numero Optuna; std/Δ a 5 decimali)")
    eta = _eta(done_seeds, total_seeds)
    if eta:
        print(f"  raccolta: extra {n_extra}/{tot_extra} + base {n_base}/{tot_base} "
              f"= {done_seeds}/{total_seeds}")
        print(eta)

    # ---- Tabella 1 (nuovo AP) — Δpos = spostamento di rango vs l'ordine Optuna a 6 semi ----
    print("\n" + "=" * 90)
    print("ORDINATE PER AP (nuovo; Δpos vs Optuna; rosso=top-10, arancio=top-15, giallo=top-20 (AP∩MCC))")
    print("=" * 90)
    print(f"{'#':>3} {'trial':>5} {'btl':>3} {'n_par':>7} {'ap_mean':>8} {'±std':>9} "
          f"{'Δap':>9} {'Δstd':>9} {'mcc':>8} {'±std':>9} {'Δpos':>5}")
    print("-" * 90)
    for i in range(min(TOP, len(by_ap))):
        l = by_ap[i]
        mov = opt_rank[l["trial_id"]] - (i + 1)        # >0 = salito rispetto a Optuna
        line = (f"{i+1:>3} {l['trial_num']:>5} {l['btl']:>3} {l['n_par']:>7} "
                f"{_fmt(l['new_ap_mean']):>8} {_fmt(l['new_ap_std']):>9} "
                f"{_fmt(l['d_ap_mean']):>9} {_fmt(l['d_ap_std']):>9} "
                f"{_fmt(l['mcc_mean']):>8} {_fmt(l['mcc_std']):>9} {arrow(mov):>5}")
        print(colorize(line, l["trial_id"]))   # rosso=top-10, arancio=top-15, giallo=top-20 (∩)

    # ---- Tabella 2 (nuovo MCC) ----  Δpos = spostamento rispetto al ranking AP (nuovo)
    mcc_src = "6 base ri-addestrati + 11 extra" if base_mcc else "soli semi extra"
    print("\n" + "=" * 88)
    print(f"ORDINATE PER MCC@FPR1% (primo stadio; {mcc_src}; Δpos vs AP; rosso=top-10, arancio=top-15, giallo=top-20 (∩))")
    print("=" * 88)
    print(f"{'#':>3} {'trial':>5} {'btl':>3} {'n_par':>7} {'mcc_mean':>9} {'±std':>9} "
          f"{'n_mcc':>5} {'ap_mean':>8} {'±std':>9} {'Δpos':>5}")
    print("-" * 88)
    for i in range(min(TOP, len(by_mcc))):
        r = by_mcc[i]
        mov = ap_rank[r["trial_id"]] - (i + 1)         # >0 = MCC lo piazza più in alto dell'AP
        line = (f"{i+1:>3} {r['trial_num']:>5} {r['btl']:>3} {r['n_par']:>7} "
                f"{_fmt(r['mcc_mean']):>9} {_fmt(r['mcc_std']):>9} {r['n_mcc']:>5} "
                f"{_fmt(r['new_ap_mean']):>8} {_fmt(r['new_ap_std']):>9} {arrow(mov):>5}")
        print(colorize(line, r["trial_id"]))    # rosso=top-10, arancio=top-15, giallo=top-20 (∩)


if __name__ == "__main__":
    main()
