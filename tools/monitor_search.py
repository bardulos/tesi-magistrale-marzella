#!/usr/bin/env python
# TAG: DIAGNOSTICA-CLUSTER | 2026-07-13 | monitor read-only search 400 | vedi docs/refactoring_censimento.md
"""tools/monitor_search.py — monitor READ-ONLY della search Optuna in corso.

Gira su workstation, legge in SOLA LETTURA (niente scritture, niente lancio, non tocca il driver
ne' le VPS) e stampa la leaderboard dei trial COMPLETE (6 semi, eleggibili a vincitore).

Sorgenti (tutte locali su workstation):
  - conteggi stato + tempi (per l'ETA) dallo studio Optuna (sqlite, mode=ro);
  - obj/ap_mean/ap_std/config dai result.json di Ray (ray_results, JSONL) sincronizzati;
  - n_par calcolato da (exp,mid,btl);
  - bordo: estremi FISSI dello spazio premuti dal trial (↓ basso, ↑ alto), vuoto se nessuno.

Riepilogo: media e varianza COMBINATE (pooled, dispersione di seme), ed ETA al target.
Colorazione della ROSA (per media), arcobaleno annidato per target e confidenza crescenti:
  - TOP-3:  ROSSO=p90, ARANCIONE=p95, GIALLO=p99;
  - TOP-8:  VERDE=p90, BLU=p95, VIOLA=p99.
(La rosa e' definita per media; la tabella resta ordinata per obj — vedi metodologia in
 metodologia della selezione della rosa.)

Uso:  python tools/monitor_search.py [--watch N] [--color auto|always|never]
"""
from __future__ import annotations
import os

import argparse
import datetime as dt
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.dae.constants import (BTL_HIGH, BTL_LOW, EXP_HIGH, EXP_LOW, LR_HIGH,  # noqa: E402
                               LR_LOW, MAX_T, MID_HIGH, MID_LOW, SEEDS, SIGMA_HIGH, SIGMA_LOW)
from lib.dae.evaluate import compute_n_params  # noqa: E402
from lib.search.perseed import build_matrix  # noqa: E402
from lib.search.shortlist import (pcs_curve, pooled_sigma, seeds_for_target_K,  # noqa: E402
                                  smallest_K)

COLS = [
    ("#", 3, ">"), ("trial", 6, ">"), ("obj", 9, ">"), ("mean", 8, ">"), ("std", 7, ">"),
    ("n_par", 7, ">"), ("exp[40,224]", 11, "^"), ("mid[20,80]", 10, "^"),
    ("btl[6,12]", 9, "^"), ("σ[.01,.20]", 10, "^"), ("lr·e-3[.1,10]", 13, "^"), ("bordo", 5, "<"),
]
EDGE = 0.06          # frazione di range entro cui un valore "preme" sull'estremo fisso
PCS_DRAWS = 50_000   # estrazioni Monte Carlo per la rosa (errore MC ~ ±0,002, K* robusto)
PCS_SEED = 99        # seme RNG fisso: K_rosso/K_giallo riproducibili tra refresh
SHRINK_KP = 20       # K' bersaglio: quanti semi per comprimere la rosa top-3@p90 a questo K'
RED, ORANGE, YELLOW = "\033[91m", "\033[38;5;208m", "\033[93m"   # top-3: p90/p95/p99
GREEN, BLUE, VIOLET = "\033[92m", "\033[94m", "\033[38;5;135m"   # top-8: p90/p95/p99
RESET = "\033[0m"


def fmt_row(cells) -> str:
    return "  ".join(f"{str(c):{a}{w}}" for c, (_, w, a) in zip(cells, COLS))


def discover(repo: str, config: str | None):
    import yaml
    cfg = yaml.safe_load((Path(config) if config else Path(repo) / "configs/dae/search.yaml").read_text())
    out_dir = Path(cfg["out_dir"])
    if not out_dir.is_absolute():
        out_dir = Path(repo) / out_dir
    return cfg.get("study_name", "dae_search"), out_dir, int(cfg.get("n_trials", 400))


def cfg_key(p) -> str:
    return f"{int(p['exp'])}_{int(p['mid'])}_{int(p['btl'])}_{float(p['sigma']):.5f}_{float(p['lr']):.6f}"


def borders(p) -> str:
    """Estremi FISSI dello spazio premuti dal trial, ordinati dal piu' estremo.
    mid solo bordi fissi 20/80 (no falso bordo dal cap geometrico exp-1); btl solo bordo
    alto (12): btl=6 e' il fondo architetturale, non un 'estendi sotto'."""
    specs = [
        ("exp", float(p["exp"]), EXP_LOW, EXP_HIGH, False, "both"),
        ("mid", float(p["mid"]), MID_LOW, MID_HIGH, False, "both"),
        ("btl", float(p["btl"]), BTL_LOW, BTL_HIGH, False, "hi"),
        ("σ", float(p["sigma"]), SIGMA_LOW, SIGMA_HIGH, False, "both"),
        ("lr", float(p["lr"]), LR_LOW, LR_HIGH, True, "both"),
    ]
    out = []
    for name, v, lo, hi, is_log, edges in specs:
        t = ((math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))) if is_log else (v - lo) / (hi - lo)
        if t <= EDGE and edges == "both":
            out.append((t, f"{name}↓"))
        elif t >= 1.0 - EDGE:
            out.append((1.0 - t, f"{name}↑"))
    out.sort(key=lambda x: x[0])
    return " ".join(lbl for _, lbl in out)


def study_state(study_name, db: Path, retries=4):
    """(counts, numbers, start_dt, completes_dt) dallo studio (read-only, retry sul lock).
    completes_dt = istanti di conclusione dei trial terminali (COMPLETE+PRUNED) per l'ETA."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    abspath = Path(db).resolve()
    for url in (f"sqlite:///file:{abspath}?mode=ro&uri=true", f"sqlite:///{abspath}"):
        for i in range(retries):
            try:
                s = optuna.load_study(study_name=study_name, storage=url)
                counts = Counter(t.state.name for t in s.trials)
                numbers = {cfg_key(t.params): t.number for t in s.trials if t.params}
                starts = [t.datetime_start for t in s.trials if t.datetime_start]
                completes = [t.datetime_complete for t in s.trials if t.datetime_complete]
                return counts, numbers, (min(starts) if starts else None), completes
            except Exception as e:  # noqa: BLE001
                if "locked" in str(e).lower():
                    time.sleep(0.4 * (i + 1)); continue
                break
    return Counter(), {}, None, []


def eta_line(start_dt, completes, n_target) -> str:
    if start_dt is None or not completes:
        return "ETA: dati insufficienti (nessun trial terminato)"
    now = dt.datetime.now()
    nterm = len(completes)
    elapsed_h = max(1e-9, (now - start_dt).total_seconds() / 3600)
    rate3 = sum(1 for c in completes if c >= now - dt.timedelta(hours=3)) / 3.0
    rate = rate3 if rate3 > 0 else nterm / elapsed_h
    rem = max(0, n_target - nterm)
    if rate <= 0:
        return f"ETA: {nterm}/{n_target} terminati · ritmo non stimabile"
    eta_h = rem / rate
    fine = now + dt.timedelta(hours=eta_h)
    return (f"ETA: {nterm}/{n_target} terminati · ritmo {rate:.1f}/h · residui {rem} "
            f"→ ~{eta_h:.1f}h → fine ~{fine:%a %d/%m %H:%M}")


def read_completed(out_dir: Path):
    """Trial COMPLETE (seeds_completed==MAX_T) da ray_results: obj/ap_mean/ap_std/config."""
    base = out_dir / "ray_results/dae_search"
    rows = []
    for d in base.glob("_run_trainable_*"):
        rj, pj = d / "result.json", d / "params.json"
        if not (rj.exists() and pj.exists()):
            continue
        try:
            last = json.loads([l for l in rj.read_text().splitlines() if l.strip()][-1])
            if int(last.get("seeds_completed", 0)) != MAX_T:
                continue
            rows.append({"obj": float(last["objective"]),
                         "ap_mean": float(last.get("ap_mean", float("nan"))),
                         "ap_std": float(last.get("ap_std", float("nan"))),
                         "cfg": json.loads(pj.read_text())})
        except Exception:  # noqa: BLE001
            continue
    return rows


def shortlist_bounds(rows):
    """(k3, k8, c3, c8, sigma, se, nu): k3/k8 = terne (K@0,90, K@0,95, K@0,99) per il contenimento
    delle vere TOP-3 e TOP-8 (per media); c3/c8 = curve PCS(K) complete. k3/c3 None se <4 COMPLETE,
    k8/c8 None se <9."""
    stds = [r["ap_std"] for r in rows]
    sigma, se, nu = pooled_sigma(stds, MAX_T)
    n = len(rows)
    k3 = k8 = c3 = c8 = None
    if n >= 4:
        means_sorted = np.array(sorted((r["ap_mean"] for r in rows), reverse=True))
        rng = np.random.default_rng(PCS_SEED)
        c3 = pcs_curve(means_sorted, se, 3, PCS_DRAWS, rng)
        k3 = (smallest_K(c3, 0.90), smallest_K(c3, 0.95), smallest_K(c3, 0.99))
        if n >= 9:
            c8 = pcs_curve(means_sorted, se, 8, PCS_DRAWS, rng)
            k8 = (smallest_K(c8, 0.90), smallest_K(c8, 0.95), smallest_K(c8, 0.99))
    return k3, k8, c3, c8, sigma, se, nu


def render(out_dir: Path, counts, numbers, start_dt, completes, n_target, color_on):
    rows = read_completed(out_dir)
    rows.sort(key=lambda r: -r["obj"])

    print("=" * 78)
    print(f"SEARCH MONITOR (read-only)  —  {counts.get('COMPLETE',0)} COMPLETE · "
          f"{counts.get('PRUNED',0)} PRUNED · {counts.get('RUNNING',0)} RUNNING · "
          f"{counts.get('FAIL',0)} FAIL · {sum(counts.values())} tot")
    print(eta_line(start_dt, completes, n_target))
    print("obj = mean(AP)−std(AP)   (solo COMPLETE = 6 semi, eleggibili a vincitore)")
    print("=" * 78)
    if not rows:
        print("  (nessun COMPLETE sincronizzato ancora)")
        return

    # rose top-3 (rosso/aranc/giallo) e top-8 (verde/blu/viola) a p90/p95/p99, per rango-MEDIA
    k3, k8, c3, c8, sigma, se, nu = shortlist_bounds(rows)
    mean_rank = {cfg_key(r["cfg"]): rk
                 for rk, r in enumerate(sorted(rows, key=lambda r: -r["ap_mean"]), 1)}

    # bande (K, colore) per (target, confidenza); colore fisso per banda
    bands = [(k3[0], RED), (k3[1], ORANGE), (k3[2], YELLOW)] if k3 else []
    if k8 is not None:
        bands += [(k8[0], GREEN), (k8[1], BLUE), (k8[2], VIOLET)]

    def paint(s, mr):
        # colora col vincolo PIÙ STRETTO (K minimo) che copre il trial: robusto ai crossing
        # (es. top-8@p90 può essere più stretto di top-3@p99), così nessuna banda sparisce
        if not color_on or not bands:
            return s
        cand = [(K, col) for K, col in bands if mr <= K]
        if not cand:
            return s
        return f"{min(cand)[1]}{s}{RESET}"

    print()
    print(fmt_row([name for name, _, _ in COLS]))
    for i, r in enumerate(rows[:100], 1):
        p = r["cfg"]
        k = cfg_key(p)
        npar = compute_n_params(int(p["exp"]), int(p["mid"]), int(p["btl"]))
        line = fmt_row([
            i, f"#{numbers.get(k,'?')}", f"{r['obj']:.5f}", f"{r['ap_mean']:.4f}", f"{r['ap_std']:.4f}",
            npar, int(p["exp"]), int(p["mid"]), int(p["btl"]),
            f"{float(p['sigma']):.3f}", f"{float(p['lr']) * 1000:.2f}", borders(p),
        ])
        print(paint(line, mean_rank[k]))

    # bordi premuti da TUTTI i rossi (banda top-3 @p90 = top-k3[0] per media), non solo i top-10
    n_reds = k3[0] if k3 else 0
    border_tally = Counter()
    for r in rows:
        if mean_rank[cfg_key(r["cfg"])] <= n_reds:
            for lbl in borders(r["cfg"]).split():
                border_tally[lbl] += 1

    objs = [r["obj"] for r in rows]
    grand_mean = float(np.mean([r["ap_mean"] for r in rows]))
    print(f"\nSOMMARIO: max={max(objs):.5f}  spread-top={max(objs)-min(objs):.4f}  "
          f"btl-top5={sorted({int(r['cfg']['btl']) for r in rows[:5]})}")
    print(f"COMBINATA (pooled su {len(rows)} COMPLETE): media={grand_mean:.4f}  "
          f"varianza={sigma**2:.6f}  σ={sigma:.4f}  SE={se:.4f}  (ν={nu})")
    # effetto-seme reale: media AP per seme FISICO (ricostruzione per-seme, lib condivisa)
    try:
        Aps, seeds_sorted, kept, _ = build_matrix(
            sorted((out_dir / "ray_results/dae_search").glob("_run_trainable_*")), SEEDS)
        if len(Aps):
            sm = Aps.mean(axis=0)
            eff = "  ".join(f"{s}:{v:.4f}" for s, v in zip(seeds_sorted, sm))
            print(f"EFFETTO-SEME (media AP per seme fisico, {len(kept)} trial): {eff}  "
                  f"spread={float(sm.max()-sm.min()):.4f}")
    except Exception:  # noqa: BLE001
        pass
    def cz(text, code):
        return f"{code}{text}{RESET}" if color_on else text
    if k3 is not None:
        print(f"ROSA top-3 (per media, colore=vincolo più stretto che copre): "
              f"{cz('rosso',RED)}=K{k3[0]}(p90) · {cz('arancione',ORANGE)}=K{k3[1]}(p95) · "
              f"{cz('giallo',YELLOW)}=K{k3[2]}(p99)")
    if k8 is not None:
        print(f"ROSA top-8 (per media): {cz('verde',GREEN)}=K{k8[0]}(p90) · "
              f"{cz('blu',BLUE)}=K{k8[1]}(p95) · {cz('viola',VIOLET)}=K{k8[2]}(p99)")
    if k3 is None:
        print("ROSA: troppi pochi COMPLETE (<4) per dimensionarla.")
    # quanti semi per comprimere la rosa top-3@p90 dal K attuale a K'=SHRINK_KP (power analysis
    # forward: medie osservate come vere, SE(n')=σ/√n', ricalcolo PCS esatto). σ = dispersione di seme
    if k3 is not None and len(rows) > SHRINK_KP:
        if k3[0] <= SHRINK_KP:
            print(f"COMPRIMI top-3@p90 → K'={SHRINK_KP}: già a K{k3[0]} ≤ {SHRINK_KP}, "
                  f"nessun seme in più")
        else:
            m_sorted = np.array(sorted((r["ap_mean"] for r in rows), reverse=True))
            n_kp = seeds_for_target_K(m_sorted, sigma, 3, 0.90, SHRINK_KP, PCS_DRAWS, PCS_SEED)
            txt = f"servono n'={n_kp} semi" if n_kp is not None else "irraggiungibile (>n_max=4000)"
            print(f"COMPRIMI top-3@p90 da K{k3[0]} → K'={SHRINK_KP}: {txt} "
                  f"(ricalcolo PCS esatto; ora {MAX_T} semi)")
    # tabella PCS(K): p raggiunto al crescere di K, per top-3 e top-8 (per media)
    if c3 is not None:
        kmax = min(len(rows), (k8[2] if k8 is not None else k3[2]))
        step = max(1, (kmax - 3) // 11)
        grid = list(range(3, kmax + 1, step))
        if grid and grid[-1] != kmax:
            grid.append(kmax)

        def prow(label, cur):
            cells = "".join(
                f"{(f'{cur[K]:.2f}' if (cur is not None and cur.get(K) is not None) else '·'):>6}"
                for K in grid)
            return f"  {label:>6}{cells}"
        print("\nPCS(K) per media — p raggiunto al crescere di K:")
        print(f"  {'K':>6}" + "".join(f"{K:>6}" for K in grid))
        print(prow("top-3", c3))
        print(prow("top-8", c8))
    if border_tally:
        tally = " ".join(f"{lbl}×{c}" for lbl, c in border_tally.most_common())
        print(f"bordi-rossi (su {n_reds} rossi): {tally}  (se un bordo domina → valutare estensione spazio)")
    else:
        print(f"bordi-rossi (su {n_reds} rossi): nessuno (i rossi non premono sugli estremi)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.expanduser("~/tesi/repov6"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--watch", type=int, default=0)
    ap.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    args = ap.parse_args()
    color_on = args.color == "always" or (args.color == "auto" and sys.stdout.isatty())

    study_name, out_dir, n_target = discover(args.repo, args.config)
    if not (out_dir / "optuna_study.db").exists():
        print(f"storage non trovato in {out_dir}"); sys.exit(1)

    while True:
        counts, numbers, start_dt, completes = study_state(study_name, out_dir / "optuna_study.db")
        if args.watch:
            print("\033[2J\033[H", end="")
        render(out_dir, counts, numbers, start_dt, completes, n_target, color_on)
        if not args.watch:
            break
        time.sleep(max(2, int(args.watch)))


if __name__ == "__main__":
    main()
