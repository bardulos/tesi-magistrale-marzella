#!/usr/bin/env python3
# TAG: ONE-SHOT | 2026-07-13 | orchestratore LOAO PA locale | vedi docs/refactoring_censimento.md
"""tools/pa_loao_fanout.py — orchestratore LOAO PA in LOCALE (workstation, nessuna VPS). Pura inferenza.

Finestra mobile di sottoprocessi (riusa run_window di sup_loao_run): un worker per coppia
(modello, seme) fra i 25 finalisti × 19 semi = 475. Censisce i pesi prima di lanciare (fallisce se il
conteggio non è 475), ripartenza sui soli job mancanti (CSV parziale assente), stagger degli import TF.
Ogni worker scrive il suo CSV parziale e il suo .npz → nessun risultato perso a interruzione.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from pa_loao_smoke import SEEDS19, census  # noqa: E402  (core riusato)
from sup_loao_run import run_window         # noqa: E402  (finestra mobile generica)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", default=str(REPO / "runs/secondo_stadio/pa_compress/hardening_candidates.json"))
    ap.add_argument("--out-dir", default=str(REPO / "runs/secondo_stadio/loao_pa"))
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    out = Path(a.out_dir)
    (out / "rows").mkdir(parents=True, exist_ok=True)

    cands = json.loads(Path(a.candidates).read_text())
    present, expected, incomplete = census(cands)
    print(f"censimento pesi: {present}/{expected}  incompleti: {len(incomplete)}")
    if incomplete:
        for it in incomplete:
            print(f"  #{it['num']}: {it['have']}/{len(SEEDS19)} mancano {it['missing']}")
        raise SystemExit("pesi incompleti: correggere prima del fan-out")

    worker = str(REPO / "tools" / "pa_loao_worker.py")
    cmds, done = [], 0
    for c in cands:
        for s in SEEDS19:
            if (out / "rows" / f"{c['num']}_seed{s}.csv").exists():
                done += 1
                continue
            cmds.append([sys.executable, worker, "--num", str(c["num"]), "--seed", str(s),
                         "--candidates", a.candidates, "--out-dir", a.out_dir])
    print(f"job totali {len(cands) * len(SEEDS19)} · già fatti {done} · da eseguire {len(cmds)} "
          f"· finestra {a.workers}")
    if not cmds:
        print("nulla da eseguire (tutti i parziali presenti)")
        return

    t0 = time.time()
    failed = run_window(cmds, a.workers)
    dt = time.time() - t0
    n_rows = len(list((out / "rows").glob("*.csv")))
    print(f"FAN-OUT COMPLETO in {dt / 60:.1f} min · parziali su disco {n_rows}/{len(cands) * len(SEEDS19)} "
          f"· falliti {failed}")
    if failed or n_rows != len(cands) * len(SEEDS19):
        print("ATTENZIONE: rilanciare per completare i mancanti (ripartenza automatica sui parziali assenti)")


if __name__ == "__main__":
    main()
