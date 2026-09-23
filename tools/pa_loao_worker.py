#!/usr/bin/env python3
# TAG: ONE-SHOT | 2026-07-13 | worker LOAO PA zero-training | vedi docs/refactoring_censimento.md
"""tools/pa_loao_worker.py — worker LOAO PA: UNA coppia (modello, seme). Pura inferenza, ZERO training.

Un solo forward per coppia (i 13 fold di classe ne derivano). Cacha i logit (ben+att) in un .npz
riusabile e scrive un CSV parziale con, per classe: DR **primaria** (a tau=p99 benigni-select, FPR1%,
class-invariant) + DR **secondaria** deploy-realistica (tau standalone = argmax MCC_bal SENZA la classe
esclusa, dagli stessi score) + d in **scala logit** (pre-sigmoide) + n. 1 thread (BLAS+TF) per non
saturare la finestra. Fallisce esplicitamente se un peso non esiste (nessun salto silenzioso).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from lib.utils import setup_blas_env  # noqa: E402

setup_blas_env(1, deterministic=False)  # 1 thread/worker (finestra su 16 core → niente oversubscription)

import numpy as np  # noqa: E402

from lib.dae.constants import FPR_TARGET          # noqa: E402
from lib.dae.evaluate import cohens_d             # noqa: E402
from lib.dae.threshold import compute_threshold   # noqa: E402
from lib.secondo_stadio import constants as C     # noqa: E402
from lib.secondo_stadio.data import load_phi_cache  # noqa: E402
from lib.secondo_stadio.fp_mining import _load_model  # noqa: E402
from lib.secondo_stadio.metrics import best_mcc_bal_standalone  # noqa: E402
from pa_loao_smoke import make_logit_fn, weight_path  # noqa: E402  (core riusato)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--candidates", default=str(REPO / "runs/secondo_stadio/pa_compress/hardening_candidates.json"))
    ap.add_argument("--cache-dir", default=str(REPO / "runs/dae/output_569"))
    ap.add_argument("--labels-val", default=str(REPO / "runs/preprocessing/val_attack_npy/val_Attack.npy"))
    ap.add_argument("--out-dir", default=str(REPO / "runs/secondo_stadio/loao_pa"))
    a = ap.parse_args()
    out = Path(a.out_dir)

    # 1 thread anche per gli op TF (BLAS già pinnato da setup_blas_env)
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    cands = json.loads(Path(a.candidates).read_text())
    cand = next((c for c in cands if str(c["num"]) == str(a.num)), None)
    if cand is None:
        raise SystemExit(f"#{a.num} assente in {a.candidates}")
    config = cand["config"]
    wpath = weight_path(cand["trial_id"], a.seed)
    if not wpath.exists():                                   # fallire esplicito, non saltare
        raise SystemExit(f"peso mancante: {wpath}")

    sh = load_phi_cache(a.cache_dir, a.labels_val, val_frac=C.VAL_FRAC,
                        split_seed=C.SPLIT_SEED, attack_split=False)
    Zs, R, labels = sh["Zs"], sh["R"], sh["val_attack"]
    ben_sel = sh["ben_pos"][sh["idx_select"]]                # split leak-free seed 0 (== DAE/ceiling)
    att_pos = sh["att_pos"]

    model = _load_model(wpath, config, {})                   # config-first
    logit_fn = make_logit_fn(model)
    logit_ben = logit_fn(Zs, R, ben_sel)
    logit_att = logit_fn(Zs, R, att_pos)
    sig_ben = 1.0 / (1.0 + np.exp(-logit_ben))
    sig_att = 1.0 / (1.0 + np.exp(-logit_att))
    labels_att = labels[att_pos]

    tau_p = compute_threshold(sig_ben, FPR_TARGET)           # primaria: p99 benigni-select

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"scores_{a.num}_seed{a.seed}.npz",
             logit_ben=logit_ben.astype(np.float32), logit_att=logit_att.astype(np.float32),
             labels_att=labels_att.astype(str), tau_primary=np.float64(tau_p),
             num=a.num, seed=a.seed)

    rows = []
    for cls in sorted(set(labels_att.tolist())):
        m = labels_att == cls
        dr_p = float((sig_att[m] >= tau_p).mean())
        tau_s = float(best_mcc_bal_standalone(sig_ben, sig_att[labels_att != cls])["tau"])  # senza la classe
        dr_s = float((sig_att[m] >= tau_s).mean())
        rows.append({"num": a.num, "seed": a.seed, "classe": cls,
                     "dr_primary": dr_p, "dr_secondary": dr_s,
                     "d_logit": cohens_d(logit_att[m], logit_ben), "n": int(m.sum()),
                     "tau_primary": float(tau_p), "tau_secondary": tau_s})

    part = out / "rows" / f"{a.num}_seed{a.seed}.csv"
    part.parent.mkdir(parents=True, exist_ok=True)
    with open(part, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"OK #{a.num} seed {a.seed}: {len(rows)} classi, tau_p={tau_p:.5f}")


if __name__ == "__main__":
    main()
