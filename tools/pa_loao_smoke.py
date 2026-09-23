#!/usr/bin/env python3
# TAG: ONE-SHOT | 2026-07-13 | smoke LOAO PA TEMPO 1 | vedi docs/refactoring_censimento.md
"""tools/pa_loao_smoke.py — SMOKE del LOAO PA (pura inferenza, una cella). TEMPO 1 (bloccante).

Principio (cap3, non re-derivare): il training del PA non vede attacchi reali -> il LOAO NON
riaddestra; gli attacchi entrano solo nei canali metrici. La soglia primaria e' class-invariant
(p99 dei benigni-select = FPR 1%, stessa regola di DAE e ceiling). Un forward per (modello, seme)
serve tutti i 13 fold di classe. Pesi BASE (mai hardened). Inferenza pura -> nessun caveat
non-bit-exact.

Questa cella: #589, seme 42, 13 fold di classe. Per classe: DR (a tau=p99 benigni-select) e
separazione d in SCALA LOGIT (pre-sigmoide; la sigmoide satura e comprime gli effect size), con n
campioni della classe. Prima del fan-out censisce i pesi su disco (25 modelli x 19 semi = 475) e
cacha gli score PA (benigni-select e attacchi, logit) per la variante soglia deploy-realistica e le
figure. Poi si ferma (via libera esplicita per il TEMPO 2 = fan-out).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.utils import setup_blas_env  # noqa: E402  (chiamato in main(); import del core side-effect-free)

from lib.dae.constants import FPR_TARGET          # noqa: E402
from lib.dae.evaluate import cohens_d             # noqa: E402
from lib.dae.threshold import compute_threshold   # noqa: E402
from lib.secondo_stadio import constants as C     # noqa: E402
from lib.secondo_stadio.data import (assemble_phi, load_phi_cache,  # noqa: E402
                                     predict_phi)
from lib.secondo_stadio.fp_mining import _load_model  # noqa: E402

SEEDS_BASE = list(C.SEEDS_PA)                      # 8 base
SEEDS_EXTRA = list(C.SEEDS_PA_EXTRA)              # 11 extra (2038-2048)
SEEDS19 = SEEDS_BASE + SEEDS_EXTRA
WF = REPO / "runs/secondo_stadio/weights_final"


def hex_of(trial_id: str) -> str:
    """8-hex del nome-dir Ray (nome dir dei pesi base)."""
    return trial_id.split("_run_trainable_")[1].split("_")[0]


def weight_path(trial_id: str, seed: int) -> Path:
    """Pesi BASE del (modello, seme): base (8 semi) in weights_final/<hex>/, extra (11 semi della
    compressione) in weights_final/pa_compress/<trial_id>/."""
    if seed in SEEDS_BASE:
        return WF / hex_of(trial_id) / f"seed_{seed}" / "best.weights.h5"
    return WF / "pa_compress" / trial_id / f"seed_{seed}" / "best.weights.h5"


def census(candidates: list[dict]) -> tuple[int, int, list[dict]]:
    """Censimento dei pesi su disco: (presenti, attesi, incompleti[num, have, missing])."""
    present, incomplete = 0, []
    for c in candidates:
        got = [s for s in SEEDS19 if weight_path(c["trial_id"], s).exists()]
        present += len(got)
        if len(got) != len(SEEDS19):
            incomplete.append({"num": c["num"], "have": len(got),
                               "missing": [s for s in SEEDS19 if s not in got]})
    return present, len(candidates) * len(SEEDS19), incomplete


def make_logit_fn(model):
    """Estrattore del LOGIT (pre-sigmoide): penultimo output @ pesi del layer di uscita `out`.
    La sigmoide del modello satura e comprime gli effect size -> la d si misura sul logit."""
    import tensorflow as tf
    penult = tf.keras.Model(model.inputs, model.layers[-2].output)
    W, b = model.get_layer("out").get_weights()

    def fn(Zs, R, idx, batch: int = 8192) -> np.ndarray:
        idx = np.asarray(idx)
        out = np.empty(len(idx), dtype=np.float64)
        for s in range(0, len(idx), batch):
            j = idx[s:s + batch]
            h = penult.predict(assemble_phi(Zs, R, j), verbose=0, batch_size=len(j))
            out[s:s + len(j)] = (h @ W + b).ravel()
        return out

    return fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num", default="589", help="trial da smoke (default #589, vincitore compressione)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--candidates", default=str(REPO / "runs/secondo_stadio/pa_compress/hardening_candidates.json"))
    ap.add_argument("--cache-dir", default=str(REPO / "runs/dae/output_569"))
    ap.add_argument("--labels-val", default=str(REPO / "runs/preprocessing/val_attack_npy/val_Attack.npy"))
    ap.add_argument("--out-dir", default=str(REPO / "runs/secondo_stadio/loao_pa"))
    a = ap.parse_args()
    setup_blas_env(8, deterministic=False)  # smoke: forward su 8 thread (una sola cella)
    out_dir = Path(a.out_dir)

    cands = json.loads(Path(a.candidates).read_text())
    cand = next((c for c in cands if str(c["num"]) == str(a.num)), None)
    if cand is None:
        raise SystemExit(f"#{a.num} assente in {a.candidates}")
    config = cand["config"]

    # --- censimento pesi (bloccante prima del fan-out) ---
    t0 = time.time()
    present, expected, incomplete = census(cands)
    t_cens = time.time() - t0

    # --- cella smoke ---
    wpath = weight_path(cand["trial_id"], a.seed)
    if not wpath.exists():
        raise SystemExit(f"pesi assenti: {wpath}")

    t0 = time.time()
    sh = load_phi_cache(a.cache_dir, a.labels_val, val_frac=C.VAL_FRAC,
                        split_seed=C.SPLIT_SEED, attack_split=False)
    Zs, R, labels = sh["Zs"], sh["R"], sh["val_attack"]
    ben_sel_abs = sh["ben_pos"][sh["idx_select"]]        # split leak-free seed 0 (== DAE/ceiling)
    att_pos = sh["att_pos"]
    t_load = time.time() - t0

    t0 = time.time()
    model = _load_model(wpath, config, {})               # config-first (h1_from_config, L2_FIXED_S2)
    logit_fn = make_logit_fn(model)
    logit_ben = logit_fn(Zs, R, ben_sel_abs)
    logit_att = logit_fn(Zs, R, att_pos)
    sig_ben = 1.0 / (1.0 + np.exp(-logit_ben))           # = uscita sigmoide del modello
    sig_att = 1.0 / (1.0 + np.exp(-logit_att))
    t_fwd = time.time() - t0

    # cross-check: la sigmoide derivata dal logit == predict_phi canonico (valida l'estrattore)
    chk = predict_phi(model, Zs, R, ben_sel_abs[:2000])
    max_dev = float(np.max(np.abs(chk - sig_ben[:2000])))

    # soglia primaria class-invariant: p99 sui benigni-select
    tau = compute_threshold(sig_ben, FPR_TARGET)
    fpr_sel = float((sig_ben >= tau).mean())

    labels_att = labels[att_pos]
    classes = sorted(set(labels_att.tolist()))
    rows = []
    for cls in classes:
        m = labels_att == cls
        dr = float((sig_att[m] >= tau).mean())
        d = cohens_d(logit_att[m], logit_ben)            # separazione in SCALA LOGIT
        rows.append({"classe": cls, "dr": dr, "d_logit": d, "n": int(m.sum())})

    # --- cache score riusabile (no ricalcolo forward per la variante deploy-realistica/figure) ---
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"smoke_{a.num}_seed{a.seed}_scores.npz"
    np.savez(cache, logit_ben=logit_ben.astype(np.float32),
             logit_att=logit_att.astype(np.float32),
             labels_att=labels_att.astype(str),
             ben_sel_abs=ben_sel_abs, att_pos=att_pos,
             tau=np.float64(tau), num=a.num, seed=a.seed)

    # --- output ---
    print("=" * 78)
    print(f"SMOKE LOAO PA — #{a.num} seme {a.seed} (pesi BASE, pura inferenza)")
    print(f"  config: exp={config['exp']:.4f} nL={config['n_layers']} ratio={config['ratio']:.4f} "
          f"dropout={config['dropout']:.4f} lr={config['lr']:.2e}")
    print(f"  pesi: {wpath.relative_to(REPO)}")
    print(f"  benigni-select (split seed {C.SPLIT_SEED}, val_frac {C.VAL_FRAC}): n={len(ben_sel_abs):,} "
          f"| attacchi val: n={len(att_pos):,}")
    print(f"  soglia primaria tau=p99(benigni-select)={tau:.6f}  ->  FPR_sel={fpr_sel*100:.3f}%")
    print(f"  cross-check sigmoide(logit) vs predict_phi: max|dev|={max_dev:.2e}")
    print("-" * 78)
    print(f"  {'classe':<26}{'DR':>9}{'d-logit':>10}{'n':>10}")
    for r in rows:
        print(f"  {r['classe']:<26}{r['dr']*100:>8.2f}%{r['d_logit']:>10.3f}{r['n']:>10,}")
    print("-" * 78)
    print(f"CENSIMENTO pesi (25 modelli x 19 semi): {present}/{expected} presenti")
    if incomplete:
        print(f"  INCOMPLETI ({len(incomplete)}):")
        for it in incomplete:
            print(f"    #{it['num']}: {it['have']}/19  mancano {it['missing']}")
    else:
        print("  tutti i 25 modelli completi a 19 semi (base + extra) — fan-out eseguibile")
    print(f"TEMPI: censimento {t_cens:.1f}s | load cache {t_load:.1f}s | forward {t_fwd:.1f}s")
    print(f"cache score: {cache.relative_to(REPO)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
