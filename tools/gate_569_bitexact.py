#!/usr/bin/env python
# TAG: DRIVER-PIPELINE | gate bit-exact #569 (rete di regressione)
"""Check bit-exact forward output for frozen DAE #569, seed 2034.

Compares the first n validation scores with the saved output. Run in a fresh process so BLAS
determinism is configured before TensorFlow initializes; the fixed batch preserves BLAS tiling.

Usage: python tools/gate_569_bitexact.py [--n 50000] [--threads 8]
Exit codes: 0 exact, 1 mismatch, 2 required artifacts absent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lib.utils import (N_BINARY, N_CONTINUOUS, N_FEATURES,  # noqa: E402
                       setup_blas_env)

WEIGHTS = REPO / "runs/dae/loao_fase3/weights/569/seed_2034.weights.h5"
VAL_X = Path.home() / "dae_data_91/val_X.npy"
SCORE_VAL = REPO / "runs/dae/output_569/dae_score_val.npy"
CFG_569 = dict(btl=16, exp=112, mid=80, sigma=0.15834)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50_000, help="righe del batch fisso (chunk 0)")
    ap.add_argument("--threads", type=int, default=8, help="thread BLAS (bit-exact anche a 1)")
    a = ap.parse_args()

    for p in (WEIGHTS, VAL_X, SCORE_VAL):
        if not p.exists():
            print(f"[gate569] artefatto assente: {p} -> SKIP", flush=True)
            return 2

    setup_blas_env(a.threads, deterministic=True)
    import numpy as np
    import tensorflow as tf

    from lib.dae import constants as C
    from lib.dae.evaluate import mse_binary_score
    from lib.dae.model import build_dae

    cont_idx = list(range(N_CONTINUOUS))
    bin_idx = list(range(N_CONTINUOUS, N_FEATURES))
    model, _enc = build_dae(n_features=N_FEATURES, btl=CFG_569["btl"], exp=CFG_569["exp"],
                            n_continuous=N_CONTINUOUS, n_binary=N_BINARY,
                            continuous_idx=cont_idx, binary_idx=bin_idx, l2_reg=C.L2_FIXED,
                            noise_std=CFG_569["sigma"], noise_type=C.NOISE_TYPE,
                            loss_continuous=C.LOSS_CONTINUOUS, mid=CFG_569["mid"])
    model.load_weights(str(WEIGHTS))

    val_X = np.load(VAL_X, mmap_mode="r")
    xb = np.ascontiguousarray(val_X[:a.n], dtype=np.float32)
    y_pred = model(tf.constant(xb), training=False).numpy()
    score = mse_binary_score(y_pred, xb)
    ref = np.load(SCORE_VAL)[:a.n]

    if score.dtype == np.float32 and ref.dtype == np.float32 and np.array_equal(score, ref):
        print(f"[gate569] OK bit-exact su n={a.n} (threads={a.threads})", flush=True)
        return 0
    n_diff = int((score != ref).sum())
    md = float(np.abs(score.astype("f8") - ref.astype("f8")).max())
    print(f"[gate569] FAIL non bit-exact: n_diff={n_diff}/{a.n} max|Δ|={md:.3e}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
