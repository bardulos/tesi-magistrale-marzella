#!/usr/bin/env python
# TAG: ONE-SHOT | dati grezzi del test UNSW
"""Rebuild the leak-free UNSW test set as raw nProbe CSV for inference.

Recreates the split and both encoder row filters (deduplication, then DNS cleanup), verifies
features and labels against encoded test artifacts, and writes all test rows. The CSE scaler
metadata is reused. Gates check counts, raw-row alignment, feature identity, coverage, and columns.
Sources are read-only; only --out is written.

Usage: python tools/make_unsw_test_raw.py [flags]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lib.preprocessing.encode import clean_dirty_dns  # noqa: E402
from lib.preprocessing.split import semisupervised_split  # noqa: E402

R5 = Path.home() / "tesi/repo_v5"

EXPECT_RAW = 2_365_424
EXPECT_DEDUP = 2_350_609
EXPECT_SURV = 2_350_577
EXPECT_TEST = 285_366
EXPECT_TEST_BENIGN = 222_290
EXPECT_TEST_MAL = 63_076
COMMON = ["IN_BYTES", "L4_SRC_PORT", "Label", "Attack"]


def log(msg):
    print(msg, flush=True)


def load_feature_lists(types_path):
    """continuous/binary nello STESSO ordine di transform.py (iterazione sul dict)."""
    with open(types_path, encoding="utf-8") as f:
        types = json.load(f)
    continuous = [c for c in types if types[c] == "continuous"]
    binary = [c for c in types if types[c] == "binary"]
    return continuous, binary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(R5 / "runs/preprocessing/ingest/unsw_raw.parquet"))
    ap.add_argument("--post-audit", default=str(R5 / "runs/preprocessing/reduce_unsw/unsw_post_audit.parquet"))
    ap.add_argument("--types", default=str(R5 / "runs/preprocessing/reduce_unsw/feature_types_post_audit.json"))
    ap.add_argument("--test-npz", default=str(R5 / "runs/preprocessing/transform_unsw/test.npz"))
    ap.add_argument("--test-labels", default=str(R5 / "runs/preprocessing/transform_unsw/test_labels.npz"))
    ap.add_argument("--scaler", default=str(ROOT / "inference/weights/cse/scaler_params.json"))
    ap.add_argument("--transform-meta", default=str(ROOT / "inference/weights/cse/transform_meta.json"))
    ap.add_argument("--out", default=str(ROOT / "inference/dati/inferenza/unsw_test_raw.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="    [split] %(message)s")

    continuous_cols, binary_cols = load_feature_lists(args.types)
    feature_cols = continuous_cols + binary_cols
    meta = json.load(open(args.transform_meta, encoding="utf-8"))
    log(f"feature_cols={len(feature_cols)} (continuous={len(continuous_cols)} binary={len(binary_cols)})  seed={args.seed}")

    # ------------------------------------------------------------------ 1. split
    log("\n[1] post_audit -> ricalcolo split deterministico...")
    df_pa = pd.read_parquet(args.post_audit).reset_index(drop=True)
    if len(df_pa) != EXPECT_SURV:
        raise SystemExit(f"post_audit righe {len(df_pa)} != {EXPECT_SURV} attese")
    train_idx, val_idx, test_idx, _ = semisupervised_split(
        df_pa, feature_cols=feature_cols, seed=args.seed)

    # G1: compare split counts with EXPECT_TEST* (the CSE transform_meta.json has no UNSW counts).
    lbl = df_pa["Label"].to_numpy()
    n_test = len(test_idx)
    n_b = int((lbl[test_idx] == 0).sum())
    n_m = int((lbl[test_idx] == 1).sum())
    log(f"[G1] test={n_test} (atteso {EXPECT_TEST})  benigni={n_b} (atteso {EXPECT_TEST_BENIGN})  "
        f"malevoli={n_m} (atteso {EXPECT_TEST_MAL})")
    if not (n_test == EXPECT_TEST and n_b == EXPECT_TEST_BENIGN and n_m == EXPECT_TEST_MAL):
        raise SystemExit("[G1] FAIL: lo split non riproduce i conteggi attesi del test UNSW")
    log("[G1] OK")

    pa_common = df_pa[COMMON].copy()
    df_pa_test = df_pa.iloc[test_idx].copy()
    del df_pa

    # ------------------------------------------------------------------ G3 (prima del raw: libera RAM)
    log("\n[G3] ricostruisco le feature del test (log1p+zscore+clip, metadati CSE) e confronto con test.npz...")
    log_features = list(meta["log_features"])
    scaler = json.load(open(args.scaler, encoding="utf-8"))
    lo, hi = meta["truncate_range"]
    Label_test = df_pa_test["Label"].to_numpy(dtype=np.int8)
    Attack_test = df_pa_test["Attack"].to_numpy().astype(object)
    for col in log_features:
        df_pa_test[col] = np.log1p(df_pa_test[col].astype(np.float64)).astype(np.float32)
    for col, p in scaler.items():
        df_pa_test[col] = ((df_pa_test[col].astype(np.float64) - p["mean"]) / p["std"]).astype(np.float32)
    for col in continuous_cols:
        df_pa_test[col] = df_pa_test[col].clip(lo, hi)
    X_recon = df_pa_test[feature_cols].to_numpy(dtype=np.float32)
    Xref = np.load(args.test_npz)["X"]
    eq = np.array_equal(X_recon, Xref)
    maxd = float(np.abs(X_recon.astype(np.float64) - Xref.astype(np.float64)).max())
    # Attack labels are trusted object arrays from the preprocessing pipeline.
    tl = np.load(args.test_labels, allow_pickle=True)
    l_ok = np.array_equal(Label_test, tl["Label"])
    a_ok = np.array_equal(Attack_test, tl["Attack"])
    log(f"[G3] X_recon{X_recon.shape} vs test.npz{Xref.shape}  array_equal={eq}  max|Δ|={maxd:.3e}  "
        f"Label_match={l_ok}  Attack_match={a_ok}")
    if not (eq and l_ok and a_ok):
        raise SystemExit("[G3] FAIL (BLOCCANTE): ricostruzione non bit-identica a test.npz/test_labels.npz")
    log("[G3] OK")
    del df_pa_test, X_recon, Xref

    # ------------------------------------------------------------------ 2. raw + dedup a DUE stadi
    log("\n[2] raw -> replico i DUE filtri di riga dell'encode (drop_duplicates + clean_dirty_dns)...")
    df_raw = pd.read_parquet(args.raw).reset_index(drop=True)
    if len(df_raw) != EXPECT_RAW:
        raise SystemExit(f"raw righe {len(df_raw)} != {EXPECT_RAW} attese")
    raw_feat_cols = [c for c in df_raw.columns if c not in ("Label", "Attack")]
    raw_dedup = df_raw.drop_duplicates(subset=raw_feat_cols, keep="first").reset_index(drop=True)
    del df_raw
    # G2a: check the intermediate row count.
    log(f"[G2a] dopo drop_duplicates: {len(raw_dedup)} (atteso {EXPECT_DEDUP})")
    if len(raw_dedup) != EXPECT_DEDUP:
        raise SystemExit(f"[G2a] FAIL: drop_duplicates {len(raw_dedup)} != {EXPECT_DEDUP}")
    raw_surv = clean_dirty_dns(raw_dedup).reset_index(drop=True)
    del raw_dedup
    log(f"[2] dopo clean_dirty_dns: raw_surv {len(raw_surv)} (atteso {EXPECT_SURV})")
    if len(raw_surv) != EXPECT_SURV:
        raise SystemExit(f"[2] FAIL: raw_surv {len(raw_surv)} != {EXPECT_SURV} (dedup a due stadi non riproduce l'encode)")

    # G2: compare aligned raw and post-audit rows.
    log("[G2] confronto raw_surv vs post_audit su colonne comuni invarianti...")
    g2 = True
    for col in COMMON:
        ok = np.array_equal(raw_surv[col].to_numpy(), pa_common[col].to_numpy())
        log(f"   {col:14s} array_equal={ok}")
        g2 = g2 and ok
    del pa_common
    if not g2:
        raise SystemExit("[G2] FAIL: raw_surv NON allineato al post_audit")
    log("[G2] OK")

    # ------------------------------------------------------------------ 3. righe grezze del test
    raw_test = raw_surv.iloc[test_idx].reset_index(drop=True)
    del raw_surv
    log(f"\n[3] raw_test righe={len(raw_test)} colonne={len(raw_test.columns)}")

    # G4: verify class coverage.
    log("[G4] copertura per Attack (nessun sottocampionamento):")
    full_c = raw_test["Attack"].value_counts()
    for a in full_c.index:
        log(f"   {str(a):28s} {int(full_c[a]):>9d}")
    nb, nm = int((raw_test["Label"] == 0).sum()), int((raw_test["Label"] == 1).sum())
    if not (len(raw_test) == EXPECT_TEST and nb == EXPECT_TEST_BENIGN and nm == EXPECT_TEST_MAL):
        raise SystemExit(f"[G4] FAIL: totale/benigni/malevoli {len(raw_test)}/{nb}/{nm} inattesi")
    log(f"[G4] OK -- totale={len(raw_test)}  benigni={nb}  malevoli={nm}")

    # ------------------------------------------------------------------ 4. scrittura CSV grezzo
    # G5: use the raw parquet schema for output.
    written_cols = list(raw_test.columns)
    if not ({"Label", "Attack"} <= set(written_cols)):
        raise SystemExit(f"[G5] FAIL: mancano Label/Attack tra le colonne ({written_cols})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    raw_test.to_csv(args.out, index=False)
    log(f"\n[G5] OK -- {len(written_cols)} colonne (schema nProbe grezzo, Label+Attack incluse)")
    log(f"==> scritto {args.out}  ({len(raw_test)} righe)")


if __name__ == "__main__":
    main()
