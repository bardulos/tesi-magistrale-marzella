#!/usr/bin/env python3
# TAG: DRIVER-PIPELINE | 2026-07-14 | genera inference/dati/inferenza/cse_test_raw.csv (WP-E2E-CSE) | vedi docs/refactoring_censimento.md
"""tools/make_cse_test_raw.py — genera inference/dati/inferenza/cse_test_raw.csv: il test set CSE
leak-free in formato CSV GREZZO nProbe, consumabile da infer.py --test --model cse.

Portato da repo_v5/tools/make_cse_test_raw.py (WP-E2E-CSE, 2026-07-14). ROOT = repov6 (per
`import lib.*`, ereditato verbatim). Gli input GREZZI (parquet raw/post_audit, test.npz,
test_labels.npz, meta/scaler) sono letti dall'oracolo read-only repo_v5 (`V5_BASE`): sono ~3GB, non
si copiano; si leggono una volta per costruire il CSV. Default `--target-rows` = n_test (2.818.847):
scrive il TEST PIENO (decisione WP-E2E-CSE), non un sottocampione.

Perche' serve: il test set CSE leak-free (split semisupervisionato cluster-aware, seed 42) esiste
solo come feature codificate (transform_cse/test.npz); infer.py vuole CSV grezzo e ricalcola lui le
feature via csv_to_features. Qui si ricostruiscono le RIGHE GREZZE del test.

Come (deterministico, riproducibile):
  1. Ricalcolo lo split su cse_post_audit.parquet riusando lib.preprocessing.split.semisupervised_split
     (seed 42) -> test_idx (posizionale).
  2. Riallineo il raw replicando l'UNICO drop di righe dell'encode -- encode.py:375:
     drop_duplicates(subset=feature_cols, keep="first"), order-preserving -- ottenendo raw_surv
     (19.486.615 righe) NELLO STESSO ORDINE del post_audit; quindi raw_surv.iloc[test_idx] sono le
     righe grezze del test.
  3. Sottocampione stratificato per Attack ~--target-rows con floor per categoria (col default = test
     pieno, k=n per ogni categoria).
  4. Scrivo il CSV con le stesse colonne/ordine di unsw_test_raw.csv.

Gate stampati (chiuso = dimostrato):
  G1 conteggi split == transform_meta.json (test/benigni/malevoli).
  G2 allineamento raw<->post_audit su colonne comuni invarianti (IN_BYTES,L4_SRC_PORT,Label,Attack).
  G3 bit-identita': feature ricostruite (log1p+zscore+clip) vs test.npz, e Label/Attack vs test_labels.npz.
  G4 copertura/proporzioni del sottocampione per Attack.
  G5 colonne identiche a unsw_test_raw.csv.

Read-only sulle sorgenti; scrive solo --out. Uso: python tools/make_cse_test_raw.py [flags].
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                       # per `import lib.*`
from lib.preprocessing.split import semisupervised_split  # noqa: E402

# Oracolo read-only: parquet/npz grezzi del preprocessing CSE (spostato su disco esterno 2026-07-14).
V5_BASE = Path("<MEDIA>/CrucialP3Plus/cartelladilavoro/repo_v5/runs/preprocessing")

EXPECT_RAW = 20_115_529       # cse_raw.parquet (encoding_log: 20115529 -> 19486615)
EXPECT_SURV = 19_486_615      # righe sopravvissute al dedup = righe del post_audit
N_TEST = 2_818_847            # righe del test (transform_meta.json) -> default target = test pieno
COMMON = ["IN_BYTES", "L4_SRC_PORT", "Label", "Attack"]   # colonne grezze invarianti presenti in entrambi


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=str(V5_BASE / "ingest/cse_raw.parquet"))
    ap.add_argument("--post-audit", default=str(V5_BASE / "reduce_cse/cse_post_audit.parquet"))
    ap.add_argument("--types", default=str(V5_BASE / "reduce_cse/feature_types_post_audit.json"))
    ap.add_argument("--transform-meta", default=str(V5_BASE / "transform_cse/transform_meta.json"))
    ap.add_argument("--scaler", default=str(V5_BASE / "transform_cse/scaler_params.json"))
    ap.add_argument("--test-npz", default=str(V5_BASE / "transform_cse/test.npz"))
    ap.add_argument("--test-labels", default=str(V5_BASE / "transform_cse/test_labels.npz"))
    ap.add_argument("--template-csv", default=str(ROOT / "inference/dati/inferenza/unsw_test_raw.csv"))
    ap.add_argument("--out", default=str(ROOT / "inference/dati/inferenza/cse_test_raw.csv"))
    ap.add_argument("--target-rows", type=int, default=N_TEST)   # default = test PIENO
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="    [split] %(message)s")

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

    # G1 -- lo split riproduce transform_meta
    lbl = df_pa["Label"].to_numpy()
    n_test, n_b, n_m = len(test_idx), int((lbl[test_idx] == 0).sum()), int((lbl[test_idx] == 1).sum())
    log(f"[G1] test={n_test} (atteso {meta['n_test']})  benigni={n_b} (atteso {meta['n_test_benign']})  "
        f"malevoli={n_m} (atteso {meta['n_test_malicious']})")
    if not (n_test == meta["n_test"] and n_b == meta["n_test_benign"] and n_m == meta["n_test_malicious"]):
        raise SystemExit("[G1] FAIL: lo split non riproduce transform_meta.json")
    log("[G1] OK")

    pa_common = df_pa[COMMON].copy()              # per G2 (allineamento su tutte le righe)
    df_pa_test = df_pa.iloc[test_idx].copy()      # per G3 (ricostruzione feature)
    del df_pa

    # ------------------------------------------------------------------ G3 (prima del raw: libera RAM)
    log("\n[G3] ricostruisco le feature del test (log1p+zscore+clip) e confronto con test.npz...")
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
    # allow_pickle: necessario per l'array Attack (object/stringhe); sorgente fidata = artefatti di
    # pipeline dell'oracolo (repo_v5/.../transform_cse/), non input esterni.
    tl = np.load(args.test_labels, allow_pickle=True)
    l_ok = np.array_equal(Label_test, tl["Label"])
    a_ok = np.array_equal(Attack_test, tl["Attack"])
    log(f"[G3] X_recon{X_recon.shape} vs test.npz{Xref.shape}  array_equal={eq}  max|Δ|={maxd:.3e}  "
        f"Label_match={l_ok}  Attack_match={a_ok}")
    if not (eq and l_ok and a_ok):
        raise SystemExit("[G3] FAIL: ricostruzione non bit-identica a test.npz/test_labels.npz")
    log("[G3] OK")
    del df_pa_test, X_recon, Xref

    # ------------------------------------------------------------------ 2. raw + dedup (allineamento)
    log("\n[2] raw -> replico il dedup dell'encode (encode.py:375)...")
    df_raw = pd.read_parquet(args.raw).reset_index(drop=True)
    if len(df_raw) != EXPECT_RAW:
        raise SystemExit(f"raw righe {len(df_raw)} != {EXPECT_RAW} attese")
    raw_feat_cols = [c for c in df_raw.columns if c not in ("Label", "Attack")]
    raw_surv = df_raw.drop_duplicates(subset=raw_feat_cols, keep="first").reset_index(drop=True)
    n_drop = len(df_raw) - len(raw_surv)
    del df_raw
    log(f"[2] raw {EXPECT_RAW} -> raw_surv {len(raw_surv)} (drop {n_drop})")
    if len(raw_surv) != EXPECT_SURV:
        raise SystemExit(f"[2] FAIL: raw_surv {len(raw_surv)} != {EXPECT_SURV} (dedup non riproduce l'encode)")

    # G2 -- raw_surv allineato al post_audit, riga per riga, sulle colonne comuni invarianti
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

    # ------------------------------------------------------------------ 4. sottocampione stratificato
    log(f"\n[4] sottocampione stratificato per Attack ~{args.target_rows} righe (seed={args.seed})...")
    rng = np.random.default_rng(args.seed)
    frac = args.target_rows / len(raw_test)
    parts = []
    for _attack, grp in raw_test.groupby("Attack", sort=True):
        n = len(grp)
        k = max(1, min(n, int(round(n * frac))))      # floor: >=1 per categoria (col default: k=n)
        sel = np.sort(rng.choice(n, size=k, replace=False))
        parts.append(grp.iloc[sel])
    sub = pd.concat(parts)
    sub = sub.iloc[rng.permutation(len(sub))].reset_index(drop=True)   # mescola benigni/malevoli
    log(f"[4] sottocampione righe={len(sub)}")

    # G4 -- copertura per Attack
    full_c = raw_test["Attack"].value_counts()
    sub_c = sub["Attack"].value_counts()
    log("[G4] copertura per Attack (test_full -> sub):")
    for a in full_c.index:
        fc, sc = int(full_c[a]), int(sub_c.get(a, 0))
        log(f"   {str(a):28s} {fc:>9d} -> {sc:>7d}  ({sc / fc * 100:5.2f}%)")
    missing = [a for a in full_c.index if a not in set(sub_c.index)]
    if missing:
        raise SystemExit(f"[G4] FAIL: categorie Attack perse nel sottocampione: {missing}")
    nb, nm = int((sub["Label"] == 0).sum()), int((sub["Label"] == 1).sum())
    log(f"[G4] OK -- totale={len(sub)}  benigni={nb}  malevoli={nm}")

    # ------------------------------------------------------------------ 5. scrittura (formato unsw)
    template_cols = pd.read_csv(args.template_csv, nrows=0).columns.tolist()
    if set(template_cols) != set(sub.columns):
        only_t = set(template_cols) - set(sub.columns)
        only_c = set(sub.columns) - set(template_cols)
        raise SystemExit(f"[G5] FAIL colonne: solo_template={only_t}  solo_cse={only_c}")
    sub = sub[template_cols]                       # stesso ordine di unsw_test_raw.csv
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    log(f"\n[G5] OK -- colonne identiche a unsw_test_raw.csv ({len(template_cols)})")
    log(f"==> scritto {args.out}  ({len(sub)} righe)")


if __name__ == "__main__":
    main()
