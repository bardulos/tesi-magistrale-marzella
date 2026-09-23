#!/usr/bin/env python
# TAG: ONE-SHOT | 2026-07-13 | WP-E2E dati grezzi UNSW dall'oracolo (citato nel report) | vedi docs/refactoring_censimento.md
# ONE-SHOT (setup/gate dati dall'oracolo repo_v5 per WP-E2E)
"""tools/make_unsw_test_raw.py — genera inference/dati/inferenza/unsw_test_raw.csv: il test set UNSW
leak-free in formato CSV GREZZO nProbe, consumabile da infer.py --test --model unsw.

Perche' serve: il test set UNSW leak-free (split semisupervisionato cluster-aware, seed 42) esiste
solo come feature codificate (transform_unsw/test.npz); infer.py vuole CSV grezzo e ricalcola lui le
feature via csv_to_features. Qui si ricostruiscono le RIGHE GREZZE del test (tutte le 285.366, nessun
sottocampionamento: il test UNSW e' gia' piccolo) e si scrive il CSV, nel formato nProbe a 55 colonne
(le stesse del CSV grezzo NF-UNSW-NB15-v3.csv, con Label e Attack a 9 classi in coda).

Differenze rispetto al gemello CSE (repo_v5/tools/make_cse_test_raw.py, riferimento di LOGICA):
  - DEDUP A DUE STADI: l'encode UNSW applica drop_duplicates POI clean_dirty_dns (drop righe con
    DNS_QUERY_TYPE>255, 32 righe su UNSW; no-op su CSE). Un porting del solo drop_duplicates
    sballerebbe l'allineamento raw<->post_audit di 32 righe. Entrambe le funzioni si importano da
    lib.preprocessing.encode (zero reimplementazione).
  - NESSUN TEMPLATE pregresso: unsw_test_raw.csv e' il primo -> il "template" e' lo schema raw stesso
    (le colonne del parquet grezzo). G5 lo verifica.
  - NESSUN SOTTOCAMPIONAMENTO: si scrivono tutte le 285.366 righe del test.
  - Scaler/metadati preprocessing: UNSW riusa lo scaler di CSE (transform_unsw gira in apply_from=
    transform_cse) -> per G3 si usano i metadati di inference/modelli/produzione/cse/ (byte-identici a
    transform_cse). transform_unsw NON ha un transform_meta.json proprio: G1 confronta i conteggi
    split contro test_labels.npz['Label'], non contro un meta inesistente.

Come (deterministico, riproducibile):
  1. Ricalcolo lo split su unsw_post_audit.parquet riusando lib.preprocessing.split.semisupervised_split
     (seed 42, feature_cols = continuous+binary dei tipi post-audit) -> test_idx (posizionale).
  2. G3 (BLOCCANTE): ricostruisco le feature del test (log1p+zscore+clip coi metadati CSE) e le
     confronto bit-exact con test.npz['X']; Label/Attack con test_labels.npz.
  3. Riallineo il raw replicando i DUE filtri di riga dell'encode:
     drop_duplicates(subset=feature_cols, keep="first") POI clean_dirty_dns -> raw_surv (2.350.577
     righe) NELLO STESSO ORDINE del post_audit; quindi raw_surv.iloc[test_idx] sono le righe grezze
     del test.
  4. Scrivo il CSV con le colonne del parquet grezzo (Label/Attack incluse).

Gate stampati (chiuso = dimostrato):
  G1  conteggi split (test/benigni/malevoli) == test_labels.npz.
  G3  bit-identita': feature ricostruite (log1p+zscore+clip) vs test.npz, e Label/Attack vs
      test_labels.npz. BLOCCANTE: se il CSV non riproduce bit-exact il .npy, il test misurerebbe un
      modello diverso da quello di laboratorio.
  G2a conteggio intermedio post-dedup-pre-dns (isola quale dei due filtri sballa, se sballa).
  G2  allineamento raw_surv<->post_audit su colonne comuni invarianti (IN_BYTES,L4_SRC_PORT,Label,Attack).
  G4  copertura per Attack (no-op difensivo: 100% per costruzione, nessun sottocampione).
  G5  colonne scritte == colonne del raw (superset dello schema nProbe atteso da infer.py).

Read-only sulle sorgenti (oracolo repo_v5 + metadati in inference/modelli/produzione/cse); scrive solo --out.
Uso: python tools/make_unsw_test_raw.py [flags].
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
from lib.preprocessing.encode import clean_dirty_dns  # noqa: E402
from lib.preprocessing.split import semisupervised_split  # noqa: E402

R5 = Path.home() / "tesi/repo_v5"

EXPECT_RAW = 2_365_424        # unsw_raw.parquet (encoding_log: loaded rows=2365424)
EXPECT_DEDUP = 2_350_609      # dopo drop_duplicates (encoding_log: dedup -> 2350609)
EXPECT_SURV = 2_350_577       # dopo clean_dirty_dns = righe del post_audit (encoding_log: -> 2350577)
EXPECT_TEST = 285_366
EXPECT_TEST_BENIGN = 222_290
EXPECT_TEST_MAL = 63_076
COMMON = ["IN_BYTES", "L4_SRC_PORT", "Label", "Attack"]   # invarianti presenti in raw E in post_audit


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

    # G1 -- conteggi split vs test_labels.npz (transform_unsw non ha transform_meta.json proprio)
    lbl = df_pa["Label"].to_numpy()
    n_test = len(test_idx)
    n_b = int((lbl[test_idx] == 0).sum())
    n_m = int((lbl[test_idx] == 1).sum())
    log(f"[G1] test={n_test} (atteso {EXPECT_TEST})  benigni={n_b} (atteso {EXPECT_TEST_BENIGN})  "
        f"malevoli={n_m} (atteso {EXPECT_TEST_MAL})")
    if not (n_test == EXPECT_TEST and n_b == EXPECT_TEST_BENIGN and n_m == EXPECT_TEST_MAL):
        raise SystemExit("[G1] FAIL: lo split non riproduce i conteggi attesi del test UNSW")
    log("[G1] OK")

    pa_common = df_pa[COMMON].copy()              # per G2 (allineamento su tutte le righe)
    df_pa_test = df_pa.iloc[test_idx].copy()      # per G3 (ricostruzione feature)
    del df_pa

    # ------------------------------------------------------------------ G3 (prima del raw: libera RAM)
    log("\n[G3] ricostruisco le feature del test (log1p+zscore+clip, metadati CSE) e confronto con test.npz...")
    log_features = list(meta["log_features"])
    scaler = json.load(open(args.scaler, encoding="utf-8"))
    lo, hi = meta["truncate_range"]
    Label_test = df_pa_test["Label"].to_numpy(dtype=np.int8)
    Attack_test = df_pa_test["Attack"].to_numpy().astype(object)
    for col in log_features:                       # 1) log1p sulle skewed (f64 -> f32)
        df_pa_test[col] = np.log1p(df_pa_test[col].astype(np.float64)).astype(np.float32)
    for col, p in scaler.items():                  # 2) z-score su tutte le 34 continue (apply a TUTTE le righe, zeri inclusi)
        df_pa_test[col] = ((df_pa_test[col].astype(np.float64) - p["mean"]) / p["std"]).astype(np.float32)
    for col in continuous_cols:                     # 3) clip [-10,10] sulle sole continue
        df_pa_test[col] = df_pa_test[col].clip(lo, hi)
    X_recon = df_pa_test[feature_cols].to_numpy(dtype=np.float32)
    Xref = np.load(args.test_npz)["X"]
    eq = np.array_equal(X_recon, Xref)
    maxd = float(np.abs(X_recon.astype(np.float64) - Xref.astype(np.float64)).max())
    # allow_pickle: array Attack (stringhe/object) da nostri artefatti di pipeline, non input esterni.
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
    # G2a -- conteggio intermedio: isola quale filtro sballa se qualcosa non torna
    log(f"[G2a] dopo drop_duplicates: {len(raw_dedup)} (atteso {EXPECT_DEDUP})")
    if len(raw_dedup) != EXPECT_DEDUP:
        raise SystemExit(f"[G2a] FAIL: drop_duplicates {len(raw_dedup)} != {EXPECT_DEDUP}")
    raw_surv = clean_dirty_dns(raw_dedup).reset_index(drop=True)
    del raw_dedup
    log(f"[2] dopo clean_dirty_dns: raw_surv {len(raw_surv)} (atteso {EXPECT_SURV})")
    if len(raw_surv) != EXPECT_SURV:
        raise SystemExit(f"[2] FAIL: raw_surv {len(raw_surv)} != {EXPECT_SURV} (dedup a due stadi non riproduce l'encode)")

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

    # G4 -- copertura per Attack (nessun sottocampione: no-op difensivo, deve essere 100%)
    log("[G4] copertura per Attack (nessun sottocampionamento):")
    full_c = raw_test["Attack"].value_counts()
    for a in full_c.index:
        log(f"   {str(a):28s} {int(full_c[a]):>9d}")
    nb, nm = int((raw_test["Label"] == 0).sum()), int((raw_test["Label"] == 1).sum())
    if not (len(raw_test) == EXPECT_TEST and nb == EXPECT_TEST_BENIGN and nm == EXPECT_TEST_MAL):
        raise SystemExit(f"[G4] FAIL: totale/benigni/malevoli {len(raw_test)}/{nb}/{nm} inattesi")
    log(f"[G4] OK -- totale={len(raw_test)}  benigni={nb}  malevoli={nm}")

    # ------------------------------------------------------------------ 4. scrittura CSV grezzo
    # G5 -- lo schema raw E' il template (primo unsw_test_raw.csv). Colonne = quelle del parquet grezzo.
    written_cols = list(raw_test.columns)
    if not ({"Label", "Attack"} <= set(written_cols)):
        raise SystemExit(f"[G5] FAIL: mancano Label/Attack tra le colonne ({written_cols})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    raw_test.to_csv(args.out, index=False)
    log(f"\n[G5] OK -- {len(written_cols)} colonne (schema nProbe grezzo, Label+Attack incluse)")
    log(f"==> scritto {args.out}  ({len(raw_test)} righe)")


if __name__ == "__main__":
    main()
