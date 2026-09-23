"""lib/preprocessing/transform.py — log1p + scaler (non_zero_only) + clip + save .npz (Stage 04).

Lo split cluster-aware vive in `split.py` (funzione pura verificata); qui restano le
trasformazioni numeriche deterministiche (skew->log1p, z-score con scaler fit sui benigni
training, clip [-10,10]) e l'orchestrazione dello stage.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lib import utils as common
from lib.preprocessing.split import semisupervised_split, log_split_summary

log = logging.getLogger(__name__)


def detect_negative_values(df: pd.DataFrame, cols: list[str]) -> dict[str, int]:
    """Conta valori negativi per colonna. Ritorna dict {col: count} solo per cols con count>0."""
    out = {}
    for col in cols:
        n_neg = int((df[col] < 0).sum())
        if n_neg > 0:
            out[col] = n_neg
    return out


def compute_skewness(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """Skewness via pandas .skew() (Fisher-Pearson, NaN-safe)."""
    return {col: float(df[col].skew()) for col in cols}


def fit_scaler(df_benign_train: pd.DataFrame, cols: list[str],
               split_features: dict[str, str]) -> dict[str, dict]:
    """Per ogni continua: mean/std sui benigni training.
    split_features (mapping feat -> flag binaria): per quelle, mean/std solo sui non-zero.
    Ritorna {col: {"mean": float, "std": float, "non_zero_only": bool}}.
    """
    params: dict[str, dict] = {}
    for col in cols:
        non_zero_only = col in split_features
        if non_zero_only:
            vals = df_benign_train[col]
            vals = vals[vals != 0]
            if len(vals) == 0:
                raise ValueError(f"split feature {col}: 0 valori non-zero nei benigni training")
        else:
            vals = df_benign_train[col]
        mean = float(vals.mean())
        std = float(vals.std(ddof=0))
        if std == 0.0:
            raise ValueError(f"feature {col}: std=0 sui benigni training (costante)")
        params[col] = {"mean": mean, "std": std, "non_zero_only": non_zero_only}
    return params


def count_out_of_range(values: np.ndarray, lo: float, hi: float) -> int:
    """Conta celle fuori dal range [lo, hi]. Atteso array 2D float."""
    return int(((values < lo) | (values > hi)).sum())


def _save_split_npz(df: pd.DataFrame, idx: np.ndarray, feature_order: list[str],
                    out_dir: Path, split_name: str) -> None:
    """Salva {split_name}.npz (X=float32) e {split_name}_labels.npz (Label int8 + Attack object)."""
    feat_path = out_dir / f"{split_name}.npz"
    label_path = out_dir / f"{split_name}_labels.npz"
    X = df.loc[idx, feature_order].to_numpy(dtype=np.float32)
    np.savez(feat_path, X=X)
    feat_mb = feat_path.stat().st_size / 1_048_576
    log.info("  %s: shape=%s float32 %.1f MB", feat_path.name, X.shape, feat_mb)
    del X

    Label = df.loc[idx, "Label"].to_numpy(dtype=np.int8)
    Attack = df.loc[idx, "Attack"].to_numpy().astype(object)
    np.savez(label_path, Label=Label, Attack=Attack)
    label_mb = label_path.stat().st_size / 1_048_576
    log.info("  %s: Label int8 + Attack object, %.1f MB", label_path.name, label_mb)


def run_transform_stage(cfg: dict) -> None:
    """post_audit.parquet -> train/val/test .npz (feature) + train/val/test_labels.npz (Label,
    Attack) + transform_meta.json + scaler_params.json + feature_order.json (JSON solo in fit).

    Il `cfg` (dict letto da YAML) deve contenere:
      - primary: path al parquet post_audit
      - schema_from: path al feature_types_post_audit.json
      - out_dir: directory output (deve gia' esistere)
      - seed: int (default 42)
      - apply_from: optional, path a transform_cse/ per modalita' apply (UNSW)
    """
    primary_path = cfg["primary"]
    schema_path = cfg["schema_from"]
    out_dir_str = cfg["out_dir"]
    seed = int(cfg.get("seed", 42))
    apply_from = cfg.get("apply_from")

    out_dir = Path(out_dir_str)
    if not out_dir.exists():
        raise FileNotFoundError(f"out_dir non esiste: {out_dir}")
    common.setup_logging(out_dir / "log.txt")

    t_total = time.perf_counter()
    mode = "apply" if apply_from else "fit"
    primary_stem = Path(primary_path).stem.replace("_post_audit", "")
    log.info("===== Stage 04 transform -- %s (%s) =====", primary_stem, mode)
    log.info("primary: %s", primary_path)
    log.info("schema:  %s", schema_path)
    if apply_from:
        log.info("apply_from: %s", apply_from)
    log.info("seed: %d", seed)
    log.info("")

    log.info("loading parquet...")
    t0 = time.perf_counter()
    df = pd.read_parquet(primary_path).reset_index(drop=True)
    with open(schema_path, encoding="utf-8") as f:
        types = json.load(f)
    continuous_cols = [c for c in types if types[c] == "continuous"]
    binary_cols = [c for c in types if types[c] == "binary"]
    feature_order = continuous_cols + binary_cols
    log.info("loaded in %.1f s: rows=%d cols=%d continuous=%d binary=%d feature_order=%d",
             time.perf_counter() - t0, len(df), len(df.columns),
             len(continuous_cols), len(binary_cols), len(feature_order))
    log.info("")

    log.info("--- 1. Split semi-supervisionato cluster-aware (seed=%d) ---", seed)
    log.info("benigni 80/10/10 per cluster  malevoli 0/50/50 per cluster (stratify Attack-majority)")
    t0 = time.perf_counter()
    train_idx, val_idx, test_idx, cluster_stats = semisupervised_split(
        df, feature_cols=continuous_cols + binary_cols, seed=seed)
    log_split_summary(df, train_idx, val_idx, test_idx)
    log.info("tempo: %.1f s", time.perf_counter() - t0)
    log.info("")

    if apply_from is None:
        log.info("--- 2. Verifica negativi nelle continue ---")
        negatives = detect_negative_values(df, continuous_cols)
        if negatives:
            for col, n in negatives.items():
                log.error("  %s: %d valori negativi", col, n)
            raise ValueError(f"valori negativi trovati in {len(negatives)} feature: stop")
        log.info("nessun valore negativo nelle %d continue", len(continuous_cols))
        log.info("")

        log.info("--- 3. log(1+x) sulle continue con skew>%.1f sul training ---", common.SKEW_THRESHOLD)
        t0 = time.perf_counter()
        skews_pre = compute_skewness(df.iloc[train_idx], continuous_cols)
        log_features = sorted([c for c, s in skews_pre.items() if abs(s) > common.SKEW_THRESHOLD])
        log.info("feature con |skew|>%.1f: %d", common.SKEW_THRESHOLD, len(log_features))
        for col in log_features:
            df[col] = np.log1p(df[col].astype(np.float64)).astype(np.float32)
        skews_post = compute_skewness(df.iloc[train_idx], log_features)
        log.info("  %-32s %12s %12s", "feature", "skew_pre", "skew_post")
        for col in log_features:
            log.info("  %-32s %12.4f %12.4f", col, skews_pre[col], skews_post[col])
        log.info("tempo: %.1f s", time.perf_counter() - t0)
        log.info("")

        log.info("--- 4. Scaler fit sul training (solo benigni per architettura semi-sup) ---")
        t0 = time.perf_counter()
        log.info("training: %d righe (tutti benigni)", len(train_idx))
        scaler_params = fit_scaler(df.iloc[train_idx], continuous_cols, common.SPARSE_SPLIT_FLAGS)
        n_normal = sum(1 for v in scaler_params.values() if not v["non_zero_only"])
        n_split = sum(1 for v in scaler_params.values() if v["non_zero_only"])
        log.info("feature continue normali: %d (scaler su tutto il training)", n_normal)
        log.info("feature continue sdoppiate: %d (scaler su non-zero del training)", n_split)
        for col, p in scaler_params.items():
            if p["non_zero_only"]:
                log.info("  %-20s flag=%s mean_nz=%.4f std_nz=%.4f",
                         col, common.SPARSE_SPLIT_FLAGS[col], p["mean"], p["std"])
        log.info("tempo: %.1f s", time.perf_counter() - t0)
        log.info("")
    else:
        src = Path(apply_from)
        log.info("--- 2-4. Carico log_features e scaler dal run sorgente ---")
        with open(src / "transform_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        with open(src / "scaler_params.json", encoding="utf-8") as f:
            scaler_params = json.load(f)
        log_features = list(meta["log_features"])
        log.info("log_features: %d feature", len(log_features))
        log.info("scaler_params: %d feature continue", len(scaler_params))
        for col in log_features:
            df[col] = np.log1p(df[col].astype(np.float64)).astype(np.float32)
        skews_pre = skews_post = {}
        log.info("")

    log.info("--- 5. Apply scaler (z-score) ---")
    t0 = time.perf_counter()
    for col, p in scaler_params.items():
        df[col] = ((df[col].astype(np.float64) - p["mean"]) / p["std"]).astype(np.float32)
    log.info("applicato a %d continue su %d righe in %.1f s",
             len(scaler_params), len(df), time.perf_counter() - t0)
    log.info("")

    log.info("--- 6. Conta truncations + clip [%.0f, %.0f] ---", *common.TRUNCATE_RANGE)
    t0 = time.perf_counter()
    truncate_counts = {}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        sub = df.loc[idx, continuous_cols].to_numpy()
        n_out = count_out_of_range(sub, *common.TRUNCATE_RANGE)
        n_cells = sub.size
        truncate_counts[name] = {"out_of_range": n_out, "total_cells": n_cells,
                                  "permille": n_out / n_cells * 1000 if n_cells else 0}
        log.info("  %s: %d/%d celle fuori range (%.4f permille)",
                 name, n_out, n_cells, truncate_counts[name]["permille"])
        del sub
    for col in continuous_cols:
        df[col] = df[col].clip(*common.TRUNCATE_RANGE)
    log.info("tempo: %.1f s", time.perf_counter() - t0)
    log.info("")

    log.info("--- 7. Save .npz per split ---")
    t0 = time.perf_counter()
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        _save_split_npz(df, idx, feature_order, out_dir, name)
    log.info("tempo: %.1f s", time.perf_counter() - t0)
    log.info("")

    if apply_from is None:
        with open(out_dir / "scaler_params.json", "w", encoding="utf-8") as f:
            json.dump(scaler_params, f, indent=2)
        label_arr = df["Label"].to_numpy()
        meta = {
            "seed": seed,
            "split_mode": "semisupervised_cluster",
            "benign_fracs": list(common.BENIGN_FRACS),
            "malicious_fracs": list(common.MALICIOUS_FRACS),
            "skew_threshold": common.SKEW_THRESHOLD,
            "truncate_range": list(common.TRUNCATE_RANGE),
            "log_features": log_features,
            "skews_pre_log": skews_pre,
            "skews_post_log": skews_post,
            "split_features": dict(common.SPARSE_SPLIT_FLAGS),
            "cluster_stats": cluster_stats,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "n_train_benign": int((label_arr[train_idx] == 0).sum()),
            "n_train_malicious": int((label_arr[train_idx] == 1).sum()),
            "n_val_benign": int((label_arr[val_idx] == 0).sum()),
            "n_val_malicious": int((label_arr[val_idx] == 1).sum()),
            "n_test_benign": int((label_arr[test_idx] == 0).sum()),
            "n_test_malicious": int((label_arr[test_idx] == 1).sum()),
            "truncate_counts": truncate_counts,
        }
        with open(out_dir / "transform_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        with open(out_dir / "feature_order.json", "w", encoding="utf-8") as f:
            json.dump({"order": feature_order,
                       "continuous": continuous_cols,
                       "binary": binary_cols}, f, indent=2)
        log.info("metadata: scaler_params.json, transform_meta.json, feature_order.json")
        log.info("")

    log.info("===== completato, tempo totale: %.1f s =====", time.perf_counter() - t_total)
