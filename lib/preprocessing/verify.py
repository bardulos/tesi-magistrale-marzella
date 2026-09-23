"""lib/preprocessing/verify.py — verifica spine del set feature: duplicati (C1-F2) + dati
Spearman per la heatmap di confronto pre/post-audit (C1-F1).

Spine-only: produce i dati gated (conteggi duplicati, % +SRC_PORT ri-derivata dal parquet
encode dello spine, matrici Spearman 39->34). Le analisi di robustezza esplorative di
repo_v3 (truncation/bootstrap/DNS, post-reduce stats) restano archiviate. Il disegno di
C1-F1/C1-F2 vive in lib/plotting (script standalone rimovibile: dal verify_stats.json
prende liste colonne/conteggi, i valori delle matrici li ricalcola dai parquet):
questo modulo contiene solo il DATO gated.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lib import utils as common

log = logging.getLogger(__name__)

# Da escludere dalla heatmap raw: non-feature (etichette, IP) + categoriche encoded numericamente
# (distance-wise non significative). L4_SRC_PORT e L4_DST_PORT RESTANO (semantiche metric-like).
EXCLUDE_RAW_HEATMAP = {
    "Label", "Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
    "PROTOCOL", "L7_PROTO", "DNS_QUERY_TYPE",
    "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "TCP_FLAGS",
}


def compute_duplicate_stats(df: pd.DataFrame, feature_cols: list[str],
                            top_k: int = 20) -> dict:
    """Conta duplicati esatti via pandas.duplicated.

    Ritorna dict con:
      total_rows, unique_keys (n. valori distinti), unique_rows (cluster k=1),
      dup_rows (righe in cluster k>=2), n_clusters_k_ge_2, top_cluster_sizes.
    """
    n_total = len(df)
    is_dup = df.duplicated(subset=feature_cols, keep=False)
    is_first = ~df.duplicated(subset=feature_cols, keep="first")

    dup_rows = int(is_dup.sum())
    unique_keys = int(is_first.sum())
    unique_rows = n_total - dup_rows
    n_clusters = unique_keys - unique_rows

    if n_clusters > 0:
        sizes = (
            df.loc[is_dup, feature_cols]
            .groupby(feature_cols, dropna=False, observed=True, sort=False)
            .size()
            .sort_values(ascending=False)
            .head(top_k)
            .to_list()
        )
    else:
        sizes = []

    return {
        "total_rows": n_total,
        "unique_keys": unique_keys,
        "unique_rows": unique_rows,
        "dup_rows": dup_rows,
        "n_clusters_k_ge_2": n_clusters,
        "top_cluster_sizes": sizes,
    }


def srcport_dup_pct(encode_parquet: Path) -> float:
    """% duplicati sui BENIGNI del parquet encode (post-SRC_PORT, pre-cluster-split).

    Sul set-91 dello spine L4_SRC_PORT e L4_DST_PORT sono entrambe presenti, quindi questo
    valore (la barra +SRC_PORT della figura C1-F2) e' un numero reale ri-derivato dallo
    spine, non il NaN documentato di repo_v3 (dove il parquet stage8 era assente).
    """
    df = pd.read_parquet(encode_parquet)
    benign = df[df["Label"] == 0]
    fc = [c for c in benign.columns if c not in ("Label", "Attack")]
    n_total = len(benign)
    n_uniq = benign[fc].drop_duplicates().shape[0]
    return 100.0 * (n_total - n_uniq) / n_total


def heatmap_continuous_cols(df: pd.DataFrame) -> list[str]:
    """Continue ammesse nella heatmap raw: numeriche, non categoriche-encoded, non apriori,
    con piu' di 2 valori distinti."""
    exclude = EXCLUDE_RAW_HEATMAP | set(common.APRIORI_DROPS_NUMERIC)
    return [c for c in df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].nunique(dropna=True) > 2]


def spearman_redundant_pairs(df: pd.DataFrame, continuous_cols: list[str],
                             threshold: float = 0.90) -> list[tuple[str, str, float]]:
    """Coppie feature-feature con |rho_Spearman| >= threshold, ordinate per |rho| desc.

    Spearman via rank+Pearson (piu' veloce della corr('spearman') nativa). Sostituisce la
    lambda di ordinamento con un reindex su indice ordinato per valore assoluto.
    """
    ranked = df[continuous_cols].rank(method="average")
    corr = ranked.corr(method="pearson")
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack()
    strong = pairs[pairs.abs() >= threshold]
    strong = strong.reindex(strong.abs().sort_values(ascending=False).index)
    return [(a, b, float(rho)) for (a, b), rho in strong.items()]


def run_verify_stage(cfg: dict) -> None:
    """Verifica spine del set feature -> verify_stats.json (le figure si generano a
    parte con lib/plotting/preprocessing_plots.py).

    Il `cfg` (dict letto da YAML) deve contenere primary, schema_from, out_dir:
      - primary: parquet post_audit (set 91)
      - schema_from: feature_types_post_audit.json
      - out_dir: directory output (deve gia' esistere)
    Opzionali (se assenti, i passi relativi sono saltati e i campi restano null):
      - encode_primary: parquet encode (per la barra +SRC_PORT, benigni)
      - raw_primary: parquet grezzo cse_raw (per le continue pre-audit, 39)
    """
    primary_path = cfg["primary"]
    schema_path = cfg["schema_from"]
    encode_primary = cfg.get("encode_primary")
    raw_primary = cfg.get("raw_primary")
    out_dir = Path(cfg["out_dir"])
    if not out_dir.exists():
        raise FileNotFoundError(f"out_dir non esiste: {out_dir}")
    common.setup_logging(out_dir / "log.txt")

    t_total = time.perf_counter()
    log.info("===== Stage verify (spine) -- %s =====", Path(primary_path).stem)
    log.info("loading post_audit parquet...")
    df = pd.read_parquet(primary_path)
    with open(schema_path, encoding="utf-8") as f:
        types = json.load(f)
    feature_cols = [c for c in df.columns if c not in ("Label", "Attack")]
    continuous_post = [c for c, t in types.items() if t == "continuous" and c in df.columns]
    log.info("loaded: rows=%d cols=%d continuous=%d", len(df), len(df.columns), len(continuous_post))
    log.info("")

    log.info("--- 1. Duplicate stats (post_audit, set 87) ---")
    t0 = time.perf_counter()
    dup = compute_duplicate_stats(df, feature_cols)
    dup_pct = dup["dup_rows"] / dup["total_rows"] * 100 if dup["total_rows"] else 0.0
    log.info("  total_rows=%d unique_keys=%d dup_rows=%d (%.2f%%)",
             dup["total_rows"], dup["unique_keys"], dup["dup_rows"], dup_pct)
    log.info("  tempo: %.1f s", time.perf_counter() - t0)
    log.info("")

    srcport_pct = None
    if encode_primary:
        log.info("--- 2. +SRC_PORT dup pct (benigni, parquet encode) ---")
        t0 = time.perf_counter()
        srcport_pct = srcport_dup_pct(Path(encode_primary))
        log.info("  +SRC_PORT (benigni) = %.4f%%  (tempo %.1f s)", srcport_pct, time.perf_counter() - t0)
        log.info("")

    log.info("--- 3. Spearman post_audit (%d continue) ---", len(continuous_post))
    t0 = time.perf_counter()
    redundant_post = spearman_redundant_pairs(df, continuous_post)
    log.info("  continue post-audit: %d | coppie |rho|>=0.90: %d",
             len(continuous_post), len(redundant_post))
    log.info("  tempo: %.1f s", time.perf_counter() - t0)
    log.info("")

    continuous_pre = None
    redundant_pre = None
    if raw_primary:
        log.info("--- 4. Spearman pre-audit (raw filtrato) ---")
        t0 = time.perf_counter()
        df_raw = pd.read_parquet(raw_primary)
        cols_pre = heatmap_continuous_cols(df_raw)
        continuous_pre = list(cols_pre)
        redundant_pre = spearman_redundant_pairs(df_raw, cols_pre)
        log.info("  continue raw filtrate: %d | coppie |rho|>=0.90: %d  (tempo %.1f s)",
                 len(cols_pre), len(redundant_pre), time.perf_counter() - t0)
        del df_raw
        log.info("")

    stats = {
        "duplicate_stats_post_audit": dup,
        "dup_pct_post_audit": dup_pct,
        "srcport_dup_pct_benign": srcport_pct,
        "spearman": {
            "n_cont_pre": len(continuous_pre) if continuous_pre is not None else None,
            "n_cont_post": len(continuous_post),
            "redundant_pairs_pre": redundant_pre,
            "redundant_pairs_post": redundant_post,
            "continuous_pre": continuous_pre,
            "continuous_post": continuous_post,
        },
    }
    with open(out_dir / "verify_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    log.info("salvato verify_stats.json")
    log.info("===== completato, tempo totale: %.1f s =====", time.perf_counter() - t_total)
