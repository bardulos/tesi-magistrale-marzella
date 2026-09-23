"""lib/preprocessing/encode.py — codifica (parquet grezzo -> parquet codificato) +
reduce (drop feature audit + flag sparse). Stage 02 + 03 della pipeline.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lib import utils as common

log = logging.getLogger(__name__)


def clean_dirty_dns(df: pd.DataFrame) -> pd.DataFrame:
    """UNSW: drop righe con DNS_QUERY_TYPE > 255 (dirty data). CSE: no-op."""
    if "DNS_QUERY_TYPE" not in df.columns:
        return df
    n_before = len(df)
    mask = df["DNS_QUERY_TYPE"] > 255
    n_dirty = int(mask.sum())
    if n_dirty == 0:
        log.info("clean_dirty_dns: nessuna riga DNS_QUERY_TYPE>255")
        return df
    df = df.loc[~mask].copy()
    log.info("clean_dirty_dns: drop %d righe (%d -> %d)", n_dirty, n_before, len(df))
    return df


def drop_apriori(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in common.APRIORI_DROPS_NUMERIC if c in df.columns]
    df = df.drop(columns=cols)
    log.info("drop_apriori: droppate %d colonne (%s)", len(cols), ", ".join(cols))
    return df


def encode_protocol(df: pd.DataFrame, types: dict) -> pd.DataFrame:
    """One-hot del protocollo L4: proto_tcp/udp/other registrate in `types`.

    `is_icmp_flow` resta come GATE INTERNO (lo usano derive_icmp e il log qui
    sotto) ma NON e' registrata in `types` -> non entra in feature_order/phi.
    La colonna viene consumata e droppata da derive_icmp (ridondante = somma dei
    7 bucket; toglierla evita una direzione quasi-collineare nel residuo).
    """
    p = df["PROTOCOL"]
    df["proto_tcp"] = (p == 6).astype(np.uint8)
    df["proto_udp"] = (p == 17).astype(np.uint8)
    df["proto_other"] = (~p.isin([6, 17])).astype(np.uint8)
    df["is_icmp_flow"] = (p == 1).astype(np.uint8)
    n = len(df)
    log.info("encode_protocol: tcp=%d (%.2f%%) udp=%d (%.2f%%) other=%d (%.2f%%) icmp=%d (%.2f%%)",
             int(df["proto_tcp"].sum()), df["proto_tcp"].sum() / n * 100,
             int(df["proto_udp"].sum()), df["proto_udp"].sum() / n * 100,
             int(df["proto_other"].sum()), df["proto_other"].sum() / n * 100,
             int(df["is_icmp_flow"].sum()), df["is_icmp_flow"].sum() / n * 100)
    types.update({"proto_tcp": "binary", "proto_udp": "binary",
                  "proto_other": "binary"})
    return df.drop(columns=["PROTOCOL"])


def encode_l4_dst_port(df: pd.DataFrame, types: dict) -> pd.DataFrame:
    """Bucket binari delle porte di destinazione (servizi + fasce residue).

    Unica encode_* che NON droppa la colonna grezza: L4_DST_PORT resta come
    feature continua e coesiste coi 10 bucket binari; il tag della continua e'
    differito a tag_continuous, a fine pipeline.
    """
    n = len(df)
    p = df["L4_DST_PORT"]
    log.info("encode_l4_dst_port: %d valori unici", p.nunique())

    is_assigned = pd.Series(False, index=df.index)
    bucket_map = {}  # port -> bucket name (per top-20 lookup)
    for name, ports in common.L4_PORT_BUCKETS:
        mask = p.isin(ports)
        df[name] = mask.astype(np.uint8)
        is_assigned = is_assigned | mask
        types[name] = "binary"
        for port in ports:
            bucket_map[port] = name
        log.info("  %s (%s): %d (%.4f%%)", name, ports, int(mask.sum()), int(mask.sum()) / n * 100)

    mask_low = (p < 1024) & (~is_assigned)
    df["port_other_low"] = mask_low.astype(np.uint8)
    is_assigned = is_assigned | mask_low
    types["port_other_low"] = "binary"
    log.info("  port_other_low (<1024 altre): %d (%.4f%%)", int(mask_low.sum()), int(mask_low.sum()) / n * 100)

    mask_high = (p >= 1024) & (~is_assigned)
    df["port_high"] = mask_high.astype(np.uint8)
    types["port_high"] = "binary"
    log.info("  port_high (>=1024 altre): %d (%.4f%%)", int(mask_high.sum()), int(mask_high.sum()) / n * 100)

    log.info("encode_l4_dst_port: top-20 con bucket assegnato:")
    for port, count in p.value_counts().head(20).items():
        pct = count / n * 100
        if port in bucket_map:
            bucket = bucket_map[port]
        elif port < 1024:
            bucket = "port_other_low"
        else:
            bucket = "port_high"
        warn = " <- WARNING: >=1% in port_high, valutare bucket dedicato" if (bucket == "port_high" and pct >= 1.0) else ""
        log.info("  port=%s count=%d (%.4f%%) -> %s%s", port, count, pct, bucket, warn)

    return df


def encode_l7_proto(df: pd.DataFrame, types: dict, schema_l7_values=None) -> pd.DataFrame:
    n = len(df)
    p = df["L7_PROTO"]
    n_uniq = p.nunique()
    is_top = pd.Series(False, index=df.index)

    if schema_l7_values is not None:
        log.info("encode_l7_proto: schema mode, %d valori L7 da schema (dataset %d unici)",
                 len(schema_l7_values), n_uniq)
        for val in schema_l7_values:
            col = f"l7_{val}"
            mask = (p == val)
            df[col] = mask.astype(np.uint8)
            is_top = is_top | mask
            types[col] = "binary"
            log.info("  %s: %d (%.4f%%)", col, int(mask.sum()), int(mask.sum()) / n * 100)
    else:
        counts = p.value_counts()
        top15 = counts.head(15)
        coverage = top15.sum() / n * 100
        log.info("encode_l7_proto: %d valori unici. Top-15 coverage: %.2f%%", n_uniq, coverage)
        if coverage < 95.0:
            log.info("  WARNING: top-15 copre %.2f%% (<95%%), valutare top-K maggiore", coverage)
        for val, count in top15.items():
            col = f"l7_{int(val)}"
            mask = (p == val)
            df[col] = mask.astype(np.uint8)
            is_top = is_top | mask
            types[col] = "binary"
            log.info("  %s (val=%s): %d (%.4f%%)", col, val, count, count / n * 100)

    mask_other = ~is_top
    df["l7_other"] = mask_other.astype(np.uint8)
    types["l7_other"] = "binary"
    log.info("  l7_other: %d (%.4f%%)", int(mask_other.sum()), int(mask_other.sum()) / n * 100)

    return df.drop(columns=["L7_PROTO"])


def encode_dns_query_type(df: pd.DataFrame, types: dict) -> pd.DataFrame:
    q = df["DNS_QUERY_TYPE"]
    is_dns = (q > 0)
    is_a = (q == 1)
    is_aaaa = (q == 28)
    is_other = is_dns & ~is_a & ~is_aaaa
    df["is_dns_flow"] = is_dns.astype(np.uint8)
    df["dns_type_a"] = is_a.astype(np.uint8)
    df["dns_type_aaaa"] = is_aaaa.astype(np.uint8)
    df["dns_type_other"] = is_other.astype(np.uint8)
    types.update({"is_dns_flow": "binary", "dns_type_a": "binary",
                  "dns_type_aaaa": "binary", "dns_type_other": "binary"})
    n = len(df)
    log.info("encode_dns_query_type: dns=%d (%.2f%%) a=%d aaaa=%d other=%d",
             int(is_dns.sum()), is_dns.sum() / n * 100,
             int(is_a.sum()), int(is_aaaa.sum()), int(is_other.sum()))
    return df.drop(columns=["DNS_QUERY_TYPE"])


def encode_tcp_flags(df: pd.DataFrame, col: str, prefix: str, types: dict, schema_bits=None) -> pd.DataFrame:
    n = len(df)
    flags = df[col].fillna(0).astype(np.int64)

    if schema_bits is not None:
        log.info("encode_tcp_flags: %s schema mode, bit attesi: %s", col, list(schema_bits))
        bits_set = set(schema_bits)
        for bit_name, bit_idx in common.TCP_FLAG_BITS:
            if bit_name in bits_set:
                new_col = f"{prefix}_{bit_name}"
                mask = ((flags & (1 << bit_idx)) > 0)
                df[new_col] = mask.astype(np.uint8)
                types[new_col] = "binary"
                log.info("  %s (bit %d): %d set (%.4f%%)", new_col, bit_idx,
                         int(mask.sum()), int(mask.sum()) / n * 100)
    else:
        threshold = 100
        log.info("encode_tcp_flags: %s (max=%d) bit decomposition (threshold attivi >=%d):",
                 col, int(flags.max()), threshold)
        for bit_name, bit_idx in common.TCP_FLAG_BITS:
            mask = ((flags & (1 << bit_idx)) > 0)
            n_set = int(mask.sum())
            new_col = f"{prefix}_{bit_name}"
            if threshold <= n_set <= n - threshold:
                df[new_col] = mask.astype(np.uint8)
                types[new_col] = "binary"
                log.info("  %s (bit %d): %d set (%.4f%%) -- generata", new_col, bit_idx, n_set, n_set / n * 100)
            else:
                reason = "costante 0" if n_set < threshold else "costante 1"
                log.info("  %s (bit %d): %d set (%.4f%%) -- SKIPPED (%s)", new_col, bit_idx, n_set, n_set / n * 100, reason)
    return df.drop(columns=[col])


# Bucket nell'ordine di EMISSIONE (= ordine delle colonne nel blocco binario).
ICMP_BUCKETS = [
    "icmp_echo",            # (8,0) richiesta U (0,0) risposta
    "icmp_unreach_port",    # (3,3)
    "icmp_unreach_admin",   # (3,9) U (3,10) U (3,13)
    "icmp_unreach_other",   # type 3, altri code validi (0..15)
    "icmp_time_exceeded",   # (11,0) U (11,1)
    "icmp_nonstandard",     # code non valido per il type (tabella RFC)
    "icmp_other",           # coda rara + novita' (redirect, timestamp, type non in tabella, ...)
]

# Codici ammessi per type (RFC 792 / IANA). Coppia (type,code) con type IN tabella
# e code NON ammesso -> nonstandard; type NON in tabella (con code qualunque) -> other.
ICMP_VALID_CODES = {
    0: {0}, 3: set(range(16)), 5: {0, 1, 2, 3}, 8: {0},
    11: {0, 1}, 12: {0, 1, 2}, 13: {0}, 17: {0},
}


def derive_icmp(df: pd.DataFrame, types: dict) -> pd.DataFrame:
    """v5 ICMP: 7 bucket one-hot semantici, GATED su is_icmp_flow.

    I 7 bucket sostituiscono le 2 continue ICMP decoded del canonico; il gating
    elimina la contaminazione del campo type grezzo (~7.76% dei non-ICMP su CSE).
    Semantica RFC 792 / IANA + frequenza; nessuna dipendenza dal label.
    Decodifica grezza (stessa aritmetica del canonico): type t = ICMP_IPV4_TYPE,
    code c = ICMP_TYPE - ICMP_IPV4_TYPE*256. Si assegna un solo bucket per flusso
    ICMP con precedenza fissa (ordine ICMP_BUCKETS); i non-ICMP restano 0 in tutti
    i bucket. Le colonne grezze ICMP e la colonna-gate is_icmp_flow sono droppate.
    """
    n = len(df)
    icmp = df["is_icmp_flow"].to_numpy().astype(bool)
    t = df["ICMP_IPV4_TYPE"].fillna(0).astype(np.int64).to_numpy()
    c = df["ICMP_TYPE"].fillna(0).astype(np.int64).to_numpy() - t * 256

    bucket = np.full(n, -1, dtype=np.int64)  # -1 = non assegnato (non-ICMP o pre-assegnazione)

    def assign(mask, idx):
        sel = icmp & mask & (bucket == -1)   # gate ICMP + precedenza (primo match vince)
        bucket[sel] = idx

    assign(((t == 8) & (c == 0)) | ((t == 0) & (c == 0)), 0)   # echo (richiesta U risposta)
    assign((t == 3) & (c == 3), 1)                             # unreach_port
    assign((t == 3) & np.isin(c, (9, 10, 13)), 2)              # unreach_admin
    assign((t == 3) & (c >= 0) & (c <= 15), 3)                 # unreach_other (code validi rimasti)
    assign((t == 11) & np.isin(c, (0, 1)), 4)                  # time_exceeded
    in_table = np.zeros(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)
    for typ, codes in ICMP_VALID_CODES.items():
        sel = (t == typ)
        in_table |= sel
        valid |= sel & np.isin(c, tuple(codes))
    assign(in_table & ~valid, 5)                               # nonstandard (code non valido per type)
    assign(np.ones(n, dtype=bool), 6)                          # other (resto dei flussi ICMP)

    for i, name in enumerate(ICMP_BUCKETS):
        df[name] = (bucket == i).astype(np.uint8)
        types[name] = "binary"

    drop = [col for col in ("ICMP_TYPE", "ICMP_IPV4_TYPE", "is_icmp_flow") if col in df.columns]
    df = df.drop(columns=drop)
    types.pop("is_icmp_flow", None)  # difensivo: il gate non deve finire in feature_order
    counts = {name: int(df[name].sum()) for name in ICMP_BUCKETS}
    log.info("derive_icmp [v5 7-bucket gated]: icmp=%d one-hot=%s drop=%s",
             int(icmp.sum()), counts, ", ".join(drop))
    return df


def normalize_second_bytes(df: pd.DataFrame) -> pd.DataFrame:
    """Sostituisce Inf con max finito CSE-derived, NaN con 0.
    Su CSE 0 sostituzioni; su UNSW NaN+Inf normalizzati."""
    for col, max_val in common.SECOND_BYTES_INF_REPLACEMENT.items():
        if col not in df.columns:
            continue
        n_inf = int(np.isinf(df[col]).sum())
        n_nan = int(df[col].isna().sum())
        if n_inf:
            df.loc[np.isinf(df[col]), col] = max_val
        if n_nan:
            df[col] = df[col].fillna(0)
        log.info("normalize_second_bytes: %s -- %d Inf -> %d, %d NaN -> 0", col, n_inf, max_val, n_nan)
    return df


def tag_continuous(df: pd.DataFrame, types: dict) -> None:
    n_cont = 0
    for col in df.columns:
        if col in types or col in ("Label", "Attack"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "continuous"
            n_cont += 1
    log.info("tag_continuous: %d feature numeriche taggate continuous", n_cont)


def load_schema(schema_path: Path) -> dict:
    """Legge feature_types.json ed estrae info strutturate per encoding schema-driven."""
    with open(schema_path, encoding="utf-8") as f:
        types_dict = json.load(f)
    columns = list(types_dict.keys())

    l7_values: list[int] = []
    for c in columns:
        if c.startswith("l7_") and c != "l7_other":
            try:
                l7_values.append(int(c.split("_", 1)[1]))
            except ValueError:
                pass

    bit_names = {b for b, _ in common.TCP_FLAG_BITS}
    client_bits, server_bits = [], []
    for c in columns:
        if c.startswith("client_") and c.split("_", 1)[1] in bit_names:
            client_bits.append(c.split("_", 1)[1])
        elif c.startswith("server_") and c.split("_", 1)[1] in bit_names:
            server_bits.append(c.split("_", 1)[1])

    return {
        "columns": columns,
        "types": types_dict,
        "l7_values": l7_values,
        "client_bits": client_bits,
        "server_bits": server_bits,
    }


def align_to_schema(df: pd.DataFrame, types: dict, schema: dict) -> pd.DataFrame:
    """Forza df a matchare schema: aggiunge mancanti (binary->0, continuous->NaN), rimuove extra, riordina."""
    expected = schema["columns"]
    expected_types = schema["types"]
    feature_cols = [c for c in df.columns if c not in ("Label", "Attack")]

    missing = [c for c in expected if c not in feature_cols]
    extra = [c for c in feature_cols if c not in expected]

    for c in missing:
        col_type = expected_types.get(c, "binary")
        if col_type == "binary":
            df[c] = np.uint8(0)
        else:
            df[c] = np.float64("nan")
        types[c] = col_type
        log.info("align_to_schema: aggiunta %s (%s, fill=%s)",
                 c, col_type, "0" if col_type == "binary" else "NaN")

    if extra:
        for c in extra:
            log.info("align_to_schema: rimossa %s (non in schema)", c)
            if c in types:
                del types[c]
        df = df.drop(columns=extra)

    target_cols = [c for c in ("Label", "Attack") if c in df.columns]
    df = df[expected + target_cols]
    log.info("align_to_schema: %d colonne aggiunte, %d rimosse, finale=%d feature + %d target",
             len(missing), len(extra), len(expected), len(target_cols))
    return df


def save_outputs(df: pd.DataFrame, types: dict, dataset: str, out_dir: Path) -> None:
    parquet_path = out_dir / f"{dataset}_encoded.parquet"
    df.to_parquet(parquet_path, compression="snappy", index=False)
    size_mb = parquet_path.stat().st_size / 1_048_576
    log.info("save_outputs: %s -- %d righe, %d colonne, %.1f MB",
             parquet_path, len(df), len(df.columns), size_mb)

    types_path = out_dir / "feature_types.json"
    with open(types_path, "w", encoding="utf-8") as f:
        json.dump(types, f, indent=2)  # no sort_keys: preserve insertion order for --schema-from
    n_bin = sum(1 for v in types.values() if v == "binary")
    n_cont = sum(1 for v in types.values() if v == "continuous")
    log.info("save_outputs: %s -- %d feature (binary=%d, continuous=%d)",
             types_path, len(types), n_bin, n_cont)


def run_encode_stage(cfg: dict) -> None:
    """Parquet grezzo -> parquet codificato + feature_types.json.

    Il `cfg` (dict letto da YAML) deve contenere:
      - primary: path al parquet input
      - out_dir: directory output (deve gia' esistere)
      - schema_from: optional, path schema JSON (per UNSW)

    Il nome dataset (cse/unsw) si ricava dallo stem del path
    (es. cse_raw.parquet -> 'cse').
    """
    primary_path = cfg["primary"]
    out_dir_str = cfg["out_dir"]
    schema_from = cfg.get("schema_from")

    out_dir = Path(out_dir_str)
    if not out_dir.exists():
        raise FileNotFoundError(f"out_dir non esiste: {out_dir}")
    common.setup_logging(out_dir / "encoding_log.txt")

    primary = Path(primary_path)
    dataset = primary.stem.replace("_raw", "")
    log.info("===== Stage 02 encode -- %s =====", dataset)
    schema = None
    if schema_from is not None:
        schema = load_schema(Path(schema_from))
        log.info("schema-from: %s, %d colonne attese (binary=%d, continuous=%d)",
                 schema_from, len(schema["columns"]),
                 sum(1 for v in schema["types"].values() if v == "binary"),
                 sum(1 for v in schema["types"].values() if v == "continuous"))

    log.info("path: %s size: %.1fMB", primary, primary.stat().st_size / 1_048_576)
    log.info("loading parquet...")
    df = pd.read_parquet(primary)
    n_rows_initial = len(df)
    n_cols_initial = len(df.columns)
    log.info("loaded: rows=%d cols=%d", n_rows_initial, n_cols_initial)
    log.info("")

    # --- dedup pre-codifica ---
    # subset = tutte le colonne TRANNE Label/Attack: il keep="first" e' order-dependent solo se esistono
    # collisioni feature-identiche con etichetta discordante. Diagnostico one-shot (script di sviluppo
    # non conservato; su CSE 20.1M righe): collisioni con LABEL discordante = 0 (dedup label-safe);
    # con ATTACK discordante = 440 gruppi / 880 righe (0.004%, solo attribuzione di sottotipo, non il Label).
    feature_cols = [c for c in df.columns if c not in ("Label", "Attack")]
    n_before = len(df)
    df = df.drop_duplicates(subset=feature_cols, keep="first")
    n_after = len(df)
    log.info("dedup pre-codifica: %d righe rimosse (%.2f%%), da %d a %d",
             n_before - n_after, (n_before - n_after) / n_before * 100,
             n_before, n_after)
    log.info("")

    df = clean_dirty_dns(df)
    df = drop_apriori(df)

    types: dict[str, str] = {}
    df = encode_protocol(df, types)
    log.info("")
    df = encode_l4_dst_port(df, types)
    log.info("")
    df = encode_l7_proto(df, types, schema_l7_values=schema["l7_values"] if schema else None)
    log.info("")
    df = encode_dns_query_type(df, types)
    log.info("")
    df = encode_tcp_flags(df, "CLIENT_TCP_FLAGS", "client", types,
                          schema_bits=schema["client_bits"] if schema else None)
    log.info("")
    df = encode_tcp_flags(df, "SERVER_TCP_FLAGS", "server", types,
                          schema_bits=schema["server_bits"] if schema else None)
    log.info("")
    df = derive_icmp(df, types)
    df = df.drop(columns=[c for c in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR") if c in df.columns])
    log.info("drop IPV4 string columns (no encoding RFC1918)")
    df = normalize_second_bytes(df)
    tag_continuous(df, types)
    log.info("")

    if schema is not None:
        df = align_to_schema(df, types, schema)
        log.info("")
    else:
        feature_cols = list(types.keys())
        target_cols = [c for c in ("Label", "Attack") if c in df.columns]
        df = df[feature_cols + target_cols]

    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        log.warning("NaN inattesi in %d colonne:", len(nan_cols))
        for col, n in nan_cols.items():
            log.warning("  %s: %d", col, int(n))
    else:
        log.info("sanity check: 0 NaN")
    log.info("")

    n_bin = sum(1 for v in types.values() if v == "binary")
    n_cont = sum(1 for v in types.values() if v == "continuous")
    log.info("riepilogo: rows %d -> %d, cols %d -> %d (binary=%d, continuous=%d)",
             n_rows_initial, len(df), n_cols_initial, len(df.columns), n_bin, n_cont)
    save_outputs(df, types, dataset, out_dir)
    log.info("===== completato =====")


def reduce_features(df: pd.DataFrame, types: dict) -> tuple[pd.DataFrame, dict]:
    """Applica decisioni audit Stage 03: drop AUDIT_DROP_FEATURES + add has_ttl, has_tcp_win_out.

    Ordine colonne finale (preserva insertion pandas):
      [binary_orig, coi 7 bucket icmp_*] + [continuous_orig minus 6 dropped]
        + [has_ttl, has_tcp_win_out] + [Label, Attack]

    Non muta df / types caller-side: lavora su copia di types e ritorna un nuovo df.
    """
    types = dict(types)

    drops = [c for c in common.AUDIT_DROP_FEATURES if c in df.columns]
    df = df.drop(columns=drops)
    log.info("reduce: drop %d feature audit (%s)", len(drops), ", ".join(drops))
    for c in common.AUDIT_DROP_FEATURES:
        types.pop(c, None)

    for src_col, flag_col in common.SPARSE_SPLIT_FLAGS.items():
        if src_col not in df.columns:
            raise KeyError(f"reduce: feature sorgente {src_col} non trovata in df")
        df[flag_col] = (df[src_col] > 0).astype(np.uint8)
        types[flag_col] = "binary"
        n_set = int(df[flag_col].sum())
        log.info("reduce: aggiunta %s = (%s > 0): %d set (%.2f%%)",
                 flag_col, src_col, n_set, n_set / len(df) * 100)

    target_cols = [c for c in ("Label", "Attack") if c in df.columns]
    feature_cols = [c for c in df.columns if c not in target_cols]
    df = df[feature_cols + target_cols]

    return df, types


def run_reduce_stage(cfg: dict) -> None:
    """encoded.parquet + feature_types.json -> post_audit.parquet + feature_types_post_audit.json.

    Il `cfg` (dict letto da YAML) deve contenere:
      - primary: path al parquet encoded
      - schema_from: path al feature_types.json
      - out_dir: directory output (deve gia' esistere)
    """
    primary_path = cfg["primary"]
    schema_path = cfg["schema_from"]
    out_dir_str = cfg["out_dir"]

    out_dir = Path(out_dir_str)
    if not out_dir.exists():
        raise FileNotFoundError(f"out_dir non esiste: {out_dir}")
    common.setup_logging(out_dir / "log.txt")

    log.info("===== Stage reduce -- %s =====", Path(primary_path).stem)
    log.info("primary: %s", primary_path)
    log.info("schema:  %s", schema_path)

    log.info("loading parquet...")
    df = pd.read_parquet(primary_path)
    with open(schema_path, encoding="utf-8") as f:
        types = json.load(f)
    log.info("loaded: rows=%d cols=%d types=%d", len(df), len(df.columns), len(types))
    log.info("")

    df, types = reduce_features(df, types)

    n_bin = sum(1 for v in types.values() if v == "binary")
    n_cont = sum(1 for v in types.values() if v == "continuous")
    log.info("")
    log.info("reduce: rows=%d cols=%d (binary=%d, continuous=%d)",
             len(df), len(df.columns), n_bin, n_cont)

    out_name = Path(primary_path).stem.replace("_encoded", "_post_audit")
    parquet_path = out_dir / f"{out_name}.parquet"
    df.to_parquet(parquet_path, compression="snappy", index=False)
    size_mb = parquet_path.stat().st_size / 1_048_576
    log.info("save: %s (%.1f MB, %d righe, %d colonne)",
             parquet_path, size_mb, len(df), len(df.columns))

    types_path = out_dir / "feature_types_post_audit.json"
    with open(types_path, "w", encoding="utf-8") as f:
        json.dump(types, f, indent=2)
    log.info("save: %s (%d feature)", types_path, len(types))
    log.info("===== completato =====")
