#!/usr/bin/env python3
"""infer.py — inferenza NIDS a 2 stadi (DAE #569 + classificatore pseudo-anomalie #589), NumPy puro.

Monolite autocontenuto: a runtime usa solo NumPy. Pipeline: CSV nProbe -> 91 feature -> DAE e PA
-> fusione dei punteggi -> alert. Le soglie sono calibrate sui benigni-select del dominio.

La modalita' A privilegia FPR minimo; la B aumenta la recall. --live cattura nProbe, --batch
analizza un CSV e --test calcola metriche con etichette. --bench aggiunge misure per-step e un JSON.

Il whitening Ledoit-Wolf e la standardizzazione del latente sono fusi nel primo layer PA. Il
forward usa float32; il piano di fusione promuove a float64. I residui per gli alert sono lazy.
"""

import argparse
import csv
import functools
import json
import math
import itertools
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

try:
    import resource  # solo Unix: serve unicamente al picco di memoria del JSON di --bench
except ImportError:  # Windows: il resto del pacchetto funziona senza
    resource = None

# Thread BLAS — impostati PRIMA di importare numpy (l'env va letto da OpenBLAS/MKL all'init).
# L'inferenza e' MONOPROCESSO a thread Python singolo: l'UNICO parallelismo e' quello interno del BLAS
# sulle matmul. Su questi matmul piccoli (91-107 dim) il default a tutti i core va in OVERSUBSCRIPTION
# ed e' il piu' lento; il PUNTO DOLCE misurato e' 4 (plateau; 8/16 peggiorano). Override via env
# (setdefault): OPENBLAS_NUM_THREADS=N davanti al comando per tarare sulla propria CPU.
# Il numero di thread BLAS si fissa qui: bench_run.sh lo eredita senza ridichiararlo, mentre
# main.BLAS_INFER ne tiene una copia da allineare a mano, perche' il benchmark misuri il deploy.
# Sweep 1-12 thread (workstation, 8 core fisici): plateau 4-6 indistinguibili, crollo del 30%
# oltre gli 8 thread (confine dell'iper-threading).
BLAS_DEFAULT = "4"
_BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
for _v in _BLAS_ENV:
    os.environ.setdefault(_v, BLAS_DEFAULT)

import numpy as np

# Geometria canonica, allineata a lib/utils.py e ai modelli #569/#589.

N_FEATURES = 91
N_CONTINUOUS = 34
N_BINARY = 57
BIN_SLICE = slice(N_CONTINUOUS, N_FEATURES)

# DAE #569: 91 -> 112 -> 80 -> 16 -> 80 -> 112 -> 91.
DAE_EXP, DAE_MID, DAE_BTL = 112, 80, 16
# PA #589: input phi=[z_s(16), r_w(91)] in R^107; hidden [420, 258], one logit.
PHI_DIM = DAE_BTL + N_FEATURES

WEIGHTS_DIR = Path(__file__).resolve().parent / "modelli"
CHUNK_SIZE = 1_024
SIGMA_K = 3.0

# Costanti di preprocessing duplicate per mantenere il monolite autocontenuto.
L4_PORT_BUCKETS = [
    ("port_dns",   [53]),
    ("port_rdp",   [3389]),
    ("port_https", [443, 8443]),
    ("port_http",  [80, 8080, 8000, 8888]),
    ("port_smb",   [445]),
    ("port_ftp",   [20, 21]),
    ("port_ssh",   [22]),
    ("port_smtp",  [25, 465, 587]),
]

# (bit_name, bit_idx). client_* usa tutti gli 8 bit; server_* esclude "urg".
TCP_FLAG_BITS = [
    ("fin", 0), ("syn", 1), ("rst", 2), ("psh", 3),
    ("ack", 4), ("urg", 5), ("ece", 6), ("cwr", 7),
]
SERVER_BIT_SET = {"fin", "syn", "rst", "psh", "ack", "ece", "cwr"}

SECOND_BYTES_INF_REPLACEMENT = {
    "SRC_TO_DST_SECOND_BYTES": 46938,
    "DST_TO_SRC_SECOND_BYTES": 2015,
}

# L7 IDs e ordine dal modello. Il match float e' esatto: non troncare nDPI master.sub.
L7_KEEP_IDS = [5, 88, 91, 7, 10, 0, 41, 1, 92, 77, 154, 131, 103, 9, 100]

TRUNCATE_RANGE = (-10.0, 10.0)

# ICMP: sette bucket one-hot attivati da is_icmp_flow, che non rientra nelle 91 feature.
ICMP_BUCKETS = [
    "icmp_echo",            # (8,0) richiesta U (0,0) risposta
    "icmp_unreach_port",    # (3,3)
    "icmp_unreach_admin",   # (3,9) U (3,10) U (3,13)
    "icmp_unreach_other",   # type 3, altri code validi (0..15)
    "icmp_time_exceeded",   # (11,0) U (11,1)
    "icmp_nonstandard",     # code non valido per il type (tabella RFC)
    "icmp_other",           # coda rara + novita' (redirect, timestamp, type non in tabella, ...)
]
# Codici ammessi per type (RFC 792 / IANA). (type in tabella, code non ammesso) -> nonstandard;
# type non in tabella -> other.
ICMP_VALID_CODES = {
    0: {0}, 3: set(range(16)), 5: {0, 1, 2, 3}, 8: {0},
    11: {0, 1}, 12: {0, 1, 2}, 13: {0}, 17: {0},
}

# Colonne continue raw del CSV nProbe, allineate a feature_order["continuous"].
_CONTINUOUS_RAW = [
    "L4_SRC_PORT", "L4_DST_PORT",
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN", "DURATION_OUT",
    "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_OUT_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_OUT", "DNS_TTL_ANSWER",
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
]

# Colonne nProbe attese: rileva input mancanti prima che la coercizione li confonda con zeri validi.
EXPECTED_INPUT_COLS = set(_CONTINUOUS_RAW) | {
    "PROTOCOL", "L7_PROTO", "DNS_QUERY_TYPE", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "ICMP_TYPE", "ICMP_IPV4_TYPE",
}


# Preprocessing: parsing ed encoding NumPy, senza pandas.


_COERCE_STATS = {"garbage_a_zero": 0}


def _to_float(s):
    if not s:
        return 0.0
    try:
        return float(s)
    except (ValueError, TypeError):
        _COERCE_STATS["garbage_a_zero"] += 1
        return 0.0


_INT64_MAX = 9223372036854775807
_INT64_MIN = -9223372036854775808


def _to_int(s):
    if not s:
        return 0
    try:
        v = int(s)
    except ValueError:
        try:
            v = int(float(s))
        except (ValueError, TypeError, OverflowError):
            return 0
    # Evita overflow in np.fromiter con valori CSV fuori dal range int64.
    if v > _INT64_MAX:
        return _INT64_MAX
    if v < _INT64_MIN:
        return _INT64_MIN
    return v


def _col_float(rows, col):
    return np.fromiter((_to_float(r.get(col, "")) for r in rows),
                       dtype=np.float64, count=len(rows))


def _col_int(rows, col):
    return np.fromiter((_to_int(r.get(col, "")) for r in rows),
                       dtype=np.int64, count=len(rows))


def _to_l7(s):
    """Parse L7_PROTO come float esatto; vuoto o illeggibile diventa NaN.

    Corrisponde alla pipeline di training: 7.37 resta distinto da 7 e va in l7_other."""
    if not s:
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def _col_l7(rows):
    return np.fromiter((_to_l7(r.get("L7_PROTO", "")) for r in rows),
                       dtype=np.float64, count=len(rows))


def encode_protocol_np(proto_arr):
    return {
        "proto_tcp":    (proto_arr == 6).astype(np.uint8),
        "proto_udp":    (proto_arr == 17).astype(np.uint8),
        "proto_other": (~np.isin(proto_arr, [6, 17])).astype(np.uint8),
        "is_icmp_flow": (proto_arr == 1).astype(np.uint8),
    }


def encode_l4_dst_port_np(port_arr):
    out = {}
    is_assigned = np.zeros(len(port_arr), dtype=bool)
    for name, ports in L4_PORT_BUCKETS:
        mask = np.isin(port_arr, ports)
        out[name] = mask.astype(np.uint8)
        is_assigned |= mask
    mask_low = (port_arr < 1024) & (~is_assigned)
    out["port_other_low"] = mask_low.astype(np.uint8)
    is_assigned |= mask_low
    mask_high = (port_arr >= 1024) & (~is_assigned)
    out["port_high"] = mask_high.astype(np.uint8)
    return out


def encode_l7_proto_np(l7_arr):
    out = {}
    is_top = np.zeros(len(l7_arr), dtype=bool)
    for v in L7_KEEP_IDS:
        mask = (l7_arr == v)
        out[f"l7_{v}"] = mask.astype(np.uint8)
        is_top |= mask
    out["l7_other"] = (~is_top).astype(np.uint8)
    return out


def encode_dns_query_type_np(qt_arr):
    is_dns = (qt_arr > 0)
    is_a = (qt_arr == 1)
    is_aaaa = (qt_arr == 28)
    is_other = is_dns & ~is_a & ~is_aaaa
    return {
        "is_dns_flow":    is_dns.astype(np.uint8),
        "dns_type_a":     is_a.astype(np.uint8),
        "dns_type_aaaa":  is_aaaa.astype(np.uint8),
        "dns_type_other": is_other.astype(np.uint8),
    }


def encode_tcp_flags_np(flags_arr, prefix, allowed_bits=None):
    out = {}
    # Valori non finiti o fuori [0,255] non impostano bit; i valori validi seguono lib.
    f = flags_arr.astype(np.float64)
    f = np.where(np.isfinite(f) & (f >= 0.0) & (f <= 255.0), f, 0.0)
    flags = f.astype(np.int64)
    for bit_name, bit_idx in TCP_FLAG_BITS:
        if allowed_bits is not None and bit_name not in allowed_bits:
            continue
        out[f"{prefix}_{bit_name}"] = ((flags & (1 << bit_idx)) > 0).astype(np.uint8)
    return out


def derive_icmp_np(rows, is_icmp_flow):
    """Assegna un bucket ICMP one-hot con precedenza fissa; i non-ICMP restano a zero."""
    n = len(rows)
    t = _col_int(rows, "ICMP_IPV4_TYPE")
    c = _col_int(rows, "ICMP_TYPE") - t * 256
    icmp = is_icmp_flow.astype(bool)
    bucket = np.full(n, -1, dtype=np.int64)

    def assign(mask, idx):
        sel = icmp & mask & (bucket == -1)
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

    return {name: (bucket == i).astype(np.uint8) for i, name in enumerate(ICMP_BUCKETS)}


def normalize_second_bytes_np(arr, col_name):
    # _col_float restituisce array float64 non aliasati, modificabili in-place.
    cap = SECOND_BYTES_INF_REPLACEMENT.get(col_name)
    arr[np.isnan(arr)] = 0.0
    arr[np.isinf(arr)] = float(cap) if cap is not None else 0.0
    return arr


# Instrumentazione opt-in: senza --bench i decoratori restituiscono le funzioni originali.
# Campioni in us/flusso; il denominatore e' il chunk fissato da timed_total.
_TIMINGS = defaultdict(list)
_BENCH_STATE = {"n_flows": 1}
_BENCH_NET = {"bytes": 0, "flows": 0}

# Distribuzione EMPIRICA della dimensione delle finestre nProbe (flussi per file), per il bench a
# micro-batch VARIABILE (simula il ritmo reale del live -> boxplot di latenza realistici). Campionata
# dalle 9.814 finestre non vuote della cattura LAN reale (share SMB): campionamento STRATIFICATO seedato
# (50 bin uguali sull'ordinata, 1 pescaggio casuale per bin, rng(0)), ordine mescolato. Media 216,9 e
# mediana 191 ~= popolazione (221,1 / 190). Statica: il bench non dipende dal file SMB.
NPROBE_WINDOW_SAMPLE = (
    464, 53, 164, 75, 297, 103, 137, 260, 166, 118,
    129, 303, 88, 278, 147, 233, 219, 45, 341, 180,
    428, 245, 203, 328, 225, 227, 109, 142, 256, 22,
    195, 287, 246, 152, 121, 213, 1291, 157, 393, 31,
    12, 313, 187, 174, 5, 65, 372, 178, 200, 269,
)


def timed(fn):
    """Con --bench registra il tempo per flusso; altrimenti lascia fn invariata."""
    if "--bench" not in sys.argv:
        return fn

    @functools.wraps(fn)
    def _wrap(*a, **k):
        t0 = time.perf_counter()
        r = fn(*a, **k)
        dt_us = (time.perf_counter() - t0) * 1e6
        n = _BENCH_STATE["n_flows"] or 1
        _TIMINGS[fn.__name__].append(dt_us / n)
        return r
    return _wrap


def timed_total(fn):
    """Imposta il denominatore, registra il totale e conta i byte fuori dal tempo misurato."""
    if "--bench" not in sys.argv:
        return fn

    @functools.wraps(fn)
    def _wrap(rows, *a, **k):
        n = max(len(rows), 1)
        _BENCH_STATE["n_flows"] = n
        t0 = time.perf_counter()
        r = fn(rows, *a, **k)
        dt_us = (time.perf_counter() - t0) * 1e6
        nb = (sum(_to_int(x.get("IN_BYTES", 0)) + _to_int(x.get("OUT_BYTES", 0)) for x in rows)
              if rows else 0)
        _TIMINGS["total"].append(dt_us / n)
        _BENCH_NET["bytes"] += nb
        _BENCH_NET["flows"] += len(rows)
        return r
    return _wrap


def bench_payload(w, n_flows, wall, arch, dataset):
    """Crea il JSON consumato da plot_bench.py."""
    lat = {s: {"samples_us": [round(v, 3) for v in c]} for s, c in _TIMINGS.items() if c}
    gbps = (_BENCH_NET["bytes"] * 8 / 1e9 / wall) if wall > 0 else 0.0
    return {
        "arch": arch,
        "mode": w["_mode"],
        "dataset": dataset,
        "blas_threads": int(os.environ.get("OPENBLAS_NUM_THREADS", BLAS_DEFAULT)),
        "throughput_flows_per_sec": round(n_flows / wall) if wall > 0 else 0,
        "throughput_gbps": round(gbps, 3),
        "latency_us": lat,
        # Il throughput strumentato non rappresenta quello di deploy.
        "instrumented": True,
        "n_flows": n_flows,
        "wall_s": round(wall, 3),
        "total_net_bytes": _BENCH_NET["bytes"],
        # In KB su Linux; il processo singolo rende questo il picco RSS del deploy. None senza resource.
        "peak_rss_mb": (round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
                        if resource is not None else None),
    }



# Feature mancanti gia' segnalate (warning una tantum per processo).
_WARNED_MISSING_FEATURES: set = set()


@timed
def csv_to_features(rows, feature_order, scaler_params, log_features):
    """Converte righe nProbe in feature float32 (N, 91), seguendo il preprocessing di training.

    Per MAX_TTL e TCP_WIN_MAX_OUT lo scaler e' fittato sui non-zero, ma applicato a tutti: lo zero
    resta un segnale del modello e diventa (0-mu)/sigma.
    """
    if not rows:
        return np.zeros((0, len(feature_order)), dtype=np.float32)

    N = len(rows)
    # Rileva le colonne mancanti prima che la coercizione le confonda con zeri validi.
    available = {k for r in rows for k in r}
    for col in sorted(EXPECTED_INPUT_COLS - available):
        if col not in _WARNED_MISSING_FEATURES:
            _WARNED_MISSING_FEATURES.add(col)
            print(f"WARNING: colonna nota '{col}' assente dallo schema in ingresso -> coercizione a 0 "
                  f"(degradazione graziosa). 'missing == 0' e' indistinguibile da un valore 0 reale.",
                  file=sys.stderr, flush=True)

    raw_cols = {}
    for col in _CONTINUOUS_RAW:
        raw_cols[col] = _col_float(rows, col)

    proto = _col_int(rows, "PROTOCOL")
    l4_dst = _col_int(rows, "L4_DST_PORT")
    l7 = _col_l7(rows)                              # float esatto, non troncato, come nel training
    dns_qt = _col_int(rows, "DNS_QUERY_TYPE")
    client_flags = _col_float(rows, "CLIENT_TCP_FLAGS")
    server_flags = _col_float(rows, "SERVER_TCP_FLAGS")

    proto_bins = encode_protocol_np(proto)
    is_icmp_flow = proto_bins.pop("is_icmp_flow")
    bins = {}
    bins.update(proto_bins)
    bins.update(encode_l4_dst_port_np(l4_dst))
    bins.update(encode_l7_proto_np(l7))
    bins.update(encode_dns_query_type_np(dns_qt))
    bins.update(encode_tcp_flags_np(client_flags, "client", allowed_bits=None))
    bins.update(encode_tcp_flags_np(server_flags, "server", allowed_bits=SERVER_BIT_SET))
    bins.update(derive_icmp_np(rows, is_icmp_flow))

    for col in ("SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"):
        raw_cols[col] = normalize_second_bytes_np(raw_cols[col], col)

    bins["has_ttl"] = (raw_cols["MAX_TTL"] > 0).astype(np.uint8)
    bins["has_tcp_win_out"] = (raw_cols["TCP_WIN_MAX_OUT"] > 0).astype(np.uint8)

    for col, arr in raw_cols.items():
        if col in ("SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"):
            continue
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    log_set = set(log_features)
    for col in raw_cols:
        if col in log_set:
            arr = raw_cols[col]
            np.maximum(arr, 0.0, out=arr)        # in-place (deploy: log1p di un negativo -> NaN)
            np.log1p(arr, out=arr)
            # lib/preprocessing arrotonda il log1p a float32 PRIMA dello scaler
            # (transform.py: df[col]=np.log1p(...).astype(float32)); replichiamo il round-trip
            # f32 per restare bit-exact coi dati su cui il DAE #569 e' stato addestrato.
            raw_cols[col] = arr.astype(np.float32).astype(np.float64)

    lo, hi = TRUNCATE_RANGE
    for col, p in scaler_params.items():
        if col not in raw_cols:
            continue
        arr = raw_cols[col]
        std_safe = p["std"] if p["std"] != 0 else 1.0
        arr = (arr - p["mean"]) / std_safe
        np.clip(arr, lo, hi, out=arr)
        raw_cols[col] = arr

    out = np.empty((N, len(feature_order)), dtype=np.float32)
    for j, name in enumerate(feature_order):
        if name in raw_cols:
            out[:, j] = raw_cols[name]
        elif name in bins:
            out[:, j] = bins[name]
        else:
            # Fallback per feature richieste dal modello ma non prodotte dal parser.
            if name not in _WARNED_MISSING_FEATURES:
                _WARNED_MISSING_FEATURES.add(name)
                print(f"WARNING: '{name}' in feature_order ma non prodotta dal parser -> riempita con 0 "
                      f"(fallback difensivo).", file=sys.stderr, flush=True)
            out[:, j] = 0.0
    return out


# ============================================================================
# FORWARD DAE (#569) + FOLD sbiancamento/standardizzazione nel 1o layer del PA (#589)
# ============================================================================
# Forward float32; fold dei pesi una tantum al load in float64, salvato in float32.


@timed
def forward_dae(X, w):
    """Esegue il DAE #569 e ritorna il latente z e la ricostruzione float32 Y_hat."""
    h = np.maximum(0, X @ w["W_enc_exp"] + w["b_enc_exp"])
    h = np.maximum(0, h @ w["W_enc_mid"] + w["b_enc_mid"])
    z = np.maximum(0, h @ w["W_btl"] + w["b_btl"])
    h = np.maximum(0, z @ w["W_dec_mid"] + w["b_dec_mid"])
    h = np.maximum(0, h @ w["W_dec_exp"] + w["b_dec_exp"])
    out_cont = h @ w["W_out_cont"] + w["b_out_cont"]
    out_bin = 1.0 / (1.0 + np.exp(-np.clip(h @ w["W_out_bin"] + w["b_out_bin"], -80.0, 80.0)))
    Y_hat = np.concatenate([out_cont, out_bin], axis=1).astype(np.float32)
    return z, Y_hat


def compute_fold(w):
    """Fonde whitening Ledoit-Wolf e standardizzazione latente nel primo layer PA.

    Scrive i pesi fusi in float32; l'operazione e' idempotente.
    """
    sz = np.maximum(w["sigma_z"].astype(np.float64), 1e-8)
    W0 = w["pa_W0"].astype(np.float64)
    b0 = w["pa_b0"].astype(np.float64)
    Wl = w["W"].astype(np.float64)                          # Cholesky di Sigma^-1 (whitening)
    mu_z = w["mu_z"].astype(np.float64)
    mu_R = w["mu_R"].astype(np.float64)
    btl = len(sz)
    W0_z, W0_r = W0[:btl], W0[btl:]
    w["W0_fused_z"] = (np.diag(1.0 / sz) @ W0_z).astype(np.float32)
    w["W0_fused_r"] = (Wl @ W0_r).astype(np.float32)
    w["b0_fused"] = (b0 - (mu_z / sz) @ W0_z - (mu_R @ Wl) @ W0_r).astype(np.float32)
    return w


@timed
def compute_scores_fused(X, Y_hat, z, w):
    """Calcola score DAE (MSE binaria), logit PA e score PA dal forward fuso."""
    R = (Y_hat - X).astype(np.float32)
    diff_bin = R[:, BIN_SLICE]
    s_dae = np.mean(diff_bin * diff_bin, axis=1).astype(np.float32)
    h = np.maximum(0, z @ w["W0_fused_z"] + R @ w["W0_fused_r"] + w["b0_fused"])
    n_layers = int(w.get("pa_n_layers", 2))
    for i in range(1, n_layers):
        h = np.maximum(0, h @ w[f"pa_W{i}"] + w[f"pa_b{i}"])
    logit_pa = (h @ w["pa_W_out"] + w["pa_b_out"]).ravel().astype(np.float32)
    s_pa = (1.0 / (1.0 + np.exp(-np.clip(logit_pa, -80.0, 80.0)))).astype(np.float32)
    return s_dae, s_pa, logit_pa


def _verify_fold(w, X_sample):
    """Verifica in float64 che il primo layer fuso coincida con quello esplicito.

    Usa i pesi originali, non i pesi fusi float32 del runtime. Ritorna max|delta| su h0.
    """
    z, Y_hat = forward_dae(X_sample, w)
    z = z.astype(np.float64)
    R = (Y_hat - X_sample).astype(np.float64)
    sz = np.maximum(w["sigma_z"].astype(np.float64), 1e-8)
    mu_z = w["mu_z"].astype(np.float64)
    mu_R = w["mu_R"].astype(np.float64)
    Wl = w["W"].astype(np.float64)
    W0 = w["pa_W0"].astype(np.float64)
    b0 = w["pa_b0"].astype(np.float64)
    btl = len(sz)
    W0z, W0r = W0[:btl], W0[btl:]
    # esplicito: [z_s, r_w] @ W0 + b0
    Z_s = (z - mu_z) / sz
    R_w = (R - mu_R) @ Wl
    h_expl = np.maximum(0, np.concatenate([Z_s, R_w], axis=1) @ W0 + b0)
    # fuso (formula in f64): z @ (diag(1/sz)W0z) + R @ (W W0r) + [b0 - (mu_z/sz)W0z - (mu_R W)W0r]
    W0_fused_z = np.diag(1.0 / sz) @ W0z
    W0_fused_r = Wl @ W0r
    b0_fused = b0 - (mu_z / sz) @ W0z - (mu_R @ Wl) @ W0r
    h_fused = np.maximum(0, z @ W0_fused_z + R @ W0_fused_r + b0_fused)
    return float(np.max(np.abs(h_expl - h_fused)))


# Fusione a tre rami, allineata a lib/secondo_stadio/fusione.py.
# Soglie derivate dai benigni-select; il piano opera in float64 dopo il forward float32.

_EPS_FUS = 1e-12

FUSION_MODES = {
    "A": {"tau_dae_pct": 97.0, "alpha_and": 0.003, "fpr_target": None, "tau_hi_pct": 99.8},
    "B": {"tau_dae_pct": 99.0, "alpha_and": 0.005, "fpr_target": 0.01, "tau_hi_pct": 99.8},
}

# Prevalenza di riferimento canonica (conteggi VAL di CSE, pi~0.3709): MCC_bal e' SEMPRE calcolato
# qui, MAI sui conteggi del set valutato -> confrontabile fra domini (prevalenza di riferimento).
FUS_N_B = 1_775_098
FUS_N_A = 1_046_755


def fus_logit(s):
    """Calcola il logit dello score sigmoide PA con clipping ai bordi."""
    s = np.clip(np.asarray(s, np.float64), _EPS_FUS, 1.0 - _EPS_FUS)
    return np.log(s / (1.0 - s))


def to_u(s_dae, pa_logit):
    """Punto grezzo [log10(s_dae), logit_PA], convertito in float64."""
    x = np.log10(np.clip(np.asarray(s_dae, np.float64), _EPS_FUS, None))
    return np.column_stack([x, np.asarray(pa_logit, np.float64)])


def fit_standardizer(u_train):
    mu = u_train.mean(0)
    sd = u_train.std(0)
    sd[sd < _EPS_FUS] = 1.0
    return mu, sd


def to_z(u, mu, sd):
    return (np.asarray(u, np.float64) - mu) / sd


def tau_z_from_dae(tau_dae, mu, sd):
    """Soglia DAE grezza portata nel piano z: (log10(tau_dae) - mu0)/sd0."""
    return float((np.log10(tau_dae) - mu[0]) / sd[0])


def and_theta_z(z_select, tau_z, alpha):
    """theta_z tale che FPR_AND(select)=alpha (tau_z fisso). Capped se alpha >= FPR del solo DAE."""
    z_select = np.asarray(z_select, np.float64)
    gated = z_select[z_select[:, 0] > tau_z, 1]
    n, n_gated = len(z_select), len(gated)
    if n_gated == 0:
        return float("inf"), True
    k = alpha * n
    if k >= n_gated:
        return float("-inf"), True
    return float(np.quantile(gated, 1.0 - k / n_gated)), False


def and_predict(z, tau_z, theta_z):
    # Convenzione ">", non ">=". results_v2.json non e' coerente con una sola convenzione: il
    # suo fpr_val si riproduce solo con ">=" (0,004487071 / 0,011137401), il fpr_test solo con ">"
    # (0,004237075 / 0,047313913). La causa sono i pareggi esatti nel selection set (204 su
    # tau_hi_z, 167 su tau_z; zero nei CSV di inferenza su 6,8 M di confronti), che spostano la
    # calibrazione e con essa fpr_test di Mod B (5,6e-7). ">" riproduce l'FPR di test pubblicato
    # e lascia invariata la decisione deployata.
    z = np.asarray(z, np.float64)
    return (z[:, 0] > tau_z) & (z[:, 1] > theta_z)


def calibrate_union_theta(z_select, tau_z, theta_z, fpr_target):
    """theta_union sul logit (z1) tale che FPR(R_AND u {z1>theta})=fpr_target sui select
    (AND fisso, soglia sui benigni-liberi non gia' presi dall'AND). Leak-free. +inf se l'AND
    copre gia' il budget."""
    z_select = np.asarray(z_select, np.float64)
    and_flag = and_predict(z_select, tau_z, theta_z)
    n = len(z_select)
    n_residual = fpr_target * n - int(and_flag.sum())
    free = ~and_flag
    if n_residual <= 0 or not free.any():
        return float("inf")
    logit_free = z_select[free, 1]
    q = 1.0 - n_residual / len(logit_free)
    return float(np.quantile(logit_free, np.clip(q, 0.0, 1.0)))


def fuse_predict(z, tau_z, theta_z, theta_union=float("inf"), tau_hi_z=float("inf")):
    """R_fus = R_AND u {z1>theta_union} u {z0>tau_hi_z} (regola canonica a tre rami). theta_union=
    +inf -> ramo union spento (v1 = caso degenere tau_hi_z=+inf; Mod A = caso degenere
    theta_union=+inf). Confronti numpy con +inf sono sempre False: un'unica funzione copre tutto."""
    z = np.asarray(z, np.float64)
    and_flag = and_predict(z, tau_z, theta_z)
    return and_flag | (z[:, 1] > theta_union) | (z[:, 0] > tau_hi_z)


def build_fusion(s_dae_train, logit_train, s_dae_select, logit_select,
                 tau_dae, fpr_target, alpha_and, tau_hi=None):
    """Parametri di fusione leak-free dai benigni train/select. tau_dae/tau_hi = percentili dello
    score DAE grezzo (portati nel piano z); theta_z e theta_union sono soglie sul logit z-scored (z1)."""
    u_train = to_u(s_dae_train, logit_train)
    mu, sd = fit_standardizer(u_train)
    tau_z = tau_z_from_dae(tau_dae, mu, sd)
    z_select = to_z(to_u(s_dae_select, logit_select), mu, sd)
    theta_z, capped = and_theta_z(z_select, tau_z, alpha_and)
    theta_union = (float("inf") if fpr_target is None
                   else calibrate_union_theta(z_select, tau_z, theta_z, fpr_target))
    tau_hi_z = tau_z_from_dae(tau_hi, mu, sd) if tau_hi is not None else float("inf")
    return {"mu": mu, "sd": sd, "tau_z": tau_z, "theta_z": theta_z,
            "theta_union": theta_union, "tau_hi_z": tau_hi_z, "tau_hi": tau_hi,
            "alpha_and": alpha_and, "fpr_target": fpr_target, "capped": bool(capped)}


@timed
def apply_fusion(params, s_dae, pa_logit):
    """Applica la regola di fusione a (s_dae, logit_PA) -> flag booleano di anomalia."""
    z = to_z(to_u(s_dae, pa_logit), params["mu"], params["sd"])
    return fuse_predict(z, params["tau_z"], params["theta_z"], params["theta_union"],
                        params.get("tau_hi_z", float("inf")))


def fired_branch(params, s_dae, pa_logit):
    """Per ogni flusso segnalato, indica quale ramo ha fatto scattare l'allarme: 'consenso' (alta
    confidenza, concordanza dei due stadi) > 'logit' (caccia) > 'solo primo stadio'
    (rete di sicurezza, DAE-alto). Precedenza per confidenza decrescente. Ritorna array di stringhe (''
    per i flussi non segnalati). Sottoprodotto gratuito di fuse_predict."""
    z = to_z(to_u(s_dae, pa_logit), params["mu"], params["sd"])
    z = np.asarray(z, np.float64)
    and_flag = and_predict(z, params["tau_z"], params["theta_z"])
    union_flag = z[:, 1] > params["theta_union"]
    dae_hi_flag = z[:, 0] > params.get("tau_hi_z", float("inf"))
    out = np.full(len(z), "", dtype=object)
    out[dae_hi_flag] = "solo primo stadio"
    out[union_flag] = "logit"
    out[and_flag] = "consenso"
    return out


def fpr_of(flag_neg):
    return float(np.asarray(flag_neg).mean())


def recall_of(flag_pos):
    return float(np.asarray(flag_pos).mean())


def mcc_bal(fpr, recall, n_b=FUS_N_B, n_a=FUS_N_A):
    """MCC dai tassi (FPR, recall) riscalati alla popolazione di riferimento (leak-free)."""
    tp = recall * n_a
    fn = (1 - recall) * n_a
    fp = fpr * n_b
    tn = (1 - fpr) * n_b
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def build_fusion_modes(sdae_tr, lg_tr, sdae_sel, lg_sel):
    """Deriva i parametri di fusione per le due modalita' A/B dalla RICETTA a percentili
    (FUSION_MODES) sui benigni train/select LOCALI (tau_dae/tau_hi = percentili di sdae_sel).
    Ritorna {mode: params}. E' il claim di trasferibilita': la ricetta, non i numeri."""
    out = {}
    for name, spec in FUSION_MODES.items():
        tau_dae = float(np.percentile(sdae_sel, spec["tau_dae_pct"]))
        tau_hi = float(np.percentile(sdae_sel, spec["tau_hi_pct"]))
        out[name] = build_fusion(sdae_tr, lg_tr, sdae_sel, lg_sel, tau_dae,
                                 spec["fpr_target"], spec["alpha_and"], tau_hi=tau_hi)
    return out


# ============================================================================
# Interpretabilita' degli alert: residuo marginale, famiglie e ramo.
# ============================================================================
# La spiegazione usa lo spazio marginale, non quello sbiancato. Il residuo si calcola solo per gli
# alert.

_RED = "\033[31m"
_RESET = "\033[0m"


def _alert_tag(color):
    return f"{_RED}[ALERT]{_RESET}" if color else "[ALERT]"


def _flow_start_iso(row, ts_emissione):
    """Istante d'inizio del flusso (FLOW_START_MILLISECONDS, epoch in ms) in ISO 8601 UTC.

    Se il campo manca o non e' leggibile ripiega sull'ora di emissione e lo dichiara nella riga.
    """
    try:
        ms = int(float(row.get("FLOW_START_MILLISECONDS", "")))
        if ms <= 0:
            raise ValueError(ms)
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat(timespec="milliseconds")
    except (TypeError, ValueError, OverflowError, OSError):
        return f"{ts_emissione} (ora di emissione)"


def _alert_prefix(row, ts_iso, color=False):
    # Prefisso a larghezza fissa per allineare le feature; ts_iso e' l'istante gia' formattato.
    src = str(row.get("IPV4_SRC_ADDR", "?"))
    dst = str(row.get("IPV4_DST_ADDR", "?"))
    dport = str(row.get("L4_DST_PORT", "?"))
    proto = str(row.get("PROTOCOL", "?")).strip()
    proto_name = {"6": "TCP", "17": "UDP", "1": "ICMP", "58": "ICMP6"}.get(proto, f"P{proto}")
    dst_port = f"{dst}:{dport}"
    return f"{_alert_tag(color)} {ts_iso} {src:<15} -> {dst_port:<21} {proto_name:<4}"


def _feature_family(name):
    """Raggruppa le feature in famiglie semantiche per il triage degli alert."""
    if name.startswith("proto_"):
        return "protocollo"
    if name.startswith("port_"):
        return "porta"
    if name.startswith("l7_"):
        return "protocollo-L7"
    if name.startswith("dns_type_") or name == "is_dns_flow" or name == "DNS_TTL_ANSWER":
        return "DNS"
    if name.startswith("client_"):
        return "flag-TCP-client"
    if name.startswith("server_"):
        return "flag-TCP-server"
    if name.startswith("icmp_"):
        return "ICMP"
    if name.startswith("has_"):
        return "presenza-campo"
    if name in ("L4_SRC_PORT", "L4_DST_PORT"):
        return "porta"
    if "IAT" in name:
        return "inter-arrivo"
    if "THROUGHPUT" in name or "SECOND_BYTES" in name:
        return "throughput"
    if "DURATION" in name:
        return "durata"
    if "BYTES" in name or "PKTS" in name:
        return "volumi"
    if "TTL" in name:
        return "TTL"
    if "WIN" in name:
        return "finestra-TCP"
    if "PKT_LEN" in name or "FLOW_PKT" in name:
        return "dimensione-pacchetti"
    return "altro"


@timed
def emit_alerts(rows, anomaly, s_dae, logit_pa, X, Y_hat, w, params,
                output_handle, sigma_k=SIGMA_K, log_handle=None):
    """Emette un alert con istante d'inizio del flusso, ramo, feature sopra sigma_k e riepilogo per famiglia.

    In console mostra le tre feature principali; alert.log contiene la tabella completa, senza codici
    colore. Il residuo e' calcolato solo per gli alert e almeno una feature e' sempre riportata.
    """
    idx = np.where(anomaly)[0]
    if len(idx) == 0:
        return
    Z = ((Y_hat[idx] - X[idx]) - w["mu_R"]) / w["sigma_R"]
    branch = fired_branch(params, s_dae, logit_pa)
    ts_emissione = datetime.now().astimezone().isoformat(timespec="seconds")
    color = hasattr(output_handle, "isatty") and output_handle.isatty()
    feat_names = w["feature_order"]
    NAMEW, TOP_PRINT, BRW = 27, 3, 17
    out_lines, log_lines = [], []
    for k, i in enumerate(idx):
        ts_iso = _flow_start_iso(rows[i], ts_emissione)
        prefix = _alert_prefix(rows[i], ts_iso, color)
        prefix_log = _alert_prefix(rows[i], ts_iso, False)
        br = branch[i] if branch[i] else "?"
        z = Z[k]
        absz = np.abs(z)
        sel = np.where(absz > sigma_k)[0]
        if sel.size == 0:
            sel = np.array([int(np.argmax(absz))])
        sel = sel[np.argsort(absz[sel])[::-1]]
        fam_max = {}
        cols = []
        for j in sel:
            jj = int(j)
            val = float(z[jj])
            name = feat_names[jj] if jj < len(feat_names) else f"idx_{jj}"
            fam = _feature_family(name)
            fam_max[fam] = max(fam_max.get(fam, 0.0), abs(val))
            sign = "+" if val >= 0 else "-"
            cols.append(f"{name:<{NAMEW}}[{fam}] {sign}{abs(val):>5.2f}")
        fam_summary = "  ".join(f"{f}({m:.1f})" for f, m in
                                sorted(fam_max.items(), key=lambda kv: -kv[1]))
        out_lines.append(f"{prefix} [{br:<{BRW}}]  " + "   ".join(cols[:TOP_PRINT])
                         + f"   | famiglie: {fam_summary}")
        if log_handle is not None:
            log_lines.append(f"{prefix_log} [{br:<{BRW}}]  " + "   ".join(cols)
                            + f"   | famiglie: {fam_summary}")
    output_handle.write("\n".join(out_lines) + "\n")
    output_handle.flush()
    if log_handle is not None:
        _scrivi_alert_log(log_handle, "\n".join(log_lines) + "\n")


# Limita la crescita del log; errori di scrittura non interrompono l'inferenza.
ALERT_LOG_MAX_BYTES = 256 * 1024 * 1024
_ALERT_LOG_STATE = {"sospeso": False}


def _apri_alert_log(path):
    """Apre alert.log in append, ruotandolo in <path>.1 se supera il limite."""
    p = Path(path)
    try:
        if p.exists() and p.stat().st_size > ALERT_LOG_MAX_BYTES:
            p.replace(p.with_name(p.name + ".1"))
            print(f"[alert.log] oltre {ALERT_LOG_MAX_BYTES // (1024*1024)} MiB: "
                  f"ruotato in {p.name}.1", file=sys.stderr)
    except OSError as e:
        print(f"[alert.log] rotazione fallita ({e}): si continua in append", file=sys.stderr)
    _ALERT_LOG_STATE["sospeso"] = False
    return open(path, "a")


def _scrivi_alert_log(handle, text):
    if _ALERT_LOG_STATE["sospeso"]:
        return
    try:
        if handle.tell() > ALERT_LOG_MAX_BYTES:
            _ALERT_LOG_STATE["sospeso"] = True
            print(f"[alert.log] tetto di {ALERT_LOG_MAX_BYTES // (1024*1024)} MiB raggiunto "
                  f"durante il run: scritture sospese (la console continua)", file=sys.stderr)
            return
        handle.write(text)
        handle.flush()
    except OSError as e:
        _ALERT_LOG_STATE["sospeso"] = True
        print(f"[alert.log] errore di scrittura ({e}): log sospeso, l'inferenza continua",
              file=sys.stderr)


# ============================================================================
# Monitoraggio del tasso di alert in --live (finestra mobile, EMA e log con timestamp).
# ============================================================================
# In modalita' B, monitorare il tasso live: il FPR realizzato puo' differire da quello nominale.
# Finestra mobile ed EMA mostrano rispettivamente il livello e le variazioni recenti.


class AlertRateMonitor:
    """Traccia il tasso su finestra mobile ed EMA; il chiamante fornisce il tempo."""

    def __init__(self, window_s=300.0, ema_alpha=0.3):
        self.window_s = float(window_s)
        self.ema_alpha = float(ema_alpha)
        self._events = deque()
        self._flows = 0
        self._alerts = 0
        self.ema_rate = None

    def update(self, t, n_flows, n_alerts):
        """Registra un batch, rimuove quelli fuori finestra e ritorna tassi e totali."""
        self._events.append((float(t), int(n_flows), int(n_alerts)))
        self._flows += int(n_flows)
        self._alerts += int(n_alerts)
        cutoff = float(t) - self.window_s
        while self._events and self._events[0][0] < cutoff:
            _, f, a = self._events.popleft()
            self._flows -= f
            self._alerts -= a
        window_rate = (self._alerts / self._flows) if self._flows else 0.0
        batch_rate = (n_alerts / n_flows) if n_flows else 0.0
        if self.ema_rate is None:
            self.ema_rate = batch_rate
        else:
            self.ema_rate = (1.0 - self.ema_alpha) * self.ema_rate + self.ema_alpha * batch_rate
        return {"window_rate": window_rate, "ema_rate": self.ema_rate,
                "flows_window": self._flows, "alerts_window": self._alerts}

    def format_line(self, ts_iso, stats):
        return (f"[RATE] {ts_iso} tasso_alert finestra={stats['window_rate'] * 100:.3f}% "
                f"EMA={stats['ema_rate'] * 100:.3f}% "
                f"(alert={stats['alerts_window']}/{stats['flows_window']} flussi nella finestra)")


# ============================================================================
# Caricamento pesi e metadati del modello.
# ============================================================================


# Contratto dei file richiesti da ogni directory modello.
FILE_MODELLO = ("dae.npz", "pa.npz", "soglie.json", "feature_order.json",
                "scaler_params.json", "transform_meta.json")


def resolve_dataset(weights_dir, dataset_override=None):
    """Modello attivo: override CLI o modelli/active_dataset.txt (default 'cse')."""
    if dataset_override:
        return dataset_override
    ad = Path(weights_dir) / "active_dataset.txt"
    if ad.exists():
        name = ad.read_text(encoding="utf-8").strip()
        if name:
            return name
    return "cse"


def _soglie_to_params(s):
    """soglie.json[mode] -> params di fusione (inf per i rami spenti)."""
    return {"mu": np.asarray(s["mu"], np.float64), "sd": np.asarray(s["sd"], np.float64),
            "tau_z": s["tau_z"], "theta_z": s["theta_z"],
            "theta_union": (float("inf") if s.get("theta_union") is None else s["theta_union"]),
            "tau_hi_z": (float("inf") if s.get("tau_hi_z") is None else s["tau_hi_z"])}


def load_weights(weights_dir, dataset=None, mode="A"):
    """Carica pesi, metadati e soglie, cercando prima in produzione e poi sperimentali."""
    wd = Path(weights_dir)
    dataset = resolve_dataset(wd, dataset)
    ds = next((wd / sub / dataset for sub in ("produzione", "sperimentali")
               if (wd / sub / dataset).exists()), None)
    if ds is None:
        disp = sorted(d.name for sub in ("produzione", "sperimentali")
                      for d in (wd / sub).glob("*") if (d / "dae.npz").exists())
        raise FileNotFoundError(f"modello '{dataset}' non trovato in {wd}/produzione/ ne' in "
                                f"{wd}/sperimentali/. Modelli disponibili: "
                                f"{', '.join(disp) if disp else '(nessuno)'}. "
                                f"Esegui il retrain (train.py) o correggi "
                                f"modelli/active_dataset.txt.")
    mancanti = [f for f in FILE_MODELLO if not (ds / f).exists()]
    if mancanti:
        raise FileNotFoundError(f"modello '{dataset}' INCOMPLETO in {ds}: manca "
                                f"{', '.join(mancanti)} ({len(mancanti)} file su "
                                f"{len(FILE_MODELLO)}). Ricopiare il modello per intero "
                                f"o rifare il retrain (train.py).")
    w = {**dict(np.load(ds / "dae.npz")), **dict(np.load(ds / "pa.npz"))}
    compute_fold(w)
    fo = json.loads((ds / "feature_order.json").read_text())
    w["feature_order"] = fo["order"]
    w["scaler_params"] = json.loads((ds / "scaler_params.json").read_text())
    w["log_features"] = json.loads((ds / "transform_meta.json").read_text())["log_features"]
    soglie = json.loads((ds / "soglie.json").read_text())
    if mode not in soglie:
        raise KeyError(f"modalita' '{mode}' assente in soglie.json (disponibili: {list(soglie)})")
    w["fusion_params"] = _soglie_to_params(soglie[mode])
    # La baseline DAE-solo usa tau_dae@p99 della modalita' B, qualunque sia la modalita' attiva.
    if "B" in soglie:
        sb = soglie["B"]
        w["tau_dae_solo"] = 10.0 ** (float(sb["tau_z"]) * float(sb["sd"][0]) + float(sb["mu"][0]))
    w["_dataset"] = dataset
    w["_mode"] = mode
    return w


# ============================================================================
# I/O CSV + pipeline per-chunk (encoding -> forward -> fusione -> alert)
# ============================================================================


# --- righe malformate da collisione di delimitatore (serie per-secondo nProbe) --------------
# I due campi SRC_TO_DST_SECOND_BYTES / DST_TO_SRC_SECOND_BYTES sono, in nProbe, ARRAY (byte per
# secondo) esportati SENZA virgolette: ogni virgola interna diventa un separatore e sfalsa a
# destra tutto il resto della riga. Misurato sulle catture LAN: solo il 28 % circa delle righe di
# lan_test_raw.csv (87.825 su 313.115) ha arita' pari all'header (53). I CSV accademici (CSE/UNSW)
# sono al 100 % conformi e non passano mai di qui.
#
# La ricostruzione e' deterministica, non euristica: sulle righe BEN FORMATE della stessa cattura
# SRC_TO_DST_SECOND_BYTES == IN_BYTES nel 99,90 % dei casi e DST_TO_SRC_SECOND_BYTES == OUT_BYTES
# nel 99,93 %, e sulle righe malformate la somma cumulata della prima serie da' esattamente
# IN_BYTES e quella della seconda OUT_BYTES (verificato su arita' 54, 59, 63, 64). Lo scalare
# atteso e' dunque il totale direzionale, e le ancore IN_BYTES/OUT_BYTES PRECEDONO il punto di
# espansione: sono intatte anche nelle righe rotte.
#
# ATTENZIONE — la riscrittura e' STRETTAMENTE condizionata all'arita': nel dataset accademico
# quei campi sono un TASSO (~OUT/durata) e coincidono con IN_BYTES solo nello 0,01 %. Applicarla
# a una riga ben formata corromperebbe la feature su CSE/UNSW.
_MALFORMED_STATS = {"riparate": 0, "corte": 0, "non_ricostruibili": 0, "avvisato": False}


def _ripara_riga(campi, header, idx):
    """Riallinea i campi usando le ancore IN/OUT; ritorna None se non e' ricostruibile."""
    i1, i2, i_in, i_out, n_coda = idx
    if i_in is None or i_out is None:
        return None
    testa = campi[:i1]
    coda = campi[len(campi) - n_coda:] if n_coda else []
    if len(testa) != i1 or len(coda) != n_coda:
        return None
    return testa + [campi[i_in], campi[i_out]] + coda


def _flussi(f):
    """Legge righe nProbe, riallineando quelle lunghe e contando quelle malformate."""
    rd = csv.reader(f)
    try:
        header = next(rd)
    except StopIteration:
        return
    nh = len(header)
    try:
        i1 = header.index("SRC_TO_DST_SECOND_BYTES")
        i2 = header.index("DST_TO_SRC_SECOND_BYTES")
    except ValueError:
        i1 = i2 = -1
    riparabile = (i2 == i1 + 1 >= 1)     # i due campi adiacenti: vero su schema LAN e accademico
    idx = (i1, i2,
           header.index("IN_BYTES") if "IN_BYTES" in header else None,
           header.index("OUT_BYTES") if "OUT_BYTES" in header else None,
           nh - i2 - 1)
    for campi in rd:
        n = len(campi)
        if n == nh:
            yield dict(zip(header, campi))
            continue
        if n > nh and riparabile:
            riallineata = _ripara_riga(campi, header, idx)
            if riallineata is not None and len(riallineata) == nh:
                _MALFORMED_STATS["riparate"] += 1
                yield dict(zip(header, riallineata))
                continue
            _MALFORMED_STATS["non_ricostruibili"] += 1
        elif n < nh:
            _MALFORMED_STATS["corte"] += 1
        else:
            _MALFORMED_STATS["non_ricostruibili"] += 1
        if not _MALFORMED_STATS["avvisato"]:
            _MALFORMED_STATS["avvisato"] = True
            print(f"WARNING: riga con {n} campi contro i {nh} dell'intestazione. Le righe piu' "
                  f"lunghe si riallineano dalle ancore (serie per-secondo nProbe non quotate); "
                  f"le altre sono contate ed emesse: le corte coi soli campi presenti (i mancanti "
                  f"diventano 0 per coercizione), le lunghe con zip posizionale (sfalsato oltre la rottura).",
                  file=sys.stderr, flush=True)
        yield {k: v for k, v in zip(header, campi)}


def riepilogo_righe_malformate():
    """Conteggi delle righe non conformi incontrate; sui CSV accademici restano tutti a zero."""
    return dict(_MALFORMED_STATS)


def _open_csv(path):
    """Apre un CSV o .gz in modalita' testo, sostituendo i byte non validi."""
    import gzip
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", newline="", errors="replace")
    return open(path, newline="", errors="replace")


@timed_total
def _process_rows(rows, w, args, output_handle, log_handle=None):
    """Elabora un chunk e ritorna righe, conteggio alert e maschere di anomalia."""
    if not rows:
        return rows, 0, None, None
    X = csv_to_features(rows, w["feature_order"], w["scaler_params"], w["log_features"])
    z, Y_hat = forward_dae(X, w)
    s_dae, _s_pa, logit = compute_scores_fused(X, Y_hat, z, w)
    anomaly = apply_fusion(w["fusion_params"], s_dae, logit)
    n_alerts = int(anomaly.sum())
    # Baseline DAE-solo per --test, senza gating del PA.
    anomaly_dae = (s_dae > w["tau_dae_solo"]) if (getattr(args, "test", False) and "tau_dae_solo" in w) else None
    if not getattr(args, "test", False) and n_alerts:
        emit_alerts(rows, anomaly, s_dae, logit, X, Y_hat, w, w["fusion_params"],
                    output_handle, sigma_k=args.sigma_k, log_handle=log_handle)
    return rows, n_alerts, anomaly, anomaly_dae


# ============================================================================
# Metriche --test (numpy puro, no sklearn); MCC_bal a prevalenza di riferimento
# ============================================================================


def confusion_chunk(rows, anomaly):
    if "Label" not in rows[0]:
        print("ERRORE: --test richiede la colonna 'Label' nel CSV", file=sys.stderr)
        sys.exit(1)
    y_true = np.fromiter((_to_int(r.get("Label", "0")) for r in rows), dtype=np.int64, count=len(rows))
    tp = int(np.sum(anomaly & (y_true == 1)))
    fp = int(np.sum(anomaly & (y_true == 0)))
    fn = int(np.sum(~anomaly & (y_true == 1)))
    tn = int(np.sum(~anomaly & (y_true == 0)))
    per_class = {}
    if "Attack" in rows[0]:
        attack = np.array([r.get("Attack", "") for r in rows], dtype=object)
        mal = y_true == 1
        for c in {a for a, m in zip(attack, mal) if m}:
            mask = (attack == c) & mal
            acc = per_class.setdefault(c, [0, 0])
            acc[0] += int(mask.sum())
            acc[1] += int(np.sum(anomaly[mask]))
    return tp, fp, tn, fn, per_class


def print_test_metrics(tp, fp, tn, fn, per_class, label=None):
    """Stampa metriche osservate e MCC_bal alla prevalenza di riferimento; ritorna un dict."""
    tp_f, tn_f, fp_f, fn_f = float(tp), float(tn), float(fp), float(fn)
    total = tp + tn + fp + fn
    denom = math.sqrt((tp_f + fp_f) * (tp_f + fn_f) * (tn_f + fp_f) * (tn_f + fn_f))
    mcc = (tp_f * tn_f - fp_f * fn_f) / denom if denom > 0 else 0.0
    fpr = fp_f / (fp_f + tn_f) if (fp + tn) else 0.0
    recall = tp_f / (tp_f + fn_f) if (tp + fn) else 0.0
    prec = tp_f / (tp_f + fp_f) if (tp + fp) else 0.0
    mccb = mcc_bal(fpr, recall)                               # prevalenza di riferimento
    suffix = "" if label is None else f" [{label}]"
    print(f"\n==> Metriche --test{suffix} su {total:,} flussi etichettati", file=sys.stderr)
    print(f"    MCC={mcc:.4f}  MCC_bal(rif)={mccb:.4f}  FPR={fpr*100:.4f}%  "
          f"recall(DR)={recall:.4f}  precision={prec:.4f}", file=sys.stderr)
    if per_class:
        print("    DR per classe:", file=sys.stderr)
        for cls in sorted(per_class):
            n, det = per_class[cls]
            print(f"      {cls:<28} {det}/{n} = {det/max(n,1):.4f}", file=sys.stderr)
    return {"mcc": mcc, "mcc_bal": mccb, "fpr": fpr, "recall": recall, "precision": prec,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ============================================================================
# run_batch — batch su CSV, anche per --test (monoprocesso; parallelismo solo BLAS)
# ============================================================================


def _iter_chunks(path, chunk_size):
    """Legge il CSV in streaming, mantenendo un solo chunk in memoria."""
    with _open_csv(path) as f:
        chunk = []
        for row in _flussi(f):
            chunk.append(row)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def _iter_windows(path, target):
    """Genera finestre di dimensione variabile fino a `target` flussi."""
    sizes = NPROBE_WINDOW_SAMPLE
    f = _open_csv(path)
    rd = _flussi(f)
    si = emitted = 0
    try:
        while emitted < target:
            want = min(sizes[si % len(sizes)], target - emitted)
            si += 1
            batch = []
            while len(batch) < want:
                row = next(rd, None)
                if row is None:
                    f.close()
                    f = _open_csv(path)
                    rd = _flussi(f)
                    row = next(rd, None)
                    if row is None:
                        return
                batch.append(row)
            emitted += len(batch)
            yield batch
    finally:
        f.close()


def run_batch(args, w):
    """Elabora un CSV in streaming; --test aggrega metriche senza emettere alert."""
    log_handle = None if args.test else _apri_alert_log(args.alert_log)
    agg = {"total": 0, "n_alerts": 0, "net_bytes": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0, "per_class": {}}
    agg_dae = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "per_class": {}}

    def handle(chunk, out_h, log_h):
        kept, n_a, anomaly, anomaly_dae = _process_rows(chunk, w, args, out_h, log_h)
        if anomaly is None:
            return
        conf = confusion_chunk(kept, anomaly) if args.test else None
        conf_dae = confusion_chunk(kept, anomaly_dae) if (args.test and anomaly_dae is not None) else None
        nb = sum(_to_int(r.get("IN_BYTES", 0)) + _to_int(r.get("OUT_BYTES", 0)) for r in kept)
        agg["total"] += len(kept)
        agg["n_alerts"] += n_a
        agg["net_bytes"] += nb
        if conf:
            c_tp, c_fp, c_tn, c_fn, c_pc = conf
            agg["tp"] += c_tp
            agg["fp"] += c_fp
            agg["tn"] += c_tn
            agg["fn"] += c_fn
            for cls, (n, det) in c_pc.items():
                a = agg["per_class"].setdefault(cls, [0, 0])
                a[0] += n; a[1] += det
        if conf_dae:
            d_tp, d_fp, d_tn, d_fn, d_pc = conf_dae
            agg_dae["tp"] += d_tp
            agg_dae["fp"] += d_fp
            agg_dae["tn"] += d_tn
            agg_dae["fn"] += d_fn
            for cls, (n, det) in d_pc.items():
                a = agg_dae["per_class"].setdefault(cls, [0, 0])
                a[0] += n; a[1] += det

    # Finestre variabili per --bench-target, chunk fissi altrimenti.
    if getattr(args, "bench_target", 0):
        src = iter(_iter_windows(args.batch, args.bench_target))
    else:
        src = iter(_iter_chunks(args.batch, args.chunk_size))
    # Con almeno due chunk il primo (warm-up) esce da tempi e aggregati della fusione; gli
    # aggregati DAE-solo di --test lo includono ancora.
    if getattr(args, "bench", False):
        first = next(src, None)
        second = next(src, None)
        if first is not None and second is not None:
            handle(first, sys.stdout, log_handle)
            _TIMINGS.clear()
            _BENCH_NET["bytes"] = 0; _BENCH_NET["flows"] = 0
            for k in ("total", "n_alerts", "net_bytes", "tp", "fp", "tn", "fn"):
                agg[k] = 0
            agg["per_class"].clear()
            src = itertools.chain([second], src)
        elif first is not None:
            src = iter([first])
        else:
            src = iter([])
    t0 = time.perf_counter()
    for chunk in src:
        handle(chunk, sys.stdout, log_handle)
    wall = time.perf_counter() - t0
    if log_handle:
        log_handle.close()

    pct = agg["n_alerts"] / max(agg["total"], 1) * 100
    thr = agg["total"] / wall if wall > 0 else 0.0
    # Throughput di rete calcolato dai byte effettivi del dataset.
    gbps = agg["net_bytes"] * 8 / 1e9 / wall if wall > 0 else 0.0
    avg_b = agg["net_bytes"] / max(agg["total"], 1)
    print(f"==> {agg['total']:,} flussi, {agg['n_alerts']:,} alert ({pct:.3f}%) "
          f"[modello={w['_dataset']} modalita'={w['_mode']}] -> {thr:,.0f} flussi/s "
          f"= {gbps:.3f} Gbit/s (flusso medio {avg_b:,.0f} B, IN+OUT)",
          file=sys.stderr)
    if _COERCE_STATS["garbage_a_zero"]:
        print(f"WARNING: {_COERCE_STATS['garbage_a_zero']:,} campi non numerici coercizzati "
              f"a 0.0 durante il run (dettagli: contatore _COERCE_STATS).", file=sys.stderr)
    if getattr(args, "bench", False):
        # Etichetta stabile per aggregare le ripetizioni, indipendentemente dal target.
        ds = args.bench_dataset or w["_dataset"]
        payload = bench_payload(w, agg["total"], wall, args.bench_arch, ds)
        out_p = Path(args.bench_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2) + "\n")
        steps = [s for s in ("csv_to_features", "forward_dae", "compute_scores_fused",
                             "apply_fusion", "emit_alerts") if _TIMINGS[s]]
        tot = sum(sum(_TIMINGS[s]) for s in steps)
        print(f"--> [bench] {out_p} — breakdown us/flusso (strumentato, NON il throughput di deploy):",
              file=sys.stderr)
        for s in steps + (["total"] if _TIMINGS["total"] else []):
            v = sum(_TIMINGS[s]) / len(_TIMINGS[s])
            quota = f"{sum(_TIMINGS[s]) / tot * 100:5.1f}%" if (tot > 0 and s != "total") else "    —"
            print(f"      {s:<22} {v:8.2f} us/flusso   {quota}", file=sys.stderr)
    if args.test:
        res = print_test_metrics(agg["tp"], agg["fp"], agg["tn"], agg["fn"], agg["per_class"])
        if any(agg_dae[k] for k in ("tp", "fp", "tn", "fn")):
            print_test_metrics(agg_dae["tp"], agg_dae["fp"], agg_dae["tn"], agg_dae["fn"],
                               agg_dae["per_class"], label="DAE-solo (tau_dae@p99, Mod B)")
        return res
    return None


# ============================================================================
# run_live — cattura nProbe continua (sniffing/cattura.sh) + monitoraggio tasso di alert
# ============================================================================


def _detect_default_iface():
    try:
        with open("/proc/net/route") as f:
            next(f)
            for line in f:
                p = line.split()
                if len(p) > 3 and p[1] == "00000000" and (int(p[3], 16) & 0x2):
                    return p[0]
    except Exception:
        pass
    return "eth0"


def run_live(args, w):
    """Legge i CSV della cattura, emette alert e monitora il tasso live."""
    base = Path(__file__).resolve().parent
    csvdir = base / "sniffing" / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    iface = args.iface or _detect_default_iface()
    seen = set()
    monitor = AlertRateMonitor()
    log_handle = _apri_alert_log(args.alert_log)

    env = dict(os.environ, NIDS_IFACE=iface, NIDS_BASEDIR=str(base / "sniffing"))
    cattura = base / "sniffing" / "cattura.sh"
    proc = subprocess.Popen(["bash", str(cattura), "--live"], env=env,
                            preexec_fn=os.setsid) if cattura.exists() else None
    print(f"==> LIVE su iface={iface} (modello={w['_dataset']} modalita'={w['_mode']}). "
          f"Ctrl-C per fermare.", file=sys.stderr)
    # Tratta SIGTERM come Ctrl-C per arrestare anche il processo di cattura.
    def _sigterm(_sig, _frm):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _sigterm)
    try:
        while True:
            nuovi = [f for f in sorted(csvdir.glob("*.csv")) if f not in seen]
            for f in nuovi:
                try:
                    with _open_csv(f) as fh:
                        rows = list(_flussi(fh))
                    kept, n_a, _, _ = _process_rows(rows, w, args, sys.stdout, log_handle)
                    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
                    stats = monitor.update(time.monotonic(), len(kept), n_a)
                    print(monitor.format_line(ts, stats), file=sys.stderr)
                    f.unlink(missing_ok=True)
                    # Un file diventa visto solo dopo il processing riuscito.
                    seen.add(f)
                except Exception as e:
                    print(f"[live] errore su {f.name}: {e}", file=sys.stderr)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n==> arresto...", file=sys.stderr)
    finally:
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
        log_handle.close()
        if _COERCE_STATS["garbage_a_zero"]:
            print(f"WARNING: {_COERCE_STATS['garbage_a_zero']:,} campi non numerici "
                  f"coercizzati a 0.0 durante la sessione live.", file=sys.stderr)


# ============================================================================
# CLI
# ============================================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Inferenza NIDS a 2 stadi (DAE #569 + PA #589, fusione a tre rami) su CSV nProbe.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="cattura live da nProbe (default)")
    mode.add_argument("--batch", metavar="FILE", help="inferenza su un CSV nProbe esistente")
    mode.add_argument("--test", action="store_true",
                      help="validazione: metriche (MCC/FPR/DR) su un test set etichettato "
                           "(dati/inferenza/<model>_test_raw.csv); nessun alert.")
    p.add_argument("--mode", choices=["A", "B"], default="A",
                   help="modalita' operativa: A = FPR minimo (pochi allarmi, alta precisione); "
                        "B = alta recall (piu' copertura, monitorare il tasso di alert). Default A.")
    p.add_argument("--model", default=None, help="modello attivo (default: modelli/active_dataset.txt o 'cse')")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help=f"righe/chunk (default {CHUNK_SIZE})")
    p.add_argument("--iface", default=None, help="interfaccia di cattura in --live (default: auto)")
    p.add_argument("--sigma-k", type=float, default=SIGMA_K,
                   help=f"soglia in sigma delle feature riportate nell'alert (default {SIGMA_K})")
    p.add_argument("--alert-log", metavar="PATH", default=str(WEIGHTS_DIR.parent / "alert.log"),
                   help="file su cui appendere la tabella completa degli alert (mai in --test)")
    p.add_argument("--bench", action="store_true",
                   help="DIAGNOSTICA (WP-10): accende il breakdown per-step delle latenze e scrive "
                        "il JSON per plot_bench.py. La run e' STRUMENTATA, quindi piu' lenta: il "
                        "throughput di deploy si misura senza questo flag.")
    p.add_argument("--bench-out", metavar="PATH", default=None,
                   help="JSON del benchmark (obbligatorio con --bench; nessun default, per non "
                        "lasciare scorie dentro il pacchetto)")
    p.add_argument("--bench-arch", metavar="NOME", default=None,
                   help="etichetta dell'architettura nel JSON (es. 'Ryzen 5800X3D (workstation)')")
    p.add_argument("--bench-dataset", metavar="NOME", default=None,
                   help="etichetta del dataset nel JSON (default: il nome del modello)")
    p.add_argument("--bench-target", type=int, default=0, metavar="N",
                   help="bench a micro-batch VARIABILE: processa N flussi in finestre di taglia presa "
                        "da NPROBE_WINDOW_SAMPLE (simula il live), ciclando il CSV base. 0 = chunk fissi "
                        "a --chunk-size. Richiede --bench e --batch <csv-base>.")
    args = p.parse_args(argv)
    if args.test:
        test_model = args.model or "unsw"
        test_csv = WEIGHTS_DIR.parent / "dati" / "inferenza" / f"{test_model}_test_raw.csv"
        if not test_csv.exists():
            p.error(f"test set per '{test_model}' non disponibile: {test_csv} mancante.")
        args.batch = str(test_csv)
        args.model = args.model or test_model
    if args.batch and not Path(args.batch).is_file():
        p.error(f"file batch non trovato: {args.batch}")
    if not args.live and not args.batch:
        args.live = True
    # Validare dopo aver risolto la modalita' predefinita, che puo' impostare args.live.
    if args.bench:
        if args.live:
            p.error("--bench richiede --batch o --test: in --live la cattura non ha fine, quindi "
                    "non esiste un wall su cui calcolare il throughput.")
        if not args.bench_out:
            p.error("--bench richiede --bench-out PATH (dove scrivere il JSON del benchmark): "
                    "nessun default, per non lasciare scorie dentro il pacchetto.")
        if not args.bench_arch:
            p.error("--bench richiede --bench-arch NOME (etichetta dell'architettura misurata: "
                    "il confronto multi-architettura del cap4 raggruppa le run per quella).")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Gli errori di selezione del modello sono errori d'uso: mostrare un messaggio senza traceback.
    try:
        w = load_weights(WEIGHTS_DIR, args.model, mode=args.mode)
    except (FileNotFoundError, KeyError) as e:
        print(f"ERRORE: {e.args[0] if e.args else e}", file=sys.stderr)
        sys.exit(2)
    if args.live:
        run_live(args, w)
    else:
        run_batch(args, w)


if __name__ == "__main__":
    main()
