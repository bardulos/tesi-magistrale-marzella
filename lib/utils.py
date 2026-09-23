"""lib/utils.py — fonte unica di costanti di decisione + helper cross-cutting.

Tiene insieme:
  * le costanti di decisione della pipeline preprocessing (stage encode/reduce/transform);
  * le dimensioni canoniche dello spazio feature (34 continue + 57 binarie = 91) e la
    definizione UNICA del punteggio di anomalia del DAE, su cui poggia l'intero gate CSE;
  * gli helper d'ambiente (`setup_blas_env`) e di logging, importabili senza trascinare
    TensorFlow ne' matplotlib.

Le soglie delle otto analisi di audit (MI/RF/Wasserstein/varianza/costo informativo) sono qui,
nella sezione
"Stage descrittivi cap1": sono parametri di decisione, non stato di calcolo. I moduli che le
consumano vivono altrove.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


def setup_blas_env(n_threads: int, deterministic: bool = False,
                   gpu_growth: bool = False) -> None:
    """Imposta in un punto solo il threading BLAS e i flag di runtime TF.

    Va invocata **prima** di importare numpy/TensorFlow. `n_threads` fissa
    OMP/OPENBLAS/MKL (1 = determinismo bit-exact su CPU, convenzione dello spine,
    incluso loao/scoring). `deterministic` attiva `TF_DETERMINISTIC_OPS`;
    `gpu_growth` attiva `TF_FORCE_GPU_ALLOW_GROWTH` (fallback robusto letto da TF
    al primo init, indipendente dall'ordine Python).
    """
    n = str(int(n_threads))
    os.environ["OMP_NUM_THREADS"] = n
    os.environ["OPENBLAS_NUM_THREADS"] = n
    os.environ["MKL_NUM_THREADS"] = n
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    if gpu_growth:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# ============================================================================
# Stage encode
# ============================================================================

APRIORI_DROPS_NUMERIC = [
    "FTP_COMMAND_RET_CODE",
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
    "DNS_QUERY_ID",
]

L4_PORT_BUCKETS = [
    ("port_dns", [53]),
    ("port_rdp", [3389]),
    ("port_https", [443, 8443]),
    ("port_http", [80, 8080, 8000, 8888]),
    ("port_smb", [445]),
    ("port_ftp", [20, 21]),
    ("port_ssh", [22]),
    ("port_smtp", [25, 465, 587]),
]

TCP_FLAG_BITS = [
    ("fin", 0), ("syn", 1), ("rst", 2), ("psh", 3),
    ("ack", 4), ("urg", 5), ("ece", 6), ("cwr", 7),
]

# Sostituzioni SECOND_BYTES per NaN/Inf nProbe (duration=0 -> div per zero).
# Valori = max finito su CSE training. Su CSE 0 sostituzioni; su UNSW NaN+Inf normalizzati.
SECOND_BYTES_INF_REPLACEMENT = {
    "SRC_TO_DST_SECOND_BYTES": 46938,
    "DST_TO_SRC_SECOND_BYTES": 2015,
}

# ============================================================================
# Stage reduce (decisioni post-audit codificate)
# ============================================================================

AUDIT_DROP_FEATURES = [
    "MAX_IP_PKT_LEN",
    "RETRANSMITTED_OUT_PKTS",
    "RETRANSMITTED_IN_PKTS",
    "MIN_TTL",
    "TCP_WIN_MAX_IN",
    "TCP_FLAGS",
]

# Coppie (continua, flag binaria companion) prodotte dalla reduce.
# Le continue restano nel parquet; le flag sono nuove (uint8 da feature > 0).
SPARSE_SPLIT_FLAGS = {
    "MAX_TTL": "has_ttl",
    "TCP_WIN_MAX_OUT": "has_tcp_win_out",
}

# ============================================================================
# Stage descrittivi cap1 (inspect / stats / audit) — portati da repo_v3.
# Esplorativi: alimentano i NUMERI/tabelle del capitolo 1, non sono artefatti gated.
# ============================================================================

EXPECTED_COLS = 55                                    # colonne attese nel CSV grezzo NF-v3
SKIP_COLS = {"Label", "Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"}

# Continue con flag-companion già presente (solo per la 'nota' della var-decomposition).
EXISTING_SPARSE_FLAGS = dict(SPARSE_SPLIT_FLAGS)

# Soglie/parametri delle 7 analisi di audit (identici all'oracolo repo_v3).
SPEARMAN_FF_THRESHOLD = 0.90
SPEARMAN_FL_THRESHOLD = 0.10
SPEARMAN_FF_REDUNDANT_THRESHOLD = 0.98
MI_TOP_N = 30
RF_TOP_N = 30
RF_N_SAMPLE = 200_000
RF_N_FOLDS = 5
WASSERSTEIN_TOP_N = 20
WASSERSTEIN_SAMPLE_SIZE = 500_000
VARIANCE_RATIO_THRESHOLD = 0.5

# Ottava analisi (costo informativo): MI di una feature-bersaglio contro TUTTE le altre, sugli
# stessi istogrammi discretizzati delle analisi 3 e 7. Serve alle rimozioni prive di gemello:
# delle 6 AUDIT_DROP_FEATURES, quattro hanno una coppia |rho|>=0.98 che ne quantifica gia' il
# costo (analisi 7) e TCP_FLAGS ha i bit client_*/server_* estratti da encode_tcp_flags;
# TCP_WIN_MAX_IN e' l'unica senza gemello ne' decomposizione. Lista estendibile.
INFORMATION_COST_TARGETS = ["TCP_WIN_MAX_IN"]
INFORMATION_COST_N_BINS = 50
INFORMATION_COST_TOP_N = 20

# ============================================================================
# Stage transform
# ============================================================================

SKEW_THRESHOLD = 5.0
TRUNCATE_RANGE = (-10.0, 10.0)

# Split semi-supervisionato: benigni 80/10/10, malevoli 0/50/50 (stratify Attack)
BENIGN_FRACS = (0.80, 0.10, 0.10)
MALICIOUS_FRACS = (0.00, 0.50, 0.50)

# ============================================================================
# Spazio feature canonico + definizione punteggio di anomalia (fonte unica)
# ============================================================================
# Spazio feature canonico (91): ICMP a 7 bucket one-hot gated (derive_icmp); la colonna
# is_icmp_flow non e' emessa e non esistono continue ICMP decoded -> 34 continue + 57
# binarie = 91 (ordine: continue, poi binarie -> feature_order.json). Le binarie occupano
# [34:91] e i 7 bucket ICMP stanno in coda al blocco binario.
# N_BINARY, BIN_SLICE (slice di scoring) e SCORE_DEFINITION (sotto) servono al punteggio del DAE,
# NON al preprocessing. VERIFICATI su #569 (DAE ricostruito su AUC-PR, seme produzione 2034): lo
# score MSE-sulle-57-binarie @p99-benigni-val è quello effettivamente usato dal LOAO e dalla
# selezione. N_CONTINUOUS/N_BINARY/N_FEATURES restano il feature-space del preprocessing.

N_CONTINUOUS = 34
N_BINARY = 57
N_FEATURES = N_CONTINUOUS + N_BINARY
BIN_SLICE = slice(N_CONTINUOUS, N_FEATURES)

# DEFINIZIONE DEL PUNTEGGIO DI ANOMALIA DEL DAE (verificato su #569, consumato da loao/selezione):
#   score = mean((X_bin - Yhat_bin)^2)  in float32, sulle SOLE 57 feature binarie [34:91].
#   NIENTE componente BCE (la BCE e' solo loss di training).
#   Residui  R = Yhat - X  su tutte le 91 feature (convenzione R = Yhat - X).
SCORE_DEFINITION = (
    "MSE float32 sulle 57 binarie [34:91]; residui R=Yhat-X su tutte le 91; nessuna BCE"
)


def setup_logging(log_file: Path | None = None) -> None:
    """Tee logging su stream + (opzionale) file. Idempotente."""
    fmt = logging.Formatter("%(message)s")
    root = logging.getLogger()
    for h in root.handlers[:]:
        h.close()
        root.removeHandler(h)
    root.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
