#!/bin/bash
# inference/sniffing/cattura.sh
#
# Cattura continua del traffico di rete con nProbe (demo), per l'inferenza in tempo reale e per la
# raccolta di traffico benigno di training.
#
# INTERFACCIA DI RETE (ATTENZIONE):
#   In --live infer.py passa NIDS_IFACE (da --iface, oppure auto-detect della ROTTA DI DEFAULT). Per una
#   cattura via PORT MIRRORING la NIC corretta e' quella collegata alla monitor-port dello switch, che e'
#   PASSIVA e NON ha rotta di default: in quel caso l'auto-detect prende la NIC sbagliata (di management) e
#   non vedrebbe il traffico specchiato. Specificare allora l'interfaccia esplicitamente:
#       infer.py --live --iface <nic_mirror>     (o impostarla nelle Opzioni del menu / NIDS_IFACE).
#
# LIMITI DI nProbe VERSIONE DEMO:
# In nProbe v11 versione demo sono permessi 5000 flussi OPPURE 300s di sniffing,
# Il timeout di 310s e' il fallback di sicurezza. Le cartelle di cattura stanno in inference/sniffing/
# (staging/ + csv/), assegnate a nprobe:ntop e group-writable da install.sh: ci scrive sia nProbe in
# training (come nprobe via sudo) sia in --live (come utente corrente, che deve essere nel gruppo ntop).
#
# Uso standalone:
#   live:      NIDS_IFACE=enp8s0 bash cattura.sh --live
#   training:  sudo bash cattura.sh            (Ctrl+C: chiusura pulita via trap)

set -u

# ======================== CONFIGURAZIONE ========================
LIVE=0
for _arg in "$@"; do
    [ "$_arg" = "--live" ] && LIVE=1
done

# Interfaccia: env NIDS_IFACE (impostata da infer.py) -> altrimenti NIC della rotta di default -> eth0.
# (cfr. avviso PORT MIRRORING nell'header: l'auto-detect prende la rotta di default, non la monitor-port.)
_default_iface() { ip -o route show default 2>/dev/null | awk '{print $5; exit}'; }
IFACE="${NIDS_IFACE:-$(_default_iface)}"; IFACE="${IFACE:-eth0}"
# Base di lavoro: env NIDS_BASEDIR (impostata da infer.py = inference/sniffing) -> altrimenti la cartella
# di QUESTO script (inference/sniffing). Sottocartelle: staging/ (.flows di nProbe) e csv/ (CSV finalizzati,
# letti da infer.py); create e assegnate a nprobe:ntop da install.sh.
BASEDIR="${NIDS_BASEDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
STAGINGDIR="$BASEDIR/staging"
CSVDIR="$BASEDIR/csv"
LOGFILE="$BASEDIR/log/cattura.log"
NPROBE_LOG="$BASEDIR/log/nprobe.log"
TIMEOUT_SEC=310  # 10s di margine oltre il limite demo (300s)

# Crea le directory di lavoro PRIMA di qualsiasi scrittura (log/csv non esistono
# al primo avvio su macchina pulita -> truncate LOGFILE e mv dei .flows fallivano).
mkdir -p "$CSVDIR" "$(dirname "$LOGFILE")"

# 53 feature di NF-UNSW-NB15-v3 / NF-CSE-CIC-IDS2018-v3 (senza Label e Attack, presenti solo nei dataset
# etichettati). IDENTICO al template della cattura di training: le colonne e il loro ordine sono il
# contratto con il preprocessing di infer.py -> NON modificare.
TEMPLATE="%IPV4_SRC_ADDR %IPV4_DST_ADDR %L4_SRC_PORT %L4_DST_PORT \
%PROTOCOL %L7_PROTO %IN_BYTES %OUT_BYTES %IN_PKTS %OUT_PKTS \
%FLOW_DURATION_MILLISECONDS %TCP_FLAGS %CLIENT_TCP_FLAGS %SERVER_TCP_FLAGS \
%DURATION_IN %DURATION_OUT %MIN_TTL %MAX_TTL %LONGEST_FLOW_PKT \
%SHORTEST_FLOW_PKT %MIN_IP_PKT_LEN %MAX_IP_PKT_LEN \
%SRC_TO_DST_SECOND_BYTES %DST_TO_SRC_SECOND_BYTES \
%RETRANSMITTED_IN_BYTES %RETRANSMITTED_IN_PKTS \
%RETRANSMITTED_OUT_BYTES %RETRANSMITTED_OUT_PKTS \
%SRC_TO_DST_AVG_THROUGHPUT %DST_TO_SRC_AVG_THROUGHPUT \
%NUM_PKTS_UP_TO_128_BYTES %NUM_PKTS_128_TO_256_BYTES \
%NUM_PKTS_256_TO_512_BYTES %NUM_PKTS_512_TO_1024_BYTES \
%NUM_PKTS_1024_TO_1514_BYTES %TCP_WIN_MAX_IN %TCP_WIN_MAX_OUT \
%ICMP_TYPE %ICMP_IPV4_TYPE %DNS_QUERY_ID %DNS_QUERY_TYPE \
%DNS_TTL_ANSWER %FTP_COMMAND_RET_CODE \
%FLOW_START_MILLISECONDS %FLOW_END_MILLISECONDS \
%SRC_TO_DST_IAT_MIN %SRC_TO_DST_IAT_MAX \
%SRC_TO_DST_IAT_AVG %SRC_TO_DST_IAT_STDDEV \
%DST_TO_SRC_IAT_MIN %DST_TO_SRC_IAT_MAX \
%DST_TO_SRC_IAT_AVG %DST_TO_SRC_IAT_STDDEV"

# variabili di stato per il trap
NPROBE_PID=""
WATCHER_PID=""

# ======================== FUNZIONI ========================

log() {
    echo "$1" | tee -a "$LOGFILE"
}

cleanup() {
    log "  ricevuto segnale di terminazione, chiusura pulita…"
    [ -n "$NPROBE_PID" ] && kill "$NPROBE_PID" 2>/dev/null
    [ -n "$WATCHER_PID" ] && kill "$WATCHER_PID" 2>/dev/null
    sleep 1
    sposta_csv
    log "=== Fine cattura — $(date '+%A %d %B %Y, %H:%M:%S') ==="
    exit 0
}
trap cleanup INT TERM

verifica_staging() {
    if [ ! -d "$STAGINGDIR" ]; then
        mkdir -p "$STAGINGDIR"
    fi

    [ "$LIVE" = "1" ] && return 0
    owner=$(stat -c '%U:%G' "$STAGINGDIR")
    if [ "$owner" != "nprobe:ntop" ]; then
        log "  AVVISO: $STAGINGDIR ha owner '$owner' invece di nprobe:ntop — correzione"
        chown nprobe:ntop "$STAGINGDIR"
    fi
}

sposta_csv() {
    # Sposta i .flows finalizzati in $CSVDIR e rinominali in .csv
    # con il path relativo appiattito (per leggibilità in ls).
    find "$STAGINGDIR" -name "*.flows" -type f 2>/dev/null | while read FILE; do
        RELPATH=$(realpath --relative-to="$STAGINGDIR" "$FILE")
        CSVNAME=$(echo "$RELPATH" | tr '/' '_' | sed 's/\.flows$/.csv/')
        mv "$FILE" "$CSVDIR/$CSVNAME"
        SIZE=$(stat --format=%s "$CSVDIR/$CSVNAME" 2>/dev/null || echo '?')
        log "  CSV: $CSVNAME (${SIZE} byte)"
    done
    find "$STAGINGDIR" -mindepth 1 -type d -empty -delete 2>/dev/null
}

# ======================== CICLO PRINCIPALE ========================

> "$LOGFILE"
log "=== Avvio cattura — $(date '+%A %d %B %Y, %H:%M:%S') ==="
log "  interfaccia=$IFACE  staging=$STAGINGDIR  destinazione=$CSVDIR"

verifica_staging

CICLO=0
FAIL_STREAK=0   # cicli consecutivi in cui nProbe esce subito senza catturare (licenza/permessi/interfaccia)
while true; do
    CICLO=$((CICLO + 1))
    T_START=$(date +%s)
    log "Ciclo $CICLO — $(date '+%d/%m/%Y %H:%M:%S')"

    verifica_staging

    # nProbe in background — output verboso nel log separato
    timeout "$TIMEOUT_SEC" nprobe -i "$IFACE" \
        -V 10 \
        --csv-separator ',' \
        -T "$TEMPLATE" \
        -P "$STAGINGDIR" \
        -b "ip" \
        >> "$NPROBE_LOG" 2>&1 &
    NPROBE_PID=$!

    log "  PID $NPROBE_PID avviato (timeout=${TIMEOUT_SEC}s)"

    # watcher: uccide nProbe non appena nel log compare "max demo"
    # (limite 5000 flussi raggiunto)
    (
        tail -f --pid="$NPROBE_PID" "$NPROBE_LOG" 2>/dev/null \
            | grep -m 1 -q "max demo"
        kill "$NPROBE_PID" 2>/dev/null
    ) &
    WATCHER_PID=$!

    # attende che nProbe esca (per uno qualunque dei tre motivi:
    #   1. limite 5000 flussi → ucciso dal watcher
    #   2. limite 300s        → uscita naturale di nProbe demo
    #   3. timeout 310s       → ucciso da `timeout`)
    wait "$NPROBE_PID" 2>/dev/null
    DURATA=$(( $(date +%s) - T_START ))

    # pulizia del watcher
    kill "$WATCHER_PID" 2>/dev/null
    wait "$WATCHER_PID" 2>/dev/null

    # determina motivo di uscita e numero flussi dall'output di nProbe
    NFLOWS=$(tail -50 "$NPROBE_LOG" | grep -oP 'Total dumped to file:\s+\[\K\d+' | tail -1 || true)
    NFLOWS=${NFLOWS:-?}

    if tail -50 "$NPROBE_LOG" | grep -q "max demo"; then
        log "  Fine ciclo: limite flussi demo ($NFLOWS flussi in ${DURATA}s)"
    elif [ "$DURATA" -ge "$TIMEOUT_SEC" ]; then
        log "  Fine ciclo: timeout di sicurezza ($NFLOWS flussi in ${DURATA}s)"
    else
        log "  Fine ciclo: limite tempo nProbe ($NFLOWS flussi in ${DURATA}s)"
    fi

    # Fail-fast: se nProbe esce in <5s senza catturare (NFLOWS assente/0 e niente "max demo") per piu' cicli
    # di fila e' una causa bloccante (licenza mancante, setcap assente, interfaccia errata): NON resta in loop
    # "muto". Segnala la causa dal log di nProbe ed esce con segnale di errore, cosi' il supervisore di infer.py la rileva.
    if [ "$DURATA" -lt 5 ] && ! tail -50 "$NPROBE_LOG" | grep -q "max demo" \
       && { [ "$NFLOWS" = "?" ] || [ "$NFLOWS" = "0" ]; }; then
        FAIL_STREAK=$((FAIL_STREAK + 1))
    else
        FAIL_STREAK=0
    fi
    if [ "$FAIL_STREAK" -ge 3 ]; then
        log "  ERRORE: nProbe esce immediatamente da ${FAIL_STREAK} cicli senza catturare nulla."
        log "  Ultime righe di nProbe ($NPROBE_LOG):"
        tail -3 "$NPROBE_LOG" 2>/dev/null | sed 's/^/      /' | tee -a "$LOGFILE"
        log "  Verificare:  licenza nProbe (/etc/nprobe.license)  |  capabilities (sudo setcap"
        log "               cap_net_raw,cap_net_admin+eip $(command -v nprobe))  |  interfaccia '$IFACE'."
        exit 3
    fi

    NPROBE_PID=""
    WATCHER_PID=""

    # spostamento CSV in background (avvia immediatamente il ciclo successivo)
    sposta_csv &

    sleep 1
done
