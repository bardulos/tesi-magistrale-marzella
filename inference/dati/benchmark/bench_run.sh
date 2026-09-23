#!/usr/bin/env bash
# bench_run.sh — misura STANDARDIZZATA del throughput di inferenza che SIMULA il live: micro-batch a
# dimensione VARIABILE (distribuzione empirica delle finestre nProbe reali, infer.NPROBE_WINDOW_SAMPLE,
# seedata e IDENTICA su ogni macchina), ciclando il CSV base fino a N flussi. Inferenza MONOPROCESSO:
# il parallelismo e' interno al BLAS. NON tocca il runtime: compone infer.py --bench (che produce il
# JSON per plot_bench.py) e /usr/bin/time -v (che cattura il RSS di picco). Vedi BENCH_README.txt.
#
# Uso:   bash bench_run.sh <etichetta-arch> [k-flussi] [dir-di-raccolta]
#   <etichetta-arch>   nome macchina concordato, es. "kub-zen3" o "$(hostname)-$(arch)".
#   [k-flussi]         flussi da processare, in MIGLIAIA (default 100 -> 100.000). Tipici: 50/100/200/500.
#   [dir-di-raccolta]  dove finiscono i JSON (default: ../../../runs/wp10_bench/multi_arch);
#                      raccogli i JSON di TUTTE le macchine nella STESSA dir per confrontarle.
set -euo pipefail

ARCH="${1:?uso: bash bench_run.sh <etichetta-arch> [k-flussi] [dir-di-raccolta]}"
KFLOWS="${2:-100}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # inference/dati/benchmark
PKG="$(cd "$DIR/../.." && pwd)"                               # inference/
OUT="${3:-$(cd "$PKG/.." && pwd)/runs/wp10_bench/multi_arch}"
PY="$PKG/infer_venv/bin/python"
MODEL="unsw"                                                  # coerente col CSV base (unsw_bench_*)
NREP=2
TARGET=$(( KFLOWS * 1000 ))
# NIENTE default qui: il numero di thread BLAS lo decide infer.py (unica sorgente di verita'), cosi'
# il benchmark misura per COSTRUZIONE la configurazione che il deploy usa davvero. Un override
# dall'esterno (OPENBLAS_NUM_THREADS=N bash bench_run.sh ...) continua a funzionare: l'ambiente si
# propaga al figlio e infer.py usa setdefault. Il valore effettivo finisce nel JSON e nel riepilogo.

command -v /usr/bin/time >/dev/null || { echo "ERRORE: manca /usr/bin/time (GNU time). Installa: apt install time"; exit 1; }
[ -x "$PY" ] || { echo "ERRORE: infer_venv assente. Installa il venv di inferenza: bash install.sh infer"; exit 1; }

CSV="$(ls -S "$DIR"/*.csv 2>/dev/null | head -1 || true)"
[ -n "$CSV" ] || { echo "ERRORE: nessun CSV di benchmark in $DIR"; exit 1; }

mkdir -p "$OUT"
RIEP="$OUT/RIEPILOGO_${ARCH}.csv"
# intestazione solo alla creazione: i rilanci APPENDONO (v. ripetizioni accumulative sotto)
[ -f "$RIEP" ] || echo "arch,k_flussi,rep,blas,throughput_fl_s,throughput_gbps,rss_MB,wall_totale_s,wall_processing_s,coldstart_s" > "$RIEP"

# Ripetizioni ACCUMULATIVE: si riparte dalla prima rep libera invece di riscrivere rep1/rep2. Prima
# un rilancio con la STESSA etichetta sovrascriveva le misure precedenti, e chi ripeteva un punto per
# mediare perdeva i dati senza accorgersene (sweep BLAS del 2026-07-20: di 8 misure per punto ne
# sopravvivevano 2 su disco). plot_bench.py aggrega tutte le rep che trova.
BASE=0
while [ -f "$OUT/${ARCH}_${KFLOWS}k_rep$((BASE + 1)).json" ]; do BASE=$((BASE + 1)); done
[ "$BASE" -gt 0 ] && echo "  (trovate $BASE ripetizioni precedenti: le nuove si aggiungono da rep$((BASE + 1)))"

echo "== bench '$ARCH' : base $(basename "$CSV") | ${TARGET} flussi in finestre VARIABILI (simula il live) | modello $MODEL | BLAS=${OPENBLAS_NUM_THREADS:-<default di infer.py>} =="
for I in $(seq 1 "$NREP"); do
    REP=$((BASE + I))
    JSON="$OUT/${ARCH}_${KFLOWS}k_rep${REP}.json"
    TLOG="$OUT/.time_${ARCH}_${KFLOWS}k_rep${REP}.txt"
    /usr/bin/time -v "$PY" "$PKG/infer.py" --batch "$CSV" --model "$MODEL" \
        --bench --bench-target "$TARGET" --bench-out "$JSON" --bench-arch "$ARCH" \
        --alert-log /dev/null >/dev/null 2> "$TLOG"
    # RSS di picco (kbytes) e wall totale (h:mm:ss o m:ss.ss) da GNU time
    RSS_KB=$(awk -F': ' '/Maximum resident set size/{print $2}' "$TLOG")
    WALL=$(awk -F': ' '/Elapsed \(wall clock\)/{print $2}' "$TLOG")
    read -r TPUT GBPS WPROC RSS_MB WTOT COLD BLAS < <("$PY" - "$JSON" "$RSS_KB" "$WALL" <<'PY'
import json, sys
j, rss_kb, wall = sys.argv[1], float(sys.argv[2]), sys.argv[3]
d = json.load(open(j))
wtot = sum(float(x) * 60**i for i, x in enumerate(reversed(wall.split(":"))))
wproc = float(d["wall_s"])
print(d["throughput_flows_per_sec"], d["throughput_gbps"], round(wproc, 3),
      round(rss_kb / 1024, 1), round(wtot, 3), round(wtot - wproc, 3), d.get("blas_threads", "?"))
PY
)
    echo "$ARCH,$KFLOWS,$REP,$BLAS,$TPUT,$GBPS,$RSS_MB,$WTOT,$WPROC,$COLD" >> "$RIEP"
    echo "  ${KFLOWS}k rep${REP}: ${TPUT} fl/s (${GBPS} Gbit/s) | BLAS ${BLAS} | RSS ${RSS_MB} MB | cold ${COLD}s"
done
echo
echo "riepilogo tabellare : $RIEP"
echo "JSON per i grafici  : $OUT/${ARCH}_*.json"
# plot_bench.py usa matplotlib/seaborn -> gira nel train_venv, NON nel infer_venv (numpy-only).
echo "grafici             : $PKG/train_venv/bin/python $PKG/plot_bench.py --bench-dir $OUT --output $OUT/plots"
