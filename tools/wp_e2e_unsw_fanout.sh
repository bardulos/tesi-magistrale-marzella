#!/usr/bin/env bash
# TAG: ONE-SHOT | 2026-07-13 | fan-out 8 semi WP-E2E | vedi docs/refactoring_censimento.md
# ONE-SHOT (WP-E2E)
# Fan-out del retrain label-free multi-seme su workstation. NESSUNA logica: compone train.py (single-seed, N1)
# su piu' semi in parallelo shell (mai multiprocessing Python). Ogni processo e' pinnato a 4 core con
# taskset (workstation = 8C/16T -> 4 paralleli x 4 core = 16, zero oversubscription; TF non rispetta gli env di
# threading, il pinning di affinita' e' l'unico modo pulito). Determinismo: il thread-count (4) e'
# costante fra i semi -> ogni seme e' bit-riproducibile con lo stesso taskset (verificato).
# Uso:  bash tools/wp_e2e_unsw_fanout.sh [seed ...]   (default 43..49; seed 42 = canary, gia' lanciato)
set -u
cd "$(dirname "$0")/.."

SEEDS=(${@:-43 44 45 46 47 48 49})
CORES=("0-3" "4-7" "8-11" "12-15")          # 4 slot su workstation (8 core fisici / 16 thread)
BENIGN=runs/unsw_local/data/train_X.npy
WSRC=inference/weights/cse
PY=$HOME/venv/bin/python
PER_SEED_GB=3                                # stima footprint/istanza (train_X mmap condivisa via page-cache)

avail=$(awk '/MemAvailable/{printf "%.0f", $2/1024/1024}' /proc/meminfo)
echo "MemAvailable=${avail}GB  paralleli=4 x ~${PER_SEED_GB}GB = ~$((4*PER_SEED_GB))GB"
[ "$avail" -lt $((4*PER_SEED_GB)) ] && { echo "STOP: RAM insufficiente"; exit 1; }

i=0
for s in "${SEEDS[@]}"; do
  out=inference/weights/unsw_seed${s}
  if [ -f "$out/train_report.json" ]; then echo "seed $s gia' completo (skip)"; continue; fi
  mkdir -p "$out"
  cset=${CORES[$(( i % 4 ))]}
  echo "[$(date +%H:%M:%S)] lancio seed $s su core $cset -> $out/train.log"
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TF_CPP_MIN_LOG_LEVEL=3 \
    taskset -c "$cset" nice -n 19 ionice -c 3 "$PY" inference/train.py \
      --benign-npy "$BENIGN" --out-dir "$out" --seed "$s" --weights-src "$WSRC" \
      > "$out/train.log" 2>&1 &
  i=$(( i + 1 ))
  if [ $(( i % 4 )) -eq 0 ]; then echo "  -- ondata piena (4 semi), attendo il completamento..."; wait; fi
done
wait
echo "[$(date +%H:%M:%S)] fan-out completato (${#SEEDS[@]} semi richiesti)"
