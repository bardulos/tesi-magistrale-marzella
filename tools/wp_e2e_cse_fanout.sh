#!/usr/bin/env bash
# TAG: ONE-SHOT | fan-out 8 semi per trainer a seme singolo
# Fan-out for the single-seed trainer.
# Uses the deterministic 1.755M-row benign sample: the full 13.8M-row shard needed chunked training
# in train.py, which diverged from unchunked training.
# This accepts a smaller-data confound relative to the lab #569 run. TensorFlow requires CPU pinning;
# keep thread count fixed across seeds. Set concurrency and memory from the measured run.
# Usage: N_PAR=8 PER_SEED_GB=10 bash tools/wp_e2e_cse_fanout.sh [seed ...] (default 43..49)
set -u
cd "$(dirname "$0")/.."

SEEDS=(${@:-43 44 45 46 47 48 49})
N_PAR=${N_PAR:-4}
PER_SEED_GB=${PER_SEED_GB:-11}
BENIGN=runs/wp_e2e_cse/benign_1755k.npy
WSRC=inference/weights/cse
PY=$HOME/venv/bin/python
T=$(( 16 / N_PAR )); [ "$T" -lt 1 ] && T=1

avail=$(awk '/MemAvailable/{printf "%.0f", $2/1024/1024}' /proc/meminfo)
need=$(( N_PAR * PER_SEED_GB ))
echo "MemAvailable=${avail}GB  N_PAR=${N_PAR} x ~${PER_SEED_GB}GB = ~${need}GB  (T=${T} thread/proc)"
[ "$avail" -lt "$need" ] && { echo "STOP: RAM insufficiente (${avail}<${need})"; exit 1; }

i=0
for s in "${SEEDS[@]}"; do
  out=inference/weights/cse_seed${s}
  if [ -f "$out/train_report.json" ]; then echo "seed $s gia' completo (skip)"; continue; fi
  mkdir -p "$out"
  slot=$(( i % N_PAR ))
  c0=$(( slot * T )); c1=$(( c0 + T - 1 )); [ "$c1" -gt 15 ] && c1=15
  echo "[$(date +%H:%M:%S)] lancio seed $s su core ${c0}-${c1} -> $out/train.log"
  OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T TF_CPP_MIN_LOG_LEVEL=3 \
    /usr/bin/time -v -o "$out/time.txt" \
    taskset -c "${c0}-${c1}" nice -n 19 ionice -c 3 "$PY" -u inference/train.py \
      --benign-npy "$BENIGN" --out-dir "$out" --seed "$s" --weights-src "$WSRC" \
      > "$out/train.log" 2>&1 &
  i=$(( i + 1 ))
  if [ $(( i % N_PAR )) -eq 0 ]; then echo "  -- ondata piena (${N_PAR} semi), attendo..."; wait; fi
done
wait
echo "[$(date +%H:%M:%S)] fan-out CSE completato (${#SEEDS[@]} semi richiesti)"
