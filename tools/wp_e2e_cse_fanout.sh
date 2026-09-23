#!/usr/bin/env bash
# TAG: ONE-SHOT | 2026-07-14 | fan-out 8 semi WP-E2E-CSE | vedi docs/refactoring_censimento.md
# ONE-SHOT (WP-E2E-CSE)
# Fan-out del retrain label-free multi-seme su workstation per il dominio primario CSE. NESSUNA logica:
# compone inference/train.py (single-seed) su piu' semi in parallelo shell (mai multiprocessing Python).
# Benigno CSE = runs/wp_e2e_cse/benign_1755k.npy (SUB-SAMPLE deterministico 1.755.000 righe, seed 0,
# da train_shard_00.npy 13,8M). Scelta 2026-07-14: il full 13,8M richiedeva un chunking di train.py
# che divergeva dall'unchunked (non validabile a ±0,02); si torna alla scala di progetto SIZE_REF
# (~1,755M) col pacchetto AS-IS, bit-exact. Confound accettato: il DAE vede meno dati del #569 lab.
# Ogni processo e' pinnato con taskset (TF 2.21 non rispetta gli env di threading -> il pinning e'
# l'unico modo pulito; thread-count costante fra i semi = bit-riproducibile).
#
# Concorrenza (N_PAR) e footprint/istanza (PER_SEED_GB) sono DECISI DAL CANARY (misura RSS reale) e
# passati via env. Core-pinning derivato: T = 16 / N_PAR thread per processo.
# Uso:  N_PAR=8 PER_SEED_GB=10 bash tools/wp_e2e_cse_fanout.sh [seed ...]   (default seed 43..49)
set -u
cd "$(dirname "$0")/.."

SEEDS=(${@:-43 44 45 46 47 48 49})
N_PAR=${N_PAR:-4}                                   # processi concorrenti (dal canary)
PER_SEED_GB=${PER_SEED_GB:-11}                      # footprint reale misurato dal canary (~10,6GB/proc)
BENIGN=runs/wp_e2e_cse/benign_1755k.npy
WSRC=inference/weights/cse                          # metadati preprocessing CSE (default corretto)
PY=$HOME/venv/bin/python
T=$(( 16 / N_PAR )); [ "$T" -lt 1 ] && T=1          # thread hw per processo

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
