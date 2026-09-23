dati/benchmark/ - come misurare il throughput di inferenza

Simula il live (finestre di dimensione variabile, identiche su ogni macchina)
sul CSV di questa cartella, con il modello unsw. Due ripetizioni per lancio.
Comandi da eseguire nella cartella inference/.

REQUISITI
  python3.12, infer_venv (bash install.sh infer), GNU time (apt install time)
  grafici: train_venv (bash install.sh train)

USO
  python main.py -> 4
      misura e grafici in inference/benchmarks/
  bash dati/benchmark/bench_run.sh ETICHETTA [K] [DIR]
      ETICHETTA   nome della macchina, es. "$(hostname)-$(arch)"
      K           migliaia di flussi (default 100)
      DIR         cartella dei risultati (default: runs/wp10_bench/multi_arch
                  nella radice del repository)
  Con la stessa etichetta le ripetizioni si accumulano.

THREAD BLAS
  Default di infer.py (4). Altri valori:
    OPENBLAS_NUM_THREADS=T bash dati/benchmark/bench_run.sh "$(hostname)-${T}T"

GRAFICI
  train_venv/bin/python plot_bench.py --bench-dir DIR --output DIR/plots
  Per confrontare più macchine: risultati di tutte nella stessa DIR.

ALTRE MACCHINE
  Copiare inference/ senza venv, dati locali e residui del traffico locale
  (alert.log, modelli/produzione/lan_contaminato), poi bash install.sh infer.
