================================================================================
KIT BENCH MULTI-MACCHINA — misurare il throughput di inferenza su più macchine
================================================================================
Scopo: eseguire la STESSA sequenza di misura, identica, su ogni macchina fisica, e
raccogliere i risultati in un formato confrontabile. Nessuna modifica al runtime:
si compongono solo infer.py --bench (produce i JSON) e /usr/bin/time (cattura il RSS).

Il numero che conta è il throughput di DEPLOY (--bench spento). --bench PAGA la
strumentazione: il breakdown per-step nel JSON è istrumentato, il throughput_flows_per_sec
del JSON è comunque la misura buona per il confronto fra macchine (tutte pagano lo stesso).


--------------------------------------------------------------------------------
0. COSA SERVE SULLA MACCHINA NUOVA
--------------------------------------------------------------------------------
  - python3.12: l'inferenza è MONOPROCESSO a thread Python singolo; il parallelismo è interno
    al BLAS (OpenBLAS) sulle matmul. Nessun runtime free-threaded richiesto.
  - /usr/bin/time (GNU time):  apt install time     (serve per il picco RSS)
  - rete, solo per install.sh (scarica numpy nel venv).
  - Per i GRAFICI serve anche python3.12 + il train_venv (matplotlib/seaborn): opzionale,
    puoi disegnare i grafici su UNA sola macchina raccogliendo lì tutti i JSON.


--------------------------------------------------------------------------------
1. PORTARE IL PACCHETTO SULLA MACCHINA NUOVA
--------------------------------------------------------------------------------
Sulla macchina di partenza, dalla radice del repo (quella che contiene inference/):

  tar -czf nids_bench_kit.tar.gz \
      --exclude='inference/infer_venv' --exclude='inference/train_venv' \
      --exclude='inference/dati/inferenza' --exclude='inference/dati/addestramento' \
      --exclude='inference/modelli/sperimentali' \
      --exclude='inference/sniffing' --exclude='__pycache__' \
      --exclude='inference/alert.log' --exclude='inference/flussi_per_file.txt' \
      --exclude='inference/modelli/produzione/lan_contaminato' \
      inference/

Cosa entra (~30 MB): gli script (infer.py, plot_bench.py, main.py, train.py, install.sh),
modelli/produzione/ (cse, cse_e2e, unsw), dati/benchmark/ (i CSV di bench + questo kit).
Cosa NON entra: i venv (si reinstallano), i test set grandi, i dataset di training, i
modelli sperimentali, e OGNI residuo del traffico locale (alert.log, il modello
lan_contaminato, i conteggi della cattura): il kit non deve portare dati della LAN su
macchine terze. Copia il tar sulla macchina nuova ed estrai:

  tar -xzf nids_bench_kit.tar.gz
  cd inference


--------------------------------------------------------------------------------
2. INSTALLARE IL SOLO VENV DI INFERENZA
--------------------------------------------------------------------------------
  bash install.sh infer

Crea infer_venv (Python 3.12 + numpy) DENTRO inference/. NON installa il venv di training
(non serve al benchmark). Idempotente: se infer_venv c'è già, non lo ricrea.


--------------------------------------------------------------------------------
3. ESEGUIRE LA MISURA STANDARDIZZATA
--------------------------------------------------------------------------------
  bash dati/benchmark/bench_run.sh <etichetta-macchina>

<etichetta-macchina> = nome concordato, es. "kub-zen3", "portatile-i7", "$(hostname)-$(arch)".
Usa la stessa convenzione su tutte le macchine: quell'etichetta finisce nel campo "arch"
del JSON e nel nome dei file.

Cosa misura, sul bench CSV più grande presente in questa cartella (oggi unsw_bench_50k.csv,
scelto automaticamente col più grande disponibile), ripetuto 2 volte:
  - throughput steady-state (flussi/s, Gbit/s)  <- dal JSON di infer.py --bench
  - picco RSS                                    <- /usr/bin/time -v
  - cold start (import numpy + caricamento pesi) <- wall totale meno wall di processing

Output (default in runs/wp10_bench/multi_arch/ nella cartella UN livello sopra inference/;
sovrascrivibile col 3o argomento — il 2o argomento è il numero di migliaia di flussi,
default 100):
  - <arch>_<K>k_rep<R>.json   un JSON per run (formato che plot_bench.py consuma), con K = migliaia
                              di flussi richieste
  - RIEPILOGO_<arch>.csv       tabella: arch,k_flussi,rep,blas,throughput_fl_s,throughput_gbps,
                               rss_MB,wall_totale_s,wall_processing_s,coldstart_s
  - .time_*.txt                output grezzo di GNU time (ausiliari, ignorabili)

LE RIPETIZIONI SI ACCUMULANO. Rilanciando lo stesso benchmark con la STESSA etichetta le nuove
misure si aggiungono (rep3, rep4, ...) invece di sovrascrivere rep1/rep2, e il RIEPILOGO viene
appeso. plot_bench.py aggrega per media tutte le ripetizioni che trova, quindi ripetere un punto
serve a stringere la stima. (Fino al 2026-07-20 i rilanci sovrascrivevano: di 8 misure per punto
ne sopravvivevano 2.)

QUANTI THREAD BLAS. Lo script NON lo decide: eredita il default di infer.py, così il benchmark
misura per costruzione la stessa configurazione che il deploy userà. Per esplorare altri valori si
antepone la variabile:

  for T in 1 2 4 6 8 12 16; do
      OPENBLAS_NUM_THREADS=$T bash bench_run.sh "$(hostname)-${T}T" 100 /tmp/sweep
  done

Il valore effettivamente usato finisce nel JSON (campo blas_threads) e nella colonna "blas" del
riepilogo, quindi una misura non è mai ambigua. Sweep di riferimento su kub (8 core fisici, 8 misure
per punto), 2026-07-20: plateau fra 4 e 6 thread, crollo del 30% oltre gli 8 —
cioè appena OpenBLAS deve usare i fratelli iper-threading.

Ripeti sulle altre macchine, raccogliendo TUTTI i JSON nella STESSA cartella
(copia i JSON delle altre macchine dentro l'unica dir di raccolta).


--------------------------------------------------------------------------------
4. GRAFICI
--------------------------------------------------------------------------------
Su UNA macchina che abbia il train_venv (matplotlib/seaborn), con tutti i JSON raccolti
nella dir unica:

  train_venv/bin/python plot_bench.py --bench-dir <dir-di-raccolta> --output <dir-di-raccolta>/plots

Produce, PER OGNI macchina (arch):
  - <arch>_throughput.png          barre di throughput (1T vs NT, mode A/B)
  - <arch>_<mode>_<T>T_latency.png boxplot latenza per-step (dove va il tempo)


--------------------------------------------------------------------------------
5. CONFRONTO FRA MACCHINE
--------------------------------------------------------------------------------
Con i JSON di >=2 macchine nella stessa dir, plot_bench.py produce ANCHE la figura di
CONFRONTO affiancato:

  confronto_multiarch_throughput.png   barre raggruppate per configurazione (1T/A, NT/A, ...),
                                       una barra per ogni macchina, ripetizioni aggregate per media.

Con una sola macchina il confronto non si genera (basta il suo <arch>_throughput.png). In
aggiunta, la tabella comparativa è immediata unendo i RIEPILOGO_<arch>.csv (stesso schema per
tutte le macchine): throughput 1T e NT, RSS, cold start, macchina per macchina. Il RSS non entra
nei grafici: sta solo nel CSV.

Nota: il titolo del boxplot per-macchina mostra "py?" perché il JSON di --bench non include la
versione di Python; è cosmetico e non tocca i numeri.
