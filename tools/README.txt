tools/ — driver e orchestratori: criterio del perimetro pubblicato

Questa cartella, nel repository di lavoro, contiene 183 file fra driver di pipeline, one-shot di
processo, diagnostiche di cluster e strumenti superati. Nel repository pubblicato ne compaiono 33.

CRITERIO
--------
Il repository documenta la tesi: ogni fase che la tesi descrive come pipeline di addestramento ha
qui il suo codice — ricerca, selezione e compressione della rosa (DAE e classificatore), classifica
a tre assi, scelta del seme di produzione, LOAO, hardening, calibrazione della fusione e dei
criteri label-free, addestramento del pacchetto di deploy, trasferimento di dominio — più i
produttori degli artefatti consegnati e il gate bit-exact del modello congelato #569.

Restano fuori i generatori di figure, gli script di analisi eseguiti una volta per sostenere una
scelta, gli script di puro testo (tabelle, macro, impaginazione di risultati già calcolati), i
monitor e le diagnostiche di cluster, le verifiche a posteriori e gli strumenti superati. Unica
eccezione: monitor_search.py, monitor in sola lettura della ricerca, incluso come dipendenza
tecnica di compress_tables.py, che lo usa solo per l'etichetta della colonna dei trial (il numero
Optuna): la classifica non ne dipende.

ONE-SHOT NON SIGNIFICA CODICE MORTO
-----------------------------------
Diciassette dei trentatré portano la classe ONE-SHOT nell'intestazione. La classe descrive il REGIME
D'USO, non lo stato di salute: una esecuzione singola, che ha prodotto l'artefatto o il risultato
consegnato e non va ripetuta. Il codice è vivo e resta la documentazione eseguibile di come
quell'artefatto è nato.

I TRENTATRÉ FILE
----------------
                                   classe            fase / prodotto

  Primo stadio (DAE)
  select_shortlist.py              DRIVER-PIPELINE   congelamento della rosa K dalla ricerca (PCS)
  refit_compress.py                DRIVER-PIPELINE   compressione multiseme K -> K'
  compress_tables.py               ONE-SHOT          classifica dei finalisti della compressione
                                                     (AP e MCC su 17 semi): elegge il #569
  monitor_search.py                DIAGNOSTICA-CLUSTER  dipendenza tecnica di compress_tables.py:
                                                     solo l'etichetta della colonna dei trial
                                                     (numero Optuna)
  loao_ray.py                      DRIVER-PIPELINE   pesi dei finalisti, fra cui il #569 seme 2034
                                                     -> fuori per dimensione (runs/), ma il derivato
                                                        è dentro: models/*/dae.npz
  loao_run.py                      DRIVER-PIPELINE   LOAO forward-only sui pesi dei finalisti
  dae_latent_history.py            ONE-SHOT          statistiche latenti per epoca, 17 semi: base
                                                     del cap-epoche del criterio d'arresto
  dae_entropy_contam.py            ONE-SHOT          criterio d'arresto EntropyStop su training
                                                     contaminato: train, analyze (griglia k x
                                                     R_down, best_config), verify; grid e plot
                                                     richiedono lib/plotting, non pubblicato
  sync_weights_from_vps.sh         DRIVER-PIPELINE   trasporto dei pesi dal cluster effimero
  dae_output_cache.py              DRIVER-PIPELINE   cache φ, runs/dae/output_569/
                                                     -> fuori (runs/), derivati incorporati in pa.npz
  gate_569_bitexact.py             DRIVER-PIPELINE   codice di uscita del gate bit-exact del #569

  Secondo stadio (classificatore a pseudo-anomalie)
  pa_shortlist.py                  DRIVER-PIPELINE   rosa del classificatore: pre-cull + PCS
  pa_compress.py                   DRIVER-PIPELINE   compressione multiseme della rosa
  pa_harden_run.py                 DRIVER-PIPELINE   hardening con FP-mining (fan-out Ray)
  sup_loao_run.py                  DRIVER-PIPELINE   LOAO supervisionato a riaddestramento (tetto)
  pa_loao_run.py                   DRIVER-PIPELINE   LOAO del classificatore a riaddestramento
  pa_loao_smoke.py                 ONE-SHOT          LOAO forward-only: una cella, censimento pesi
  pa_loao_worker.py                ONE-SHOT          LOAO forward-only: una coppia (modello, seme)
  pa_loao_fanout.py                ONE-SHOT          LOAO forward-only: orchestratore locale
  pa_loao_aggregate.py             ONE-SHOT          LOAO forward-only: aggregato sui 19 semi,
                                                     soglia dichiarata, criterio graduato
  pa_tre_assi.py                   ONE-SHOT          classifica a tre assi dei finalisti
  pa_production_seed.py            ONE-SHOT          scelta del seme di produzione del #589
  pa_wp4v2_repass.py               ONE-SHOT          calibrazione label-free (WP-4_v2), fase 0:
                                                     re-pass del #589 con pesi per epoca
  pa_wp4v2_search.py               ONE-SHOT          calibrazione label-free (WP-4_v2), fasi 1-3:
                                                     precalc, ricerca (stage wp4v2_es_search), validate

  Deploy e trasferimento di dominio
  convert_weights_to_npz.py        DRIVER-PIPELINE   models/cse/{dae,pa}.npz, soglie.json
                                                     -> nel repository
  wp_e2e_cse_fanout.sh             ONE-SHOT          models/cse_e2e/ (orchestra inference/train.py)
  wp_e2e_unsw_fanout.sh            ONE-SHOT          models/unsw/ (orchestra inference/train.py)
  deploy_unsw_train.py             ONE-SHOT          riaddestramento del DAE su UNSW (retrain delle
                                                     configurazioni finaliste, sweep del collo di
                                                     bottiglia)

  Dati
  materialize_test_npy.py          DRIVER-PIPELINE   runs/preprocessing/test_npy -> fuori (runs/)
  make_cse_test_raw.py             DRIVER-PIPELINE   cse_test_raw.csv (570 MB)
                                                     -> nel repository come .7z; rigenerabile
  make_unsw_test_raw.py            ONE-SHOT          unsw_test_raw.csv (66 MB) -> nel repository
  make_unsw_contaminated.py        ONE-SHOT          unsw_train_contam1_raw.csv (417 MB)
                                                     -> nel repository come .7z; rigenerabile
  make_lan_contaminato.py          ONE-SHOT          lan_contaminato.csv (610 MB)
                                                     -> cattura di rete privata, NON DISTRIBUITO

Nota su loao_ray.py e loao_run.py: il primo ha scritto i pesi del #569 ed è quindi la sorgente
di models/*/dae.npz; il secondo lo supera per il solo calcolo LOAO forward-only, non per la
produzione dei pesi. Entrambi importano lib/plotting, non pubblicato, per le figure: l'import
sta in un blocco protetto e senza figure il calcolo procede.

lib/plotting non è pubblicato. Oltre ai due sopra lo usano i sottocomandi grid e plot di
dae_entropy_contam.py (l'import è dentro il sottocomando: train, analyze e verify non lo
toccano) e il ramo figure di dae_output_cache.py (protetto). Nessun calcolo dipende da esso.

Controllo incrociato non eseguito: pa_wp4v2_repass.py confronta la configurazione del #589, letta
da hardening_candidates.json, con il database Optuna della ricerca del classificatore, usando un
helper di monitor_search4.py, monitor in sola lettura non incluso. Nel repository pubblicato il
controllo quindi non viene eseguito: lo script lo segnala nello stdout («cross-check DB saltato»)
e procede. Non influisce sui risultati: la configurazione viene comunque da
hardening_candidates.json, e l'esito del controllo non entra né nel JSON né nei pesi prodotti.

ASSUNZIONI D'AMBIENTE
---------------------
Alcuni valori di runtime riflettono la macchina di laboratorio e non sono stati modificati.

  Segnaposto nelle configurazioni, da fornire con --override o copiando il config:
    configs/dae/{search,search_canonical,search_extended,output_cache}.yaml — shard_dir, val_X,
      val_y (<HOME>/dae_data_91) e weights_home (<HOME>/repov6_weights);
    configs/secondo_stadio/{pa_search,sup_search}.yaml — weights_home (<HOME>/repov6_weights_s2)
      e, in pa_search.yaml, enqueue_configs (<HOME>/repov6_ops/...: il file è in
      configs/ops_archive/);
    tools/make_cse_test_raw.py (V5_BASE) e tools/make_lan_contaminato.py (RAW): sorgenti
      <MEDIA>/... dell'archivio dei repository precedenti.

  Tool che presuppongono il repository in ~/tesi/repov6 (via Path.home() o expanduser):
    tools/make_lan_contaminato.py, tools/make_unsw_contaminated.py, tools/monitor_search.py
    (default di --repo); tools/make_unsw_test_raw.py presuppone anche l'oracolo in
    ~/tesi/repo_v5, e l'aiuto di --oracle-dir di tools/materialize_test_npy.py rimanda allo
    stesso oracolo esterno (runs/preprocessing/transform_cse/). tools/gate_569_bitexact.py legge
    ~/dae_data_91/val_X.npy. tools/pa_compress.py e tools/pa_harden_run.py hanno come default di
    --weights-home la cartella ~/repov6_weights_s2.

  Tool che presuppongono il venv in ~/venv: tools/wp_e2e_cse_fanout.sh, tools/wp_e2e_unsw_fanout.sh.

  tools/sync_weights_from_vps.sh: nodo1…nodo5 sono alias SSH dei nodi del cluster, definiti
  nella configurazione SSH locale; la radice remota di default è ~/repov6_weights.

TESTI D'AIUTO E DEFAULT STORICI
-------------------------------
Alcuni testi d'aiuto e valori di default dei tool riflettono il laboratorio e non sono stati
modificati.

  tools/make_unsw_test_raw.py: i default di --scaler e --transform-meta puntano a
  inference/weights/cse/, cartella storica che non esiste nel repository; vanno passati i file
  scaler_params.json e transform_meta.json di inference/modelli/produzione/cse/.

  tools/pa_wp4v2_search.py: l'aiuto del sottocomando search e quello di --worker-id citano
  pa_wp4v2_run.sh, il lanciatore di cluster della fase 2 non incluso (descritto sotto).

  tools/dae_entropy_contam.py: gli aiuti dei sottocomandi plot e grid descrivono figure che
  richiedono lib/plotting, non pubblicato (v. sopra).

SEGNAPOSTO NELLE CONFIGURAZIONI
-------------------------------
Le configurazioni in configs/ portano i segnaposto letterali <HOME> e <MEDIA> al posto dei
percorsi della macchina di laboratorio (dati esterni al repository come dae_data_91, radici dei
pesi). Nessun codice li espande: i percorsi reali si forniscono al lancio con
--override chiave=valore, oppure copiando il config e sostituendoli.

ORCHESTRAZIONE SU CLUSTER: DESCRITTA, NON INCLUSA
-------------------------------------------------
Le fasi pesanti girarono su un cluster effimero di cinque nodi più la workstation. I lanciatori
di cluster (sincronizzazione dei nodi, gate dei dati, fan-out via ssh) non sono inclusi; questo è
ciò che facevano.

  Ricerche (dae.py, secondo_stadio.py): motore Ray (lib/search/engine.py) con ray_address=auto sui
  nodi; i pesi dei trial nella radice weights_home di ciascun nodo, raccolti con
  sync_weights_from_vps.sh.

  pa_compress.py, pa_harden_run.py, loao_ray.py, refit_compress.py: fan-out Ray sugli stessi
  nodi; i pesi in una radice fuori dall'albero (--weights-home o weights_home del config).

  dae_latent_history.py: 17 semi in parallelo sulla workstation, un processo per seme a un thread,
  cap 60 epoche senza early-stop, un JSON per seme in runs/dae/latent_history/.

  WP-4_v2 (pa_wp4v2_repass.py, pa_wp4v2_search.py): fase 0, re-pass del #589 con pesi salvati a
  ogni epoca, 19 semi ripartiti su due nodi (10 + 9); fase 1, precalc sulla workstation; fase 2,
  in due forme: il sottocomando search di pa_wp4v2_search.py, più processi worker in parallelo (16
  sulla workstation, 12 sui nodi da 30 GB) lanciati da uno script di cluster non incluso, con uno
  storage JournalStorage per nodo e un id di worker globale (con un offset per nodo) che entra nel
  seme del campionatore, e lo stage wp4v2_es_search del
  dispatcher su Ray, da cui viene il vincitore (runs/secondo_stadio/wp4v2_search/ray/winner.json);
  fase 3, validate sulla workstation.

  Trasferimento UNSW (deploy_unsw_train.py), due lanciatori locali:
  1. retrain di 4 configurazioni finaliste (tag exp mid btl sigma lr: 569 112 80 16 0.1583
     0.00207; 341 109 46 12 0.1400 0.00354; 556 145 79 10 0.1442 0.00209; 404 76 46 14 0.1635
     0.00184) su 4 semi (42, 123, 2024, 7): 16 job, 4 thread ciascuno, priorità minima, guardia
     sulla RAM disponibile; uscite in runs/unsw_local/results/, più una variante senza
     corruzione (sigma forzato a 0) in results_nodrop/;
  2. sweep del collo di bottiglia: mid fisso, un seme (42), un job per valore di btl, uscite in
     runs/unsw_local/results_sweep_<tag>/btl<btl>_seed<seme>.csv; valori usati: #341 (109, 46,
     0.1400, 0.00354) con btl 12 11 10 9 8 7 6 5 4 3; #477 (240, 50, 0.1592, 0.00039) con btl 16
     14 12 10 8 7 6 5 4 3; #569 (112, 80, 0.1583, 0.00207) con btl 15 14 13 12 11 10 9 8 7 6 5 4 3.
     Ogni punto: deploy_unsw_train.py --exp --mid --btl --sigma --lr --seed --tag
     --data-dir runs/unsw_local/data --out <csv> --threads 1.

L'INTESTAZIONE DI OGNI FILE
---------------------------
Ogni .py e ogni .sh di questa cartella porta, entro le prime tre righe:

  # TAG: <classe> | <data> | <motivo> | vedi docs/refactoring_censimento.md

con classe fra DRIVER-PIPELINE, ONE-SHOT, DIAGNOSTICA-CLUSTER, SUPERATO.

Il documento citato in coda al tag, docs/refactoring_censimento.md, è il censimento interno del
lavoro e NON fa parte di questa pubblicazione: il riferimento resta nell'intestazione perché
riscriverlo avrebbe significato toccare l'intestazione di ogni sorgente. Ciò che serve per leggere
il perimetro pubblicato è in questo file.

Per la stessa ragione i sorgenti, qui e in lib/, citano anche strumenti e documenti non inclusi:
riferimenti storici, mai invocati a runtime.

COME SI ESEGUE UNO STAGE DELLA PIPELINE
---------------------------------------
I driver di questa cartella non sono il punto d'ingresso della pipeline. Gli stage si lanciano dai
tre entry-point in radice:

  python <entry>.py configs/<pilastro>/<stage>.yaml [--override chiave=valore ...]

con <entry> fra preprocessing.py, dae.py, secondo_stadio.py.
