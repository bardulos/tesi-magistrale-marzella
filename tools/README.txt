tools/ - driver e orchestratori della pipeline

Serie di script non integrati nella architettura a 3 pilastri.
Nati come script per compito singolo e rimasti in questo stato.
Gli stage si occupano di importarli e lanciarli.


PRIMO STADIO (DAE)
  select_shortlist.py        rosa K dalla ricerca (PCS)
  refit_compress.py          compressione multiseme K -> K'
  compress_tables.py         classifica dei finalisti su 17 semi: elegge il #569
  monitor_search.py          per compress_tables.py: numero Optuna del trial
  loao_ray.py                LOAO su Ray; ha scritto i pesi dei finalisti, fra
                             cui il #569 seme 2034 (fonte di models/*/dae.npz)
  loao_run.py                LOAO forward-only sui pesi dei finalisti
  dae_latent_history.py      statistiche latenti per epoca (cap delle epoche)
  dae_entropy_contam.py      EntropyStop su training contaminato:
                             train, analyze, verify
  sync_weights_from_vps.sh   trasporto dei pesi dal cluster
  dae_output_cache.py        cache φ (runs/dae/output_569/)
  gate_569_bitexact.py       gate bit-exact del #569

SECONDO STADIO
  pa_shortlist.py            rosa: pre-cull + PCS
  pa_compress.py             compressione multiseme
  pa_harden_run.py           hardening con FP-mining (Ray)
  sup_loao_run.py            LOAO supervisionato a riaddestramento (tetto)
  pa_loao_run.py             LOAO a riaddestramento
  pa_loao_smoke.py           LOAO forward-only: una cella, censimento dei pesi
  pa_loao_worker.py          LOAO forward-only: una coppia (modello, seme)
  pa_loao_fanout.py          LOAO forward-only: orchestratore locale
  pa_loao_aggregate.py       LOAO forward-only: aggregato sui 19 semi
  pa_tre_assi.py             classifica a tre assi dei finalisti
  pa_production_seed.py      seme di produzione del #589
  pa_wp4v2_repass.py         calibrazione label-free, fase 0: re-pass del #589
  pa_wp4v2_search.py         calibrazione label-free, fasi 1-3: precalc,
                             ricerca, validate

DEPLOY E DOMINIO
  convert_weights_to_npz.py  models/cse/{dae,pa}.npz e soglie.json
  wp_e2e_cse_fanout.sh       models/cse_e2e/ (orchestra inference/train.py)
  wp_e2e_unsw_fanout.sh      models/unsw/ (orchestra inference/train.py)
  deploy_unsw_train.py       riaddestramento del DAE su UNSW, sweep del btl

DATI
  materialize_test_npy.py    runs/preprocessing/test_npy/
  make_cse_test_raw.py       cse_test_raw.csv (rigenerabile, v. inference/dati/)
  make_unsw_test_raw.py      unsw_test_raw.csv
  make_unsw_contaminated.py  unsw_train_contam1_raw.csv (rigenerabile)
  make_lan_contaminato.py    lan_contaminato.csv (cattura privata, non
                             distribuito)

AMBIENTE (valori del laboratorio, non modificati)
  <HOME> e <MEDIA> nei config: segnaposto da passare con --override
    (enqueue_configs di pa_search.yaml è in configs/ops_archive/).
  Percorsi fissi in ~ o <MEDIA>/: make_*.py, materialize_test_npy.py,
    monitor_search.py, gate_569_bitexact.py, pa_compress.py, pa_harden_run.py.
  wp_e2e_*_fanout.sh: venv ~/venv. sync_weights_from_vps.sh: alias SSH
    nodo1..nodo5.
  Aiuti storici: make_unsw_test_raw.py (--scaler e --transform-meta: usare
    inference/modelli/produzione/cse/), pa_wp4v2_search.py (cita
    pa_wp4v2_run.sh, non incluso).

CLUSTER (lanciatori non inclusi)
  Ricerche, compressioni, hardening e loao_ray.py: Ray su cinque nodi più la
    workstation; pesi raccolti con sync_weights_from_vps.sh.
  WP-4_v2: worker paralleli, un JournalStorage per nodo; il vincitore viene
    dallo stage wp4v2_es_search su Ray.
  dae_latent_history.py e deploy_unsw_train.py: lanciatori locali.
