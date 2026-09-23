"""lib/secondo_stadio/constants.py — costanti del pilastro secondo_stadio (cap3).

Protocollo congelato al GATE A (2026-07-02, decisioni D1-D11).
Le dimensioni feature restano a sorgente unica in lib.utils; batch e semi base a sorgente
unica in lib.dae.constants (re-import, non ridefinizione).
"""
from lib.dae.constants import BATCH_FIXED, SEEDS  # noqa: F401 (re-export sorgente unica)
from lib.utils import N_FEATURES

# ---- Spazio phi = [z_s, r_w] ----
Z_DIM = 16               # latente del DAE #569, GIA' standardizzato nella cache output_569
R_DIM = N_FEATURES       # residui sbiancati Ledoit-Wolf (91)
PHI_DIM = Z_DIM + R_DIM  # 107 (v5 era 97 = 6+91: dimensioni SEMPRE via questa costante)

# ---- Prevalenza di RIFERIMENTO per il rate-scaling di MCC_bal (GATE A, D5) ----
# Terminologia VINCOLANTE: "prevalenza di riferimento (nativa del dataset)", MAI "prevalenza
# di deployment" (una rete reale ha prevalenza di attacco <<1%; il legacy v5 usa il nome
# sbagliato). La calibrazione e' rispetto a un reference prior scelto [siblini2020calibration];
# il riferimento nativo rende MCC_bal comparabile con le metriche del resto della tesi
# (MCC@FPR1% del DAE, stessa prevalenza). La robustezza rispetto alla prevalenza di deploy
# e' coperta dalla curva MCC_bal(pi) in forma chiusa (fase P7, metrics.mcc_bal_pi).
N_B = 1_775_098          # benigni val (ricalcolo 2026-07-02 su val_Attack.npy: identico al legacy)
N_A = 1_046_755          # attacchi val
PI_REF = N_A / (N_A + N_B)  # ~0.3709

P_GRID = [90.0, 92.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 99.3, 99.5, 99.7, 99.9]  # sweep soglia standalone (best_mcc_bal_standalone)

# ---- Split leak-free (D2, D7, D8) ----
# Split FISSO (non per-seme): il rumore-di-split e' sistematico anziche' mediato — stessa
# limitazione dichiarata del cap2 (FixHOptEst). Il TEST non entra MAI nella ricerca.
SPLIT_SEED = 0
VAL_FRAC = 0.20          # 80% train / 20% select (benigni; e attacchi per il ceiling, D2)
ES_N_ATTACKS = 50_000    # subsample attacchi per la metrica per-epoca (D7); finale su completi

# ---- Semi (D6) ----
# 8 semi base PA: i 6 canonici + 2 nuovi. 2036/2037 proseguono la serie senza collisioni
# con SEEDS_EXTRA del DAE (2025..2035, gia' spesi nella compressione cap2).
SEEDS_PA = list(SEEDS) + [2036, 2037]
# Semi extra della compressione P4 (K->K'): lista nuova, dimensione finale al GATE C.
SEEDS_PA_EXTRA = list(range(2038, 2049))

# ---- Rung ASHA (time_attr=seeds_completed) ----
# I knob del rung vivono nei config YAML del motore (grace_period/reduction_factor):
# P1 sup_search.yaml = 4/2 (rung unico a 4 semi su 6, come il DAE cap2);
# P3 pa_search.yaml  = 4/1.5 (rung a 4 e 6 semi su 8; rf FLOAT: engine patchato).
# max_t = len(seeds) via SearchComponent (max_t=None): niente costanti-eco mai lette.

# ---- Fissi MLP (D1, D3, D10, D11) ----
# L2 fissa: a ~1e-6 e' inerte (misurato nel cap2, test causale UNSW) — cercarla sprecherebbe
# una dimensione; il regolarizzatore CERCATO e' il dropout. Sentinella D3: pressione al bordo
# del dropout a 0.3 nella fANOVA di P1 => rivedere la regolarizzazione.
L2_FIXED_S2 = 1.0819231482664046e-06
MLP_MAX_EPOCHS = 100     # D10 (con patience 15 il cap e' raramente attivo; se attivo, segnalarlo)
MLP_PATIENCE = 15        # D1: ES per-epoca su AUC-PR del selection set, restore-best, MAI su MCC

# ---- Spazio di ricerca P1 (ceiling supervisionato): 5 dim, imbuto espansione->compressione ----
# Geometria fondata sulla letteratura (Bourlard-Kabil 2022 "Autoencoders reloaded", espansione
# nonlineare-poi-compressione alla kernel; Masters funnel a rapporto costante). DUE knob distinti
# (piu' il numero di layer, terzo knob discreto):
#   1) PRIMO hidden ESPANSO — h1 = e * N_in, e = fattore di espansione. Il primo layer NON puo'
#      comprimere sotto le N_in = PHI_DIM = 107 feature di phi (mappa overcomplete): e >= 1,5.
#      Letteratura: e in [1,5x; 4x] (1,5-3x consenso MLP; 4x = FFN transformer, Vaswani 2017)
#      -> con N_in=107, h1 in [160, 428].
#   2) COMPRESSIONE geometrica verso l'output (1 neurone) — h_{k+1} = lambda * h_k, lambda = ratio.
#      Letteratura: lambda in [0,5; 0,7] (Masters halving 0,5 -> taper 0,7). La compressione agisce
#      DOPO l'espansione (layer >= 2); che un hidden profondo scenda sotto N_in e' voluto (imbuto).
# Revisione 2026-07-02: il primo layer deve espandere.
SUP_N_LAYERS = (2, 3, 4)                            # categorico (numero di hidden layer)
SUP_EXPANSION_LOW, SUP_EXPANSION_HIGH = 1.5, 4.0    # e = h1 / N_in (1,5-3x consenso MLP; 4x = FFN transformer)
SUP_H1_LOW = round(SUP_EXPANSION_LOW * PHI_DIM)      # 160 = round(1,5*107): primo layer sempre ~>=1,5x input
SUP_H1_HIGH = round(SUP_EXPANSION_HIGH * PHI_DIM)    # 428 = 4*107
SUP_RATIO_LOW, SUP_RATIO_HIGH = 0.5, 0.7            # lambda: compressione geometrica (Masters 0,5 - taper 0,7)
SUP_LR_LOW, SUP_LR_HIGH = 2e-4, 5e-3               # log
SUP_DROPOUT_LOW, SUP_DROPOUT_HIGH = 0.0, 0.3       # regolarizzatore cercato (v. sentinella D3)

# ---- Spazio di ricerca P3 (classificatore PA) — 8 dim ----
# La PA CERCA l'architettura (intervalli di P1, SUP_*, INTATTI): compito diverso (pseudo-anomalie,
# non attacchi reali) -> la fANOVA di P1 non trasferisce. UNICO restringimento: lr in [1e-4,1e-3]
# (il ceiling e' ottimo al floor 2e-4). Pseudo: delta = frazione radiale; alpha_lo>=1 evita
# pseudo-benigni, alpha_hi<=5 tiene il confine non banale. Warm-start dalle 10 architetture piu'
# diverse del ceiling P1. Redesign 2026-07-03.
PA_LR_LOW, PA_LR_HIGH = 1e-4, 1e-3          # lr ristretto (UNICO narrowing; log): ceiling ottimo al floor
DELTA_LOW, DELTA_HIGH = 0.0, 1.0
ALPHA_LO_LOW, ALPHA_LO_HIGH = 1.0, 2.0
ALPHA_HI_LOW, ALPHA_HI_HIGH = 2.0, 5.0
# NB: architettura via SUP_EXPANSION/SUP_RATIO/SUP_DROPOUT/SUP_N_LAYERS (P1); niente piu' PA_H1/PA_N_LAYERS.

# ---- fp-mining (P6, valori canonici da config legacy) ----
FPM_K = 4          # copie EXTRA per falso positivo minato (molteplicita' x(K+1))
FPM_K_MIN = 0.002  # default legacy di stop_threshold: oggi la soglia si applica al Delta pAUC_MCC
FPM_N_MAX = 5
# --- metrica partial_auc_mcc (P6 v6): AUC_MCC PARZIALE (Orlova et al. 2025, arXiv 2507.09338v2)
#     ristretta al band operativo di FPR, NORMALIZZATA = MCC medio sul band (scala MCC). ---
FPM_FPR_LO = 0.005      # estremo inferiore del band operativo
FPM_FPR_HI = 0.015      # estremo superiore del band operativo
FPM_CURVE_N = 41        # punti-griglia per l'integrale del band (trapezi)
FPM_PLOT_FPR_LO = 0.001  # curva ESTESA per il plotting / dati grezzi
FPM_PLOT_FPR_HI = 0.030
FPM_PLOT_N = 59
FPM_STOP_SIGMA = 1.0    # auto-stop per-modello: Delta(pAUC_MCC) < c*sigma_m  (c=1 -> 1 sigma)

ALPHA_AND = 0.005  # FPR bersaglio della regione AND sul selection set (P7)

# ---- WP-4_v2: pseudo-anomalie di MONITORING del PA (criterio d'arresto label-free) ----
# Spazio DISTINTO dallo spazio P3 (ALPHA_LO/ALPHA_HI/DELTA sopra, che e' l'architettura+training-
# pseudo del PA): qui si cercano i parametri delle pseudo usate SOLO come monitor BCE (mai nel
# training), range piu' ampio per includere pseudo sia piu' difficili (alpha verso 1) sia piu'
# facili delle canoniche (piano approvato 2026-07-10).
MON_SEED_V2 = 20260710          # seme FISSO partizione monitor_v2 (indipendente dal seme modello;
                                 # diverso da MON_SEED=20260706 di WP-3b(i), archiviato/non riusato)
PSEUDO_MONITOR_SEED = 20260711  # seme FISSO base per le pseudo-monitor generate nel loop Optuna
                                 # (deriva per trial: generators.pseudo_seed_for_trial, usata dal driver)
WP4V2_MON_N = 200_000           # taglia del pool monitor_v2 (4x il monitor_v1 di WP-3b(i), 50k)
WP4V2_ALPHA_LO_LOW, WP4V2_ALPHA_LO_HIGH = 1.0, 2.5
WP4V2_ALPHA_HI_LOW, WP4V2_ALPHA_HI_HIGH = 1.2, 5.0
WP4V2_DELTA_LOW, WP4V2_DELTA_HIGH = 0.0, 1.0
WP4V2_RUNG_1, WP4V2_RUNG_2 = 7, 14   # pruning SuccessiveHalving: rung a 7 e 14 semi (su 19 totali)
WP4V2_OPTUNA_SEED = 20260712    # base sampler TPE (+worker_id per worker, evita first-sample
                                 # correlati fra processi — stesso fix di tools/synth_search.py
                                 # `seed=1000+worker_id`). Distinto da POOL_SEED/GEN_SEED_REF/
                                 # GEN_SEED_BASE (20260709-11) di tools/synth_search.py: WP-4
                                 # legacy, obiettivo diverso (copertura open-space), non toccato.
WP4V2_N_STARTUP_TRIALS = 100    # trial random iniziali prima che il TPE subentri (spazio 3-dim)
