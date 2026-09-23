"""lib/dae/constants.py — costanti del DAE e della ricerca AUC-PR (repov6).

Le dimensioni dello spazio feature sono a sorgente unica in lib/utils
(N_CONTINUOUS/N_BINARY/N_FEATURES): qui si re-importano, non si ridefiniscono.
Lo spazio di ricerca e l'objective sono fissati dal piano repov6 (motore AUC-PR).
"""
from lib.utils import N_BINARY, N_CONTINUOUS, N_FEATURES  # noqa: F401 (re-export sorgente unica)

# ---- Loss (TRAINING) ----
LOSS_CONTINUOUS = "mse"   # MSE sulle continue + BCE sulle binarie (bloccato, dichiarato)

# ---- Iperparametri BLOCCATI (non in ricerca, scelta di contenimento dichiarata) ----
BATCH_FIXED = 512
L2_FIXED = 1e-6
NOISE_TYPE = "gaussian"
# n_layers: encoder a 3 livelli (exp -> mid -> btl), autoencoder simmetrico (build_dae con mid>0).

# ---- Spazio di ricerca TPE (repov6) ----
# btl CERCATO [6,16] e exp [40,256]: allargati per la search +200 dove premevano i bordi
# (btl forte: ~34/64 rossi a 12; exp moderato). mid in [max(btl+1, MID_LOW), min(MID_HIGH, exp-1)]
# -> garantisce btl < mid < exp; sigma/lr invariati. Vincolo in lib.dae.objective.is_valid_geometry.
EXP_LOW, EXP_HIGH = 40, 256
MID_LOW, MID_HIGH = 20, 80
BTL_LOW, BTL_HIGH = 6, 16
SIGMA_LOW, SIGMA_HIGH = 0.01, 0.20
LR_LOW, LR_HIGH = 1e-4, 1e-2   # log

# ---- Objective Optuna ----
# mean(AUC-PR) − K_STD·std(AUC-PR) sui semi genuini. La penalita' −std e' la varianza
# inter-seme (robustezza/deploy), NON parsimonia (esclusa: nessuna penalita' sui parametri).
K_STD = 1.0

# ---- Early-stopping su AUC-PR (B3: PROVVISORIO) ----
# PATIENCE_PROVVISORIO e' un placeholder-per-il-dry-run, NON il valore finale: era tarato
# sulla rumorosita' MCC; l'AUC-PR e' piu' liscia -> patience ottimale verosimilmente piu'
# basso. Il valore DEFINITIVO esce dallo Step 5 (misura sulle traiettorie AUC-PR reali della
# lunghezza tipica dell'avvallamento-prima-di-risalita). Il restore-best rende il patience
# alto privo di rischio.
PATIENCE_PROVVISORIO = 17

# ---- Filtro non-apprendimento (B1) ----
# Ancorato alla PREVALENZA MISURATA sul val (AUC-PR casuale ≈ prevalenza, NON 0): un seme e'
# scartato se dopo NON_LEARNING_MIN_EPOCHS l'AUC-PR migliore <= prevalenza*(1+NON_LEARNING_MARGIN).
# La prevalenza si misura a runtime su val_y; il margine e' da rifinire allo Step 5.
NON_LEARNING_MIN_EPOCHS = 10
NON_LEARNING_MARGIN = 0.05

# ---- Multi-seed ----
# 6 semi canonici: la varianza inter-seme e' reale (l'AUC-PR liscia riduce il rumore
# entro-seme, non la dispersione tra-seme). Valori = semi RNG arbitrari (nessuna
# contaminazione di metrica). L'ORDINE dei semi per-trial si deriva dal trial id (lib.search.engine).
SEEDS = [42, 123, 2024, 7, 99, 1337]

# Semi EXTRA per la compressione multiseed (FASE 2, K->k'): il 7°…17° seme, usati DIRETTAMENTE
# (nessuna permutazione: ogni job refit_compress addestra un (config, seme) preciso). Valori
# arbitrari congelati, distinti dai 6 base, continuano dal base 2024 → riproducibilità bit-exact.
# 11 valori: n'=6+11=17, taratura canonica per comprimere K=71→k'=25 a conf 0.80 (PCS 0.803).
SEEDS_EXTRA = [2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]

# ---- ASHA (rung unico al 4 seme) ----
# max_t = len(SEEDS) = 6; al 4 seme completato ASHA decide: chi promuove prosegue a 6,
# chi no e' marcato PRUNED dal searcher (PruningAwareOptunaSearch.on_trial_complete ->
# study.tell(PRUNED): il raise di optuna.TrialPruned NON funziona sotto Ray 2.55, v. engine)
# e valutato sul mean−std parziale dei 4 semi. NB: il TPE di Optuna 4.8 INCLUDE i PRUNED
# nel fit col valore-al-rung (scelta accettata e dichiarata).
# B2: REDUCTION_FACTOR e frazione di promozione sono PLACEHOLDER, da ricalibrare sui valori
# AUC-PR reali (compressi in [prevalenza,1]) allo Step 5 — NON ereditati dalla config MCC di v5.
MAX_T = 6
# grace_period (4) e reduction_factor (2) sono knob del MOTORE, letti dal config YAML
# (lib/search/engine.py; search_canonical.yaml): qui non vivono costanti-eco mai lette.

# ---- Soglia di anomalia (inferenza / LOAO, FASE 3) ----
# tau = percentile (1 - FPR_TARGET) degli score sui benigni di validazione (p99 per FPR 1%).
# Identica per tutti i fold del LOAO (escludere una classe non cambia training né soglia).
FPR_TARGET = 0.01
