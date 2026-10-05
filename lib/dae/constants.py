"""Costanti del DAE e della ricerca AUC-PR. Le dimensioni feature vengono da lib.utils."""
from lib.utils import N_BINARY, N_CONTINUOUS, N_FEATURES  # noqa: F401 (re-export sorgente unica)

# ---- Loss (TRAINING) ----
LOSS_CONTINUOUS = "mse"

# ---- Iperparametri BLOCCATI (non in ricerca, scelta di contenimento dichiarata) ----
BATCH_FIXED = 512
L2_FIXED = 1e-6
NOISE_TYPE = "gaussian"
# n_layers: encoder a 3 livelli (exp -> mid -> btl), autoencoder simmetrico (build_dae con mid>0).

# ---- Spazio di ricerca TPE ----
# mid e' vincolato a [max(btl+1, MID_LOW), min(MID_HIGH, exp-1)], quindi btl < mid < exp.
# Rispetto a runs/dae/search_400 sono allargati solo i tetti di btl (12->16) ed exp
# (224->256); i range di sigma e lr sono gli stessi.
EXP_LOW, EXP_HIGH = 40, 256
MID_LOW, MID_HIGH = 20, 80
BTL_LOW, BTL_HIGH = 6, 16
SIGMA_LOW, SIGMA_HIGH = 0.01, 0.20
LR_LOW, LR_HIGH = 1e-4, 1e-2   # log

# ---- Objective Optuna ----
# Objective: mean(AUC-PR) - K_STD * std sui semi genuini; penalizza la variabilita'
# tra semi, non il numero di parametri.
K_STD = 1.0

# ---- Early-stopping su AUC-PR (provvisorio) ----
# Placeholder per il dry-run; il valore definitivo va tarato sulle traiettorie AUC-PR
# reali. Restore-best conserva i pesi dell'epoca migliore.
PATIENCE_PROVVISORIO = 17

# ---- Filtro non-apprendimento ----
# Scarta un seme se, dopo NON_LEARNING_MIN_EPOCHS, l'AUC-PR migliore non supera
# prevalenza * (1 + NON_LEARNING_MARGIN). La prevalenza e' misurata su val_y.
NON_LEARNING_MIN_EPOCHS = 10
NON_LEARNING_MARGIN = 0.05

# ---- Multi-seed ----
# Semi RNG canonici; l'ordine per trial deriva dal trial ID (lib.search.engine).
SEEDS = [42, 123, 2024, 7, 99, 1337]

# Undici semi extra, distinti dai sei base: gli stessi della compressione multiseed, che
# refit_compress riceve da --extra-seeds. Nel codice li importa solo tools/loao_ray.py
# (LOAO a 17 semi).
SEEDS_EXTRA = [2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]

# ---- ASHA (rung unico al 4 seme) ----
# ASHA valuta i trial al quarto seme; il searcher marca PRUNED quelli non promossi.
# Reduction factor e frazione di promozione sono placeholder da ricalibrare.
MAX_T = 6
# grace_period e reduction_factor sono letti dal config YAML.

# ---- Soglia di anomalia (inferenza / LOAO) ----
# tau e' il percentile (1 - FPR_TARGET) degli score benigni di validazione.
FPR_TARGET = 0.01
