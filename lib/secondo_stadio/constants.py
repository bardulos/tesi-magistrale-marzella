"""Costanti del secondo stadio; dimensioni feature, batch e semi hanno sorgente unica."""
from lib.dae.constants import BATCH_FIXED, SEEDS  # noqa: F401 (re-export sorgente unica)
from lib.utils import N_FEATURES

# ---- Spazio phi = [z_s, r_w] ----
Z_DIM = 16
R_DIM = N_FEATURES
PHI_DIM = Z_DIM + R_DIM

# ---- Prevalenza di riferimento per il rate-scaling di MCC_bal ----
# Terminologia: prevalenza di riferimento, non di deployment. La curva MCC_bal(pi)
# descrive la robustezza a prevalenze diverse.
N_B = 1_775_098
N_A = 1_046_755
PI_REF = N_A / (N_A + N_B)

P_GRID = [90.0, 92.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 99.3, 99.5, 99.7, 99.9]  # standalone MCC sweep

# ---- Split di validazione ----
# Split fisso, non per-seme. Il test non entra nella ricerca.
SPLIT_SEED = 0
VAL_FRAC = 0.20          # 80% train / 20% selection
ES_N_ATTACKS = 50_000    # Subsample per-epoca; finale sugli attacchi completi.

# ---- Semi ----
# Gli otto semi PA includono i sei canonici e due nuovi, distinti dai SEEDS_EXTRA DAE.
SEEDS_PA = list(SEEDS) + [2036, 2037]
# Semi extra per la compressione.
SEEDS_PA_EXTRA = list(range(2038, 2049))

# ---- Rung ASHA (time_attr=seeds_completed) ----
# Rung nei config: ricerca supervisionata 4/2, PA 4/1.5. `max_t` deriva dai semi.

# ---- Parametri fissi MLP ----
# L2 e' fissa; il dropout e' il regolarizzatore cercato. Pressione sul limite 0.3
# nella fANOVA segnala di rivedere la regolarizzazione.
L2_FIXED_S2 = 1.0819231482664046e-06
MLP_MAX_EPOCHS = 100
MLP_PATIENCE = 15        # Early stopping su AUC-PR, mai su MCC.

# ---- Ricerca supervisionata (ceiling): 5 dim, imbuto espansione->compressione ----
# Il primo hidden layer espande l'input; i successivi comprimono con rapporto fisso.
SUP_N_LAYERS = (2, 3, 4)
SUP_EXPANSION_LOW, SUP_EXPANSION_HIGH = 1.5, 4.0
SUP_H1_LOW = round(SUP_EXPANSION_LOW * PHI_DIM)
SUP_H1_HIGH = round(SUP_EXPANSION_HIGH * PHI_DIM)
SUP_RATIO_LOW, SUP_RATIO_HIGH = 0.5, 0.7
SUP_LR_LOW, SUP_LR_HIGH = 2e-4, 5e-3
SUP_DROPOUT_LOW, SUP_DROPOUT_HIGH = 0.0, 0.3

# ---- Ricerca PA (classificatore) — 8 dim ----
# La PA cerca architettura e generazione di pseudo-anomalie. Il learning rate ha un
# intervallo piu' stretto e spostato in basso rispetto al supervisionato, il cui
# ottimo cade sul limite inferiore 2e-4.
PA_LR_LOW, PA_LR_HIGH = 1e-4, 1e-3
DELTA_LOW, DELTA_HIGH = 0.0, 1.0
ALPHA_LO_LOW, ALPHA_LO_HIGH = 1.0, 2.0
ALPHA_HI_LOW, ALPHA_HI_HIGH = 2.0, 5.0

# ---- fp-mining (valori canonici) ----
FPM_K = 4          # Numero di copie extra per falso positivo minato.
FPM_K_MIN = 0.002  # Soglia applicata al delta pAUC_MCC.
FPM_N_MAX = 5
# --- metrica partial_auc_mcc: AUC_MCC PARZIALE (Orlova et al. 2025, arXiv 2507.09338v2)
#     ristretta al band operativo di FPR, NORMALIZZATA = MCC medio sul band (scala MCC). ---
FPM_FPR_LO = 0.005
FPM_FPR_HI = 0.015
FPM_CURVE_N = 41
FPM_PLOT_FPR_LO = 0.001
FPM_PLOT_FPR_HI = 0.030
FPM_PLOT_N = 59
FPM_STOP_SIGMA = 1.0    # Auto-stop quando il delta scende sotto c*sigma_m.

ALPHA_AND = 0.005

# ---- Pseudo-anomalie di monitoraggio per l'early stopping label-free della PA ----
# Parametri delle pseudo usate solo per il monitor BCE, non per il training.
MON_SEED_V2 = 20260710
PSEUDO_MONITOR_SEED = 20260711
WP4V2_MON_N = 200_000
WP4V2_ALPHA_LO_LOW, WP4V2_ALPHA_LO_HIGH = 1.0, 2.5
WP4V2_ALPHA_HI_LOW, WP4V2_ALPHA_HI_HIGH = 1.2, 5.0
WP4V2_DELTA_LOW, WP4V2_DELTA_HIGH = 0.0, 1.0
WP4V2_RUNG_1, WP4V2_RUNG_2 = 7, 14
WP4V2_OPTUNA_SEED = 20260712
WP4V2_N_STARTUP_TRIALS = 100
