"""lib/secondo_stadio/fusione.py — fusione DAE+PA: regola canonica a TRE rami
R_fus = R_AND u {logit_PA > theta_union} u {s_dae > tau_hi}.

Piano u = [log10(s_dae), logit_PA] z-scored sui benigni-train; l'AND (tau_z, theta_z) gata
DAE & PA (ramo di alta confidenza); l'unione aggiunge i benigni-liberi col logit alto fino al
budget FPR (ramo di caccia, calibrato LEAK-FREE sui benigni-select, held-out: nessun confine
cotto sul test); il ramo DAE-alto (tau_hi, soglia estrema sullo score DAE grezzo) recupera le
classi DAE-dominate che il gate logit esclude (SQL Injection, BF-Web). Due modalita'
operative (FUSION_MODES sotto): la Modalita' A, regola canonica della tesi e default, usa i
due rami AND e DAE-alto con un unico punto operativo calibrato senza etichette sui benigni
(theta_union=+inf); la Modalita' B aggiunge il ramo union a budget FPR, valutata e non
adottata, mantenuta nel codice e nei soglie.json per riprodurre i confronti riportati.
v1 (due rami AND+union, FUSION_PRESETS sotto) e' il caso degenere tau_hi=+inf: fuse_predict
copre tutto lo spettro con un'unica funzione.

logit_PA = log(s/(1-s)), s = punteggio sigmoide del PA. Tutto NumPy puro (riusabile in
deploy). v1 port quasi verbatim dal repository precedente (equivalenza statica); convenzioni
d'operatore documentate in and_predict. La regola a tre rami e' stata aggiunta il 2026-07-11
come codice riproducibile della regola canonica del cap3 (decisa il 2026-07-06), che prima
viveva solo in JSON congelati.
"""

import numpy as np

EPS = 1e-12

# Preset del knob --fpr: FPR target dell'unione (None = solo-AND, nessuna unione).
# DEPRECATO (v1 del 2026-07-02): superato dalla regola canonica a tre rami del 2026-07-06.
# Mantenuto perche' lo stage v1 `fusione` (fusione_eval.run_fusione_stage) lo istanzia e
# `ablazione.py` ne valida la fedelta' bit-exact (fusione/results.json congelato) — non usare
# per nuovi sviluppi: v. FUSION_MODES.
FUSION_PRESETS = {"andonly": None, "rec": 0.01, "high": 0.015}

# Ricetta canonica delle due modalita' operative (cap3). Codifica PERCENTILI/budget, non
# valori numerici: le soglie si
# derivano dai benigni-select del dominio in input (label-free, verificato trasferire a CSE e
# UNSW pur con soglie numeriche completamente diverse — la ricetta e' l'invariante).
#   Modalita' A (alta confidenza; regola canonica della tesi e default): AND stretto
#     (tau_dae@p97, alpha=0.003) + rete di sicurezza DAE-alto (tau_hi@p99.8); ramo union
#     spento (theta_union=+inf): due rami, un solo punto operativo.
#   Modalita' B (caccia; alternativa a tre rami, valutata e non adottata): AND alla soglia p99
#     (= la stessa del DAE-solo canonico) + union a budget FPR 1% + la stessa rete di
#     sicurezza DAE-alto. Resta nel codice e nei soglie.json per riprodurre i confronti.
FUSION_MODES = {
    "A": {"tau_dae_pct": 97.0, "alpha_and": 0.003, "fpr_target": None, "tau_hi_pct": 99.8},
    "B": {"tau_dae_pct": 99.0, "alpha_and": 0.005, "fpr_target": 0.01, "tau_hi_pct": 99.8},
}


def logit(s: np.ndarray) -> np.ndarray:
    """logit = log(s/(1-s)) con clip per evitare +-inf ai bordi 0/1."""
    s = np.clip(np.asarray(s, np.float64), EPS, 1.0 - EPS)
    return np.log(s / (1.0 - s))


# Piano u=[log10(s_dae), logit] z-scored sui benigni-train

def to_u(s_dae, pa_logit):
    """Punto 2D grezzo: [log10(s_dae) (clip), logit_PA]."""
    x = np.log10(np.clip(np.asarray(s_dae, np.float64), EPS, None))
    return np.column_stack([x, np.asarray(pa_logit, np.float64)])


def fit_standardizer(u_train):
    """(mu, sd) sui benigni-train; sd<EPS -> 1 (feature costante)."""
    mu = u_train.mean(0)
    sd = u_train.std(0)
    sd[sd < EPS] = 1.0
    return mu, sd


def to_z(u, mu, sd):
    return (np.asarray(u, np.float64) - mu) / sd


def tau_z_from_dae(tau_dae: float, mu, sd) -> float:
    """Soglia DAE tau (sorgente unica) portata nel piano z: (log10(tau_dae) - mu0)/sd0."""
    return float((np.log10(tau_dae) - mu[0]) / sd[0])


# AND: (z0 > tau_z) & (z1 > theta_z)

def and_theta_z(z_select, tau_z, alpha):
    """theta_z tale che FPR_AND(select) = alpha (tau_z fisso). Capped se alpha >= FPR del solo DAE."""
    z_select = np.asarray(z_select, np.float64)
    gated = z_select[z_select[:, 0] > tau_z, 1]
    n, n_gated = len(z_select), len(gated)
    if n_gated == 0:
        return float("inf"), True
    k = alpha * n
    if k >= n_gated:
        return float("-inf"), True
    return float(np.quantile(gated, 1.0 - k / n_gated)), False


def and_predict(z, tau_z, theta_z):
    # CONVENZIONE OPERATORE: qui (fusione, spazio z-score) si usa `>`; lib/secondo_stadio/metrics
    # usa `>=` su quantili di score GREZZI. NON "unificare" i due operatori: l'equivalenza e' a
    # misura-zero, non un teorema, e un tie esatto sul quantile operativo li separa.
    #
    # MISURATO il 2026-07-22 (obiezione n.64) -- l'equivalenza NON e' inerte come questa nota
    # affermava. Sul SELECTION set di runs/secondo_stadio/fusione/scores.npz (355.020 righe) i
    # pareggi esatti sono 204 su tau_hi_z (entrambe le modalita') e 167 su tau_z (Mod B). Non
    # spostano una decisione a soglia fissa, ma spostano la CALIBRAZIONE: portare questa funzione
    # a `>=` riproduce fpr_val dell'artefatto (0,004487071 / 0,011137401) e ROMPE fpr_test di
    # Mod B (0,047314478 contro 0,047313913 congelato). results_v2.json e' internamente misto --
    # fpr_val da `>=`, fpr_test da `>` -- quindi nessuna convenzione unica lo riproduce per intero.
    # Si tiene `>`: riproduce l'FPR di test pubblicato. Stessa scelta in inference/infer.py.
    # Sui CSV di inferenza i pareggi sono ZERO su 6,8 M di confronti: la decisione deployata e'
    # invariata in ogni caso.
    z = np.asarray(z, np.float64)
    return (z[:, 0] > tau_z) & (z[:, 1] > theta_z)


# Unione leak-free: soglia-logit 1D sui benigni-liberi

def calibrate_union_theta(z_select, tau_z, theta_z, fpr_target):
    """theta_union sul logit (z1) tale che FPR(R_AND u {z1>theta}) = fpr_target sui select.

    L'AND (tau_z, theta_z) resta FISSO. Si calibra la soglia-logit sui benigni NON gia' presi
    dall'AND (free), facendo passare i top-k con k = fpr_target*N - |AND|. Leak-free: calibrata
    sui select (held-out), non sul test. Se l'AND copre gia' il budget -> +inf (solo-AND).
    """
    z_select = np.asarray(z_select, np.float64)
    and_flag = and_predict(z_select, tau_z, theta_z)
    n = len(z_select)
    n_residual = fpr_target * n - int(and_flag.sum())
    free = ~and_flag
    if n_residual <= 0 or not free.any():
        return float("inf")
    logit_free = z_select[free, 1]
    q = 1.0 - n_residual / len(logit_free)
    return float(np.quantile(logit_free, np.clip(q, 0.0, 1.0)))


def fuse_predict(z, tau_z, theta_z, theta_union=float("inf"), tau_hi_z=float("inf")):
    """R_fus = R_AND u {z1 > theta_union} u {z0 > tau_hi_z} (regola canonica a tre rami).

    theta_union=+inf -> ramo union spento (v1 e' il caso degenere tau_hi_z=+inf; la
    Modalita' A e' il caso degenere theta_union=+inf). Confronti numpy con +inf sono sempre
    False: nessun ramo speciale necessario, la funzione unica copre tutto lo spettro."""
    z = np.asarray(z, np.float64)
    and_flag = and_predict(z, tau_z, theta_z)
    return and_flag | (z[:, 1] > theta_union) | (z[:, 0] > tau_hi_z)


# Metriche

def fpr_of(flag_neg) -> float:
    return float(np.asarray(flag_neg).mean())


def recall_of(flag_pos) -> float:
    return float(np.asarray(flag_pos).mean())


def perclass_dr(flag_pos, classes) -> dict:
    """DR per classe d'attacco da un flag booleano (esclude 'Benign')."""
    classes = np.asarray(classes)
    return {str(c): float(flag_pos[classes == c].mean())
            for c in np.unique(classes) if c != "Benign"}


def build_fusion(s_dae_train, logit_train, s_dae_select, logit_select,
                 tau_dae, fpr_target, alpha_and, tau_hi=None):
    """Costruisce i parametri di fusione leak-free dai benigni train/select.

    alpha_and = FPR base dell'AND (es. 0.005); fpr_target = FPR totale dell'unione (None=solo-AND);
    tau_hi = soglia estrema sullo score DAE grezzo per il terzo ramo (None=ramo spento, caso
    degenere v1). tau_hi e tau_dae sono percentili dello score DAE GREZZO (stessa convenzione,
    portati nel piano z da tau_z_from_dae); theta_z vive nel piano z-scored; theta_union e' sul
    logit (anch'esso nel piano z, coerente con fuse_predict). Ritorna dict {mu, sd, tau_z,
    theta_z, theta_union, tau_hi_z, capped, tau_hi, alpha_and, fpr_target}.
    """
    u_train = to_u(s_dae_train, logit_train)
    mu, sd = fit_standardizer(u_train)
    tau_z = tau_z_from_dae(tau_dae, mu, sd)
    z_select = to_z(to_u(s_dae_select, logit_select), mu, sd)
    theta_z, capped = and_theta_z(z_select, tau_z, alpha_and)
    if fpr_target is None:
        theta_union = float("inf")
    else:
        theta_union = calibrate_union_theta(z_select, tau_z, theta_z, fpr_target)
    tau_hi_z = tau_z_from_dae(tau_hi, mu, sd) if tau_hi is not None else float("inf")
    return {"mu": mu, "sd": sd, "tau_z": tau_z, "theta_z": theta_z,
            "theta_union": theta_union, "tau_hi_z": tau_hi_z, "tau_hi": tau_hi,
            "alpha_and": alpha_and, "fpr_target": fpr_target, "capped": bool(capped)}


def apply_fusion(params, s_dae, pa_logit):
    """Applica la regola di fusione a (s_dae, logit_PA) -> flag booleano."""
    z = to_z(to_u(s_dae, pa_logit), params["mu"], params["sd"])
    return fuse_predict(z, params["tau_z"], params["theta_z"], params["theta_union"],
                        params.get("tau_hi_z", float("inf")))
