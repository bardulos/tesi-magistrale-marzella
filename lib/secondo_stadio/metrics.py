"""lib/secondo_stadio/metrics.py — metriche pure del secondo stadio (NumPy, deterministiche).

MCC_bal = MCC calcolato sulla matrice di confusione RISCALATA dai tassi (FPR, recall) —
prevalenza-invarianti — ai conteggi della prevalenza di RIFERIMENTO (nativa del val,
pi≈0,3709; GATE A D5: mai «prevalenza di deployment»). La robustezza rispetto alla
prevalenza di deploy e' la curva mcc_bal_pi (fase P7). Port da
repo_v5/lib/secondo_stadio/pseudo_anomalie.py:52-72 (equivalenza statica).

Convenzione di soglia: score >= tau (contratto legacy del PA; diverge dal `>` di
lib.dae.evaluate.binary_metrics_at_threshold). La divergenza NON e' inerte in generale
(misurato 2026-07-22, obiezione n.64): sul selection set di runs/secondo_stadio/fusione/
scores.npz (355.020 righe) i pareggi esatti float32 sono 204 su tau_hi_z e 167 su tau_z;
sono 0 sui CSV di test (6.834.656 confronti), dove le due convenzioni coincidono — inerte
SUL TEST, non in generale. V. nota operatore in lib/secondo_stadio/fusione.py (and_predict).
"""
from __future__ import annotations

import numpy as np

from lib.dae.threshold import compute_threshold
from lib.secondo_stadio import constants as C


def pearson_or_worst(curve_a: np.ndarray, curve_b: np.ndarray, worst: float = 1.0) -> float:
    """Pearson(curve_a, curve_b) NaN-safe per un obiettivo Optuna da MINIMIZZARE (WP-4_v2, Fase 2:
    Pearson(BCE-monitor, AUC-PR oracolo) — si cerca il piu' negativo/anti-correlato). Se una curva
    e' degenere (varianza nulla su un lato o <3 punti finiti in comune), ritorna `worst` (default
    1.0, il PEGGIOR Pearson possibile per la minimizzazione) invece di NaN: il chiamante resta
    valutabile (es. sugli altri semi di un trial) invece di propagare un NaN che farebbe fallire
    silenziosamente un aggregato (mediana/media)."""
    a = np.asarray(curve_a, dtype=np.float64)
    b = np.asarray(curve_b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return float(worst)
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def bce(y: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> float:
    """BCE dati puri sui punteggi sigmoid (clip per stabilita' log). Materiale grezzo per le
    curve di loss (WP-4_v2); stessa formula di tools/pa_es_history.py:74-78 (_bce locale,
    grandfathered li' — qui estratta perche' usata da >=2 file nuovi: pa_wp4v2_repass.py,
    pa_wp4v2_search.py)."""
    p = np.clip(np.asarray(p, np.float64), eps, 1.0 - eps)
    y = np.asarray(y, np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def mcc_bal(fpr: float, recall: float, n_b: int = C.N_B, n_a: int = C.N_A) -> float:
    """MCC dai tassi (FPR, recall) riscalati alla popolazione di riferimento (leak-free)."""
    tp = recall * n_a
    fn = (1 - recall) * n_a
    fp = fpr * n_b
    tn = (1 - fpr) * n_b
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def mcc_bal_pi(fpr: float, recall: float, pi: float) -> float:
    """MCC_bal a prevalenza arbitraria pi in (0,1), in forma chiusa dai tassi: la curva
    prevalenza-positiva della fase P7 (l'MCC riscalato e' invariante alla scala dei
    conteggi, quindi n_a=pi e n_b=1-pi bastano)."""
    return mcc_bal(fpr, recall, n_b=(1.0 - pi), n_a=pi)


def best_mcc_bal_standalone(clf_ben: np.ndarray, clf_att: np.ndarray,
                            p_grid: list | None = None) -> dict:
    """Sweep soglia (percentile p dei benigni-select) -> argmax MCC_bal standalone (no DAE).
    La soglia e' scelta UNA volta, deliberatamente, su un set grande (mai per-epoca: la
    patologia MCC-per-epoca del legacy e' documentata nel cap2)."""
    if p_grid is None:
        p_grid = C.P_GRID
    best = {"mcc_bal": -1.0, "fpr": 1.0, "recall": 0.0, "p_star": p_grid[0], "tau": 0.0}
    for p in p_grid:
        tau = float(np.percentile(clf_ben, p))
        fpr = float((clf_ben >= tau).mean())
        recall = float((clf_att >= tau).mean())
        mc = mcc_bal(fpr, recall)
        if mc > best["mcc_bal"]:
            best = {"mcc_bal": mc, "fpr": fpr, "recall": recall, "p_star": p, "tau": tau}
    return best


def and_at_oppoint(clf_neg, clf_pos, dae_neg, dae_pos, tau_clf: float,
                   tau_dae: float) -> tuple[float, float]:
    """FPR/DR del gate AND `(s_dae >= tau_dae) & (s_clf >= tau_clf)` al punto operativo dato
    (port da repo_v5 posthoc_metrics.py:99-103 — unica funzione portata da quel modulo)."""
    and_neg = (np.asarray(dae_neg) >= tau_dae) & (np.asarray(clf_neg) >= tau_clf)
    and_pos = (np.asarray(dae_pos) >= tau_dae) & (np.asarray(clf_pos) >= tau_clf)
    return float(and_neg.mean()), float(and_pos.mean())


def mcc_at_fpr(clf_ben_sel: np.ndarray, clf_att: np.ndarray,
               fpr_target: float = 0.01) -> dict:
    """MCC@FPR — punto operativo comparabile col primo stadio (cap2): tau = percentile
    (1 - fpr_target) dei benigni-select (compute_threshold, sorgente unica della soglia),
    tassi realizzati sul selection set, MCC sui conteggi REALI del set (coincide con
    l'MCC della matrice effettiva: fp = fpr*n_b, esatto a meno di 1 ulp float64)."""
    clf_ben_sel = np.asarray(clf_ben_sel)
    clf_att = np.asarray(clf_att)
    tau = compute_threshold(clf_ben_sel, fpr_target)
    fpr = float((clf_ben_sel >= tau).mean())
    recall = float((clf_att >= tau).mean())
    mcc = mcc_bal(fpr, recall, n_b=len(clf_ben_sel), n_a=len(clf_att))
    return {"tau": float(tau), "fpr": fpr, "recall": recall, "mcc": mcc}


def mcc_fpr_curve(clf_ben_sel: np.ndarray, clf_att: np.ndarray,
                  fpr_grid: np.ndarray) -> np.ndarray:
    """Curva MCC(FPR): MCC (conteggi reali del set) a ogni FPR target della griglia, via
    `mcc_at_fpr` (stessa soglia/convenzione di mcc_fpr1 → il punto a FPR=1% coincide con
    mcc@fpr1%)."""
    return np.array([mcc_at_fpr(clf_ben_sel, clf_att, float(f))["mcc"] for f in fpr_grid])


def partial_auc_mcc(clf_ben_sel: np.ndarray, clf_att: np.ndarray,
                    fpr_lo: float = C.FPM_FPR_LO, fpr_hi: float = C.FPM_FPR_HI,
                    n: int = C.FPM_CURVE_N) -> tuple[float, list, list]:
    """AUC_MCC PARZIALE (Orlova et al. 2025, arXiv 2507.09338) ristretta al band [fpr_lo,fpr_hi]
    e NORMALIZZATA per l'ampiezza = **MCC medio sul band** (scala MCC in [-1,1], confrontabile
    con mcc@fpr1%). Integrazione trapezoidale su griglia FPR uniforme. Ritorna
    (valore, griglia_fpr, curva_mcc): il valore per la selezione/stop, la curva per i dati grezzi."""
    g = np.linspace(fpr_lo, fpr_hi, n)
    m = mcc_fpr_curve(clf_ben_sel, clf_att, g)
    area = float(np.sum((m[:-1] + m[1:]) / 2.0 * np.diff(g)) / (fpr_hi - fpr_lo))
    return area, g.tolist(), m.tolist()
