"""lib/secondo_stadio/metrics.py — metriche pure del secondo stadio (NumPy, deterministiche).

MCC_bal = MCC calcolato sulla matrice di confusione RISCALATA dai tassi (FPR, recall) —
prevalenza-invarianti — ai conteggi della prevalenza di RIFERIMENTO (nativa del val,
pi≈0,3709; mai «prevalenza di deployment»). La robustezza rispetto alla
prevalenza di deploy e' la curva mcc_bal_pi.

Convenzione di soglia: score >= tau, come nel classificatore PA. Il DAE usa `>`;
i due confronti possono differire in presenza di valori esattamente uguali alla soglia.
"""
from __future__ import annotations

import numpy as np

from lib.dae.threshold import compute_threshold
from lib.secondo_stadio import constants as C


def pearson_or_worst(curve_a: np.ndarray, curve_b: np.ndarray, worst: float = 1.0) -> float:
    """Pearson NaN-safe; restituisce `worst` se ci sono meno di tre punti o varianza nulla."""
    a = np.asarray(curve_a, dtype=np.float64)
    b = np.asarray(curve_b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return float(worst)
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def bce(y: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> float:
    """BCE dei punteggi sigmoid, con clipping per stabilita' numerica."""
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
    """MCC_bal a prevalenza arbitraria, calcolato dai tassi."""
    return mcc_bal(fpr, recall, n_b=(1.0 - pi), n_a=pi)


def best_mcc_bal_standalone(clf_ben: np.ndarray, clf_att: np.ndarray,
                            p_grid: list | None = None) -> dict:
    """Seleziona la soglia percentile che massimizza MCC_bal standalone."""
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
    """FPR e DR della regola AND alle soglie indicate."""
    and_neg = (np.asarray(dae_neg) >= tau_dae) & (np.asarray(clf_neg) >= tau_clf)
    and_pos = (np.asarray(dae_pos) >= tau_dae) & (np.asarray(clf_pos) >= tau_clf)
    return float(and_neg.mean()), float(and_pos.mean())


def mcc_at_fpr(clf_ben_sel: np.ndarray, clf_att: np.ndarray,
               fpr_target: float = 0.01) -> dict:
    """Calcola MCC al target FPR usando una soglia dai benigni-select."""
    clf_ben_sel = np.asarray(clf_ben_sel)
    clf_att = np.asarray(clf_att)
    tau = compute_threshold(clf_ben_sel, fpr_target)
    fpr = float((clf_ben_sel >= tau).mean())
    recall = float((clf_att >= tau).mean())
    mcc = mcc_bal(fpr, recall, n_b=len(clf_ben_sel), n_a=len(clf_att))
    return {"tau": float(tau), "fpr": fpr, "recall": recall, "mcc": mcc}


def mcc_fpr_curve(clf_ben_sel: np.ndarray, clf_att: np.ndarray,
                  fpr_grid: np.ndarray) -> np.ndarray:
    """Calcola MCC ai target FPR della griglia."""
    return np.array([mcc_at_fpr(clf_ben_sel, clf_att, float(f))["mcc"] for f in fpr_grid])


def partial_auc_mcc(clf_ben_sel: np.ndarray, clf_att: np.ndarray,
                    fpr_lo: float = C.FPM_FPR_LO, fpr_hi: float = C.FPM_FPR_HI,
                    n: int = C.FPM_CURVE_N) -> tuple[float, list, list]:
    """MCC medio sul band FPR, integrato con trapezi; restituisce valore e curva."""
    g = np.linspace(fpr_lo, fpr_hi, n)
    m = mcc_fpr_curve(clf_ben_sel, clf_att, g)
    area = float(np.sum((m[:-1] + m[1:]) / 2.0 * np.diff(g)) / (fpr_hi - fpr_lo))
    return area, g.tolist(), m.tolist()
