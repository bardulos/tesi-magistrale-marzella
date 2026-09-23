"""lib/secondo_stadio/ablazione.py — stage `ablazione`: confronto delle regole di decisione.

Confronta QUATTRO regole sugli STESSI punteggi (s_DAE, logit_PA) prodotti dallo stage
`fusione` (dump scores npz; stesso split leak-free, MAI ri-splittato/riaddestrato), ognuna
calibrata LEAK-FREE sui benigni-select a una griglia di FPR-bersaglio e valutata su test:

  1. DAE-solo    : z0 > tau                             (primo stadio da solo)
  2. logit-solo  : z1 > theta                           (il "solo classificatore")
  3. AND-solo    : (z0 > tau_z) & (z1 > theta_z)        (nucleo conservativo, tau_z dal DAE)
  4. AND∪logit   : AND(alpha_and) u {z1 > theta_union}  (fusione canonica)

Calc-only (NumPy + lib.secondo_stadio.fusione + mcc_bal; NESSUN matplotlib: la figura vive
in lib/plotting/secondo_stadio_plots.py). MAI calibrare sul test: le soglie si fissano sui
select-benigni; lo sweep su test e' RENDICONTAZIONE (interpolazione alle ancore comuni),
non ri-calibrazione. Gate di fedelta' bit-exact: riproduce i preset canonici di
fusione/results.json (STOP se Delta > 1e-6, non riconciliare). Verdetto sulla soglia
OPERATIVA |Delta recall| < delta_operativa per ancora; CI di Wilson caratterizzanti
(per-classe, n basso), MAI gate — a n grande la significativita' degenererebbe.

Knob scientifici (griglia/ancore/delta/criteri) OBBLIGATORI nel config YAML (nessun default
silenzioso): la run e' legata univocamente a configs/secondo_stadio/ablazione.yaml.
Port da repo_v5 ablazione.py:49-388; la sintesi del verdetto e' riscritta FATTUALE
(regimi/ancore/scarti misurati), senza le conclusioni della run legacy.
"""
import csv
import json
import logging
from pathlib import Path

import numpy as np

from lib.secondo_stadio import fusione as F
from lib.secondo_stadio.metrics import mcc_bal

log = logging.getLogger(__name__)

# Classi-chiave (spelling esatto delle etichette) e regole — strutturali, non knob.
HOIC = "DDOS_attack-HOIC"
INFIL = "Infilteration"            # (sic, con la 'e': spelling storico del dataset)
RULES = ["DAE-solo", "logit-solo", "AND-solo", "AND∪logit"]


# metriche
def _mcc(recall, fpr, n_a, n_b):
    """MCC dai tassi: riusa metrics.mcc_bal (forma chiusa identica) con guardia NaN per le
    ancore non racchiuse (mcc_bal su input NaN tornerebbe 0.0; qui torna NaN)."""
    if not (np.isfinite(recall) and np.isfinite(fpr)):
        return float("nan")
    return mcc_bal(fpr, recall, n_b=n_b, n_a=n_a)


def _wilson(p, n, z=1.96):
    """Intervallo di Wilson 95% (lo, hi) per una proporzione p su n campioni."""
    if n <= 0 or not np.isfinite(p):
        return float("nan"), float("nan")
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return float(c - h), float(c + h)


def _ci_disjoint(ci_a, ci_b):
    """True se i due intervalli (lo,hi) sono disgiunti; None se non valutabile."""
    (alo, ahi), (blo, bhi) = ci_a, ci_b
    if any(not np.isfinite(v) for v in (alo, ahi, blo, bhi)):
        return None
    return bool(ahi < blo or bhi < alo)


# sweep leak-free
def _eval_flag(f_bt, f_at, cls_at, n_b, n_a):
    """Metriche da flag booleani su test-benigni (f_bt) e test-attacchi (f_at)."""
    fpr = float(f_bt.mean())
    recall = float(f_at.mean())
    return {"fpr_test": fpr, "recall": recall,
            "mcc_nat": _mcc(recall, fpr, n_a, n_b), "mcc_bil": _mcc(recall, fpr, 1.0, 1.0),
            "perclass_dr": F.perclass_dr(f_at, cls_at)}


def _sweep_rule(rule, z_sel, z_bt, z_at, cls_at, tau_z, theta_z_and, grid):
    """Calibra `rule` su select a ogni bersaglio FPR del grid (LEAK-FREE), valuta su test.
    Tutte le regole usano lo STESSO estimatore np.quantile e la STESSA definizione di
    bersaglio (frazione di select-benigni segnalati = t): confronto equo sullo stesso asse."""
    n_b, n_a = len(z_bt), len(z_at)
    pts = []
    for t in grid:
        capped = False
        if rule == "DAE-solo":
            thr = float(np.quantile(z_sel[:, 0], 1.0 - t))
            f_bt, f_at = z_bt[:, 0] > thr, z_at[:, 0] > thr
        elif rule == "logit-solo":
            thr = float(np.quantile(z_sel[:, 1], 1.0 - t))
            f_bt, f_at = z_bt[:, 1] > thr, z_at[:, 1] > thr
        elif rule == "AND-solo":
            thr, capped = F.and_theta_z(z_sel, tau_z, t)            # tau_z FISSO (DAE p99)
            f_bt = F.and_predict(z_bt, tau_z, thr)
            f_at = F.and_predict(z_at, tau_z, thr)
        elif rule == "AND∪logit":
            thr = F.calibrate_union_theta(z_sel, tau_z, theta_z_and, t)   # AND fisso
            f_bt = F.fuse_predict(z_bt, tau_z, theta_z_and, thr)
            f_at = F.fuse_predict(z_at, tau_z, theta_z_and, thr)
        else:
            raise ValueError(rule)
        m = _eval_flag(f_bt, f_at, cls_at, n_b, n_a)
        m.update({"fpr_target_val": t, "theta_val": thr, "capped": bool(capped)})
        pts.append(m)
    return pts


def _curve(pts, key, exclude_capped=True):
    """(fpr_test, value[key]) ordinati per fpr crescente, dedup x (np.interp vuole xp
    crescente). key 'pc:<classe>' -> DR di classe. Esclude i punti capped (degeneri)."""
    rows = [p for p in pts if not (exclude_capped and p["capped"])]
    xs = np.array([p["fpr_test"] for p in rows], float)
    if key.startswith("pc:"):
        cls = key[3:]
        ys = np.array([p["perclass_dr"].get(cls, np.nan) for p in rows], float)
    else:
        ys = np.array([p[key] for p in rows], float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if len(xs):
        keep = np.concatenate([[True], np.diff(xs) > 0])
        xs, ys = xs[keep], ys[keep]
    return xs, ys


def _interp_at(pts, key, anchor):
    """Interpola value[key] all'ancora di FPR-test; NaN (niente estrapolazione) se non
    racchiusa."""
    xs, ys = _curve(pts, key)
    if len(xs) >= 2 and xs.min() <= anchor <= xs.max():
        return float(np.interp(anchor, xs, ys)), True
    return float("nan"), False


# gate di fedelta'
def _gate(canon, sdae_tr, lg_tr, sdae_sel, lg_sel, sdae_bt, lg_bt, sdae_at, lg_at, cls_at,
          tau_dae, alpha_and, tol=1e-6):
    """Riproduce i preset canonici via build_fusion+apply_fusion e confronta con
    fusione/results.json (tol 1e-6). Se fallisce -> STOP (non riconciliare)."""
    checks = []
    for name, ft in F.FUSION_PRESETS.items():
        prm = F.build_fusion(sdae_tr, lg_tr, sdae_sel, lg_sel, tau_dae, ft, alpha_and)
        fpr = float(F.apply_fusion(prm, sdae_bt, lg_bt).mean())
        f_at = F.apply_fusion(prm, sdae_at, lg_at)
        rec = float(f_at.mean())
        hoic = float(f_at[cls_at == HOIC].mean()) if (cls_at == HOIC).any() else float("nan")
        c = canon["presets"][name]
        tu, tuc = prm["theta_union"], c["theta_union"]
        if not np.isfinite(tu):
            # canonico senza theta_union effettiva: combacia se anche tuc e' assente/non-finito
            tu_ok = tuc is None or not np.isfinite(tuc)
        elif isinstance(tuc, (int, float)):
            tu_ok = abs(tu - tuc) < tol
        else:
            tu_ok = False
        d_fpr = abs(fpr - c["fused_fpr_test"])
        d_rec = abs(rec - c["fused_recall_test"])
        hoic_c = c["perclass_dr_fused"].get(HOIC)
        d_hoic = (abs(hoic - hoic_c) if (hoic_c is not None and np.isfinite(hoic))
                  else 0.0)
        ok = bool(d_fpr < tol and d_rec < tol and tu_ok and d_hoic < tol)
        checks.append({"preset": name, "fpr_test": fpr, "recall": rec,
                       "theta_union": (None if not np.isfinite(tu) else tu),
                       "d_fpr": d_fpr, "d_recall": d_rec, "d_hoic": d_hoic,
                       "theta_union_ok": bool(tu_ok), "ok": ok})
    return checks


# driver
def _write_csv(sweep, out_path):
    cols = ["rule", "fpr_target_val", "theta_val", "capped", "fpr_test", "recall",
            "mcc_nat", "mcc_bil", "dr_HOIC", "dr_Infilteration"]
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in RULES:
            for p in sweep[r]:
                w.writerow([r, p["fpr_target_val"], p["theta_val"], p["capped"],
                            p["fpr_test"], p["recall"], p["mcc_nat"], p["mcc_bil"],
                            p["perclass_dr"].get(HOIC, float("nan")),
                            p["perclass_dr"].get(INFIL, float("nan"))])


def run_ablazione(scores_path, canonico_path, out_dir, grid, anchors, delta_op, criteri):
    """Core dell'ablazione (riusabile/testabile). out_dir deve gia' esistere (shell-first).

    Regimi strutturali restituiti in results['regimi_strutturali'] — bound INDIPENDENTI
    dalla griglia di soglie:
      - logit_solo_floor_fpr_test: frazione di test-benigni col logit oltre il MAX logit
        dei benigni-select (la soglia-quantile tende a max(select) per t->0); e' l'FPR-test
        minimo strutturalmente raggiungibile da logit-solo;
      - and_solo_cap_fpr_test: FPR-test del solo gate DAE (z0 > tau_z) — cap di AND-solo,
        indipendente dal ramo logit;
      - and_union_floor_fpr_test: FPR-test dell'AND puro a theta_z_and (theta_union=+inf,
        nessun contributo dell'unione) — pavimento di AND∪logit.
    """
    out_dir = Path(out_dir)
    # allow_pickle: cls_at e' un array object di etichette-stringa prodotto dalla NOSTRA
    # pipeline (fusione_eval.dump_scores_npz) -> artefatto fidato interno, non input esterno.
    d = np.load(scores_path, allow_pickle=True)
    sdae_tr, lg_tr = d["sdae_tr"], d["lg_tr"]
    sdae_sel, lg_sel = d["sdae_sel"], d["lg_sel"]
    sdae_bt, lg_bt = d["sdae_bt"], d["lg_bt"]
    sdae_at, lg_at = d["sdae_at"], d["lg_at"]
    cls_at = np.array([c.decode() if isinstance(c, (bytes, bytearray)) else str(c)
                       for c in d["cls_at"]])
    tau_dae, alpha_and = float(d["tau_dae"]), float(d["alpha_and"])
    canon = json.loads(Path(canonico_path).read_text())
    n_b, n_a = len(sdae_bt), len(sdae_at)
    log.info("punteggi: train-ben %d, select-ben %d, test-ben %d, test-att %d; "
             "tau_dae=%.6g alpha_and=%.4g", len(sdae_tr), len(sdae_sel), n_b, n_a,
             tau_dae, alpha_and)

    u, cnt = np.unique(cls_at, return_counts=True)
    supports = {str(k): int(v) for k, v in zip(u, cnt)}

    # --- GATE di fedelta' (STOP se fallisce, non riconciliare) ---
    checks = _gate(canon, sdae_tr, lg_tr, sdae_sel, lg_sel, sdae_bt, lg_bt, sdae_at,
                   lg_at, cls_at, tau_dae, alpha_and)
    for ch in checks:
        log.info("[gate] %-8s fpr=%.6f (D%.1e) recall=%.6f (D%.1e) HOIC D%.1e "
                 "thu_ok=%s -> %s", ch["preset"], ch["fpr_test"], ch["d_fpr"],
                 ch["recall"], ch["d_recall"], ch["d_hoic"], ch["theta_union_ok"],
                 "OK" if ch["ok"] else "FAIL")
    if not all(ch["ok"] for ch in checks):
        raise SystemExit("GATE FALLITO: l'ablazione non riproduce i preset canonici a "
                         "1e-6. STOP.")

    # --- standardizer (train) + z (riproduce build_fusion) ---
    mu, sd = F.fit_standardizer(F.to_u(sdae_tr, lg_tr))
    tau_z = F.tau_z_from_dae(tau_dae, mu, sd)
    z_sel = F.to_z(F.to_u(sdae_sel, lg_sel), mu, sd)
    z_bt = F.to_z(F.to_u(sdae_bt, lg_bt), mu, sd)
    z_at = F.to_z(F.to_u(sdae_at, lg_at), mu, sd)
    theta_z_and, capped_and = F.and_theta_z(z_sel, tau_z, alpha_and)
    log.info("piano z: tau_z=%.6f theta_z(AND@alpha=%.4g)=%.6f (capped=%s)",
             tau_z, alpha_and, theta_z_and, capped_and)

    # --- regimi strutturali (bound INDIPENDENTI dalla griglia; semantica in docstring) ---
    logit_floor = float((z_bt[:, 1] > z_sel[:, 1].max()).mean())
    and_cap = float((z_bt[:, 0] > tau_z).mean())
    and_union_floor = float(F.and_predict(z_bt, tau_z, theta_z_and).mean())
    regimi = {"logit_solo_floor_fpr_test": logit_floor, "and_solo_cap_fpr_test": and_cap,
              "and_union_floor_fpr_test": and_union_floor}
    log.info("[regimi] logit-solo floor=%.4f%% | AND-solo cap=%.4f%% | AND-union "
             "floor=%.4f%%", logit_floor * 100, and_cap * 100, and_union_floor * 100)

    # --- sweep leak-free delle 4 regole ---
    sweep = {r: _sweep_rule(r, z_sel, z_bt, z_at, cls_at, tau_z, theta_z_and, grid)
             for r in RULES}

    # --- bracketing per regola (niente estrapolazione) ---
    bracket = {}
    for r in RULES:
        xs, _ = _curve(sweep[r], "recall")
        lo, hi = (float(xs.min()), float(xs.max())) if len(xs) else (float("nan"),
                                                                     float("nan"))
        per_anchor = {f"{a:.4f}": bool(len(xs) >= 2 and lo <= a <= hi) for a in anchors}
        bracket[r] = {"fpr_test_min": lo, "fpr_test_max": hi,
                      "anchor_bracketed": per_anchor}
        for a, ok in per_anchor.items():
            if not ok:
                log.warning("[bracket] %-10s NON racchiude FPR-test %s (range "
                            "[%.4f,%.4f]).", r, a, lo, hi)

    # --- tabella a FPR appaiato + testa-a-testa logit-solo vs AND∪logit ---
    def _compare(v_logit, v_fus, n, soglia=None):
        delta = ((v_fus - v_logit) if (np.isfinite(v_logit) and np.isfinite(v_fus))
                 else float("nan"))
        ci_l, ci_f = _wilson(v_logit, n), _wilson(v_fus, n)
        out = {"logit_solo": v_logit, "fusione": v_fus, "delta_fus_meno_logit": delta,
               "n": int(n), "wilson95_logit": list(ci_l), "wilson95_fusione": list(ci_f),
               "ci_disgiunti": _ci_disjoint(ci_l, ci_f)}
        if soglia is not None:
            out["operativamente_viabile"] = (bool(abs(delta) < soglia)
                                             if np.isfinite(delta) else None)
        return out

    anchor_table = []
    for a in anchors:
        per_rule = {}
        for r in RULES:
            rec_v, brk = _interp_at(sweep[r], "recall", a)
            hoic_v, _ = _interp_at(sweep[r], "pc:" + HOIC, a)
            infil_v, _ = _interp_at(sweep[r], "pc:" + INFIL, a)
            per_rule[r] = {"recall": rec_v, "bracketed": brk,
                           "mcc_nat": _mcc(rec_v, a, n_a, n_b),
                           "dr_HOIC": hoic_v, "dr_Infilteration": infil_v}
        h2h = {
            "recall": _compare(per_rule["logit-solo"]["recall"],
                               per_rule["AND∪logit"]["recall"], n_a, soglia=delta_op),
            "HOIC": _compare(per_rule["logit-solo"]["dr_HOIC"],
                             per_rule["AND∪logit"]["dr_HOIC"], supports.get(HOIC, 0)),
            "Infilteration": _compare(per_rule["logit-solo"]["dr_Infilteration"],
                                      per_rule["AND∪logit"]["dr_Infilteration"],
                                      supports.get(INFIL, 0)),
        }
        anchor_table.append({"fpr_test_anchor": a, "rules": per_rule,
                             "head_to_head_logit_vs_fusione": h2h})

    # --- verdetto: criteri A/B PRE-REGISTRATI sulla soglia OPERATIVA ---
    cls_per_anchor, deltas_both = {}, []
    for row in anchor_table:
        a = row["fpr_test_anchor"]
        rc = row["head_to_head_logit_vs_fusione"]["recall"]
        rl, rf = rc["logit_solo"], rc["fusione"]
        if np.isfinite(rl) and np.isfinite(rf):
            dd = rf - rl
            deltas_both.append((a, dd))
            if abs(dd) < delta_op:
                cls_per_anchor[f"{a:.4f}"] = "equivalenti"
            elif dd > 0:
                cls_per_anchor[f"{a:.4f}"] = "fusione_meglio"
            else:
                cls_per_anchor[f"{a:.4f}"] = "logit_meglio"
        elif (not np.isfinite(rl)) and np.isfinite(rf):
            cls_per_anchor[f"{a:.4f}"] = "logit_assente"
        else:
            cls_per_anchor[f"{a:.4f}"] = "non_valutabile"
    reachable_both = [a for a, _ in deltas_both]
    logit_absent = [a for a in anchors if cls_per_anchor[f"{a:.4f}"] == "logit_assente"]
    max_abs = max((abs(dd) for _, dd in deltas_both), default=float("nan"))
    # (A) semplificazione viabile: logit-solo raggiunge TUTTE le ancore e le eguaglia entro
    #     delta_op ovunque. (B) la fusione DOMINA la recall a FPR appaiato in >=1 ancora.
    crit_A = bool(reachable_both and not logit_absent
                  and all(abs(dd) < delta_op for _, dd in deltas_both))
    crit_B = bool(any(dd >= delta_op for _, dd in deltas_both))
    # Sintesi FATTUALE dai valori misurati (nessuna conclusione precotta: quella e' prosa
    # dell'autore, informata da criteri A/B, regimi e classificazione per ancora).
    sintesi = (
        f"Regimi misurati: logit-solo floor FPR-test {logit_floor * 100:.2f}%; AND-solo cap "
        f"{and_cap * 100:.2f}%; AND-union floor {and_union_floor * 100:.2f}%. Ancore coperte "
        f"da entrambe le regole del testa-a-testa: {[f'{a * 100:.2f}%' for a in reachable_both]}; "
        f"ancore senza logit-solo: {[f'{a * 100:.2f}%' for a in logit_absent]}. "
        f"Max |Delta recall(fusione-logit)| dove entrambe operano: "
        f"{(max_abs * 100 if np.isfinite(max_abs) else float('nan')):.2f} pp "
        f"(soglia operativa {delta_op * 100:.2f} pp). Criterio A={crit_A}, criterio B={crit_B}.")
    verdetto = {
        "criterio_A_semplificazione_viabile": crit_A,
        "criterio_B_fusione_domina_recall_a_fpr_appaiato": crit_B,
        "soglia_operativa_delta_recall": delta_op,
        "logit_solo_floor_fpr_test": logit_floor,
        "and_solo_cap_fpr_test": and_cap,
        "ancore_non_raggiungibili_da_logit_solo": logit_absent,
        "ancore_reachable_da_entrambe": reachable_both,
        "classificazione_per_ancora": cls_per_anchor,
        "max_abs_delta_recall_dove_entrambe": max_abs,
        "sintesi": sintesi,
        "nota": ("Verdetto sulla soglia operativa; CI di Wilson caratterizzanti "
                 "(per-classe/n basso), non gate."),
    }
    log.info("[verdetto] A_viabile=%s B_fus_domina_recall=%s | logit_floor=%.2f%% "
             "logit_assente@%s | max|D|dove_entrambe=%.2fpp", crit_A, crit_B,
             logit_floor * 100, [f"{a * 100:.1f}%" for a in logit_absent], max_abs * 100)

    # --- artefatti (NIENTE figura: la disegna lib/plotting/secondo_stadio_plots.py) ---
    results = {
        "descrizione": ("Ablazione regole di fusione cap3 (CSE), calibrazione leak-free "
                        "val->test."),
        "criteri_preregistrati": criteri,
        "regole": RULES,
        "grid_val_fpr": list(grid),
        "ancore_fpr_test": list(anchors),
        "support_classi_test": supports,
        "regimi_strutturali": regimi,
        "gate_fedelta_1e-6": checks,
        "bracketing": bracket,
        "sweep": sweep,
        "tabella_fpr_appaiato": anchor_table,
        "verdetto": verdetto,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2,
                                                     ensure_ascii=False))
    _write_csv(sweep, out_dir / "frontiera.csv")
    log.info("scritti: %s/{results.json, frontiera.csv}", out_dir)


def run_ablazione_stage(cfg: dict) -> None:
    """Stage `ablazione`: consuma gli score dumpati dallo stage `fusione` (scores_npz) +
    il results.json canonico, confronta le 4 regole leak-free, scrive results.json +
    frontiera.csv.

    Knob scientifici OBBLIGATORI (nessun default silenzioso): grid_val_fpr,
    ancore_fpr_test_basse, ancore_preset_canonici, delta_operativa, criteri. Le ancore
    rec/high si DERIVANO dagli FPR-test realizzati canonici (fused_fpr_test): coincidono
    coi punti di deploy.
    """
    out_dir = Path(cfg["out_dir"])
    if not out_dir.exists():
        raise FileNotFoundError(f"out_dir non esiste (shell-first): {out_dir}")
    scores_path = Path(cfg["scores_npz"])
    if not scores_path.exists():
        raise FileNotFoundError(f"scores_npz assente: {scores_path}. Eseguire prima lo "
                                f"stage 'fusione' con dump_scores_npz.")
    canonico = Path(cfg["canonico"])
    missing = [k for k in ("grid_val_fpr", "ancore_fpr_test_basse",
                           "ancore_preset_canonici", "delta_operativa", "criteri")
               if k not in cfg]
    if missing:
        raise SystemExit(f"config ablazione: chiavi obbligatorie mancanti {missing} "
                         f"(niente default).")

    grid = [float(g) for g in cfg["grid_val_fpr"]]
    delta_op = float(cfg["delta_operativa"])
    criteri = cfg["criteri"]
    canon = json.loads(canonico.read_text())
    preset_fprs = {p: float(canon["presets"][p]["fused_fpr_test"])
                   for p in cfg["ancore_preset_canonici"]}
    anchors = sorted(set([float(a) for a in cfg["ancore_fpr_test_basse"]]
                         + list(preset_fprs.values())))
    log.info("ancore FPR-test: basse %s + preset-derivate %s -> %s",
             cfg["ancore_fpr_test_basse"], preset_fprs, anchors)
    run_ablazione(scores_path, canonico, out_dir, grid, anchors, delta_op, criteri)
