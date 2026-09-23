"""lib/secondo_stadio/fusione_eval.py — stage `fusione` (v1) e `fusione_canonica` (v2).

`fusione` (v1, storico): valutazione della regola R_fus = R_AND u {logit>theta} ai preset
FPR (andonly/rec/high) con theta calibrato LEAK-FREE sui benigni-select, misurata su TEST
(GATE E). Riporta per preset FPR/recall di AND e fusione + DR per classe di ENTRAMBE. Dump
opzionale degli score (scores npz): input dell'ablazione, delle figure e (quando presente)
dello stage `fusione_canonica`.

`fusione_canonica` (v2, canonico dal 2026-07-06): regola a TRE rami, due modalita' operative
A/B (F.FUSION_MODES: A = regola canonica della tesi e default, due rami con punto operativo
unico; B = alternativa a tre rami, valutata e non adottata) — v. `run_fusione_canonica_stage`.

Adattamenti dal legacy (fusione_eval v5:55-135): la cache v6 ha z_s GIA' standardizzato
(niente ricalcolo mu_z/sd_z, che in v5 era necessario e qui sarebbe un bug); accesso dati
SOLO via data.load_phi_cache/load_test_context; niente emissione deploy (fuori scope cap3).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from lib.secondo_stadio import fusione as F
from lib.secondo_stadio.data import (load_phi_cache, load_tau_dae,
                                     load_test_context, predict_phi)
from lib.secondo_stadio.metrics import mcc_bal

log = logging.getLogger(__name__)

_SCORE_KEYS = ("sdae_tr", "lg_tr", "sdae_sel", "lg_sel",
              "sdae_bt", "lg_bt", "sdae_at", "lg_at")


def compute_scores(cfg: dict) -> dict:
    """Calcola gli score di piano (s_dae, logit_PA) per benigni-train/select/test e
    attacchi-test dalla cache phi + PA congelato. Forward-only, nessun training. Riusata da
    `run_fusione_stage` (v1) e `run_fusione_canonica_stage` (v2, quando `scores_npz` non e'
    gia' disponibile — es. UNSW, dove il dump non esiste ancora).

    cfg: cache_dir, labels_val, labels_test, l1l2_csv, pa_weights, model_config, mlp_fixed.
    tau_dae (dal l1l2_csv, sorgente unica v1) e' incluso nel dict di ritorno per retro-
    compatibilita' col chiamante v1; `fusione_canonica` lo ignora (usa percentili live, v.
    F.FUSION_MODES: verificato bit-exact equivalente a tau_dae@p99 per la Modalita' B).
    """
    from lib.secondo_stadio.fp_mining import _load_model

    tau_dae = load_tau_dae(cfg["l1l2_csv"])
    data = load_phi_cache(cfg["cache_dir"], cfg["labels_val"])
    model = _load_model(cfg["pa_weights"], cfg["model_config"], cfg["mlp_fixed"])

    ben_train_abs = data["ben_pos"][data["idx_train"]]
    ben_sel_abs = data["ben_pos"][data["idx_select"]]
    sdae_tr = data["dae_score"][ben_train_abs]
    sdae_sel = data["dae_score"][ben_sel_abs]
    lg_tr = F.logit(predict_phi(model, data["Zs"], data["R"], ben_train_abs))
    lg_sel = F.logit(predict_phi(model, data["Zs"], data["R"], ben_sel_abs))

    tctx = load_test_context(cfg["cache_dir"], cfg["labels_test"])
    lg_bt = F.logit(predict_phi(model, tctx["Zs"], tctx["R"], tctx["ben_pos"]))
    lg_at = F.logit(predict_phi(model, tctx["Zs"], tctx["R"], tctx["att_pos"]))
    sdae_bt = tctx["dae_score"][tctx["ben_pos"]]
    sdae_at = tctx["dae_score"][tctx["att_pos"]]
    cls_at = tctx["test_attack"][tctx["att_pos"]]

    return {"sdae_tr": sdae_tr, "lg_tr": lg_tr, "sdae_sel": sdae_sel, "lg_sel": lg_sel,
            "sdae_bt": sdae_bt, "lg_bt": lg_bt, "sdae_at": sdae_at, "lg_at": lg_at,
            "cls_at": cls_at, "tau_dae": tau_dae,
            # `tctx` e' locale a questa funzione: il supporto va restituito qui, altrimenti
            # run_fusione_stage lo cerca in uno scope in cui non esiste (NameError).
            "support_attacchi": int(len(tctx["att_pos"]))}


def run_fusione_stage(cfg: dict) -> None:
    """Valuta la fusione v1 leak-free -> results.json (+ dump score opzionale).

    cfg: cache_dir, labels_val, labels_test, l1l2_csv, pa_weights, model_config, mlp_fixed,
    out_dir; alpha_and (default 0.005); dump_scores_npz (opzionale, per ablazione/figure/
    stage `fusione_canonica`).
    """
    from lib.secondo_stadio import constants as C

    out_dir = Path(cfg["out_dir"])
    if not out_dir.exists():
        raise FileNotFoundError(f"out_dir non esiste (shell-first): {out_dir}")
    alpha_and = float(cfg.get("alpha_and", C.ALPHA_AND))

    s = compute_scores(cfg)
    tau_dae = s["tau_dae"]
    sdae_tr, lg_tr = s["sdae_tr"], s["lg_tr"]
    sdae_sel, lg_sel = s["sdae_sel"], s["lg_sel"]
    sdae_bt, lg_bt = s["sdae_bt"], s["lg_bt"]
    sdae_at, lg_at = s["sdae_at"], s["lg_at"]
    cls_at = s["cls_at"]

    results = {"tau_dae": tau_dae, "alpha_and": alpha_and,
               "support_attacchi": s["support_attacchi"], "presets": {}}
    for preset, fpr_t in F.FUSION_PRESETS.items():
        p = F.build_fusion(sdae_tr, lg_tr, sdae_sel, lg_sel, tau_dae, fpr_t, alpha_and)
        fl_bt = F.apply_fusion(p, sdae_bt, lg_bt)
        fl_at = F.apply_fusion(p, sdae_at, lg_at)
        z_bt = F.to_z(F.to_u(sdae_bt, lg_bt), p["mu"], p["sd"])
        z_at = F.to_z(F.to_u(sdae_at, lg_at), p["mu"], p["sd"])
        and_bt = F.and_predict(z_bt, p["tau_z"], p["theta_z"])
        and_at = F.and_predict(z_at, p["tau_z"], p["theta_z"])
        results["presets"][preset] = {
            "fpr_target_val": fpr_t,
            "and_fpr_test": F.fpr_of(and_bt), "and_recall_test": F.recall_of(and_at),
            "fused_fpr_test": F.fpr_of(fl_bt), "fused_recall_test": F.recall_of(fl_at),
            "theta_z": p["theta_z"],
            "theta_union": (None if not np.isfinite(p["theta_union"])
                            else float(p["theta_union"])),
            "capped": p["capped"],
            # DR per classe di ENTRAMBE le regole: il recupero fused-vs-AND si legge qui.
            "perclass_dr_and": F.perclass_dr(and_at, cls_at),
            "perclass_dr_fused": F.perclass_dr(fl_at, cls_at),
        }
        log.info("preset %-8s | AND rec=%.4f fpr=%.4f | FUS rec=%.4f fpr=%.4f (theta_u=%s)",
                 preset, F.recall_of(and_at), F.fpr_of(and_bt), F.recall_of(fl_at),
                 F.fpr_of(fl_bt), results["presets"][preset]["theta_union"])

    # Dump OPT-IN degli score (input di ablazione e figure piano-2D/frontiera). Additivo.
    dump = cfg.get("dump_scores_npz")
    if dump:
        np.savez_compressed(dump, sdae_tr=sdae_tr, lg_tr=lg_tr, sdae_sel=sdae_sel,
                            lg_sel=lg_sel, sdae_bt=sdae_bt, lg_bt=lg_bt,
                            sdae_at=sdae_at, lg_at=lg_at,
                            cls_at=np.asarray(cls_at, dtype=object),
                            tau_dae=np.float64(tau_dae), alpha_and=np.float64(alpha_and))
        log.info("dump_scores_npz -> %s", dump)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    log.info("scritto results.json (canonico leak-free) in %s", out_dir)


def run_fusione_canonica_stage(cfg: dict) -> None:
    """Stage `fusione_canonica`: regola v2 a TRE rami (AND u {logit>theta_union} u
    {s_dae>tau_hi}), due modalita' operative A/B (F.FUSION_MODES). Le soglie SONO LA RICETTA
    (percentili dei benigni-select del dominio in input + budget FPR), non valori congelati:
    si derivano dagli score forniti, qualunque sia il dominio (CSE o UNSW) — e' il claim di
    trasferibilita' della tesi ("la ricetta si calibra sui benigni locali") reso verificabile.

    cfg: out_dir (obbligatorio). Poi UNA delle due vie per gli score:
      - `scores_npz` (se il file esiste): riusa gli score gia' dumpati (no forward) — path CSE,
        dove lo stage `fusione` ha gia' prodotto scores.npz.
      - altrimenti lo stesso contratto di `run_fusione_stage` (cache_dir, labels_val,
        labels_test, l1l2_csv, pa_weights, model_config, mlp_fixed): gli score si calcolano
        fresh (forward-only, nessun training) via `compute_scores` — path UNSW, dove il dump
        non esiste ancora. `dump_scores_npz` opzionale per persistere il calcolo.

    Scrive results_canonico.json (mai results.json/results_v2.json: quelli sono gli
    artefatti congelati contro cui questo stage si verifica, non li sovrascrive mai).
    """
    out_dir = Path(cfg["out_dir"])
    if not out_dir.exists():
        raise FileNotFoundError(f"out_dir non esiste (shell-first): {out_dir}")

    scores_path = cfg.get("scores_npz")
    if scores_path and Path(scores_path).exists():
        # allow_pickle: cls_at e' un array object di etichette-stringa prodotto dalla NOSTRA
        # pipeline (run_fusione_stage/run_fusione_canonica_stage dump_scores_npz sopra) ->
        # artefatto fidato interno, non input esterno. Stesso pattern di run_ablazione (ablazione.py).
        d = np.load(scores_path, allow_pickle=True)
        s = {k: d[k] for k in _SCORE_KEYS}
        s["cls_at"] = d["cls_at"]
        log.info("fusione_canonica: score riusati da %s (nessun forward)", scores_path)
    else:
        s = compute_scores(cfg)
        log.info("fusione_canonica: score calcolati fresh (forward-only, nessun training)")
        dump = cfg.get("dump_scores_npz")
        if dump:
            np.savez_compressed(dump, **{k: s[k] for k in _SCORE_KEYS},
                                cls_at=np.asarray(s["cls_at"], dtype=object))
            log.info("dump_scores_npz -> %s", dump)

    # MCC_bal e' SEMPRE alla PREVALENZA DI RIFERIMENTO canonica (mcc_bal(fpr, recall) coi
    # default C.N_B/C.N_A da constants.py — la costante e' quella di CSE, invariante rispetto
    # al dominio valutato). NON i conteggi assoluti del test in input (len(sdae_bt)/
    # len(sdae_at)): verificato bit-exact contro results_v2.json (CSE) E contro
    # fusione_unsw_retrain/results.json (UNSW, stessi C.N_B/C.N_A pur avendo un test set di
    # dimensione completamente diversa) — e' esattamente cio' che rende MCC_bal confrontabile
    # CSE<->UNSW nella stessa tabella (memoria project_pi_prevalenza_riferimento: pi nativa,
    # MAI "di deployment", MAI i conteggi del set valutato).
    results = {"descrizione": "Fusione canonica v2 (tre rami), due modalita' operative A/B "
                              "(cap3, journal 2026-07-06)", "modalita": {}}
    for name, spec in F.FUSION_MODES.items():
        tau_dae = float(np.percentile(s["sdae_sel"], spec["tau_dae_pct"]))
        tau_hi = float(np.percentile(s["sdae_sel"], spec["tau_hi_pct"]))
        p = F.build_fusion(s["sdae_tr"], s["lg_tr"], s["sdae_sel"], s["lg_sel"],
                           tau_dae, spec["fpr_target"], spec["alpha_and"], tau_hi=tau_hi)
        f_bt = F.apply_fusion(p, s["sdae_bt"], s["lg_bt"])
        f_at = F.apply_fusion(p, s["sdae_at"], s["lg_at"])
        fpr, recall = F.fpr_of(f_bt), F.recall_of(f_at)
        mcc = mcc_bal(fpr, recall)
        results["modalita"][name] = {
            "ricetta": spec,
            "soglie": {"tau_dae": tau_dae, "tau_hi": tau_hi, "theta_z": p["theta_z"],
                       "theta_union": (None if not np.isfinite(p["theta_union"])
                                       else float(p["theta_union"])),
                       "mu": p["mu"].tolist(), "sd": p["sd"].tolist()},
            "fpr_test": fpr, "recall_test": recall, "mcc_bal_test": mcc,
            "perclass_dr": F.perclass_dr(f_at, s["cls_at"]),
        }
        log.info("[Mod %s] tau_dae(p%.1f)=%.6g tau_hi(p%.1f)=%.6g theta_z=%.6f "
                 "theta_union=%s -> fpr=%.6f%% recall=%.6f mcc_bal=%.6f",
                 name, spec["tau_dae_pct"], tau_dae, spec["tau_hi_pct"], tau_hi,
                 p["theta_z"], results["modalita"][name]["soglie"]["theta_union"],
                 fpr * 100, recall, mcc)

    (out_dir / "results_canonico.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    log.info("scritto results_canonico.json (regola v2, tre rami) in %s", out_dir)
