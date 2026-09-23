#!/usr/bin/env python
# TAG: DRIVER-PIPELINE | 2026-07-13 | compressione multiseed K->k' FASE 2 (testato) | vedi docs/refactoring_censimento.md
"""tools/refit_compress.py — FASE 2: compressione multiseed K→k'.

Per ogni finalista (trial_id, config) × seme extra (SEEDS_EXTRA):
  - addestra il DAE (NO Optuna, NO ASHA, NO pruning) tramite Ray grezzo, SUL WORKER VPS;
  - calcola AUC-PR inline (come la search);
  - calcola MCC@FPR1% inline (forward gratis: modello già in RAM)
      τ   = compute_threshold(score_val[val_y==0])       # calibra su val benigni
      mcc = binary_metrics_at_threshold(test_y, score_test, τ)["mcc"]  # misura su test
  - scrive per job: (trial_id, seed, ap, mcc) — append incrementale (robusto a crash).

ARCHITETTURA (CRITICA): il caricamento dati (val/test/train) avviene DENTRO il worker Ray sul VPS
(che ha <HOME>/dae_data_91), NON sul driver workstation (che ne è privo). Cache a livello di PROCESSO
worker (`_WORKER_DATA`): ogni worker carica val/test/train UNA volta e li riusa tra i suoi task
(~80 caricamenti, non K×extra). I PATH (non gli array) viaggiano nel payload del task.

Funzioni pure (testabili senza TF/Ray):
  merge_ap_and_mcc   — fonde AP base (6 semi da result.json) con extra (compressione)
  check_pipeline_homogeneity — vigilanza runtime: search↔compressione stessa pipeline
  compress_summary   — tabella riassuntiva ordinata per ap_mean desc

Nota: i pesi NON si salvano in FASE 2 (la maggior parte dei K verrebbe scartata; i pesi
dei soli k' sopravvissuti si recuperano in FASE 3 con censimento + ri-addestramento mirato).
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Chiavi critiche: devono coincidere tra search e refit_compress per rendere confrontabili
# gli AUC-PR base (da result.json) con quelli extra (dal ri-addestramento).
_HOMOGENEITY_KEYS = ("patience", "batch", "shard_dir", "val_X")


# ===================================================================== logica pura ====


def merge_ap_and_mcc(
    base_aps: dict[str, list[float]],
    extra_rows: list[dict[str, Any]],
) -> dict[str, dict]:
    """Fonde gli AUC-PR base (6 semi da result.json) con i risultati extra (compressione).

    Parametri
    ---------
    base_aps   : {trial_id: [ap_seed1, …, ap_seed6]} — AUC-PR ricostruiti dai result.json;
                 può essere {} per finalisti senza AP base.
    extra_rows : [{trial_id, seed, ap, mcc}, …] — output dei job di compressione.

    Restituisce
    -----------
    {trial_id: {ap_all, mcc_extra, ap_mean, ap_std, mcc_mean, mcc_std}}

    MCC è calcolato SOLO sui semi extra (N−6): i pesi base sono frammentati (400 pregressi
    non li salvavano, due VPS cadute hanno azzerato il resto). La media MCC resta su N−6
    semi extra: accettabile come caratterizzazione al punto operativo (non è selezione).
    """
    # Raggruppa extra per trial_id
    extra_by_trial: dict[str, list] = defaultdict(list)
    for r in extra_rows:
        extra_by_trial[r["trial_id"]].append(r)

    all_trial_ids = set(base_aps.keys()) | set(extra_by_trial.keys())
    merged = {}
    for tid in all_trial_ids:
        ap_base   = list(base_aps.get(tid, []))
        ap_extra  = [r["ap"]  for r in extra_by_trial.get(tid, [])]
        mcc_extra = [r["mcc"] for r in extra_by_trial.get(tid, [])]
        ap_all    = ap_base + ap_extra

        ap_arr  = np.asarray(ap_all,    dtype=float)
        mcc_arr = np.asarray(mcc_extra, dtype=float)

        merged[tid] = {
            "ap_all":    ap_all,
            "mcc_extra": mcc_extra,
            "ap_mean":   float(np.mean(ap_arr))  if len(ap_arr)  else float("nan"),
            "ap_std":    float(np.std(ap_arr))   if len(ap_arr)  else float("nan"),
            "mcc_mean":  float(np.mean(mcc_arr)) if len(mcc_arr) else float("nan"),
            "mcc_std":   float(np.std(mcc_arr))  if len(mcc_arr) else float("nan"),
            "n_seeds_ap":  len(ap_all),
            "n_seeds_mcc": len(mcc_extra),
        }
    return merged


def check_pipeline_homogeneity(base_cfg: dict, compress_cfg: dict) -> bool:
    """Verifica che la compressione usi la stessa pipeline della search.

    Controlla le chiavi critiche (_HOMOGENEITY_KEYS): patience, batch, shard_dir, val_X.
    Chiavi extra (n_new_trials, max_concurrent, ecc.) vengono ignorate.

    Ritorna True se tutte le chiavi critiche coincidono, False se c'è almeno una differenza.
    La chiamata deve avvenire PRIMA di unire AP-base e AP-extra: un parametro diverso
    renderebbe i due set non confrontabili (regimi di training diversi).
    """
    for key in _HOMOGENEITY_KEYS:
        v_base    = base_cfg.get(key)
        v_compress = compress_cfg.get(key)
        if v_base != v_compress:
            return False
    return True


def compress_summary(merged: dict[str, dict]) -> list[dict]:
    """Tabella riassuntiva ordinata per ap_mean discendente.

    Per ogni finalista: trial_id, ap_mean, ap_std, mcc_mean, mcc_std, n_seeds_ap, n_seeds_mcc.
    L'ordinamento è su AP (la selezione K→k' è su AUC-PR, non su MCC).
    MCC riportato SEMPRE con std (e con n_seeds_mcc dichiarato) — mai senza incertezza.
    """
    rows = []
    for tid, info in merged.items():
        rows.append({
            "trial_id":    tid,
            "ap_mean":     info["ap_mean"],
            "ap_std":      info["ap_std"],
            "mcc_mean":    info["mcc_mean"],
            "mcc_std":     info["mcc_std"],
            "n_seeds_ap":  info["n_seeds_ap"],
            "n_seeds_mcc": info["n_seeds_mcc"],
        })
    rows.sort(key=lambda r: -(r["ap_mean"] if math.isfinite(r["ap_mean"]) else -1e9))
    return rows


# ===================================================================== worker (VPS) ====

# Cache a livello di PROCESSO worker: val/test/train caricati UNA volta per processo Ray e
# riusati tra tutti i task che il processo serve (~80 caricamenti totali, non K×extra).
_WORKER_DATA: dict = {}


def _get_worker_data(hp: dict) -> dict:
    """Carica (con cache di processo) val/test/train dai PATH locali del VPS.

    val_X/test_X via MEMORY-MAP (mmap_mode='r'): la page cache del SO è CONDIVISA tra i ~16
    processi worker dello stesso VPS → ~2 GB totali, non 16×2 GB (le VPS hanno 30 GB; il load
    pieno andrebbe in OOM). compute_scores_chunked copia ogni chunk a float32 contiguo → stessi
    valori, nessun cambio di risultato. val_y/test_y sono minuscoli (int8 ~3 MB): load pieno.
    Chiave di cache = la tupla dei path → un solo set per processo."""
    key = (hp["shard_dir"], hp["val_X"], hp["val_y"], hp["test_X"], hp["test_y"])
    if key not in _WORKER_DATA:
        shard_dir = Path(hp["shard_dir"])
        shards = sorted(shard_dir.glob("train_shard_*.npy"))
        if not shards:
            raise SystemExit(f"nessun train_shard_*.npy in {shard_dir}")
        train_mm = [np.load(p, mmap_mode="r") for p in shards]
        val_X = np.load(hp["val_X"], mmap_mode="r")          # memmap float32 (condiviso)
        test_X = np.load(hp["test_X"], mmap_mode="r")         # memmap float32 (condiviso)
        val_y = np.asarray(np.load(hp["val_y"])).astype(np.int8)
        test_y = np.asarray(np.load(hp["test_y"])).astype(np.int8)
        _WORKER_DATA[key] = {
            "train_mm": train_mm, "val_X": val_X, "val_y": val_y,
            "test_X": test_X, "test_y": test_y, "prevalence": float(val_y.mean()),
        }
    return _WORKER_DATA[key]


def _train_one_compress(config: dict, seed: int, hp: dict, trial_id: str | None = None) -> dict:
    """Addestra il DAE per (config, seed) SUL WORKER — NO Optuna, NO ASHA, NO pruning.

    Replica fedele del training della search (KEEP-and-flag: la search è congelata post-run,
    estrarre un helper la retrofitterebbe) + due forward extra (val-benigni→τ, test→MCC).
    Se `hp["weights_home"]` è impostato (e c'è trial_id), salva i pesi dei semi extra nella HOME
    DUREVOLE del worker (<weights_home>/<trial_id>/seed_<s>/best.weights.h5) — stessa struttura
    della search, così i 6 base + gli 11 extra convivono per trial e la raccolta in locale (runs/)
    è uniforme. Off se weights_home è None (comportamento storico: nessun peso). Ritorna {seed,ap,mcc}.
    """
    import tensorflow as tf

    from lib.dae.callback import PrAucCallback
    from lib.dae.evaluate import binary_metrics_at_threshold, compute_scores_chunked
    from lib.dae.model import build_dae, make_dae_loss
    from lib.dae.objective import seed_ap
    from lib.dae.threshold import compute_threshold
    from lib.utils import N_BINARY, N_CONTINUOUS, N_FEATURES

    tf.keras.utils.set_random_seed(int(seed))
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    data = _get_worker_data(hp)
    val_X, val_y = data["val_X"], data["val_y"]
    test_X, test_y = data["test_X"], data["test_y"]
    train_mm = data["train_mm"]
    prevalence = data["prevalence"]
    batch = int(hp["batch"])
    cont_idx = list(range(N_CONTINUOUS))
    bin_idx  = list(range(N_CONTINUOUS, N_FEATURES))

    model, _enc = build_dae(
        n_features=N_FEATURES, btl=int(config["btl"]), exp=int(config["exp"]),
        n_continuous=N_CONTINUOUS, n_binary=N_BINARY,
        continuous_idx=cont_idx, binary_idx=bin_idx,
        l2_reg=hp["l2_reg"], noise_std=float(config["sigma"]),
        noise_type=hp["noise_type"],
        loss_continuous=hp["loss_continuous"], mid=int(config["mid"]))
    model.compile(optimizer=tf.keras.optimizers.Adam(float(config["lr"])),
                  loss=make_dae_loss(cont_idx, bin_idx, hp["loss_continuous"]))

    @tf.function(reduce_retracing=True,
                 input_signature=[tf.TensorSpec(shape=(None, N_FEATURES), dtype=tf.float32)])
    def predict_fn(x):
        return model(x, training=False)

    def ap_fn():
        scores = compute_scores_chunked(predict_fn, val_X)
        return seed_ap(val_y, scores)

    cb = PrAucCallback(ap_fn=ap_fn, patience=int(hp["patience"]),
                       prevalence=prevalence,
                       min_epochs=int(hp["non_learning_min_epochs"]),
                       margin=float(hp["non_learning_margin"]))

    # train_ds: stesso metodo della search (ordine batch permutato dal seme)
    steps_data = [(si, b * batch)
                  for si, s in enumerate(train_mm) for b in range(len(s) // batch)]
    order = np.random.default_rng(seed).permutation(len(steps_data))

    def gen():
        for k in order:
            si, start = steps_data[int(k)]
            chunk = np.ascontiguousarray(train_mm[si][start:start + batch], dtype=np.float32)
            yield chunk, chunk

    sig = (tf.TensorSpec((batch, N_FEATURES), tf.float32),) * 2
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig).repeat().prefetch(tf.data.AUTOTUNE)
    model.fit(ds, epochs=int(hp["max_epochs"]),
              steps_per_epoch=len(steps_data), callbacks=[cb], verbose=0)

    # ---- salva i pesi del seme extra nella HOME durevole (mai /tmp), se richiesto ----
    if hp.get("weights_home") and trial_id:
        seed_dir = Path(hp["weights_home"]) / trial_id / f"seed_{int(seed)}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(seed_dir / "best.weights.h5"))

    # ---- AUC-PR (stesso path della search) ----
    score_val = compute_scores_chunked(predict_fn, val_X)
    ap        = seed_ap(val_y, score_val)

    # ---- MCC@FPR1% — calibra su val benigni, misura su test (MAI stesso set) ----
    tau = compute_threshold(score_val[val_y == 0])
    score_test = compute_scores_chunked(predict_fn, test_X)
    mcc = binary_metrics_at_threshold(test_y, score_test, tau)["mcc"]

    return {"seed": int(seed), "ap": float(ap), "mcc": float(mcc)}


def _worker_fn(trial_id: str, config: dict, seed: int, hp: dict) -> dict:
    """Funzione Ray remote: addestra (trial_id, config, seed) e ritorna la riga risultato.
    Si ricalcolano+salvano SOLO i semi passati (per la compressione a k' = gli n'-6 extra; i 6
    base non arrivano qui, vengono dai result.json della search)."""
    res = _train_one_compress(config, seed, hp, trial_id)
    return {"trial_id": trial_id, **res}


# ===================================================================== driver (workstation) ====

def _build_hp(cfg: dict):
    """Iperparametri fissi + path dati per i worker, dalla config della search."""
    from lib.dae import constants as C
    val_X = cfg["val_X"]
    val_y = cfg["val_y"]
    return {
        "shard_dir": cfg["shard_dir"],
        "val_X": val_X, "val_y": val_y,
        "test_X": cfg.get("test_X", str(Path(val_X).parent / "test_X.npy")),
        "test_y": cfg.get("test_y", str(Path(val_y).parent / "test_y.npy")),
        "patience": int(cfg.get("patience", C.PATIENCE_PROVVISORIO)),
        "batch": int(cfg.get("batch", C.BATCH_FIXED)),
        "max_epochs": int(cfg.get("max_epochs", 100)),
        "non_learning_min_epochs": int(cfg.get("non_learning_min_epochs", C.NON_LEARNING_MIN_EPOCHS)),
        "non_learning_margin": float(cfg.get("non_learning_margin", C.NON_LEARNING_MARGIN)),
        "l2_reg": C.L2_FIXED,
        "noise_type": C.NOISE_TYPE,
        "loss_continuous": C.LOSS_CONTINUOUS,
        # Radice DUREVOLE pesi (HOME worker, mai /tmp): se impostata, salva gli n'-6 semi extra.
        "weights_home": cfg.get("weights_home"),
    }


def _read_shortlist(path: Path) -> list[dict]:
    """Righe della shortlist CSV (rank, trial_id, source, ap_mean, ap_std, ray_dir)."""
    with open(path) as f:
        return list(csv.DictReader(f))


def _base_aps_for(ray_dir: Path) -> list[float]:
    """AP per-seme base (6 semi) ricostruiti dal result.json del finalista."""
    from lib.search.perseed import load_result_records, reconstruct_seed_aps
    recs = load_result_records(Path(ray_dir) / "result.json")
    if not recs:
        return []
    aps, clean = reconstruct_seed_aps(recs)
    return aps if clean else []


def main(argv=None):
    """Driver FASE 2 (su workstation): legge la shortlist, schedula i job Ray (worker su VPS),
    raccoglie i risultati (append incrementale), unisce con gli AP base e stampa la tabella.

    Uso:
      python tools/refit_compress.py \\
          --shortlist runs/dae/compress_fase2/shortlist.csv \\
          --config    configs/dae/search_extended.yaml \\
          --extra-seeds 2025 2026 2027 2028 2029 2030 2031 2032 2033 2034 \\
          --out-dir   runs/dae/compress_fase2 \\
          [--limit N]   # smoke: solo i primi N job
    """
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shortlist", required=True, help="CSV con trial_id + ray_dir (select_shortlist)")
    ap.add_argument("--config", required=True, help="YAML della search (iperparametri fissi)")
    ap.add_argument("--extra-seeds", nargs="+", type=int, required=True, help="semi extra (7°…N°)")
    ap.add_argument("--out-dir", required=True, help="dir output in HOME (deve esistere)")
    ap.add_argument("--limit", type=int, default=0, help="smoke: usa solo i primi N job (0=tutti)")
    args = ap.parse_args(argv)

    import ray
    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        raise SystemExit(f"--out-dir non esiste: {out_dir}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    hp = _build_hp(cfg)

    # Vigilanza omogeneità: i parametri critici provengono dalla STESSA config della search.
    print(f"[VIGILANZA pipeline] patience={hp['patience']} batch={hp['batch']} "
          f"shard_dir={hp['shard_dir']} max_epochs={hp['max_epochs']}")
    if hp["patience"] != 18:
        print(f"  ATTENZIONE: patience={hp['patience']} ≠ 18 (valore Optuna). Verificare.")

    # Shortlist + config/base-AP per finalista (dal ray_dir, robusto)
    rows = _read_shortlist(Path(args.shortlist))
    finalists = []   # (trial_id, config_dict, base_aps)
    base_aps_finalisti = {}
    for r in rows:
        tid = r["trial_id"]
        rdir = Path(r["ray_dir"])
        params_f = rdir / "params.json"
        if not params_f.exists():
            raise SystemExit(f"params.json mancante per {tid}: {params_f}")
        config = json.loads(params_f.read_text())
        base_aps_finalisti[tid] = _base_aps_for(rdir)
        finalists.append((tid, config))
    print(f"Finalisti: {len(finalists)}  (AP base ricostruiti: "
          f"{sum(1 for v in base_aps_finalisti.values() if v)}/{len(finalists)})")

    # Job per-(trial, seme extra)
    jobs = [(tid, config, seed) for (tid, config) in finalists for seed in args.extra_seeds]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"Job totali: {len(jobs)} ({len(finalists)} finalisti × {len(args.extra_seeds)} semi extra"
          f"{'  [LIMIT '+str(args.limit)+']' if args.limit else ''})", flush=True)

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)
    remote_fn = ray.remote(num_cpus=1, max_retries=3)(_worker_fn)
    hp_ref = ray.put(hp)   # un solo oggetto condiviso (path dati + iperparametri) per tutti i task

    # Append incrementale del CSV extra (robusto a crash: i risultati arrivati sopravvivono)
    extra_csv = out_dir / "compress_extra.csv"
    extra_rows = []
    futures = {remote_fn.remote(tid, config, seed, hp_ref): (tid, seed)
               for (tid, config, seed) in jobs}
    pending = list(futures.keys())
    with open(extra_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial_id", "seed", "ap", "mcc"])
        w.writeheader(); f.flush()
        done = 0
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            for fut in ready:
                try:
                    res = ray.get(fut)
                    extra_rows.append(res)
                    w.writerow({k: res[k] for k in ("trial_id", "seed", "ap", "mcc")}); f.flush()
                except Exception as e:  # noqa: BLE001
                    tid, seed = futures[fut]
                    print(f"  JOB FALLITO {tid} seme {seed}: {e}", flush=True)
                done += 1
                if done % 25 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)} job completati", flush=True)
    print(f"Risultati extra → {extra_csv}")

    # Merge con gli AP base + tabella
    merged  = merge_ap_and_mcc(base_aps_finalisti, extra_rows)
    summary = compress_summary(merged)

    print("\n=== TABELLA COMPRESSIONE (ordinata per ap_mean) ===")
    print(f"{'#':>3}  {'ap_mean':>8} {'ap_std':>7} {'n_ap':>4}  "
          f"{'mcc_mean':>8} {'mcc_std':>7} {'n_mcc':>5}  trial")
    for i, row in enumerate(summary[:30], 1):
        print(f"{i:>3}  {row['ap_mean']:.4f}  {row['ap_std']:.4f}  {row['n_seeds_ap']:>4}  "
              f"{row['mcc_mean']:.4f}   {row['mcc_std']:.4f}  {row['n_seeds_mcc']:>5}  "
              f"{row['trial_id'][:48]}")

    summary_out = out_dir / "compress_summary.csv"
    with open(summary_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nSummary → {summary_out}")


if __name__ == "__main__":
    main()
