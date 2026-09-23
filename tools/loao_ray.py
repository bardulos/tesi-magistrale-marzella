#!/usr/bin/env python
# TAG: DRIVER-PIPELINE | 2026-09-22 | produttore dei pesi #569 (runs/dae/loao_fase3/weights/), sorgente di models/*/dae.npz; superato da loao_run.py SOLO per il calcolo LOAO forward-only | vedi docs/refactoring_censimento.md
# Il superamento da parte di tools/loao_run.py riguarda il SOLO calcolo LOAO forward-only: la produzione dei pesi resta di questo file.
"""tools/loao_ray.py — FASE 3: LOAO dei finalisti su Ray (recupero pesi + inferenza per-classe).

Per ogni finalista (trial_num, config) × seme (17 = 6 base + 11 extra), UN task Ray sul worker VPS:
  - addestra il DAE (NO Optuna/ASHA/pruning) — replica fedele del training di search/compressione
    (KEEP-and-flag, come refit_compress: la search è congelata post-run; un helper la retrofitterebbe);
  - SALVA i pesi nella HOME del VPS (mai /tmp): <weights_base>/<trial_num>/seed_<seme>.weights.h5;
  - forward su val_X (modello già in RAM) → loao_metrics(score, val_attack):
      DR per-classe, Z-DR (macro), overall_dr (micro, tutte le classi unite), Cohen's d;
  - ritorna il record per (trial_num, seme) — append incrementale su JSONL (robusto a crash).

Il training con il VALORE GREZZO del seme riproduce bit-exact i modelli di search/compressione (il
base-retrain l'ha verificato: AP ricostruito = result.json, Δ=0) → i 425 modelli sono identici a
quelli che hanno prodotto AP/MCC. I pesi salvati sui VPS si raccolgono poi via rsync nel repo.

ARCHITETTURA (come refit_compress): caricamento dati DENTRO il worker VPS (workstation non ha <HOME>/
dae_data_91); cache di processo (`_WORKER_DATA`); val_X via mmap (page cache condivisa tra i 16
worker → ~2 GB, non 16×2); val_attack (object, stringhe per-classe) caricato una volta per processo.

Funzioni pure (testabili senza TF/Ray):
  build_loao_jobs   — espande finalisti × semi preservando l'ordine dato
  build_figure_data — record per-(modello,seme) → strutture delle due figure, nell'ordine dato
"""
from __future__ import annotations

import csv
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


# ===================================================================== logica pura ====


def build_loao_jobs(finalists: list[dict], seeds: list[int]) -> list[dict]:
    """Espande (finalista × seme) preservando l'ordine: tutti i semi del 1o finalista, poi del 2o…
    Ogni job = {trial_num, trial_id, ray_dir, config, seed}."""
    jobs = []
    for fin in finalists:
        trial_id = fin.get("trial_id") or Path(fin["ray_dir"]).name
        for seed in seeds:
            jobs.append({"trial_num": fin["trial_num"], "trial_id": trial_id,
                         "ray_dir": fin["ray_dir"], "config": fin["config"], "seed": int(seed)})
    return jobs


def build_figure_data(records: list[dict], order: list[int]):
    """Dai record per-(modello,seme) costruisce le strutture delle due figure, NELL'ORDINE DATO.

    Ritorna (overall_ordered, perclass_by_model):
      - overall_ordered  = [(label, [overall_dr per seme]), …]  → figura 1 (un box per modello);
      - perclass_by_model = OrderedDict{label: {classe: [dr per seme]}}  → figura 2 (per modello).
    L'ordine di `order` (numeri Optuna) è preservato 1:1. I modelli senza record compaiono con
    liste vuote (tengono la posizione)."""
    by_model: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_model[int(r["trial_num"])].append(r)

    overall_ordered = []
    perclass_by_model: "OrderedDict[str, dict]" = OrderedDict()
    for num in order:
        recs = by_model.get(int(num), [])
        label = str(num)
        overall_ordered.append((label, [r["overall_dr"] for r in recs]))
        cls_map: dict[str, list] = defaultdict(list)
        for r in recs:
            for c, dr in r.get("dr_by_class", {}).items():
                cls_map[c].append(dr)
        perclass_by_model[label] = dict(cls_map)
    return overall_ordered, perclass_by_model


# ===================================================================== worker (VPS) ====

# Cache a livello di PROCESSO worker: val_X (mmap) + val_attack (object) + train shards (mmap),
# caricati UNA volta per processo Ray e riusati tra tutti i task del processo.
_WORKER_DATA: dict = {}


def _get_worker_data_loao(hp: dict) -> dict:
    """Carica (con cache di processo) val_X (mmap), val_attack (object) e i train shard (mmap).

    val_X via mmap (page cache condivisa tra i ~16 worker dello stesso VPS → ~2 GB totali, non
    16×2). compute_scores_chunked copia ogni chunk a float32 contiguo → stessi valori. val_attack
    è l'array di stringhe per-classe (dtype=object), stesso ordine riga di val_X."""
    key = (hp["shard_dir"], hp["val_X"], hp["val_attack"])
    if key not in _WORKER_DATA:
        shard_dir = Path(hp["shard_dir"])
        shards = sorted(shard_dir.glob("train_shard_*.npy"))
        if not shards:
            raise SystemExit(f"nessun train_shard_*.npy in {shard_dir}")
        train_mm = [np.load(p, mmap_mode="r") for p in shards]
        val_X = np.load(hp["val_X"], mmap_mode="r")               # memmap float32 (condiviso)
        val_attack = np.load(hp["val_attack"], allow_pickle=True)  # stringhe per-classe (object)
        val_y = (np.asarray(val_attack).astype(str) != hp["benign_name"]).astype(np.int8)
        _WORKER_DATA[key] = {"train_mm": train_mm, "val_X": val_X,
                             "val_attack": val_attack, "val_y": val_y,
                             "prevalence": float(val_y.mean())}
    return _WORKER_DATA[key]


def _recover_and_loao(trial_num, trial_id: str, config: dict, seed: int, hp: dict) -> dict:
    """Recupera i pesi (config, seed) SUL WORKER e forward LOAO su val.

    RIUSO vs RICOSTRUZIONE (determinismo bit-exact, già provato Δ=0 dal base-retrain):
      - se i pesi del seme esistono già (search, pre-piazzati su tutti i VPS) → load_weights,
        mode="reused" (NESSUN ri-addestramento);
      - altrimenti → addestra (replica fedele search/compressione, KEEP-and-flag), mode="trained".
    In ENTRAMBI i casi salva i pesi nel layout uniforme weights_base/<trial_num>/seed_<seme>.weights.h5
    (così tutti i 425 finiscono nel repo con la stessa chiave) e fa il forward LOAO.
    Ritorna {trial_num, seed, mode, overall_dr, z_dr, dr_by_class, cohens_d_by_class, threshold,
    host, weights}.
    """
    import socket

    import tensorflow as tf

    from lib.dae.callback import PrAucCallback
    from lib.dae.evaluate import compute_scores_chunked
    from lib.dae.loao import loao_metrics
    from lib.dae.model import build_dae, make_dae_loss
    from lib.dae.objective import seed_ap
    from lib.utils import N_BINARY, N_CONTINUOUS, N_FEATURES

    tf.keras.utils.set_random_seed(int(seed))
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    data = _get_worker_data_loao(hp)
    val_X, val_y, val_attack = data["val_X"], data["val_y"], data["val_attack"]
    train_mm = data["train_mm"]
    prevalence = data["prevalence"]
    batch = int(hp["batch"])
    cont_idx = list(range(N_CONTINUOUS))
    bin_idx = list(range(N_CONTINUOUS, N_FEATURES))

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

    # pesi della search pre-piazzati (riuso, nessun retrain): <existing_base>/<trial_id>/seed_<s>/best.weights.h5
    existing = Path(hp["existing_base"]) / trial_id / f"seed_{int(seed)}" / "best.weights.h5"
    if existing.exists():
        model.load_weights(str(existing))
        mode = "reused"
    else:
        def ap_fn():
            scores = compute_scores_chunked(predict_fn, val_X)
            return seed_ap(val_y, scores)

        cb = PrAucCallback(ap_fn=ap_fn, patience=int(hp["patience"]),
                           prevalence=prevalence,
                           min_epochs=int(hp["non_learning_min_epochs"]),
                           margin=float(hp["non_learning_margin"]))
        steps_data = [(si, b * batch)
                      for si, s in enumerate(train_mm) for b in range(len(s) // batch)]
        order = np.random.default_rng(seed).permutation(len(steps_data))

        def gen():
            for k in order:
                si, start = steps_data[int(k)]
                chunk = np.ascontiguousarray(train_mm[si][start:start + batch], dtype=np.float32)
                yield chunk, chunk

        sig = (tf.TensorSpec((batch, N_FEATURES), tf.float32),) * 2
        ds = (tf.data.Dataset.from_generator(gen, output_signature=sig)
              .repeat().prefetch(tf.data.AUTOTUNE))
        model.fit(ds, epochs=int(hp["max_epochs"]),
                  steps_per_epoch=len(steps_data), callbacks=[cb], verbose=0)
        mode = "trained"

    # ---- salva i pesi nel layout uniforme (HOME del VPS, mai /tmp) — tutti i 425 nel repo ----
    wpath = Path(hp["weights_base"]) / str(trial_num) / f"seed_{int(seed)}.weights.h5"
    wpath.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(wpath))

    # ---- forward LOAO su val (stesso score della search: MSE sulle binarie) ----
    score_val = compute_scores_chunked(predict_fn, val_X)
    m = loao_metrics(score_val, val_attack, fpr_target=float(hp["fpr_target"]),
                     benign_name=hp["benign_name"])
    return {"trial_num": trial_num, "seed": int(seed), "mode": mode,
            "overall_dr": m["overall_dr"], "z_dr": m["z_dr"],
            "dr_by_class": m["dr_by_class"], "cohens_d_by_class": m["cohens_d_by_class"],
            "threshold": m["threshold"], "host": socket.gethostname(),
            "weights": str(wpath)}


def _worker_fn(job: dict, hp: dict) -> dict:
    """Funzione Ray remote: recupera (riuso o ricostruzione) + forward per un job."""
    return _recover_and_loao(job["trial_num"], job["trial_id"], job["config"],
                             job["seed"], hp)


# ===================================================================== driver (workstation) ====

def _build_hp(cfg: dict, val_attack: str, weights_base: str, existing_base: str) -> dict:
    """Iperparametri fissi + path dati per i worker, dalla config della search (val-only + attack)."""
    from lib.dae import constants as C
    return {
        "shard_dir": cfg["shard_dir"],
        "val_X": cfg["val_X"],
        "val_attack": val_attack,
        "benign_name": "Benign",
        "fpr_target": C.FPR_TARGET,
        "weights_base": weights_base,
        "existing_base": existing_base,
        "patience": int(cfg.get("patience", C.PATIENCE_PROVVISORIO)),
        "batch": int(cfg.get("batch", C.BATCH_FIXED)),
        "max_epochs": int(cfg.get("max_epochs", 100)),
        "non_learning_min_epochs": int(cfg.get("non_learning_min_epochs", C.NON_LEARNING_MIN_EPOCHS)),
        "non_learning_margin": float(cfg.get("non_learning_margin", C.NON_LEARNING_MARGIN)),
        "l2_reg": C.L2_FIXED,
        "noise_type": C.NOISE_TYPE,
        "loss_continuous": C.LOSS_CONTINUOUS,
    }


def _read_finalists(path: Path) -> list[dict]:
    """Legge finalists_ordered.csv (order, trial_num, …, ray_dir) e carica la config da params.json,
    preservando l'ordine del file."""
    finalists = []
    with open(path) as f:
        for r in sorted(csv.DictReader(f), key=lambda r: int(r["order"])):
            rdir = Path(r["ray_dir"])
            cfg = json.loads((rdir / "params.json").read_text())
            finalists.append({"trial_num": int(r["trial_num"]), "trial_id": r["trial_id"],
                              "ray_dir": str(rdir), "config": cfg})
    return finalists


def _generate_figures(records: list[dict], order: list[int], figs_dir: Path) -> None:
    """Figura 1 (overall_dr per modello, ordine dato) + figure 2 (per-modello, per-classe)."""
    from lib.plotting.loao_plots import (plot_dr_by_class_for_model,
                                         plot_overall_dr_by_model)
    figs_dir.mkdir(parents=True, exist_ok=True)
    overall, perclass = build_figure_data(records, order)
    plot_overall_dr_by_model(
        overall, figs_dir / "loao_overall_dr.png",
        title="LOAO — detection-rate complessivo per modello (distribuzione sui 17 semi)")
    for label, cls_map in perclass.items():
        if any(len(v) for v in cls_map.values()):
            plot_dr_by_class_for_model(
                cls_map, figs_dir / f"loao_perclass_{label}.png",
                title=f"Modello {label} — detection-rate per classe (distribuzione sui 17 semi)")


def _write_summary(records: list[dict], order: list[int], out_csv: Path) -> None:
    """CSV riassuntivo per modello (ordine dato): overall_dr e z_dr media±std sui semi disponibili."""
    overall, perclass = build_figure_data(records, order)
    overall_map = dict(overall)
    z_by_model: dict[str, list] = defaultdict(list)
    for r in records:
        z_by_model[str(r["trial_num"])].append(r["z_dr"])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "trial_num", "n_seeds", "overall_dr_mean", "overall_dr_std",
                    "z_dr_mean", "z_dr_std"])
        for i, num in enumerate(order, 1):
            ov = np.asarray(overall_map.get(str(num), []), float)
            zz = np.asarray(z_by_model.get(str(num), []), float)
            w.writerow([i, num, len(ov),
                        f"{ov.mean():.6f}" if len(ov) else "",
                        f"{ov.std():.6f}" if len(ov) else "",
                        f"{zz.mean():.6f}" if len(zz) else "",
                        f"{zz.std():.6f}" if len(zz) else ""])


def main(argv=None):
    """Driver FASE 3 (su workstation): legge i finalisti ordinati, schedula i job Ray (worker su VPS),
    raccoglie i record (JSONL append-incrementale), genera figure + summary.

    Uso:
      python tools/loao_ray.py \\
          --finalists runs/dae/loao_fase3/finalists_ordered.csv \\
          --config    configs/dae/search_extended.yaml \\
          --val-attack <HOME>/dae_data_91/val_Attack.npy \\
          --weights-base <HOME>/repov6_loao_weights \\
          --out-dir   runs/dae/loao_fase3 \\
          [--limit N]   # smoke: solo i primi N job
    """
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finalists", required=True, help="finalists_ordered.csv (ordine ESATTO)")
    ap.add_argument("--config", required=True, help="YAML della search (iperparametri fissi)")
    ap.add_argument("--val-attack", required=True, help="path a val_Attack.npy sul VPS")
    ap.add_argument("--weights-base", required=True, help="dir pesi OUTPUT in HOME del VPS (mai /tmp)")
    ap.add_argument("--existing-base", required=True,
                    help="dir pesi search pre-piazzati sui VPS (<id>/seed_<s>/best.weights.h5) per il riuso")
    ap.add_argument("--out-dir", required=True, help="dir output su workstation (deve esistere)")
    ap.add_argument("--limit", type=int, default=0, help="smoke: solo i primi N job (0=tutti)")
    args = ap.parse_args(argv)

    import ray
    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        raise SystemExit(f"--out-dir non esiste: {out_dir}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    hp = _build_hp(cfg, args.val_attack, args.weights_base, args.existing_base)
    print(f"[VIGILANZA pipeline] patience={hp['patience']} batch={hp['batch']} "
          f"shard_dir={hp['shard_dir']} max_epochs={hp['max_epochs']}", flush=True)

    finalists = _read_finalists(Path(args.finalists))
    order = [f["trial_num"] for f in finalists]
    from lib.dae.constants import SEEDS, SEEDS_EXTRA
    seeds = list(SEEDS) + list(SEEDS_EXTRA)            # 17 = 6 base + 11 extra
    jobs = build_loao_jobs(finalists, seeds)
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"Finalisti: {len(finalists)}  semi: {len(seeds)}  job: {len(jobs)}"
          f"{'  [LIMIT '+str(args.limit)+']' if args.limit else ''}", flush=True)

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)
    remote_fn = ray.remote(num_cpus=1, max_retries=3)(_worker_fn)
    hp_ref = ray.put(hp)

    jsonl = out_dir / "loao_records.jsonl"
    records: list[dict] = []
    futures = {remote_fn.remote(job, hp_ref): job for job in jobs}
    pending = list(futures.keys())
    with open(jsonl, "w") as f:
        done = 0
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            for fut in ready:
                try:
                    res = ray.get(fut)
                    records.append(res)
                    f.write(json.dumps(res) + "\n"); f.flush()
                except Exception as e:  # noqa: BLE001
                    job = futures[fut]
                    print(f"  JOB FALLITO #{job['trial_num']} seme {job['seed']}: {e}", flush=True)
                done += 1
                if done % 25 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)} job completati", flush=True)
    n_reused = sum(1 for r in records if r.get("mode") == "reused")
    n_trained = sum(1 for r in records if r.get("mode") == "trained")
    print(f"Record LOAO → {jsonl}  ({len(records)} record; riusati={n_reused} ricostruiti={n_trained})",
          flush=True)

    _write_summary(records, order, out_dir / "loao_summary.csv")
    try:
        _generate_figures(records, order, out_dir / "figs")
        print(f"Figure → {out_dir / 'figs'}")
    except Exception as e:  # noqa: BLE001
        print(f"  (plotting saltato: {e})")
    print(f"Summary → {out_dir / 'loao_summary.csv'}")


if __name__ == "__main__":
    main()
