"""lib/dae/search.py — componente DAE per il motore di ricerca AUC-PR (repov6).

Wira le parti pure (model/objective/space/callback/evaluate) nel `SearchComponent`
generico di lib.search.engine:
  - setup        : carica val (X,y) + shard di train (memmap), misura la PREVALENZA su val;
  - train_one_seed: addestra un DAE (config) con PrAucCallback (early-stop su AUC-PR +
                    restore-best + filtro non-apprendimento), ritorna best_ap + filter_reason;
  - aggregate    : objective = mean(AUC-PR) − K·std(AUC-PR) sui semi genuini (aggregate_seeds);
  - run_search_stage: entry per lo stage 'search' (make_dae_component + engine.run_search).

train_one_seed/setup sono lo STOCASTICO (TF): validati dal dry-run (Step 5), non da unit test.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from lib.dae import constants as C
from lib.dae.objective import aggregate_seeds
from lib.dae.space import sample_dae_space
from lib.search.engine import SearchComponent
from lib.utils import N_BINARY, N_CONTINUOUS, N_FEATURES

log = logging.getLogger(__name__)


def define_space(trial) -> dict:
    """Spazio di ricerca: delega a sample_dae_space (btl<mid<exp per costruzione)."""
    return sample_dae_space(trial)


def build_parent_args(cfg: dict) -> dict:
    """Path e iperparametri fissi spediti ai worker (serializzati)."""
    return {
        "shard_dir": cfg["shard_dir"],
        "val_X": cfg["val_X"],
        "val_y": cfg["val_y"],
        "max_epochs": int(cfg.get("max_epochs", 100)),
        "patience": int(cfg.get("patience", C.PATIENCE_PROVVISORIO)),
        "batch": int(cfg.get("batch", C.BATCH_FIXED)),
        "non_learning_margin": float(cfg.get("non_learning_margin", C.NON_LEARNING_MARGIN)),
        "non_learning_min_epochs": int(cfg.get("non_learning_min_epochs", C.NON_LEARNING_MIN_EPOCHS)),
        # Salvataggio pesi per-seme (best-epoch). Off di default (search normale/test); on nella
        # FASE 1 allargata. NB: si salva a OGNI seme che termina -> i trial poi-PRUNED lasciano
        # i pesi dei loro 4 semi (orfani): la raccolta (FASE 2) filtra i soli COMPLETE.
        "save_weights": bool(cfg.get("save_weights", False)),
        # Radice DUREVOLE dei pesi (HOME del worker, mai /tmp): se impostata, engine ancora i pesi
        # a <weights_home>/<trial_id>/seed_<s>/ invece della working-dir Ray (cwd, sotto /tmp, che
        # un crash del VPS azzera). None -> comportamento storico (cwd). Vedi engine._run_trainable.
        "weights_home": cfg.get("weights_home"),
    }


def setup(parent_args: dict) -> dict:
    """Caricato UNA VOLTA per trainable: val (X,y) in RAM, shard train in memmap
    (page-cache condivisa tra i semi), prevalenza misurata su val_y (per il filtro B1)."""
    shard_dir = Path(parent_args["shard_dir"])
    shards = sorted(shard_dir.glob("train_shard_*.npy"))
    if not shards:
        raise SystemExit(f"nessun train_shard_*.npy in {shard_dir}")
    train_mm = [np.load(p, mmap_mode="r") for p in shards]
    val_X = np.load(parent_args["val_X"])          # gia' float32 su disco: nessuna copia (v. guardia)
    if val_X.dtype != np.float32:
        log.warning("val_X %s != float32: conversione esplicita (copia in RAM)", val_X.dtype)
        val_X = val_X.astype(np.float32)
    val_y = np.asarray(np.load(parent_args["val_y"])).astype(np.int8)
    prevalence = float(val_y.mean())
    log.info("setup DAE: %d shard train, val=%d (prevalenza=%.4f)",
             len(train_mm), len(val_y), prevalence)
    return {
        "train_mm": train_mm, "val_X": val_X, "val_y": val_y, "prevalence": prevalence,
        "cont_idx": list(range(N_CONTINUOUS)), "bin_idx": list(range(N_CONTINUOUS, N_FEATURES)),
    }


def _make_train_ds(train_mm, batch: int, seed: int):
    """tf.data dai memmap: ordine dei batch permutato deterministicamente dal seme
    (page-cache condivisa, niente materializzazione). Corruzione gaussiana DENTRO il modello."""
    import tensorflow as tf
    desc = [(si, b * batch) for si, s in enumerate(train_mm) for b in range(len(s) // batch)]
    order = np.random.default_rng(seed).permutation(len(desc))

    def gen():
        for k in order:
            si, start = desc[int(k)]
            chunk = np.ascontiguousarray(train_mm[si][start:start + batch], dtype=np.float32)
            yield chunk, chunk

    sig = (tf.TensorSpec((batch, N_FEATURES), tf.float32),) * 2
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig).repeat().prefetch(tf.data.AUTOTUNE)
    return ds, len(desc)   # .repeat() + steps_per_epoch: ogni epoca = un passaggio completo


def train_one_seed(config: dict, seed: int, shared_state: dict,
                   parent_args: dict, seed_dir: Path) -> dict:
    """Addestra un DAE per un seme; ritorna best_ap (= AUC-PR al restore-best) + filter_reason."""
    import tensorflow as tf

    from lib.dae.callback import PrAucCallback
    from lib.dae.evaluate import compute_scores_chunked
    from lib.dae.model import build_dae, make_dae_loss
    from lib.dae.objective import seed_ap

    tf.keras.utils.set_random_seed(int(seed))
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    cont_idx, bin_idx = shared_state["cont_idx"], shared_state["bin_idx"]
    val_X, val_y = shared_state["val_X"], shared_state["val_y"]
    batch = int(parent_args["batch"])

    model, _enc = build_dae(
        n_features=N_FEATURES, btl=int(config["btl"]), exp=int(config["exp"]),
        n_continuous=N_CONTINUOUS, n_binary=N_BINARY,
        continuous_idx=cont_idx, binary_idx=bin_idx,
        l2_reg=C.L2_FIXED, noise_std=float(config["sigma"]), noise_type=C.NOISE_TYPE,
        loss_continuous=C.LOSS_CONTINUOUS, mid=int(config["mid"]))
    model.compile(optimizer=tf.keras.optimizers.Adam(float(config["lr"])),
                  loss=make_dae_loss(cont_idx, bin_idx, C.LOSS_CONTINUOUS))

    @tf.function(reduce_retracing=True,
                 input_signature=[tf.TensorSpec(shape=(None, N_FEATURES), dtype=tf.float32)])
    def predict_fn(x):
        return model(x, training=False)

    def ap_fn():
        scores = compute_scores_chunked(predict_fn, val_X)
        return seed_ap(val_y, scores)

    cb = PrAucCallback(ap_fn=ap_fn, patience=int(parent_args["patience"]),
                       prevalence=float(shared_state["prevalence"]),
                       min_epochs=int(parent_args["non_learning_min_epochs"]),
                       margin=float(parent_args["non_learning_margin"]))
    train_ds, steps = _make_train_ds(shared_state["train_mm"], batch, int(seed))
    model.fit(train_ds, epochs=int(parent_args["max_epochs"]), steps_per_epoch=steps,
              callbacks=[cb], verbose=0)

    # pesi al best-epoch (gia' ripristinati dal restore-best del callback): salvati per-seme
    # solo se richiesto (FASE 1). Path strutturato seed_dir/best.weights.h5 sul FS locale del worker.
    # Accanto ai pesi si persiste la traiettoria AUC-PR per-epoca (byproduct del PrAucCallback, gia' in
    # RAM): cosi' e' disponibile per i modelli selezionati da K le curve di training, senza re-pass.
    weights_path = None
    if parent_args.get("save_weights"):
        seed_dir.mkdir(parents=True, exist_ok=True)
        weights_path = str(seed_dir / "best.weights.h5")
        model.save_weights(weights_path)
        (seed_dir / "history_ap.json").write_text(json.dumps(
            [float(x) if x is not None else None for x in cb.history_ap]))

    return {
        "seed": int(seed),
        "best_ap": float(cb.best_ap) if cb.best_ap > -1e18 else float("nan"),
        "best_epoch": int(cb.best_epoch),
        "filter_reason": cb.filter_reason,
        "n_epochs": len(cb.history_ap),
        "weights_path": weights_path,
    }


def aggregate(per_seed_results: list, config: dict) -> dict:
    """objective = mean(AUC-PR) − K_STD·std(AUC-PR) sui semi GENUINI (non filtrati). I semi
    non-apprendenti sono esclusi; se nessun genuino -> objective sentinella bassa (-1)."""
    genuine = [r["best_ap"] for r in per_seed_results if r.get("filter_reason") != "non_learning"]
    obj = aggregate_seeds(genuine, C.K_STD)
    import math
    if math.isnan(obj):
        obj = -1.0
    arr = [g for g in genuine if g is not None and math.isfinite(float(g))]
    return {
        "objective": float(obj),
        "n_genuine": len(arr),
        "ap_mean": float(np.mean(arr)) if arr else float("nan"),
        "ap_std": float(np.std(arr)) if arr else float("nan"),
    }


def make_dae_component(seeds=None) -> SearchComponent:
    seeds_list = list(seeds) if seeds else list(C.SEEDS)
    return SearchComponent(
        name="dae",
        seeds=seeds_list,
        metric_name="objective",
        define_space=define_space,
        build_parent_args=build_parent_args,
        setup=setup,
        train_one_seed=train_one_seed,
        aggregate=aggregate,
        report_extra_keys=("n_genuine", "ap_mean", "ap_std"),
        max_t=None,   # = len(seeds): max_t pari al numero di semi, ogni trial li percorre tutti (no mismatch coi semi override)
    )


def run_search_stage(cfg: dict) -> dict:
    """Entry dello stage 'search': costruisce il componente DAE e lancia il motore."""
    from lib.search.engine import run_search
    component = make_dae_component(seeds=cfg.get("seeds"))
    return run_search(component, cfg)
