#!/usr/bin/env python
# TAG: ONE-SHOT | 2026-07-13 | WP-4_v2 Fase 0 | vedi docs/refactoring_censimento.md
"""tools/pa_wp4v2_repass.py — Fase 0 di WP-4_v2: re-pass PA #589 con salvataggio pesi per-epoca.

Mirror del re-pass di WP-3b(i) (driver non incluso, invariato) con tre differenze deliberate:
  1. **Partizione a tre vie v2** dei benigni: gradiente_v2/monitor_v2/select. Il monitor_v2 e'
     200k (4x il monitor_v1 di WP-3b(i)), seme FISSO `MON_SEED_V2` diverso da `MON_SEED` di
     WP-3b(i) (quel monitor_v1 e' archiviato, verdetto negativo, non riusato qui).
     Partizione via lib.secondo_stadio.data.partition_gradient_monitor (stessa
     funzione di WP-4_v2 precalc/search: nessuna necessita' di salvare gli indici, sono
     ri-derivabili deterministicamente da (ben_train, mon_n, seed)).
  2. **Salvataggio pesi a OGNI epoca** (.npz, e0 pre-gradiente + 100 = 101 file/seme) — prerequisito
     per la Fase 2 (ricerca Optuna sui parametri delle pseudo-monitor: serve poter fare forward
     su pesi congelati di qualunque epoca, senza ri-addestrare). save_weights_npz da
     lib.secondo_stadio.mlp_numpy (stesso schema di tools/dae_entropy_contam.py, isolato in lib
     senza side-effect d'import).
  3. **Nessuna generazione di pseudo-monitor qui**: quella e' nel loop Optuna di Fase 2 (parametri
     delta/alpha_lo/alpha_hi DA CERCARE, non fissi). Si logga solo l'oracolo vero (history_ap,
     su TUTTI gli attacchi del val — decisione di pianificazione: la prevalenza non conta per una
     correlazione di TREND, la stabilita' epoca-per-epoca si') e le due loss pure (train, val vero).

Tre curve per-epoca (NON 5+4 come WP-3b(i), che aveva bisogno delle curve vs-pseudo):
  history_ap         — AUC-PR oracolo, ben_select(0) vs att_oracle-TUTTI(1) — il bersaglio con cui
                        Fase 2 correla la BCE-monitor.
  history_train_loss — BCE+L2 TOTALE sui batch (da logs["loss"], regolarizzata: dropout+L2 attivi).
  history_val_loss   — BCE dati puri, ben_select(0) vs att_oracle-TUTTI(1) (stesso lato di history_ap,
                        loss anziche' AUC-PR — diagnostica di coerenza, non il monitor label-free).

Il re-pass gira senza early-stop fino a --epochs (nessuno stop qui: lo stop si SIMULA offline in
Fase 3 sulla BCE-monitor delle pseudo calibrate). NON e' ri-selezione: valida il METODO d'arresto
(caveat: non bit-exact vs pesi di compressione, gradiente diverso da WP-3b(i) e da weights_final).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.utils import setup_blas_env  # noqa: E402

_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--seed", type=int, required=True)
_p.add_argument("--epochs", type=int, default=100)          # cap; nessun early-stop
_p.add_argument("--mon-n", type=int, default=None)            # benigni-monitor_v2 (default C.WP4V2_MON_N)
_p.add_argument("--threads", type=int, default=1)
_p.add_argument("--candidates", default=str(REPO / "runs/secondo_stadio/pa_compress/hardening_candidates.json"))
_p.add_argument("--cache-dir", default=str(REPO / "runs/dae/output_569"))
_p.add_argument("--labels-val", default=str(REPO / "runs/preprocessing/val_attack_npy/val_Attack.npy"))
_p.add_argument("--out", default=None)
_p.add_argument("--weights-dir", default=None,
                help="se presente, salva i pesi a ogni epoca in DIR/weights_epoch_NNN.npz (e0=000). "
                     "Prerequisito di Fase 1/2: se assente il re-pass logga solo le curve.")
a = _p.parse_args()

setup_blas_env(int(a.threads), deterministic=False)

import numpy as np  # noqa: E402

from lib.dae.objective import seed_ap                       # noqa: E402
from lib.secondo_stadio import constants as C                # noqa: E402
from lib.secondo_stadio.data import (assemble_phi,           # noqa: E402
                                     load_phi_cache,
                                     partition_gradient_monitor,
                                     predict_phi)
from lib.secondo_stadio.metrics import bce                   # noqa: E402
from lib.secondo_stadio.mlp_numpy import save_weights_npz    # noqa: E402
from lib.secondo_stadio.model import build_mlp, h1_from_config  # noqa: E402
from lib.secondo_stadio.pa_search import _make_pa_train_ds   # noqa: E402

# MON_SEED_V2/WP4V2_MON_N: costanti centralizzate in lib.secondo_stadio.constants (condivise con
# pa_wp4v2_search.py precalc/search, che devono ri-derivare ESATTAMENTE la stessa partizione).


def load_config():
    """Config #589 config-first: hardening_candidates.json (num 589) + cross-check DB Optuna (params
    trial 589). Identico a tools/pa_es_history.py:80-96 (stesso contratto config-first)."""
    cands = json.loads(Path(a.candidates).read_text())
    cfg = next(c["config"] for c in cands if str(c["num"]) == "589")
    try:
        sys.path.insert(0, str(REPO / "tools"))
        from monitor_search4 import pa_key
        import optuna
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        db = (REPO / "runs/secondo_stadio/pa_search/optuna_study.db").resolve()
        st = optuna.load_study(study_name="pa_search", storage=f"sqlite:///file:{db}?mode=ro&uri=true")
        t589 = next(t for t in st.trials if t.number == 589)
        assert pa_key(t589.params) == pa_key(cfg), "config #589: hardening_candidates != DB"
        db_ok = True
    except Exception as e:  # noqa: BLE001
        db_ok = f"cross-check DB saltato: {e}"
    return cfg, db_ok


def main():
    import tensorflow as tf
    tf.keras.utils.set_random_seed(int(a.seed))
    tf.config.threading.set_intra_op_parallelism_threads(int(a.threads))
    tf.config.threading.set_inter_op_parallelism_threads(int(a.threads))

    cfg, db_ok = load_config()
    delta, alo, ahi = float(cfg["delta"]), float(cfg["alpha_lo"]), float(cfg["alpha_hi"])
    mon_n = int(a.mon_n) if a.mon_n is not None else int(C.WP4V2_MON_N)

    sh = load_phi_cache(a.cache_dir, a.labels_val, val_frac=C.VAL_FRAC,
                        split_seed=C.SPLIT_SEED, es_n_attacks=0, attack_split=False)
    Zs, R = sh["Zs"], sh["R"]
    ben_train = sh["ben_pos"][sh["idx_train"]]          # 80% (assoluti) — 1.420.078
    ben_select = sh["ben_pos"][sh["idx_select"]]        # 20% (INTATTO) — 355.020
    att_oracle = sh["att_es"]                           # es_n_attacks=0 -> att_pos INTERO (~1.046.755)

    # --- partizione v2 a tre vie: monitor_v2 ritagliato da DENTRO ben_train (seme FISSO v2) ---
    ben_gradient, ben_monitor = partition_gradient_monitor(ben_train, mon_n, C.MON_SEED_V2)
    assert len(np.intersect1d(ben_gradient, ben_select)) == 0
    assert len(np.intersect1d(ben_monitor, ben_select)) == 0

    # --- modello config-first (identico a _fit_pa) ---
    params = {"h1": h1_from_config(cfg), "n_layers": int(cfg["n_layers"]),
              "ratio": float(cfg["ratio"]), "dropout": float(cfg["dropout"]), "l2_reg": C.L2_FIXED_S2}
    model = build_mlp(params)
    model.compile(optimizer=tf.keras.optimizers.Adam(float(cfg["lr"])), loss="binary_crossentropy")

    weights_dir = Path(a.weights_dir) if a.weights_dir else None
    if weights_dir is not None:
        weights_dir.mkdir(parents=True, exist_ok=True)
        save_weights_npz(model.get_weights(), weights_dir / "weights_epoch_000.npz")

    y_true_oracle = np.r_[np.zeros(len(ben_select)), np.ones(len(att_oracle))]

    class RepassCb(tf.keras.callbacks.Callback):
        def __init__(self):
            self.history_ap, self.loss_train, self.loss_val = [], [], []

        def on_epoch_end(self, epoch, logs=None):
            m = self.model
            s_bs = predict_phi(m, Zs, R, ben_select)
            s_ae = predict_phi(m, Zs, R, att_oracle)
            scores = np.r_[s_bs, s_ae]
            self.history_ap.append(seed_ap(y_true_oracle, scores))
            self.loss_train.append(float(logs["loss"]) if logs and "loss" in logs else float("nan"))
            self.loss_val.append(bce(y_true_oracle, scores))
            if weights_dir is not None:
                save_weights_npz(m.get_weights(), weights_dir / f"weights_epoch_{epoch + 1:03d}.npz")

    cb = RepassCb()
    train_ds, steps = _make_pa_train_ds(Zs, R, ben_gradient, C.BATCH_FIXED, int(a.seed), delta, alo, ahi)
    t0 = time.time()
    model.fit(train_ds, epochs=int(a.epochs), steps_per_epoch=steps, callbacks=[cb], verbose=0)
    dt = time.time() - t0

    out = Path(a.out) if a.out else REPO / "runs/secondo_stadio/wp4v2_repass" / f"seed_{a.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    weights_dir_rec = None
    if weights_dir is not None:
        try:
            weights_dir_rec = str(weights_dir.relative_to(REPO))
        except ValueError:
            weights_dir_rec = str(weights_dir)  # fuori dal repo (es. smoke in scratchpad): path assoluto
    rec = {"seed": int(a.seed), "n_epochs": len(cb.history_ap), "epochs_cap": int(a.epochs),
           "n_gradient": int(len(ben_gradient)), "n_monitor": int(len(ben_monitor)),
           "n_select": int(len(ben_select)), "n_att_oracle": int(len(att_oracle)),
           "mon_seed_v2": int(C.MON_SEED_V2),
           "history_ap": cb.history_ap, "history_train_loss": cb.loss_train,
           "history_val_loss": cb.loss_val,
           "weights_dir": weights_dir_rec,
           "config": cfg, "sec_total": dt}
    out.write_text(json.dumps(rec))

    # --- riassunto smoke ---
    print("=" * 84)
    print(f"WP-4_v2 FASE 0 — RE-PASS PA #589 — seme {a.seed}  ({len(cb.history_ap)} epoche, cap {a.epochs})")
    print(f"  config-first DB cross-check: {db_ok}")
    print(f"  CARDINALITA' POOL: gradiente_v2 {len(ben_gradient):,} · monitor_v2 {len(ben_monitor):,} · "
          f"select {len(ben_select):,}  (disgiunti: assert OK)")
    print(f"  ORACOLO: att_oracle (TUTTI gli attacchi) {len(att_oracle):,}")
    def show(name, h):
        print(f"    {name:<20} [{', '.join(f'{v:.4f}' for v in h[:10])}{' …' if len(h) > 10 else ''}]")
    print("  CURVE (prime ~10 epoche):")
    show("history_ap", cb.history_ap)
    show("loss_train", cb.loss_train)
    show("loss_val", cb.loss_val)
    if weights_dir is not None:
        n_npz = len(list(weights_dir.glob("weights_epoch_*.npz")))
        print(f"  PESI: {n_npz} file .npz in {weights_dir_rec} (attesi {len(cb.history_ap) + 1})")
    print(f"  TEMPI: {dt:.1f}s totali · {dt/max(1,len(cb.history_ap)):.2f}s/epoca "
          f"→ cap 100 ≈ {dt/max(1,len(cb.history_ap))*100/60:.1f} min")
    try:
        rel = out.relative_to(REPO)
    except ValueError:
        rel = out
    print(f"  JSON: {rel}")
    print("=" * 84)


if __name__ == "__main__":
    main()
