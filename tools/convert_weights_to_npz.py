# TAG: DRIVER-PIPELINE | produttore models/
"""Convert canonical CSE .weights.h5 files to the NumPy inference package.

Writes DAE/PA weights, whitening and z-scaler data, fusion thresholds, and preprocessing metadata
under models/cse/, then copies them to inference/modelli/produzione/cse/. Keras 3 kernels are
matched by unique shape; each bias stays paired with its layer kernel. Optimizer state is ignored.

Usage: python tools/convert_weights_to_npz.py (requires h5py and NumPy, not TensorFlow).
"""
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))
import infer  # noqa: E402  (la copia inline della fusione: le soglie salvate DEVONO coincidere col pacchetto)

# --- sorgenti canoniche (CSE) ---
DAE_H5 = ROOT / "runs/dae/loao_fase3/weights/569/seed_2034.weights.h5"
# ATTENZIONE (falso amico): i pesi PA #589 sono sotto l'hash 21a2e100; la dir 589a7bc9 contiene
# "589" nel nome ma e' un'ALTRA config (trial Ray _493). Verificato via hardening_candidates.json
# (num=589 -> base_weights 21a2e100/seed_2037) — NON usare 589a7bc9.
PA_H5 = ROOT / "runs/secondo_stadio/weights_final/21a2e100/seed_2037/best.weights.h5"
WHITENING = ROOT / "runs/dae/output_569/whitening_params.npz"
ZSCALER = ROOT / "runs/dae/output_569/z_scaler.npz"
SCORES = ROOT / "runs/secondo_stadio/fusione/scores.npz"
META_SRC = ROOT / "inference/modelli/produzione/cse"

MODELS = ROOT / "models" / "cse"
WEIGHTS = ROOT / "inference" / "modelli" / "produzione" / "cse"

DAE_SHAPE_TO_NAME = {
    (91, 112): "enc_exp", (112, 80): "enc_mid", (80, 16): "btl",
    (16, 80): "dec_mid", (80, 112): "dec_exp", (112, 34): "out_cont", (112, 57): "out_bin",
}
PA_SHAPE_TO_NAME = {
    (107, 420): ("pa_W0", "pa_b0"), (420, 258): ("pa_W1", "pa_b1"), (258, 1): ("pa_W_out", "pa_b_out"),
}


def _extract_layers(h5_path):
    """{shape_kernel: (kernel, bias)} da un .weights.h5 (Keras 3), solo il gruppo layers/."""
    out = {}
    with h5py.File(h5_path, "r") as f:
        layers = f["layers"]
        for lname in layers:
            vg = layers[lname].get("vars")
            if vg is None or "0" not in vg:
                continue
            k = vg["0"][()]
            if getattr(k, "ndim", 0) != 2:
                continue
            b = vg["1"][()] if "1" in vg else np.zeros(k.shape[1], dtype=k.dtype)
            out[tuple(k.shape)] = (np.asarray(k, np.float32), np.asarray(b, np.float32))
    return out


def convert_dae():
    layers = _extract_layers(DAE_H5)
    d = {}
    tot = 0
    for shape, (k, b) in layers.items():
        name = DAE_SHAPE_TO_NAME.get(shape)
        if name is None:
            raise ValueError(f"DAE: kernel shape inattesa {shape}")
        d[f"W_{name}"], d[f"b_{name}"] = k, b
        tot += k.size + b.size
    missing = {f"W_{n}" for n in DAE_SHAPE_TO_NAME.values()} - set(d)
    if missing:
        raise ValueError(f"DAE: layer mancanti {missing}")
    print(f"[DAE] {len(layers)} layer, {tot} parametri (atteso 41355)")
    assert tot == 41355, f"DAE parametri {tot} != 41355"
    return d


def convert_pa():
    layers = _extract_layers(PA_H5)
    d = {}
    tot = 0
    for shape, (k, b) in layers.items():
        names = PA_SHAPE_TO_NAME.get(shape)
        if names is None:
            raise ValueError(f"PA: kernel shape inattesa {shape}")
        wname, bname = names
        d[wname], d[bname] = k, b
        tot += k.size + b.size
    d["pa_n_layers"] = np.int64(2)
    print(f"[PA] {len(layers)} layer, {tot} parametri (atteso 154237)")
    assert tot == 154237, f"PA parametri {tot} != 154237"
    # Copy the cached whitening and scaling parameters used by inference.
    wp = np.load(WHITENING)
    zs = np.load(ZSCALER)
    d["mu_R"] = np.asarray(wp["mu"], np.float32)
    d["W"] = np.asarray(wp["L"], np.float32)
    d["sigma_R"] = np.asarray(wp["sigma_R"], np.float32)
    d["mu_z"] = np.asarray(zs["mu_z"], np.float32)
    d["sigma_z"] = np.asarray(zs["sd_z"], np.float32)
    return d


def calibra_soglie():
    """Build fusion parameters for modes A/B from cached scores."""
    d = np.load(SCORES, allow_pickle=True)
    modes = infer.build_fusion_modes(d["sdae_tr"], d["lg_tr"], d["sdae_sel"], d["lg_sel"])
    out = {}
    for name, p in modes.items():
        out[name] = {
            "mu": [float(x) for x in p["mu"]],
            "sd": [float(x) for x in p["sd"]],
            "tau_z": p["tau_z"], "theta_z": p["theta_z"],
            "theta_union": (None if not np.isfinite(p["theta_union"]) else float(p["theta_union"])),
            "tau_hi_z": (None if not np.isfinite(p["tau_hi_z"]) else float(p["tau_hi_z"])),
            "ricetta": infer.FUSION_MODES[name],
        }
    return out


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    np.savez(MODELS / "dae.npz", **convert_dae())
    np.savez(MODELS / "pa.npz", **convert_pa())
    (MODELS / "soglie.json").write_text(json.dumps(calibra_soglie(), indent=2, ensure_ascii=False))
    for f in ("feature_order.json", "scaler_params.json", "transform_meta.json"):
        shutil.copy(META_SRC / f, MODELS / f)
    print(f"[models] scritto {MODELS}")
    # Populate inference weights from the canonical models output.
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    for f in ("dae.npz", "pa.npz", "soglie.json", "feature_order.json",
              "scaler_params.json", "transform_meta.json"):
        shutil.copy(MODELS / f, WEIGHTS / f)
    print(f"[weights] copiato models/cse -> {WEIGHTS}")


if __name__ == "__main__":
    main()
