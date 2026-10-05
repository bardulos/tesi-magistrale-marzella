#!/usr/bin/env python
# TAG: DRIVER-PIPELINE | materializza test NPY dall'output del transform e verifica opzionale della validation
"""Materialize test arrays from the transform output directory given by --oracle-dir.

Writes float32 features, int8 labels, and Attack strings. An optional check regenerates
val_X and compares it exactly with --val-ref before materialization.

Usage: python tools/materialize_test_npy.py --oracle-dir DIR --out-dir DIR [--val-ref PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np


# Attack is a trusted object array produced by the preprocessing pipeline.
_ALLOW_PICKLE = True


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def materialize(oracle_dir: Path, out_dir: Path) -> dict:
    """Materializza test_X.npy, test_y.npy, test_Attack.npy da test.npz e test_labels.npz in oracle_dir.

    Restituisce n_rows, n_features, prevalence e lo sha256 di test_X.npy e test_y.npy;
    test_Attack.npy non ha hash.
    """
    oracle_dir, out_dir = Path(oracle_dir), Path(out_dir)
    if not oracle_dir.is_dir():
        raise SystemExit(f"oracle_dir non esiste: {oracle_dir}")
    if not out_dir.is_dir():
        raise SystemExit(f"out_dir non esiste (crearla prima): {out_dir}")

    test_npz = np.load(oracle_dir / "test.npz")
    test_X = test_npz["X"].astype(np.float32)

    test_labels = np.load(oracle_dir / "test_labels.npz", allow_pickle=_ALLOW_PICKLE)
    test_y = test_labels["Label"].astype(np.int8)
    test_Attack = test_labels["Attack"]

    expected_y = (test_Attack != "Benign").astype(np.int8)
    if not np.array_equal(test_y, expected_y):
        n_mismatch = int((test_y != expected_y).sum())
        raise ValueError(f"test_y/Attack non allineati: {n_mismatch} righe discordanti")

    np.save(out_dir / "test_X.npy", test_X)
    np.save(out_dir / "test_y.npy", test_y)
    np.save(out_dir / "test_Attack.npy", test_Attack)

    return {
        "n_rows":        int(test_X.shape[0]),
        "n_features":    int(test_X.shape[1]),
        "prevalence":    float(test_y.mean()),
        "sha256_test_X": _sha256(out_dir / "test_X.npy"),
        "sha256_test_y": _sha256(out_dir / "test_y.npy"),
    }


def val_gate(oracle_dir: Path, ref_val_X: Path) -> bool:
    """Regenerate validation features and compare them exactly with the reference."""
    oracle_dir, ref_val_X = Path(oracle_dir), Path(ref_val_X)
    val_X_regen = np.load(oracle_dir / "val.npz")["X"].astype(np.float32)
    val_X_ref   = np.load(ref_val_X).astype(np.float32)
    return bool(np.array_equal(val_X_regen, val_X_ref))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle-dir", required=True,
                    help="path a repo_v5/runs/preprocessing/transform_cse/")
    ap.add_argument("--out-dir", required=True,
                    help="directory di output (deve esistere)")
    ap.add_argument("--val-ref", default=None,
                    help="path a val_X.npy di riferimento per il gate bit-exact (opzionale)")
    args = ap.parse_args()

    oracle_dir = Path(args.oracle_dir)
    out_dir    = Path(args.out_dir)

    if args.val_ref:
        ref = Path(args.val_ref)
        ok = val_gate(oracle_dir, ref)
        print(f"[GATE val_X] {'OK — bit-exact' if ok else 'FAIL — DIVERGENZA!'} ({ref})")
        if not ok:
            raise SystemExit("Gate val fallito: la ricetta produce val_X diverso dal riferimento.")

    info = materialize(oracle_dir, out_dir)
    print(f"test_X.npy  shape=({info['n_rows']}, {info['n_features']})  dtype=float32")
    print(f"test_y.npy  dtype=int8  prevalenza_attacchi={info['prevalence']:.4f}")
    print(f"sha256 test_X: {info['sha256_test_X']}")
    print(f"sha256 test_y: {info['sha256_test_y']}")
    print("OK")


if __name__ == "__main__":
    main()
