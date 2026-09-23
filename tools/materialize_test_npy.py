#!/usr/bin/env python
# TAG: DRIVER-PIPELINE | 2026-07-13 | materializzazione test npy dall'oracolo + val_gate (testato; tag armonizzato da ONE-SHOT storico, censimento A3.5) | vedi docs/refactoring_censimento.md
"""tools/materialize_test_npy.py — materializza test_X.npy / test_y.npy dall'oracolo repo_v5.

Ricetta (identica a quella usata per val_X.npy / val_y.npy nella search):
  test_X.npy  = test.npz["X"].astype(float32)          shape [N, 91]
  test_y.npy  = test_labels.npz["Label"].astype(int8)  shape [N]   (0=benigno, 1=attacco)
  test_Attack.npy = test_labels.npz["Attack"]          shape [N]   dtype=object (stringhe)

Gate val (obbligatorio prima di fidarsi della ricetta):
  Rigenera val_X da val.npz["X"] e ne calcola lo sha256; lo confronta con il file di
  riferimento indicato (--val-ref, es. il val_X.npy deployato su un VPS). Bit-exact.

Uso:
  python tools/materialize_test_npy.py \\
      --oracle-dir <HOME>/tesi/repo_v5/runs/preprocessing/transform_cse \\
      --out-dir <HOME>/tesi/repov6/runs/preprocessing/test_npy \\
      [--val-ref /path/a/val_X.npy]   # confronto sha256 per gate
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np


# allow_pickle=True sulle label: colonna Attack = array di stringhe Python (dtype=object)
# prodotto dal preprocessing (pipeline controllata), mai input esterno.
_ALLOW_PICKLE = True


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def materialize(oracle_dir: Path, out_dir: Path) -> dict:
    """Materializza test_X.npy, test_y.npy, test_Attack.npy a partire dall'oracolo.

    Returns dict con sha256 di ogni file prodotto (per verifica esterna).
    """
    oracle_dir, out_dir = Path(oracle_dir), Path(out_dir)
    if not oracle_dir.is_dir():
        raise SystemExit(f"oracle_dir non esiste: {oracle_dir}")
    if not out_dir.is_dir():
        raise SystemExit(f"out_dir non esiste (crearla prima): {out_dir}")

    # Carica test.npz
    test_npz = np.load(oracle_dir / "test.npz")
    test_X = test_npz["X"].astype(np.float32)

    # Carica test_labels.npz (allow_pickle per Attack dtype=object)
    test_labels = np.load(oracle_dir / "test_labels.npz", allow_pickle=_ALLOW_PICKLE)
    test_y = test_labels["Label"].astype(np.int8)
    test_Attack = test_labels["Attack"]  # dtype=object (stringhe)

    # Controllo di coerenza interno: test_y == (Attack!="Benign") bit-exact
    expected_y = (test_Attack != "Benign").astype(np.int8)
    if not np.array_equal(test_y, expected_y):
        n_mismatch = int((test_y != expected_y).sum())
        raise ValueError(f"test_y/Attack non allineati: {n_mismatch} righe discordanti")

    # Salvataggio
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
    """Rigenera val_X da val.npz e confronta bit-exact col file di riferimento.

    Carica entrambi come array float32 e usa np.array_equal (zero-divergenza).
    True = ricette identiche → la stessa ricetta applicata al test è affidabile.
    False = divergenza (dtype/valori diversi → interrompere prima di distribuire i file).
    """
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

    # Gate val (se fornito il riferimento)
    if args.val_ref:
        ref = Path(args.val_ref)
        ok = val_gate(oracle_dir, ref)
        print(f"[GATE val_X] {'OK — bit-exact' if ok else 'FAIL — DIVERGENZA!'} ({ref})")
        if not ok:
            raise SystemExit("Gate val fallito: la ricetta produce val_X diverso dal riferimento.")

    # Materializzazione
    info = materialize(oracle_dir, out_dir)
    print(f"test_X.npy  shape=({info['n_rows']}, {info['n_features']})  dtype=float32")
    print(f"test_y.npy  dtype=int8  prevalenza_attacchi={info['prevalence']:.4f}")
    print(f"sha256 test_X: {info['sha256_test_X']}")
    print(f"sha256 test_y: {info['sha256_test_y']}")
    print("OK")


if __name__ == "__main__":
    main()
