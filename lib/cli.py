"""Dispatcher condiviso per config YAML, override e stage.

Gli entry-point definiscono `STAGES` e `PATH_KEYS`. I moduli stage sono importati
lazy, evitando TensorFlow negli stage che non lo richiedono. `out_dir` deve esistere.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import shutil
import sys
import time
from pathlib import Path

import yaml


def parse_value(raw: str):
    """Converte `--override key=value` con YAML; le forme nulle diventano `None`."""
    if raw.strip().lower() in ("null", "~", "none"):
        return None
    try:
        v = yaml.safe_load(raw)
        return v if v is not None else raw
    except yaml.YAMLError:
        return raw


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Applica gli override al cfg in-place e lo ritorna."""
    for ov in overrides:
        if "=" not in ov:
            raise SystemExit(f"--override invalido: '{ov}' (atteso 'key=value')")
        key, raw = ov.split("=", 1)
        cfg[key.strip()] = parse_value(raw)
    return cfg


def resolve_paths(cfg: dict, project_root: Path, path_keys: set) -> dict:
    """Risolve rispetto a project_root i path relativi nelle chiavi indicate."""
    for key in path_keys:
        if key in cfg and isinstance(cfg[key], str):
            p = Path(cfg[key])
            if not p.is_absolute():
                cfg[key] = str((project_root / p).resolve())
    return cfg


def setup_logging():
    """Logging minimale verso stdout (i moduli stage hanno proprio logging su file)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )


def reject_if_illustrative(cfg: dict, config_path) -> None:
    """Termina se il config e' marcato `status: illustrative`."""
    if cfg.get("status") == "illustrative":
        raise SystemExit(
            f"config '{config_path}' marcata 'status: illustrative' (reperto storico NON eseguibile): "
            f"la riproduzione canonica va via run_pa_search_stage/run_fp_mining_stage con cfg minimale "
            f"(vedi l'header del file).")


def run_orchestrator(stages: dict, path_keys: set, log_name: str,
                     project_root: Path, doc: str | None = None) -> None:
    """Carica e valida il config, risolve i path e avvia lo stage richiesto.

    Rifiuta `status: illustrative` prima degli override e salva in `out_dir` il config
    originale e quello effettivo. La directory deve esistere.
    """
    parser = argparse.ArgumentParser(
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=Path,
                        help="Path al file YAML di configurazione")
    parser.add_argument("--override", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Sovrascrive un campo del YAML (ripetibile)")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger(log_name)

    if not args.config.exists():
        raise SystemExit(f"Config non trovato: {args.config}")

    with args.config.open() as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Il YAML deve contenere un dict al top-level, non {type(cfg).__name__}")

    reject_if_illustrative(cfg, args.config)

    cfg = apply_overrides(cfg, args.override)

    stage = cfg.get("stage")
    if stage is None:
        raise SystemExit("Manca chiave 'stage:' nel YAML")
    if stage not in stages:
        raise SystemExit(
            f"stage '{stage}' sconosciuto. Validi: {sorted(stages.keys())}"
        )

    cfg["project_root"] = str(project_root)
    cfg = resolve_paths(cfg, project_root, path_keys)

    out_dir = cfg.get("out_dir")
    if out_dir:
        out_path = Path(out_dir)
        if not out_path.is_dir():
            raise SystemExit(f"out_dir non esiste (crearla prima, shell-first): {out_path}")
        cfg_snapshot = out_path / "config.yaml"
        try:
            shutil.copy(args.config, cfg_snapshot)
        except shutil.SameFileError:
            pass
        log.info(f"YAML salvato in {cfg_snapshot}")
        eff_path = out_path / "config_effective.yaml"
        with eff_path.open("w") as fh:
            fh.write("# Config EFFETTIVO post-override/post-resolve; config.yaml accanto e' la"
                     " copia PRE-merge dello yaml sorgente.\n")
            yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
        log.info(f"Config effettivo salvato in {eff_path}")

    mod_name, fn_name = stages[stage]
    log.info(f"=== Stage: {stage} ({mod_name}.{fn_name}) ===")
    t0 = time.perf_counter()
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    fn(cfg)
    log.info(f"=== Stage '{stage}' completato in {time.perf_counter() - t0:.1f}s ===")
