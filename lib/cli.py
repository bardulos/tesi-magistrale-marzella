"""lib/cli.py — glue condiviso degli orchestratori (dispatch YAML -> stage).

Estratto dal pattern identico di `preprocessing.py` / `dae.py` / `secondo_stadio.py`:
parsing degli `--override`, risoluzione dei path relativi, logging minimale e il loop
`run_orchestrator()` di dispatch (carica YAML, valida lo stage, importa pigramente il
modulo dello stage e ne esegue la funzione). Ogni entry-point definisce `STAGES`,
`PATH_KEYS` e il nome-logger e chiama `run_orchestrator(...)`; resta cosi' un dispatcher
sottile.

L'import del modulo dello stage e' LAZY (`importlib.import_module` dentro
`run_orchestrator`): uno stage leggero (puro numpy/pandas) non paga il costo di
TensorFlow.

Convenzione shell-first: la `out_dir` la crea lo script di orchestrazione (bash); qui non
si crea nessuna cartella, si verifica solo che esista (errore esplicito altrimenti),
coerentemente con la validazione che fanno i singoli stage.
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
    """Converte una stringa da `--override key=value` nel tipo Python naturale.

    Usa `yaml.safe_load` per supportare int/float/bool/lista; fallback a stringa.
    Le forme nulle (`null`/`~`/`none`) diventano `None`.
    """
    if raw.strip().lower() in ("null", "~", "none"):
        return None
    try:
        v = yaml.safe_load(raw)
        return v if v is not None else raw
    except yaml.YAMLError:
        return raw


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Applica una lista di `"key=value"` al cfg. Modifica in-place + ritorna."""
    for ov in overrides:
        if "=" not in ov:
            raise SystemExit(f"--override invalido: '{ov}' (atteso 'key=value')")
        key, raw = ov.split("=", 1)
        cfg[key.strip()] = parse_value(raw)
    return cfg


def resolve_paths(cfg: dict, project_root: Path, path_keys: set) -> dict:
    """Risolve i path relativi (solo le chiavi in `path_keys`) rispetto a project_root.

    I path gia' assoluti e i valori non-stringa restano invariati.
    """
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
    """Rigetta (SystemExit) una config marcata `status: illustrative` — reperto storico NON eseguibile.

    VINCOLO machine-checkable: l'header testuale "non eseguibile as-is" era comment-only (non vincolava
    nulla, come mostrato da REVIEW_3); questo SOLLEVA. Le config canoniche non hanno il campo
    -> `cfg.get("status")` is None -> nessun raise. (Bypass deliberato: rimuovere il campo dal file.)
    """
    if cfg.get("status") == "illustrative":
        raise SystemExit(
            f"config '{config_path}' marcata 'status: illustrative' (reperto storico NON eseguibile): "
            f"la riproduzione canonica va via run_pa_search_stage/run_fp_mining_stage con cfg minimale "
            f"(vedi l'header del file).")


def run_orchestrator(stages: dict, path_keys: set, log_name: str,
                     project_root: Path, doc: str | None = None) -> None:
    """Loop di dispatch condiviso degli entry-point.

    Ordine REALE (il docstring descrive cio' che il codice fa): legge il config YAML (argomento posizionale);
    RIGETTA subito le config marcate `status: illustrative` (`reject_if_illustrative`, PRIMA degli override
    -> `--override status=...` non bypassa, scelta intenzionale); poi applica gli `--override`, valida lo
    `stage:` contro `stages`, risolve i path, salva in `out_dir` (se presente nel cfg; deve gia' esistere) lo snapshot
    PRE-merge del YAML (`config.yaml`) E il config effettivo post-override/post-resolve
    (`config_effective.yaml`), e infine importa+esegue il modulo dello stage.
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
