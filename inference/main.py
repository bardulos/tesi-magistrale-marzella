#!/usr/bin/env python3
"""main.py — orchestratore interattivo del pacchetto NIDS per reti PMI.

Compone comandi di infer.py e train.py, mostrando le opzioni disponibili e il comando eseguito.
Gestisce retrain multi-seme con limiti di memoria e seleziona il seme di produzione.
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
INFER = DIR / "infer_venv" / "bin" / "python"
TRAIN = DIR / "train_venv" / "bin" / "python"
MODELLI = DIR / "modelli"
DATI_INF = DIR / "dati" / "inferenza"
DATI_TRAIN = DIR / "dati" / "addestramento"
DATI_BENCH = DIR / "dati" / "benchmark"
BLAS_INFER = 4          # Deve restare allineato a infer.BLAS_DEFAULT.

# --- Dimensionamento automatico del retrain multi-seme -----------------------------------------
# I semi per ondata dipendono dalla RAM disponibile e sono limitati ai thread logici. Oltre i core
# fisici, ogni processo usa un thread logico e BLAS=1.
SIZE_REF = 1_755_000          # riferimento CSE solo informativo, non usato (train.py ha la sua copia)
# Riserva fissa per il sistema; la RAM disponibile e' rilevata al lancio.
RAM_RESERVE_GB = 1.0
# RAM totale minima per il riaddestramento (non per l'inferenza, che gira in ~50 MB): sotto questa
# soglia il retrain viene rifiutato invece di essere tentato. TensorFlow costa ~2 GiB fissi e un seme
# arriva a ~5 GiB su ~3M righe (misurazione di riferimento), quindi su una macchina piccola il
# dimensionamento sceglie comunque 1 seme e il tetto del cgroup lo uccide dopo ore di paginazione.
# La soglia e' 14,5 e non 16,0 perche' MemTotal e' sempre sotto la taglia nominale (firmware e grafica
# integrata ne riservano una parte): misurati 62,72 GiB su 64 nominali e 7,44 su 8. Una macchina da
# 16 GB dichiara ~15 GiB e deve passare; una da 8 ne dichiara ~7,4 e deve essere fermata.
RAM_MIN_TOTAL_GB = 14.5
# Margine del cgroup oltre il bersaglio, per contenere eventuali sforamenti dell'ondata.
CONTAINMENT_MARGIN_GB = 0.5
# Stima della memoria per riga durante il retrain completo, misurata su 2.966.479 righe:
# picco 5,14 GiB, di cui circa 2 GiB per TensorFlow e l'interprete. Il valore è arrotondato
# a 1.200 byte per riga per lasciare margine. Il retrain usa tutte le righe per default.
BYTES_PER_ROW = 1_200         # Stima di memoria per riga nel retrain completo.
TF_OVERHEAD_GB = 2.0          # runtime TensorFlow + interprete (indipendente dalla taglia)
# Aggiornare insieme righe e RSS quando si ripete la misurazione della memoria.
CANARY_FULL_CHAIN_N_ROWS = 2_966_479   # righe della misurazione di riferimento
CANARY_FULL_CHAIN_RSS_GB = 5.14        # picco RSS in GiB nella misurazione di riferimento


def _py(which):
    p = INFER if which == "infer" else TRAIN
    return str(p) if p.exists() else sys.executable


def _check_venv(which):
    """Verifica il venv richiesto e mostra il comando d'installazione se manca."""
    if (INFER if which == "infer" else TRAIN).exists():
        return True
    nome = "infer_venv (inferenza)" if which == "infer" else "train_venv (training)"
    arg = "infer" if which == "infer" else "train"
    print(f"\n  ATTENZIONE: manca il venv {nome}.")
    print(f"  Installalo con:   bash install.sh {arg}")
    return False


def _scopri_csv(directory):
    """CSV non symlink presenti nella cartella, ordinati per nome."""
    d = Path(directory)
    return sorted(f for f in d.iterdir()
                  if f.is_file() and not f.is_symlink() and f.suffix == ".csv") if d.exists() else []


def _scopri_modelli():
    """Ritorna i modelli validi, prima quelli di produzione."""
    trovati = {}
    for sub in ("produzione", "sperimentali"):
        d = MODELLI / sub
        if d.exists():
            for m in sorted(d.iterdir()):
                if m.is_dir() and (m / "dae.npz").exists():
                    trovati[m.name] = (m, sub)
    return trovati


def _scegli_da_lista(voci, etichetta, prompt_vuoto):
    """Mostra una lista numerata di (display, valore) e ritorna il valore scelto ('' se non valido).

    Con lista vuota chiede il valore a mano con prompt_vuoto.
    """
    if not voci:
        return ask(prompt_vuoto)
    print(etichetta)
    for i, (disp, _) in enumerate(voci, 1):
        print(f"    {i}) {disp}")
    scelta = ask("Numero", "1")
    try:
        return voci[int(scelta) - 1][1]
    except (ValueError, IndexError):
        print("  scelta non valida: annullo")
        return ""


def ask(prompt, default=""):
    r = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    return r or default


def ask_yesno(prompt, default=True):
    d = "S/n" if default else "s/N"
    r = input(f"{prompt} [{d}]: ").strip().lower()
    return default if not r else r.startswith("s")


def run(py, script, args, background=False, cpus=None, threads=None):
    """Compone ed esegue un comando, con pinning CPU e thread opzionali."""
    cmd = [py, str(DIR / script)] + args
    if cpus:
        cmd = ["taskset", "-c", cpus] + cmd
    env = None
    if threads:
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
                   MKL_NUM_THREADS=str(threads))
    prefix = f"OMP_NUM_THREADS={threads} " if threads else ""
    print("\n$ " + prefix + " ".join(shlex.quote(c) for c in cmd) + ("  &" if background else ""))
    if background:
        return subprocess.Popen(["setsid"] + cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, env=env)
    return subprocess.run(cmd, env=env)


def _meminfo_gb(chiave):
    """Legge un campo di /proc/meminfo in GiB, oppure ritorna None."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(chiave + ":"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return None


def ram_totale_gb():
    """Ritorna la RAM fisica totale, non quella disponibile."""
    return _meminfo_gb("MemTotal")


def ram_sufficiente_per_retrain(totale_gb=None):
    """Verifica la RAM totale minima; se non e' rilevabile, consente il tentativo."""
    tot = ram_totale_gb() if totale_gb is None else totale_gb
    if tot is None:
        return True, None
    return tot >= RAM_MIN_TOTAL_GB, tot


def _ram_gb():
    """Ritorna la RAM disponibile per dimensionare i semi dell'ondata."""
    return _meminfo_gb("MemAvailable")


def _sibling_sets(sys_root="/sys/devices/system/cpu"):
    """Raggruppa CPU logiche per core fisico usando la topologia del kernel."""
    gruppi = {}
    root = Path(sys_root)
    for cpu in sorted(root.glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        topo = cpu / "topology"
        try:
            core = (topo / "core_id").read_text().strip()
            pkg = (topo / "physical_package_id").read_text().strip()
        except OSError:
            continue
        gruppi.setdefault((int(pkg), int(core)), []).append(int(cpu.name[3:]))
    return [sorted(v) for _, v in sorted(gruppi.items())]


def _physical_cores_lscpu():
    """Ripiego quando la topologia sysfs non e' leggibile."""
    try:
        out = subprocess.run(["lscpu", "-p=Core,Socket"], capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:
        return None
    righe = {ln.strip() for ln in out.splitlines() if ln.strip() and not ln.startswith("#")}
    return len(righe) or None


def physical_cores(sys_root="/sys/devices/system/cpu"):
    """Ritorna i core fisici, usati per scegliere la strategia di pinning."""
    gruppi = _sibling_sets(sys_root)
    return len(gruppi) if gruppi else _physical_cores_lscpu()


def logical_cores(sys_root="/sys/devices/system/cpu"):
    """Ritorna i thread logici, limite massimo dei semi per ondata."""
    gruppi = _sibling_sets(sys_root)
    if gruppi:
        return sum(len(g) for g in gruppi)
    return os.cpu_count()


def count_csv_rows(path):
    """Conta le righe dati a blocchi, senza caricare il file in RAM."""
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            n += b.count(b"\n")
    return max(n - 1, 0)


def footprint_gb(n_rows):
    """Stima la RAM per retrain in funzione delle righe e dell'overhead TensorFlow."""
    return n_rows * BYTES_PER_ROW / 2**30 + TF_OVERHEAD_GB


def n_seeds_auto(avail_gb, f_gb, max_par):
    """Calcola quanti semi entrano in RAM, entro max_par e lasciando la riserva di sistema."""
    if not avail_gb or not f_gb or not max_par:
        return 1
    return max(1, min(int((avail_gb - RAM_RESERVE_GB) / f_gb), max_par))


def containment_cap_gb(avail_gb):
    """Calcola il tetto cgroup dell'ondata, con margine sul dimensionamento."""
    if not avail_gb:
        return None
    return max(1.0, avail_gb - CONTAINMENT_MARGIN_GB)


def taskset_sets(n_proc, sib_sets):
    """Ripartisce core fisici fra processi; ritorna pinning taskset e core per processo."""
    if not sib_sets or n_proc < 1:
        return [], 0
    per = max(1, len(sib_sets) // n_proc)
    fette = []
    for i in range(n_proc):
        gruppo = sib_sets[i * per:(i + 1) * per] or [sib_sets[i % len(sib_sets)]]
        fette.append(",".join(str(c) for g in gruppo for c in sorted(g)))
    return fette, per


def taskset_logici(n_proc, sib_sets):
    """Assegna un thread logico per processo, distribuendo prima un thread per core."""
    if not sib_sets or n_proc < 1:
        return [], 0
    profondita = max(len(g) for g in sib_sets)
    logici = [g[k] for k in range(profondita) for g in sib_sets if k < len(g)]
    return [str(logici[i % len(logici)]) for i in range(n_proc)], 1


def pinning_ondata(n_proc, p_cores, sib_sets):
    """Sceglie il pinning dell'ondata in base al numero di core fisici."""
    if p_cores and n_proc > p_cores:
        return taskset_logici(n_proc, sib_sets)
    return taskset_sets(n_proc, sib_sets)


def piano_turni(n_semi, n_auto):
    """Divide i semi in ondate sequenziali di massimo n_auto elementi."""
    par = max(1, n_auto)
    return [list(range(i, min(i + par, n_semi))) for i in range(0, n_semi, par)]


def _riga_seme(py, dataset, out, seed, cpus, threads):
    """Compone la riga shell per un seme, con pinning, thread e log dedicato."""
    cmd = [py, str(DIR / "train.py"), "--dataset", dataset, "--out-dir", str(out),
           "--seed", str(seed)]
    if cpus:
        cmd = ["taskset", "-c", cpus] + cmd
    env = ""
    if threads:
        env = (f"OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} "
               f"MKL_NUM_THREADS={threads} ")
    log = shlex.quote(str(Path(out) / "train.log"))
    return env + " ".join(shlex.quote(c) for c in cmd) + f" > {log} 2>&1 &"


def lancia_ondata(righe, cap_gb, out_dirs=None):
    """Lancia l'ondata in un unico cgroup con MemoryMax e scrive l'exit code di ogni seme.

    Senza cap_gb o senza systemd-run l'ondata parte senza cgroup; l'exit code va in
    <out>/.exitcode solo per le cartelle passate in out_dirs.
    """
    parts = []
    pid_vars = []
    n = len(righe)
    for i, r in enumerate(righe):
        parts.append(r)
        pid_vars.append(f"_P{i}")
        parts.append(f"_P{i}=$!")
        if i < n - 1:
            parts.append("sleep 2")   # Sfalsa i picchi RAM dei semi nello stesso cgroup.
    dirs = out_dirs or [None] * len(righe)
    for var, out in zip(pid_vars, dirs):
        if out:
            ec = shlex.quote(str(Path(out) / ".exitcode"))
            parts.append(f"wait ${var}; echo $? > {ec}")
        else:
            parts.append(f"wait ${var}")
    inner = "\n".join(parts)
    cmd = ["bash", "-c", inner]
    if cap_gb:
        if shutil.which("systemd-run"):
            cmd = ["systemd-run", "--user", "--scope", "-p", f"MemoryMax={cap_gb:.1f}G", "--"] + cmd
        else:
            print("  AVVISO: systemd-run non trovato — contenimento MemoryMax disabilitato (ondata senza cgroup).")
    return subprocess.run(cmd)


def _annota_dimensionamento(out_dir, info):
    """Aggiunge al train_report.json i dati di dimensionamento senza perdere le chiavi esistenti."""
    rep = Path(out_dir) / "train_report.json"
    if not rep.exists():
        return
    try:
        r = json.loads(rep.read_text())
    except (json.JSONDecodeError, OSError) as e:      # report troncato (kill a meta' scrittura)
        print(f"  [{Path(out_dir).name}] train_report.json illeggibile ({e}): annotazione saltata")
        return
    r["dimensionamento"] = info
    rep.write_text(json.dumps(r, indent=2))


def op_inferenza():
    """Gestisce l'inferenza dal menu."""
    print("\n-- Inferenza --")
    mod = _scopri_modelli()
    if not mod:
        print("  nessun modello disponibile in modelli/: esegui prima un riaddestramento.")
        return
    voci_mod = [(f"{n}  ({sub})", n) for n, (_, sub) in mod.items()]
    modello = _scegli_da_lista(voci_mod, "\n  Modelli disponibili:", "")
    if not modello:
        return
    modo = ask("Modalita' (A=FPR minimo, pochi allarmi; B=alta recall, piu' copertura)", "A").upper()
    modo = modo if modo in ("A", "B") else "A"
    live = ask_yesno("Cattura LIVE dalla rete? (no = analizza un file CSV)", default=False)
    args = ["--model", modello, "--mode", modo]
    if not _check_venv("infer"):
        return
    if live:
        iface = ask("Interfaccia di rete (vuoto = auto)")
        if iface:
            args += ["--iface", iface]
        run(_py("infer"), "infer.py", ["--live"] + args, threads=BLAS_INFER)
    else:
        csv_list = _scopri_csv(DATI_INF)
        voci = [(f.name, str(f)) for f in csv_list]
        f = _scegli_da_lista(voci, "\n  CSV in dati/inferenza/:",
                             "Percorso del CSV nProbe (dati/inferenza/ e' vuota)")
        if not f:
            print("  nessun file: annullo")
            return
        run(_py("infer"), "infer.py", ["--batch", f] + args, threads=BLAS_INFER)


def op_retrain():
    print("\n-- Riaddestramento label-free --")
    # Requisito di macchina PRIMA di ogni domanda: su meno di 16 GB il retrain non parte proprio.
    # Non e' prudenza eccessiva: sotto soglia il dimensionamento sceglierebbe comunque 1 seme e lo
    # lancerebbe sotto un tetto di memoria che lo uccide a meta' strada, dopo ore di paginazione.
    ok_ram, tot_gb = ram_sufficiente_per_retrain()
    if not ok_ram:
        print(f"\n  BLOCCATO: il riaddestramento richiede una macchina da almeno 16 GB di RAM.")
        print(f"  Questa macchina ne ha {tot_gb:.1f} GiB in tutto.")
        print(f"  Il solo runtime TensorFlow ne occupa ~{TF_OVERHEAD_GB:.0f} GiB e ogni seme parte da "
              f"~5 GiB: nessun parametro (nemmeno --max-rows) puo' compensare, e tentare comunque "
              f"significa perdere ore e finire ucciso dal contenimento di memoria.")
        print(f"  L'INFERENZA invece gira benissimo qui (picco ~50 MB): usa la voce 1 del menu.")
        print(f"  Per riaddestrare: una macchina con >= 16 GB, oppure riaddestra altrove e copia il "
              f"modello prodotto in modelli/.")
        return
    # Un solo dataset in ingresso, gia' contaminato al naturale (nessuna domanda benigno/sospetto).
    dataset = _scegli_da_lista([(f.name, str(f)) for f in _scopri_csv(DATI_TRAIN)],
                               "\n  Dataset in dati/addestramento/ (il tuo traffico catturato):",
                               "Percorso del CSV del traffico (dati/addestramento/ e' vuota)")
    if not dataset:
        return
    nome = ask("Nome del modello da produrre (cartella in modelli/sperimentali/)", "lan")
    if not _check_venv("train"):
        return

    # La macchina dimensiona il parallelismo sicuro (semi per ondata in base alla RAM). Il limite
    # sono i thread logici (non i core fisici): si riempie la RAM sfruttando anche l'iper-threading;
    # oltre i core reali ogni processo va a 1 thread logico + BLAS=1 (v. pinning_ondata sotto).
    p_cores = physical_cores()
    l_cores = logical_cores()
    avail = _ram_gb()
    try:
        n_rows = count_csv_rows(dataset)
    except OSError as e:
        print(f"  impossibile leggere {dataset}: {e}")
        return
    f_gb = footprint_gb(n_rows)
    n_auto = max(1, n_seeds_auto(avail, f_gb, l_cores))
    n_eff = n_rows
    cap = containment_cap_gb(avail)
    print("\n  -- Dimensionamento automatico --")
    print(f"    core fisici (P)           : {p_cores if p_cores else 'non rilevabili'}")
    print(f"    thread logici (T)         : {l_cores if l_cores else 'non rilevabili'}   (tetto ai semi per ondata)")
    print(f"    RAM disponibile (A)       : {avail:.1f} GB" if avail else
          "    RAM disponibile (A)       : non rilevabile")
    print(f"    righe dataset             : {n_rows:,}  (tutte usate, ingest a chunk)")
    print(f"    footprint per seme (F)    : ~{f_gb:.1f} GB")
    print(f"    semi per ondata (N)       : {n_auto}   (piu' RAM -> piu' semi in parallelo)")
    if p_cores and n_auto > p_cores:
        print(f"    -> {n_auto} semi > {p_cores} core reali: 1 thread logico/processo + BLAS=1 "
              f"(l'iper-threading riempie la RAM, niente contesa BLAS sui core)")

    # Quanti semi TOTALI: predefinito = n_auto (INVIO). Un numero maggiore si esegue in piu' TURNI.
    n_semi = _N_SEEDS_CLI if _N_SEEDS_CLI else int(ask(
        f"Quanti semi addestrare? (INVIO = {n_auto}, il predefinito dalla RAM)", str(n_auto)))
    n_semi = max(1, n_semi)
    turni = piano_turni(n_semi, n_auto)
    if n_semi > n_auto:
        print(f"\n  ATTENZIONE: {n_semi} semi > {n_auto} che la RAM regge in parallelo. "
              f"L'addestramento sara' LUNGO: {len(turni)} turni sequenziali "
              f"(~{len(turni)} volte il tempo di un'ondata). La macchina resta impegnata a lungo.")
        if not ask_yesno("Procedo comunque?", default=True):
            return
    if n_semi < 4:
        print(f"\n  ATTENZIONE: {n_semi} semi sono pochi. Con pochi semi la probabilita' di trovare "
              f"il seme ottimale e' bassa: le prestazioni di rilevazione possono calare.")
        if not ask_yesno("Procedo comunque?", default=True):
            return

    sib = _sibling_sets()
    dimensionamento = {"core_fisici": p_cores, "thread_logici": l_cores,
                       "oversubscription_blas1": bool(p_cores and n_auto > p_cores),
                       "ram_disponibile_gb": round(avail, 1) if avail else None,
                       "riserva_fissa_gb": RAM_RESERVE_GB, "footprint_per_seme_gb": round(f_gb, 1),
                       "righe_dataset": n_rows, "righe_usate": n_eff, "semi_totali": n_semi,
                       "semi_per_ondata": n_auto, "turni": len(turni),
                       "contenimento_memorymax_gb": round(cap, 1) if cap else None}
    print(f"\n  {n_semi} semi in {len(turni)} turno/i"
          + (f", contenuti a MemoryMax={cap:.1f}G" if cap else "")
          + f" (log per seme in modelli/sperimentali/<nome>_seed<S>/train.log):")

    outs = []
    for t, ondata in enumerate(turni, 1):
        fette, core_per_proc = pinning_ondata(len(ondata), p_cores, sib)
        dimensionamento["core_per_processo"] = core_per_proc
        righe, outs_t = [], []
        for j, off in enumerate(ondata):
            s = 42 + off
            out = MODELLI / "sperimentali" / f"{nome}_seed{s}"
            # Conserva il modello precedente in _prev prima del retrain.
            if out.exists():
                prev = out.with_name(out.name + "_prev")
                if prev.exists():
                    shutil.rmtree(prev)
                out.rename(prev)
                print(f"  backup: {out.name} -> {prev.name}")
            out.mkdir(parents=True, exist_ok=True)
            outs_t.append(out)
            righe.append(_riga_seme(_py("train"), dataset, out, s,
                                    fette[j] if fette else None, core_per_proc or None))
        if len(turni) > 1:
            print(f"\n  -- turno {t}/{len(turni)} (semi {[42 + o for o in ondata]}) --")
        for r in righe:
            print("    $ " + r)
        esito = lancia_ondata(righe, cap, outs_t)
        # Riporta per nome i semi che terminano con errore.
        falliti = []
        for out, off in zip(outs_t, ondata):
            ec_file = Path(out) / ".exitcode"
            if ec_file.exists():
                rc = ec_file.read_text().strip()
                if rc != "0":
                    falliti.append((f"seed{42 + off}", f"exit={rc}"))
            else:
                falliti.append((f"seed{42 + off}", "exit=? (processo interrotto prima del completamento)"))
        if falliti:
            print(f"\n  ERRORE (H5): {len(falliti)} semi falliti nel turno {t}:")
            for nome_seme, motivo in falliti:
                print(f"    {nome_seme}: {motivo}")
        elif esito.returncode != 0:
            print(f"  ATTENZIONE: turno {t} uscito con codice {esito.returncode} (possibile kill del "
                  f"contenimento MemoryMax — controlla i train.log).")
        for out in outs_t:
            _annota_dimensionamento(out, dimensionamento)
            _mostra_esito_es(out)
        outs.extend(outs_t)
    best = _select_best_seed(outs)
    if best:
        _promuovi_e_pulisci(best, nome, outs)


def _fmt_fpr(fm, key):
    """FPR dal train_report: una stringa diagnostica passa com'e', un numero diventa percentuale."""
    v = fm.get(key)
    if isinstance(v, str):
        return v
    return f"{fm.get(key, 0) * 100:.3f}%"


def _mostra_esito_es(out_dir):
    rep = Path(out_dir) / "train_report.json"
    if not rep.exists():
        print(f"  [{Path(out_dir).name}] nessun train_report.json (retrain fallito?)")
        return
    try:
        r = json.loads(rep.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [{Path(out_dir).name}] train_report.json illeggibile ({e}): retrain troncato?")
        return
    dae, pa, fm = r.get("dae", {}), r.get("pa", {}), r.get("fpr_monitor", {})
    print(f"  [{Path(out_dir).name}] DAE: arresto = {dae.get('ramo','?')} (ep {dae.get('stop_epoch','?')}) | "
          f"PA: {pa.get('ramo','?')} (ep {pa.get('stop_epoch','?')}) | "
          f"FPR monitor A={_fmt_fpr(fm, 'A')} "
          f"B={_fmt_fpr(fm, 'B')}")


def _select_best_seed(out_dirs):
    """Seleziona il seme con FPR monitor minimo in A, usando B come spareggio."""
    cands = []
    for out in out_dirs:
        rep = Path(out) / "train_report.json"
        if not rep.exists():
            continue
        try:
            fm = json.loads(rep.read_text()).get("fpr_monitor", {})
        except (json.JSONDecodeError, OSError):       # un report corrotto esclude il seme, non la selezione
            print(f"  [{Path(out).name}] train_report.json illeggibile: seme escluso dalla selezione")
            continue
        if "A" in fm and "B" in fm:
            cands.append((Path(out).name, float(fm["A"]), float(fm["B"])))
    if not cands:
        print("\n  (nessun train_report.json valido: impossibile selezionare il seme)")
        return
    cands.sort(key=lambda c: (c[1], c[2]))            # minimo FPR Mod A, tie-break Mod B
    best = cands[0]
    print("\n  === Selezione del seme di produzione (N2, label-free) ===")
    print("  Criterio: minimo FPR realizzato sul MONITOR (holdout benigno), non il mediano di laboratorio.")
    for name, a, b in cands:
        mark = "  <== VINCITORE" if name == best[0] else ""
        print(f"    {name:<24} FPR monitor  Mod A {a*100:.3f}%   Mod B {b*100:.3f}%{mark}")
    if len(cands) > 1:
        secondo = cands[1]
        print(f"\n  Vince '{best[0]}': FPR realizzato piu' basso a Mod A ({best[1]*100:.3f}%), il minimo "
              f"tra i {len(cands)} semi (il secondo, '{secondo[0]}', e' a {secondo[1]*100:.3f}%). "
              f"Essendo label-free, il FPR sui benigni-monitor e' l'unico segnale disponibile.")
    else:
        print(f"\n  Vince '{best[0]}' (unico seme con report valido): FPR a Mod A {best[1]*100:.3f}%.")
    return best[0]


def _promuovi_e_pulisci(best, nome, outs):
    """Promuove il seme vincitore e propone di rimuovere gli sperimentali perdenti."""
    src = MODELLI / "sperimentali" / best
    if not src.exists():
        print(f"  ATTENZIONE: la cartella del vincitore {src} non esiste piu': salto la promozione.")
        return
    dest = MODELLI / "produzione" / nome
    if dest.exists():
        if ask_yesno(f"\n  produzione/{nome} esiste gia'. Sovrascriverlo col nuovo vincitore?", default=True):
            shutil.rmtree(dest)
        else:
            dest = MODELLI / "produzione" / best
            if dest.exists():
                print(f"  ERRORE: anche produzione/{best} esiste gia': promozione annullata per evitare "
                      f"l'annidamento (shutil.move inserirebbe il modello dentro la cartella esistente). "
                      f"Spostare o rinominare produzione/{best} prima di riprovare.")
                return
            print(f"  (conservo produzione/{nome}; il vincitore va in produzione/{best})")
    shutil.move(str(src), str(dest))
    print(f"\n  ==> '{best}' PROMOSSO in produzione/{dest.name}")
    print(f"      ./infer.py --batch traffico.csv --mode A --model {dest.name}")
    perdenti = [Path(o) for o in outs if Path(o).name != best and Path(o).exists()]
    if perdenti and ask_yesno(f"\n  Cancellare i {len(perdenti)} semi NON vincenti da sperimentali/?", default=False):
        for p in perdenti:
            shutil.rmtree(p, ignore_errors=True)
        print(f"  cancellati {len(perdenti)} semi non vincenti.")
    elif perdenti:
        print(f"  {len(perdenti)} semi non vincenti conservati in sperimentali/.")


def op_cattura():
    print("\n-- Cattura nProbe --")
    iface = ask("Interfaccia di rete (vuoto = auto)")
    env = dict(os.environ, NIDS_BASEDIR=str(DIR / "sniffing"))
    if iface:
        env["NIDS_IFACE"] = iface
    print("$ bash sniffing/cattura.sh   (Ctrl-C per fermare)")
    subprocess.run(["bash", str(DIR / "sniffing" / "cattura.sh")], env=env)


def _parse_k(s, default):
    """Converte input in migliaia di flussi; usa default se non valido."""
    s = s.strip().lower().rstrip("k")
    try:
        return max(1, int(s))
    except ValueError:
        return default


def op_benchmark():
    """Esegue il benchmark a micro-batch variabili e genera i grafici disponibili."""
    import socket
    print("\n-- Benchmark (throughput + latenza, simula il live) --")
    if not _check_venv("infer"):
        return
    arch = ask("Etichetta di questa macchina (per confrontare piu' PC)", socket.gethostname())
    k = _parse_k(ask("Quanti campioni, in migliaia? es. 50 / 100 / 200 / 500 (accetta '100' o '100k')",
                     "100"), 100)
    out = DIR / "benchmarks"
    # Conserva i benchmark esistenti per aggregare macchine e ripetizioni.
    print(f"\n  Misura in corso: {k * 1000:,} flussi in finestre VARIABILI (simula il live), x2 ripetizioni...")
    r = subprocess.run(["bash", str(DATI_BENCH / "bench_run.sh"), arch, str(k), str(out)])
    if r.returncode != 0:
        print("  la misura e' uscita con errore: controlla l'output sopra.")
        return
    if TRAIN.exists():
        print("\n  Genero i grafici (matplotlib/seaborn)...")
        subprocess.run([str(TRAIN), str(DIR / "plot_bench.py"),
                        "--bench-dir", str(out), "--output", str(out / "plots")])
        print(f"  grafici in {out / 'plots'}")
    else:
        print(f"\n  Riepilogo testuale pronto in {out}. Per i grafici serve il venv di plotting:")
        print(f"    bash install.sh train")
        print(f"    train_venv/bin/python plot_bench.py --bench-dir {out} --output {out / 'plots'}")


def _stampa_stato():
    """Mostra modelli e dataset disponibili."""
    mod = _scopri_modelli()
    prod = [n for n, (_, s) in mod.items() if s == "produzione"]
    nsper = sum(1 for _, (_, s) in mod.items() if s == "sperimentali")
    inf = _scopri_csv(DATI_INF)
    tr = _scopri_csv(DATI_TRAIN)
    print("\n" + "-" * 60)
    print("  Contenuto del pacchetto:")
    print(f"    Modelli produzione : {', '.join(prod) if prod else '(nessuno)'}"
          + (f"   + {nsper} sperimentali" if nsper else ""))
    print(f"    CSV inferenza      : {', '.join(f.name for f in inf) if inf else '(nessuno)'}")
    print(f"    CSV addestramento  : {', '.join(f.name for f in tr) if tr else '(nessuno)'}")
    print("-" * 60)


_N_SEEDS_CLI = None


def main(argv=None):
    global _N_SEEDS_CLI
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-seeds", type=int, default=None,
                   help="forza il numero TOTALE di semi del retrain (il parallelismo resta cappato "
                        "al valore sicuro: l'eccesso genera turni; avviso se sopra il cap).")
    _N_SEEDS_CLI = p.parse_args(argv).n_seeds

    actions = {"1": ("Inferenza (file o live)", op_inferenza),
               "2": ("Riaddestramento label-free", op_retrain),
               "3": ("Cattura traffico (nProbe)", op_cattura),
               "4": ("Benchmark (throughput + plot)", op_benchmark)}
    while True:
        _stampa_stato()
        print("\n=== NIDS a 2 stadi — menu ===")
        for k, (label, _) in actions.items():
            print(f"  {k}) {label}")
        print("  q) Esci")
        c = input("Scelta: ").strip().lower()
        if c == "q":
            break
        if c in actions:
            try:
                actions[c][1]()
            except KeyboardInterrupt:
                print("\n(interrotto)")


if __name__ == "__main__":
    main()
