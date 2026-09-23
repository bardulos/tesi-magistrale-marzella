#!/usr/bin/env python3
"""main.py — orchestratore interattivo del pacchetto NIDS (per l'amministratore di rete PMI).

Interfaccia a domande e risposte che compone ed esegue i comandi di infer.py / train.py: NESSUNA
logica propria, solo composizione (la sequenza resta riproducibile a mano senza il menu). Ogni
comando eseguito viene stampato prima, cosi' e' ripetibile da shell.

Pensato per un utente SENZA shell aperta: all'avvio MOSTRA cosa ha trovato nelle cartelle (modelli
disponibili in modelli/, dataset in dati/) e fa scegliere da una LISTA NUMERATA, mai chiedere un
percorso a mano. Il controllo dei venv e' MIRATO: verifica infer_venv solo prima di infer.py e
train_venv solo prima di train.py, cosi' chi fa solo inferenza non deve installare il venv di training.

Novita' rispetto al vecchio corso:
  - selezione della MODALITA' operativa A/B ("FPR minimo" / "alta recall");
  - retrain LABEL-FREE con parallelismo scelto dall'utente (N core -> N semi in parallelo, shell,
    mai multiprocessing) e guardia RAM; il dataset si sceglie da una lista numerata (un solo
    insieme in ingresso, gia' contaminato al naturale: nessuna domanda benigno/sospetto);
  - backup automatico "solo ultimo" del modello precedente in <nome>_prev prima di riaddestrarlo;
  - lettura dell'esito del criterio d'arresto automatico (ramo ES del DAE, punto di stop del PA)
    da train_report.json in linguaggio leggibile, non log grezzo.
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
BLAS_INFER = 4          # thread BLAS per l'inferenza monoprocesso. DEVE restare uguale a
                        # infer.BLAS_DEFAULT (lo verifica test_blas_infer_allineato_con_infer):
                        # li' sta la misura che lo giustifica.

# --- N9: dimensionamento automatico del retrain multi-seme -------------------------------------
# Il numero di semi si calcola a runtime da (thread logici come TETTO, RAM disponibile, footprint per
# seme), in una o piu' ondate. Dal 2026-07-20 il tetto e' salito dai core fisici ai thread logici
# (riempire la RAM, non i soli core): oltre i core reali si va a 1 thread logico/processo + BLAS=1
# (v. op_retrain). Il footprint NON e' una costante di comodo: e' il modello lineare ricavato
# dall'audit di memoria del 2026-07-16, misurato sul codice post-fix.
SIZE_REF = 1_755_000          # riferimento canonico CSE (log/guardie N4 di train.py); NON piu' un cap
# Margine di sicurezza FISSO, non proporzionale: si usa quasi tutta la RAM disponibile tenendo da
# parte 1 GiB per il sistema. A e' una fotografia al lancio, quindi l'ondata pretende una macchina
# senza carichi concorrenti significativi (l'avviso lo dice a schermo).
RAM_RESERVE_GB = 1.0
# Requisito MINIMO di macchina per il RIADDESTRAMENTO (non per l'inferenza, che gira in ~50 MB): sotto
# questa soglia il retrain viene RIFIUTATO invece di essere tentato. Motivo: il costo FISSO del runtime
# TensorFlow e' ~2 GiB e il footprint per seme parte da ~5 GiB, quindi su una macchina piccola il
# dimensionamento sceglie comunque 1 seme e gli mette addosso un tetto che lo uccide dopo ore di
# paginazione (incidente del 2026-07-20: OOM del cgroup dopo 80 minuti al 34% di un core).
# La soglia e' 14,5 e non 16,0 perche' MemTotal e' sempre SOTTO la taglia nominale (firmware e grafica
# integrata ne riservano una parte): misurati 62,72 GiB su 64 nominali e 7,44 su 8. Una macchina da
# 16 GB dichiara ~15 GiB e deve passare; una da 8 ne dichiara ~7,4 e deve essere fermata.
RAM_MIN_TOTAL_GB = 14.5
# L'ondata gira dentro un cgroup con questo margine SOPRA il bersaglio di dimensionamento: se
# sfonda, il kernel uccide il cgroup e non l'intero sistema (workstation e' senza swap).
CONTAINMENT_MARGIN_GB = 0.5
# Byte vivi per riga di input al picco della catena COMPLETA (7 passi; il collo di bottiglia e' il
# training PA). Con l'ingest a CHUNK (train.py, 2026-07-19) il picco NON e' piu' la lista di dict del
# DictReader (~2,7 KB/riga, la vecchia causa dell'OOM a ~15 GB/processo). Con i forward TF BATCHATI
# (Step B, 2026-07-20) non e' piu' nemmeno l'arena TF del forward intero su 2,41M righe (che portava
# il picco a ~13,6 GiB): restano gli array di feature + le copie dei pool (gradiente/monitor/select) +
# phi + il transiente del PA. Calibrato sul canary full-chain post-Step-B (5,14 GiB @ 2.966.479 righe
# -> 1137 B/riga sopra i 2,0 GiB fissi del runtime TF, arrotondato a 1200 per margine; aritmetica nella
# "Stratigrafia del drift" sotto). Nessun cap: si addestra su TUTTE le righe (train.py --max-rows 0).
BYTES_PER_ROW = 1_200         # Byte vivi per riga al picco, catena COMPLETA (7 passi), forward batchati.
# Stratigrafia del drift C1:
#   2026-07-16 Parte 5: 3600 B/riga calibrato su catena completa, RSS 7,84 GiB @ 1.755.000 righe.
#   2026-07-19 fix ingest-a-chunk: canary con --dae-only (usciva prima di phi+PA) -> RSS 3,93 GiB
#              @ 2.966.479 righe -> BYTES_PER_ROW erroneamente abbassato a 750 (sottostima 2,4x).
#   2026-07-19 mandato WP-9: canary FULL-CHAIN (7 passi, 3 epoche max) ->
#              RSS 13,95 GiB @ 2.966.479 righe -> (13,95-2,0)*2^30/2.966.479 = 4322 B/riga -> 4500.
#   2026-07-20 Step B (forward TF batchati, FWD_CHUNK): canary FULL-CHAIN post-fix ->
#              VmHWM 5,14 GiB @ 2.966.479 righe -> (5,14-2,0)*2^30/2.966.479 = 1137 B/riga -> 1200.
TF_OVERHEAD_GB = 2.0          # runtime TensorFlow + interprete (indipendente dalla taglia)
# Ancora empirica del canary full-chain (7 passi, 3 epoche max) su lan_contaminato.csv:
# N_ROWS e RSS_GB si aggiornano in coppia dopo ogni canary (non modificare separatamente).
CANARY_FULL_CHAIN_N_ROWS = 2_966_479   # righe del canary di riferimento
CANARY_FULL_CHAIN_RSS_GB = 5.14        # VmHWM picco full-chain post-Step-B (2026-07-20, max su semi 42/46)


def _py(which):
    p = INFER if which == "infer" else TRAIN
    return str(p) if p.exists() else sys.executable   # safety net (non raggiunto se _check_venv ok)


def _check_venv(which):
    """Controllo MIRATO del venv: infer_venv prima di infer.py, train_venv prima di train.py.
    Se manca, avvisa con il comando esatto e ritorna False (nessun fallback silenzioso, e non si
    forza l'installazione dell'altro venv)."""
    if (INFER if which == "infer" else TRAIN).exists():
        return True
    nome = "infer_venv (inferenza)" if which == "infer" else "train_venv (training)"
    arg = "infer" if which == "infer" else "train"
    print(f"\n  ATTENZIONE: manca il venv {nome}.")
    print(f"  Installalo con:   bash install.sh {arg}")
    return False


def _scopri_csv(directory):
    """CSV reali (non symlink) presenti in una cartella, ordinati per nome."""
    d = Path(directory)
    return sorted(f for f in d.iterdir()
                  if f.is_file() and not f.is_symlink() and f.suffix == ".csv") if d.exists() else []


def _scopri_modelli():
    """{nome: (path, 'produzione'|'sperimentali')} dei modelli validi (con dae.npz), produzione prima."""
    trovati = {}
    for sub in ("produzione", "sperimentali"):
        d = MODELLI / sub
        if d.exists():
            for m in sorted(d.iterdir()):
                if m.is_dir() and (m / "dae.npz").exists():
                    trovati[m.name] = (m, sub)
    return trovati


def _scegli_da_lista(voci, etichetta, prompt_vuoto):
    """Presenta una lista numerata e ritorna la voce scelta (o '' per annullare). `voci` = lista di
    (display, valore). Se vuota, ricade su un prompt manuale (solo perche' la cartella e' vuota)."""
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
    """Compone ed esegue un comando. `cpus` (lista taskset) e `threads` (thread BLAS/TF) servono al
    retrain multi-seme: TensorFlow ignora gli env di threading, quindi il pinning e' obbligatorio
    per non far litigare i processi sugli stessi core."""
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
    """Un campo di /proc/meminfo in GiB (None se illeggibile). Unico parse per MemTotal e MemAvailable."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(chiave + ":"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return None


def ram_totale_gb():
    """RAM FISICA della macchina (MemTotal), non quella libera: serve a decidere se questa macchina
    puo' fare un riaddestramento, a prescindere da chi la sta occupando in questo momento."""
    return _meminfo_gb("MemTotal")


def ram_sufficiente_per_retrain(totale_gb=None):
    """Il riaddestramento richiede una macchina da almeno 14,5 GB (RAM_MIN_TOTAL_GB: la soglia
    e' sotto i 16 GB nominali perche' una macchina da 16 GB ne dichiara ~15). Ritorna
    (ok, totale_gb). Se la RAM non e' rilevabile si LASCIA passare: meglio un tentativo che un blocco
    su una macchina che il pacchetto non sa misurare."""
    tot = ram_totale_gb() if totale_gb is None else totale_gb
    if tot is None:
        return True, None
    return tot >= RAM_MIN_TOTAL_GB, tot


def _ram_gb():
    """RAM DISPONIBILE ora (MemAvailable): dimensiona i semi per ondata. Diversa da ram_totale_gb()."""
    return _meminfo_gb("MemAvailable")


def _sibling_sets(sys_root="/sys/devices/system/cpu"):
    """CPU logiche raggruppate per core FISICO, dalla topologia esposta dal kernel.
    Pinnare un processo a un core fisico vuol dire dargli entrambi i suoi thread (iper-threading):
    due processi sui due thread dello stesso core si contenderebbero le stesse unita' di calcolo."""
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
    """P = numero di core FISICI (reali). NON e' piu' il tetto ai semi per ondata (ora sono i thread
    logici, v. logical_cores): P e' la SOGLIA sopra la quale i semi superano i core reali e si passa a
    1 thread logico/processo + BLAS=1 (op_retrain), e resta il riferimento per il pinning a core interi
    sotto quella soglia."""
    gruppi = _sibling_sets(sys_root)
    return len(gruppi) if gruppi else _physical_cores_lscpu()


def logical_cores(sys_root="/sys/devices/system/cpu"):
    """T = numero di THREAD logici (iper-threading incluso). E' il TETTO ai semi per ondata quando la
    RAM lo consente: si riempie la RAM sfruttando anche i thread logici, non solo i core fisici (oltre
    i core reali ogni processo va a 1 thread + BLAS=1, v. op_retrain). Su 8c/16t vale 16. Ripiego:
    os.cpu_count()."""
    gruppi = _sibling_sets(sys_root)
    if gruppi:
        return sum(len(g) for g in gruppi)
    return os.cpu_count()


def count_csv_rows(path):
    """Righe dati del CSV (header escluso), contate a blocchi: il file non entra in RAM."""
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            n += b.count(b"\n")
    return max(n - 1, 0)


def footprint_gb(n_rows):
    """F = RAM attesa per UN retrain: lineare nelle righe (array di feature + copie dei pool; ingest a
    chunk, quindi NESSUN cap: tutte le righe) piu' l'overhead fisso del runtime TF. Calibrato sul
    canary full-chain (~5,14 GiB a 2,97M righe, 2026-07-20)."""
    return n_rows * BYTES_PER_ROW / 2**30 + TF_OVERHEAD_GB


def n_seeds_auto(avail_gb, f_gb, max_par):
    """N = clamp(floor((A - 1 GiB) / F), 1, max_par): quanti semi entrano in memoria tenendo da parte
    una riserva FISSA per il sistema, mai piu' del parallelismo massimo consentito. Dal 2026-07-20
    max_par sono i THREAD logici della CPU (non i soli core fisici): si riempie la RAM sfruttando anche
    l'iper-threading (oltre i core reali ogni processo va a 1 thread + BLAS=1, v. op_retrain), cosi' e'
    la RAM a comandare quasi sempre. Se non basta la RAM nemmeno per un seme si prova comunque con 1,
    sotto contenimento: e' il chiamante ad avvisare."""
    if not avail_gb or not f_gb or not max_par:
        return 1
    return max(1, min(int((avail_gb - RAM_RESERVE_GB) / f_gb), max_par))


def containment_cap_gb(avail_gb):
    """Tetto di memoria dell'ondata (cgroup): 0,5 GiB sopra il bersaglio di dimensionamento, cosi'
    uno sforamento modesto viene contenuto invece di essere ucciso subito, e il caso peggiore resta
    il kill dell'ondata e non la cascata OOM di sistema."""
    if not avail_gb:
        return None
    return max(1.0, avail_gb - CONTAINMENT_MARGIN_GB)


def taskset_sets(n_proc, sib_sets):
    """Ripartisce i core fisici fra i processi, senza sovrapposizioni: ogni processo riceve
    interi core fisici (entrambi i thread). Ritorna (lista di stringhe per taskset -c, core/proc)."""
    if not sib_sets or n_proc < 1:
        return [], 0
    per = max(1, len(sib_sets) // n_proc)
    fette = []
    for i in range(n_proc):
        gruppo = sib_sets[i * per:(i + 1) * per] or [sib_sets[i % len(sib_sets)]]
        fette.append(",".join(str(c) for g in gruppo for c in sorted(g)))
    return fette, per


def taskset_logici(n_proc, sib_sets):
    """Oversubscription controllata (piu' semi che core FISICI): ogni processo e' pinnato a UNA sola
    CPU logica (un thread hardware), non a un intero core. Con BLAS=1 per processo (core_per_proc=1,
    impostato a valle) l'iper-threading riempie la RAM invece di far contendere due thread BLAS sullo
    stesso core fisico.

    L'ORDINE conta: si prende il PRIMO fratello di OGNI core prima di tornare sui secondi
    (0,1,...,7 poi 8,...,15), cosi' i primi n_proc <= core_fisici processi cadono su core DISTINTI e si
    raddoppia su un core solo quando i core sono esauriti. Appiattire i sibling-set per core darebbe
    invece 0,8,1,9,... e appaierebbe i processi CONSECUTIVI sullo stesso core: con 11 processi si
    impegnavano 6 core su 8 (2 fermi), e i processi appaiati misuravano 67% di occupazione contro
    l'85,7% di quello rimasto solo sul proprio core (osservazione del 2026-07-20).
    Ritorna (lista di stringhe per taskset -c, 1 = thread BLAS per processo)."""
    if not sib_sets or n_proc < 1:
        return [], 0
    profondita = max(len(g) for g in sib_sets)
    logici = [g[k] for k in range(profondita) for g in sib_sets if k < len(g)]
    return [str(logici[i % len(logici)]) for i in range(n_proc)], 1


def pinning_ondata(n_proc, p_cores, sib_sets):
    """Pinning di un'ondata di n_proc semi. Oltre i core FISICI (p_cores) si oversottoscrive con un
    thread logico per processo e BLAS=1 (taskset_logici); entro i core reali si danno interi core
    fisici con BLAS libero (taskset_sets). Ritorna (fette per taskset -c, core_per_proc = thread BLAS)."""
    if p_cores and n_proc > p_cores:
        return taskset_logici(n_proc, sib_sets)
    return taskset_sets(n_proc, sib_sets)


def piano_turni(n_semi, n_auto):
    """Distribuisce n_semi in ONDATE sequenziali da al piu' n_auto semi ciascuna (n_auto = quanti
    la RAM regge in parallelo). Ritorna la lista delle ondate, ognuna lista di offset-seme
    (0..n_semi-1). Con n_semi <= n_auto: una sola ondata. E' la logica dei 'turni' del retrain."""
    par = max(1, n_auto)
    return [list(range(i, min(i + par, n_semi))) for i in range(0, n_semi, par)]


def _riga_seme(py, dataset, out, seed, cpus, threads):
    """Una riga di shell per un seme: env di threading + pinning + train.py + log dedicato.
    Resta leggibile e ripetibile a mano, come ogni comando composto dal menu."""
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
    """Lancia l'INTERA ondata dentro UN SOLO cgroup con tetto di memoria.

    Il cap e' a livello di ondata, non per-processo, perche' la quantita' che minaccia il sistema e'
    l'AGGREGATO: i picchi per seme variano (EntropyStop ferma a epoche diverse), quindi un tetto
    per-processo tarato sulla media ucciderebbe un seme legittimo mentre l'ondata nel suo insieme
    sta larga. Cosi' invece si viene uccisi solo quando l'aggregato sfonda davvero — che e'
    esattamente il caso da contenere.

    Se out_dirs e' fornito, ogni seme scrive il proprio exit code in <out_dir>/.exitcode prima di
    uscire: H5 — un seme fallito deve essere riportato per nome, mai silenzioso.
    """
    parts = []
    pid_vars = []
    n = len(righe)
    for i, r in enumerate(righe):
        parts.append(r)         # termina con &
        pid_vars.append(f"_P{i}")
        parts.append(f"_P{i}=$!")
        if i < n - 1:
            parts.append("sleep 2")   # A5 (M7): sfalsa i picchi RAM per seme dentro lo stesso cgroup
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
    """Scrive nel train_report.json la scelta di dimensionamento (P/A/F/N). Merge: le chiavi
    prodotte da train.py restano intatte."""
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
    """Operazione di inferenza dal menu: il modello si sceglie da un ELENCO dei disponibili,
    mai digitandone il nome a memoria."""
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

    # N9: la macchina dimensiona il PARALLELISMO sicuro (n_auto = semi per ondata dalla RAM). Il TETTO
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
    n_eff = n_rows                    # nessun cap: si addestra su TUTTE le righe (ingest a chunk)
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
            # backup "solo ultimo": il modello precedente va in <nome>_seed<s>_prev (mai sovrascrivere
            # senza backup); il _prev precedente viene rimpiazzato. rename = istantaneo (stesso fs).
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
        # H5 — raccolta exit code per-seme: un seme fallito è riportato per nome, mai silenzioso
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
            for nome_seme, motivo in falliti:      # variabile propria: `nome` (modello) non va ricoperto
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
    """N2 — sceglie il seme di produzione: il MIGLIORE per FPR realizzato sul pool MONITOR (holdout
    label-free), non il mediano. Criterio: minimo FPR al punto operativo Mod A, tie-break Mod B.
    La selezione e' una lettura dei train_report.json prodotti dai retrain paralleli."""
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
    """Sposta SEMPRE il seme vincitore in modelli/produzione/ (col nome base del modello) e propone
    di cancellare gli sperimentali non vincenti (solo su conferma)."""
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
    # pulizia degli sperimentali non vincenti (il vincitore ora e' in produzione, quindi non e' fra questi)
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
    """'100' o '100k'/'100K' -> 100 (migliaia di flussi). Ripiego al default su input non valido."""
    s = s.strip().lower().rstrip("k")
    try:
        return max(1, int(s))
    except ValueError:
        return default


def op_benchmark():
    """Benchmark del throughput di inferenza che SIMULA il live: micro-batch a dimensione variabile
    (distribuzione reale delle finestre nProbe). Misura testuale (bench_run.sh, nel infer_venv) +
    grafici (plot_bench.py, nel train_venv). I JSON di piu' macchine nella stessa dir -> confronto."""
    import socket
    print("\n-- Benchmark (throughput + latenza, simula il live) --")
    if not _check_venv("infer"):
        return
    arch = ask("Etichetta di questa macchina (per confrontare piu' PC)", socket.gethostname())
    k = _parse_k(ask("Quanti campioni, in migliaia? es. 50 / 100 / 200 / 500 (accetta '100' o '100k')",
                     "100"), 100)
    out = DIR / "benchmarks"
    # NB: NON si cancella la cartella: puo' contenere i JSON di ALTRE macchine (confronto multi-arch).
    # Le rep successive della STESSA macchina si accumulano (bench_run.sh aggiunge, non sovrascrive).
    print(f"\n  Misura in corso: {k * 1000:,} flussi in finestre VARIABILI (simula il live), x2 ripetizioni...")
    r = subprocess.run(["bash", str(DATI_BENCH / "bench_run.sh"), arch, str(k), str(out)])
    if r.returncode != 0:
        print("  la misura e' uscita con errore: controlla l'output sopra.")
        return
    # grafici: matplotlib/seaborn stanno nel train_venv
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
    """Mostra all'utente cosa c'e' nelle cartelle: modelli e dataset disponibili (niente ls a mano)."""
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


_N_SEEDS_CLI = None      # override esplicito di N9 (--n-seeds), None = la macchina decide


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
