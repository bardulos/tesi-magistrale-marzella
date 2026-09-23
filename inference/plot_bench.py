#!/usr/bin/env python3
"""plot_bench.py — visualizza i benchmark JSON prodotti da infer.py --bench (3o script del
pacchetto, offline). Separazione netta: infer.py PRODUCE i dati (NumPy-puro a runtime),
plot_bench.py li VISUALIZZA (matplotlib/seaborn — questo NON e' il runtime di inferenza).

Nota WP-6: il benchmark DETTAGLIATO (breakdown per-step delle latenze, segmentazione dentro il
preprocessing) e' oggetto del WP-10 (dopo il modello end-to-end). Questo file e' la superficie di
visualizzazione, pronta per quei JSON: gli step sono gia' quelli delle funzioni @timed di infer.py.

Uso:
    python plot_bench.py                          # tutti i bench in benchmarks/
    python plot_bench.py --arch Ryzen             # solo una architettura
    python plot_bench.py --output plots/          # dir PNG (default plots/)
"""
import argparse
import glob
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                   # backend non interattivo (file PNG)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", rc={"axes.axisbelow": True, "axes.grid.axis": "y", "figure.dpi": 120})

# Ordine canonico degli step = funzioni @timed di infer.py + 'total' di @timed_total (assenti ignorati).
STEP_ORDER = ["csv_to_features", "forward_dae", "compute_scores_fused",
              "apply_fusion", "emit_alerts", "total"]
STEP_LABELS = {
    "csv_to_features": "preprocessing",      # parsing CSV + encoding a 91 + scaling/clip
    "forward_dae": "dae",                    # forward del DAE (ricostruzione)
    "compute_scores_fused": "classificatore",  # s_dae (MSE) + forward fuso del PA
    "apply_fusion": "fusione",               # decisione a tre rami (AND u logit u DAE-alto)
    "emit_alerts": "alert",                  # formattazione alert + spiegazione per-feature
    "total": "totale",
}


# --- stile: allineato a lib/plotting/base.py, ma COPIATO qui ------------------------------------
# inference/ e' un pacchetto autocontenuto: non importa da lib/ (girerebbe anche estratto da solo).
# I valori seguono le convenzioni del resto del progetto; se cambiano la', vanno riallineati qui.
DPI = 600                       # come DPI_FINAL di lib/plotting/base.py: testo nitido nel PDF
BOX_FACE, BOX_EDGE = "#cfe0f3", "#33567a"      # stile box condiviso col progetto (azzurro tenue)

# Palette CATEGORICA delle ARCHITETTURE (le macchine confrontate). Volutamente NON e' METHOD_COLORS
# di lib/plotting: quella codifica i METODI (DAE/PA/AND/fused), un dominio semantico diverso, e
# riusarla qui darebbe a colori gia' "prenotati" un secondo significato.
# Perche' la "colorblind" di seaborn (derivata dalla Okabe-Ito) e non la famiglia del progetto:
# quest'ultima fallisce la verifica oltre le due serie (verde #2ca02c e arancio #ff7f0e a DeltaE 0,7 in protanopia,
# cioe' sono lo STESSO colore per chi non distingue rosso e verde). Questa passa a 4-5 serie; le
# etichette numeriche sulle barre restano come codifica secondaria, mai il colore da solo.
# Palette "colorblind" di seaborn: fra quelle predefinite e' l'unica che supera la verifica a 4 serie
# (deep, muted e Set2 falliscono: due aranci o due verdi indistinguibili anche a vista normale).
ARCH_COLORS = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC"]
UNA_SERIE = "#0173B2"           # grafici a una sola serie: il colore non codifica nulla, uno basta


def _rifinisci_barre(bars, spessore=0.8):
    """Bordo sottile in una tinta piu' scura del riempimento: da' definizione alla barra senza
    appiattirla in un blocco di colore pieno.

    NB: niente cime arrotondate. Il trucco usuale (bordo spesso + giunzioni tonde) qui non funziona
    su due fronti: a 600 dpi il raggio risultante e' di pochi pixel su immagini da migliaia, quindi
    invisibile; e soprattutto il bordo e' centrato sul contorno, percio' estende la barra di meta'
    spessore SOPRA il valore vero (~0,7% con 3,5 pt). Su un grafico di benchmark e' inchiostro che
    sovrastima il dato: si rinuncia all'effetto."""
    for b in bars:
        r, g, bl, _ = b.get_facecolor()
        b.set_linewidth(spessore)
        b.set_edgecolor((r * 0.72, g * 0.72, bl * 0.72, 1.0))


def _colore_arch(i):
    """Colore dell'i-esima architettura nell'ordine globale di throughput, condiviso dai tre
    confronti: a parco macchine invariato, la stessa macchina ha sempre lo stesso colore."""
    return ARCH_COLORS[i % len(ARCH_COLORS)]


def _asse_gemello_gbps(ax, coppie):
    """Secondo asse a destra con la STESSA grandezza in Gbit/s.

    Lecito perche' non e' una seconda misura: e' la stessa in un'altra unita' (i byte per flusso di un
    dataset sono una costante), quindi l'asse di destra e' una rietichettatura di quello di sinistra,
    non una seconda scala arbitraria. Vale pero' solo se il rapporto Gbit/flusso e' lo stesso per
    TUTTE le barre: con dataset di taglia media diversa un unico fattore sarebbe giusto per una sola
    e sbagliato per le altre (era il difetto della versione precedente, che lo prendeva dalla prima
    configurazione e lo applicava a tutte). In quel caso si rinuncia all'asse invece di mentire.
    `coppie` = [(flussi_al_secondo, gbit_al_secondo), ...]; ritorna True se l'asse e' stato disegnato."""
    ratti = [g / t for t, g in coppie if t > 0 and g > 0]
    if not ratti:
        return False
    if max(ratti) / min(ratti) > 1.01:                # oltre l'1%: byte/flusso diversi fra i dataset
        print("  nota: dataset con byte/flusso diversi -> niente asse Gbit/s (sarebbe valido per uno solo)")
        return False
    k = sum(ratti) / len(ratti)
    lo, hi = ax.get_ylim()
    ax2 = ax.twinx()
    ax2.set_ylim(lo * k, hi * k)
    ax2.set_ylabel("throughput di rete (Gbit/s)")
    ax2.grid(False)
    return True


def _step_label(s):
    return STEP_LABELS.get(s, s)


def load_runs(bench_dir, arch_filter=None):
    runs = []
    for f in sorted(glob.glob(str(Path(bench_dir) / "**" / "*.json"), recursive=True)):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARN: skip {f}: {e}")
            continue
        d["_arch"] = d.get("arch", Path(f).parent.name)
        d["_file"] = f
        if arch_filter and d["_arch"] != arch_filter:
            continue
        runs.append(d)
    return runs


def plot_throughput_bars(arch, runs, out_dir):
    """Throughput di deploy per config (dataset, mode): UN grafico, flussi/s a sinistra e Gbit/s a
    destra. Non sono due misure ma la stessa in due unita', quindi un secondo pannello ridisegnerebbe
    le stesse barre; l'asse destro viene pero' disegnato solo se il fattore di conversione e' lo
    stesso per tutte le barre (v. _asse_gemello_gbps). Ripetizioni aggregate per media."""
    agg = {}                                             # (dataset, mode) -> {"tput":[...], "gbps":[...]}
    for r in runs:
        a = agg.setdefault((r.get("dataset", "?"), r.get("mode", "?")), {"tput": [], "gbps": []})
        a["tput"].append(r.get("throughput_flows_per_sec", 0))
        a["gbps"].append(r.get("throughput_gbps", 0))
    configs = sorted(agg)
    n = len(configs)
    tput = [statistics.mean(agg[c]["tput"]) for c in configs]
    gbps = [statistics.mean(agg[c]["gbps"]) for c in configs]
    fig, ax = plt.subplots(figsize=(max(6.0, 2.4 * n + 3.0), 5.0))
    bars = ax.bar(range(n), tput, width=min(0.5, 0.16 * n + 0.2), color=UNA_SERIE)
    _rifinisci_barre(bars)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{ds} · Mod {m}" for ds, m in configs], fontsize=9)
    ax.set_xlim(-0.75, n - 0.25)
    ax.set_ylabel("throughput (flussi/s)")
    ax.set_ylim(0, (max(tput) * 1.14) if tput else 1.0)
    ax.grid(axis="y", alpha=0.3)
    for i, b in enumerate(bars):
        if tput[i] > 0:
            ax.annotate(f"{tput[i]:,.0f} fl/s  ·  {gbps[i]:.2f} Gbit/s",
                        (b.get_x() + b.get_width() / 2, tput[i]), textcoords="offset points",
                        xytext=(0, 4), ha="center", va="bottom", fontsize=9, fontweight="bold")
    _asse_gemello_gbps(ax, list(zip(tput, gbps)))
    ax.set_title(f"Throughput deploy (simula il live) — {arch}")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"{arch}_throughput.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {arch}_throughput.png")


def plot_memory_bars(arch, runs, out_dir):
    """Picco RSS (memoria del processo) per config (dataset, mode): UNA barra = media sulle ripetizioni,
    barretta d'errore = semi-scarto fra rep. L'inferenza e' monoprocesso, quindi questo E' il picco di
    memoria del deploy. Salta se i JSON non hanno peak_rss_mb (bench vecchi)."""
    agg = {}
    for r in runs:
        agg.setdefault((r.get("dataset", "?"), r.get("mode", "?")), []).append(r.get("peak_rss_mb", 0))
    configs = sorted(agg)
    rss = [statistics.mean(agg[c]) for c in configs]
    if not any(v > 0 for v in rss):                      # nessun peak_rss nei JSON
        return
    n = len(configs)
    labels = [f"{ds}\nMod {m}" for ds, m in configs]
    err = [(max(agg[c]) - min(agg[c])) / 2 for c in configs]
    fig, ax = plt.subplots(figsize=(max(5.0, 2.6 * n + 1.5), 5.0))
    bars = ax.bar(range(n), rss, width=min(0.5, 0.16 * n + 0.2), color=UNA_SERIE)
    _rifinisci_barre(bars)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlim(-0.75, n - 0.25)
    ax.set_ylabel("picco RSS (MB)"); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, (max(rss) * 1.14) if rss else 1.0)
    for i, b in enumerate(bars):
        if rss[i] > 0:
            ax.annotate(f"{rss[i]:,.0f} MB", (b.get_x() + b.get_width() / 2, rss[i]),
                        textcoords="offset points", xytext=(0, 3), ha="center", va="bottom",
                        fontsize=9, fontweight="bold")
    ax.set_title(f"Picco memoria (RSS) — {arch}")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"{arch}_memory.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {arch}_memory.png")


def _whisker_top(samp):
    """Baffo superiore del boxplot (matplotlib): il valore piu' grande <= Q3 + 1.5*IQR. Robusto agli
    outlier: le finestre piccole (pochi flussi) danno us/flusso altissimi che, se usati come tetto,
    schiaccerebbero i box. Con <2 campioni ripiega al massimo."""
    if len(samp) < 2:
        return max(samp) if samp else 1.0
    q1, _, q3 = statistics.quantiles(samp, n=4)
    hi = q3 + 1.5 * (q3 - q1)
    return max((x for x in samp if x <= hi), default=q3)


def _merge_latency(runs):
    """Unisce i campioni us/flusso delle ripetizioni della stessa config, per step -> boxplot con piu'
    dati (una finestra = un campione; con 2 rep si raddoppiano)."""
    merged = {}
    for r in runs:
        for s, v in r.get("latency_us", {}).items():
            merged.setdefault(s, []).extend(v.get("samples_us", []))
    return merged


def plot_latency_boxplot(arch, mode, dataset, latency, out_dir):
    """Boxplot latenza us/flusso per step (campioni delle rep uniti): mostra dove va il tempo
    (tipicamente il preprocessing domina, la fusione e' una frazione trascurabile). La varieta' delle
    finestre (bench a micro-batch variabile) da' box con spread REALE."""
    steps = [s for s in STEP_ORDER if s in latency]
    if [s for s in steps if s != "total"] == []:
        return
    data = [latency[s] for s in steps]
    if not any(data):
        return
    fig, ax = plt.subplots(figsize=(9.5, 5))
    bp = ax.boxplot(data, showfliers=False, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.5})
    for patch in bp["boxes"]:                      # stile box condiviso col progetto
        patch.set_facecolor(BOX_FACE); patch.set_edgecolor(BOX_EDGE); patch.set_alpha(1.0)
    ax.set_xticks(range(1, len(steps) + 1))
    ax.set_xticklabels([_step_label(s) for s in steps], rotation=20, ha="right")
    ax.set_ylabel("Latenza (us / flusso)")
    ax.set_title(f"Latenza per step — {arch} · Mod {mode} · {dataset}")
    # Tetto y e annotazioni ancorati al BAFFO superiore (non al max, che e' un outlier delle finestre
    # piccole): cosi' i box restano leggibili e la mediana e' attaccata alla cima del box.
    tops = [_whisker_top(s) for s in data if s]
    gmax = max(tops, default=1.0)
    ax.set_ylim(top=gmax * 1.15)
    for i, samp in enumerate(data, start=1):
        if not samp:
            continue
        med = statistics.median(samp)
        ax.annotate(f"{med:.2f} us", (i, _whisker_top(samp)), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=8, fontweight="bold",
                    annotation_clip=False)
    fig.tight_layout()
    name = f"{arch}_{mode}_latency.png"
    fig.savefig(Path(out_dir) / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {name}")


def _ordine_architetture(runs):
    """Architetture dalla MENO alla PIU' performante = throughput medio crescente. UN solo ordine,
    riusato da TUTTI i confronti (throughput, memoria, latenza): cosi' la stessa macchina resta nella
    stessa posizione e con lo stesso colore in ogni grafico (il colore segue la MACCHINA, non il
    singolo grafico ne' il suo rango in quella metrica)."""
    tput = {}
    for r in runs:
        tput.setdefault(r["_arch"], []).append(r.get("throughput_flows_per_sec", 0))
    return sorted(tput, key=lambda a: statistics.mean(tput[a]))


def _griglia_barre_arch(configs, archs, valore):
    """Disposizione comune ai confronti fra macchine: UNA barra per (config, architettura), il nome
    della macchina sotto la barra. `valore(c, a)` ritorna il valore o None se la combinazione manca.
    Ritorna (posizioni, etichette, colori, valori, centri_dei_gruppi)."""
    pos, etich, colori, vals, centri = [], [], [], [], []
    x = 0.0
    for c in configs:
        inizio = x
        for ai, a in enumerate(archs):
            v = valore(c, a)
            if v is None:
                continue
            pos.append(x)
            etich.append(a)
            colori.append(_colore_arch(ai))
            vals.append(v)
            x += 1.0
        centri.append((inizio + x - 1.0) / 2)
        x += 0.9                                         # stacco fra un gruppo-config e il successivo
    return pos, etich, colori, vals, centri


def _rifinisci_assi_arch(ax, pos, etich, configs, centri, ylabel, vals):
    """Assi dei confronti fra macchine: nomi delle macchine sulle ascisse (mai una legenda: le
    etichette dirette si leggono senza rimbalzare avanti e indietro, e non lasciano l'identita' al
    solo colore). Il nome della configurazione compare sotto il gruppo solo se ce n'e' piu' d'una."""
    # etichette in DIAGONALE: i nomi delle macchine sono lunghi (es. "Raspberry Pi 3B+ 1GiB") e
    # dritti si sovrapporrebbero; ruotati a 25 gradi con ancoraggio a destra restano separati.
    ax.set_xticks(pos); ax.set_xticklabels(etich, fontsize=10, rotation=25, ha="right")
    ax.set_xlim(pos[0] - 0.85, pos[-1] + 0.85)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(vals) * 1.16)
    ax.grid(axis="y", alpha=0.3)
    if len(configs) > 1:
        for centro, (ds, m) in zip(centri, configs):
            ax.annotate(f"{ds} · Mod {m}", (centro, -0.20), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=9, fontweight="bold")


def plot_multiarch_throughput(runs, out_dir):
    """Confronto del throughput FRA architetture: una barra per macchina, nome sotto la barra.
    Flussi/s a sinistra e Gbit/s a destra: stessa grandezza in due unita', non due misure.
    Macchine ordinate dalla meno alla piu' performante (throughput crescente)."""
    archs = _ordine_architetture(runs)
    if len(archs) < 2:
        return                                           # con una sola macchina bastano i per-arch
    if len(archs) > len(ARCH_COLORS):
        print(f"  AVVISO: {len(archs)} architetture ma {len(ARCH_COLORS)} colori: si ripetono "
              f"(l'identita' resta nelle etichette sotto le barre).")
    agg = {}
    for r in runs:
        key = (r.get("dataset", "?"), r.get("mode", "?"))
        a = agg.setdefault(key, {}).setdefault(r["_arch"], {"tput": [], "gbps": []})
        a["tput"].append(r.get("throughput_flows_per_sec", 0))
        a["gbps"].append(r.get("throughput_gbps", 0))
    configs = sorted(agg)
    media = lambda c, a, k: statistics.mean(agg[c][a][k]) if agg[c].get(a) else None
    pos, etich, colori, tput, centri = _griglia_barre_arch(configs, archs,
                                                           lambda c, a: media(c, a, "tput"))
    if not pos:
        return
    # stesso doppio ciclo e stesso filtro None di _griglia_barre_arch: gbps[i] corrisponde a tput[i]
    gbps = []
    for c in configs:
        for a in archs:
            v = media(c, a, "gbps")
            if v is not None:
                gbps.append(v)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(pos) + 3.5), 5.2))
    bars = ax.bar(pos, tput, width=0.62, color=colori)
    _rifinisci_barre(bars)
    _rifinisci_assi_arch(ax, pos, etich, configs, centri, "throughput (flussi/s)", tput)
    for b, v, g in zip(bars, tput, gbps):
        ax.annotate(f"{v:,.0f} fl/s  ·  {g:.2f} Gbit/s", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    _asse_gemello_gbps(ax, list(zip(tput, gbps)))
    coda = "" if len(configs) > 1 else f" — {configs[0][0]} · Mod {configs[0][1]}"
    ax.set_title(f"Confronto throughput fra architetture{coda}")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "confronto_multiarch_throughput.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [plot] confronto_multiarch_throughput.png")


def plot_multiarch_memory(runs, out_dir):
    """Confronto del picco di memoria FRA architetture, stessa impaginazione e stesso ordine
    (meno -> piu' performante) del throughput. Salta se i JSON non portano peak_rss_mb (bench vecchi)."""
    archs = _ordine_architetture(runs)
    if len(archs) < 2:
        return
    agg = {}
    for r in runs:
        key = (r.get("dataset", "?"), r.get("mode", "?"))
        agg.setdefault(key, {}).setdefault(r["_arch"], []).append(r.get("peak_rss_mb", 0))
    configs = sorted(agg)
    pos, etich, colori, rss, centri = _griglia_barre_arch(
        configs, archs, lambda c, a: statistics.mean(agg[c][a]) if agg[c].get(a) else None)
    if not pos or not any(v > 0 for v in rss):
        return
    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(pos) + 3.5), 5.2))
    bars = ax.bar(pos, rss, width=0.62, color=colori)
    _rifinisci_barre(bars)
    _rifinisci_assi_arch(ax, pos, etich, configs, centri, "picco RSS (MB)", rss)
    for b, v in zip(bars, rss):
        if v > 0:
            ax.annotate(f"{v:,.0f} MB", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 4), ha="center", va="bottom",
                        fontsize=9, fontweight="bold")
    coda = "" if len(configs) > 1 else f" — {configs[0][0]} · Mod {configs[0][1]}"
    ax.set_title(f"Confronto memoria (RSS) fra architetture{coda}")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "confronto_multiarch_memory.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [plot] confronto_multiarch_memory.png")


def plot_multiarch_latency(runs, out_dir):
    """Confronto della latenza TOTALE per flusso FRA architetture: un boxplot per macchina, dalla meno
    alla piu' performante (stesso ordine e stessi colori degli altri confronti). 'total' e' il tempo
    per flusso dell'intera catena; ogni finestra del bench e' un campione, quindi il box mostra la
    dispersione reale. I campioni sono uniti su tutte le ripetizioni e configurazioni della macchina.

    Scala LOG automatica quando le mediane spaziano oltre 5x (il caso tipico con un Raspberry Pi
    accanto a un desktop: ~20x): su scala lineare i box veloci sarebbero schiacciati contro l'asse e
    non se ne leggerebbe piu' la dispersione. Le mediane sono comunque annotate in us espliciti."""
    archs = _ordine_architetture(runs)
    dati = {a: [] for a in archs}
    for r in runs:
        dati[r["_arch"]].extend(r.get("latency_us", {}).get("total", {}).get("samples_us", []))
    archs = [a for a in archs if dati[a]]                 # solo macchine con campioni 'total'
    if len(archs) < 2:
        return
    data = [dati[a] for a in archs]
    medie = [statistics.median(s) for s in data]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.9 * len(archs) + 2.5), 5.2))
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6,
                    medianprops={"color": "black", "linewidth": 1.5})
    for ai, patch in enumerate(bp["boxes"]):
        c = matplotlib.colors.to_rgb(_colore_arch(ai))
        patch.set_facecolor(c); patch.set_edgecolor(tuple(x * 0.72 for x in c))
        patch.set_linewidth(1.0); patch.set_alpha(1.0)
    log = min(medie) > 0 and max(medie) / min(medie) > 5
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(1, len(archs) + 1))
    # nomi macchina in diagonale (lunghi, si sovrapporrebbero dritti)
    ax.set_xticklabels(archs, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Latenza totale (us / flusso)" + (" — scala log" if log else ""))
    ax.grid(axis="y", alpha=0.3, which="both" if log else "major")
    for i, (samp, med) in enumerate(zip(data, medie), start=1):
        ax.annotate(f"{med:.1f} us", (i, _whisker_top(samp)), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=8, fontweight="bold", annotation_clip=False)
    if not log:
        ax.set_ylim(0, max(_whisker_top(s) for s in data) * 1.18)
    ax.set_title("Confronto latenza totale per flusso fra architetture")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "confronto_multiarch_latenza.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [plot] confronto_multiarch_latenza.png")


def main():
    p = argparse.ArgumentParser(description="Visualizza i benchmark JSON di infer.py --bench (WP-10).")
    p.add_argument("--bench-dir", default=str(Path(__file__).resolve().parent / "benchmarks"))
    p.add_argument("--arch", default=None, help="filtra una sola architettura")
    p.add_argument("--output", default=str(Path(__file__).resolve().parent / "plots"))
    args = p.parse_args()
    runs = load_runs(args.bench_dir, args.arch)
    if not runs:
        raise SystemExit(f"nessun benchmark JSON in {args.bench_dir} (il --bench dettagliato e' WP-10)")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_arch = {}
    for r in runs:
        by_arch.setdefault(r["_arch"], []).append(r)
    for arch, arch_runs in sorted(by_arch.items()):
        plot_throughput_bars(arch, arch_runs, out_dir)
        plot_memory_bars(arch, arch_runs, out_dir)
        by_cfg = {}                                  # (dataset, mode) -> [run...]; boxplot per config
        for r in arch_runs:
            by_cfg.setdefault((r.get("dataset", "?"), r.get("mode", "?")), []).append(r)
        for (ds, mode), rs in sorted(by_cfg.items()):
            plot_latency_boxplot(arch, mode, ds, _merge_latency(rs), out_dir)
    plot_multiarch_throughput(runs, out_dir)
    plot_multiarch_memory(runs, out_dir)
    plot_multiarch_latency(runs, out_dir)


if __name__ == "__main__":
    main()
