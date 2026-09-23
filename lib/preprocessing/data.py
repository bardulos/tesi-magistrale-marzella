"""lib/preprocessing/data.py — risoluzione handle, ingest CSV grezzo -> Parquet,
e stage descrittivi del capitolo 1 (inspect / stats / audit).

- run_ingest_stage  : CSV grezzo -> Parquet snappy.
- run_inspect_stage : conteggi per-colonna sul CSV grezzo (non_zero/zero/min/max).
- run_stats_stage   : statistiche per-feature (mean/std/skew/kurt/%zero + split benigni/malevoli).
- run_audit_stage   : 8 analisi post-reduce (Spearman ff/fl, MI, RF-importance, KS/Wasserstein,
                      decomposizione varianza, entropia condizionata, costo informativo).

Gli stage descrittivi sono ESPLORATIVI (alimentano i NUMERI del cap1, non artefatti gated).
Portati da repo_v3, adattati a 91 feature; oltre al log scrivono un JSON persistente in out_dir
(per inventario e referenziazione). scipy/sklearn importati lazy nelle funzioni che li
usano (l'import del modulo resta leggero per gli stage che non li richiedono, es. ingest).
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lib import utils as common

log = logging.getLogger(__name__)

# Path verso la radice del progetto (lib/preprocessing/data.py -> 3 livelli sopra).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
HANDLES = {
    "cse": "NF-CICIDS2018-v3.csv",
    "unsw": "NF-UNSW-NB15-v3.csv",
    "ton": "NF-ToN-IoT-v3.csv",
}


def resolve_dataset(handle: str) -> Path:
    p = DATA_RAW / HANDLES[handle]
    if not p.exists():
        raise FileNotFoundError(f"file non trovato: {p}")
    return p


def load_dataset(handle: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Legge il CSV grezzo dell'handle in un DataFrame pandas.

    `encoding_errors="replace"` e' la stessa guardia di inference/infer.py::_open_csv:
    i CSV accademici misurati (CSE/UNSW) sono ASCII 7-bit puliti, ma le catture LAN
    reali contengono byte non-UTF8 (~1 riga su 150 000): il preprocessing "fuori" e
    quello del pacchetto prevedono la stessa possibilita'.
    """
    return pd.read_csv(resolve_dataset(handle), low_memory=False, usecols=usecols,
                       encoding_errors="replace")


def ingest_to_parquet(handle: str, out_path: Path) -> None:
    """CSV grezzo -> Parquet snappy. Tipi 1:1 da pd.read_csv (no cast).
    Cosi' il dataframe risultante e' identico a load_dataset(handle),
    garantendo md5-equivalence dell'encode downstream.
    """
    src = resolve_dataset(handle)
    csv_mb = src.stat().st_size / 1_048_576
    log.info("ingest %s: leggo CSV %s (%.1f MB)...", handle, src.name, csv_mb)
    t0 = time.perf_counter()
    df = pd.read_csv(src, low_memory=False, encoding_errors="replace")
    t_csv = time.perf_counter() - t0
    log.info("CSV caricato in %.1f s: rows=%d cols=%d", t_csv, len(df), len(df.columns))

    t0 = time.perf_counter()
    df.to_parquet(out_path, compression="snappy", index=False)
    t_pq = time.perf_counter() - t0
    pq_mb = out_path.stat().st_size / 1_048_576
    log.info("Parquet scritto in %.1f s: %s (%.1f MB)", t_pq, out_path, pq_mb)
    log.info("compressione: %.2fx (%.1f MB -> %.1f MB)",
             csv_mb / pq_mb if pq_mb > 0 else 0, csv_mb, pq_mb)


def run_ingest_stage(cfg: dict) -> None:
    """CSV grezzo -> Parquet snappy per uno o piu' dataset.

    Il `cfg` (dict letto da YAML) deve contenere:
      - dataset: stringa o lista di stringhe (es. "cse" o ["cse", "unsw"])
      - out_dir: directory output (deve gia' esistere)
    """
    out_dir_str = cfg["out_dir"]
    out_dir = Path(out_dir_str)
    if not out_dir.exists():
        raise FileNotFoundError(f"out_dir non esiste: {out_dir}")

    datasets = cfg["dataset"]
    if isinstance(datasets, str):
        datasets = [datasets]

    for ds in datasets:
        common.setup_logging(out_dir / f"ingest_{ds}.log")
        log.info("===== Stage 00 ingest -- %s =====", ds)
        out_path = out_dir / f"{ds}_raw.parquet"
        ingest_to_parquet(ds, out_path)
        log.info("===== completato =====")


def _json_default(o):
    """Coercizione per json.dump di tipi numpy/None/str."""
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def run_inspect_stage(cfg: dict) -> None:
    """Conteggi per-colonna sul CSV grezzo (non_zero/zero/min/max sui non-zero).

    cfg: dataset (handle singolo cse/unsw/ton), out_dir (deve esistere).
    Output: log + out_dir/inspect_stats.json.
    """
    dataset = cfg["dataset"]
    if not isinstance(dataset, str):
        dataset = dataset[0]
    out_dir = Path(cfg["out_dir"])
    if not out_dir.exists():
        raise FileNotFoundError(f"out_dir non esiste: {out_dir}")
    common.setup_logging(out_dir / "log.txt")

    path = resolve_dataset(dataset)
    log.info("===== Stage inspect -- %s =====", dataset)
    log.info("path: %s (%.1f MB)", path, path.stat().st_size / 1_048_576)
    df = load_dataset(dataset)
    n_rows, n_cols = len(df), len(df.columns)
    log.info("caricato: rows=%d cols=%d", n_rows, n_cols)

    stats: dict[str, dict] = {}
    for col in df.columns:
        non_nan = df[col].dropna()
        n_zero = int((non_nan == 0).sum())
        nz = non_nan[non_nan != 0]
        stats[col] = {
            "non_zero": int(len(nz)),
            "zero": n_zero,
            "min": (nz.min() if len(nz) else None),
            "max": (nz.max() if len(nz) else None),
        }

    schema = ("OK" if n_cols == common.EXPECTED_COLS
              else f"MISMATCH (atteso {common.EXPECTED_COLS}, trovato {n_cols})")
    log.info("schema: %s", schema)
    log.info("--- per-column stats (non_zero/zero/min/max sui non-zero) ---")
    col_w = max(len(c) for c in stats)
    log.info("%-4s %-*s %15s %15s %25s %25s", "#", col_w, "colonna", "non_zero", "zero", "min", "max")
    for i, (col, s) in enumerate(stats.items(), 1):
        log.info("%-4d %-*s %15s %15s %25s %25s", i, col_w, col,
                 f"{s['non_zero']:,}", f"{s['zero']:,}", str(s["min"]), str(s["max"]))

    out_json = out_dir / "inspect_stats.json"
    out_json.write_text(json.dumps(
        {"dataset": dataset, "n_rows": n_rows, "n_cols": n_cols, "schema": schema,
         "columns": stats}, indent=2, default=_json_default))
    log.info("scritto %s", out_json)


def run_stats_stage(cfg: dict) -> None:
    """Statistiche descrittive per-feature sul CSV grezzo (richiede colonna Label):
    n_uniq/n_nan/n_inf/n_zero/%zero/min/max/mean/std/median/skew/kurt + split benigni/malevoli.

    cfg: dataset (handle singolo), out_dir (deve esistere).
    Output: log + out_dir/feature_stats.json.
    """
    from scipy.stats import kurtosis, skew

    dataset = cfg["dataset"]
    if not isinstance(dataset, str):
        dataset = dataset[0]
    out_dir = Path(cfg["out_dir"])
    if not out_dir.exists():
        raise FileNotFoundError(f"out_dir non esiste: {out_dir}")
    common.setup_logging(out_dir / "log.txt")

    path = resolve_dataset(dataset)
    log.info("===== Stage stats -- %s =====", dataset)
    log.info("path: %s (%.1f MB)", path, path.stat().st_size / 1_048_576)
    df = load_dataset(dataset)
    log.info("caricato: rows=%d cols=%d", len(df), len(df.columns))
    if "Label" not in df.columns:
        raise SystemExit("dataset senza colonna Label, stage stats non applicabile")

    feature_cols = sorted(c for c in df.columns
                          if c not in common.SKIP_COLS and pd.api.types.is_numeric_dtype(df[c]))
    benign = df[df["Label"] == 0]
    malicious = df[df["Label"] == 1]
    log.info("benign=%d malicious=%d features=%d", len(benign), len(malicious), len(feature_cols))

    headers = ["feature", "n_uniq", "n_nan", "n_inf", "n_zero", "%zero", "min", "max",
               "mean", "std", "median", "skew", "kurt",
               "mean_b", "mean_m", "std_b", "std_m", "med_b", "med_m"]
    name_w = max(len(headers[0]), max(len(c) for c in feature_cols))
    num_w = 12
    log.info(f"{headers[0]:<{name_w}}" + "".join(f" {h:>{num_w}}" for h in headers[1:]))

    rows = []
    for col in feature_cols:
        n_nan = int(df[col].isna().sum())
        s_all = df[col].dropna()
        n_inf = int(np.isinf(s_all).sum())
        s_b, s_m = benign[col].dropna(), malicious[col].dropna()
        n_uniq = int(s_all.nunique())
        n_zero = int((s_all == 0).sum())
        pct_zero = n_zero / len(s_all) * 100 if len(s_all) else 0.0
        rec = {
            "feature": col, "n_uniq": n_uniq, "n_nan": n_nan, "n_inf": n_inf,
            "n_zero": n_zero, "pct_zero": pct_zero,
            "min": float(s_all.min()), "max": float(s_all.max()),
            "mean": float(s_all.mean()), "std": float(s_all.std()),
            "median": float(s_all.median()),
            "skew": float(skew(s_all.values)), "kurt": float(kurtosis(s_all.values)),
            "mean_b": float(s_b.mean()) if len(s_b) else float("nan"),
            "mean_m": float(s_m.mean()) if len(s_m) else float("nan"),
            "std_b": float(s_b.std()) if len(s_b) else float("nan"),
            "std_m": float(s_m.std()) if len(s_m) else float("nan"),
            "med_b": float(s_b.median()) if len(s_b) else float("nan"),
            "med_m": float(s_m.median()) if len(s_m) else float("nan"),
        }
        rows.append(rec)
        values = [rec[h.replace("%zero", "pct_zero")] for h in headers[1:]]
        log.info(f"{col:<{name_w}}" + "".join(f" {v:>{num_w}.4g}" for v in values))

    out_json = out_dir / "feature_stats.json"
    out_json.write_text(json.dumps({"dataset": dataset, "n_features": len(rows),
                                    "features": rows}, indent=2, default=_json_default))
    log.info("summary: %d feature processate -> %s", len(rows), out_json)


# --- 7 helper di audit (logica identica all'oracolo repo_v3) ----------------


def compute_spearman_ff(df, continuous_cols, threshold=0.90):
    """Spearman feature-vs-feature via rank+pearson. Ritorna (a,b,rho) |rho|>=threshold, desc."""
    ranked = df[continuous_cols].rank(method="average")
    corr = ranked.corr(method="pearson")
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack().sort_values(key=lambda s: s.abs(), ascending=False)
    filtered = pairs[pairs.abs() >= threshold]
    return [(a, b, float(rho)) for (a, b), rho in filtered.items()]


def compute_spearman_fl(df, continuous_cols, label_col="Label", threshold=0.10):
    """Spearman feature-vs-label. Skip feature con rank costante (std=0). |rho|>=threshold, desc."""
    label_ranked = df[label_col].rank(method="average")
    feature_ranked = df[continuous_cols].rank(method="average")
    results = []
    for col in continuous_cols:
        fr = feature_ranked[col]
        if fr.std() == 0:
            continue
        rho = float(fr.corr(label_ranked, method="pearson"))
        if abs(rho) >= threshold:
            results.append((col, rho))
    results.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return results


def compute_mi_label(df, feature_types, label_col="Label", n_bins=50, top_n=30):
    """NMI feature-vs-label. Continue: pd.qcut(n_bins). Binarie: dirette. Top-N desc."""
    from sklearn.metrics import normalized_mutual_info_score
    label = df[label_col].values
    results = []
    for feat, ftype in feature_types.items():
        if feat not in df.columns:
            continue
        col = df[feat]
        if ftype == "binary":
            score = float(normalized_mutual_info_score(col.values, label))
        elif ftype == "continuous":
            mask = col.notna()
            non_nan = col[mask]
            if non_nan.nunique() < 2:
                results.append((feat, 0.0))
                continue
            try:
                disc = pd.qcut(non_nan, q=n_bins, labels=False, duplicates="drop")
            except ValueError:
                results.append((feat, 0.0))
                continue
            score = float(normalized_mutual_info_score(disc.values, label[mask.values]))
        else:
            continue
        results.append((feat, score))
    results.sort(key=lambda kv: kv[1], reverse=True)
    return results[:top_n]


def compute_rf_importance(df, feature_cols, label_col="Label", n_sample=200_000,
                          n_folds=5, top_n=30, random_state=42):
    """Sample stratificato 50/50, RF n_estimators=100, n_folds CV, Gini importance media. Top-N desc."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    rng = np.random.default_rng(random_state)
    benign_idx = df.index[df[label_col] == 0].to_numpy()
    malicious_idx = df.index[df[label_col] == 1].to_numpy()
    n_each = n_sample // 2
    n_b = min(n_each, len(benign_idx))
    n_m = min(n_each, len(malicious_idx))
    sample_idx = np.concatenate([rng.choice(benign_idx, n_b, replace=False),
                                 rng.choice(malicious_idx, n_m, replace=False)])
    rng.shuffle(sample_idx)
    log.info("  sample: benigni=%d maligni=%d totale=%d", n_b, n_m, len(sample_idx))
    sub = df.loc[sample_idx]
    X = sub[feature_cols].fillna(0).to_numpy()
    y = sub[label_col].to_numpy()
    importances = np.zeros((n_folds, len(feature_cols)))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for i, (train_idx, _) in enumerate(skf.split(X, y)):
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state)
        rf.fit(X[train_idx], y[train_idx])
        importances[i] = rf.feature_importances_
        log.info("  fold %d/%d: fit completato", i + 1, n_folds)
    avg = importances.mean(axis=0)
    paired = sorted(zip(feature_cols, avg), key=lambda kv: kv[1], reverse=True)
    return [(c, float(imp)) for c, imp in paired[:top_n]]


def compute_distributional_shift(primary_benign, compare_named, continuous_cols,
                                 sample_size=500_000, top_n=20, random_state=42):
    """Wasserstein + KS tra primary_benign e compare benign su continue. Top-N per Wasserstein desc."""
    from scipy.stats import ks_2samp, wasserstein_distance
    rng = np.random.default_rng(random_state)
    name, compare_df = compare_named
    results = []
    for feat in continuous_cols:
        if feat not in primary_benign.columns or feat not in compare_df.columns:
            continue
        p_vals = primary_benign[feat].dropna().to_numpy()
        c_vals = compare_df[feat].dropna().to_numpy()
        if len(p_vals) == 0 or len(c_vals) == 0:
            continue
        if len(p_vals) > sample_size:
            p_vals = rng.choice(p_vals, sample_size, replace=False)
        if len(c_vals) > sample_size:
            c_vals = rng.choice(c_vals, sample_size, replace=False)
        w = float(wasserstein_distance(p_vals, c_vals))
        ks = float(ks_2samp(p_vals, c_vals).statistic)
        results.append({"feature": feat, f"w_{name}": w, f"ks_{name}": ks})
    results.sort(key=lambda r: r[f"w_{name}"], reverse=True)
    return results[:top_n]


def compute_variance_decomposition(df, continuous_cols, existing_flags=None, threshold=0.5):
    """Decomposizione varianza per continua: var_presenza=p(1-p)mu_nz^2, rapporto=var_pres/var_tot.
    SPLIT se rapporto>threshold. existing_flags: nota flag companion. Ordinata per rapporto desc."""
    existing = existing_flags or {}
    results = []
    for col in continuous_cols:
        if col not in df.columns:
            continue
        x = df[col].dropna()
        n = len(x)
        if n == 0:
            continue
        n_zero = int((x == 0).sum())
        p = n_zero / n
        nz = x[x != 0]
        mu_nz = float(nz.mean()) if len(nz) > 0 else 0.0
        var_total = float(x.var())
        var_presenza = p * (1.0 - p) * (mu_nz ** 2)
        ratio = var_presenza / var_total if var_total > 0 else 0.0
        nota = ["SPLIT" if ratio > threshold else "KEEP"]
        if col in existing:
            nota.append(f"flag esistente: {existing[col]}")
        results.append({"feature": col, "pct_zero": p * 100, "mu_nz": mu_nz,
                        "var_totale": var_total, "var_presenza": var_presenza,
                        "rapporto": ratio, "nota": " — ".join(nota)})
    results.sort(key=lambda r: r["rapporto"], reverse=True)
    return results


def compute_conditional_entropy_pairs(df, pairs, n_bins=50):
    """Per coppia (A,B,rho): H(A),H(B),MI,H(A|B),H(B|A) in nat; 'keep'=membro con entropia maggiore."""
    from scipy.stats import entropy
    from sklearn.metrics import mutual_info_score
    results = []
    for a, b, rho in pairs:
        if a not in df.columns or b not in df.columns:
            continue
        sa, sb = df[a].dropna(), df[b].dropna()
        idx = sa.index.intersection(sb.index)
        sa, sb = sa.loc[idx], sb.loc[idx]
        if len(sa) < 2:
            continue
        try:
            disc_a = pd.qcut(sa, q=n_bins, labels=False, duplicates="drop")
            disc_b = pd.qcut(sb, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        m = disc_a.notna() & disc_b.notna()
        disc_a = disc_a[m].astype(int).to_numpy()
        disc_b = disc_b[m].astype(int).to_numpy()
        if len(disc_a) < 2:
            continue
        _, ca = np.unique(disc_a, return_counts=True)
        _, cb = np.unique(disc_b, return_counts=True)
        h_a, h_b = float(entropy(ca)), float(entropy(cb))
        mi = float(mutual_info_score(disc_a, disc_b))
        results.append({"a": a, "b": b, "rho": rho, "H_a": h_a, "H_b": h_b, "MI": mi,
                        "H_a_given_b": max(0.0, h_a - mi), "H_b_given_a": max(0.0, h_b - mi),
                        "keep": a if h_a >= h_b else b})
    return results


def compute_information_cost(df, targets, feature_types, n_bins=50, top_n=20):
    """Costo informativo di una feature: H(target) e quota catturata da ogni altra colonna.

    Ottava analisi. Generalizzazione feature-contro-feature della mutua informazione gia' usata
    da compute_mi_label (feature-vs-Label) e da compute_conditional_entropy_pairs (solo le
    coppie |rho|>=0.98): stessa discretizzazione, pd.qcut(n_bins) sulle continue e valori
    diretti sulle binarie. H e MI in nat.

    quota = MI / H: quanta informazione del target sopravvive in un'altra colonna. Il
    complemento della quota MASSIMA e' l'informazione unica che la rimozione butta via.

    H e' calcolata sulle righe valide del target; la MI sulle righe valide di entrambi. Sul
    parquet canonico i NaN sono zero (sanity check dell'encode) e i due insiemi coincidono;
    'n_rows' e 'n_rows_comuni' restano nel referto per rendere visibile un eventuale scarto.
    """
    from scipy.stats import entropy
    from sklearn.metrics import mutual_info_score

    def discretize(feat):
        """(valori discretizzati sulle righe valide, maschera booleana) oppure (None, None)."""
        ftype = feature_types.get(feat)
        if feat not in df.columns or ftype not in ("binary", "continuous"):
            return None, None
        col = df[feat]
        mask = col.notna().to_numpy()
        valid = col[mask]
        if valid.nunique() < 2:
            return None, None
        if ftype == "binary":
            return valid.to_numpy(), mask
        try:
            disc = pd.qcut(valid, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            return None, None
        return disc.to_numpy(), mask

    results = []
    for target in targets:
        t_vals, t_mask = discretize(target)
        if t_vals is None:
            log.info("  %s: assente o non discretizzabile, saltata", target)
            continue
        _, counts = np.unique(t_vals, return_counts=True)
        h_target = float(entropy(counts))

        scored = []
        for other in feature_types:
            if other == target:
                continue
            o_vals, o_mask = discretize(other)
            if o_vals is None:
                continue
            t_common = t_vals[o_mask[t_mask]]
            o_common = o_vals[t_mask[o_mask]]
            if len(t_common) < 2:
                continue
            mi = float(mutual_info_score(t_common, o_common))
            scored.append({"other": other, "MI": mi,
                           "quota_pct": mi / h_target * 100 if h_target > 0 else 0.0,
                           "n_rows_comuni": int(len(t_common))})
        scored.sort(key=lambda r: r["MI"], reverse=True)

        quota_max = scored[0]["quota_pct"] if scored else 0.0
        results.append({"feature": target, "H": h_target,
                        "n_bins_effettivi": int(len(counts)),
                        "n_rows": int(t_mask.sum()),
                        "n_confrontate": len(scored),
                        "argmax": scored[0]["other"] if scored else None,
                        "quota_max_pct": quota_max,
                        "unica_pct": 100.0 - quota_max,
                        "top": scored[:top_n]})
    return results


def run_audit_stage(cfg: dict) -> None:
    """8 analisi post-reduce su un parquet primary, con confronto distribuzionale vs compare.

    cfg: primary (parquet post_reduce, l'ingresso dell'audit), schema_from
    (feature_types_post_audit.json),
    compare (parquet di confronto), out_dir (deve esistere).
    Output: log + out_dir/audit_results.json (le 8 analisi strutturate).
    """
    primary_path, schema_path = cfg["primary"], cfg["schema_from"]
    compare_path = cfg["compare"]
    out_dir = Path(cfg["out_dir"])
    if not out_dir.exists():
        raise FileNotFoundError(f"out_dir non esiste: {out_dir}")
    common.setup_logging(out_dir / "log.txt")

    t_total = time.perf_counter()
    log.info("===== Stage audit -- %s =====", Path(primary_path).stem)
    df = pd.read_parquet(primary_path)
    with open(schema_path, encoding="utf-8") as f:
        types = json.load(f)
    continuous = [c for c, t in types.items() if t == "continuous"]
    binary = [c for c, t in types.items() if t == "binary"]
    log.info("loaded: rows=%d cols=%d continuous=%d binary=%d",
             len(df), len(df.columns), len(continuous), len(binary))

    res: dict = {"primary": str(primary_path), "n_continuous": len(continuous),
                 "n_binary": len(binary)}

    log.info("--- 1. Spearman feature-feature (continue %dx%d) ---", len(continuous), len(continuous))
    pairs = compute_spearman_ff(df, continuous, threshold=common.SPEARMAN_FF_THRESHOLD)
    for a, b, rho in pairs:
        log.info("  %s <-> %s: rho=%.4f", a, b, rho)
    res["spearman_ff"] = [{"a": a, "b": b, "rho": rho} for a, b, rho in pairs]

    log.info("--- 2. Spearman feature-label (continue) ---")
    fl = compute_spearman_fl(df, continuous, threshold=common.SPEARMAN_FL_THRESHOLD)
    for feat, rho in fl:
        log.info("  %s: rho=%.4f", feat, rho)
    res["spearman_fl"] = [{"feature": f, "rho": r} for f, r in fl]

    log.info("--- 3. Mutua informazione feature-label (tutte %d) ---", len(continuous) + len(binary))
    mi = compute_mi_label(df, types, top_n=common.MI_TOP_N)
    for feat, nmi in mi:
        log.info("  %s: nmi=%.4f", feat, nmi)
    res["mi_label"] = [{"feature": f, "nmi": v} for f, v in mi]

    log.info("--- 4. Importanza RF (campione %dk, %d-fold) ---",
             common.RF_N_SAMPLE // 1000, common.RF_N_FOLDS)
    rf_imp = compute_rf_importance(df, continuous + binary, n_sample=common.RF_N_SAMPLE,
                                   n_folds=common.RF_N_FOLDS, top_n=common.RF_TOP_N)
    for feat, imp in rf_imp:
        log.info("  %s: importance=%.4f", feat, imp)
    res["rf_importance"] = [{"feature": f, "importance": v} for f, v in rf_imp]

    log.info("--- 5. Scostamento distribuzionale (benigni primary vs compare) ---")
    primary_benign = df[df["Label"] == 0] if "Label" in df.columns else df
    cdf = pd.read_parquet(compare_path)
    cbenign = cdf[cdf["Label"] == 0] if "Label" in cdf.columns else cdf
    name = Path(compare_path).stem.replace("_encoded", "").replace("_post_audit", "")
    log.info("primary benign rows=%d, compare (%s) benign rows=%d", len(primary_benign), name, len(cbenign))
    shift = compute_distributional_shift(primary_benign, (name, cbenign), continuous,
                                         sample_size=common.WASSERSTEIN_SAMPLE_SIZE,
                                         top_n=common.WASSERSTEIN_TOP_N)
    for r in shift:
        log.info("  %-32s w=%.4g ks=%.4f", r["feature"], r[f"w_{name}"], r[f"ks_{name}"])
    res["distributional_shift"] = shift

    log.info("--- 6. Decomposizione varianza feature continue ---")
    var_dec = compute_variance_decomposition(df, continuous, existing_flags=common.EXISTING_SPARSE_FLAGS,
                                             threshold=common.VARIANCE_RATIO_THRESHOLD)
    n_split = sum(1 for r in var_dec if r["rapporto"] > common.VARIANCE_RATIO_THRESHOLD)
    log.info("%d feature, %d SPLIT, %d KEEP", len(var_dec), n_split, len(var_dec) - n_split)
    for r in var_dec:
        log.info("  %-32s %%zero=%.2f rapporto=%.4f  %s", r["feature"], r["pct_zero"], r["rapporto"], r["nota"])
    res["variance_decomposition"] = var_dec

    log.info("--- 7. Entropia condizionata coppie ridondanti (|rho|>=%.2f) ---",
             common.SPEARMAN_FF_REDUNDANT_THRESHOLD)
    red_pairs = [(a, b, rho) for a, b, rho in pairs if abs(rho) >= common.SPEARMAN_FF_REDUNDANT_THRESHOLD]
    ce = compute_conditional_entropy_pairs(df, red_pairs)
    for r in ce:
        log.info("  %s <-> %s: rho=%.4f H(A)=%.4f H(B)=%.4f MI=%.4f keep=%s",
                 r["a"], r["b"], r["rho"], r["H_a"], r["H_b"], r["MI"], r["keep"])
    res["conditional_entropy"] = ce

    log.info("--- 8. Costo informativo (MI feature-contro-feature, %d bin) ---",
             common.INFORMATION_COST_N_BINS)
    info_cost = compute_information_cost(df, common.INFORMATION_COST_TARGETS, types,
                                         n_bins=common.INFORMATION_COST_N_BINS,
                                         top_n=common.INFORMATION_COST_TOP_N)
    for r in info_cost:
        log.info("  %s: H=%.6f nat (%d bin effettivi, %d righe), %d feature confrontate",
                 r["feature"], r["H"], r["n_bins_effettivi"], r["n_rows"], r["n_confrontate"])
        for e in r["top"]:
            log.info("    %-32s MI=%.6f quota=%.4f%%", e["other"], e["MI"], e["quota_pct"])
        log.info("  %s: quota massima %.4f%% (%s) -> informazione unica %.4f%%",
                 r["feature"], r["quota_max_pct"], r["argmax"], r["unica_pct"])
    res["information_cost"] = info_cost

    out_json = out_dir / "audit_results.json"
    out_json.write_text(json.dumps(res, indent=2, default=_json_default))
    log.info("===== completato in %.1f s -> %s =====", time.perf_counter() - t_total, out_json)
