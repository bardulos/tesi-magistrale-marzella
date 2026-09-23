"""lib/preprocessing/split.py — split semi-supervisionato cluster-aware (Stage 04, parte 1).

Estratto da transform.py: e' la funzione pura, deterministica e pesantemente verificata che
suddivide i flussi in train/val/test a livello di CLUSTER (righe con feature_cols identiche),
evitando il leakage da vettori duplicati. Benigni 80/10/10 per cluster, malevoli 0/50/50
stratificati su Attack-majority.
"""

import logging
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from lib import utils as common

log = logging.getLogger(__name__)


def _majority_value(s: pd.Series):
    """Valore piu' frequente di una Series (Attack-majority del cluster).

    Sostituisce la lambda `lambda s: s.value_counts().idxmax()` usata da .agg(),
    coerentemente con la convenzione 'niente lambda'.
    """
    return s.value_counts().idxmax()


def semisupervised_split(df: pd.DataFrame,
                         feature_cols: list[str],
                         label_col: str = "Label",
                         attack_col: str = "Attack",
                         benign_fracs: tuple[float, float, float] = common.BENIGN_FRACS,
                         malicious_fracs: tuple[float, float, float] = common.MALICIOUS_FRACS,
                         seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Split semi-supervisionato a livello di CLUSTER (gruppi di righe con lo stesso
    bit-pattern delle feature_cols: hash dei byte dei valori, non un confronto numerico).
    Evita il leakage da vettori duplicati.

    1. Hash su feature_cols -> cluster_id deterministico (vettoriale).
    2. Classifica cluster: PURE_BENIGN (solo Label=0), PURE_MALICIOUS (solo
       Label=1), MIXED (entrambe).
    3. PURE_BENIGN: shuffle determ. + accumulo per righe con `benign_fracs`
       (mai spezza un cluster).
    4. PURE_MALICIOUS: train=0; val/test 50/50 stratificato su Attack majority (il 50/50 e' cablato: malicious_fracs e' validata — somma 1, [0]=0 — ma non pilota le proporzioni).
    5. MIXED: train=0 (mai); val/test 50/50 stratificato sull'Attack majority
       delle sole righe MALEVOLE del cluster (porta anche queste categorie in val).
    6. Espansione cluster -> indici riga (maschere booleane compatte).
    7. Verifiche: zero malevoli train, zero cluster condivisi, conteggi.
    8. Shuffle finale intra-split, per mescolare benigni e malevoli.

    Ritorna (train_idx, val_idx, test_idx, cluster_stats).
    """
    if not np.isclose(sum(benign_fracs), 1.0):
        raise ValueError(f"benign_fracs non sommano a 1.0: {benign_fracs}")
    if not np.isclose(sum(malicious_fracs), 1.0):
        raise ValueError(f"malicious_fracs non sommano a 1.0: {malicious_fracs}")
    if malicious_fracs[0] != 0.0:
        raise ValueError(
            f"split cluster-aware: malicious_fracs[0] deve essere 0.0 (mai malevoli in train), trovato {malicious_fracs[0]}"
        )

    label = df[label_col].to_numpy()

    t0 = time.perf_counter()
    hashes = pd.util.hash_pandas_object(df[feature_cols], index=False).to_numpy()
    cluster_id, _ = pd.factorize(hashes, sort=False)
    n_clusters = int(cluster_id.max()) + 1
    log.info("hash + factorize: %d cluster su %d righe in %.1f s",
             n_clusters, len(df), time.perf_counter() - t0)

    grouped = (pd.DataFrame({"cid": cluster_id, "label": label})
               .groupby("cid")["label"].agg(["sum", "count"]))
    grouped["malicious"] = grouped["sum"]
    grouped["benign"] = grouped["count"] - grouped["sum"]
    grouped["kind"] = "MIXED"
    grouped.loc[grouped["malicious"] == 0, "kind"] = "PURE_BENIGN"
    grouped.loc[grouped["benign"] == 0, "kind"] = "PURE_MALICIOUS"

    pure_benign = grouped[grouped["kind"] == "PURE_BENIGN"]
    pure_malicious = grouped[grouped["kind"] == "PURE_MALICIOUS"]
    mixed = grouped[grouped["kind"] == "MIXED"]

    rows_pb = int(pure_benign["count"].sum())
    rows_pm = int(pure_malicious["count"].sum())
    rows_mx = int(mixed["count"].sum())
    rows_mx_b = int(mixed["benign"].sum())
    rows_mx_m = int(mixed["malicious"].sum())

    log.info("cluster totali: %d", n_clusters)
    log.info("cluster puri benigni:  %d (%d righe)", len(pure_benign), rows_pb)
    log.info("cluster puri malevoli: %d (%d righe)", len(pure_malicious), rows_pm)
    log.info("cluster mixed:         %d (%d righe, di cui %d benigne %d malevole)",
             len(mixed), rows_mx, rows_mx_b, rows_mx_m)

    # Sanity: la classificazione in tre specie ({0}, {1}, {0,1}) deve coprire ogni cluster.
    n_label_kinds = (pure_benign.shape[0] +
                     pure_malicious.shape[0] +
                     mixed.shape[0])
    assert n_label_kinds == n_clusters, "classificazione cluster non esaustiva (bug)"

    rng = np.random.default_rng(seed)

    pb_cids = pure_benign.index.to_numpy()
    pb_sizes = pure_benign["count"].to_numpy()
    perm_pb = rng.permutation(len(pb_cids))
    pb_cids_shuf = pb_cids[perm_pb]
    pb_sizes_shuf = pb_sizes[perm_pb]

    target_train = rows_pb * benign_fracs[0]
    target_val = rows_pb * (benign_fracs[0] + benign_fracs[1])

    train_clusters: list[int] = []
    val_clusters_b: list[int] = []
    test_clusters_b: list[int] = []
    running = 0
    n_train_rows = n_val_rows_b = n_test_rows_b = 0
    for cid, size in zip(pb_cids_shuf, pb_sizes_shuf):
        size_int = int(size)
        cid_int = int(cid)
        if running < target_train:
            train_clusters.append(cid_int)
            n_train_rows += size_int
        elif running < target_val:
            val_clusters_b.append(cid_int)
            n_val_rows_b += size_int
        else:
            test_clusters_b.append(cid_int)
            n_test_rows_b += size_int
        running += size_int

    log.info("PURE_BENIGN: train=%d cluster (%d righe, target %d=%.1f%%)  val=%d cluster (%d righe)  test=%d cluster (%d righe)",
             len(train_clusters), n_train_rows, int(target_train),
             benign_fracs[0] * 100,
             len(val_clusters_b), n_val_rows_b,
             len(test_clusters_b), n_test_rows_b)

    attack = df[attack_col].to_numpy()

    val_clusters_m: list[int] = []
    test_clusters_m: list[int] = []
    if len(pure_malicious) > 0:
        pm_cids = pure_malicious.index.to_numpy()
        pm_mask = np.isin(cluster_id, pm_cids)
        pm_rows_df = pd.DataFrame({"cid": cluster_id[pm_mask],
                                    "attack": attack[pm_mask]})
        attack_per_cluster = (pm_rows_df.groupby("cid")["attack"]
                              .agg(_majority_value))
        attack_labels = attack_per_cluster.reindex(pm_cids).to_numpy()

        try:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
            val_rel, test_rel = next(sss.split(np.zeros(len(pm_cids)), attack_labels))
            val_clusters_m = [int(c) for c in pm_cids[val_rel]]
            test_clusters_m = [int(c) for c in pm_cids[test_rel]]
        except ValueError as e:
            log.warning("StratifiedShuffleSplit fallita su PURE_MALICIOUS (%s); fallback alternato per categoria", e)
            for i, cat in enumerate(sorted(np.unique(attack_labels))):
                cat_cids = pm_cids[attack_labels == cat]
                rng_cat = np.random.default_rng(seed + 1000 + i)
                cat_perm = rng_cat.permutation(len(cat_cids))
                cat_shuf = cat_cids[cat_perm]
                half = len(cat_shuf) // 2
                val_clusters_m.extend(int(c) for c in cat_shuf[:half])
                test_clusters_m.extend(int(c) for c in cat_shuf[half:])

        n_val_rows_m = int(pure_malicious.loc[val_clusters_m, "count"].sum()) if val_clusters_m else 0
        n_test_rows_m = int(pure_malicious.loc[test_clusters_m, "count"].sum()) if test_clusters_m else 0
        log.info("PURE_MALICIOUS: val=%d cluster (%d righe)  test=%d cluster (%d righe)",
                 len(val_clusters_m), n_val_rows_m, len(test_clusters_m), n_test_rows_m)
    else:
        n_val_rows_m = n_test_rows_m = 0

    val_clusters_x: list[int] = []
    test_clusters_x: list[int] = []
    if len(mixed) > 0:
        mx_cids = mixed.index.to_numpy()
        mx_mal_mask = np.isin(cluster_id, mx_cids) & (label == 1)
        mx_rows_df = pd.DataFrame({"cid": cluster_id[mx_mal_mask],
                                    "attack": attack[mx_mal_mask]})
        attack_per_mx = (mx_rows_df.groupby("cid")["attack"]
                         .agg(_majority_value))
        attack_labels_x = attack_per_mx.reindex(mx_cids).to_numpy()

        try:
            sss_x = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
            val_rel, test_rel = next(sss_x.split(np.zeros(len(mx_cids)), attack_labels_x))
            val_clusters_x = [int(c) for c in mx_cids[val_rel]]
            test_clusters_x = [int(c) for c in mx_cids[test_rel]]
        except ValueError as e:
            log.warning("StratifiedShuffleSplit fallita su MIXED (%s); fallback alternato per categoria", e)
            for i, cat in enumerate(sorted(np.unique(attack_labels_x))):
                cat_cids = mx_cids[attack_labels_x == cat]
                rng_cat = np.random.default_rng(seed + 2000 + i)
                cat_perm = rng_cat.permutation(len(cat_cids))
                cat_shuf = cat_cids[cat_perm]
                half = len(cat_shuf) // 2
                val_clusters_x.extend(int(c) for c in cat_shuf[:half])
                test_clusters_x.extend(int(c) for c in cat_shuf[half:])

        n_val_rows_x = int(mixed.loc[val_clusters_x, "count"].sum()) if val_clusters_x else 0
        n_test_rows_x = int(mixed.loc[test_clusters_x, "count"].sum()) if test_clusters_x else 0
        log.info("MIXED: val=%d cluster (%d righe)  test=%d cluster (%d righe)",
                 len(val_clusters_x), n_val_rows_x, len(test_clusters_x), n_test_rows_x)
    else:
        n_val_rows_x = n_test_rows_x = 0

    mask_train = np.zeros(n_clusters, dtype=bool)
    mask_train[train_clusters] = True
    mask_val = np.zeros(n_clusters, dtype=bool)
    mask_val[val_clusters_b + val_clusters_m + val_clusters_x] = True
    mask_test = np.zeros(n_clusters, dtype=bool)
    mask_test[test_clusters_b + test_clusters_m + test_clusters_x] = True

    train_idx = np.where(mask_train[cluster_id])[0]
    val_idx = np.where(mask_val[cluster_id])[0]
    test_idx = np.where(mask_test[cluster_id])[0]

    assert (label[train_idx] == 1).sum() == 0, "malevoli trovati in train (bug cluster-split)"
    train_set = set(train_clusters)
    val_set = set(val_clusters_b + val_clusters_m + val_clusters_x)
    test_set = set(test_clusters_b + test_clusters_m + test_clusters_x)
    assert len(train_set & val_set) == 0, "leakage: cluster condivisi train-val"
    assert len(train_set & test_set) == 0, "leakage: cluster condivisi train-test"
    assert len(val_set & test_set) == 0, "leakage: cluster condivisi val-test"
    total = len(train_idx) + len(val_idx) + len(test_idx)
    assert total == len(df), f"conteggio righe: {total} != {len(df)}"

    rng_shuffle = np.random.default_rng(seed)
    rng_shuffle.shuffle(train_idx)
    rng_shuffle.shuffle(val_idx)
    rng_shuffle.shuffle(test_idx)

    log.info("")
    log.info("split cluster-aware:")
    log.info("  train: %d righe (%d cluster, 100%% benigni)",
             len(train_idx), len(train_clusters))
    log.info("  val:   %d righe (%d cluster: %d pure_ben + %d pure_mal + %d mixed)",
             len(val_idx), len(val_set),
             len(val_clusters_b), len(val_clusters_m), len(val_clusters_x))
    log.info("  test:  %d righe (%d cluster: %d pure_ben + %d pure_mal + %d mixed)",
             len(test_idx), len(test_set),
             len(test_clusters_b), len(test_clusters_m), len(test_clusters_x))
    log.info("leakage check: 0 cluster condivisi tra split")

    cluster_stats = {
        "n_total": n_clusters,
        "n_pure_benign": int(len(pure_benign)),
        "n_pure_malicious": int(len(pure_malicious)),
        "n_mixed": int(len(mixed)),
        "rows_pure_benign": rows_pb,
        "rows_pure_malicious": rows_pm,
        "rows_mixed": rows_mx,
        "rows_mixed_benign": rows_mx_b,
        "rows_mixed_malicious": rows_mx_m,
        "n_train_clusters": len(train_clusters),
        "n_val_clusters": len(val_set),
        "n_test_clusters": len(test_set),
    }

    return (train_idx.astype(np.int64),
            val_idx.astype(np.int64),
            test_idx.astype(np.int64),
            cluster_stats)


def log_split_summary(df: pd.DataFrame, train_idx: np.ndarray,
                      val_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """Log riassuntivo di composizione split (benigni/malevoli, copertura Attack)."""
    label = df["Label"].to_numpy()
    n_b_train = int((label[train_idx] == 0).sum())
    n_b_val = int((label[val_idx] == 0).sum())
    n_b_test = int((label[test_idx] == 0).sum())
    n_m_train = int((label[train_idx] == 1).sum())
    n_m_val = int((label[val_idx] == 1).sum())
    n_m_test = int((label[test_idx] == 1).sum())
    n_b_tot = n_b_train + n_b_val + n_b_test
    n_m_tot = n_m_train + n_m_val + n_m_test

    log.info("benigni:  totale=%d  train=%d (%.1f%%)  val=%d (%.1f%%)  test=%d (%.1f%%)",
             n_b_tot,
             n_b_train, n_b_train / n_b_tot * 100 if n_b_tot else 0,
             n_b_val, n_b_val / n_b_tot * 100 if n_b_tot else 0,
             n_b_test, n_b_test / n_b_tot * 100 if n_b_tot else 0)
    log.info("malevoli: totale=%d  train=%d (%.1f%%)  val=%d (%.1f%%)  test=%d (%.1f%%)",
             n_m_tot,
             n_m_train, n_m_train / n_m_tot * 100 if n_m_tot else 0,
             n_m_val, n_m_val / n_m_tot * 100 if n_m_tot else 0,
             n_m_test, n_m_test / n_m_tot * 100 if n_m_tot else 0)
    log.info("")

    attack = df["Attack"].to_numpy()
    mal_categories = sorted(set(attack[label == 1].tolist()))
    log.info("distribuzione Attack in val/test (%d categorie malevole):", len(mal_categories))
    log.info("  %-30s %12s %12s", "Attack", "val", "test")
    missing = []
    for cat in mal_categories:
        v = int(((attack[val_idx] == cat) & (label[val_idx] == 1)).sum())
        t = int(((attack[test_idx] == cat) & (label[test_idx] == 1)).sum())
        log.info("  %-30s %12s %12s", cat, f"{v:,}", f"{t:,}")
        if v == 0:
            missing.append((cat, "val"))
        if t == 0:
            missing.append((cat, "test"))
    if missing:
        log.warning("verifica: %d categorie malevole con 0 campioni: %s",
                    len(missing), ", ".join(f"{c}/{s}" for c, s in missing))
    else:
        log.info("verifica: %d/%d categorie malevole in val e test",
                 len(mal_categories), len(mal_categories))
    log.info("")
    log.info("composizione split finali:")
    log.info("  train: %d righe (%d benigni + %d malevoli)",
             len(train_idx), n_b_train, n_m_train)
    log.info("  val:   %d righe (%d benigni + %d malevoli)",
             len(val_idx), n_b_val, n_m_val)
    log.info("  test:  %d righe (%d benigni + %d malevoli)",
             len(test_idx), n_b_test, n_m_test)
