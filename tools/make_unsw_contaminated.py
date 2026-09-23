#!/usr/bin/env python3
# TAG: ONE-SHOT | 2026-07-18 | crea un UNSW gia' contaminato all'1% (dato dimostrativo retrain) | vedi docs/refactoring_censimento.md
"""Crea un dataset UNSW gia' CONTAMINATO all'1% da usare come traffico locale "reale" per il retrain.

Semantica ADD (decisa dal maintainer): ai benigni di dati/addestramento/unsw_train_raw.csv
(1.778.334, tutti Label=0) si AGGIUNGONO k attacchi (Label=1) presi dal raw etichettato
data/raw/NF-UNSW-NB15-v3.csv, tali che gli attacchi siano l'1% del TOTALE:
    k / (N + k) = 0.01  ->  k = round(N * 0.01 / 0.99) = 17.963  (totale 1.796.297, prevalenza 1,000%).

Attenzioni (richieste dal maintainer):
  * NO-LEAK: si escludono gli attacchi gia' presenti nel test set (dati/inferenza/unsw_test_raw.csv),
    confrontando per CHIAVE numerica (i primi 6 campi: tempi + IP + porte, stabili). Cosi' il
    contaminato resta valutabile onestamente su quel test. Ne restano a sufficienza.
  * FORMATO: il raw e' CRLF, il train e' LF -> si strippa il \r dagli attacchi.
  * MESCOLAMENTO: gli attacchi vanno RANDOM nel file, NON in fondo. Le posizioni sono scelte con
    rng(0) deterministico; i benigni si leggono in streaming (non entrano in RAM), gli attacchi
    (piccoli) stanno in RAM, e si fondono per posizione.

Output: dati/addestramento/unsw_train_contam1_raw.csv (LF, header identico al train).
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path.home() / "tesi/repov6"
TRAIN = REPO / "inference/dati/addestramento/unsw_train_raw.csv"      # 1.778.334 benigni (LF)
RAW = REPO / "data/raw/NF-UNSW-NB15-v3.csv"                            # etichettato (CRLF)
TEST = REPO / "inference/dati/inferenza/unsw_test_raw.csv"            # per il no-leak
OUT = REPO / "inference/dati/addestramento/unsw_train_contam1_raw.csv"
LABEL_COL = 53           # 0-based (54a colonna)
PREVALENZA = 0.01
SEED = 0
N_CHIAVE = 6             # primi 6 campi: FLOW_START/END, IP_SRC, PORT_SRC, IP_DST, PORT_DST


def chiave(campi):
    return b",".join(campi[:N_CHIAVE])


def main():
    header = open(TRAIN, "rb").readline().rstrip(b"\r\n")
    n_benign = sum(1 for _ in open(TRAIN, "rb")) - 1
    k = int(round(n_benign * PREVALENZA / (1.0 - PREVALENZA)))
    print(f"benigni: {n_benign:,} | attacchi da aggiungere: {k:,} "
          f"| totale: {n_benign + k:,} | prevalenza attesa: {k / (n_benign + k) * 100:.4f}%")

    # chiavi degli attacchi gia' nel test set (no-leak)
    test_att = set()
    with open(TEST, "rb") as f:
        next(f)
        for l in f:
            c = l.rstrip(b"\r\n").split(b",")
            if len(c) > LABEL_COL and c[LABEL_COL] == b"1":
                test_att.add(chiave(c))
    print(f"attacchi nel test set (da escludere): {len(test_att):,}")

    # raccogli k attacchi dal raw NON presenti nel test, normalizzando il \r
    attacchi = []
    visti = 0
    with open(RAW, "rb") as f:
        next(f)
        for l in f:
            riga = l.rstrip(b"\r\n")
            c = riga.split(b",")
            if len(c) <= LABEL_COL or c[LABEL_COL] != b"1":
                continue
            visti += 1
            if chiave(c) in test_att:
                continue
            attacchi.append(riga)
            if len(attacchi) >= k:
                break
    if len(attacchi) < k:
        print(f"STOP: solo {len(attacchi):,} attacchi non-test disponibili (servono {k:,})")
        return 1
    print(f"attacchi raccolti (non-test): {len(attacchi):,} su {visti:,} attacchi visti nel raw")

    # posizioni degli attacchi nel file di output (random, sparse, non in fondo)
    tot = n_benign + k
    pos_att = set(int(x) for x in np.random.default_rng(SEED).choice(tot, size=k, replace=False))

    # scrittura: scorro le posizioni; benigni in streaming, attacchi da RAM
    it_att = iter(attacchi)
    n_att_scritti = n_ben_scritti = 0
    with open(TRAIN, "rb") as fb, open(OUT, "wb") as fo:
        next(fb)                                    # salta header del train
        fo.write(header + b"\n")
        for p in range(tot):
            if p in pos_att:
                fo.write(next(it_att) + b"\n")
                n_att_scritti += 1
            else:
                fo.write(fb.readline().rstrip(b"\r\n") + b"\n")
                n_ben_scritti += 1
    print(f"scritte {n_ben_scritti:,} benigne + {n_att_scritti:,} attacchi = {n_ben_scritti + n_att_scritti:,}")
    print(f"output: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
