#!/usr/bin/env python3
# TAG: ONE-SHOT | LAN contaminato = file_unito meno le righe del lan_test (no-leak)
"""Create a naturally contaminated LAN training CSV without test-row leakage.

Removes every full row matching the held-out test, including duplicate matches. Reports row
accounting, checks that suspicious traffic remains, and verifies a multiset checksum identity.
"""
import hashlib
import sys
from pathlib import Path

REPO = Path.home() / "tesi/repov6"
RAW = Path("<MEDIA>/CrucialP3Plus/cartelladilavoro/archivio_tesi/repo_v4/data/raw")
UNITO = RAW / "file_unito.csv"
SOSP = RAW / "file_sospetti.csv"
TEST = REPO / "inference/dati/inferenza/lan_test_raw.csv"
OUT = REPO / "inference/dati/addestramento/lan_contaminato.csv"
MOD = 1 << 128


def data_rows(path):
    """Yield data rows without CR/LF endings."""
    with open(path, "rb") as f:
        next(f)
        for line in f:
            yield line.rstrip(b"\r\n")


def _h(row):
    return int.from_bytes(hashlib.md5(row).digest(), "big")


def main():
    header = open(UNITO, "rb").readline().rstrip(b"\r\n")
    if header != open(TEST, "rb").readline().rstrip(b"\r\n"):
        print("STOP: header file_unito != header lan_test")
        return 1

    test_set = set(data_rows(TEST))
    sosp_set = set(data_rows(SOSP))
    n_sosp_rows = sum(1 for _ in data_rows(SOSP))
    print(f"lan_test: righe uniche da escludere = {len(test_set):,}")
    print(f"file_sospetti: righe dati = {n_sosp_rows:,} (uniche {len(sosp_set):,})")

    kept = removed = kept_susp = removed_susp = 0
    sum_out = sum_removed = 0
    with open(OUT, "wb") as fo:
        fo.write(header + b"\n")
        for row in data_rows(UNITO):
            susp = row in sosp_set
            if row in test_set:
                removed += 1
                sum_removed = (sum_removed + _h(row)) % MOD
                if susp:
                    removed_susp += 1
            else:
                fo.write(row + b"\n")
                kept += 1
                sum_out = (sum_out + _h(row)) % MOD
                if susp:
                    kept_susp += 1

    tot = kept + removed
    print(f"\nkept={kept:,}  removed={removed:,}  totale letto={tot:,}")
    print(f"sospetti tenuti={kept_susp:,}  sospetti rimossi={removed_susp:,}")
    print(f"prevalenza sospetto nell'output = {kept_susp / kept * 100:.4f}%  (n={kept_susp:,})")

    # ---- Verifica rileggendo da disco ----
    print("\n=== GATE ===")
    # accounting
    print(f"[accounting] kept+removed == righe file_unito : {tot} (atteso 3.279.654)")
    # no-leak: nessuna riga di output nel lan_test
    leak = sum(1 for r in data_rows(OUT) if r in test_set)
    print(f"[no-leak]    righe di output presenti nel lan_test : {leak}  -> {'OK' if leak == 0 else 'FALLITO'}")
    # contaminazione preservata: TUTTE le righe sospette presenti nell'output (nessuna rimossa)
    susp_in_out = sum(1 for r in data_rows(OUT) if r in sosp_set)
    contam_ok = (susp_in_out == n_sosp_rows and removed_susp == 0)
    print(f"[contam]     sospetti presenti nell'output : {susp_in_out:,} / {n_sosp_rows:,} righe "
          f"(rimossi {removed_susp})  -> {'OK' if contam_ok else 'FALLITO'}")
    # identita' omomorfa: sum(output riletto) + sum(rimosse) == sum(file_unito riletto)
    sum_out_re = 0
    for r in data_rows(OUT):
        sum_out_re = (sum_out_re + _h(r)) % MOD
    sum_unito = 0
    for r in data_rows(UNITO):
        sum_unito = (sum_unito + _h(r)) % MOD
    ok_id = (sum_out_re + sum_removed) % MOD == sum_unito
    print(f"[identita']  sum(output)+sum(rimosse) == sum(file_unito) : {'OK' if ok_id else 'FALLITO'}")
    print(f"             sum_output   = {sum_out_re:032x}")
    print(f"             sum_unito    = {sum_unito:032x}")

    print(f"\noutput: {OUT}  ({OUT.stat().st_size:,} byte, {kept:,} righe dati)")
    return 0 if (leak == 0 and contam_ok and ok_id and tot == 3_279_654) else 2


if __name__ == "__main__":
    sys.exit(main())
