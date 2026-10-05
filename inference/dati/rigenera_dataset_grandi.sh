#!/usr/bin/env bash
# Rigenera o verifica i dataset CSE/UNSW non distribuiti; ogni output e' controllato con sha256.
# I CSV della cattura LAN privata non sono distribuiti e non si rigenerano.
# Uso dalla radice del repository:
#   bash inference/dati/rigenera_dataset_grandi.sh [cse_test|cse_benigni|unsw_train|unsw_contam|tutto]
# Prerequisiti: pandas/pyarrow (o PY=/percorso/python); cse_test richiede gli artefatti CSE.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATI="$ROOT/inference/dati"
PY="${PY:-python}"
PRE="${PREPROC_DIR:-$ROOT/runs/preprocessing}"

# sha256 attesi (PROVENIENZA.txt)
SHA_CSE_TEST=7ab7a0430894b9466efa2ac4d84cb7cb31954d2a8f1ed6f3877ccb041047d68a
SHA_CSE_BENIGNI=dc42312cbd8b7ce4114616b76611839243da1c91a61d666769b46bb08ab18787
SHA_UNSW_TRAIN=fbf16e272046f7c2b0fee753d66e5c9b34dd345f5ca5a888f3e5d76c04a6308f
SHA_UNSW_CONTAM=debe648f35c824aaad8d9fb7c1eb613c0ce7d24d84124905fee016ccfa86c15f

verifica() {  # verifica <file> <sha_atteso>
  local got; got="$(sha256sum "$1" | cut -c1-64)"
  if [ "$got" = "$2" ]; then echo "OK  sha256 $1"; else echo "ERRORE sha256 $1: atteso $2, ottenuto $got" >&2; exit 2; fi
}

cse_test() {
  # Ricostruisce il test CSE nProbe dallo split cluster-aware (seed 42).
  for f in ingest/cse_raw.parquet reduce_cse/cse_post_audit.parquet reduce_cse/feature_types_post_audit.json \
           transform_cse/transform_meta.json transform_cse/scaler_params.json transform_cse/test.npz transform_cse/test_labels.npz; do
    [ -e "$PRE/$f" ] || { echo "manca $PRE/$f: eseguire prima gli stage del preprocessing CSE (ingest, encode, audit, reduce, transform)" >&2; exit 3; }
  done
  "$PY" "$ROOT/tools/make_cse_test_raw.py" --raw "$PRE/ingest/cse_raw.parquet" \
      --post-audit "$PRE/reduce_cse/cse_post_audit.parquet" --types "$PRE/reduce_cse/feature_types_post_audit.json" \
      --transform-meta "$PRE/transform_cse/transform_meta.json" --scaler "$PRE/transform_cse/scaler_params.json" \
      --test-npz "$PRE/transform_cse/test.npz" --test-labels "$PRE/transform_cse/test_labels.npz" \
      --template-csv "$DATI/inferenza/unsw_test_raw.csv" --out "$DATI/inferenza/cse_test_raw.csv" --seed 42
  verifica "$DATI/inferenza/cse_test_raw.csv" "$SHA_CSE_TEST"
}

cse_benigni() {
  # Estrae le righe benigne del test CSE, mantenendo l'intestazione.
  [ -e "$DATI/inferenza/cse_test_raw.csv" ] || { echo "manca cse_test_raw.csv: rigenerarlo prima (cse_test)" >&2; exit 3; }
  awk -F',' 'NR==1 || $54=="0"' "$DATI/inferenza/cse_test_raw.csv" > "$DATI/inferenza/cse_benigni.csv"
  verifica "$DATI/inferenza/cse_benigni.csv" "$SHA_CSE_BENIGNI"
}

unsw_train() {
  # Questo CSV proviene dall'archivio del repository precedente e non e' rigenerabile qui.
  if [ -e "$DATI/addestramento/unsw_train_raw.csv" ]; then
    verifica "$DATI/addestramento/unsw_train_raw.csv" "$SHA_UNSW_TRAIN"
  else
    echo "unsw_train_raw.csv non presente e non rigenerabile con i tool del repository:" >&2
    echo "  ottenerlo dall'archivio (sha256 atteso $SHA_UNSW_TRAIN) o ricostruire lo split UNSW con preprocessing.py" >&2
    exit 4
  fi
}

unsw_contam() {
  # Richiede il train UNSW, il dataset pubblico e il test UNSW (no-leak).
  [ -e "$DATI/addestramento/unsw_train_raw.csv" ] || { echo "manca unsw_train_raw.csv (v. unsw_train)" >&2; exit 3; }
  [ -e "$ROOT/data/raw/NF-UNSW-NB15-v3.csv" ] || { echo "manca data/raw/NF-UNSW-NB15-v3.csv (dataset pubblico UQ)" >&2; exit 3; }
  "$PY" "$ROOT/tools/make_unsw_contaminated.py"
  verifica "$DATI/addestramento/unsw_train_contam1_raw.csv" "$SHA_UNSW_CONTAM"
}

case "${1:-tutto}" in
  cse_test) cse_test ;;
  cse_benigni) cse_benigni ;;
  unsw_train) unsw_train ;;
  unsw_contam) unsw_contam ;;
  tutto) cse_test; cse_benigni; unsw_train; unsw_contam ;;
  *) echo "uso: $0 [cse_test|cse_benigni|unsw_train|unsw_contam|tutto]" >&2; exit 1 ;;
esac
