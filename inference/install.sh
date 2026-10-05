#!/usr/bin/env bash
# install.sh — setup della macchina target per il pacchetto NIDS a 2 stadi (repov6, cap4).
# Crea i due virtualenv co-locati nella cartella del pacchetto, ENTRAMBI Python 3.12:
#   infer_venv (Python 3.12, solo numpy)                 -> inferenza (infer.py)
#   train_venv (Python 3.12, TensorFlow + scikit-learn)  -> retrain locale (train.py)
# e scompatta i dataset oltre i 100 MB, distribuiti come archivi .7z in dati/.
# Sono separati perche' a runtime l'inferenza e' NumPy-pura (nessun framework pesante), mentre il
# training usa TensorFlow+sklearn. L'inferenza e' MONOPROCESSO a thread Python singolo: l'unico
# parallelismo e' quello interno del BLAS sulle matmul. Le versioni sono PINNATE nei requirements.
#
#   bash install.sh          INTERATTIVO: chiede s/n prima di OGNI venv e dei dataset (default)
#   bash install.sh infer    installa solo infer_venv, senza chiedere
#   bash install.sh train    installa solo train_venv, senza chiedere
#   bash install.sh dati     scompatta solo i dataset, senza chiedere
#   bash install.sh all      installa entrambi e scompatta i dataset, senza chiedere
# Output su console + install.log.
set -euo pipefail
MODE="${1:-ask}"
case "$MODE" in
    ask|infer|train|dati|all) ;;
    *) echo "ERRORE: modalita' '$MODE' non valida. Usa: bash install.sh [ask|infer|train|dati|all]"; exit 1 ;;
esac
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ask_yn() {   # $1 = domanda; default SI (INVIO); ritorna 0=si, 1=no
    local r
    read -r -p "$1 [S/n]: " r || true
    case "${r,,}" in n|no) return 1 ;; *) return 0 ;; esac
}

# Decide COSA installare PRIMA di avviare il log su file, cosi' i prompt vanno diretti al terminale.
do_infer=0; do_train=0; do_dati=0
case "$MODE" in
    infer) do_infer=1 ;;
    train) do_train=1 ;;
    dati)  do_dati=1 ;;
    all)   do_infer=1; do_train=1; do_dati=1 ;;
    ask)
        ask_yn "Installare infer_venv (inferenza, Python 3.12 + numpy)?" && do_infer=1
        ask_yn "Installare train_venv (training, TensorFlow + scikit-learn)?" && do_train=1
        ask_yn "Scompattare i dataset di dati/ (archivi .7z, circa 1,4 GB su disco)?" && do_dati=1
        ;;
esac
if [ "$do_infer" = 0 ] && [ "$do_train" = 0 ] && [ "$do_dati" = 0 ]; then
    echo "Niente da installare."
    exit 0
fi
# 7-Zip serve solo ai dataset: si controlla subito, prima di installare qualunque cosa.
if [ "$do_dati" = 1 ]; then
    Z="$(command -v 7z || command -v 7za || command -v 7zz || true)"
    [ -n "$Z" ] || { echo "ERRORE: per scompattare i dataset serve 7-Zip (Debian/Ubuntu: sudo apt install p7zip-full)"; exit 1; }
fi

exec > >(tee -a "$DIR/install.log") 2>&1
echo "==> install (infer=$do_infer train=$do_train dati=$do_dati) $(LC_ALL=C date -u +%Y-%m-%dT%H:%M:%SZ) in $DIR"
PY312="${PY312:-python3.12}"

# Dataset oltre i 100 MB, distribuiti come .7z accanto alla loro posizione (passo 3): ogni CSV
# scompattato si verifica contro lo sha256 di dati/PROVENIENZA.txt; cse_benigni.csv (righe con
# Label=0) si ricava da cse_test_raw.csv.
verifica() {   # $1 = file, $2 = sha256 atteso: 0 se coincide
    [ -f "$1" ] && [ "$(sha256sum "$1" | cut -c1-64)" = "$2" ]
}
scompatta() {  # $1 = archivio .7z, $2 = sha256 atteso del CSV
    # controlli espliciti: in una funzione chiamata con '|| return 1' bash sospende set -e
    local out="${1%.7z}"
    if verifica "$out" "$2"; then echo "==> [dati] $(basename "$out"): gia' presente, sha256 ok"; return 0; fi
    [ -f "$1" ] || { echo "ERRORE: [dati] manca l'archivio $(basename "$1")"; return 1; }
    if ! "$Z" e -so "$1" > "$out.tmp"; then
        rm -f "$out.tmp"; echo "ERRORE: [dati] 7z non riesce a scompattare $(basename "$1")"; return 1
    fi
    if ! verifica "$out.tmp" "$2"; then
        rm -f "$out.tmp"; echo "ERRORE: [dati] sha256 di $(basename "$out") diverso dall'atteso"; return 1
    fi
    mv "$out.tmp" "$out" || return 1
    echo "==> [dati] $(basename "$out"): scompattato, sha256 ok"
}
scompatta_dataset() {
    local I="$DIR/dati/inferenza" A="$DIR/dati/addestramento"
    scompatta "$I/cse_test_raw.csv.7z" 7ab7a0430894b9466efa2ac4d84cb7cb31954d2a8f1ed6f3877ccb041047d68a || return 1
    scompatta "$A/unsw_train_raw.csv.7z" fbf16e272046f7c2b0fee753d66e5c9b34dd345f5ca5a888f3e5d76c04a6308f || return 1
    scompatta "$A/unsw_train_contam1_raw.csv.7z" debe648f35c824aaad8d9fb7c1eb613c0ce7d24d84124905fee016ccfa86c15f || return 1
    local B="$I/cse_benigni.csv" SB=dc42312cbd8b7ce4114616b76611839243da1c91a61d666769b46bb08ab18787
    if verifica "$B" "$SB"; then echo "==> [dati] cse_benigni.csv: gia' presente, sha256 ok"; return 0; fi
    if ! awk -F',' 'NR==1 || $54=="0"' "$I/cse_test_raw.csv" > "$B.tmp" || ! verifica "$B.tmp" "$SB"; then
        rm -f "$B.tmp"; echo "ERRORE: [dati] cse_benigni.csv non ricavato o sha256 diverso dall'atteso"; return 1
    fi
    mv "$B.tmp" "$B" || return 1
    echo "==> [dati] cse_benigni.csv: ricavato da cse_test_raw.csv (Label=0), sha256 ok"
}

# 1. venv di INFERENZA (numpy-only, Python 3.12)
if [ "$do_infer" = 1 ]; then
    command -v "$PY312" >/dev/null || { echo "ERRORE: $PY312 non trovato"; exit 1; }
    [ -x "$DIR/infer_venv/bin/python" ] || "$PY312" -m venv "$DIR/infer_venv"
    "$DIR/infer_venv/bin/pip" install --quiet --upgrade pip
    "$DIR/infer_venv/bin/pip" install --quiet -r "$DIR/requirements.txt"
    echo "==> infer_venv: $("$DIR/infer_venv/bin/python" -c 'import sys,numpy;print("Python",sys.version.split()[0],"| numpy",numpy.__version__)')"
    sed -i "1s|.*|#!$DIR/infer_venv/bin/python|" "$DIR/infer.py"
    chmod +x "$DIR/infer.py"
fi

# 2. venv di TRAINING (TF+sklearn, Python 3.12)
if [ "$do_train" = 1 ]; then
    command -v "$PY312" >/dev/null || { echo "ERRORE: $PY312 non trovato"; exit 1; }
    [ -x "$DIR/train_venv/bin/python" ] || "$PY312" -m venv "$DIR/train_venv"
    "$DIR/train_venv/bin/pip" install --quiet --upgrade pip
    "$DIR/train_venv/bin/pip" install --quiet -r "$DIR/requirements_train.txt"
    echo "==> train_venv: $("$DIR/train_venv/bin/python" -c 'import sys;print("Python",sys.version.split()[0])')"
    sed -i "1s|.*|#!$DIR/train_venv/bin/python|" "$DIR/train.py"
    chmod +x "$DIR/train.py"
fi

# 3. dataset: scompattazione e verifica degli sha256, un archivio dopo l'altro
if [ "$do_dati" = 1 ]; then
    scompatta_dataset || { echo "ERRORE: scompattazione dei dataset non riuscita (v. sopra)"; exit 1; }
    echo "==> [dati] dataset pronti in dati/inferenza/ e dati/addestramento/"
fi

# 4. cartelle di lavoro della cattura nProbe
mkdir -p "$DIR/sniffing/csv" "$DIR/sniffing/staging" "$DIR/sniffing/log"

# 5. nProbe demo (opzionale): installa i .deb locali + setcap per catturare senza root
if ls "$DIR/nprobe_demo/"*.deb >/dev/null 2>&1; then
    echo "==> nProbe demo: dpkg -i + setcap (richiede sudo)"
    sudo dpkg -i "$DIR/nprobe_demo/"*.deb || sudo apt-get -f install -y
    NB="$(command -v nprobe || true)"
    [ -n "$NB" ] && sudo setcap cap_net_raw,cap_net_admin+eip "$NB"
else
    echo "==> nProbe demo assente: --live richiede nProbe installato a parte (l'inferenza su CSV/batch funziona comunque)"
fi

echo "==> FATTO."
echo "    menu:       python main.py    (interattivo, consigliato)"
echo "    inferenza:  ./infer.py --batch traffico.csv --mode A --model cse   (A=FPR minimo, B=alta recall)"
echo "    retrain:    ./train.py --dataset traffico_locale.csv --out-dir modelli/sperimentali/lan_seed42 --seed 42"
