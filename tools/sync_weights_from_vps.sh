#!/usr/bin/env bash
# TAG: DRIVER-PIPELINE | raccolta pesi dalle HOME dei VPS
# Collect weights from the five VPS homes into one local runs/ directory for LOAO.
# Each trial/seed exists on one worker; rsync merges without deleting local files.
#
# Uso:
#   tools/sync_weights_from_vps.sh [remote_dir] [local_dest]
# Default:
#   remote_dir = $HOME/repov6_weights
#   local_dest = runs/dae/weights
#
# Esempi:
#   tools/sync_weights_from_vps.sh
#   tools/sync_weights_from_vps.sh $HOME/repov6_loao_weights runs/dae/loao_fase3/weights
set -euo pipefail

REMOTE_DIR="${1:-$HOME/repov6_weights}"
LOCAL_DEST="${2:-runs/dae/weights}"
VPS=(nodo1 nodo2 nodo3 nodo4 nodo5)

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/$LOCAL_DEST"
mkdir -p "$DEST"

echo "raccolta pesi: ${REMOTE_DIR} (5 VPS) -> ${DEST}"
for v in "${VPS[@]}"; do
  echo "--- $v ---"
  rsync -a --ignore-existing "$v:${REMOTE_DIR}/" "$DEST/" 2>&1 | tail -1 || echo "  (nessun peso o VPS irraggiungibile)"
done

N=$(find "$DEST" -name '*.weights.h5' | wc -l)
echo "totale pesi locali in $DEST: $N file"
