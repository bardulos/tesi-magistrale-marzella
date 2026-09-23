#!/usr/bin/env bash
# TAG: DRIVER-PIPELINE | 2026-07-13 | raccolta pesi dalle HOME dei VPS | vedi docs/refactoring_censimento.md
# tools/sync_weights_from_vps.sh — raccoglie in LOCALE i pesi sparsi nelle HOME dei 5 VPS.
#
# I worker scrivono i pesi nella HOME DUREVOLE del VPS (mai /tmp): la search e la compressione a k'
# in <remote_dir>/<trial_id>/seed_<s>/best.weights.h5, ciascun (trial,seme) su UN solo VPS (dove è
# girato). Questo script fa il merge-union dei 5 VPS in una sola dir LOCALE sotto runs/ — pronta per
# il LOAO (che legge i pesi da runs/, senza ricalcolo) e per il repo.
#
# Uso:
#   tools/sync_weights_from_vps.sh [remote_dir] [local_dest]
# Default:
#   remote_dir = $HOME/repov6_weights         (pesi search+compressione sui VPS)
#   local_dest = runs/dae/weights                 (destinazione locale, sotto runs/)
#
# Esempi:
#   tools/sync_weights_from_vps.sh                                   # search+compressione
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
  # -a: preserva struttura <trial_id>/seed_<s>/best.weights.h5; merge-union (no --delete).
  rsync -a --ignore-existing "$v:${REMOTE_DIR}/" "$DEST/" 2>&1 | tail -1 || echo "  (nessun peso o VPS irraggiungibile)"
done

N=$(find "$DEST" -name '*.weights.h5' | wc -l)
echo "totale pesi locali in $DEST: $N file"
