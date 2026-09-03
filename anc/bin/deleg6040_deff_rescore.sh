#!/usr/bin/env bash
# Re-score retained 60/40 attacker bundles with per-row eval prediction dumping.
#
# The frozen nine-arm attacker is re-run unchanged; --dump-eval-predictions is
# a diagnostic side channel that writes each arm's per-row argmax and touches
# no scored quantity (attacker/attacks/latent_probe.py:37,201-203,232-237).
# Every re-score is checked against the cell's committed *_attacker.json by
# bin/deleg6040_design_effect.py before any dump is read.
#
# Runs on gx10-odysseus.nord out of an isolated tree (~/dtraining-deff) so no
# running job's checkout is touched. Bundles are read-only.
#
#   BUNDLES="a b c" bash bin/deleg6040_deff_rescore.sh
set -euo pipefail

TREE=${TREE:-$HOME/dtraining-deff}
OUT=${OUT:-$HOME/experiments/results/training/deleg6040/deff}
BUNDLEDIR=${BUNDLEDIR:-$HOME/experiments/results/training/deleg6040/bundles}
IMAGE=${IMAGE:-split-inference:spark}

mkdir -p "$OUT"

for cell in $BUNDLES; do
  bundle="$BUNDLEDIR/${cell}_bundle.pt"
  [ -f "$bundle" ] || { echo "missing bundle $bundle" >&2; exit 1; }
  if [ -f "$OUT/${cell}_pred.pt" ] && [ -f "$OUT/${cell}_rescore.json" ]; then
    echo "=== $cell: dump present, skipping ==="
    continue
  fi
  echo "=== $cell: re-score start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  docker run --rm --network host --ipc host \
    -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e CONTAINER_IMAGE="$IMAGE" \
    -v "$HOME/experiments:/workspace/experiments" \
    -v "$TREE:/workspace/dtraining-deff" -w /workspace/dtraining-deff \
    "$IMAGE" python3 -m attacker --attack latent-probe \
      --bundle "/workspace/experiments/results/training/deleg6040/bundles/${cell}_bundle.pt" \
      --output "/workspace/experiments/results/training/deleg6040/deff/${cell}_rescore.json" \
      --dump-eval-predictions \
        "/workspace/experiments/results/training/deleg6040/deff/${cell}_pred.pt" \
    2>&1 | tail -3
  echo "=== $cell: re-score done  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
done
echo "ALL_RESCORES_DONE"
