#!/usr/bin/env bash
# W2.4 axis-2: dose-response across TRAINING SIZE, reusing the gradaudit
# budget bundles (2k ladder s13, 10k, 40k, 100k). Same metric-sweep driver.
set -uo pipefail
PKG=$HOME/dtraining-packaged
IMAGE=split-inference:spark
BASE=/workspace/experiments/results/training
MOUNTS=(-v "$HOME/experiments:/workspace/experiments" -v "$PKG:$PKG")

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

run_cell() { # cell steps_label
  local cell=$1
  local bundle="$BASE/gradaudit/bundles/${cell}_bundle.pt"
  [ -f "$HOME/experiments/results/training/gradaudit/bundles/${cell}_bundle.pt" ] || { log "no bundle $cell"; return 1; }
  log "axis-2 dose-response on $cell"
  docker run --rm --gpus all --network host --ipc host \
    -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -e BUNDLE="$bundle" -e CELL_SEED=42 \
    -e WORK="$BASE/w24_axis2_${cell}" -e OUT="$BASE/w24_axis2_${cell}" \
    "${MOUNTS[@]}" -w "$PKG" "$IMAGE" bash bin/w24_metric_sweep.sh \
    && log "axis-2 $cell complete" || log "axis-2 $cell FAILED"
}

for cell in gradaudit_ladder_s13_2k gradaudit_a2b_40k gradaudit_a2c_100k; do
  run_cell "$cell"
done
log "AXIS-2 QUEUE COMPLETE"
