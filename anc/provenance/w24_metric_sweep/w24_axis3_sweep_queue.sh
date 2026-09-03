#!/usr/bin/env bash
# W2.4 axis-3 sweeps: dose-response on the seq-64 and seq-128 bundles,
# against the seq-32 axis-1 baseline. Waits for the training queue to drain.
set -uo pipefail
PKG=$HOME/dtraining-packaged
BASE=/workspace/experiments/results/training
IMAGE=split-inference:spark
MOUNTS=(-v "$HOME/experiments:/workspace/experiments" -v "$PKG:$PKG")
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

while pgrep -f w24_axis3_queue.sh >/dev/null 2>&1; do sleep 60; done
sleep 20

for sl in 64 128; do
  cell="w24_seqlen${sl}_40k"
  log "axis-3 dose-response on $cell (seq_len=$sl)"
  docker run --rm --gpus all --network host --ipc host \
    -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -e BUNDLE="$BASE/w24_axis3/bundles/${cell}_bundle.pt" -e CELL_SEED=42 \
    -e WORK="$BASE/w24_axis3_sweep_seqlen${sl}" -e OUT="$BASE/w24_axis3_sweep_seqlen${sl}" \
    "${MOUNTS[@]}" -w "$PKG" "$IMAGE" bash bin/w24_metric_sweep.sh \
    && log "axis-3 seqlen$sl complete" || log "axis-3 seqlen$sl FAILED"
done
log "AXIS-3 SWEEP COMPLETE"
