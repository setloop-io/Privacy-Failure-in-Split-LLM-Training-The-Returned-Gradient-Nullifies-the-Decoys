#!/usr/bin/env bash
# W2.4 axis-3: frame-length training runs (seq-len 64, 128) then dose-response
# sweeps on each, against the seq-len-32 baseline (axis-1 on s44).
set -uo pipefail
PKG=$HOME/dtraining-packaged
OUT=/workspace/experiments/results/training/w24_axis3
IMAGE=split-inference:spark
MOUNTS=(-v "$HOME/experiments:/workspace/experiments" -v "$PKG:$PKG")
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# Phase 1: train the frame-length cells (40k, the budget where the leak resolves)
for sl in 64 128; do
  cell="w24_seqlen${sl}_40k"
  if [ -f "$HOME/experiments/results/training/w24_axis3/bundles/${cell}_bundle.pt" ]; then
    log "$cell bundle exists, skipping training"; continue
  fi
  log "training $cell seq_len=$sl steps=40000"
  docker run --rm --gpus all --network host --ipc host \
    -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -e SPLIT_AFTER=14 -e STEPS=40000 -e TRAIN_BLOCKS=4096 -e EVAL_BLOCKS=4096 \
    -e FRAMES=4096 -e LATENT_DIM=64 -e SEQ_LEN=$sl -e CELL=$cell -e GRAD_DP=off \
    -e "ARMS=grad_real grad_real_shuffled wire_real" \
    -e OUTDIR=/workspace/experiments/results/training/w24_axis3 \
    "${MOUNTS[@]}" -w "$PKG" "$IMAGE" bash bin/e6_gradaudit_cell.sh \
    && log "$cell training complete" || log "$cell FAILED"
done
log "AXIS-3 TRAINING COMPLETE"
