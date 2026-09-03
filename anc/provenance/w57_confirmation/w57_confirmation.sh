#!/usr/bin/env bash
# W5.7: post-freeze confirmatory E1 replication. Three fresh seeds (48/49/50),
# GRAD_DP=off, captured after freeze commit 222c4a7, scored against the frozen
# thresholds. These are the confirmatory numbers W6.1 may cite.
set -uo pipefail
PKG=$HOME/dtraining-packaged
OUT=/workspace/experiments/results/training/e1_confirmation
IMAGE=split-inference:spark
MOUNTS=(-v "$HOME/experiments:/workspace/experiments" -v "$PKG:$PKG")
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

for seed in 48 49 50; do
  cell="e1_confirm_s${seed}"
  if [ -f "$HOME/experiments/results/training/e1_confirmation/${cell}_arm_wire_real_paired.json" ]; then
    log "$cell exists, skipping"; continue
  fi
  log "training $cell seed=$seed"
  docker run --rm --gpus all --network host --ipc host \
    -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -e SEED=$seed -e CELL=$cell -e GRAD_DP=off \
    -e OUTDIR="$OUT" \
    "${MOUNTS[@]}" -w "$PKG" "$IMAGE" bash bin/e6_gradaudit_cell.sh \
    && log "$cell training+scoring complete" || { log "$cell FAILED"; continue; }
  for arm in grad_real grad_real_shuffled wire_real; do
    docker run --rm --gpus all --network host --ipc host \
      -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
      "${MOUNTS[@]}" -w "$PKG" "$IMAGE" \
      python3 bin/paired_advantage.py --dump $OUT/bundles/$cell/${arm}_pred.pt \
        --output $OUT/${cell}_arm_${arm}_paired.json >/dev/null 2>&1 \
      && log "paired $cell $arm" || log "paired FAILED $cell $arm"
  done
done
log "W5.7 CONFIRMATION COMPLETE"
