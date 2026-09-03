#!/usr/bin/env bash
# W5.6 follow-up: D=96/128 at 4096 frames (the "wide CI at 512" caveat).
set -uo pipefail
PKG=$HOME/dtraining-packaged
OUT=/workspace/experiments/results/training/gradaudit
IMAGE=split-inference:spark
MOUNTS=(-v "$HOME/experiments:/workspace/experiments" -v "$PKG:$PKG")
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

run_d() {
  local dim=$1 port=$2
  local cell="gradaudit_d${dim}_10k_4k"
  if [ -f "$HOME/experiments/results/training/gradaudit/${cell}_arm_wire_real_paired.json" ]; then
    log "$cell already complete, skipping"; return 0
  fi
  log "cell $cell split=14 steps=10000 blocks=256 frames=4096 D=$dim port=$port"
  docker run --rm --gpus all --network host --ipc host \
    -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -e SPLIT_AFTER=14 -e STEPS=10000 -e TRAIN_BLOCKS=256 -e EVAL_BLOCKS=256 \
    -e FRAMES=4096 -e LATENT_DIM=$dim -e CELL=$cell -e GRAD_DP=off \
    -e "ARMS=grad_real grad_real_shuffled wire_real" \
    -e CLOUD_URL=wss://poseidon.cluster:$port \
    "${MOUNTS[@]}" -w "$PKG" "$IMAGE" bash bin/e6_gradaudit_cell.sh \
    && log "$cell training+scoring complete" || { log "$cell FAILED"; return 1; }
  for arm in grad_real grad_real_shuffled wire_real; do
    docker run --rm --gpus all --network host --ipc host \
      -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
      "${MOUNTS[@]}" -w "$PKG" "$IMAGE" \
      python3 bin/paired_advantage.py --dump $OUT/bundles/$cell/${arm}_pred.pt \
        --output $OUT/${cell}_arm_${arm}_paired.json >/dev/null 2>&1 \
      && log "paired $cell $arm" || log "paired FAILED $cell $arm"
  done
}

run_d 96 5296
run_d 128 5328
log "D-DEEP QUEUE COMPLETE"
