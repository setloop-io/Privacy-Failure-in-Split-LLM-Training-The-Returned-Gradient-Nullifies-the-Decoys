#!/usr/bin/env bash
# W5.6 (PLAN.md E6): gradient-channel audit on the untested cells.
# Runs AFTER the W1.7 extra-seeds queue drains; refuses to start concurrent.
# Packaged tree both sides. Every cell: --outbound-grad-dp off, seed 42,
# arms grad_real grad_real_shuffled wire_real, --dump-eval-predictions.
set -uo pipefail
PKG=$HOME/dtraining-packaged
OUT=/workspace/experiments/results/training/gradaudit
IMAGE=split-inference:spark
MOUNTS=(-v "$HOME/experiments:/workspace/experiments" -v "$PKG:$PKG")

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

in_container() {
  docker run --rm --gpus all --network host --ipc host \
    -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    "${MOUNTS[@]}" -w "$PKG" "$IMAGE" bash -lc "$1"
}

run_cell() { # name split steps blocks frames dim
  local cell=$1 split=$2 steps=$3 blocks=$4 frames=$5 dim=$6
  if [ -f "$HOME/experiments/results/training/gradaudit/${cell}_arm_wire_real_paired.json" ]; then
    log "$cell already complete, skipping"; return 0
  fi
  log "cell $cell split=$split steps=$steps blocks=$blocks frames=$frames D=$dim"
  docker run --rm --gpus all --network host --ipc host \
    -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -e SPLIT_AFTER=$split -e STEPS=$steps -e TRAIN_BLOCKS=$blocks \
    -e EVAL_BLOCKS=$blocks -e FRAMES=$frames -e LATENT_DIM=$dim \
    -e CELL=$cell -e GRAD_DP=off \
    -e "ARMS=grad_real grad_real_shuffled wire_real" \
    "${MOUNTS[@]}" -w "$PKG" "$IMAGE" bash bin/e6_gradaudit_cell.sh \
    && log "$cell training+scoring complete" || { log "$cell FAILED"; return 1; }
  for arm in grad_real grad_real_shuffled wire_real; do
    in_container "python3 bin/paired_advantage.py --dump $OUT/bundles/$cell/${arm}_pred.pt --output $OUT/${cell}_arm_${arm}_paired.json" >/dev/null 2>&1 \
      && log "paired $cell $arm" || log "paired FAILED $cell $arm"
  done
}

# Stage E6-2k: ladder rungs (2k steps, 256 blocks, 512 frames, D=64)
for split in 13 17 19; do
  run_cell "gradaudit_ladder_s${split}_2k" $split 2000 256 512 64
done
# Stage E6-D: latent-width arms (10k steps, 256 blocks, 512 frames)
run_cell "gradaudit_d96_10k"  14 10000 256 512 96
run_cell "gradaudit_d128_10k" 14 10000 256 512 128
# Stage E6-40k: matches a2b exactly (40k, 4096 blocks, 4096 frames, D=64)
run_cell "gradaudit_a2b_40k"  14 40000 4096 4096 64
# Stage E6-100k: the 10x-exposure cell (100k, 4096 blocks, 4096 frames, D=64)
run_cell "gradaudit_a2c_100k" 14 100000 4096 4096 64

log "W5.6 QUEUE COMPLETE"
