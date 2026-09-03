#!/usr/bin/env bash
# W1.7: E1 reproduction from packaged code, seeds 43 and 44.
# Trusted side: ~/dtraining-packaged (privacy-probe-transcript-leaks @ 65a7aa8).
# Cloud side:   poseidon latent-cloud, restarted from ~/dtraining-packaged.
set -uo pipefail
PKG=$HOME/dtraining-packaged
OUT=/workspace/experiments/results/training/e1_unprotected
IMAGE=split-inference:spark
MOUNTS=(-v "$HOME/experiments:/workspace/experiments" -v "$PKG:$PKG")

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

in_container() {
  docker run --rm --gpus all --network host --ipc host \
    -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    "${MOUNTS[@]}" -w "$PKG" "$IMAGE" bash -lc "$1"
}

for seed in 43 44; do
  cell="e1_repro_w12_s${seed}"
  if [ -f "$OUT/${cell}_arm_wire_real_paired.json" ] && [ -f "$HOME/experiments/results/training/e1_unprotected/${cell}_arm_wire_real_paired.json" ]; then
    log "$cell already complete, skipping"; continue
  fi
  log "training $cell (expect ~65 min)"
  in_container "SEED=$seed CELL=$cell bash bin/e1_unprotected_cell.sh" \
    && log "$cell training+scoring complete" || { log "$cell FAILED"; continue; }
  for arm in grad_real grad_real_shuffled wire_real; do
    d="$OUT/bundles/${cell}/${arm}_pred.pt"
    in_container "python3 bin/paired_advantage.py --dump $d --output $OUT/${cell}_arm_${arm}_paired.json" >/dev/null 2>&1 \
      && log "paired $cell $arm" || log "paired FAILED $cell $arm"
  done
done
log "QUEUE COMPLETE"
