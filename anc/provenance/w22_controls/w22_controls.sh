#!/usr/bin/env bash
# W2.2 / gate item 1: the three distinct controls.
#  (i)   D=64, randomization OFF (no rotation/permutation), inert padding
#        -- isolates the randomization layers
#  (ii)  full-width D=1024 naked -- isolates the bottleneck
#  (iii) exact-codeword injection inside the defended representation -- the
#        sensitivity control; this is the W2.4 injection mechanism run on a
#        defended bundle (deleg6040_gate_sensitivity on the defended cell),
#        not a separate training run.
set -uo pipefail
PKG=$HOME/dtraining-packaged
OUT=/workspace/experiments/results/training/w22_controls
IMAGE=split-inference:spark
MOUNTS=(-v "$HOME/experiments:/workspace/experiments" -v "$PKG:$PKG")
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
mkdir -p "$HOME/experiments/results/training/w22_controls/bundles"

# Controls (i) and (ii): runner-driven training cells. The rotation/permutation
# flags are store_true, so omitting them turns the randomization OFF for (i).
run_cell() {
  local cell=$1; shift
  local extra=("$@")
  if [ -f "$HOME/experiments/results/training/w22_controls/${cell}.json" ]; then
    log "$cell exists, skipping"; return 0
  fi
  log "control cell $cell"
  docker run --rm --gpus all --network host --ipc host \
    -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    "${MOUNTS[@]}" -w "$PKG" "$IMAGE" \
    python3 bin/run_latent_native_v5_06b.py \
      --model /workspace/experiments/models/qwen3-0.6b \
      --corpus /workspace/experiments/models/wikitext2_corpus.txt \
      --output "$OUT/${cell}.json" \
      --cloud-tls-ca /workspace/experiments/tls/ca.crt \
      --cloud-kind monomial_moe_radial --cloud-experts 8 --cloud-layers 2 \
      --noise-multiplier 0.35 --clip-norm 1.0 \
      --split-after 14 --resume-after 26 --seq-len 32 \
      --steps 10000 --warmup-steps 200 \
      --train-blocks 256 --eval-blocks 256 \
      --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
      --lr 3e-4 --remote-grad-clip 1.0 \
      --token-scale-sigma 0.75 --chaff-tokens 48 --seed 42 \
      --outbound-grad-dp off \
      --attacker-bundle "$OUT/bundles/${cell}_bundle.pt" \
      --grad-channel-bundle "$OUT/bundles/${cell}_gradchannel.pt" \
      --grad-channel-frames 512 \
      --cloud-url wss://poseidon.cluster:5025 --adversary-strength 1.0 --mine-penalty 0.1 \
      "${extra[@]}" \
    && log "$cell complete" || log "$cell FAILED"
}

# (i) randomization OFF: rotation/permutation omitted. D=64.
run_cell "ctrl_i_d64_norand" --latent-dim 64
# (ii) full-width naked: D=1024, rotation/permutation ON (the defended config,
#       but no bottleneck).
run_cell "ctrl_ii_d1024_naked" --latent-dim 1024 --secret-wire-rotation --secret-token-permutation

log "W2.2 CONTROLS COMPLETE"
