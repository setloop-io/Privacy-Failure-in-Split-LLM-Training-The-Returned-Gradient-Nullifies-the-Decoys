#!/usr/bin/env bash
# W3.4: complete-transcript capture for a forward-gate-passing config, 3 seeds,
# with trusted-side checkpoints on (W3.3) so W3.5 can verify and index.
set -uo pipefail
PKG=$HOME/dtraining-packaged
OUT=/workspace/experiments/results/training/w34_capture
IMAGE=split-inference:spark
MOUNTS=(-v "$HOME/experiments:/workspace/experiments" -v "$PKG:$PKG")
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

for seed in 42 43 44; do
  cell="w34_complete_s${seed}"
  if [ -f "$HOME/experiments/results/training/w34_capture/${cell}/trusted_boundary_usage.jsonl" ]; then
    log "$cell exists, skipping"; continue
  fi
  log "capture $cell seed=$seed"
  docker run --rm --gpus all --network host --ipc host \
    -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    "${MOUNTS[@]}" -w "$PKG" "$IMAGE" \
    python3 bin/run_latent_native_v5_06b.py \
      --model /workspace/experiments/models/qwen3-0.6b \
      --corpus /workspace/experiments/models/wikitext2_corpus.txt \
      --output "$OUT/${cell}.json" \
      --cloud-tls-ca /workspace/experiments/tls/ca.crt \
      --cloud-kind monomial_moe_radial --cloud-experts 8 --cloud-layers 2 \
      --secret-wire-rotation --secret-token-permutation \
      --latent-dim 64 --noise-multiplier 0.35 --clip-norm 1.0 \
      --split-after 14 --resume-after 26 --seq-len 32 \
      --steps 10000 --warmup-steps 200 \
      --train-blocks 256 --eval-blocks 256 \
      --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
      --lr 3e-4 --remote-grad-clip 1.0 \
      --token-scale-sigma 0.75 --chaff-tokens 48 --seed "$seed" \
      --outbound-grad-dp off \
      --attacker-bundle "$OUT/bundles/${cell}_bundle.pt" \
      --grad-channel-bundle "$OUT/bundles/${cell}_gradchannel.pt" \
      --grad-channel-frames 512 \
      --trusted-checkpoint "$OUT/${cell}" \
      --cloud-url wss://poseidon.cluster:5025 --adversary-strength 1.0 --mine-penalty 0.1 \
    && log "$cell complete" || log "$cell FAILED"
done
log "W3.4 CAPTURE COMPLETE"
