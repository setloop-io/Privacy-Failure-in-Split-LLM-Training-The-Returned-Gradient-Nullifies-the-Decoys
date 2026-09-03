#!/usr/bin/env bash
# Compromise battery: regenerate v9.2-winner bundles (canonical fraction 0.2
# for the gauge-compromise arm), score with the frozen latent_probe gate AND
# the new latent-sensitivity arms, then delete the bundles (hygiene).
set -euo pipefail

OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
CA=/workspace/experiments/tls/ca.crt
mkdir -p "$BUNDLEDIR"

run_sens_cell () {
  local name="$1"; shift
  local model="$1"; shift
  local split="$1"; shift
  local resume="$1"; shift
  local noise="$1"; shift
  local seed="$1"

  echo "=== sens cell $name (model=$(basename $model) seed=$seed) ==="
  python3 bin/run_latent_native_v5_06b.py \
    --model "$model" --corpus /workspace/experiments/models/wikitext2_corpus.txt \
    --output "$OUTDIR/latent_v9_${name}.json" \
    --cloud-url wss://ucn:5025 --cloud-tls-ca "$CA" \
    --cloud-kind monomial_moe_radial --cloud-experts 8 --cloud-layers 2 \
    --cloud-channels 2 \
    --secret-wire-rotation --secret-token-permutation \
    --latent-dim 64 --noise-multiplier "$noise" --clip-norm 1.0 \
    --split-after "$split" --resume-after "$resume" --seq-len 32 \
    --steps 2000 --warmup-steps 200 --train-blocks 256 --eval-blocks 256 \
    --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
    --lr 3e-4 --adversary-strength 1.0 --remote-grad-clip 1.0 \
    --token-scale-sigma 0.75 --chaff-tokens 48 --seed "$seed" \
    --bundle-canonical-fraction 0.2 \
    --attacker-bundle "$BUNDLEDIR/latent_v9_${name}_bundle.pt" \
    || { echo "CELL $name FAILED"; return 1; }

  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/latent_v9_${name}_bundle.pt" \
    --output "$OUTDIR/latent_v9_${name}_attacker.json" || echo "PROBE FAILED"
  python3 -m attacker --attack latent-sensitivity \
    --bundle "$BUNDLEDIR/latent_v9_${name}_bundle.pt" \
    --output "$OUTDIR/latent_v9_${name}_sensitivity.json" \
    || echo "SENSITIVITY FAILED"
  rm -f "$BUNDLEDIR/latent_v9_${name}_bundle.pt"
  echo "=== sens cell $name done ==="
}

run_sens_cell v92_s42 /workspace/experiments/models/qwen3-0.6b     21 26 0.35 42
run_sens_cell v92_s43 /workspace/experiments/models/qwen3-0.6b     21 26 0.35 43
run_sens_cell v92_s44 /workspace/experiments/models/qwen3-0.6b     21 26 0.35 44
run_sens_cell v92_35b_s42 /workspace/experiments/models/qwen36-35b-a3b 31 36 0.40 42

echo "SENS_BATTERY_COMPLETE"
