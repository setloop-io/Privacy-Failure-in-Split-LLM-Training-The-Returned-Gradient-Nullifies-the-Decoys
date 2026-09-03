#!/usr/bin/env bash
# Latent-v8 sweep 6: equivariant-MoE cloud, first measurements.
# 0.6B E=8 (privacy/utility/runtime vs v7 monomial), then 35B E=32
# (capacity question: cloud contribution vs v7's 0.40 zero-cloud penalty).
set -uo pipefail

CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
CA=/workspace/experiments/tls/ca.crt
mkdir -p "$BUNDLEDIR"

run_cell () {
  local name="$1"; shift
  local model="$1"; shift
  local split="$1"; shift
  local resume="$1"; shift
  local experts="$1"; shift
  local noise="$1"; shift
  local seed="$1"; shift
  local port="$1"

  echo "=== cell $name (model=$model split=$split/$resume E=$experts n=$noise seed=$seed port=$port) ==="
  python3 bin/run_latent_native_v5_06b.py \
    --model "$model" --corpus "$CORPUS" \
    --output "$OUTDIR/latent_v8_${name}.json" \
    --cloud-url wss://ucn:$port --cloud-tls-ca "$CA" \
    --cloud-kind monomial_moe --cloud-experts "$experts" \
    --secret-wire-rotation --secret-token-permutation --secret-token-gauge \
    --latent-dim 16 --noise-multiplier "$noise" --clip-norm 1.0 \
    --split-after "$split" --resume-after "$resume" --seq-len 32 \
    --steps 2000 --warmup-steps 200 --train-blocks 256 --eval-blocks 256 \
    --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
    --lr 3e-4 --adversary-strength 1.0 --remote-grad-clip 1.0 \
    --token-scale-sigma 0.75 --chaff-tokens 16 --seed "$seed" \
    --attacker-bundle "$BUNDLEDIR/latent_v8_${name}_bundle.pt" \
    || { echo "CELL $name FAILED"; return 1; }

  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/latent_v8_${name}_bundle.pt" \
    --output "$OUTDIR/latent_v8_${name}_attacker.json" \
    || echo "ATTACKER $name FAILED"
  rm -f "$BUNDLEDIR/latent_v8_${name}_bundle.pt"
  echo "=== cell $name done ==="
}

run_cell moe8_06b_s42   /workspace/experiments/models/qwen3-0.6b          21 26 8  0.35 42 5014
run_cell moe32_35b_s42  /workspace/experiments/models/qwen36-35b-a3b      31 36 32 0.40 42 5015

echo "SWEEP6 COMPLETE"
