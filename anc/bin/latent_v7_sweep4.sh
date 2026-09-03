#!/usr/bin/env bash
# Latent-v7 sweep 4: final v7 configuration over encrypted wss transport.
# Config: split-21, D=16, monomial cloud, all gauges (v2 key stream),
# chaff-16, noise 0.35, clip 1.0 — seeds 43/44 (seed 42 = cell L).
set -euo pipefail

MODEL=/workspace/experiments/models/qwen3-0.6b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
CA=/workspace/experiments/tls/ca.crt
mkdir -p "$BUNDLEDIR"

run_cell () {
  local name="$1"; shift
  local seed="$1"

  echo "=== cell $name (seed=$seed, wss) ==="
  python3 bin/run_latent_native_v5_06b.py \
    --model "$MODEL" --corpus "$CORPUS" \
    --output "$OUTDIR/latent_v7_${name}.json" \
    --cloud-url wss://ucn:5013 --cloud-tls-ca "$CA" \
    --cloud-kind monomial --secret-wire-rotation --secret-token-permutation \
    --secret-token-gauge \
    --latent-dim 16 --noise-multiplier 0.35 --clip-norm 1.0 \
    --split-after 21 --resume-after 26 --seq-len 32 \
    --steps 2000 --warmup-steps 200 --train-blocks 256 --eval-blocks 256 \
    --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
    --lr 3e-4 --adversary-strength 1.0 --remote-grad-clip 1.0 \
    --token-scale-sigma 0.75 --chaff-tokens 16 --seed "$seed" \
    --attacker-bundle "$BUNDLEDIR/latent_v7_${name}_bundle.pt" \
    || { echo "CELL $name FAILED"; return 1; }

  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/latent_v7_${name}_bundle.pt" \
    --output "$OUTDIR/latent_v7_${name}_attacker.json" \
    || echo "ATTACKER $name FAILED"
  rm -f "$BUNDLEDIR/latent_v7_${name}_bundle.pt"
  echo "=== cell $name done ==="
}

run_cell L_wss_chaff16_s43 43
run_cell L_wss_chaff16_s44 44

echo "SWEEP4 COMPLETE"
