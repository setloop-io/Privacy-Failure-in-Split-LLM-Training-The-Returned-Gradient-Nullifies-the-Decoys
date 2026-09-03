#!/usr/bin/env bash
# Latent-v7 sweep 5 (boundary push, 0.6B, seed 42, wss):
# chaff dose-response and the gram-flattening regularizer.
set -euo pipefail

MODEL=/workspace/experiments/models/qwen3-0.6b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
CA=/workspace/experiments/tls/ca.crt
mkdir -p "$BUNDLEDIR"

run_cell () {
  local name="$1"; shift
  local chaff="$1"; shift
  local gram="$1"; shift

  echo "=== cell $name (chaff=$chaff gram_flatten=$gram) ==="
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
    --token-scale-sigma 0.75 --chaff-tokens "$chaff" --gram-flatten "$gram" \
    --seed 42 \
    --attacker-bundle "$BUNDLEDIR/latent_v7_${name}_bundle.pt" \
    || { echo "CELL $name FAILED"; return 1; }

  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/latent_v7_${name}_bundle.pt" \
    --output "$OUTDIR/latent_v7_${name}_attacker.json" \
    || echo "ATTACKER $name FAILED"
  rm -f "$BUNDLEDIR/latent_v7_${name}_bundle.pt"
  echo "=== cell $name done ==="
}

run_cell P1_chaff32      32 0.0
run_cell P2_chaff48      48 0.0
run_cell P3_gram005      16 0.05
run_cell P4_gram020      16 0.20

echo "SWEEP5 COMPLETE"
