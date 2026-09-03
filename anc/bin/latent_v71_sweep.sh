#!/usr/bin/env bash
# Latent-v7.1: 35B-A3B privacy-margin replication (seeds 43/44 at n0.35)
# plus noise-0.40 countermeasure cell (seed 42). Final v7 config over wss.
set -euo pipefail

MODEL=/workspace/experiments/models/qwen36-35b-a3b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
CA=/workspace/experiments/tls/ca.crt
mkdir -p "$BUNDLEDIR"

run_cell () {
  local name="$1"; shift
  local noise="$1"; shift
  local seed="$1"

  echo "=== cell $name (noise=$noise seed=$seed, 35B wss) ==="
  python3 bin/run_latent_native_v5_06b.py \
    --model "$MODEL" --corpus "$CORPUS" \
    --output "$OUTDIR/latent_v71_${name}.json" \
    --cloud-url wss://ucn:5013 --cloud-tls-ca "$CA" \
    --cloud-kind monomial --secret-wire-rotation --secret-token-permutation \
    --secret-token-gauge \
    --latent-dim 16 --noise-multiplier "$noise" --clip-norm 1.0 \
    --split-after 31 --resume-after 36 --seq-len 32 \
    --steps 2000 --warmup-steps 200 --train-blocks 256 --eval-blocks 256 \
    --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
    --lr 3e-4 --adversary-strength 1.0 --remote-grad-clip 1.0 \
    --token-scale-sigma 0.75 --chaff-tokens 16 --seed "$seed" \
    --attacker-bundle "$BUNDLEDIR/latent_v71_${name}_bundle.pt" \
    || { echo "CELL $name FAILED"; return 1; }

  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/latent_v71_${name}_bundle.pt" \
    --output "$OUTDIR/latent_v71_${name}_attacker.json" \
    || echo "ATTACKER $name FAILED"
  rm -f "$BUNDLEDIR/latent_v71_${name}_bundle.pt"
  echo "=== cell $name done ==="
}

run_cell 35b_s43_n035 0.35 43
run_cell 35b_s44_n035 0.35 44
run_cell 35b_s42_n040 0.40 42

echo "V71 COMPLETE"
