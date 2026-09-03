#!/usr/bin/env bash
# Latent-v7.1 follow-up: boundary-push mechanisms on the 35B margin case.
# Operating point: noise 0.40 (v7.1 finding). Seed 42.
set -euo pipefail

MODEL=/workspace/experiments/models/qwen36-35b-a3b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
CA=/workspace/experiments/tls/ca.crt
mkdir -p "$BUNDLEDIR"

run_cell () {
  local name="$1"; shift
  local chaff="$1"; shift
  local gram="$1"; shift

  echo "=== cell $name (35B n0.40 chaff=$chaff gram=$gram) ==="
  python3 bin/run_latent_native_v5_06b.py \
    --model "$MODEL" --corpus "$CORPUS" \
    --output "$OUTDIR/latent_v71_${name}.json" \
    --cloud-url wss://ucn:5013 --cloud-tls-ca "$CA" \
    --cloud-kind monomial --secret-wire-rotation --secret-token-permutation \
    --secret-token-gauge \
    --latent-dim 16 --noise-multiplier 0.40 --clip-norm 1.0 \
    --split-after 31 --resume-after 36 --seq-len 32 \
    --steps 2000 --warmup-steps 200 --train-blocks 256 --eval-blocks 256 \
    --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
    --lr 3e-4 --adversary-strength 1.0 --remote-grad-clip 1.0 \
    --token-scale-sigma 0.75 --chaff-tokens "$chaff" --gram-flatten "$gram" \
    --seed 42 \
    --attacker-bundle "$BUNDLEDIR/latent_v71_${name}_bundle.pt" \
    || { echo "CELL $name FAILED"; return 1; }

  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/latent_v71_${name}_bundle.pt" \
    --output "$OUTDIR/latent_v71_${name}_attacker.json" \
    || echo "ATTACKER $name FAILED"
  rm -f "$BUNDLEDIR/latent_v71_${name}_bundle.pt"
  echo "=== cell $name done ==="
}

run_cell 35b_s42_n040_gram010 16 0.10
run_cell 35b_s42_n040_chaff48 48 0.0

echo "V71B COMPLETE"
