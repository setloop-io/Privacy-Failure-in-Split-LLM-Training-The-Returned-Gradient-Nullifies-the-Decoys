#!/usr/bin/env bash
# PR-101 D4: e2_seq64 / e2_seq128 against the PR author's
# reconstructed corpus (sha256-verified byte-identical) on this stack.
set -euo pipefail

MODEL=/workspace/experiments/models/qwen3-0.6b
CORPUS=/workspace/experiments/models/wikitext2_reconstructed.txt
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
CA=/workspace/experiments/tls/ca.crt
mkdir -p "$BUNDLEDIR"

run_cell () {
  local name="$1"; shift
  local seq="$1"
  echo "=== settle-it $name (seq=$seq, reconstructed corpus) ==="
  python3 bin/run_latent_native_v5_06b.py \
    --model "$MODEL" --corpus "$CORPUS" \
    --output "$OUTDIR/latent_pr101_${name}.json" \
    --cloud-url wss://ucn:5013 --cloud-tls-ca "$CA" \
    --cloud-kind monomial --secret-wire-rotation --secret-token-permutation \
    --secret-token-gauge \
    --latent-dim 16 --noise-multiplier 0.35 --clip-norm 1.0 \
    --split-after 21 --resume-after 26 --seq-len "$seq" \
    --steps 2000 --warmup-steps 200 --train-blocks 256 --eval-blocks 256 \
    --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
    --lr 3e-4 --adversary-strength 1.0 --remote-grad-clip 1.0 \
    --token-scale-sigma 0.75 --chaff-tokens 48 --seed 42 \
    --attacker-bundle "$BUNDLEDIR/latent_pr101_${name}_bundle.pt" \
    || { echo "CELL $name FAILED"; return 1; }
  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/latent_pr101_${name}_bundle.pt" \
    --output "$OUTDIR/latent_pr101_${name}_attacker.json" || echo "PROBE FAILED"
  rm -f "$BUNDLEDIR/latent_pr101_${name}_bundle.pt"
  echo "=== settle-it $name done ==="
}

run_cell e2_seq64_repro 64
run_cell e2_seq128_repro 128

echo "SETTLE_IT_COMPLETE"
