#!/usr/bin/env bash
# Latent-v7 defense sweep round 1: knob variants, attacker unchanged.
# Runs on tln inside split-inference:spark; UCN server already up at ws://ucn:5013.
set -euo pipefail

MODEL=/workspace/experiments/models/qwen3-0.6b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
mkdir -p "$BUNDLEDIR"

run_cell () {
  local name="$1"; shift
  local noise="$1"; shift
  local sigma="$1"; shift
  local clip="$1"; shift
  local seed="${1:-42}"

  echo "=== cell $name (noise=$noise sigma=$sigma clip=$clip seed=$seed) ==="
  python3 bin/run_latent_native_v5_06b.py \
    --model "$MODEL" --corpus "$CORPUS" \
    --output "$OUTDIR/latent_v7_${name}.json" \
    --cloud-url ws://ucn:5013 --cloud-kind monomial \
    --secret-wire-rotation --secret-token-permutation --secret-token-gauge \
    --latent-dim 16 --noise-multiplier "$noise" --clip-norm "$clip" \
    --split-after 21 --resume-after 26 --seq-len 32 \
    --steps 2000 --warmup-steps 200 --train-blocks 256 --eval-blocks 256 \
    --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
    --lr 3e-4 --adversary-strength 1.0 --remote-grad-clip 1.0 \
    --token-scale-sigma "$sigma" --seed "$seed" \
    --attacker-bundle "$BUNDLEDIR/latent_v7_${name}_bundle.pt" \
    || { echo "CELL $name FAILED"; return 1; }

  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/latent_v7_${name}_bundle.pt" \
    --output "$OUTDIR/latent_v7_${name}_attacker.json" \
    || echo "ATTACKER $name FAILED"
  rm -f "$BUNDLEDIR/latent_v7_${name}_bundle.pt"
  echo "=== cell $name done ==="
}

run_cell base_n035    0.35 0.75 1.0 42
run_cell A_n045       0.45 0.75 1.0 42
run_cell B_n055       0.55 0.75 1.0 42
run_cell C_n045_s125  0.45 1.25 1.0 42
run_cell D_n045_c050  0.45 0.75 0.5 42

echo "SWEEP1 COMPLETE"
