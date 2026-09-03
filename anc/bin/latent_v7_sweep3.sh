#!/usr/bin/env bash
# Latent-v7 sweep 3: int8 mechanism hypothesis test + multi-seed replication.
set -euo pipefail

MODEL=/workspace/experiments/models/qwen3-0.6b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
mkdir -p "$BUNDLEDIR"

run_cell () {
  local name="$1"; shift
  local noise="$1"; shift
  local chaff="$1"; shift
  local quant="$1"; shift
  local seed="${1:-42}"

  echo "=== cell $name (noise=$noise chaff=$chaff quant=$quant seed=$seed) ==="
  python3 bin/run_latent_native_v5_06b.py \
    --model "$MODEL" --corpus "$CORPUS" \
    --output "$OUTDIR/latent_v7_${name}.json" \
    --cloud-url ws://ucn:5013 --cloud-kind monomial \
    --secret-wire-rotation --secret-token-permutation --secret-token-gauge \
    --latent-dim 16 --noise-multiplier "$noise" --clip-norm 1.0 \
    --split-after 21 --resume-after 26 --seq-len 32 \
    --steps 2000 --warmup-steps 200 --train-blocks 256 --eval-blocks 256 \
    --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
    --lr 3e-4 --adversary-strength 1.0 --remote-grad-clip 1.0 \
    --token-scale-sigma 0.75 --chaff-tokens "$chaff" --wire-quant "$quant" \
    --seed "$seed" \
    --attacker-bundle "$BUNDLEDIR/latent_v7_${name}_bundle.pt" \
    || { echo "CELL $name FAILED"; return 1; }

  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/latent_v7_${name}_bundle.pt" \
    --output "$OUTDIR/latent_v7_${name}_attacker.json" \
    || echo "ATTACKER $name FAILED"
  rm -f "$BUNDLEDIR/latent_v7_${name}_bundle.pt"
  echo "=== cell $name done ==="
}

# Hypothesis test: fixed-grid int8 must NOT raise graph-arm recovery
run_cell I2_int8fixed_s42  0.35 0 int8 42
# Replicate the absmax anomaly on new seeds (old grid kept as int8row)
run_cell F2_int8row_s43    0.35 0 int8row 43
run_cell F3_int8row_s44    0.35 0 int8row 44
# Chaff multi-seed confirmation
run_cell E2_chaff16_s43    0.35 16 none 43
run_cell E3_chaff16_s44    0.35 16 none 44
# Best-combo candidate multi-seed
run_cell G2_chaff16_int8_s43 0.35 16 int8 43
run_cell G3_chaff16_int8_s44 0.35 16 int8 44
# Base control on new seeds (variance reference)
run_cell base_n035_s43     0.35 0 none 43
run_cell base_n035_s44     0.35 0 none 44

echo "SWEEP3 COMPLETE"
