#!/usr/bin/env bash
# Latent-v8 sweep 7: cloud DEPTH ladder on the 35B MoE (E=32 fixed).
# Question: does giving UCN more iterative message-passing steps per
# forward (5 -> 8 -> 11, from the default 2) compromise privacy or finally
# move the cloud contribution?  Operating point: noise 0.40, chaff 48.
set -uo pipefail

MODEL=/workspace/experiments/models/qwen36-35b-a3b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
CA=/workspace/experiments/tls/ca.crt
mkdir -p "$BUNDLEDIR"

run_cell () {
  local name="$1"; shift
  local layers="$1"; shift
  local port="$1"

  echo "=== cell $name (35B MoE E=32 L=$layers port=$port) ==="
  python3 bin/run_latent_native_v5_06b.py \
    --model "$MODEL" --corpus "$CORPUS" \
    --output "$OUTDIR/latent_v8_${name}.json" \
    --cloud-url wss://ucn:$port --cloud-tls-ca "$CA" \
    --cloud-kind monomial_moe --cloud-experts 32 --cloud-layers "$layers" \
    --secret-wire-rotation --secret-token-permutation --secret-token-gauge \
    --latent-dim 16 --noise-multiplier 0.40 --clip-norm 1.0 \
    --split-after 31 --resume-after 36 --seq-len 32 \
    --steps 2000 --warmup-steps 200 --train-blocks 256 --eval-blocks 256 \
    --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
    --lr 3e-4 --adversary-strength 1.0 --remote-grad-clip 1.0 \
    --token-scale-sigma 0.75 --chaff-tokens 48 --seed 42 \
    --attacker-bundle "$BUNDLEDIR/latent_v8_${name}_bundle.pt" \
    || { echo "CELL $name FAILED"; return 1; }

  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/latent_v8_${name}_bundle.pt" \
    --output "$OUTDIR/latent_v8_${name}_attacker.json" \
    || echo "ATTACKER $name FAILED"
  rm -f "$BUNDLEDIR/latent_v8_${name}_bundle.pt"
  echo "=== cell $name done ==="
}

run_cell moe32_L5_35b_s42  5  5016
run_cell moe32_L8_35b_s42  8  5017
run_cell moe32_L11_35b_s42 11 5018

echo "SWEEP7 COMPLETE"
