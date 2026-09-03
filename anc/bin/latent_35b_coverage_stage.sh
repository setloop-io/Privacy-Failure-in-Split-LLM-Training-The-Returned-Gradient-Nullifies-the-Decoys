#!/usr/bin/env bash
# 35B coverage stage: v10/v12/v13 headline cells on Qwen3.5-35B-A3B.
# Operating point: noise 0.40, chaff-48 (the hybrid-MoE margin config).
set -euo pipefail

MODEL=/workspace/experiments/models/qwen36-35b-a3b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
CA=/workspace/experiments/tls/ca.crt
mkdir -p "$BUNDLEDIR"

run_cell () {
  local name="$1"; shift
  local runner="$1"; shift
  local extra=("$@")
  echo "=== 35B cell $name ==="
  python3 "bin/$runner" \
    --model "$MODEL" --corpus "$CORPUS" \
    --output "$OUTDIR/latent_v13_35b_${name}.json" \
    --cloud-tls-ca "$CA" \
    --secret-wire-rotation --secret-token-permutation \
    --latent-dim 64 --noise-multiplier 0.40 --clip-norm 1.0 --seq-len 32 \
    --steps 2000 --warmup-steps 200 --train-blocks 256 --eval-blocks 256 \
    --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
    --lr 3e-4 --remote-grad-clip 1.0 \
    --token-scale-sigma 0.75 --chaff-tokens 48 --seed 42 \
    --attacker-bundle "$BUNDLEDIR/latent_v13_35b_${name}_bundle.pt" \
    "${extra[@]}" \
    || { echo "CELL $name FAILED"; return 1; }

  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/latent_v13_35b_${name}_bundle.pt" \
    --output "$OUTDIR/latent_v13_35b_${name}_attacker.json" || echo "PROBE FAILED"
  rm -f "$BUNDLEDIR/latent_v13_35b_${name}_bundle.pt"
  echo "=== 35B cell $name done ==="
}

# v10.0 two-segment: prefix 0-27 | A 28-31 | island 32-33 | B 34-37 | tail 38-39 (20% delegated)
run_cell v10_2seg run_latent_native_v10_2seg.py \
  --cloud-url-a wss://ucn:5025 --cloud-url-b wss://ucn:5027 \
  --cloud-kind monomial_moe_radial --cloud-experts 8 --cloud-layers 2 \
  --split-after-a 27 --resume-after-a 32 --split-after-b 33 --resume-after-b 38

# v12.0 invariant-MLP, split 31/36, K=1
run_cell v12_invmlp run_latent_native_v5_06b.py \
  --cloud-url wss://ucn:5030 --cloud-kind invariant_mlp --cloud-layers 2 \
  --split-after 31 --resume-after 36 --adversary-strength 1.0

# v13 trio
run_cell v130_mine run_latent_native_v5_06b.py \
  --cloud-url wss://ucn:5025 --cloud-kind monomial_moe_radial --cloud-experts 8 --cloud-layers 2 \
  --split-after 31 --resume-after 36 --adversary-strength 1.0 --mine-penalty 0.1

run_cell v131_mionly run_latent_native_v5_06b.py \
  --cloud-url wss://ucn:5025 --cloud-kind monomial_moe_radial --cloud-experts 8 --cloud-layers 2 \
  --split-after 31 --resume-after 36 --adversary-strength 0.0 --mine-penalty 0.1

run_cell v132_fragment run_latent_native_v5_06b.py \
  --cloud-urls wss://ucn:5031,wss://ucn:5032 \
  --cloud-kind monomial_moe_radial --cloud-experts 8 --cloud-layers 2 \
  --cloud-channels 2 --fragment-channels 2 \
  --split-after 31 --resume-after 36 --adversary-strength 1.0

echo "STAGE_35B_COVERAGE_COMPLETE"
