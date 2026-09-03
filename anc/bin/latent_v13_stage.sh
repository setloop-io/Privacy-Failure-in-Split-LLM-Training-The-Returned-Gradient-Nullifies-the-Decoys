#!/usr/bin/env bash
# v13 stage: the v13 cells on the v9.2 winner base
# (D=64, chaff-48, radial MoE E=8, scale gauge off, noise 0.35, wss).
#   v13.0 (A9): + MINE MI penalty beta=0.1 (K=1)
#   v13.1 (A2): MI-penalty-only objective (adversary strength 0) (K=1)
#   v13.2 (A1): PrivDFS-style fragmentation across two D=32 servers (K=2)
set -uo pipefail

MODEL=/workspace/experiments/models/qwen3-0.6b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
CA=/workspace/experiments/tls/ca.crt
mkdir -p "$BUNDLEDIR"

run_cell () {
  local name="$1"; shift
  local extra=("$@")
  echo "=== v13 cell $name ==="
  python3 bin/run_latent_native_v5_06b.py \
    --model "$MODEL" --corpus "$CORPUS" \
    --output "$OUTDIR/latent_v13_${name}.json" \
    --cloud-tls-ca "$CA" \
    --cloud-kind monomial_moe_radial --cloud-experts 8 --cloud-layers 2 \
    --secret-wire-rotation --secret-token-permutation \
    --latent-dim 64 --noise-multiplier 0.35 --clip-norm 1.0 \
    --split-after 21 --resume-after 26 --seq-len 32 \
    --steps 2000 --warmup-steps 200 --train-blocks 256 --eval-blocks 256 \
    --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
    --lr 3e-4 --remote-grad-clip 1.0 \
    --token-scale-sigma 0.75 --chaff-tokens 48 --seed 42 \
    --attacker-bundle "$BUNDLEDIR/latent_v13_${name}_bundle.pt" \
    "${extra[@]}" \
    || { echo "CELL $name FAILED"; return 1; }

  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/latent_v13_${name}_bundle.pt" \
    --output "$OUTDIR/latent_v13_${name}_attacker.json" || echo "PROBE FAILED"
  rm -f "$BUNDLEDIR/latent_v13_${name}_bundle.pt"
  echo "=== v13 cell $name done ==="
}

run_cell a9_mine010        --cloud-url wss://ucn:5025 --adversary-strength 1.0 --mine-penalty 0.1
run_cell a2_mionly010      --cloud-url wss://ucn:5025 --adversary-strength 0.0 --mine-penalty 0.1
run_cell a1_fragment2      --cloud-urls wss://ucn:5031,wss://ucn:5032 --adversary-strength 1.0 --cloud-channels 2 --fragment-channels 2

echo "V13_COMPLETE"
