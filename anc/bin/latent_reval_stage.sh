#!/usr/bin/env bash
# v13.x verdict re-evaluation: rerun the failed cells under the v13 stack
# (v9.2 winner + MINE penalty beta=0.1) on the 0.6B, frozen gate throughout.
set -euo pipefail

MODEL=/workspace/experiments/models/qwen3-0.6b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
OUTDIR=/workspace/experiments/results/training
BUNDLEDIR=$OUTDIR/bundles
CA=/workspace/experiments/tls/ca.crt
mkdir -p "$BUNDLEDIR"

run_cell () {
  local name="$1"; shift
  local seq="$1"; shift
  local noise="$1"; shift
  local public_steps="$1"; shift
  local priv_steps="$1"; shift
  local kind="$1"; shift
  local url="$1"; shift
  local extra=("$@")

  local pub_args=()
  if [ "$public_steps" != "0" ]; then
    pub_args=(--public-corpus /workspace/experiments/models/wikitext2_public.txt
              --public-steps "$public_steps")
  fi

  echo "=== re-eval $name (seq=$seq n=$noise pub=$public_steps priv=$priv_steps kind=$kind) ==="
  python3 bin/run_latent_native_v5_06b.py \
    --model "$MODEL" --corpus "$CORPUS" \
    --output "$OUTDIR/latent_reval_${name}.json" \
    --cloud-url "$url" --cloud-tls-ca "$CA" \
    --cloud-kind "$kind" --cloud-experts 8 --cloud-layers 2 \
    --secret-wire-rotation --secret-token-permutation \
    --latent-dim 64 --noise-multiplier "$noise" --clip-norm 1.0 \
    --split-after 21 --resume-after 26 --seq-len "$seq" \
    --steps "$priv_steps" --warmup-steps 200 --train-blocks 256 --eval-blocks 256 \
    --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
    --lr 3e-4 --remote-grad-clip 1.0 \
    --token-scale-sigma 0.75 --chaff-tokens 48 --seed 42 \
    --attacker-bundle "$BUNDLEDIR/latent_reval_${name}_bundle.pt" \
    "${pub_args[@]}" "${extra[@]}"
  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/latent_reval_${name}_bundle.pt" \
    --output "$OUTDIR/latent_reval_${name}_attacker.json"
  rm -f "$BUNDLEDIR/latent_reval_${name}_bundle.pt"
  echo "=== re-eval $name done ==="
}

RAD=(--cloud-kind monomial_moe_radial --adversary-strength 1.0)

# E2 failures re-evaluated WITH MINE penalty (fragment port for d64 radial is 5025)
run_cell e2_seq64_mine  64  0.35 0 2000 monomial_moe_radial wss://ucn:5025 --adversary-strength 1.0 --mine-penalty 0.1
run_cell e2_seq128_mine 128 0.35 0 2000 monomial_moe_radial wss://ucn:5025 --adversary-strength 1.0 --mine-penalty 0.1
# E5 original failed recipe WITH MINE penalty
run_cell e5_pub4k_priv1k_mine 32 0.35 4000 1000 monomial_moe_radial wss://ucn:5025 --adversary-strength 1.0 --mine-penalty 0.1
# noise tripwire 0.30 with the current composite
run_cell n030_tripwire 32 0.30 0 2000 monomial_moe_radial wss://ucn:5025 --adversary-strength 1.0
# v9.2 base with MINE penalty (control delta) = v13.0 rerun for reference
run_cell v92_mine_control 32 0.35 0 2000 monomial_moe_radial wss://ucn:5025 --adversary-strength 1.0 --mine-penalty 0.1

echo "REVAL_COMPLETE"
