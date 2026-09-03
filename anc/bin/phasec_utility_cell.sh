#!/usr/bin/env bash
# Phase C1 (2026-08-27): does the backward gradient channel leak on a
# configuration that passes BOTH gates?
#
# The published E1 leak cell (a2b) fails its own utility gate (+0.9137 vs
# 0.35). This driver runs the utility-passing topology (split_after=21,
# resume_after=26 -- the v9.2-class 4-layer delegation) with the forward AND
# backward channels recorded, scored by the frozen nine-arm attacker with
# matched shuffled nulls and per-row prediction dumps for the paired statistic.
#
# Arms (GRAD_DP): off = leak exposure; clip_noise = the frozen defense.
# NAKED=1: full-width control (latent 1024, gauges off, chaff 0) -- the
# representation-matched sensitivity control at this boundary.
#
# FRAMES <= STEPS - WARMUP by construction: the packaged runner predates the
# pre-warmup capture guard, so the capture window must sit entirely after
# warmup.
#
# NOTE: --secret-token-gauge is intentionally rejected for the
# monomial_moe_radial cloud (runner validation: "the radial MoE cloud reads
# norms and is intentionally incompatible with the scale gauge"). The defended
# posture here is therefore the E1/deleg6040 one: rotation + permutation +
# noise + chaff.
set -euo pipefail

SPLIT_AFTER=${SPLIT_AFTER:-21}
RESUME_AFTER=${RESUME_AFTER:-26}
STEPS=${STEPS:-2000}
WARMUP=${WARMUP:-200}
FRAMES=${FRAMES:-1024}
TRAIN_BLOCKS=${TRAIN_BLOCKS:-256}
EVAL_BLOCKS=${EVAL_BLOCKS:-256}
SEED=${SEED:-51}
GRAD_DP=${GRAD_DP:-off}
NAKED=${NAKED:-0}
CELL=${CELL:-c1_def${GRAD_DP}_s${SEED}}
ARMS=${ARMS:-"grad_real wire_real joint_real_scaled grad_real_shuffled"}

MODEL=/workspace/experiments/models/qwen3-0.6b
CORPUS=${CORPUS:-/workspace/experiments/models/wikitext2_corpus.txt}
CA=/workspace/experiments/tls/ca.crt
CLOUD_URL=${CLOUD_URL:-wss://ucn:5321}
OUTDIR=/workspace/experiments/results/training/phasec
BUNDLEDIR=$OUTDIR/bundles

if [ "$FRAMES" -gt $((STEPS - WARMUP)) ]; then
  echo "FRAMES must be <= STEPS - WARMUP (pre-warmup capture guard)" >&2
  exit 2
fi

mkdir -p "$BUNDLEDIR/$CELL"

if [ "$NAKED" = "1" ]; then
  DEFENSE_FLAGS="--latent-dim 1024 --noise-multiplier 0.001 --chaff-tokens 0"
else
  DEFENSE_FLAGS="--secret-wire-rotation --secret-token-permutation \
    --latent-dim 64 --noise-multiplier 0.35 --chaff-tokens 48"
fi

echo "=== cell $CELL: split=$SPLIT_AFTER/$RESUME_AFTER steps=$STEPS frames=$FRAMES blocks=$TRAIN_BLOCKS/$EVAL_BLOCKS seed=$SEED grad_dp=$GRAD_DP naked=$NAKED url=$CLOUD_URL ==="
date -u +%Y-%m-%dT%H:%M:%SZ

# shellcheck disable=SC2086
python3 bin/run_latent_native_v5_06b.py \
  --model "$MODEL" --corpus "$CORPUS" \
  --output "$OUTDIR/${CELL}.json" \
  --cloud-tls-ca "$CA" \
  --cloud-kind monomial_moe_radial \
  --cloud-experts 8 --cloud-layers 2 \
  $DEFENSE_FLAGS --clip-norm 1.0 \
  --split-after "$SPLIT_AFTER" --resume-after "$RESUME_AFTER" --seq-len 32 \
  --steps "$STEPS" --warmup-steps "$WARMUP" \
  --train-blocks "$TRAIN_BLOCKS" --eval-blocks "$EVAL_BLOCKS" \
  --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
  --lr 3e-4 --remote-grad-clip 1.0 \
  --token-scale-sigma 0.75 --seed "$SEED" \
  --attacker-bundle "$BUNDLEDIR/${CELL}_bundle.pt" \
  --grad-channel-bundle "$BUNDLEDIR/${CELL}_gradchannel.pt" \
  --grad-channel-frames "$FRAMES" \
  --outbound-grad-dp "$GRAD_DP" \
  --cloud-url "$CLOUD_URL" --adversary-strength 1.0 --mine-penalty 0.1

echo "=== runner done; scoring the forward-frame bundle ==="
date -u +%Y-%m-%dT%H:%M:%SZ
python3 -m attacker --attack latent-probe \
  --bundle "$BUNDLEDIR/${CELL}_bundle.pt" \
  --output "$OUTDIR/${CELL}_attacker.json" \
  --dump-eval-predictions "$BUNDLEDIR/${CELL}_forward_pred.pt"

echo "=== building outbound-wire arms ==="
date -u +%Y-%m-%dT%H:%M:%SZ
python3 bin/deleg6040_grad_bundle.py \
  --capture "$BUNDLEDIR/${CELL}_gradchannel.pt" \
  --outdir "$BUNDLEDIR/$CELL" \
  --report "$OUTDIR/${CELL}_bundles.json"

for arm in $ARMS; do
  echo "=== scoring arm $arm with the frozen nine-arm attacker ==="
  date -u +%Y-%m-%dT%H:%M:%SZ
  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/$CELL/${arm}.pt" \
    --output "$OUTDIR/${CELL}_arm_${arm}.json" \
    --dump-eval-predictions "$BUNDLEDIR/$CELL/${arm}_pred.pt"
done

echo "=== $CELL complete ==="
date -u +%Y-%m-%dT%H:%M:%SZ
