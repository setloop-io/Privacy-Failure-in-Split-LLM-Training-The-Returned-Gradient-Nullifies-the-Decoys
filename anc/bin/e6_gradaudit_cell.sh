#!/usr/bin/env bash
# Experiment W5.6 / E6: gradient-channel audit on the cells E1 did not
# cover -- the 2k ladder rungs, the D=96/128 latent-width arms, and the
# 40k/100k budget arms.
#
# Base: bin/e1_unprotected_cell.sh, flags verbatim. Two generalizations:
#   1. LATENT_DIM is an env override (E1 hardcodes 64; E6's D-arms need 96/128).
#   2. OUTDIR is an env override (E6 writes to training/gradaudit, keeping the
#      e1_unprotected tree exclusively W1.7's).
# Defaults are E1's, so this script is a drop-in for the E1 configuration too.
# The unprotected wire (GRAD_DP=off) stays the default for the same reason as E1:
# the audit measures the unfixed case, which is what every published artifact
# was produced with.
set -euo pipefail

SPLIT_AFTER=${SPLIT_AFTER:-14}
STEPS=${STEPS:-40000}
FRAMES=${FRAMES:-4096}
TRAIN_BLOCKS=${TRAIN_BLOCKS:-4096}
EVAL_BLOCKS=${EVAL_BLOCKS:-4096}
SEED=${SEED:-42}
CELL=${CELL:-gradaudit_a2b_split14}
ARMS=${ARMS:-"grad_real grad_real_shuffled wire_real"}
GRAD_DP=${GRAD_DP:-off}
LATENT_DIM=${LATENT_DIM:-64}
SEQ_LEN=${SEQ_LEN:-32}

MODEL=/workspace/experiments/models/qwen3-0.6b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
CA=/workspace/experiments/tls/ca.crt
CLOUD_URL=${CLOUD_URL:-wss://poseidon.cluster:5025}
OUTDIR=${OUTDIR:-/workspace/experiments/results/training/gradaudit}
BUNDLEDIR="$OUTDIR/bundles"

mkdir -p "$BUNDLEDIR/$CELL"

echo "=== $CELL split=$SPLIT_AFTER steps=$STEPS frames=$FRAMES blocks=$TRAIN_BLOCKS/$EVAL_BLOCKS D=$LATENT_DIM seed=$SEED grad_dp=$GRAD_DP ==="
date -u +%Y-%m-%dT%H:%M:%SZ

python3 bin/run_latent_native_v5_06b.py \
  --model "$MODEL" --corpus "$CORPUS" \
  --output "$OUTDIR/${CELL}.json" \
  --cloud-tls-ca "$CA" \
  --cloud-kind monomial_moe_radial \
  --cloud-experts 8 --cloud-layers 2 \
  --secret-wire-rotation --secret-token-permutation \
  --latent-dim "$LATENT_DIM" --noise-multiplier 0.35 --clip-norm 1.0 \
  --split-after "$SPLIT_AFTER" --resume-after 26 --seq-len "$SEQ_LEN" \
  --steps "$STEPS" --warmup-steps 200 \
  --train-blocks "$TRAIN_BLOCKS" --eval-blocks "$EVAL_BLOCKS" \
  --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
  --lr 3e-4 --remote-grad-clip 1.0 \
  --token-scale-sigma 0.75 --chaff-tokens 48 --seed "$SEED" \
  --outbound-grad-dp "$GRAD_DP" \
  --attacker-bundle "$BUNDLEDIR/${CELL}_bundle.pt" \
  --grad-channel-bundle "$BUNDLEDIR/${CELL}_gradchannel.pt" \
  --grad-channel-frames "$FRAMES" \
  --cloud-url "$CLOUD_URL" --adversary-strength 1.0 --mine-penalty 0.1

echo "=== runner done; scoring the forward-frame bundle ==="
date -u +%Y-%m-%dT%H:%M:%SZ
python3 -m attacker --attack latent-probe \
  --bundle "$BUNDLEDIR/${CELL}_bundle.pt" \
  --output "$OUTDIR/${CELL}_attacker.json"

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
