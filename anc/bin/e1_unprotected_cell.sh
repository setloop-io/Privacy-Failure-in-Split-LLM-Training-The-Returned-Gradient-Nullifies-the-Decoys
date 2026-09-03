#!/usr/bin/env bash
# Experiment W0.1 / E1: does the UNPROTECTED gradient channel break a cell
# whose forward frame passes the gate?
#
# Base: bin/deleg6040_grad_cell_t2b.sh (not included in this release), runner
# flags verbatim.
# Three deliberate changes:
#   1. a2b scale -- 40k steps, 4096 train/eval blocks, 4096 recorded grad frames,
#      giving n = 65,536 independent evaluation rows on a forward-passing config.
#   2. --outbound-grad-dp off (GRAD_DP) -- the unprotected backward wire. The runner default
#      is clip_noise (the issue #105 fix); gradfix_a2b_split14 already measured the
#      FIXED case and read at floor. E1 is the unfixed case, which is what every
#      published artifact was produced with and what the audit never resolved.
#   3. --dump-eval-predictions on every arm -- per-row output the W2.1a paired,
#      cluster-aware statistic needs. Without it that statistic would require a
#      full re-run.
set -euo pipefail

SPLIT_AFTER=${SPLIT_AFTER:-14}
STEPS=${STEPS:-40000}
FRAMES=${FRAMES:-4096}
TRAIN_BLOCKS=${TRAIN_BLOCKS:-4096}
EVAL_BLOCKS=${EVAL_BLOCKS:-4096}
SEED=${SEED:-42}
CELL=${CELL:-e1_unprot_a2b_split14}
ARMS=${ARMS:-"grad_real grad_real_shuffled wire_real"}
# E1 is the UNPROTECTED case, so 'off' is the default and must stay the default.
# Set GRAD_DP=clip_noise to measure the fix instead; the runner's own default is
# clip_noise, and gradfix_a2b_split14 was produced that way.
GRAD_DP=${GRAD_DP:-off}

MODEL=/workspace/experiments/models/qwen3-0.6b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
CA=/workspace/experiments/tls/ca.crt
CLOUD_URL=${CLOUD_URL:-wss://poseidon.cluster:5025}
OUTDIR=/workspace/experiments/results/training/e1_unprotected
BUNDLEDIR="$OUTDIR/bundles"

mkdir -p "$BUNDLEDIR/$CELL"

echo "=== $CELL split=$SPLIT_AFTER steps=$STEPS frames=$FRAMES blocks=$TRAIN_BLOCKS/$EVAL_BLOCKS seed=$SEED grad_dp=$GRAD_DP ==="
date -u +%Y-%m-%dT%H:%M:%SZ

python3 bin/run_latent_native_v5_06b.py \
  --model "$MODEL" --corpus "$CORPUS" \
  --output "$OUTDIR/${CELL}.json" \
  --cloud-tls-ca "$CA" \
  --cloud-kind monomial_moe_radial \
  --cloud-experts 8 --cloud-layers 2 \
  --secret-wire-rotation --secret-token-permutation \
  --latent-dim 64 --noise-multiplier 0.35 --clip-norm 1.0 \
  --split-after "$SPLIT_AFTER" --resume-after 26 --seq-len 32 \
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
