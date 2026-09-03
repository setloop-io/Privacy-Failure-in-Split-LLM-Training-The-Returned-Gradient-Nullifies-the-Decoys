#!/usr/bin/env bash
# Threat-model scope audit cell: the 60/40 delegation configuration, run with
# the outbound backward wire recorded.
#
# WHY THIS EXISTS. The external adversarial review (finding 1; the review
# document is not included in this release) established that the committed
# attacker bundles hold only forward frames, while the compromised node also
# receives the per-step output gradient. This cell captures that gradient in
# situ and scores it with the SAME frozen nine-arm attacker the gate uses.
#
# Every runner flag below is copied verbatim from bin/deleg6040_cell.sh. The
# only additions are --grad-channel-bundle / --grad-channel-frames, which are
# inert in that script and change no RNG draw. The attacker bundle is retained.
set -euo pipefail

SPLIT_AFTER=${SPLIT_AFTER:-14}
STEPS=${STEPS:-10000}
FRAMES=${FRAMES:-512}
TRAIN_BLOCKS=${TRAIN_BLOCKS:-256}
EVAL_BLOCKS=${EVAL_BLOCKS:-256}
CELL=${CELL:-grad_channel_split${SPLIT_AFTER}}
ARMS=${ARMS:-"wire_all grad_all grad_real wire_real joint_real joint_real_scaled grad_shuffled"}

MODEL=/workspace/experiments/models/qwen3-0.6b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
CA=/workspace/experiments/tls/ca.crt
CLOUD_URL=${CLOUD_URL:-wss://poseidon.cluster:5025}
OUTDIR=/workspace/experiments/results/training/gradaudit
BUNDLEDIR=$OUTDIR/bundles

mkdir -p "$BUNDLEDIR/$CELL"

echo "=== cell $CELL: split_after=$SPLIT_AFTER steps=$STEPS frames=$FRAMES blocks=$TRAIN_BLOCKS/$EVAL_BLOCKS url=$CLOUD_URL ==="
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
  --token-scale-sigma 0.75 --chaff-tokens 48 --seed 42 \
  --attacker-bundle "$BUNDLEDIR/${CELL}_bundle.pt" \
  --grad-channel-bundle "$BUNDLEDIR/${CELL}_gradchannel.pt" \
  --grad-channel-frames "$FRAMES" \
  --cloud-url "$CLOUD_URL" --adversary-strength 1.0 --mine-penalty 0.1

echo "=== runner done; scoring the committed forward-frame bundle ==="
date -u +%Y-%m-%dT%H:%M:%SZ
python3 -m attacker --attack latent-probe \
  --bundle "$BUNDLEDIR/${CELL}_bundle.pt" \
  --output "$OUTDIR/${CELL}_attacker.json"

echo "=== building outbound-wire arms ==="
python3 bin/deleg6040_grad_bundle.py \
  --capture "$BUNDLEDIR/${CELL}_gradchannel.pt" \
  --outdir "$BUNDLEDIR/$CELL" \
  --report "$OUTDIR/${CELL}_bundles.json"

for arm in $ARMS; do
  echo "=== scoring arm $arm with the frozen nine-arm attacker ==="
  date -u +%Y-%m-%dT%H:%M:%SZ
  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/$CELL/${arm}.pt" \
    --output "$OUTDIR/${CELL}_arm_${arm}.json"
done

echo "=== cell $CELL complete ==="
date -u +%Y-%m-%dT%H:%M:%SZ
