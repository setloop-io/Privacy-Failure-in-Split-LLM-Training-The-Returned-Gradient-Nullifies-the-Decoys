#!/usr/bin/env bash
# The 60/40 delegation cell: the v13.0 (a9_mine010) configuration with the
# private/cloud boundary moved earlier, and nothing else changed.
#
# Delegation is (resume_after - split_after - 1) / 28 layers:
#   split_after 21, resume_after 26 ->  4/28 = 14.3%  (v13.0 baseline)
#   split_after 14, resume_after 26 -> 11/28 = 39.3%  (this cell, primary)
#   split_after 13, resume_after 26 -> 12/28 = 42.9%  (optional second point)
#
# Every other flag is copied verbatim from bin/latent_v13_stage.sh's run_cell()
# plus its a9_mine010 extras. Two deliberate deviations, both recorded in the
# write-up:
#   1. --cloud-url uses poseidon.cluster, not ucn. Containment pins `ucn`
#      to 127.0.0.1 on both nodes, so the original URL would silently reach
#      localhost. transport_tls is a string test on the wss:// prefix and
#      cloud_url is not recorded in the artifact, so this is invisible to the
#      config comparison.
#   2. The attacker bundle is NOT deleted after scoring (the v13 stage script
#      removes it). Retaining it permits re-scoring and a shuffled-label
#      negative control without a full re-run.
#
# Uses `set -e` (unlike latent_v13_stage.sh's `set -uo pipefail`), so a failed
# stage cannot print a success line.
set -euo pipefail

SPLIT_AFTER=${SPLIT_AFTER:-14}
CELL=${CELL:-deleg_6040_split${SPLIT_AFTER}}

# STEPS is 2000 for every cell that answers the seed's question, so split_after
# stays the only thing that varies across D1 and D3. It is overridable ONLY for
# the labelled convergence control, which deliberately changes the training
# budget to test whether 2,000 steps ceasing to suffice past 8 delegated layers
# -- rather than delegation share itself -- is what moves the gate. That control
# is never reported as a ladder rung. warmup stays at 200 at every budget,
# matching latent_v12_converge10k_06b_s42.
STEPS=${STEPS:-2000}

# Capability-experiment knobs. All default to the v13 values, so every ladder
# cell is unaffected; a cell that overrides one is NOT a ladder rung and
# bin/deleg6040_config_diff.py reports CONFIG violations for it by design.
#
# A1 raises surrogate capacity at fixed D=64: experts 8->32 and layers 2->4 take
# the cloud module from 161 to 1281 parameters. The handshake validates both
# against the server, so this needs a server started with matching values --
# :5026, not the ladder's :5025.
# A2 raises the training window: 256 blocks is 0.7% of the corpus and makes
# 10,000 steps roughly 39 passes over the train split.
# A3 widens the channel itself. This is the ONLY knob here that touches the
# trust axis: D is the width of everything ucn ever sees, and the whole
# privacy result rests on it. Every A3 cell needs a full privacy re-gate, and a
# pass at D=64 licenses nothing at D=96 or D=128.
# latent_dim must be divisible by cloud_heads (default 4).
# A4 sublayer-granular delegation. "full" is the default and is the path every
# committed artifact was produced with; "mlp"/"attn" delegate one sublayer and
# keep the other trusted. Still one channel crossing per block, so the trusted
# sublayers run on a trajectory that omits the delegated ones: this holds the
# boundary CAPACITY fixed and varies only how it is ALLOCATED, which is exactly
# the hypothesis A4 tests.
DELEGATE_SUBLAYER=${DELEGATE_SUBLAYER:-full}
LATENT_DIM=${LATENT_DIM:-64}
CLOUD_EXPERTS=${CLOUD_EXPERTS:-8}
CLOUD_LAYERS=${CLOUD_LAYERS:-2}
TRAIN_BLOCKS=${TRAIN_BLOCKS:-256}
EVAL_BLOCKS=${EVAL_BLOCKS:-256}

MODEL=/workspace/experiments/models/qwen3-0.6b
CORPUS=/workspace/experiments/models/wikitext2_corpus.txt
CA=/workspace/experiments/tls/ca.crt
CLOUD_URL=${CLOUD_URL:-wss://poseidon.cluster:5025}
OUTDIR=/workspace/experiments/results/training/deleg6040
BUNDLEDIR=$OUTDIR/bundles

mkdir -p "$BUNDLEDIR"

echo "=== cell $CELL: split_after=$SPLIT_AFTER resume_after=26 steps=$STEPS experts=$CLOUD_EXPERTS layers=$CLOUD_LAYERS blocks=$TRAIN_BLOCKS/$EVAL_BLOCKS D=$LATENT_DIM sublayer=$DELEGATE_SUBLAYER url=$CLOUD_URL ==="
date -u +%Y-%m-%dT%H:%M:%SZ

python3 bin/run_latent_native_v5_06b.py \
  --model "$MODEL" --corpus "$CORPUS" \
  --output "$OUTDIR/${CELL}.json" \
  --cloud-tls-ca "$CA" \
  --cloud-kind monomial_moe_radial \
  --cloud-experts "$CLOUD_EXPERTS" --cloud-layers "$CLOUD_LAYERS" \
  --secret-wire-rotation --secret-token-permutation \
  --latent-dim "$LATENT_DIM" --noise-multiplier 0.35 --clip-norm 1.0 \
  --delegate-sublayer "$DELEGATE_SUBLAYER" \
  --split-after "$SPLIT_AFTER" --resume-after 26 --seq-len 32 \
  --steps "$STEPS" --warmup-steps 200 \
  --train-blocks "$TRAIN_BLOCKS" --eval-blocks "$EVAL_BLOCKS" \
  --attack-steps 256 --probe-restarts 3 --attacker-updates 3 \
  --lr 3e-4 --remote-grad-clip 1.0 \
  --token-scale-sigma 0.75 --chaff-tokens 48 --seed 42 \
  --attacker-bundle "$BUNDLEDIR/${CELL}_bundle.pt" \
  --cloud-url "$CLOUD_URL" --adversary-strength 1.0 --mine-penalty 0.1

echo "=== runner done; scoring the frozen nine-arm attacker ==="
date -u +%Y-%m-%dT%H:%M:%SZ

python3 -m attacker --attack latent-probe \
  --bundle "$BUNDLEDIR/${CELL}_bundle.pt" \
  --output "$OUTDIR/${CELL}_attacker.json"

echo "=== cell $CELL complete (bundle retained at $BUNDLEDIR/${CELL}_bundle.pt) ==="
date -u +%Y-%m-%dT%H:%M:%SZ
