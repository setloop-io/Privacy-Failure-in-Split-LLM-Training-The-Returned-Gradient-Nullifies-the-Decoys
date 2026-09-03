#!/usr/bin/env bash
# Phase C2 (2026-08-27): rescore a completed phasec cell with the XG^T arms
# and emit the leakage-metrics records (including semantic_cosine) for every
# scored dump.
#
# Rebuilds the arm bundles with the current deleg6040_grad_bundle.py (which
# carries the W4.2 xgxt_real / xgxt_real_shuffled arms), scores both new arms
# with the frozen nine-arm attacker, and runs bin/leakage_metrics.py over every
# per-row prediction dump with the public embedding table.
#
# Usage: EMB=/workspace/experiments/models/qwen3-0.6b_embed.pt bash bin/phasec_rescore_xgxt.sh CELL
set -euo pipefail

CELL=${CELL:?cell name, e.g. c1_defoff_s51}
SEED=${SEED:-$(echo "$CELL" | sed -E 's/.*_s([0-9]+)$/\1/')}
EMB=${EMB:-/workspace/experiments/models/qwen3-0.6b_embed.pt}
OUTDIR=/workspace/experiments/results/training/phasec
BUNDLEDIR=$OUTDIR/bundles

echo "=== rescore $CELL (seed $SEED): rebuild arms with XG^T ==="
date -u +%Y-%m-%dT%H:%M:%SZ
python3 bin/deleg6040_grad_bundle.py \
  --capture "$BUNDLEDIR/${CELL}_gradchannel.pt" \
  --outdir "$BUNDLEDIR/$CELL" \
  --report "$OUTDIR/${CELL}_bundles.json"

for arm in xgxt_real xgxt_real_shuffled xgxt_raw_real xgxt_raw_real_shuffled; do
  echo "=== scoring arm $arm ==="
  python3 -m attacker --attack latent-probe \
    --bundle "$BUNDLEDIR/$CELL/${arm}.pt" \
    --output "$OUTDIR/${CELL}_arm_${arm}.json" \
    --dump-eval-predictions "$BUNDLEDIR/$CELL/${arm}_pred.pt"
done

echo "=== leakage metrics over all dumps (semantic_cosine enabled) ==="
for dump in "$BUNDLEDIR/${CELL}_forward_pred.pt" \
            "$BUNDLEDIR/$CELL"/grad_real_pred.pt \
            "$BUNDLEDIR/$CELL"/wire_real_pred.pt \
            "$BUNDLEDIR/$CELL"/joint_real_scaled_pred.pt \
            "$BUNDLEDIR/$CELL"/grad_real_shuffled_pred.pt \
            "$BUNDLEDIR/$CELL"/xgxt_real_pred.pt \
            "$BUNDLEDIR/$CELL"/xgxt_real_shuffled_pred.pt; do
  [ -f "$dump" ] || continue
  name=$(basename "$dump" _pred.pt)
  attack=forward_only; case "$name" in grad_*|xgxt_*) attack=gradient_only;; joint_*) attack=joint_forward_gradient;; esac
  python3 bin/leakage_metrics.py --dump "$dump" --arm "$name" \
    --attack "$attack" --seed "$SEED" ${EMB:+--embeddings "$EMB"} \
    --output "$OUTDIR/${CELL}_metrics_${name}.json" \
    --jsonl "$OUTDIR/${CELL}_metrics_${name}.jsonl"
done

echo "=== rescore $CELL complete ==="
date -u +%Y-%m-%dT%H:%M:%SZ
