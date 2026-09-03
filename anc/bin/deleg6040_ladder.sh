#!/usr/bin/env bash
# Delegation dose-response ladder. Run ON THE TRUSTED NODE (odysseus).
#
# WHY THIS EXISTS, beyond the seed's D1 and D3: main.tex:170-172 states the
# campaign's own evidentiary rule -- below the statistical floor "claims are
# made only on three-seed runs or monotone dose-response". D1 is a single seed,
# and its +2.090 pp reading is far above the floor, but a single point cannot
# show that delegation share is what moves the gate. A monotone ladder across
# split points can, and each cell is about six minutes.
#
# split_after -> delegated layers (resume_after 26, 28 layers total):
#   21 ->  4  = 14.3%   the published corner and the same-stack control
#   19 ->  6  = 21.4%   added here
#   17 ->  8  = 28.6%   added here; same share as the published v10.0
#                       two-segment cell, but as ONE boundary
#   14 -> 11  = 39.3%   the seed's primary cell (D1)
#   13 -> 12  = 42.9%   the seed's optional second point (D3)
#
# Cells run strictly sequentially, never concurrently: eval_time_ratio and
# train_seconds are reported numbers, and GPU contention would corrupt them.
# Containment and corpus identity are re-verified before every cell, per the
# seed's invariant 1, and a failed precheck aborts the whole ladder.
set -euo pipefail

SPLITS=${SPLITS:-"19 17 13"}
OUTDIR=$HOME/experiments/results/training/deleg6040

for split in $SPLITS; do
  echo "######## ladder: split_after=$split ########"
  OUT=$OUTDIR/ladder_precheck_s${split}.json \
    bash "$HOME/dtraining/bin/deleg6040_precheck.sh" > /dev/null

  docker run --rm --name "pxrun_deleg6040_s${split}" --gpus all \
    --network host --ipc host \
    -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -e CONTAINER_IMAGE=split-inference:spark \
    -e SPLIT_AFTER="$split" -e CELL="deleg_6040_ladder_split${split}" \
    -v "$HOME/experiments:/workspace/experiments" \
    -v "$HOME/dtraining:$HOME/dtraining" -w "$HOME/dtraining" \
    split-inference:spark \
    bash bin/deleg6040_cell.sh \
    > "$OUTDIR/ladder_split${split}.log" 2>&1

  echo "ladder split_after=$split done"
done

echo "LADDER_COMPLETE"
