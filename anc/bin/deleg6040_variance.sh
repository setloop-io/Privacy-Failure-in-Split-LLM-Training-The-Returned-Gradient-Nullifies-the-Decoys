#!/usr/bin/env bash
# T1.1 -- variance of the reported metrics under identical configuration.
#
# WHY THIS EXISTS. Every cell in this effort is n=1, and the external adversarial
# review (finding 10; the review document is not included in this release)
# established that no variance estimate exists for closure or residual, while
# sequences like 86.1 / 86.0 / 86.4 / 86.5 were nonetheless used to declare
# mechanisms closed.
#
# Repeating the SAME command yields a different result here, by design:
# privacy_runtime/latent_native.py:105 draws fresh CSPRNG entropy per DP call and
# the gauge is redrawn per block, so --seed does not pin the released frames. That
# makes an honest variance estimate possible without varying any input at all --
# every run below is byte-identical in configuration.
#
# Target: the configuration this work leans on most heavily,
# deleg_6040_conv10k_split14 (split_after 14, 10,000 steps, 256 blocks, D=64).
# The already-committed cell is sample 1; this adds four more.
set -euo pipefail

REPEATS=${REPEATS:-4}
OUTDIR=$HOME/experiments/results/training/deleg6040

for i in $(seq 1 "$REPEATS"); do
  echo "######## variance repeat $i/$REPEATS ########"
  OUT=$OUTDIR/t1_precheck_rep${i}.json \
    bash "$HOME/dtraining/bin/deleg6040_precheck.sh" > /dev/null

  docker run --rm --name "pxrun_var${i}" --gpus all --network host --ipc host \
    -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -e CONTAINER_IMAGE=split-inference:spark \
    -e SPLIT_AFTER=14 -e STEPS=10000 \
    -e CELL="t1_variance_rep${i}_split14" \
    -v "$HOME/experiments:/workspace/experiments" \
    -v "$HOME/dtraining:$HOME/dtraining" -w "$HOME/dtraining" \
    split-inference:spark \
    bash bin/deleg6040_cell.sh \
    > "$OUTDIR/t1_variance_rep${i}.log" 2>&1

  echo "repeat $i done"
done

echo "VARIANCE_COMPLETE"
