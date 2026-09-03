#!/usr/bin/env bash
# Phase D1 + E1 tail (2026-08-27): runs after the defended-cell queue.
#
# D1: the gradient-leak exposure cell on a SECOND corpus (wikitext2_public.txt),
#     three seeds -- a limited robustness check of the token-class effect's
#     existence (not a corpus-independence claim: one extra corpus).
# E1: one clean rerun of the provenance-anomalous seed 42 in the a2b
#     configuration from packaged code (the W1.7 s42 draw was served by the
#     pre-existing dirty container). Long cell: 40k steps.
set -uo pipefail
LOG=$HOME/phasec_queue3.log
run_cell() {
  local cell=$1 seed=$2 dp=$3 naked=$4 url=$5 corpus=$6 steps=$7 tblocks=$8 eblocks=$9 frames=${10} split=${11} resume=${12}
  echo "[$(date -u +%FT%TZ)] START $cell seed=$seed corpus=$corpus steps=$steps split=$split" >> $LOG
  docker run --rm --gpus all --network host --ipc host \
    -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -v $HOME/experiments:/workspace/experiments -v $HOME/phasec_packaged:/workspace/phasec \
    -w /workspace/phasec split-inference:spark \
    bash -c "SEED=$seed GRAD_DP=$dp NAKED=$naked CELL=$cell CLOUD_URL=$url \
             CORPUS=$corpus STEPS=$steps TRAIN_BLOCKS=$tblocks EVAL_BLOCKS=$eblocks FRAMES=$frames \
             SPLIT_AFTER=$split RESUME_AFTER=$resume \
             bash bin/phasec_utility_cell.sh" >> $LOG 2>&1
  local rc=$?
  echo "[$(date -u +%FT%TZ)] DONE $cell rc=$rc" >> $LOG
}

# D1: second corpus, defended topology, gradient unprotected, 3 fresh seeds.
for s in 51 52 53; do
  run_cell d1_puboff_s$s $s off 0 wss://ucn:5322 \
    /workspace/experiments/models/wikitext2_public.txt 2000 256 256 1024 21 26
done

# E1 tail: the provenance-anomalous seed 42, clean rerun from packaged code in
# the a2b configuration (split 14/26, 40k steps, 4096 blocks, 4096 frames).
run_cell e1_s42_clean 42 off 0 wss://ucn:5322 \
  /workspace/experiments/models/wikitext2_corpus.txt 40000 4096 4096 4096 14 26

echo "[$(date -u +%FT%TZ)] QUEUE3 COMPLETE" >> $LOG
