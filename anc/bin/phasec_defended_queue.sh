#!/usr/bin/env bash
# Phase C1 defended cells (2026-08-27): the six defended cells of the C1
# matrix. The token-gauge flag is rejected by the monomial_moe_radial cloud
# (see phasec_utility_cell.sh).
#
# Runs on tln as: nohup ~/phasec_defended_queue.sh >/dev/null 2>&1 &
# Cells are anonymous `docker run --rm`; every cell is
# verified by its output artifacts, never by rc alone (a command substitution
# in the DONE line masks the real status unless rc is captured first).
set -uo pipefail
LOG=$HOME/phasec_queue2.log
run_cell() {
  local cell=$1 seed=$2 dp=$3
  echo "[$(date -u +%FT%TZ)] START $cell seed=$seed dp=$dp" >> $LOG
  docker run --rm --gpus all --network host --ipc host \
    -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -v $HOME/experiments:/workspace/experiments -v $HOME/phasec_packaged:/workspace/phasec \
    -w /workspace/phasec split-inference:spark \
    bash -c "SEED=$seed GRAD_DP=$dp NAKED=0 CELL=$cell CLOUD_URL=wss://ucn:5321 bash bin/phasec_defended_cell.sh" >> $LOG 2>&1
  local rc=$?
  echo "[$(date -u +%FT%TZ)] DONE $cell rc=$rc" >> $LOG
}
for s in 51 52 53; do
  run_cell c1_defoff_s$s $s off
  run_cell c1_defon_s$s $s clip_noise
done
echo "[$(date -u +%FT%TZ)] QUEUE2 COMPLETE" >> $LOG
