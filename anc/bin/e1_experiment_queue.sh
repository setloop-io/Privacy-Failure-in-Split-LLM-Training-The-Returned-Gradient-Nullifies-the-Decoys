#!/usr/bin/env bash
# Overnight/day experiment queue for the E1 line (experiments W0.1, W5.7, gate item 2).
#
# Runs detached and sequentially, because TLN carries ~99.7% of the compute and two
# concurrent training cells would contend on the same GPU.
#
# Stage 1  score the four E1 arms already on disk -- no retraining needed.
#          grad_all, grad_shuffled, joint_real, joint_real_scaled.
#          joint_real / joint_real_scaled are what gate item 2's "joint attack"
#          requires; this stage scores them on a forward-passing cell.
# Stage 2  replicate E1 at seeds 43 and 44, taking the finding from n=1 to
#          three seeds.
#
# Resilience: the cloud server on UCN is (re)started before each training cell and
# health-checked; a stage that fails is logged and the queue continues rather than
# aborting the rest.
set -uo pipefail

CELL=e1_unprot_a2b_split14
OUT=/workspace/experiments/results/training/e1_unprotected
BUNDLES="$OUT/bundles/$CELL"
IMAGE=split-inference:spark
MOUNTS=(-v "$HOME/experiments:/workspace/experiments" -v "$HOME/dtraining:$HOME/dtraining")
UCN=poseidon

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

in_container() {
  docker run --rm --gpus all --network host --ipc host \
    -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    "${MOUNTS[@]}" -w "$HOME/dtraining" "$IMAGE" bash -lc "$1"
}

ensure_cloud() {
  if ssh -o BatchMode=yes "$UCN" 'docker ps --filter name=latent-cloud --format "{{.Names}}"' 2>/dev/null | grep -q latent-cloud; then
    log "cloud server already up"; return 0
  fi
  log "starting cloud server on $UCN"
  ssh -o BatchMode=yes "$UCN" '
    docker rm -f latent-cloud >/dev/null 2>&1 || true
    docker run -d --name latent-cloud --gpus all --network host --ipc host \
      -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
      -v $HOME/experiments:/workspace/experiments -v $HOME/dtraining:$HOME/dtraining \
      -w $HOME/dtraining split-inference:spark \
      python3 split-training/latent_cloud_server.py \
        --latent-dim 64 --forbidden-hidden-dim 1024 \
        --cloud-kind monomial_moe_radial --cloud-experts 8 --cloud-layers 2 \
        --port 5025 \
        --tls-cert /workspace/experiments/tls/cloud-server.crt \
        --tls-key /workspace/experiments/tls/cloud-server.key' >/dev/null 2>&1
  sleep 10
  ssh -o BatchMode=yes "$UCN" 'docker logs latent-cloud 2>&1 | tail -1' 2>/dev/null | grep -q '"status": "ready"' \
    && log "cloud server ready" || log "WARNING cloud server did not report ready"
}

# stage 1
log "STAGE 1: scoring the four unscored E1 arms (no retraining)"
for arm in joint_real joint_real_scaled grad_all grad_shuffled; do
  if [ -f "$HOME/experiments/results/training/e1_unprotected/bundles/$CELL/${arm}.pt" ]; then
    log "  scoring $arm"
    in_container "python3 -m attacker --attack latent-probe \
      --bundle $BUNDLES/${arm}.pt \
      --output $OUT/${CELL}_arm_${arm}.json \
      --dump-eval-predictions $BUNDLES/${arm}_pred.pt" >/dev/null 2>&1 \
      && log "  scored $arm" || log "  FAILED $arm"
    in_container "python3 bin/paired_advantage.py --dump $BUNDLES/${arm}_pred.pt \
      --output $OUT/${CELL}_arm_${arm}_paired.json" >/dev/null 2>&1 \
      && log "  paired $arm" || log "  paired FAILED $arm"
  else
    log "  no bundle for $arm"
  fi
done

# stage 2
log "STAGE 2: replicating E1 at seeds 43 and 44 (W5.7 / gate item 5)"
for seed in 43 44; do
  target="$OUT/${CELL}_s${seed}.json"
  if [ -f "$HOME/experiments/results/training/e1_unprotected/${CELL}_s${seed}.json" ]; then
    log "  seed $seed already present, skipping"; continue
  fi
  ensure_cloud
  log "  training seed $seed (expect ~65 min)"
  in_container "SEED=$seed CELL=${CELL}_s${seed} bash bin/e1_unprotected_cell.sh" \
    > "$HOME/e1_seed${seed}.log" 2>&1 \
    && log "  seed $seed complete" || log "  seed $seed FAILED (see ~/e1_seed${seed}.log)"
  for arm in grad_real grad_real_shuffled wire_real; do
    d="$OUT/bundles/${CELL}_s${seed}/${arm}_pred.pt"
    in_container "python3 bin/paired_advantage.py --dump $d \
      --output $OUT/${CELL}_s${seed}_arm_${arm}_paired.json" >/dev/null 2>&1 \
      && log "  paired s${seed} $arm" || log "  paired s${seed} $arm unavailable"
  done
done

log "QUEUE COMPLETE"
