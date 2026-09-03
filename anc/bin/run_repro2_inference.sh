#!/bin/bash
# run_repro2_inference.sh — clean inference reproduction with n=3 repetitions.
# Per model: rtt_analysis ×3 at each profile; identity ×3 (near-tie behavior
# is numerics-dependent, so a single identity run is not enough).
# Usage: run_repro2_inference.sh <model-dir> <split-after> <resume-at> <tag> [cloud-start cloud-end]
set -e
MODEL=${1:?}; SA=${2:?}; RA=${3:?}; TAG=${4:?}
CLOUD_SPLIT=""
[ -n "$5" ] && [ -n "$6" ] && CLOUD_SPLIT="--cloud-start $5 --cloud-end $6"
BIN=$HOME/experiments/bin
CLOUD=http://10.10.10.2:5000
OUTC=/workspace/experiments/results/v2/$TAG
mkdir -p $HOME/experiments/results/v2/$TAG

wait_health () { for i in $(seq 1 60); do curl -s -m 2 "$1" >/dev/null && return 0; sleep 8; done; echo "HEALTH TIMEOUT"; exit 1; }

# --- preflight: both nodes must run the SAME dtraining commit and the SAME
# container image, otherwise cross-node results are not comparable. Hard-fail.
img_digest () {  # img_digest [ssh-host] — repo digest, falling back to image ID
  local CMD='D=$(docker images --digests split-inference:spark --format "{{.Digest}}");
             if [ -z "$D" ] || [ "$D" = "<none>" ]; then
               D=$(docker image inspect split-inference:spark --format "{{.Id}}");
             fi; echo "$D"'
  if [ -n "$1" ]; then ssh "$1" "$CMD"; else eval "$CMD"; fi
}
LOCAL_HEAD=$(git -C $HOME/dtraining rev-parse HEAD)
REMOTE_HEAD=$(ssh ucn 'git -C $HOME/dtraining rev-parse HEAD')
if [ "$LOCAL_HEAD" != "$REMOTE_HEAD" ]; then
  echo "PREFLIGHT FAIL: dtraining commit mismatch (tln=$LOCAL_HEAD ucn=$REMOTE_HEAD)"; exit 1
fi
LOCAL_IMG=$(img_digest)
REMOTE_IMG=$(img_digest ucn)
if [ -z "$LOCAL_IMG" ] || [ "$LOCAL_IMG" != "$REMOTE_IMG" ]; then
  echo "PREFLIGHT FAIL: split-inference:spark image mismatch (tln=$LOCAL_IMG ucn=$REMOTE_IMG)"; exit 1
fi
echo "[preflight] commit=$LOCAL_HEAD image=$LOCAL_IMG"

# --- ALWAYS restore WAN shaping on both nodes, whatever happens below.
cleanup () {
  $BIN/set_wan.sh off >/dev/null 2>&1 || $BIN/set_rtt.sh 0 >/dev/null 2>&1 || true
  ssh ucn "$BIN/set_wan.sh off >/dev/null 2>&1 || $BIN/set_rtt.sh 0 >/dev/null 2>&1" || true
}
trap cleanup EXIT

echo "=== REPRO2 INFERENCE $TAG START $(date -Is) ==="
ssh ucn "$BIN/start_server.sh cloud $MODEL $CLOUD_SPLIT" >/dev/null
wait_health $CLOUD/health

for PROFILE in 0 20 80 hostile; do
  case $PROFILE in
    hostile) $BIN/set_wan.sh hostile >/dev/null; ssh ucn "$BIN/set_wan.sh hostile >/dev/null";;
    *) $BIN/set_rtt.sh $PROFILE >/dev/null; ssh ucn "$BIN/set_rtt.sh $PROFILE >/dev/null";;
  esac
  M=$(ping -c 5 -i 0.3 10.10.10.2 | tail -1 | awk -F/ '{print $5}')
  for REP in 1 2 3; do
    echo "--- $TAG profile=$PROFILE rep=$REP (measured ${M}ms) $(date -Is)"
    $BIN/si python3 rtt_analysis.py --model /workspace/experiments/models/$MODEL \
      --cloud $CLOUD --split-after $SA --resume-at $RA \
      --output $OUTC/rtt_${PROFILE}ms_rep${REP}.json 2>&1 | tail -1
  done
done

echo "--- $TAG identity @80ms (x3 reps)"
$BIN/set_rtt.sh 80 >/dev/null; ssh ucn "$BIN/set_rtt.sh 80 >/dev/null"
for REP in 1 2 3; do
  $BIN/si python3 perplexity_comparison.py --model /workspace/experiments/models/$MODEL \
    --cloud $CLOUD --split-after $SA --resume-at $RA --max-tokens 200 \
    --output $OUTC/identity_80ms_rep${REP}.json 2>&1 | tail -2
done

echo "=== REPRO2 INFERENCE $TAG DONE $(date -Is) ==="
