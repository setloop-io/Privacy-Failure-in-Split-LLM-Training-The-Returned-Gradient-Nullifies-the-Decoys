#!/usr/bin/env bash
# Containment + corpus precheck. Run ON THE TRUSTED NODE (odysseus) before every cell.
#
# The seed's invariant 1 requires re-verifying before each cell that `tln` and
# `ucn` -- a third party's machines -- are pinned to 127.0.0.1 on BOTH nodes and
# that ssh to them is refused. This script asserts that and the corpus identity,
# and writes a machine-readable receipt. It exits non-zero rather than warning, so
# a cell wrapper can gate on it.
#
# It reaches the untrusted node only as `poseidon.cluster` (the direct link,
# 192.168.100.11), never as `ucn`.
set -euo pipefail

CLOUD_HOST=${CLOUD_HOST:-poseidon.cluster}
CORPUS=${CORPUS:-$HOME/experiments/models/wikitext2_corpus.txt}
EXPECT_SHA=${EXPECT_SHA:-78b6bfb90cfd718f0c27d42b1fd2231b139d1dda75d7d796e6a603b2e5cd7efe}
OUT=${OUT:-$HOME/experiments/results/training/deleg6040/d0_precheck.json}

remote () { ssh -o BatchMode=yes -o ForwardAgent=no "$CLOUD_HOST" "$@"; }

fail=0
note () { echo "$1"; }

# containment, both nodes
local_ucn=$(getent hosts ucn | awk '{print $1}' | head -1)
local_tln=$(getent hosts tln | awk '{print $1}' | head -1)
remote_ucn=$(remote "getent hosts ucn | awk '{print \$1}' | head -1")
remote_tln=$(remote "getent hosts tln | awk '{print \$1}' | head -1")

for pair in "trusted/ucn:$local_ucn" "trusted/tln:$local_tln" \
            "untrusted/ucn:$remote_ucn" "untrusted/tln:$remote_tln"; do
  name=${pair%%:*}; addr=${pair##*:}
  if [ "$addr" != "127.0.0.1" ]; then
    note "CONTAINMENT BROKEN: $name resolves to '$addr', expected 127.0.0.1"
    fail=1
  fi
done

ssh_ucn=$(ssh -o BatchMode=yes -o ConnectTimeout=5 -o ForwardAgent=no ucn true 2>&1 || true)
case "$ssh_ucn" in
  *"Connection refused"*) ;;
  *) note "CONTAINMENT BROKEN: ssh ucn did not refuse: $ssh_ucn"; fail=1 ;;
esac

# corpus identity
corpus_sha=$(sha256sum "$CORPUS" | awk '{print $1}')
if [ "$corpus_sha" != "$EXPECT_SHA" ]; then
  note "CORPUS MISMATCH: $corpus_sha != $EXPECT_SHA"
  fail=1
fi

mkdir -p "$(dirname "$OUT")"
cat > "$OUT" <<JSON
{
  "schema": "dtraining.deleg6040.precheck.v1",
  "utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "trusted_node": "$(hostname)",
  "untrusted_node_alias": "$CLOUD_HOST",
  "containment": {
    "trusted": {"ucn": "$local_ucn", "tln": "$local_tln"},
    "untrusted": {"ucn": "$remote_ucn", "tln": "$remote_tln"},
    "ssh_ucn_result": "$(echo "$ssh_ucn" | tr -d '"' | head -1)"
  },
  "corpus": {"path": "$CORPUS", "sha256": "$corpus_sha", "expected": "$EXPECT_SHA",
             "match": $( [ "$corpus_sha" = "$EXPECT_SHA" ] && echo true || echo false )},
  "pass": $( [ "$fail" -eq 0 ] && echo true || echo false )
}
JSON

cat "$OUT"
[ "$fail" -eq 0 ] || { echo "PRECHECK FAILED"; exit 1; }
echo "PRECHECK OK"
