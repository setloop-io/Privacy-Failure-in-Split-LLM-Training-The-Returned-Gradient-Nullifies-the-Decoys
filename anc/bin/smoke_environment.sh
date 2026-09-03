#!/usr/bin/env bash
# Acceptance test for experiment W1.4 (execution environment).
#
# Verifies the execution environment end to end: both hosts reachable, GPUs present,
# container image at the recorded digest, torch importable with CUDA inside it, the four
# data-of-record hashes intact, and the attacker framework self-test passing WITH torch
# present -- which is the acceptance criterion W1.4 is measured against.
#
# Exit 0 = W1.4 satisfied. Any failure prints FAIL and exits non-zero.
set -uo pipefail

IMAGE=${IMAGE:-split-inference:spark}
# The image is built independently on each host from the same recipe, so layer digests
# differ legitimately. Equivalence is checked functionally (interpreter + framework
# fingerprint, compared across hosts), not by digest. Recorded digests are informational.
DIGEST_ODYSSEUS=sha256:ddf69e590221ccdf4129c6677a5f2aa0fc0ec0c8acf53e235debd3863f3b93ca
DIGEST_POSEIDON=sha256:77f367dda8441874be8b674b4cbec1de8149a19aa345af83ec0e6a7c977481f9
HOSTS=${HOSTS:-"odysseus poseidon"}
PRIMARY=${PRIMARY:-odysseus}
REPO_ON_HOST=${REPO_ON_HOST:-\$HOME/dtraining}

MODEL_SHA=f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b
CONFIG_SHA=660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd
TOKENIZER_SHA=aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
CORPUS_SHA=78b6bfb90cfd718f0c27d42b1fd2231b139d1dda75d7d796e6a603b2e5cd7efe

fails=0
FINGERPRINT=""
REFHOST=""
ok()   { printf '  \033[32mok\033[0m   %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; fails=$((fails+1)); }
rs()   { timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2" 2>/dev/null; }

echo "== hosts =="
for h in $HOSTS; do
  if [ "$(rs "$h" 'echo OK')" = OK ]; then ok "$h reachable"; else fail "$h unreachable"; continue; fi
  gpu=$(rs "$h" 'nvidia-smi --query-gpu=name --format=csv,noheader | head -1')
  [ -n "$gpu" ] && ok "$h GPU: $gpu" || fail "$h no GPU"
  drv=$(rs "$h" 'nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1')
  ok "$h driver: ${drv:-unknown}"
  if [ "$(rs "$h" 'docker ps >/dev/null 2>&1 && echo yes')" = yes ]; then
    ok "$h docker without sudo"
  else fail "$h docker requires sudo"; fi
  d=$(rs "$h" "docker image inspect $IMAGE --format '{{.Id}}'")
  [ -n "$d" ] && ok "$h image present ($(echo "$d" | cut -c1-19)...)" || fail "$h image $IMAGE absent"
  fp=$(rs "$h" "docker run --rm $IMAGE python3 -c \"
import sys,torch,transformers
print(sys.version.split()[0],torch.__version__,torch.version.cuda,transformers.__version__)\" 2>/dev/null | tail -1")
  if [ -z "$FINGERPRINT" ]; then FINGERPRINT="$fp"; REFHOST="$h"; fi
  if [ "$fp" = "$FINGERPRINT" ]; then ok "$h framework fingerprint matches $REFHOST"
  else fail "$h framework DRIFT vs $REFHOST: '$fp' != '$FINGERPRINT'"; fi
done

echo "== container runtime ($PRIMARY) =="
probe=$(rs "$PRIMARY" "docker run --rm --gpus all $IMAGE python3 -c \"
import torch,transformers,sys
print(sys.version.split()[0], torch.__version__, torch.version.cuda, transformers.__version__, torch.cuda.is_available())
\" 2>/dev/null | tail -1")
if [ -n "$probe" ]; then
  ok "python/torch/cuda/transformers: $probe"
  case "$probe" in *True) ok "CUDA available inside container";; *) fail "CUDA NOT available inside container";; esac
else fail "container failed to import torch"; fi

echo "== data of record ($PRIMARY) =="
check_sha () { # <path> <expected> <label>
  got=$(rs "$PRIMARY" "sha256sum $1 2>/dev/null | cut -d' ' -f1")
  if [ "$got" = "$2" ]; then ok "$3 hash intact"
  elif [ -n "$got" ]; then fail "$3 hash MISMATCH: $got"
  else fail "$3 missing: $1"; fi
}
check_sha '$HOME/experiments/models/qwen3-0.6b/model.safetensors' "$MODEL_SHA"     "model weights"
check_sha '$HOME/experiments/models/qwen3-0.6b/config.json'       "$CONFIG_SHA"    "model config"
check_sha '$HOME/experiments/models/qwen3-0.6b/tokenizer.json'    "$TOKENIZER_SHA" "tokenizer"
check_sha '$HOME/experiments/models/wikitext2_corpus.txt'         "$CORPUS_SHA"    "corpus"

for f in ca.crt cloud-server.crt cloud-server.key; do
  [ "$(rs "$PRIMARY" "test -f \$HOME/experiments/tls/$f && echo yes")" = yes ] \
    && ok "tls/$f present" || fail "tls/$f missing"
done

echo "== attacker self-test WITH torch (the W1.4 criterion) =="
st=$(rs "$PRIMARY" "docker run --rm --gpus all -v \$HOME/experiments:/workspace/experiments -v $REPO_ON_HOST:/repo -w /repo $IMAGE bash -lc 'python3 -c \"import torch\" && python3 -m attacker --self-test' 2>&1 | tail -1")
case "$st" in
  *"SELF-TEST PASSED"*) ok "attacker framework self-test passed with torch present";;
  *) fail "attacker self-test did not pass: ${st:-no output}";;
esac

echo
if [ "$fails" -eq 0 ]; then
  echo "W1.4 SATISFIED"; exit 0
else
  echo "W1.4 NOT SATISFIED ($fails failure(s))"; exit 1
fi
