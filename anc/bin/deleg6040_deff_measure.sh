#!/usr/bin/env bash
# Measure the design effect on every retained bundle whose dump exists.
#
# Reads only: the prediction dump, the bundle, the re-score that wrote the
# dump, the cell's committed *_attacker.json and *.json, the corpus and the
# tokenizer. bin/deleg6040_design_effect.py aborts unless the re-score
# reproduces the committed per-arm counts exactly.
#
#   BUNDLES="a b c" bash bin/deleg6040_deff_measure.sh
set -euo pipefail

TREE=${TREE:-$HOME/dtraining-deff}
OUT=${OUT:-$HOME/experiments/results/training/deleg6040/deff}
ARTIFACTS=${ARTIFACTS:-paper-data/collected/diagnostic/deleg_60_40}
IMAGE=${IMAGE:-split-inference:spark}
BOOTSTRAP=${BOOTSTRAP:-10000}
NULL_DRAWS=${NULL_DRAWS:-2000}
W=/workspace/experiments/results/training/deleg6040

for cell in $BUNDLES; do
  [ -f "$OUT/${cell}_pred.pt" ] || { echo "no dump for $cell yet"; continue; }
  if [ -f "$OUT/${cell}_design_effect.json" ]; then
    echo "=== $cell: measured already, skipping ==="
    continue
  fi
  echo "=== $cell: measure start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  docker run --rm --network host --ipc host \
    -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e CONTAINER_IMAGE="$IMAGE" \
    -v "$HOME/experiments:/workspace/experiments" \
    -v "$TREE:/workspace/dtraining-deff" -w /workspace/dtraining-deff \
    "$IMAGE" python3 bin/deleg6040_design_effect.py \
      --dump "$W/deff/${cell}_pred.pt" \
      --bundle "$W/bundles/${cell}_bundle.pt" \
      --rescore-json "$W/deff/${cell}_rescore.json" \
      --recorded-attacker-json "$ARTIFACTS/${cell}_attacker.json" \
      --run-json "$ARTIFACTS/${cell}.json" \
      --model /workspace/experiments/models/qwen3-0.6b \
      --corpus /workspace/experiments/models/wikitext2_corpus.txt \
      --bootstrap "$BOOTSTRAP" --null-draws "$NULL_DRAWS" \
      --output "$W/deff/${cell}_design_effect.json" \
    2>&1 | grep -v "^Copyright\|^All rights\|^NOTE:\|^  Using CUDA\|^  See http\|^Various files\|^GOVERNING\|^(found at\|^and the Product"
  echo "=== $cell: measure done  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
done
echo "ALL_MEASUREMENTS_DONE"
