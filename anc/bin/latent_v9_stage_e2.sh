#!/usr/bin/env bash
# Stage E2: sequence-length scaling (more rows/context per frame) with
# chaff-48 at the best D from stage E1 (D=16 reference cells on 0.6B).
set -euo pipefail
BIN=$HOME/dtraining/bin

bash "$BIN/latent_v9_cell.sh" e2_seq64_06b /workspace/experiments/models/qwen3-0.6b \
  21 26 16 0.35 48 64 monomial 1 2 5013 0.0 2000
bash "$BIN/latent_v9_cell.sh" e2_seq128_06b /workspace/experiments/models/qwen3-0.6b \
  21 26 16 0.35 48 128 monomial 1 2 5013 0.0 2000

echo "STAGE_E2_COMPLETE"
