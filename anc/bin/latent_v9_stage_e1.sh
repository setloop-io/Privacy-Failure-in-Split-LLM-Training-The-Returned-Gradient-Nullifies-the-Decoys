#!/usr/bin/env bash
# Stage E1: boundary-width frontier with the v7.1 defenses (chaff-48).
# 0.6B, monomial cloud, D=32 and D=64. Gate: privacy must hold at each width.
set -euo pipefail
BIN=$HOME/dtraining/bin

bash "$BIN/latent_v9_cell.sh" e1_d32_06b /workspace/experiments/models/qwen3-0.6b \
  21 26 32 0.35 48 32 monomial 1 2 5021 0.0 2000
bash "$BIN/latent_v9_cell.sh" e1_d64_06b /workspace/experiments/models/qwen3-0.6b \
  21 26 64 0.35 48 32 monomial 1 2 5022 0.0 2000

echo "STAGE_E1_COMPLETE"
