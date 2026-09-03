#!/usr/bin/env bash
# Stage E3 (v9.1 candidate): radial+Gram MoE experts — no token-scale gauge.
# E2 failed and was dropped; v9.0 combination = E1 (D=64, chaff-48).
# Two cells for attribution: radial at D=16 (isolate scale-gauge effect) and
# at D=64 (the E1 winner; the v9.1 combination candidate).
set -euo pipefail
BIN=$HOME/dtraining/bin

bash "$BIN/latent_v9_cell.sh" v91_radial_d16_06b /workspace/experiments/models/qwen3-0.6b \
  21 26 16 0.35 48 32 monomial_moe_radial 8 2 5023 0.0 2000 0
bash "$BIN/latent_v9_cell.sh" v91_radial_d64_06b /workspace/experiments/models/qwen3-0.6b \
  21 26 64 0.35 48 32 monomial_moe_radial 8 2 5025 0.0 2000 0

echo "STAGE_E3_COMPLETE"
