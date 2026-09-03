#!/usr/bin/env bash
# Stage E4 (v9.2 candidate): K=2 parallel cloud channels.
# Reference at D=16 (isolate the channel effect) + combination at D=64
# (E1 winner, chaff-48). Monomial cloud, scale gauge ON.
set -euo pipefail
BIN=$HOME/dtraining/bin

bash "$BIN/latent_v9_cell.sh" v92_k2_d16_06b /workspace/experiments/models/qwen3-0.6b \
  21 26 16 0.35 48 32 monomial 1 2 5013 0.0 2000 1 2
bash "$BIN/latent_v9_cell.sh" v92_k2_d64_06b /workspace/experiments/models/qwen3-0.6b \
  21 26 64 0.35 48 32 monomial 1 2 5022 0.0 2000 1 2
# v9.2 combination: v9.1 (D=64 + radial MoE, scale gauge off) + K=2 channels
bash "$BIN/latent_v9_cell.sh" v92_k2_radial_d64_06b /workspace/experiments/models/qwen3-0.6b \
  21 26 64 0.35 48 32 monomial_moe_radial 8 2 5025 0.0 2000 0 2

echo "STAGE_E4_COMPLETE"
