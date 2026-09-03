#!/usr/bin/env bash
# Stage E6 (v9.4 candidate): K=4 channels across two radial-D64 endpoints
# (5025 + 5027, stand-ins for separate datacenters; with a single untrusted
# host the multi-datacenter property follows from per-channel gauge
# independence and session isolation — documented as argued, not physically
# tested).  Config: v9.2 winner (D=64, chaff-48, radial MoE E=8).
set -euo pipefail
BIN=$HOME/dtraining/bin

bash "$BIN/latent_v9_cell.sh" v94_k4_2endpoints_d64_06b \
  /workspace/experiments/models/qwen3-0.6b 21 26 64 0.35 48 32 \
  monomial_moe_radial 8 2 5025 0.0 2000 0 4 0 "wss://ucn:5025,wss://ucn:5027"

echo "STAGE_E6_COMPLETE"
