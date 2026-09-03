#!/usr/bin/env bash
# Canonical latent cloud server launcher: every cell starts the server through
# this script rather than an inline docker run.
#
# usage: latent_cloud_launch.sh <name> <latent_dim> <cloud_kind> <port> [extra server args...]
# example: latent_cloud_launch.sh latent-cloud 64 monomial_moe_radial 5025 --cloud-experts 8
set -euo pipefail

NAME="$1"; LDIM="$2"; KIND="$3"; PORT="$4"; shift 4
TLS_DIR="${TLS_DIR:-/workspace/experiments/tls}"
TLS_ARGS=(--tls-cert "$TLS_DIR/ucn-server.crt" --tls-key "$TLS_DIR/ucn-server.key")

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --network host --ipc host \
  -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -v "$HOME/experiments:/workspace/experiments" \
  -v "$HOME/dtraining:$HOME/dtraining" -w "$HOME/dtraining" \
  split-inference:spark \
  python3 split-training/latent_cloud_server.py \
    --latent-dim "$LDIM" --forbidden-hidden-dim 1024 \
    --cloud-kind "$KIND" --port "$PORT" "${TLS_ARGS[@]}" "$@"
sleep 3
docker logs "$NAME" 2>&1 | tail -1
