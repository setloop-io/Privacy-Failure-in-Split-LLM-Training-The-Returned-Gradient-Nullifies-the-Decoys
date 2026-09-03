#!/usr/bin/env bash
# Stage E5 (v9.3 candidate): public-data cloud pretraining, then a HALVED
# private phase.  Claim under test: a warm-started cloud reaches the same
# utility with fewer private steps (fleet does the bulk compute on public
# data; privacy surface unchanged).  Config: the current v9.2 winner.
set -euo pipefail
BIN=$HOME/dtraining/bin
PUBLIC=/workspace/experiments/models/wikitext2_public.txt

# Build the disjoint public corpus once: skip the first 200KB of the private
# corpus (private split uses only the first 512 blocks x 33 tokens).
docker run --rm -v $HOME/experiments:/workspace/experiments split-inference:spark \
  sh -c "tail -c +200000 /workspace/experiments/models/wikitext2_corpus.txt > $PUBLIC && wc -c $PUBLIC"

# E5 cell: 4000 public steps + 1000 private steps on the v9.2 winner
# (D=64, chaff-48, radial MoE E=8, K=2 channels).  Compare utility against
# the 2000-private-step v9.2 reference (loss delta +0.073).
bash "$BIN/latent_v9_cell.sh" v93_public4k_priv1k_d64_06b \
  /workspace/experiments/models/qwen3-0.6b 21 26 64 0.35 48 32 \
  monomial_moe_radial 8 2 5025 0.0 1000 0 2 4000

echo "STAGE_E5_COMPLETE"
