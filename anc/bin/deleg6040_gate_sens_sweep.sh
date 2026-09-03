#!/usr/bin/env bash
# Dose-response sweeps for bin/deleg6040_gate_sensitivity.py.
#
# Runs INSIDE the split-inference:spark container. Each sweep is one
# invocation of the tool over a list of doses; every dose writes an injected
# bundle, re-scores the frozen nine-arm attacker unmodified, and re-derives the
# gate from raw counts.
#
#   A  coordinate mode, amplitude 1.0, coverage sweep   (11 doses)
#   B  coordinate mode, coverage 1.0, amplitude sweep   (6 doses)
#   C  invariant  mode, amplitude 1.0, coverage sweep   (4 doses)
#   D  invariant  mode, coverage 1.0, amplitude sweep   (2 doses)
#   E  coordinate mode, amplitude 1.0, coverage refinement between A's
#      0.04 and 0.06 rungs, where A brackets the detection threshold (3 doses)
#   F  coordinate mode, coverage 1.0, amplitude refinement between B's
#      0.10 and 0.15 rungs, where B brackets the threshold (3 doses)
#   G  invariant  mode, amplitude 1.0, coverage refinement between C's
#      0.25 and 0.50 rungs, where C brackets the threshold (3 doses)
#
# Sweep A's first dose is coverage 0, which must reproduce the recorded
# attacker artifact exactly; the tool aborts if it does not.
set -euo pipefail

SWEEP=${1:?usage: deleg6040_gate_sens_sweep.sh A|B|C|D}
BUNDLEDIR=${BUNDLEDIR:-/workspace/experiments/results/training/deleg6040/bundles}
BUNDLE=${BUNDLE:-$BUNDLEDIR/deleg_6040_conv10k_split14_bundle.pt}
WORK=${WORK:-/workspace/experiments/results/training/deleg6040/gate_sens}
RECORDED=${RECORDED:-paper-data/collected/diagnostic/deleg_60_40/deleg_6040_conv10k_split14_attacker.json}
TOOL=${TOOL:-bin/deleg6040_gate_sensitivity.py}

# REUSE=1 re-derives every report from the frozen-attacker artifacts already on
# disk instead of scoring again. The injected bundles are rewritten and their
# sha256 recorded, so a changed injection cannot pass unnoticed.
mkdir -p "$WORK"
common=(python3 "$TOOL" --bundle "$BUNDLE" --workdir "$WORK")
[ "${REUSE:-0}" = "1" ] && common+=(--reuse)

case "$SWEEP" in
  A) "${common[@]}" --mode coordinate --amplitude 1.0 \
       --coverage 0 0.0025 0.005 0.01 0.02 0.03 0.04 0.06 0.10 0.30 1.0 \
       --recorded-attacker-json "$RECORDED" \
       --output "$WORK/gate_sens_A_coordinate_coverage.json" ;;
  B) "${common[@]}" --mode coordinate --coverage 1.0 \
       --amplitude 0.05 0.10 0.15 0.20 0.30 0.50 \
       --output "$WORK/gate_sens_B_coordinate_amplitude.json" ;;
  C) "${common[@]}" --mode invariant --amplitude 1.0 \
       --coverage 0.05 0.25 0.50 1.0 \
       --output "$WORK/gate_sens_C_invariant_coverage.json" ;;
  D) "${common[@]}" --mode invariant --coverage 1.0 \
       --amplitude 0.25 0.50 \
       --output "$WORK/gate_sens_D_invariant_amplitude.json" ;;
  E) "${common[@]}" --mode coordinate --amplitude 1.0 \
       --coverage 0.045 0.05 0.055 \
       --output "$WORK/gate_sens_E_coordinate_refine.json" ;;
  F) "${common[@]}" --mode coordinate --coverage 1.0 \
       --amplitude 0.11 0.12 0.13 \
       --output "$WORK/gate_sens_F_coordinate_amp_refine.json" ;;
  G) "${common[@]}" --mode invariant --amplitude 1.0 \
       --coverage 0.30 0.35 0.40 \
       --output "$WORK/gate_sens_G_invariant_refine.json" ;;
  *) echo "unknown sweep $SWEEP" >&2; exit 2 ;;
esac
