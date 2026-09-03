#!/usr/bin/env bash
# Experiment W2.4: a leakage-injection dose-response curve per emitting metric.
#
# token_top1 already has its 32-dose curve (deleg6040_gate_sensitivity.py on
# the deleg_6040 bundle). This driver produces the curve for ALL FOUR metrics
# W2.3 emits -- token_top1, rare_token_top1, token_cross_entropy,
# membership_auc -- on one retained bundle, the E1 packaged seed 44
# (e1_repro_w12_s44, the cleanest packaged cell: no duplicate-run history).
#
# Each dose: inject -> score with --dump-eval-probabilities -> leakage_metrics.
# One attacker run per dose yields all four curves at once.
#
# AXIS SCOPE, stated honestly: this is axis-1 of the W2.4 matrix --
#   dose (coverage x amplitude) x token frequency (the rare_token_top1 curve
#   is itself the frequency axis). Frame length and training size need
#   DIFFERENT bundles (seq-len and steps are training-time choices, not
#   dosing choices); those are axis-2 and are recorded as deferred, not
#   silently dropped. Every dose also writes the token_top1 gate reading so
#   the new curves sit next to the published one.
set -euo pipefail

BUNDLE=${BUNDLE:-/workspace/experiments/results/training/e1_unprotected/bundles/e1_repro_w12_s44_bundle.pt}
CELL_SEED=${CELL_SEED:-44}
WORK=${WORK:-/workspace/experiments/results/training/w24_metric_sweep}
OUT=${OUT:-/workspace/experiments/results/training/w24_metric_sweep}
IMAGE=${IMAGE:-split-inference:spark}
PKG=${PKG:-$HOME/dtraining-packaged}
TOOL=bin/deleg6040_gate_sensitivity.py

mkdir -p "$WORK"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

dose() { # tag mode coverage amplitude
  local tag=$1 mode=$2 coverage=$3 amplitude=$4
  # Spec.tag() inside the tool is {mode}_rho{coverage:g}_a{amplitude:g} --
  # Python's %g strips trailing zeros (1.0 -> "1"), and the injected bundle
  # follows that name, not ours.
  local cov_g amp_g spec_tag
  cov_g=$(python3 -c "print('%g' % float('$coverage'))")
  amp_g=$(python3 -c "print('%g' % float('$amplitude'))")
  spec_tag="${mode}_rho${cov_g}_a${amp_g}"
  local injected="$WORK/inj_${spec_tag}.pt"
  local artifact="$WORK/attacker_${tag}.json"
  local dump="$WORK/dump_${tag}.pt"
  local metrics="$OUT/metrics_${tag}.json"
  if [ -f "$metrics" ]; then log "$tag exists, skipping"; return 0; fi
  log "dose $tag mode=$mode coverage=$coverage amplitude=$amplitude"
  if [ "$coverage" = "0" ] || [ "$amplitude" = "0" ]; then
    # zero dose: write_injected saves nothing (injection is a no-op), so the
    # "injected" bundle is the source itself. The gate-sensitivity tool treats
    # this as the reproduce-the-recorded-gate control; here it is the control
    # dose the four curves are measured against.
    cp "$BUNDLE" "$injected"
  else
    python3 "$TOOL" --bundle "$BUNDLE" --workdir "$WORK" \
      --mode "$mode" --coverage "$coverage" --amplitude "$amplitude" \
      --seed 20260823 \
      --output "$WORK/sweep_${tag}.json" >/dev/null
  fi
  # the tool re-scores internally; re-score here with probabilities so the
  # dump carries what token_cross_entropy and membership_auc need
  python3 -m attacker --attack latent-probe \
    --bundle "$injected" --output "$artifact" \
    --dump-eval-probabilities "$dump" >/dev/null
  python3 bin/leakage_metrics.py --dump "$dump" \
    --arm injected_leak --attack forward_only --seed "$CELL_SEED" \
    --output "$metrics" >/dev/null
  # Per-dose tensors are ~12 GB (the invariant_graph arm's full per-row
  # probability tensor); 19 doses do not fit on the disk. The metric JSON, the
  # sweep accounting JSON, and the attacker artifact JSON are the record -- the
  # injected bundle and dumps are re-derivable from this script and the
  # committed seed, so they are removed once the metrics exist.
  rm -f "$injected" "$dump" "$WORK/inj_${spec_tag}_pred.pt"
  log "dose $tag metrics written"
}

# Axis 1a: coverage sweep at amplitude 1.0, coordinate mode (the published
# curve's family). Coverage 0 is the control dose.
for cov in 0 0.005 0.01 0.02 0.04 0.06 0.10 0.30 1.0; do
  dose "coord_cov${cov}" coordinate "$cov" 1.0
done
# Axis 1b: amplitude sweep at coverage 1.0, coordinate mode.
for amp in 0.05 0.10 0.15 0.20 0.30 0.50; do
  dose "coord_amp${amp}" coordinate 1.0 "$amp"
done
# Axis 1c: invariant mode, coverage sweep (the second injection family).
for cov in 0.05 0.25 0.50 1.0; do
  dose "inv_cov${cov}" invariant "$cov" 1.0
done

log "W2.4 AXIS-1 SWEEP COMPLETE"
