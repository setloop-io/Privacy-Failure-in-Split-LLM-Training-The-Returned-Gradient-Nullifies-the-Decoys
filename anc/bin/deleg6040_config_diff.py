#!/usr/bin/env python3
"""AC1.2 / D2(a): prove that only `split_after` changed.

Flattens both runner artifacts and sorts every leaf into one of four classes:

  INTENDED     split_after, and nothing else may appear here.
  CONFIG       knobs and protocol state that MUST be identical. Any difference
               here means the cell answers a different question than the
               baseline, and is a stop condition (seed section 6).
  INVARIANT    not knobs, but split-independent by construction: the undefended
               baseline loss and the label-free majority control depend on the
               model, corpus window and chaff pool only. A difference here means
               the corpus or eval window moved, which would also invalidate the
               comparison.
  OUTCOME      measured results that are expected to move; reported, never gated.

A leaf that matches no rule falls through to CONFIG, which is the fail-safe
direction: an unrecognised field is gated rather than waved past, so a schema
addition cannot slip through unexamined. Nothing is silently ignored.

Membership of OUTCOME is therefore the only place a real difference could hide,
and every entry in it must be a measured result rather than a CLI knob. Checked
against the runner's argument parser: none of the fields below is settable from
the command line.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

INTENDED = {"split_after"}

# Split-independent by construction, not a tunable knob. `baseline_eval_loss` is
# the undefended model on the eval window and is the strongest available check
# that the corpus window did not move: it reads 4.977673412300646 in every 0.6B
# cell in the campaign, across every split, latent_dim, chaff and cloud kind
# (v6 split-8/split-20, v13 a1/a2/a9). NOTE: label_free_majority_pct is NOT in
# this set -- it reads 5.127 / 5.195 / 5.376 across v13 a1/a2/a9, which share
# corpus, chaff and split_after, so it tracks the released-row sample and is an
# outcome, not an invariant.
INVARIANT = {
    "baseline_eval_loss",
    "hidden_dim",
    "model",
}

# Measured outcomes: expected to move when the boundary moves.
OUTCOME_PREFIXES = (
    "candidate_eval_loss", "candidate_eval_seconds", "baseline_eval_seconds",
    "eval_loss_delta", "eval_time_ratio", "zero_cloud_eval_loss",
    "cloud_correction_loss_improvement", "train_seconds", "mean_step_seconds",
    "tln_peak_cuda_memory_mb", "probe_recovery_pct", "token_recovery_pct",
    "recovery_above_label_free_pct", "mine_mi_nats_held_out", "training_tail",
    "label_free_majority_pct",
    "gates/", "limitations", "status",
    "remote_protocol/session_id",          # fresh per connection
    "byzantine_max_relative_deviation", "byzantine_verified_frames",
    "byzantine_flagged_frames", "remote_tampered_frames",
)
# NOT in OUTCOME, though it looks like a measurement: bundle_canonical_fraction
# is a real CLI knob (--bundle-canonical-fraction, "also store the pre-gauge
# canonical latent rows"). Classifying it as an outcome would let pre-gauge rows
# be added to the released bundle without failing AC1.2, which would invalidate
# any attacker result computed from that bundle. It stays in CONFIG.


def flatten(obj: object, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten(value, f"{prefix}/{key}" if prefix else key))
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj)
    else:
        out[prefix] = obj
    return out


def classify(key: str) -> str:
    if key in INTENDED:
        return "INTENDED"
    if key in INVARIANT:
        return "INVARIANT"
    if key.startswith(OUTCOME_PREFIXES):
        return "OUTCOME"
    return "CONFIG"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    base = flatten(json.loads(Path(args.baseline).read_text()))
    cell = flatten(json.loads(Path(args.cell).read_text()))

    rows = []
    for key in sorted(set(base) | set(cell)):
        b, c = base.get(key, "<absent>"), cell.get(key, "<absent>")
        rows.append({"field": key, "class": classify(key),
                     "baseline": b, "cell": c, "same": b == c})

    changed = [r for r in rows if not r["same"]]
    config_changed = [r for r in changed if r["class"] == "CONFIG"]
    invariant_changed = [r for r in changed if r["class"] == "INVARIANT"]
    intended_changed = [r for r in changed if r["class"] == "INTENDED"]

    verdict = {
        "schema": "dtraining.deleg6040.config_diff.v1",
        "baseline_file": args.baseline,
        "cell_file": args.cell,
        "n_fields": len(rows),
        "intended_changes": intended_changed,
        "config_violations": config_changed,
        "invariant_violations": invariant_changed,
        "outcome_changes": [r for r in changed if r["class"] == "OUTCOME"],
        "unchanged_config_fields": sorted(
            r["field"] for r in rows if r["same"] and r["class"] == "CONFIG"),
        "ac1_2_pass": (not config_changed and not invariant_changed
                       and {r["field"] for r in intended_changed} == {"split_after"}),
    }
    Path(args.output).write_text(json.dumps(verdict, indent=2) + "\n")

    print(f"fields compared      : {verdict['n_fields']}")
    print(f"INTENDED changes     : "
          f"{[r['field'] for r in intended_changed]}")
    print(f"CONFIG violations    : {len(config_changed)}")
    for r in config_changed:
        print(f"  ! {r['field']}: {r['baseline']!r} -> {r['cell']!r}")
    print(f"INVARIANT violations : {len(invariant_changed)}")
    for r in invariant_changed:
        print(f"  ! {r['field']}: {r['baseline']!r} -> {r['cell']!r}")
    print(f"OUTCOME changes      : {len(verdict['outcome_changes'])}")
    print(f"AC1.2 PASS           : {verdict['ac1_2_pass']}")
    return 0 if verdict["ac1_2_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
