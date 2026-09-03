#!/usr/bin/env python3
"""Phase C4 aggregation: per-cell paired statistics over the phasec captures.

For every completed cell in the phasec results tree, pairs each scored arm's
prediction dump against the constant control on the same rows (paired,
frame-clustered, bootstrapped -- bin/paired_advantage.py, the W2.1a statistic),
alongside the frozen-gate upper-95 excess readings, the support-leak account,
and the utility delta. Emits one summary JSON.

Usage (cluster, from the packaged tree):
  python3 bin/phasec_paired_summary.py --results /workspace/experiments/results/training/phasec
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paired_advantage import analyse  # noqa: E402  (bin/ is the import root)


def cell_summary(results: Path, cell: str) -> dict:
    import torch
    entry = {"cell": cell}
    main = json.loads((results / f"{cell}.json").read_text())
    entry["utility_delta"] = main.get("eval_loss_delta")
    entry["gates"] = main.get("gates")
    fwd = json.loads((results / f"{cell}_attacker.json").read_text())
    entry["forward_excess_pp"] = fwd["summary"][0][
        "upper95_excess_over_majority_pp"]
    bundles = json.loads((results / f"{cell}_bundles.json").read_text())
    entry["support_leak"] = bundles["support_leak"]
    arms = {}
    for arm_json in sorted(results.glob(f"{cell}_arm_*.json")):
        arm = arm_json.stem.replace(f"{cell}_arm_", "")
        gate = json.loads(arm_json.read_text())["summary"][0]
        record = {"frozen_gate_excess_pp":
                  gate["upper95_excess_over_majority_pp"]}
        dump = results / "bundles" / cell / f"{arm}_pred.pt"
        if dump.exists():
            paired = analyse(torch.load(dump, map_location="cpu",
                                        weights_only=False), 2000, 42)
            best = paired.get("best_eligible") or {}
            record["paired_best_pp"] = best.get("paired_advantage_pp")
            record["paired_best_arm"] = best.get("model")
            record["paired_best_ci95"] = [best.get("ci95_low_pp"),
                                          best.get("ci95_high_pp")]
            record["paired_verdict"] = paired.get("verdict")
        arms[arm] = record
    entry["arms"] = arms
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    cells = sorted(p.stem.removesuffix("_attacker")
                   for p in args.results.glob("*_attacker.json"))
    summaries = [cell_summary(args.results, cell) for cell in cells]
    out = {"schema": "dtraining.phasec_paired_summary.v1",
           "n_cells": len(summaries), "cells": summaries}
    text = json.dumps(out, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
