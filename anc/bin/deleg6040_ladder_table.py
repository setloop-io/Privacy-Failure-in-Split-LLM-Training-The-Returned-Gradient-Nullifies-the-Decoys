#!/usr/bin/env python3
"""Assemble the delegation dose-response table from committed artifacts.

Reports, per split point, three views of the same nine arms so the headline is
not resting on one statistic:

  gated excess    the campaign metric: worst-of-nine Bonferroni-Wilson upper
                  bound minus the label-free majority control. This is the
                  number the +1.0 pp gate applies to, and the number that is
                  vulnerable to the objection that it takes a MAX over nine arms.

  raw lift        best arm's raw top-1 minus the majority control. No confidence
                  bound, so no floor inflation -- but still a max.

  invariant-arm   mean raw top-1 of the six invariant-family arms minus the
  mean lift       majority control. NO selection at all: a fixed, pre-declared
                  subset averaged over all its restarts. If this moves
                  monotonically with delegation share, the effect cannot be a
                  max-selection artifact.

It also decomposes the excess into how much came from recovery RISING versus
the majority control FALLING, because a drop in the control inflates the excess
without any attacker improving.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

TOTAL_LAYERS = 28
RESUME_AFTER = 26
INVARIANT_ARMS = ("invariant_only", "invariant_graph")


def summarise(runner_path: Path, attacker_path: Path) -> dict:
    run = json.loads(runner_path.read_text())
    att = json.loads(attacker_path.read_text())
    summary = att["summary"][0]

    split_after = run["split_after"]
    delegated = run["resume_after"] - split_after - 1
    majority = summary["label_free_majority_pct"]

    arms = att["results"]
    best = max(arms, key=lambda a: a["top1_pct"])
    inv = [a for a in arms if a["model"] in INVARIANT_ARMS]
    inv_mean = sum(a["top1_pct"] for a in inv) / len(inv)
    coord = [a for a in arms if a["model"] not in INVARIANT_ARMS]
    coord_mean = sum(a["top1_pct"] for a in coord) / len(coord)

    return {
        "split_after": split_after,
        "delegated_layers": delegated,
        "delegated_pct": 100.0 * delegated / TOTAL_LAYERS,
        "gated_excess_pp": summary["upper95_excess_over_majority_pp"],
        "gate_pass": summary["upper95_excess_over_majority_pp"] <= 1.0,
        "majority_pct": majority,
        "best_arm_top1_pct": best["top1_pct"],
        "best_arm_model": best["model"],
        "raw_lift_pp": best["top1_pct"] - majority,
        "invariant_mean_top1_pct": inv_mean,
        "invariant_mean_lift_pp": inv_mean - majority,
        "coordinate_mean_top1_pct": coord_mean,
        "coordinate_mean_lift_pp": coord_mean - majority,
        "eval_loss_delta": run["eval_loss_delta"],
        "utility_gate_pass": run["gates"]["utility_delta_le_0_35"],
        "eval_time_ratio": run["eval_time_ratio"],
        "in_run_probe_excess_pp": run["recovery_above_label_free_pct"],
        "in_run_privacy_gate": run["gates"]["privacy_above_band_le_1pct"],
        "cloud_latent_only": run["gates"]["cloud_latent_only"],
        "cloud_state_contains_hidden_width": run["cloud_state_contains_hidden_width"],
        "state_shape_digest": run["remote_protocol"]["state_shape_digest"],
        "cloud_correction_loss_improvement": run["cloud_correction_loss_improvement"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.dir)
    rows = []
    for runner in sorted(root.glob("*.json")):
        if runner.name.endswith("_attacker.json") or "_split" not in runner.name:
            continue
        attacker = runner.with_name(runner.stem + "_attacker.json")
        if not attacker.exists():
            continue
        rows.append(summarise(runner, attacker))

    rows.sort(key=lambda r: r["delegated_pct"])

    # Decompose the change in gated excess relative to the lowest-delegation row.
    ref = rows[0]
    for r in rows:
        r["delta_excess_vs_ref_pp"] = r["gated_excess_pp"] - ref["gated_excess_pp"]
        r["from_recovery_rising_pp"] = r["best_arm_top1_pct"] - ref["best_arm_top1_pct"]
        r["from_majority_falling_pp"] = ref["majority_pct"] - r["majority_pct"]

    digests = {r["state_shape_digest"] for r in rows}
    report = {
        "schema": "dtraining.deleg6040.ladder.v1",
        "reference_row_split_after": ref["split_after"],
        "rows": rows,
        "state_shape_digest_identical_across_ladder": len(digests) == 1,
        "state_shape_digest": sorted(digests),
        "gated_excess_monotone_in_delegation": all(
            rows[i]["gated_excess_pp"] <= rows[i + 1]["gated_excess_pp"]
            for i in range(len(rows) - 1)),
        "invariant_mean_lift_monotone_in_delegation": all(
            rows[i]["invariant_mean_lift_pp"] <= rows[i + 1]["invariant_mean_lift_pp"]
            for i in range(len(rows) - 1)),
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")

    hdr = (f"{'split':>6} {'deleg':>7} {'layers':>7} {'excess pp':>10} {'gate':>6} "
           f"{'majority':>9} {'best top1':>10} {'raw lift':>9} "
           f"{'inv-mean lift':>14} {'coord lift':>11} {'util d':>8} {'util':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['split_after']:>6} {r['delegated_pct']:>6.1f}% "
              f"{r['delegated_layers']:>7} {r['gated_excess_pp']:>+10.4f} "
              f"{'PASS' if r['gate_pass'] else 'FAIL':>6} "
              f"{r['majority_pct']:>8.4f}% {r['best_arm_top1_pct']:>9.4f}% "
              f"{r['raw_lift_pp']:>+9.4f} {r['invariant_mean_lift_pp']:>+14.4f} "
              f"{r['coordinate_mean_lift_pp']:>+11.4f} "
              f"{r['eval_loss_delta']:>+8.3f} "
              f"{'PASS' if r['utility_gate_pass'] else 'FAIL':>5}")
    print()
    print("Decomposition of the gated-excess change vs the "
          f"split_after {ref['split_after']} row:")
    for r in rows:
        print(f"  split {r['split_after']:>2} ({r['delegated_pct']:>4.1f}%): "
              f"delta excess {r['delta_excess_vs_ref_pp']:>+7.4f} pp  = "
              f"recovery rising {r['from_recovery_rising_pp']:>+7.4f} pp  "
              f"+ majority falling {r['from_majority_falling_pp']:>+7.4f} pp")
    print()
    print(f"state_shape_digest identical across ladder : "
          f"{report['state_shape_digest_identical_across_ladder']}")
    print(f"gated excess monotone in delegation        : "
          f"{report['gated_excess_monotone_in_delegation']}")
    print(f"invariant-arm mean lift monotone           : "
          f"{report['invariant_mean_lift_monotone_in_delegation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
