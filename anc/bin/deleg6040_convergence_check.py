#!/usr/bin/env python3
"""Does encoder non-convergence explain the privacy jump, or does delegation?

The two hypotheses are confounded along the delegation axis, because utility
damage rises with delegation share. They are separable if any split point shows
one without the other. This prints, per split, the privacy reading beside every
convergence indicator the artifacts carry:

  eval_loss_delta            utility damage against the undefended baseline
  distill_loss (tail)        how far the surrogate is from the teacher middle
  language_loss (tail)       the LM objective at the last recorded step
  attacker_loss (tail)       the in-loop adversary's loss
  distill_loss slope         change over the recorded tail window; a still-
                             falling loss is the signature of "more steps would
                             have helped", which is the convergence hypothesis
  zero_cloud_eval_loss       loss with the cloud bypassed

The convergence hypothesis predicts privacy degrades with utility damage. If a
split shows large utility damage and a clean floor privacy reading, that
hypothesis does not survive in its simple form.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

TOTAL_LAYERS = 28


def tail_stats(run: dict) -> dict:
    tail = run.get("training_tail") or []
    if not tail:
        return {}
    first, last = tail[0], tail[-1]
    span = max(1, last.get("step", 0) - first.get("step", 0))
    out = {}
    for key in ("distill_loss", "language_loss", "attacker_loss"):
        if key in last:
            out[f"{key}_last"] = last[key]
            out[f"{key}_slope_per_1k"] = (
                1000.0 * (last[key] - first.get(key, last[key])) / span)
    out["tail_first_step"] = first.get("step")
    out["tail_last_step"] = last.get("step")
    return out


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
        run = json.loads(runner.read_text())
        att = json.loads(attacker.read_text())
        s = att["summary"][0]
        best = max(att["results"], key=lambda a: a["top1_pct"])
        p0 = s["label_free_majority_pct"] / 100.0
        n = best["total"]
        z = (best["correct"] - n * p0) / math.sqrt(n * p0 * (1 - p0))
        delegated = run["resume_after"] - run["split_after"] - 1
        rows.append({
            "split_after": run["split_after"],
            "steps": run["steps"],
            "delegated_layers": delegated,
            "delegated_pct": 100.0 * delegated / TOTAL_LAYERS,
            "excess_pp": s["upper95_excess_over_majority_pp"],
            "privacy_gate_pass": s["upper95_excess_over_majority_pp"] <= 1.0,
            "best_arm_binomial_z": z,
            "eval_loss_delta": run["eval_loss_delta"],
            "utility_gate_pass": run["gates"]["utility_delta_le_0_35"],
            "zero_cloud_eval_loss": run["zero_cloud_eval_loss"],
            **tail_stats(run),
        })
    rows.sort(key=lambda r: (r["delegated_pct"], r["steps"]))

    # A split with big utility damage but a floor privacy reading separates the
    # two hypotheses.
    failing_utility = [r for r in rows if not r["utility_gate_pass"]]
    dissociating = [r for r in failing_utility if r["privacy_gate_pass"]]

    report = {
        "schema": "dtraining.deleg6040.convergence_check.v1",
        "rows": rows,
        "splits_with_failed_utility_but_floor_privacy":
            [r["split_after"] for r in dissociating],
        "max_utility_damage_with_privacy_at_floor":
            max((r["eval_loss_delta"] for r in dissociating), default=None),
        "min_utility_damage_with_privacy_failed":
            min((r["eval_loss_delta"] for r in rows
                 if not r["privacy_gate_pass"]), default=None),
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")

    hdr = (f"{'split':>6} {'deleg':>7} {'steps':>7} {'excess':>8} {'z':>8} {'priv':>5} "
           f"{'util d':>8} {'util':>5} {'distill':>8} {'dist/1k':>9} "
           f"{'lang':>7} {'lang/1k':>9} {'zerocloud':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['split_after']:>6} {r['delegated_pct']:>6.1f}% "
              f"{r['steps']:>7} "
              f"{r['excess_pp']:>+8.4f} {r['best_arm_binomial_z']:>8.2f} "
              f"{'PASS' if r['privacy_gate_pass'] else 'FAIL':>5} "
              f"{r['eval_loss_delta']:>+8.3f} "
              f"{'PASS' if r['utility_gate_pass'] else 'FAIL':>5} "
              f"{r.get('distill_loss_last', float('nan')):>8.4f} "
              f"{r.get('distill_loss_slope_per_1k', float('nan')):>+9.4f} "
              f"{r.get('language_loss_last', float('nan')):>7.3f} "
              f"{r.get('language_loss_slope_per_1k', float('nan')):>+9.4f} "
              f"{r['zero_cloud_eval_loss']:>10.3f}")
    print()
    print("splits with FAILED utility but privacy still at the floor: "
          f"{report['splits_with_failed_utility_but_floor_privacy']}")
    print("  largest utility damage that still gave floor privacy : "
          f"{report['max_utility_damage_with_privacy_at_floor']}")
    print("  smallest utility damage that failed privacy          : "
          f"{report['min_utility_damage_with_privacy_failed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
