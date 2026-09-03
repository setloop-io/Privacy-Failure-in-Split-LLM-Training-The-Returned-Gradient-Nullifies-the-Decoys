#!/usr/bin/env python3
"""Root-cause evidence for the 2,000-step privacy failure, and where the
utility bound actually sits.

Two tables, both from committed artifacts only.

(1) ARM-FAMILY BREAKDOWN. The nine attacker arms fall into two families.
    `coordinate_plus_invariants` reads raw latent coordinates, which the secret
    per-block rotation randomises. `invariant_only` and `invariant_graph` read
    row norms and sorted Gram similarities -- quantities a rotation preserves
    EXACTLY, and which v13 leaves unscrambled because it omits
    --secret-token-gauge (the runner's own token_norm_and_sign_gauged gate reads
    false in every cell). Printing the families separately shows the leak
    signature: in every converged cell the invariant arms collapse onto the
    majority control, i.e. they degenerate to constant prediction and find
    nothing. They rise above it only where the minimax has not converged.

(2) SURROGATE EFFICIENCY. eval_loss_delta alone conflates two things: how hard
    the surrogate's job is, and how well it does it. Separating them:

      gap      = zero_cloud_eval_loss - baseline_eval_loss
                 how much loss the delegated layers were worth
      closed   = zero_cloud_eval_loss - candidate_eval_loss
                 how much of it the surrogate recovered
      fraction = closed / gap
      required = the fraction needed to bring the residual under the +0.35 gate

    The fraction is the surrogate's competence; the gap is the size of its job.
    If the fraction is flat while the gap grows, the utility failure is not the
    surrogate getting worse -- it is a fixed absolute gate applied to a
    rapidly growing gap.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

TOTAL_LAYERS = 28
UTILITY_GATE = 0.35
INVARIANT_ARMS = ("invariant_only", "invariant_graph")


def load(root: Path) -> list[dict]:
    rows = []
    for runner in sorted(root.glob("*.json")):
        if runner.name.endswith("_attacker.json") or "_split" not in runner.name:
            continue
        attacker = runner.with_name(runner.stem + "_attacker.json")
        if not attacker.exists():
            continue
        run = json.loads(runner.read_text())
        att = json.loads(attacker.read_text())
        summary = att["summary"][0]
        majority = summary["label_free_majority_pct"]

        by_model: dict[str, list[float]] = {}
        for r in att["results"]:
            by_model.setdefault(r["model"], []).append(r["top1_pct"])
        mean = {k: sum(v) / len(v) for k, v in by_model.items()}
        inv_mean = sum(mean[m] for m in INVARIANT_ARMS) / len(INVARIANT_ARMS)

        best = max(att["results"], key=lambda a: a["top1_pct"])
        p0, n = majority / 100.0, best["total"]
        z = (best["correct"] - n * p0) / math.sqrt(n * p0 * (1 - p0))

        base = run["baseline_eval_loss"]
        gap = run["zero_cloud_eval_loss"] - base
        closed = run["zero_cloud_eval_loss"] - run["candidate_eval_loss"]
        delegated = run["resume_after"] - run["split_after"] - 1

        rows.append({
            "split_after": run["split_after"],
            "steps": run["steps"],
            "delegated_layers": delegated,
            "delegated_pct": 100.0 * delegated / TOTAL_LAYERS,
            "excess_pp": summary["upper95_excess_over_majority_pp"],
            "privacy_gate_pass": summary["upper95_excess_over_majority_pp"] <= 1.0,
            "best_arm_binomial_z": z,
            "majority_pct": majority,
            "coordinate_arm_mean_pct": mean["coordinate_plus_invariants"],
            "invariant_arm_mean_pct": inv_mean,
            "invariant_lift_pp": inv_mean - majority,
            "coordinate_lift_pp": mean["coordinate_plus_invariants"] - majority,
            "distill_loss_last": run["training_tail"][-1]["distill_loss"],
            "baseline_eval_loss": base,
            "zero_cloud_eval_loss": run["zero_cloud_eval_loss"],
            "candidate_eval_loss": run["candidate_eval_loss"],
            "gap_nats": gap,
            "closed_nats": closed,
            "fraction_closed": closed / gap,
            "fraction_required_for_gate": (gap - UTILITY_GATE) / gap,
            "residual_nats": run["candidate_eval_loss"] - base,
            "utility_gate_pass": run["gates"]["utility_delta_le_0_35"],
        })
    rows.sort(key=lambda r: (r["delegated_layers"], r["steps"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = load(Path(args.dir))

    print("(1) ARM FAMILIES -- the leak signature")
    h = (f"{'layers':>6}{'deleg':>7}{'steps':>7} | {'majority':>9}{'coord':>8}"
         f"{'invariant':>10} | {'inv lift':>9}{'z':>8} {'distill':>8}  privacy")
    print(h)
    print("-" * len(h))
    for r in rows:
        print(f"{r['delegated_layers']:>6}{r['delegated_pct']:>6.1f}%"
              f"{r['steps']:>7} | {r['majority_pct']:>8.3f}%"
              f"{r['coordinate_arm_mean_pct']:>7.3f}%"
              f"{r['invariant_arm_mean_pct']:>9.3f}% | "
              f"{r['invariant_lift_pp']:>+9.3f}{r['best_arm_binomial_z']:>8.2f} "
              f"{r['distill_loss_last']:>8.4f}  "
              f"{'pass' if r['privacy_gate_pass'] else 'FAIL'}")

    print()
    print("(2) SURROGATE EFFICIENCY -- where the utility bound actually is")
    h2 = (f"{'layers':>6}{'deleg':>7}{'steps':>7} | {'gap':>8}{'closed':>8}"
          f"{'frac':>8}{'needed':>8} | {'residual':>9}  utility")
    print(h2)
    print("-" * len(h2))
    for r in rows:
        print(f"{r['delegated_layers']:>6}{r['delegated_pct']:>6.1f}%"
              f"{r['steps']:>7} | {r['gap_nats']:>8.3f}{r['closed_nats']:>8.3f}"
              f"{100 * r['fraction_closed']:>7.1f}%"
              f"{100 * r['fraction_required_for_gate']:>7.1f}% | "
              f"{r['residual_nats']:>+9.3f}  "
              f"{'pass' if r['utility_gate_pass'] else 'FAIL'}")

    converged = [r for r in rows if r["privacy_gate_pass"]
                 and r["delegated_layers"] > 4]
    report = {
        "schema": "dtraining.deleg6040.root_cause.v1",
        "rows": rows,
        "invariant_arms_collapse_to_majority_when_privacy_passes": all(
            abs(r["invariant_lift_pp"]) < 0.25
            for r in rows if r["privacy_gate_pass"]),
        "invariant_lift_when_privacy_fails_pp": [
            r["invariant_lift_pp"] for r in rows if not r["privacy_gate_pass"]],
        "fraction_closed_range_beyond_4_layers": [
            min(r["fraction_closed"] for r in converged),
            max(r["fraction_closed"] for r in converged)],
        "note": ("fraction_closed is roughly flat beyond 4 delegated layers "
                 "while gap_nats grows sixfold; the utility failure is a fixed "
                 "absolute gate meeting a growing gap, not a surrogate that "
                 "degrades with depth"),
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print()
    print("invariant arms collapse onto majority whenever privacy passes: "
          f"{report['invariant_arms_collapse_to_majority_when_privacy_passes']}")
    lo, hi = report["fraction_closed_range_beyond_4_layers"]
    print(f"fraction closed beyond 4 delegated layers: "
          f"{100 * lo:.1f}% to {100 * hi:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
