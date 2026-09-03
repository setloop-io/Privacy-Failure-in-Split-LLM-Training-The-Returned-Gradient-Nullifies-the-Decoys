#!/usr/bin/env python3
"""T1.1 -- run-to-run variance of the reported metrics, and what it resolves.

Five byte-identical runs of the same configuration (split_after 14, 10,000 steps,
256 blocks, D=64). Nothing varies between them: `--seed` is 42 in all five, and
the spread comes entirely from `privacy_runtime/latent_native.py:105` drawing
fresh CSPRNG entropy per DP call and the gauge being redrawn per block.

The external adversarial review (finding 10) established that no such estimate
existed while closure sequences like 86.1 / 86.0 / 86.4 / 86.5 were nonetheless
used to declare mechanisms closed. This produces the missing standard deviation
and then applies it: for every capability cell measured in this effort, it reports
the difference from the repeat mean in units of the standard error of that
difference, so an over-read difference cannot survive the table.

The standard error of a difference between two single runs is sd*sqrt(2); between
a single run and the mean of the five repeats it is sqrt(sd^2 + sd^2/5).
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path
from statistics import NormalDist

ALPHA = 0.05


def wilson_upper(correct: int, total: int, z: float) -> float:
    p = correct / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / total
                                   + z * z / (4 * total * total))
    return 100.0 * (centre + half)


def metrics(stem: Path) -> dict:
    run = json.loads(stem.with_suffix(".json").read_text())
    att = json.loads(Path(str(stem) + "_attacker.json").read_text())
    summary = att["summary"][0]
    base = run["baseline_eval_loss"]
    gap = run["zero_cloud_eval_loss"] - base
    closed = run["zero_cloud_eval_loss"] - run["candidate_eval_loss"]
    arms = att["results"]
    n = arms[0]["total"]
    z = NormalDist().inv_cdf(1.0 - ALPHA / len(arms))
    p0 = summary["label_free_majority_pct"] / 100.0
    best = max(arms, key=lambda a: a["top1_pct"])
    return {
        "name": stem.name,
        "closure_pct": 100.0 * closed / gap,
        "residual_nats": run["candidate_eval_loss"] - base,
        "excess_pp": summary["upper95_excess_over_majority_pp"],
        "binomial_z": (best["correct"] - n * p0) / math.sqrt(n * p0 * (1 - p0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--repeats", nargs="+", required=True,
                    help="artifact stems of the identical repeats")
    ap.add_argument("--compare", nargs="*", default=[],
                    help="capability cells to test against the repeat spread")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.dir)
    reps = [metrics(root / s) for s in args.repeats]
    if len(reps) < 3:
        raise ValueError("need at least three repeats for a usable sd")

    fields = ("closure_pct", "residual_nats", "excess_pp")
    stats = {}
    for f in fields:
        values = [r[f] for r in reps]
        sd = statistics.stdev(values)
        stats[f] = {
            "n": len(values), "mean": statistics.mean(values), "sd": sd,
            "range": max(values) - min(values),
            # resolution floor: the smallest difference between two single runs
            # that reaches two standard errors
            "two_se_single_pair": 2.0 * sd * math.sqrt(2.0),
            "two_se_vs_mean": 2.0 * math.sqrt(sd * sd + sd * sd / len(values)),
        }

    print("Identical repeats")
    hdr = f"{'run':38s}{'closure%':>10}{'residual':>11}{'excess pp':>11}{'z':>8}"
    print(hdr); print("-" * len(hdr))
    for r in reps:
        print(f"{r['name'][:37]:38s}{r['closure_pct']:>10.3f}"
              f"{r['residual_nats']:>+11.4f}{r['excess_pp']:>+11.4f}"
              f"{r['binomial_z']:>8.2f}")
    print()
    for f in fields:
        s = stats[f]
        print(f"{f:15s} mean {s['mean']:+9.4f}  sd {s['sd']:.4f}  "
              f"range {s['range']:.4f}  |  a difference needs "
              f"{s['two_se_vs_mean']:.4f} to reach 2 se of the repeat mean")

    compared = []
    if args.compare:
        print()
        print("Capability cells against the repeat spread "
              "(sigma = difference / se of that difference)")
        hdr2 = (f"{'cell':38s}{'closure%':>10}{'sig':>7}"
                f"{'residual':>11}{'sig':>7}{'excess pp':>11}{'sig':>7}")
        print(hdr2); print("-" * len(hdr2))
        for stem in args.compare:
            m = metrics(root / stem)
            row = {"name": m["name"]}
            cells = []
            for f in fields:
                s = stats[f]
                se = math.sqrt(s["sd"] ** 2 + s["sd"] ** 2 / s["n"])
                sig = (m[f] - s["mean"]) / se if se else float("nan")
                row[f] = m[f]
                row[f + "_sigma_vs_repeats"] = sig
                cells.append((m[f], sig))
            compared.append(row)
            print(f"{m['name'][:37]:38s}"
                  f"{cells[0][0]:>10.3f}{cells[0][1]:>+7.1f}"
                  f"{cells[1][0]:>+11.4f}{cells[1][1]:>+7.1f}"
                  f"{cells[2][0]:>+11.4f}{cells[2][1]:>+7.1f}")

    Path(args.output).write_text(json.dumps({
        "schema": "dtraining.deleg6040.variance.v1",
        "repeats": reps, "statistics": stats, "compared": compared,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
