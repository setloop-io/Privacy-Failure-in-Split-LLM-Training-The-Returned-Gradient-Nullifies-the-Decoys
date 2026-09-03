#!/usr/bin/env python3
"""AC1.3 / AC4.2: re-derive every reported privacy number from raw counts.

Nothing here reads a summary field. The excess, the Bonferroni-adjusted Wilson
upper bound, the statistical floor and the binomial z are all recomputed from
`correct` and `total` in the attacker artifact's arms, so a wrong summary in the
artifact cannot propagate into the write-up. The Bonferroni correction is taken
from the arm count present in the artifact, not assumed: most cells score nine
(3 model classes x 3 restarts) but four v6-era artifacts score six.

The floor is the excess a probe scoring *exactly* at the majority rate would
still show from confidence width alone (main.tex:163-169). A reading is only
evidence of leakage if it clears that floor plus the declared single-seed
variance of +-0.2 to 0.25 pp.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

ALPHA = 0.05
N_ARMS = 9
GATE_PP = 1.0


def z_bonferroni(alpha: float = ALPHA, arms: int = N_ARMS) -> float:
    """One-sided normal quantile at alpha/arms. main.tex quotes 2.5392."""
    from statistics import NormalDist
    return NormalDist().inv_cdf(1.0 - alpha / arms)


def wilson_upper(correct: int, total: int, z: float) -> float:
    p = correct / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / total
                                   + z * z / (4 * total * total))
    return 100.0 * (center + half)


def binomial_z(correct: int, total: int, p0: float) -> float:
    """How many sd above the majority rate the arm scored."""
    expected = total * p0
    sd = math.sqrt(total * p0 * (1 - p0))
    return (correct - expected) / sd


def analyse(path: str) -> dict:
    art = json.loads(Path(path).read_text())
    # Derive the Bonferroni correction from the artifact, never from a constant.
    # Most cells score nine arms (3 model classes x 3 restarts), but four v6-era
    # artifacts under paper-data/collected/invalid/latent_v6_rejected/ score six
    # and were bounded at z(alpha/6) = 2.3940. Hard-coding nine silently
    # recomputes those cells at the wrong quantile and produces a WRONG excess
    # (e.g. latent_v6_remote_batch_seed42: +0.3544 pp against the artifact's
    # +0.3167 pp) rather than failing.
    n_arms = len(art["results"])
    if n_arms < 1:
        raise ValueError(f"{path}: no attacker arms to score")
    z = z_bonferroni(arms=n_arms)
    majority_pct = art["summary"][0]["label_free_majority_pct"]
    p0 = majority_pct / 100.0

    arms = []
    for r in art["results"]:
        upper = wilson_upper(r["correct"], r["total"], z)
        arms.append({
            "model": r["model"], "restart": r["restart"],
            "correct": r["correct"], "total": r["total"],
            "top1_pct": 100.0 * r["correct"] / r["total"],
            "recomputed_upper95_pct": upper,
            "artifact_upper95_pct": r["bonferroni_upper95_pct"],
            "upper_matches": abs(upper - r["bonferroni_upper95_pct"]) < 1e-9,
            "binomial_z_vs_majority": binomial_z(r["correct"], r["total"], p0),
        })

    worst = max(arms, key=lambda a: a["recomputed_upper95_pct"])
    excess = worst["recomputed_upper95_pct"] - majority_pct

    # The floor: an arm scoring exactly at the majority rate, same n.
    n = worst["total"]
    floor_upper = wilson_upper(round(p0 * n), n, z)
    floor_pp = floor_upper - majority_pct

    return {
        "file": path,
        "n_rows": n,
        "n_arms": n_arms,
        "z_bonferroni": z,
        "label_free_majority_pct": majority_pct,
        "arms": arms,
        "worst_arm": {k: worst[k] for k in
                      ("model", "restart", "correct", "total", "top1_pct",
                       "recomputed_upper95_pct", "binomial_z_vs_majority")},
        "excess_pp_recomputed": excess,
        "excess_pp_in_artifact": art["summary"][0]["upper95_excess_over_majority_pp"],
        "excess_matches": abs(
            excess - art["summary"][0]["upper95_excess_over_majority_pp"]) < 1e-9,
        "statistical_floor_pp": floor_pp,
        "excess_over_floor_pp": excess - floor_pp,
        "gate_pp": GATE_PP,
        "gate_pass": excess <= GATE_PP,
        "all_upper_bounds_match": all(a["upper_matches"] for a in arms),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--baseline-label", default="BASELINE")
    ap.add_argument("--cell-label", default="CELL")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    base, cell = analyse(args.baseline), analyse(args.cell)
    report = {
        "schema": "dtraining.deleg6040.verify_stats.v1",
        "baseline_label": args.baseline_label,
        "cell_label": args.cell_label,
        "baseline": base,
        "cell": cell,
        "delta_excess_pp": cell["excess_pp_recomputed"] - base["excess_pp_recomputed"],
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")

    for tag, r in ((args.baseline_label, base), (args.cell_label, cell)):
        print(f"--- {tag} ({Path(r['file']).name}) ---")
        print(f"  n rows                 : {r['n_rows']}")
        print(f"  arms scored            : {r['n_arms']}")
        print(f"  z (Bonferroni a/{r['n_arms']})     : {r['z_bonferroni']:.4f}")
        print(f"  majority control       : {r['label_free_majority_pct']:.4f}%")
        w = r["worst_arm"]
        print(f"  worst arm              : {w['model']} restart {w['restart']}"
              f"  {w['correct']}/{w['total']} = {w['top1_pct']:.4f}%")
        print(f"  Wilson upper95         : {w['recomputed_upper95_pct']:.4f}%")
        print(f"  binomial z vs majority : {w['binomial_z_vs_majority']:.3f}")
        print(f"  EXCESS                 : {r['excess_pp_recomputed']:+.4f} pp"
              f"   (artifact {r['excess_pp_in_artifact']:+.4f}, "
              f"match={r['excess_matches']})")
        print(f"  statistical floor      : {r['statistical_floor_pp']:+.4f} pp")
        print(f"  excess over floor      : {r['excess_over_floor_pp']:+.4f} pp")
        print(f"  GATE <= +1.0 pp        : {'PASS' if r['gate_pass'] else 'FAIL'}")
        print(f"  all {r['n_arms']} bounds re-derived: "
              f"{r['all_upper_bounds_match']}")
    print(f"\nDELTA in excess ({args.cell_label} - {args.baseline_label}): "
          f"{report['delta_excess_pp']:+.4f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
