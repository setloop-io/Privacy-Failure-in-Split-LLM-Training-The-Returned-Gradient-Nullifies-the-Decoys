#!/usr/bin/env python3
"""E1 matched-pair report: the outbound gradient with and without protection.

Two cells, same a2b configuration, same seed, same corpus, differing in one setting:

  e1_unprot_a2b_split14   --outbound-grad-dp off   (dp.releases has no gradient entry)
  gradfix_a2b_split14     --outbound-grad-dp clip_noise, clip 0.01, noise 0.35

Neither was designed as half of a pair: gradfix measured the FIXED case, and E1
supplies the leak-exposure half.

For each arm this prints the frozen gate's reading and, where a prediction dump exists,
the paired cluster-aware advantage of experiment W2.1a. The frozen reading is included for
comparability with published cells, not because it is the statistic of record.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ARMS = ("grad_real", "grad_real_shuffled", "wire_real")


def frozen_reading(path: Path) -> dict | None:
    if not path.is_file():
        return None
    summary = json.loads(path.read_text())["summary"][0]
    best = summary["best_probe_top1_pct"]
    majority = summary["label_free_majority_pct"]
    total = json.loads(path.read_text())["results"][0]["total"]
    se = 100.0 * math.sqrt((majority / 100) * (1 - majority / 100) / total)
    return {
        "n": total,
        "best_top1_pct": round(best, 4),
        "majority_pct": round(majority, 4),
        "point_lift_pp": round(best - majority, 4),
        "se_pp": round(se, 4),
        "z_independent_rows": round((best - majority) / se, 3) if se else None,
        "bonferroni_u95_excess_pp": round(
            summary["upper95_excess_over_majority_pp"], 4),
    }


def paired_reading(path: Path) -> dict | None:
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    best = report.get("best_eligible")
    if not best:
        return None
    return {
        "paired_advantage_pp": best["paired_advantage_pp"],
        "ci95": [best["ci95_low_pp"], best["ci95_high_pp"]],
        "verdict": report["verdict"],
        "degenerate_excluded": report["arms_degenerate_excluded"],
    }


def collect(root: Path, cell: str) -> dict:
    out = {"cell": cell, "arms": {}}
    runner = root / f"{cell}.json"
    if runner.is_file():
        data = json.loads(runner.read_text())
        out["dp_releases"] = (data.get("dp") or {}).get("releases")
        out["gradient_protected"] = "gradient" in (out["dp_releases"] or {})
        out["eval_loss_delta"] = data.get("eval_loss_delta")
        out["train_seconds"] = data.get("train_seconds")
        out["gates"] = data.get("gates")
    for arm in ARMS:
        out["arms"][arm] = {
            "frozen": frozen_reading(root / f"{cell}_arm_{arm}.json"),
            "paired": paired_reading(root / f"{cell}_arm_{arm}_paired.json"),
        }
    return out


def render(cells: list[dict]) -> str:
    lines = []
    for cell in cells:
        protected = cell.get("gradient_protected")
        label = ("gradient PROTECTED (clip+noise)" if protected
                 else "gradient UNPROTECTED and unaccounted")
        lines.append(f"\n{cell['cell']}  --  {label}")
        lines.append(f"  eval_loss_delta {cell.get('eval_loss_delta')}   "
                     f"dp.releases {sorted((cell.get('dp_releases') or {}))}")
        lines.append(f"  {'arm':22s} {'n':>7s} {'lift_pp':>9s} {'z':>7s} "
                     f"{'u95_pp':>8s} {'paired_pp':>10s}  {'verdict':>10s}")
        for arm, reading in cell["arms"].items():
            frozen, paired = reading["frozen"], reading["paired"]
            if not frozen:
                lines.append(f"  {arm:22s} {'-- not scored yet --':>40s}")
                continue
            paired_txt = (f"{paired['paired_advantage_pp']:+10.4f}" if paired
                          else f"{'--':>10s}")
            verdict = paired["verdict"] if paired else "--"
            lines.append(
                f"  {arm:22s} {frozen['n']:7d} {frozen['point_lift_pp']:+9.4f} "
                f"{frozen['z_independent_rows']:+7.2f} "
                f"{frozen['bonferroni_u95_excess_pp']:+8.4f} {paired_txt}  "
                f"{verdict:>10s}")
    lines.append("\nz assumes independent rows, which the evaluation protocol forbids; "
                 "it is shown for comparability with published cells only.")
    lines.append("paired_pp is the statistic of record (PLAN.md W2.1a): paired against "
                 "the same control on the same rows, clustered by frame.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-root", type=Path, required=True)
    parser.add_argument("--gradfix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cells = [collect(args.e1_root, "e1_unprot_a2b_split14"),
             collect(args.gradfix_root, "gradfix_a2b_split14")]
    if args.output:
        args.output.write_text(json.dumps(cells, indent=2, sort_keys=True) + "\n")
    print(render(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
