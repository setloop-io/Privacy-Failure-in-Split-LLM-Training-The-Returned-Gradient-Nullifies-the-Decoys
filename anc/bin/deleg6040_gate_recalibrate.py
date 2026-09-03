#!/usr/bin/env python3
"""Re-gate any attacker artifact against a floor-relative privacy gate.

The campaign metric is the label-free majority control excess: the worst of the
nine arms' Bonferroni-adjusted Wilson upper-95 bounds, minus the majority
control. The pre-declared gate is <= +1.0 pp (main.tex:158).

That gate is n-dependent in the wrong direction. The metric has a statistical
floor -- the excess a probe scoring *exactly* at the majority rate still shows
from confidence width alone -- and the floor shrinks as the evaluation grows:
+0.639 pp at n=8,192, +0.410 pp at n=20,480, +0.091 pp at n=327,680. A fixed
+1.0 pp gate therefore hands the attacker a *real recovery* budget of
1.0 - floor(n), which grows from +0.36 pp to +0.91 pp over the campaign's range.
More evaluation rows make the gate easier to pass.

The proposed gate caps the excess at the floor plus a fixed real-recovery
budget, and keeps the legacy ceiling as a second leg:

    excess <= min(LEGACY_GATE_PP, floor(n, p0, k) + FLOOR_BUDGET_PP)

Because the cap is a minimum with the legacy value it can only ever tighten:
no cell the legacy gate failed can pass the new gate. The budget is set to
+0.70 pp = the largest floor-relative reading among accepted campaign cells
(+0.4446 pp, v9.4 E6 K=4 two endpoints) plus one declared single-seed variance
(0.25 pp, main.tex:169), so seed noise alone cannot flip an accepted cell.

Significance is reported but never gates. The excess/floor ratio is exactly
1 + z_binomial/z_bonferroni, so a ratio gate and a significance gate are the
same gate, and its tolerated recovery falls as 1/sqrt(n) -- the opposite
n-dependence. Magnitude decides the verdict; significance classifies whether
the reading is distinguishable from the floor at all.

Nothing derived is read from a summary field. The majority control is the one
input taken from `summary` (it is not recoverable from the arms); every bound,
excess, floor and z is recomputed from raw `correct`/`total` counts, and the
artifact's own excess is read back only to cross-check the re-derivation.

Sweep the whole corpus and write the committed artifact:

    python3 bin/deleg6040_gate_recalibrate.py --sweep paper-data/collected \\
      --output paper-data/collected/diagnostic/deleg_60_40/\\
gate_recalibration_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deleg6040_verify_stats import binomial_z, wilson_upper, z_bonferroni  # noqa: E402

LEGACY_GATE_PP = 1.0
FLOOR_BUDGET_PP = 0.70
RESOLVED_RATIO = 2.0  # ratio = 1 + z/z_bonferroni, so ratio 2 <=> z = z_bonferroni
SCHEMA = "dtraining.deleg6040.gate_recalibration.v1"


def classify(excess_pp: float, floor_pp: float, best_z: float,
             z_bonf: float) -> str:
    """Is the reading distinguishable from a probe scoring at the majority rate?"""
    if excess_pp <= floor_pp:
        return "at-floor"
    return "resolved" if best_z > z_bonf else "at-floor"


def arm_record(arm: dict[str, Any], n: int, upper_pct: float) -> dict[str, Any]:
    """One attacker arm, with its re-derived Bonferroni-Wilson upper bound."""
    return {
        "model": arm["model"],
        "restart": arm["restart"],
        "correct": arm["correct"],
        "total": n,
        "top1_pct": 100.0 * arm["correct"] / n,
        "recomputed_upper95_pct": upper_pct,
    }


def recalibrate(path: str, legacy_gate_pp: float = LEGACY_GATE_PP,
                floor_budget_pp: float = FLOOR_BUDGET_PP) -> dict[str, Any]:
    """Re-derive every gate input for one attacker artifact from raw counts."""
    art = json.loads(Path(path).read_text())
    arms = art["results"]
    assert arms, f"{path}: no attacker arms"
    totals = {a["total"] for a in arms}
    assert len(totals) == 1, f"{path}: ragged arm totals {sorted(totals)}"
    n = totals.pop()
    assert n > 0, f"{path}: empty evaluation"

    k = len(arms)
    z_bonf = z_bonferroni(arms=k)
    majority_pct = art["summary"][0]["label_free_majority_pct"]
    p0 = majority_pct / 100.0
    assert 0.0 < p0 < 1.0, f"{path}: majority control {majority_pct} out of range"

    uppers = [wilson_upper(a["correct"], n, z_bonf) for a in arms]
    worst = max(range(k), key=uppers.__getitem__)
    excess_pp = uppers[worst] - majority_pct
    floor_pp = wilson_upper(round(p0 * n), n, z_bonf) - majority_pct
    assert floor_pp > 0.0, f"{path}: non-positive statistical floor"
    best_z = max(binomial_z(a["correct"], n, p0) for a in arms)
    reported = art["summary"][0]["upper95_excess_over_majority_pp"]

    cell = {
        "file": path,
        "n_rows": n,
        "n_arms": k,
        "z_bonferroni": z_bonf,
        "label_free_majority_pct": majority_pct,
        "worst_arm": arm_record(arms[worst], n, uppers[worst]),
        "excess_pp": excess_pp,
        "excess_pp_in_artifact": reported,
        "excess_matches_artifact": abs(excess_pp - reported) < 1e-9,
        "statistical_floor_pp": floor_pp,
        "excess_over_floor_pp": excess_pp - floor_pp,
        "excess_over_floor_ratio": excess_pp / floor_pp,
        "best_arm_binomial_z": best_z,
        "classification": classify(excess_pp, floor_pp, best_z, z_bonf),
    }
    cell.update(apply_gates(excess_pp, floor_pp, legacy_gate_pp, floor_budget_pp))
    return cell


def apply_gates(excess_pp: float, floor_pp: float, legacy_gate_pp: float,
                floor_budget_pp: float) -> dict[str, Any]:
    """Legacy fixed gate and proposed floor-relative gate, plus agreement."""
    cap_pp = min(legacy_gate_pp, floor_pp + floor_budget_pp)
    legacy = "PASS" if excess_pp <= legacy_gate_pp else "FAIL"
    proposed = "PASS" if excess_pp <= cap_pp else "FAIL"
    return {
        "legacy_gate_pp": legacy_gate_pp,
        "legacy_verdict": legacy,
        "floor_budget_pp": floor_budget_pp,
        "floor_relative_cap_pp": cap_pp,
        "proposed_verdict": proposed,
        "gates_agree": legacy == proposed,
    }


def find_artifacts(root: str) -> list[str]:
    base = Path(root)
    assert base.is_dir(), f"{root}: not a directory"
    found = sorted(str(p) for p in base.rglob("*attacker*.json"))
    assert found, f"{root}: no attacker artifacts found"
    return found


HEADER = (f"{'excess':>9} {'floor':>8} {'over':>8} {'ratio':>7} {'bestz':>9} "
          f"{'n':>7} {'k':>2} {'cap':>7} {'legacy':>7} {'new':>5} {'':>9}  cell")


def print_table(cells: list[dict[str, Any]], root: str) -> None:
    print(HEADER)
    for c in sorted(cells, key=lambda x: x["excess_pp"]):
        name = c["file"].replace(root.rstrip("/") + "/", "")
        agree = "AGREE" if c["gates_agree"] else "DISAGREE"
        print(f"{c['excess_pp']:>+9.4f} {c['statistical_floor_pp']:>+8.4f} "
              f"{c['excess_over_floor_pp']:>+8.4f} "
              f"{c['excess_over_floor_ratio']:>7.3f} "
              f"{c['best_arm_binomial_z']:>+9.3f} {c['n_rows']:>7} "
              f"{c['n_arms']:>2} {c['floor_relative_cap_pp']:>7.4f} "
              f"{c['legacy_verdict']:>7} {c['proposed_verdict']:>5} "
              f"{agree:>9}  {name}")


def summarise(cells: list[dict[str, Any]], legacy_gate_pp: float,
              floor_budget_pp: float, root: str) -> dict[str, Any]:
    disagreements = [c for c in cells if not c["gates_agree"]]
    return {
        "schema": SCHEMA,
        "root": root,
        "legacy_gate_pp": legacy_gate_pp,
        "floor_budget_pp": floor_budget_pp,
        "resolved_ratio_threshold": RESOLVED_RATIO,
        "n_cells": len(cells),
        "n_legacy_pass": sum(c["legacy_verdict"] == "PASS" for c in cells),
        "n_proposed_pass": sum(c["proposed_verdict"] == "PASS" for c in cells),
        "n_agree": sum(c["gates_agree"] for c in cells),
        "n_disagree": len(disagreements),
        "n_excess_matches_artifact": sum(c["excess_matches_artifact"] for c in cells),
        "n_resolved": sum(c["classification"] == "resolved" for c in cells),
        "disagreements": [
            {k: c[k] for k in ("file", "n_rows", "excess_pp", "statistical_floor_pp",
                               "excess_over_floor_pp", "best_arm_binomial_z",
                               "legacy_verdict", "proposed_verdict")}
            for c in disagreements
        ],
        "cells": cells,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifact", action="append", default=[],
                    help="attacker artifact to re-gate (repeatable)")
    ap.add_argument("--sweep", help="directory to search recursively for artifacts")
    ap.add_argument("--output", help="write the sweep report JSON here")
    ap.add_argument("--legacy-gate-pp", type=float, default=LEGACY_GATE_PP)
    ap.add_argument("--floor-budget-pp", type=float, default=FLOOR_BUDGET_PP)
    ap.add_argument("--quiet", action="store_true", help="counts only, no table")
    args = ap.parse_args()

    paths = list(args.artifact)
    root = args.sweep or "."
    if args.sweep:
        paths += find_artifacts(args.sweep)
    assert paths, "give --artifact and/or --sweep"

    cells = [recalibrate(p, args.legacy_gate_pp, args.floor_budget_pp)
             for p in paths]
    report = summarise(cells, args.legacy_gate_pp, args.floor_budget_pp, root)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    if not args.quiet:
        print_table(cells, root)
    print(f"\ncells {report['n_cells']}  legacy-pass {report['n_legacy_pass']}  "
          f"proposed-pass {report['n_proposed_pass']}  "
          f"agree {report['n_agree']}  DISAGREE {report['n_disagree']}  "
          f"resolved {report['n_resolved']}  "
          f"excess re-derived {report['n_excess_matches_artifact']}/{report['n_cells']}")
    for d in report["disagreements"]:
        print(f"  DISAGREE {d['legacy_verdict']}->{d['proposed_verdict']}  "
              f"excess {d['excess_pp']:+.4f}  floor {d['statistical_floor_pp']:+.4f}  "
              f"over {d['excess_over_floor_pp']:+.4f}  z {d['best_arm_binomial_z']:+.3f}"
              f"  {d['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
