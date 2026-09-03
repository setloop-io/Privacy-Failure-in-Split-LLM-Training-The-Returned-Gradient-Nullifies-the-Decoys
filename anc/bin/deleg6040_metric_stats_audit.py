#!/usr/bin/env python3
"""Two statistical defects in the label-free majority control excess.

DEFECT 1 -- the control estimator does not match the paper.
main.tex:153 defines the control as the best constant predictor ON THE
EVALUATION ROWS.  attacker/attacks/latent_probe.py:215-217 instead takes
mode(train_tokens) and scores that one constant on the evaluation rows.
Writing L_v for the number of evaluation rows carrying token v, m for the
train mode and U for the worst Bonferroni-Wilson upper bound,

    excess_code  = U - 100 * L_m / n
    excess_paper = U - 100 * max_v L_v / n

so the two differ by exactly the control shift, one pp for one pp, and
because max_v L_v >= L_m the paper's excess is never larger.  This module
turns that into a decision: for every attacker artifact it computes the
SMALLEST control count at which the cell's verdict would change, under both
legs of the gate, so a bounded question ("could this cell flip?") replaces an
unbounded one.  max_v L_v itself is measured by bin/deleg6040_paper_control.py
for the cells whose evaluation rows are corpus rows, and is unrecoverable for
the cells that release chaff; see --help-limits.

DEFECT 2 -- the intervals assume independence the design does not provide.
The n rows are eval_blocks released frames of rows_per_frame rows each (32
real + 48 chaff at the v13 operating point).  The invariant_graph arm makes
every row's prediction a function of the whole frame's Gram matrix
(latent_probe.py:115-135) and chaff rows are recycled inside frames, so rows
are not independent Bernoulli trials.  Separately, the control and the arm are
scored on the SAME rows, so the comparison is paired: subtracting a point
control from a one-sample upper bound is not a bound on their difference.

This module reports, per cell, (a) the paired (McNemar) difference and the
exact interval its standard error must lie in given only the marginal counts,
and (b) the largest that standard error can become once frames are the
clustering unit -- an exact extremal bound, never an estimate of the true
design effect, which needs per-row per-frame correctness that no committed
artifact carries.

Nothing is read from a summary field except label_free_majority_pct, which is
not recoverable from the arms; every bound is recomputed from raw counts.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deleg6040_verify_stats import wilson_upper, z_bonferroni  # noqa: E402

LEGACY_GATE_PP = 1.0
FLOOR_BUDGET_PP = 0.70
SCHEMA = "dtraining.deleg6040.metric_stats_audit.v1"

LIMITS = """\
What cannot be computed from committed artifacts, and what would close it:

1. max_v L_v (the paper's control) needs the released evaluation label
   multiset.  For the 26 chaff-free cells that multiset IS the corpus, and
   bin/deleg6040_paper_control.py measures it: the shift is exactly zero
   there.  For the 85 cells that release chaff it is unrecoverable in
   principle, not merely unavailable -- chaff rows are drawn by a CSPRNG
   permutation seeded inside TLN and never stored
   (bin/run_latent_native_v5_06b.py:295-304), so no corpus regeneration
   reaches them.  What would close it is bundle["eval_tokens"], carried only
   by the trusted .pt bundles, which are gitignored ("*.pt") and were
   retained for three cells only (docs/experiments/BUNDLE_FORENSICS.md,
   root-owned under ~/experiments/results/training/deleg6040/bundles/ on
   gx10-odysseus.nord).  One command per retained bundle:
   python3 -c 'import torch,collections;print(collections.Counter(
   torch.load(B)["eval_tokens"].reshape(-1).tolist()).most_common(5))'

2. The true design effect needs per-row correctness grouped by frame.  The
   frozen attacker emits that only under --dump-eval-predictions, whose .pt
   output is written to a scratch workdir and is not committed.  Without it
   the correction can only be bounded, which is what this tool does.
"""


def majority_count(majority_pct: float, n: int) -> int:
    """The control's correct-row count, recovered exactly from its percentage."""
    raw = majority_pct * n / 100.0
    count = round(raw)
    if abs(raw - count) > 1e-6:
        raise ValueError(f"majority {majority_pct}% of {n} is not integral")
    return count


def load_cell(path: Path) -> dict[str, Any]:
    """Raw counts and the released frame structure, from one attacker artifact."""
    art = json.loads(path.read_text())
    arms, summary, config = art["results"], art["summary"][0], art["config"]
    if not arms:
        raise ValueError(f"{path}: no attacker arms")
    totals = {a["total"] for a in arms}
    if len(totals) != 1:
        raise ValueError(f"{path}: ragged arm totals {sorted(totals)}")
    n = totals.pop()
    frames, per_frame = config["eval_blocks"], config["sequence_length"]
    if frames * per_frame != n:
        raise ValueError(f"{path}: {frames} frames x {per_frame} != {n} rows")
    return {
        "file": str(path), "n_rows": n, "frames": frames,
        "rows_per_frame": per_frame, "n_arms": len(arms),
        "z_bonferroni": z_bonferroni(arms=len(arms)),
        "label_free_majority_pct": summary["label_free_majority_pct"],
        "control_correct": majority_count(
            summary["label_free_majority_pct"], n),
        "arms": [{"model": a["model"], "restart": a["restart"],
                  "correct": a["correct"]} for a in arms],
        "excess_pp_in_artifact": summary["upper95_excess_over_majority_pp"],
    }


def worst_arm(cell: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """The arm the gate reads: highest Bonferroni-Wilson upper bound."""
    n, z = cell["n_rows"], cell["z_bonferroni"]
    scored = [(wilson_upper(a["correct"], n, z), a) for a in cell["arms"]]
    upper, arm = max(scored, key=lambda pair: pair[0])
    return arm, upper


def smallest_control_passing(upper_pct: float, n: int, z: float,
                             gate_pp: float, budget_pp: float) -> int:
    """Least control count C at which BOTH gate legs pass.

    Leg 1 is  U - 100C/n <= gate.  Leg 2 is  excess - floor <= budget, and
    since floor(C) = WilsonUpper(C,n,z) - 100C/n the control term cancels
    outright, leaving  U - WilsonUpper(C,n,z) <= budget.  Both are monotone
    non-increasing in C, so the passing set is upward closed and a single
    threshold decides the cell.
    """
    leg1 = max(0, math.ceil(n * (upper_pct - gate_pp) / 100.0))
    target = upper_pct - budget_pp
    low, high = 0, n
    while low < high:
        mid = (low + high) // 2
        if wilson_upper(mid, n, z) >= target:
            high = mid
        else:
            low = mid + 1
    return max(leg1, low)


def control_fix(cell: dict[str, Any], gate_pp: float,
                budget_pp: float) -> dict[str, Any]:
    """Defect 1: how far the control must move before this cell's verdict does."""
    n, z, control = cell["n_rows"], cell["z_bonferroni"], cell["control_correct"]
    arm, upper = worst_arm(cell)
    excess = upper - cell["label_free_majority_pct"]
    floor_pp = wilson_upper(control, n, z) - cell["label_free_majority_pct"]
    cap = min(gate_pp, floor_pp + budget_pp)
    needed = smallest_control_passing(upper, n, z, gate_pp, budget_pp)
    return {
        "worst_arm": {**arm, "upper95_pct": upper},
        "excess_pp": excess,
        "excess_matches_artifact":
            abs(excess - cell["excess_pp_in_artifact"]) < 1e-9,
        "statistical_floor_pp": floor_pp,
        "gate_cap_pp": cap,
        "verdict": "PASS" if excess <= cap else "FAIL",
        "control_correct": control,
        "control_correct_to_pass": needed,
        "control_rows_needed": max(0, needed - control),
        "control_shift_pp_needed": max(0.0, 100.0 * (needed - control) / n),
        "verdict_can_change": needed > control,
    }


def excess_and_cap_at(deff: float, cell: dict[str, Any], correct: int,
                      gate_pp: float, budget_pp: float) -> tuple[float, float]:
    """The published excess and gate cap recomputed at effective sample n/DEFF.

    A design effect of D means the clustered design carries the information of
    n/D independent rows, so every Wilson bound is re-evaluated there at the
    same observed rates.  D=1 reproduces the published numbers exactly.
    """
    n, z = cell["n_rows"], cell["z_bonferroni"]
    control_pct = cell["label_free_majority_pct"]
    effective = n / deff
    rate, base = correct / n, control_pct / 100.0
    excess = wilson_upper(rate * effective, effective, z) - control_pct
    floor = wilson_upper(base * effective, effective, z) - control_pct
    return excess, min(gate_pp, floor + budget_pp)


def design_effect_to_fail_published(cell: dict[str, Any], correct: int,
                                    gate_pp: float,
                                    budget_pp: float) -> float:
    """Design effect at which the cell's PUBLISHED verdict turns to FAIL.

    Relaxes independence and nothing else: the estimator, the arm, the control
    and both gate legs are the committed ones.  Returned as a multiple of the
    independent-row assumption, against a structural ceiling of one frame size.
    """
    ceiling = float(cell["rows_per_frame"])
    excess, cap = excess_and_cap_at(1.0, cell, correct, gate_pp, budget_pp)
    if excess > cap:
        return 1.0
    excess, cap = excess_and_cap_at(ceiling, cell, correct, gate_pp, budget_pp)
    if excess <= cap:
        return float("inf")
    low, high = 1.0, ceiling
    for _ in range(60):
        mid = 0.5 * (low + high)
        excess, cap = excess_and_cap_at(mid, cell, correct, gate_pp, budget_pp)
        low, high = (low, mid) if excess > cap else (mid, high)
    return 0.5 * (low + high)


def overlap_bounds(arm_correct: int, control: int, n: int,
                   majority_predicted: int | None) -> tuple[int, int]:
    """Rows both the arm and the control score correct: exact feasible range.

    Without the predictions only the marginals constrain it.  Where an arm's
    prediction concentration is committed (a forensics artifact), the count of
    rows on which the arm predicts the control's own token pins it further.
    """
    low, high = max(0, arm_correct + control - n), min(arm_correct, control)
    if majority_predicted is not None:
        low = max(low, arm_correct - (n - majority_predicted))
        high = min(high, majority_predicted)
    if low > high:
        raise ValueError(f"infeasible overlap [{low},{high}]")
    return low, high


def paired_se_pp(discordant: int, difference: int, n: int) -> float:
    """SE of the paired accuracy difference, in pp, from the McNemar counts.

    d_i = 1{arm correct} - 1{control correct} takes values in {-1,0,+1} with
    b at +1 and c at -1, so var(d) = (b+c)/n - ((b-c)/n)^2 and the mean's
    standard error is sqrt(b + c - (b-c)^2/n) / n.
    """
    variance = discordant - difference * difference / n
    if variance < 0.0:
        raise ValueError(f"negative paired variance for b+c={discordant}")
    return 100.0 * math.sqrt(variance) / n


def max_cluster_sum_squares(pos: int, neg: int, frames: int,
                            per_frame: int) -> float:
    """Largest possible sum of squared frame totals of d, given the marginals.

    Variance is maximised by packing the +1 rows into as few frames as
    possible and the -1 rows into disjoint frames.  This is an exact extremal
    bound over every allocation the design permits; it never exceeds the
    textbook design-effect ceiling of one cluster size.
    """
    if math.ceil(pos / per_frame) + math.ceil(neg / per_frame) > frames:
        raise ValueError("discordant rows cannot be packed into disjoint frames")
    total = 0.0
    for count in (pos, neg):
        full, rest = divmod(count, per_frame)
        total += full * per_frame * per_frame + rest * rest
    return total


def cluster_se_pp(pos: int, neg: int, frames: int, per_frame: int,
                  n: int) -> float:
    """Cluster-robust SE of the paired difference at its extremal allocation."""
    mean = (pos - neg) / frames
    sum_squares = max_cluster_sum_squares(pos, neg, frames, per_frame)
    centred = sum_squares - frames * mean * mean
    variance = (frames / (frames - 1.0)) * max(0.0, centred)
    return 100.0 * math.sqrt(variance) / n


def design_effect_to_fail(point_pp: float, se_pp: float, z: float,
                          gate_pp: float) -> float:
    """Design effect at which this arm's corrected bound first hits the gate.

    Under clustering the SE scales as sqrt(DEFF), so the bound crosses the gate
    at DEFF = ((gate - point) / (z * se))^2.  Reported against the structural
    ceiling of one frame size, this turns an uninformatively wide worst case
    into a per-cell sensitivity: below this value the verdict is safe.
    """
    if se_pp <= 0.0:
        return float("inf")
    if point_pp >= gate_pp:
        return 0.0
    return ((gate_pp - point_pp) / (z * se_pp)) ** 2


def paired_arm(arm: dict[str, Any], cell: dict[str, Any],
               predicted_majority: int | None, gate_pp: float) -> dict[str, Any]:
    """Defect 2 for one arm: paired point, SE range, and the clustered ceiling."""
    n, z = cell["n_rows"], cell["z_bonferroni"]
    frames, per_frame = cell["frames"], cell["rows_per_frame"]
    control, correct = cell["control_correct"], arm["correct"]
    low, high = overlap_bounds(correct, control, n, predicted_majority)
    difference = correct - control
    tightest, widest = correct + control - 2 * high, correct + control - 2 * low
    point_pp = 100.0 * difference / n
    se_lo, se_hi = (paired_se_pp(tightest, difference, n),
                    paired_se_pp(widest, difference, n))
    se_cluster = cluster_se_pp(correct - low, control - low, frames,
                               per_frame, n)
    se_cluster_lo = cluster_se_pp(correct - high, control - high, frames,
                                  per_frame, n)
    return {
        "model": arm["model"], "restart": arm["restart"], "correct": correct,
        "both_correct_lo": low, "both_correct_hi": high,
        "discordant_lo": tightest, "discordant_hi": widest,
        "paired_point_pp": point_pp,
        "paired_se_pp_lo": se_lo, "paired_se_pp_hi": se_hi,
        "paired_upper_pp_lo": point_pp + z * se_lo,
        "paired_upper_pp_hi": point_pp + z * se_hi,
        "cluster_se_pp_min": se_cluster_lo,
        "cluster_se_pp_max": se_cluster,
        "cluster_inflation_max": se_cluster / se_hi if se_hi > 0 else 1.0,
        "paired_cluster_upper_pp_max": point_pp + z * se_cluster,
        "unpaired_se_pp": 100.0 * math.sqrt(
            (correct / n) * (1 - correct / n) / n),
        "design_effect_to_fail": design_effect_to_fail(point_pp, se_hi, z,
                                                       gate_pp),
        "design_effect_ceiling": per_frame,
    }


def predicted_majority_rows(forensics: dict[str, Any] | None,
                            arm: dict[str, Any], n: int) -> int | None:
    """Rows on which one arm predicts the control's token, if it is committed."""
    if forensics is None:
        return None
    control_token = forensics["majority_control"]["token"]
    for record in forensics["arms"]:
        if (record["model"], record["restart"]) != (arm["model"],
                                                    arm["restart"]):
            continue
        if record["modal_token"] != control_token:
            return None
        rows = record["modal_token_share"] * n
        if abs(rows - round(rows)) > 1e-6:
            raise ValueError(f"modal share {record['modal_token_share']} of "
                             f"{n} is not integral")
        return int(round(rows))
    return None


def clustering_fix(cell: dict[str, Any], forensics: dict[str, Any] | None,
                   gate_pp: float, budget_pp: float) -> dict[str, Any]:
    """Defect 2 for the cell: the arm that maximises each corrected bound."""
    scored = [paired_arm(a, cell,
                         predicted_majority_rows(forensics, a, cell["n_rows"]),
                         gate_pp)
              for a in cell["arms"]]
    paired = max(scored, key=lambda a: a["paired_upper_pp_hi"])
    clustered = max(scored, key=lambda a: a["paired_cluster_upper_pp_max"])
    gate_arm, _ = worst_arm(cell)
    return {
        "forensics_refined": forensics is not None,
        "published_design_effect_to_fail": design_effect_to_fail_published(
            cell, gate_arm["correct"], gate_pp, budget_pp),
        "arms": scored,
        "paired_worst_arm": paired,
        "clustered_worst_arm": clustered,
        "paired_upper_pp": paired["paired_upper_pp_hi"],
        "paired_upper_pp_best_case": paired["paired_upper_pp_lo"],
        "paired_cluster_upper_pp": clustered["paired_cluster_upper_pp_max"],
        "paired_verdict": "PASS" if paired["paired_upper_pp_hi"] <= gate_pp
                          else "FAIL",
        "paired_cluster_verdict":
            "PASS" if clustered["paired_cluster_upper_pp_max"] <= gate_pp
            else "FAIL",
        "design_effect_to_fail": min(a["design_effect_to_fail"]
                                     for a in scored),
        "design_effect_ceiling": cell["rows_per_frame"],
    }


def sibling_forensics(path: Path) -> dict[str, Any] | None:
    """The cell's own forensics artifact, when one was committed."""
    name = path.name.replace("_attacker.json", "_forensics.json")
    candidate = path.with_name(name)
    if candidate == path or not candidate.exists():
        return None
    report = json.loads(candidate.read_text())
    return report if report.get("schema", "").endswith("bundle_forensics.v1") \
        else None


def declared_chaff(path: Path) -> int | None:
    """The cell's released chaff width, if its run artifact records one."""
    run = Path(str(path).replace("_attacker.json", ".json"))
    if run == path or not run.exists():
        return None
    return json.loads(run.read_text()).get("chaff_tokens")


def audit(path: Path, gate_pp: float, budget_pp: float) -> dict[str, Any]:
    """Both defects, for one attacker artifact."""
    cell = load_cell(path)
    forensics = sibling_forensics(path)
    return {**{k: cell[k] for k in
               ("file", "n_rows", "frames", "rows_per_frame", "n_arms",
                "z_bonferroni", "label_free_majority_pct")},
            "declared_chaff_tokens": declared_chaff(path),
            "control_estimator": control_fix(cell, gate_pp, budget_pp),
            "clustering": clustering_fix(cell, forensics, gate_pp,
                                        budget_pp)}


def control_invariance(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Which frame shapes release chaff, measured rather than assumed.

    Chaff is drawn by a fresh CSPRNG permutation on every run, so independent
    cells sharing a frame shape land on distinct control counts iff chaff is
    released.  A group of many cells with ONE control count is chaff-free, and
    its evaluation rows are the corpus rows alone -- deterministic in corpus,
    tokenizer and seq_len, hence the paper's control is recomputable for it.
    """
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for cell in cells:
        groups.setdefault((cell["frames"], cell["rows_per_frame"]),
                          []).append(cell)
    report = []
    for (frames, per_frame), members in sorted(groups.items()):
        counts = sorted({m["control_estimator"]["control_correct"]
                         for m in members})
        chaff = sorted({m["declared_chaff_tokens"] for m in members},
                       key=lambda v: (v is None, v))
        chaff_free = len(members) > 1 and len(counts) == 1
        for member in members:
            member["control_recomputable"] = (
                chaff_free or member["declared_chaff_tokens"] == 0)
        report.append({
            "frames": frames, "rows_per_frame": per_frame,
            "n_cells": len(members), "distinct_control_counts": len(counts),
            "control_counts": counts[:4],
            "declared_chaff_tokens": chaff,
            "chaff_free_by_invariance": chaff_free})
    return report


def find_artifacts(root: Path) -> list[Path]:
    found = sorted(p for p in root.rglob("*attacker*.json")
                   if "results" in json.loads(p.read_text()))
    if not found:
        raise SystemExit(f"{root}: no attacker artifacts found")
    return found


def summarise(cells: list[dict[str, Any]], root: str, gate_pp: float,
              budget_pp: float) -> dict[str, Any]:
    groups = control_invariance(cells)
    control = [c["control_estimator"] for c in cells]
    cluster = [c["clustering"] for c in cells]
    movable = [c for c in cells if c["control_estimator"]["verdict_can_change"]]
    return {
        "schema": SCHEMA, "root": root, "legacy_gate_pp": gate_pp,
        "floor_budget_pp": budget_pp, "n_cells": len(cells),
        "frame_shape_groups": groups,
        "n_control_recomputable": sum(c["control_recomputable"] for c in cells),
        "n_excess_matches_artifact":
            sum(c["excess_matches_artifact"] for c in control),
        "n_verdict_pass": sum(c["verdict"] == "PASS" for c in control),
        "n_verdict_fail": sum(c["verdict"] == "FAIL" for c in control),
        "n_control_fix_can_change_verdict": len(movable),
        "control_shift_needed_pp": sorted(
            c["control_estimator"]["control_shift_pp_needed"] for c in movable),
        "n_paired_pass": sum(c["paired_verdict"] == "PASS" for c in cluster),
        "n_paired_cluster_pass":
            sum(c["paired_cluster_verdict"] == "PASS" for c in cluster),
        "n_forensics_refined": sum(c["forensics_refined"] for c in cluster),
        "published_design_effect_to_fail": sorted(
            c["published_design_effect_to_fail"] for c in cluster),
        "cells": cells,
    }


HEADER = (f"{'excess':>9} {'floor':>8} {'gate':>6} {'rows':>6} {'shift pp':>9} "
          f"{'rec':>4} {'paired':>9} {'pairedLO':>9} {'deff*':>7} {'m':>4}"
          f"  cell")


def print_table(report: dict[str, Any]) -> None:
    print(HEADER)
    root = report["root"].rstrip("/") + "/"
    for cell in sorted(report["cells"],
                       key=lambda c: c["control_estimator"]["excess_pp"]):
        one, two = cell["control_estimator"], cell["clustering"]
        print(f"{one['excess_pp']:>+9.4f} {one['statistical_floor_pp']:>+8.4f} "
              f"{one['verdict']:>6} {one['control_rows_needed']:>6} "
              f"{one['control_shift_pp_needed']:>9.4f} "
              f"{'yes' if cell['control_recomputable'] else 'no':>4} "
              f"{two['paired_upper_pp']:>+9.4f} "
              f"{two['paired_upper_pp_best_case']:>+9.4f} "
              f"{two['published_design_effect_to_fail']:>7.2f} "
              f"{cell['rows_per_frame']:>4}  "
              f"{cell['file'].replace(root, '')}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", action="append", default=[])
    ap.add_argument("--sweep", help="directory searched recursively")
    ap.add_argument("--output", help="report JSON path")
    ap.add_argument("--gate-pp", type=float, default=LEGACY_GATE_PP)
    ap.add_argument("--budget-pp", type=float, default=FLOOR_BUDGET_PP)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--help-limits", action="store_true",
                    help="print what cannot be computed and why")
    ap.add_argument("--self-test", action="store_true")
    return ap


def self_test() -> int:
    """Arithmetic checks that do not depend on any artifact."""
    checks = [
        ("majority count integral", majority_count(4.9755859375, 20480) == 1019),
        ("paired identity b-c = A-M",
         paired_se_pp(0, 0, 20480) == 0.0),
        ("overlap range", overlap_bounds(1354, 1019, 20480, 19151) == (25, 1019)),
        ("cluster ss packs extremally",
         max_cluster_sum_squares(160, 0, 256, 80) == 2 * 80 * 80),
        ("design effect ceiling",
         cluster_se_pp(800, 800, 256, 80, 20480)
         <= math.sqrt(80) * paired_se_pp(1600, 0, 20480) * 1.01),
        ("gate leg 1 threshold",
         smallest_control_passing(7.066005801721505, 20480,
                                  z_bonferroni(arms=9), 1.0, 0.70) >= 1243),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in checks) else 1


def main() -> int:
    args = build_parser().parse_args()
    if args.help_limits:
        print(LIMITS)
        return 0
    if args.self_test:
        return self_test()
    paths = [Path(p) for p in args.artifact]
    if args.sweep:
        paths += find_artifacts(Path(args.sweep))
    if not paths:
        raise SystemExit("pass --artifact or --sweep")
    cells = [audit(p, args.gate_pp, args.budget_pp) for p in paths]
    root = args.sweep or ""
    report = summarise(cells, root, args.gate_pp, args.budget_pp)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    if not args.quiet:
        print_table(report)
        print(f"\ncells {report['n_cells']}  "
              f"excess re-derived {report['n_excess_matches_artifact']}  "
              f"PASS {report['n_verdict_pass']}  FAIL {report['n_verdict_fail']}  "
              f"control fix could move {report['n_control_fix_can_change_verdict']}"
              f"  paired PASS {report['n_paired_pass']}  "
              f"paired+cluster PASS {report['n_paired_cluster_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
