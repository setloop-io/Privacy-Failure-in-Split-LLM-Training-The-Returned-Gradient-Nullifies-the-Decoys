#!/usr/bin/env python3
"""W5.1: the post-freeze confirmation matrix.

For each of the nine protocol attack families x five required arms x three
seeds, score the constructible cells against the frozen thresholds on the
verified w34 index, and declare the rest unmeasured per the W2.6 family map.
Uses paired_advantage where dumps exist; the verified index supplies the
frozen-protocol arms.

CONSTRUCTIBILITY RULE: a cell is constructible only if
its family has BOTH an applicable metric AND a positive control in
family_metric_map.json. The map's own header: "A family/metric pair without a
control may not appear in a headline claim", and the freeze record excludes
families without a representation-matched positive control from the
confirmation matrix.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

RULE = ("constructible requires an applicable metric AND a positive control "
        "(family_metric_map.json; freeze record: no headline cell without a "
        "representation-matched positive control)")


def cell_status(fam_info: dict) -> tuple[str, str | None]:
    """A family without a positive control is unmeasured even if a metric applies."""
    applicable = fam_info.get("applicable_metrics", [])
    if not applicable:
        return "unmeasured", fam_info.get("note", "no applicable metric")
    if not fam_info.get("positive_control"):
        return "unmeasured", ("applicable metric but no positive control: "
                              + fam_info.get("note", ""))
    return "constructible", None


def build_matrix(index_path: Path, protocol_path: Path, family_map_path: Path,
                 out_path: Path) -> dict:
    index = json.loads(index_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    family_map = json.loads(family_map_path.read_text())
    arms = protocol["required_arms"]
    families = protocol["attack_families"]

    cells = []
    for family in families:
        fam_info = family_map["families"].get(family, {})
        applicable = fam_info.get("applicable_metrics", [])
        status, reason = cell_status(fam_info)
        for arm in arms:
            for seed in (42, 43, 44):
                cell = {"family": family, "arm": arm, "seed": seed,
                        "status": status}
                if status == "unmeasured":
                    cell["reason"] = reason
                else:
                    cell["metrics"] = applicable
                    cell["positive_control"] = fam_info.get("positive_control")
                cells.append(cell)

    measured = sum(1 for c in cells if c["status"] == "constructible")
    result = {
        "schema": "dtraining.w51_confirmation_matrix.v2",
        "constructibility_rule": RULE,
        "index_source": str(index_path),
        "arms": arms, "families": families,
        "seeds": [42, 43, 44],
        "cells": cells,
        "constructible": measured,
        "unmeasured": len(cells) - measured,
        "frozen_thresholds": family_map["metrics"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def self_test() -> int:
    """The corrected rule: metric-without-control is unmeasured."""
    family_map = {"metrics": {}, "families": {
        "f_ok": {"applicable_metrics": ["token_top1"],
                 "positive_control": "injected_leak", "note": ""},
        "f_metric_no_control": {"applicable_metrics": ["token_top1"],
                                "positive_control": None, "note": "no control"},
        "f_no_metric": {"applicable_metrics": [], "positive_control": None,
                        "note": "no metric"},
    }}
    protocol = {"required_arms": ["arm_a", "arm_b"],
                "attack_families": list(family_map["families"])}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for name, payload in (("index", {}), ("protocol", protocol),
                              ("fam", family_map)):
            (td / f"{name}.json").write_text(json.dumps(payload))
        result = build_matrix(td / "index.json", td / "protocol.json",
                              td / "fam.json", td / "out.json")
    # 2 arms x 3 seeds per family: f_ok constructible (6), others unmeasured.
    checks = [
        (result["constructible"] == 6, "metric+control family is constructible"),
        (result["unmeasured"] == 12, "metric-without-control and no-metric are unmeasured"),
        (any(c["family"] == "f_metric_no_control"
             and "no positive control" in c.get("reason", "")
             for c in result["cells"]),
         "metric-without-control reason names the missing control"),
        (result["constructibility_rule"] == RULE, "the rule is recorded in the artifact"),
    ]
    ok = True
    for passed, label in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    root = Path("paper-data")
    result = build_matrix(
        root / "collected/diagnostic/w34_complete/w34_index.json",
        root / "evaluation_protocol.json",
        root / "family_metric_map.json",
        root / "collected/diagnostic/w51_confirmation_matrix.json")
    print(f"matrix: {result['constructible']} constructible, "
          f"{result['unmeasured']} unmeasured of {len(result['cells'])} cells")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(self_test())
    raise SystemExit(main())
