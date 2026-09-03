#!/usr/bin/env python3
"""Compare a packaged-code E1 re-run against the committed E1 seeds (experiment W1.7).

W1.7's acceptance criterion is reproduction of the E1 *finding*, not of its
bits: this package carries the W1.3 fragmentation fix and the tree that
produced E1 does not, so the cloud module initialises from a different
seed. The test is therefore whether the re-run lands inside the spread the
committed seeds already show.

    python3 bin/e1_reproduction_report.py --repro-dir <dir> --repro-cell <name>

Reads only committed artifacts plus the re-run's own, and prints a table plus a
machine-readable verdict. Every number is re-derived here; none is copied from a
document.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "paper-data" / "collected" / "diagnostic" / "e1_unprotected"
BASELINE_CELLS = ("e1_unprot_a2b_split14", "e1_unprot_a2b_split14_s43",
                  "e1_unprot_a2b_split14_s44")
ARMS = ("grad_real", "grad_real_shuffled", "wire_real")

# METRIC_VARIANCE.md records sd 0.0481 pp for excess under identical
# configuration. Three sd is the band a re-run must land inside.
EXCESS_SD_PP = 0.0481
SD_MULTIPLE = 3.0


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def arm_reading(directory: Path, cell: str, arm: str) -> dict | None:
    """Lift and paired advantage for one arm, re-derived from its artifacts."""
    scored = load(directory / f"{cell}_arm_{arm}.json")
    if scored is None:
        return None
    summary = scored["summary"][0]
    lift = summary["best_probe_top1_pct"] - summary["label_free_majority_pct"]
    reading = {"lift_pp": round(lift, 4),
               "u95_excess_pp": round(summary["upper95_excess_over_majority_pp"], 4)}

    paired = load(directory / f"{cell}_arm_{arm}_paired.json")
    if paired is not None:
        best = paired["best_eligible"]
        reading["paired_pp"] = round(best["paired_advantage_pp"], 4)
        reading["paired_ci95"] = [round(best["ci95_low_pp"], 4),
                                  round(best["ci95_high_pp"], 4)]
        reading["verdict"] = paired["verdict"]
    return reading


def cell_reading(directory: Path, cell: str) -> dict:
    """Everything W1.7 checks for one cell."""
    result = load(directory / f"{cell}.json")
    bundles = load(directory / f"{cell}_bundles.json")
    reading = {"cell": cell,
               "arms": {arm: arm_reading(directory, cell, arm) for arm in ARMS}}
    if result is not None:
        reading["dp_releases"] = sorted(result["dp"]["releases"])
        reading["outbound_grad_dp"] = result.get("outbound_grad_dp")
        reading["train_seconds"] = round(result["train_seconds"], 1)
    if bundles is not None:
        leak = bundles["support_leak"]
        reading["support_leak"] = {
            "frames": leak["frames"],
            "frames_with_exact_agreement": leak["frames_with_exact_agreement"],
            "row_agreement": round(leak["row_agreement_zero_support_vs_real"], 4),
        }
    return reading


def spread(values: list[float]) -> tuple[float, float]:
    return (min(values), max(values))


def verdict(repro: dict, baselines: list[dict]) -> dict:
    """Did the finding reproduce? Each check is pass/fail with its evidence."""
    checks: dict[str, dict] = {}

    checks["gradient_unaccounted"] = {
        "pass": repro.get("dp_releases") == ["forward", "return"]
                and repro.get("outbound_grad_dp") is None,
        "got": repro.get("dp_releases"),
    }

    leak = repro.get("support_leak") or {}
    checks["chaff_partition_disclosed"] = {
        "pass": leak.get("frames", 0) > 0
                and leak.get("frames") == leak.get("frames_with_exact_agreement")
                and leak.get("row_agreement") == 1.0,
        "got": leak,
    }

    for arm in ARMS:
        repro_arm = (repro["arms"].get(arm) or {})
        base = [b["arms"][arm]["paired_pp"] for b in baselines
                if (b["arms"].get(arm) or {}).get("paired_pp") is not None]
        value = repro_arm.get("paired_pp")
        if value is None or not base:
            checks[f"{arm}_paired"] = {"pass": False, "got": value,
                                       "baseline": base}
            continue
        low, high = spread(base)
        band = SD_MULTIPLE * EXCESS_SD_PP
        checks[f"{arm}_paired"] = {
            "pass": low - band <= value <= high + band,
            "got": value,
            "baseline_spread": [round(low, 4), round(high, 4)],
            "band_pp": round(band, 4),
            "repro_verdict": repro_arm.get("verdict"),
            "baseline_verdicts": sorted({b["arms"][arm]["verdict"]
                                         for b in baselines
                                         if b["arms"].get(arm)}),
        }

    null = checks.get("grad_real_shuffled_paired", {})
    checks["null_at_floor"] = {
        "pass": (repro["arms"].get("grad_real_shuffled") or {}).get("verdict")
                == "at-floor",
        "got": (repro["arms"].get("grad_real_shuffled") or {}).get("verdict"),
        "note": "any arm named shuffled is a null and must read at-floor",
    }
    del null

    return {"checks": checks,
            "reproduced": all(check["pass"] for check in checks.values())}


def render(repro: dict, baselines: list[dict], report: dict) -> None:
    print(f"E1 REPRODUCTION FROM PACKAGED CODE -- {repro['cell']}\n")
    header = f"{'arm':<22}{'repro paired':>14}{'committed seeds':>34}"
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        got = (repro["arms"].get(arm) or {}).get("paired_pp")
        base = [(b["arms"].get(arm) or {}).get("paired_pp") for b in baselines]
        base_text = "  ".join("--" if v is None else f"{v:+.4f}" for v in base)
        got_text = "--" if got is None else f"{got:+.4f}"
        print(f"{arm:<22}{got_text:>14}{base_text:>34}")

    print(f"\ndp.releases        {repro.get('dp_releases')}")
    print(f"support_leak       {repro.get('support_leak')}")
    print(f"train_seconds      {repro.get('train_seconds')}  "
          f"(committed: {[b.get('train_seconds') for b in baselines]})")
    print("\nchecks")
    for name, check in report["checks"].items():
        print(f"  {'PASS' if check['pass'] else 'FAIL'}  {name}")
        if not check["pass"]:
            print(f"        {json.dumps(check)}")
    print(f"\nREPRODUCED: {report['reproduced']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro-dir", type=Path, required=True)
    ap.add_argument("--repro-cell", required=True, nargs="+",
                    help="one or more packaged-code reproduction cells")
    ap.add_argument("--baseline-dir", type=Path, default=COMMITTED)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    repros = [cell_reading(args.repro_dir, cell) for cell in args.repro_cell]
    baselines = [cell_reading(args.baseline_dir, cell)
                 for cell in BASELINE_CELLS]

    reports = []
    for repro in repros:
        report = verdict(repro, baselines)
        render(repro, baselines, report)
        reports.append({"repro": repro, **report})
        print()

    # Cross-seed reading: does the packaged distribution separate from the
    # committed one?  With n <= 3 per side this is descriptive, not a test.
    if len(repros) > 1:
        print("packaged seeds (paired pp)")
        for arm in ARMS:
            values = [r["arms"][arm]["paired_pp"] for r in repros
                      if (r["arms"].get(arm) or {}).get("paired_pp") is not None]
            base = [b["arms"][arm]["paired_pp"] for b in baselines
                    if (b["arms"].get(arm) or {}).get("paired_pp") is not None]
            if values and base:
                print(f"  {arm:<22} packaged {['%+.4f' % v for v in values]}"
                      f"  committed {['%+.4f' % v for v in base]}")

    if args.output:
        args.output.write_text(json.dumps(
            {"schema": "dtraining.e1_reproduction.v1",
             "repro_cells": args.repro_cell,
             "seeds": reports,
             "baselines": baselines,
             "reproduced": all(r["reproduced"] for r in reports)},
            indent=2, sort_keys=True) + "\n")
        print(f"\n[artifact] wrote {args.output}")
    return 0 if all(r["reproduced"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
