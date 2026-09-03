#!/usr/bin/env python3
"""Summarize complete-view attack results with paired cluster bootstrap CIs.

Input is JSONL. Each record is one metric value for one held-out document or
block and must contain: arm, attack, metric, seed, cluster_id, and value.
The tool never treats rows inside a cluster as independent observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "dtraining.complete_view_matrix_summary.v1"
REQUIRED = {"arm", "attack", "metric", "seed", "cluster_id", "value"}


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile of an empty sample")
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_bootstrap(values: list[float], draws: int, seed: int) -> tuple[float, float]:
    """Flat bootstrap over exchangeable units. Retained for single-seed inputs only."""
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        means.append(sum(values[rng.randrange(len(values))]
                         for _ in values) / len(values))
    return quantile(means, 0.025), quantile(means, 0.975)


def hierarchical_bootstrap(by_seed: dict[int, list[float]], draws: int,
                           seed: int) -> tuple[float, float]:
    """Resample training seeds, then clusters within each resampled seed.

    The design is hierarchical: clusters scored by the same trained model are not
    independent of each other. Flattening seed x cluster into one exchangeable pool
    understates the interval, because it treats
    between-run variance as if it were within-run variance. The
    evaluation protocol requires the inferential unit to be the run or the
    document/block, never independently sampled rows.
    """
    seeds = sorted(by_seed)
    if len(seeds) < 2:
        pooled = [value for values in by_seed.values() for value in values]
        return cluster_bootstrap(pooled, draws, seed)
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        drawn = []
        for _ in seeds:
            picked = by_seed[seeds[rng.randrange(len(seeds))]]
            if not picked:
                continue
            drawn.append(sum(picked[rng.randrange(len(picked))]
                             for _ in picked) / len(picked))
        if drawn:
            means.append(sum(drawn) / len(drawn))
    if len(means) < 2:
        return float("nan"), float("nan")
    return quantile(means, 0.025), quantile(means, 0.975)


def stable_salt(*parts: str) -> int:
    data = "\x00".join(parts).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big")


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = REQUIRED.difference(row)
        if missing:
            raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
        if not isinstance(row["value"], (int, float)):
            raise ValueError(f"{path}:{line_number}: value must be numeric")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no result rows")
    return rows


def summarize(rows: list[dict], draws: int, seed: int) -> dict:
    values: dict[tuple[str, str, str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["arm"]), str(row["attack"]), str(row["metric"]),
               int(row["seed"]), str(row["cluster_id"]))
        values[key].append(float(row["value"]))
    units = {key: statistics.fmean(observations)
             for key, observations in values.items()}
    cells: dict[tuple[str, str, str], list[tuple[int, str, float]]] = defaultdict(list)
    for (arm, attack, metric, run_seed, cluster), value in units.items():
        cells[(arm, attack, metric)].append((run_seed, cluster, value))
    summaries = []
    for (arm, attack, metric), observations in sorted(cells.items()):
        unit_values = [value for _, _, value in observations]
        by_seed: dict[int, list[float]] = defaultdict(list)
        for run_seed, _, value in observations:
            by_seed[run_seed].append(value)
        summaries.append({
            "arm": arm, "attack": attack, "metric": metric,
            "clusters": len(unit_values),
            "training_seeds": sorted(by_seed),
            "mean": statistics.fmean(unit_values),
            "hierarchical_bootstrap95": hierarchical_bootstrap(
                dict(by_seed), draws, seed + stable_salt(arm, attack, metric)),
        })

    # Per-arm matched nulls. A single global shuffled_label baseline cannot express
    # separate naked-, defended-, and injected-view nulls, which is what experiments
    # W2.2 and W2.6 introduce. `baseline_for` on a row names its own null; rows
    # without one fall back to the global shuffled_label arm.
    baseline_of: dict[str, str] = {}
    for row in rows:
        named = row.get("baseline_for")
        if named:
            baseline_of[str(row["arm"])] = str(named)

    contrasts = []
    baselines = set(baseline_of.values()) | {"shuffled_label"}
    for arm, attack, metric in sorted(cells):
        if arm in baselines:
            continue
        baseline_arm = baseline_of.get(arm, "shuffled_label")
        candidate = {(run_seed, cluster): value
                     for run_seed, cluster, value in cells[(arm, attack, metric)]}
        baseline = {(run_seed, cluster): value
                    for run_seed, cluster, value in cells.get(
                        (baseline_arm, attack, metric), [])}
        paired = sorted(set(candidate).intersection(baseline))
        if not paired:
            continue
        differences = [candidate[key] - baseline[key] for key in paired]
        by_seed_diff: dict[int, list[float]] = defaultdict(list)
        for (run_seed, _cluster), difference in zip(paired, differences):
            by_seed_diff[run_seed].append(difference)
        contrasts.append({
            "arm": arm, "baseline": baseline_arm,
            "attack": attack, "metric": metric,
            "paired_clusters": len(paired),
            "effect": statistics.fmean(differences),
            "paired_hierarchical_bootstrap95": hierarchical_bootstrap(
                dict(by_seed_diff), draws,
                seed + stable_salt(arm, attack, metric, "contrast")),
        })
    return {
        "schema": SCHEMA,
        "statistical_unit": "training seed (outer), held-out document/block cluster within seed (inner)",
        "cells": summaries,
        "contrasts": contrasts,
    }


def self_test() -> int:
    checks = {}

    rows = []
    for arm, shift in (("defended", 0.02), ("shuffled_label", 0.0)):
        for run_seed in (42, 43, 44):
            for cluster in range(8):
                rows.append({"arm": arm, "attack": "gradient_only",
                             "metric": "token_top1", "seed": run_seed,
                             "cluster_id": f"doc-{cluster}",
                             "value": 0.1 + shift + 0.001 * cluster})
    report = summarize(rows, draws=200, seed=7)
    contrast = report["contrasts"][0]
    checks["basic_contrast"] = (len(report["cells"]) == 2
                                and contrast["paired_clusters"] == 24
                                and 0.015 < contrast["effect"] < 0.025)
    checks["baseline_defaults_to_shuffled_label"] = contrast["baseline"] == "shuffled_label"

    # Per-arm matched nulls: naked and defended each get their own baseline.
    per_arm = []
    for arm, base, shift in (("naked_full_width", "naked_null", 0.05),
                             ("naked_null", None, 0.0),
                             ("defended", "defended_null", 0.01),
                             ("defended_null", None, 0.0)):
        for run_seed in (42, 43, 44):
            for cluster in range(8):
                row = {"arm": arm, "attack": "gradient_only", "metric": "token_top1",
                       "seed": run_seed, "cluster_id": f"doc-{cluster}",
                       "value": 0.1 + shift}
                if base:
                    row["baseline_for"] = base
                per_arm.append(row)
    report2 = summarize(per_arm, draws=200, seed=7)
    pairs = {c["arm"]: c["baseline"] for c in report2["contrasts"]}
    checks["per_arm_baselines_respected"] = (
        pairs.get("naked_full_width") == "naked_null"
        and pairs.get("defended") == "defended_null")
    checks["baseline_arms_not_contrasted_against_themselves"] = (
        "naked_null" not in pairs and "defended_null" not in pairs)

    # The hierarchical interval must be wider than the flat one when between-run
    # variance dominates: three seeds far apart, tight within each seed.
    spread = {42: [0.10] * 8, 43: [0.20] * 8, 44: [0.30] * 8}
    flat = [v for values in spread.values() for v in values]
    flat_low, flat_high = cluster_bootstrap(flat, 2000, 3)
    hier_low, hier_high = hierarchical_bootstrap(spread, 2000, 3)
    checks["hierarchical_interval_wider_than_flat"] = (
        (hier_high - hier_low) > (flat_high - flat_low))

    for name, passed in checks.items():
        print(f"  {'ok  ' if passed else 'FAIL'} {name}")
    ok = all(checks.values())
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        parser.error("--bootstrap-draws must be at least 100")
    report = summarize(load_rows(args.input), args.bootstrap_draws,
                       args.bootstrap_seed)
    report["input"] = str(args.input)
    report["bootstrap_draws"] = args.bootstrap_draws
    report["bootstrap_seed"] = args.bootstrap_seed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test() if "--self-test" in __import__("sys").argv else main())
