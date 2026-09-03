#!/usr/bin/env python3
"""Aggregate E-A4 attacker metrics across independent training seeds."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
       6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
       16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
       21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
       26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def summarize(values):
    n = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    critical = T95.get(n - 1, 1.96)
    half_width = critical * std / math.sqrt(n)
    return {"values": values, "mean": mean, "sample_std": std,
            "mean_95ci": [mean - half_width, mean + half_width]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True,
                        metavar="TRAINING_SEED=RESULT.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runs = []
    for item in args.input:
        seed_text, path_text = item.split("=", 1)
        path = Path(path_text)
        data = json.loads(path.read_text())
        runs.append({"training_seed": int(seed_text), "path": str(path),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "attacker_seeds": data["seeds"], "data": data})
    seeds = [run["training_seed"] for run in runs]
    if len(runs) < 3 or len(set(seeds)) != len(seeds):
        parser.error("at least three distinct training seeds are required")
    if any(run["attacker_seeds"] != runs[0]["attacker_seeds"] for run in runs):
        parser.error("attacker seeds differ across training runs")

    metrics = {}
    for condition in ("split_ft", "fedavg"):
        metrics[condition] = {}
        for attack in ("membership", "property"):
            metrics[condition][attack] = {}
            for metric in ("roc_auc_mean", "tpr_at_1pct_fpr_mean"):
                values = [run["data"]["conditions"][condition][attack][metric]
                          for run in runs]
                metrics[condition][attack][metric] = summarize(values)

    output = {
        "schema": "dtraining.ea4.training_seed_aggregate.v1",
        "experiment": "E-A4 membership/property inference repeatability",
        "training_seeds": seeds,
        "attacker_seeds_per_training_run": runs[0]["attacker_seeds"],
        "n_independent_training_runs": len(runs),
        "inputs": [{k: run[k] for k in ("training_seed", "path", "sha256")}
                   for run in runs],
        "metrics": metrics,
        "ci_method": "two-sided Student t interval across training-seed means",
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
