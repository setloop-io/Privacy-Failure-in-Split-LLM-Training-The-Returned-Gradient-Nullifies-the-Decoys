#!/usr/bin/env python3
"""Tabulate latent_v7 sweep results: runner gates + external attacker excess."""
import json
import sys
from pathlib import Path

outdir = Path("/workspace/experiments/results/training")
rows = []
for path in sorted(outdir.glob("latent_v7_*.json")):
    name = path.stem
    if name.endswith("_attacker") or "_repro_" in name:
        continue
    try:
        r = json.loads(path.read_text())
    except Exception:
        continue
    attacker_path = path.with_name(name + "_attacker.json")
    excess = worst = None
    if attacker_path.exists():
        a = json.loads(attacker_path.read_text())
        summary = a["summary"]
        if isinstance(summary, list):
            summary = summary[0]
        excess = summary["upper95_excess_over_majority_pp"]
        worst = summary["worst_bonferroni_upper95_pct"]
    rows.append({
        "cell": name.replace("latent_v7_", ""),
        "loss_delta": r.get("eval_loss_delta"),
        "time_ratio": r.get("eval_time_ratio"),
        "probe_max_pct": max(r.get("probe_recovery_pct", [float("nan")])),
        "majority_pct": r.get("label_free_majority_pct"),
        "upper95_excess_pp": excess,
        "worst_upper95_pct": worst,
        "gates_all": all(r.get("gates", {}).values()),
        "chaff": r.get("chaff_tokens", 0),
        "quant": r.get("wire_quant", "none"),
        "noise": r.get("noise_multiplier"),
        "sigma": r.get("token_scale_sigma"),
        "clip": r.get("remote_grad_clip"),
    })

if "--json" in sys.argv:
    print(json.dumps(rows, indent=1))
    sys.exit(0)

header = ("cell", "loss_d", "ratio", "probe%", "maj%",
          "exc_pp", "gates", "chaff", "quant")
print("{:<22} {:>7} {:>6} {:>7} {:>6} {:>8} {:>6} {:>6} {:>5}".format(*header))
for row in rows:
    exc = ("%.3f" % row["upper95_excess_pp"]) if row["upper95_excess_pp"] is not None else "-"
    print("{:<22} {:>7.3f} {:>6.3f} {:>7.2f} {:>6.2f} {:>8} {:>6} {:>6} {:>5}".format(
        row["cell"], row["loss_delta"], row["time_ratio"], row["probe_max_pct"],
        row["majority_pct"], exc, str(row["gates_all"]), row["chaff"], row["quant"]))
