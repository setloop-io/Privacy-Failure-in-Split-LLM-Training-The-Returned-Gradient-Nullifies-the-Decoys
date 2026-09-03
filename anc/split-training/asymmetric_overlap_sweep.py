#!/usr/bin/env python3
"""Sweep split points (local vs cloud layers) and compute pipeline overlap efficiency."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from trained_inversion import make_provenance

def run_split_trainer(split_after, resume_after, pipeline, rtt_ms):
    base = Path(__file__).resolve().parent
    output_file = base / f"experiment_data/asymmetric_sweep_sa{split_after}_ra{resume_after}_{pipeline}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, str(base / "split_trainer.py"),
        "--toy",
        "--smoke",
        "--split-after", str(split_after),
        "--resume-after", str(resume_after),
        "--pipeline", pipeline,
        "--steps", "5",
        "--output", str(output_file)
    ]
    
    start_time = time.time()
    subprocess.run(cmd, check=True)
    total_time = time.time() - start_time
    
    with open(output_file) as f:
        res = json.load(f)
        
    return res

def analyze_timings(res, rtt_ms):
    mb_timings = res.get("microbatch_timing", [])
    if not mb_timings:
        return {}
        
    t_local_fwd = sum(t["t_local_fwd"] for t in mb_timings) / len(mb_timings)
    t_local_bwd = sum(t["t_local_bwd"] for t in mb_timings) / len(mb_timings)
    t_tail_fwd = sum(t["t_tail_fwd"] for t in mb_timings) / len(mb_timings)
    t_head_bwd = sum(t["t_head_bwd"] for t in mb_timings) / len(mb_timings)
    
    t_local_total = t_local_fwd + t_local_bwd + t_tail_fwd + t_head_bwd
    
    t_cloud_fwd = sum(t.get("t_cloud_fwd", 0) for t in mb_timings) / len(mb_timings)
    t_cloud_bwd = sum(t.get("t_cloud_bwd", 0) for t in mb_timings) / len(mb_timings)
    
    # Comm time = 2 * RTT (fwd + bwd)
    t_comm = (rtt_ms * 2) / 1000.0
    
    t_cloud_compute = max(0, (t_cloud_fwd + t_cloud_bwd) - t_comm)
    
    # overlap efficiency: 1.0 when local compute exactly hides the RTT
    step_time = sum(s["t_step"] for s in res["steps"]) / len(res["steps"])
    diff = abs(t_local_total - t_comm)
    efficiency = 1.0 - (diff / max(1e-6, step_time))
    
    return {
        "t_local_total_s": round(t_local_total, 4),
        "t_cloud_compute_s": round(t_cloud_compute, 4),
        "t_comm_s": t_comm,
        "step_time_s": round(step_time, 4),
        "overlap_efficiency": round(max(0.0, min(1.0, efficiency)), 3)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtt", type=int, default=80, help="Simulated WAN RTT in ms")
    parser.add_argument("--output", default="experiment_data/asymmetric_overlap_sweep.json")
    args = parser.parse_args()

    # toy model has 4 layers (0..3); sweep head/cloud/tail split points
    splits = [
        {"sa": 0, "ra": 2},
        {"sa": 0, "ra": 3},
        {"sa": 1, "ra": 3}
    ]

    results = []

    for s in splits:
        print(f"--- Running sweep sa={s['sa']} ra={s['ra']} ---")
        sync_res = run_split_trainer(s["sa"], s["ra"], "sync", args.rtt)
        sync_metrics = analyze_timings(sync_res, args.rtt)
        
        results.append({
            "split_after": s["sa"],
            "resume_after": s["ra"],
            "pipeline": "sync",
            "metrics": sync_metrics
        })
        print(f"  Sync Step Time: {sync_metrics['step_time_s']}s, Efficiency: {sync_metrics['overlap_efficiency']}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out = {
        "schema": "dtraining.asymmetric_overlap.analytical.v2",
        "experiment": "asymmetric overlap planning sweep",
        "config": {**vars(args), "measurement_kind": "analytical"},
        "evidence_status": "supporting",
        "known_limitations": [
            "Toy-model compute is measured locally, while WAN communication is inserted analytically from the requested RTT.",
            "The output is a planning estimate and must not be reported as a distributed WAN measurement."
        ],
        "provenance": make_provenance(None, "toy_synthetic", 0, [],
                                      model_path=None),
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
