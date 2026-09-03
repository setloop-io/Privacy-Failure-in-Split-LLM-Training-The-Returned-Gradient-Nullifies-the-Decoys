#!/usr/bin/env python3
"""Validate the pre-freeze complete-view evaluation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "dtraining.complete_view_evaluation_protocol.v1"
ARMS = {"naked_no_randomization", "naked_full_width", "defended",
        "shuffled_label", "injected_leak"}
ATTACKS = {"forward_only", "gradient_only", "joint_forward_gradient",
           "accumulated_history", "stateful_remote_state", "timing_metadata",
           "active_perturbation", "membership_property", "response_side"}
METRICS = {"token_top1", "token_cross_entropy", "rare_token_top1",
           "sequence_exact_match", "semantic_recovery", "membership_auc",
           "property_auc", "response_side_recovery"}


def validate(protocol: dict) -> dict:
    if protocol.get("schema") != SCHEMA:
        raise ValueError("unsupported evaluation protocol schema")
    if set(protocol.get("required_arms", {})) != ARMS:
        raise ValueError("protocol must define exactly the five control arms of PLAN.md W2.2")
    if set(protocol.get("attack_families", [])) != ATTACKS:
        raise ValueError("protocol must enumerate the complete attack matrix")
    if set(protocol.get("metrics", [])) != METRICS:
        raise ValueError("protocol must enumerate the required leakage metrics")
    plan = protocol.get("statistical_plan", {})
    if plan.get("minimum_independent_training_seeds", 0) < 3:
        raise ValueError("protocol requires at least three independent training seeds")
    if protocol.get("status") not in {"pending_freeze", "frozen", "complete"}:
        raise ValueError("invalid protocol status")
    if protocol.get("status") == "frozen" and not protocol.get("candidate_frozen"):
        raise ValueError("frozen protocol must state that the candidate is frozen")
    emitters = protocol.get("metric_emitters", {})
    missing = METRICS - set(emitters)
    if missing:
        raise ValueError(
            "protocol must declare an emitter for every metric (W2.3); "
            f"missing: {sorted(missing)}")
    for metric, emitter in emitters.items():
        if metric not in METRICS:
            raise ValueError(f"emitter declared for undeclared metric: {metric}")
        if emitter.get("implemented") and not emitter.get("producer"):
            raise ValueError(f"emitter for {metric} must name its producer")
        if not emitter.get("implemented") and not emitter.get("unavailable_reason"):
            raise ValueError(
                f"emitter for {metric} must either be implemented or state "
                "the reason it cannot be computed (W2.3: no silent gaps)")
    return {
        "status": protocol["status"],
        "candidate_frozen": bool(protocol["candidate_frozen"]),
        "arms": len(ARMS), "attacks": len(ATTACKS), "metrics": len(METRICS),
        "emitters_implemented": sum(1 for e in emitters.values() if e.get("implemented")),
        "emitters_unavailable": sorted(m for m, e in emitters.items()
                                       if not e.get("implemented")),
        "ready_for_confirmation": protocol["status"] == "frozen"
    }


def self_test() -> int:
    protocol = json.loads((ROOT / "paper-data" / "evaluation_protocol.json").read_text())
    report = validate(protocol)
    # The self-test asserts the protocol is internally consistent for its
    # declared status. After F.1 the protocol is frozen, so the invariant is
    # ready_for_confirmation == (status == "frozen") with candidate_frozen set.
    consistent = (report["ready_for_confirmation"]
                  == (report["status"] == "frozen"))
    if report["status"] == "frozen":
        consistent = consistent and report["candidate_frozen"]
    ok = report["arms"] == 5 and consistent
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "paper-data" / "evaluation_protocol.json")
    args = parser.parse_args()
    print(json.dumps(validate(json.loads(args.protocol.read_text())),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test() if "--self-test" in __import__("sys").argv else main())
