#!/usr/bin/env python3
"""full-history: the full-remote-view consumer attack (W4.1/W4.5).

The first registered attack that consumes the complete-view index. It selects
payload references for one of three views over the verified transcript —
final-window, per-step, accumulated — and reports what the view contains and
whether payloads are present. Payload tensors are never in the index (it is
label-free by construction); when they are absent the attack says exactly what
it needs instead of silently degrading.

Torch-free: this attack computes no features yet (that is per-family feature
extraction, scored downstream); it establishes that the full view is
consumable and complete.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .. import artifacts
from ..full_view import (INDEX_SCHEMA, PAYLOAD_KINDS, build_index,
                         check_index_coverage, view_accumulated,
                         view_final_window, view_per_step)
from .common import add_common_args

EXPERIMENT_ID = "full_history"
MODES = ("training",)
REQUIRES_LABELS = False
DESCRIPTION = ("full-remote-view consumer: final-window / per-step / "
               "accumulated-history reads of a verified complete transcript")

VIEWS = {"final-window": view_final_window, "per-step": view_per_step,
         "accumulated": view_accumulated}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--index", help="full_remote_view_index.v1 JSON")
    ap.add_argument("--view", choices=sorted(VIEWS), default="accumulated")
    ap.add_argument("--last-steps", type=int, default=512)
    return ap


def _fixture_collection(root: Path, sessions: int = 2, steps: int = 3) -> None:
    """A synthetic complete collection through the real transcript writer."""
    from privacy_runtime.remote_transcript import RemoteTranscript
    for name in [f"fixture_{i}" for i in range(sessions)]:
        transcript = RemoteTranscript(root, name)
        for step in range(steps):
            for event, direction in (("forward", "received"),
                                     ("forward_result", "sent"),
                                     ("backward", "received"),
                                     ("backward_result", "sent")):
                payload = f"{name}-{event}-{step}".encode()
                transcript.record_wire(
                    direction, payload, event=event,
                    header={"op": event, "mb_id": 0, "training": True},
                    step=step, mb_id=0)
        transcript.record_state_bytes(b"s0", step=0, reason="initial")
        transcript.finalize("complete")


def run(args) -> int:
    if args.toy:
        with tempfile.TemporaryDirectory() as directory:
            _fixture_collection(Path(directory))
            index = build_index(directory)
    else:
        if not args.index:
            raise SystemExit("--index is required (or --toy)")
        index_path = Path(args.index)
        index = json.loads(index_path.read_text())
        if index.get("schema") != INDEX_SCHEMA:
            raise SystemExit(f"{index_path} is not a {INDEX_SCHEMA} index")

    coverage = check_index_coverage(index)
    view = (view_final_window(index, args.last_steps)
            if args.view == "final-window" else VIEWS[args.view](index))
    missing = 0
    base = Path(args.index).parent if args.index else None
    if base is not None:
        for session in view["sessions"]:
            for frame in session["frames"]:
                for kind in PAYLOAD_KINDS:
                    if not (base / frame[kind]["path"]).is_file():
                        missing += 1

    out = artifacts.make_artifact(
        "dtraining.attacker.full_history.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "view": args.view,
         "last_steps": args.last_steps,
         "index": str(args.index) if args.index else "toy"},
        "Label-free full-remote-view consumer. The index carries paths and "
        "metadata only; no trusted labels, canonical latents, or gauge "
        "material are joined.",
        interpretation="payload_mode 'blocked' means the raw transcript is not "
                       "on this host; the view then carries references, not "
                       "tensors.")
    out["results"].append({"coverage": coverage,
                           "view_frames": [len(s["frames"])
                                           for s in view["sessions"]],
                           "payloads_missing": missing,
                           "payload_mode": "blocked" if missing else "ready"})
    if missing and not args.toy:
        out["summary"].append({
            "blocker": "payload tensors live in the raw transcript collection "
                       "(w34: poseidon, 1.9 GB); the committed index supports "
                       "all three views but no payload feature can be computed "
                       "until the transcript is reachable"})
    artifacts.write_artifact(args.output, out)
    return 0


def self_test() -> int:
    with tempfile.TemporaryDirectory() as directory:
        _fixture_collection(Path(directory), sessions=2, steps=3)
        index = build_index(directory)
        checks = [
            check_index_coverage(index)["total_training_frames"] == 6,
            len(view_per_step(index)["sessions"]) == 2,
            len(view_final_window(index, 1)["sessions"][0]["frames"]) == 1,
            len(view_accumulated(index)["sessions"][0]["states"]) == 1,
        ]
    ok = all(checks)
    print(f"  [{'PASS' if ok else 'FAIL'}] full-history fixture: two sessions, "
          f"three views, coverage exact")
    return 0 if ok else 1
