"""Build a label-free attack index from a complete remote transcript.

The index deliberately contains only paths and metadata already available to
the compromised cloud. It never joins in trusted-side labels, canonical
latents, gauge material, or private optimizer state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from privacy_runtime.remote_transcript import COLLECTION_SCHEMA, SCHEMA


INDEX_SCHEMA = "dtraining.attacker.full_remote_view_index.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_events(session_root: Path, manifest: dict) -> list[dict]:
    entries = {entry["path"]: entry for entry in manifest["files"]}
    event_entry = entries.get("events.jsonl")
    if event_entry is None:
        raise ValueError("session manifest has no event log")
    event_path = session_root / "events.jsonl"
    if _sha256(event_path) != event_entry["sha256"]:
        raise ValueError("event log digest does not match session manifest")
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    if [event["sequence"] for event in events] != list(range(len(events))):
        raise ValueError("session event sequence is not contiguous")
    return events


def _relative(root: Path, session_root: Path, event: dict) -> str:
    relative = event.get("file", {}).get("path")
    if not relative:
        raise ValueError(f"event {event['event']} has no payload file")
    path = session_root / relative
    if not path.is_file():
        raise ValueError(f"event payload is missing: {path}")
    if _sha256(path) != event["file"]["sha256"]:
        raise ValueError(f"event payload digest mismatch: {path}")
    return str(path.relative_to(root))


def _session_index(root: Path, session_root: Path, entry: dict) -> dict:
    manifest_path = session_root / "TRANSCRIPT_MANIFEST.json"
    if _sha256(manifest_path) != entry["manifest_sha256"]:
        raise ValueError(f"session manifest digest mismatch: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError(f"session is not a complete {SCHEMA} artifact")
    if manifest.get("session_id") != entry["session_id"]:
        raise ValueError("collection/session identity mismatch")
    events = _load_events(session_root, manifest)
    frames: dict[tuple[int, int], dict] = {}
    states = []
    controls = []
    for event in events:
        name = event["event"]
        if name in {"forward", "forward_result", "backward", "backward_result"}:
            key = (int(event["step"]), int(event["mb_id"]))
            frame = frames.setdefault(key, {"step": key[0], "mb_id": key[1]})
            if name in frame:
                raise ValueError(f"duplicate {name} for step/mb {key}")
            frame[name] = {
                "path": _relative(root, session_root, event),
                "sha256": event["file"]["sha256"],
                "bytes": event["file"]["bytes"],
                "header": event.get("header", {}),
                "recorded_monotonic_ns": event["recorded_monotonic_ns"],
            }
        elif name == "state_snapshot":
            states.append({
                "step": int(event["step"]), "reason": event["reason"],
                "path": _relative(root, session_root, event),
                "sha256": event["file"]["sha256"],
                "bytes": event["file"]["bytes"],
            })
        elif event.get("transport") == "text":
            controls.append({
                "event": name, "direction": event.get("direction"),
                "step": event.get("step"), "mb_id": event.get("mb_id"),
                "parsed": event.get("parsed"),
                "path": _relative(root, session_root, event),
                "sha256": event["file"]["sha256"],
            })
    required = {"forward", "forward_result", "backward", "backward_result"}
    for key, frame in frames.items():
        missing = required.difference(frame)
        if missing and frame.get("forward", {}).get("header", {}).get("training"):
            raise ValueError(f"training frame {key} is missing {sorted(missing)}")
    return {
        "session_id": manifest["session_id"],
        "frames": [frames[key] for key in sorted(frames)],
        "states": sorted(states, key=lambda state: (state["step"], state["reason"])),
        "controls": controls,
    }


def build_index(collection_root: str | Path) -> dict:
    root = Path(collection_root)
    collection = json.loads((root / "COLLECTION_MANIFEST.json").read_text())
    if collection.get("schema") != COLLECTION_SCHEMA:
        raise ValueError("unsupported collection schema")
    sessions = collection.get("sessions", [])
    if not sessions:
        raise ValueError("collection has no sessions")
    indexed = []
    for entry in sessions:
        path = Path(entry["manifest_path"])
        if path.is_absolute() or ".." in path.parts or path.name != "TRANSCRIPT_MANIFEST.json":
            raise ValueError(f"unsafe session manifest path: {path}")
        indexed.append(_session_index(root, root / path.parent, entry))
    return {
        "schema": INDEX_SCHEMA,
        "source_collection": "COLLECTION_MANIFEST.json",
        "sessions": indexed,
        "scope": "Remote-process messages, controls, timing, and remote state only; no trusted labels or secrets.",
    }


PAYLOAD_KINDS = ("forward", "forward_result", "backward", "backward_result")


def training_frames(session: dict) -> list[dict]:
    """Frames with all four payloads and a training header."""
    out = []
    for frame in session["frames"]:
        if not all(kind in frame for kind in PAYLOAD_KINDS):
            continue
        if not frame["forward"].get("header", {}).get("training", True):
            continue
        out.append(frame)
    return out


def view_final_window(index: dict, last_steps: int) -> dict:
    """The last `last_steps` training steps per session."""
    views = []
    for session in index["sessions"]:
        frames = training_frames(session)
        views.append({"session_id": session["session_id"], "view": "final_window",
                      "frames": frames[-last_steps:] if last_steps else frames})
    return {"view": "final_window", "last_steps": last_steps, "sessions": views}


def view_per_step(index: dict) -> dict:
    """Every training frame as its own attack unit (the W4.3 surface)."""
    views = [{"session_id": s["session_id"], "view": "per_step",
              "frames": training_frames(s)} for s in index["sessions"]]
    return {"view": "per_step", "sessions": views}


def view_accumulated(index: dict) -> dict:
    """Full frame history + state snapshots: the W4.5 accumulated surface."""
    views = [{"session_id": s["session_id"], "view": "accumulated",
              "frames": training_frames(s), "states": s["states"]}
             for s in index["sessions"]]
    return {"view": "accumulated", "sessions": views}


VIEWS = {"final-window": view_final_window, "per-step": view_per_step,
         "accumulated": view_accumulated}


def check_index_coverage(index: dict) -> dict:
    """Structural coverage of an index: what a full-view attack could consume.

    Structure-only: runs against a committed index without any payloads."""
    per_session = []
    for session in index["sessions"]:
        frames = training_frames(session)
        steps = sorted({f["step"] for f in frames})
        states = session["states"]
        state_steps = sorted({s["step"] for s in states})
        gaps = [b - a for a, b in zip(state_steps, state_steps[1:])]
        per_session.append({
            "session_id": session["session_id"],
            "training_frames": len(frames),
            "distinct_steps": len(steps),
            "step_range": [steps[0], steps[-1]] if steps else None,
            "four_payload_frames": len(frames),
            "state_snapshots": len(states),
            "state_interval_max": max(gaps) if gaps else None,
            "control_events": len(session["controls"]),
        })
    return {"schema": "dtraining.full_view_coverage.v1",
            "sessions": per_session,
            "total_training_frames": sum(s["training_frames"]
                                         for s in per_session)}


def self_test() -> int:
    import tempfile
    from privacy_runtime.remote_transcript import RemoteTranscript

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        transcript = RemoteTranscript(root, "fixture")
        transcript.record_text("received", '{"op":"hello"}', event="hello",
                               parsed={"op": "hello"})
        transcript.record_text("sent", '{"op":"hello_ack"}', event="hello_ack",
                               parsed={"op": "hello_ack"})
        transcript.record_state_bytes(b"initial", step=0, reason="initial")
        for event, direction in (("forward", "received"), ("forward_result", "sent"),
                                 ("backward", "received"), ("backward_result", "sent")):
            transcript.record_wire(direction, event.encode(), event=event,
                                   header={"op": event, "mb_id": 0, "training": True},
                                   step=0, mb_id=0)
        transcript.record_text("received", '{"op":"optimizer_step"}',
                               event="optimizer_step", step=0,
                               parsed={"op": "optimizer_step"})
        transcript.record_text("sent", '{"op":"step_ack"}', event="step_ack",
                               step=1, parsed={"op": "step_ack"})
        transcript.record_state_bytes(b"updated", step=1,
                                      reason="after_optimizer_step")
        transcript.record_text("received", '{"op":"close"}', event="close",
                               step=1, parsed={"op": "close"})
        transcript.finalize("complete")
        index = build_index(root)
        ok = (index["schema"] == INDEX_SCHEMA and len(index["sessions"]) == 1
              and len(index["sessions"][0]["frames"]) == 1
              and len(index["sessions"][0]["states"]) == 2)
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1
