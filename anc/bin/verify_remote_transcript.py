#!/usr/bin/env python3
"""Verify the integrity and minimum coverage of one remote transcript bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from privacy_runtime.remote_transcript import (COLLECTION_SCHEMA, SCHEMA,
                                               quarantine_session)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _state_interval(manifest: dict) -> int:
    """Snapshot thinning declared by the recorder.

    latent_cloud_server.py's --transcript-state-interval N persists a remote
    state checkpoint only every N optimizer steps; the recorder declares N in
    the manifest so this stays verifiable (1 = every step, the only mode the
    original recorder had).
    """
    interval = manifest.get("state_interval", 1)
    if (not isinstance(interval, int) or isinstance(interval, bool)
            or interval < 1):
        raise ValueError(f"invalid state_interval: {interval!r}")
    return interval


def _cross_check_session_json(root: Path, seen: set, manifest: dict,
                              interval: int) -> None:
    if Path("session.json") not in seen:
        return
    meta = json.loads((root / "session.json").read_text())
    if meta.get("session_id") != manifest.get("session_id"):
        raise ValueError("session.json session_id disagrees with manifest")
    if meta.get("state_interval", interval) != interval:
        raise ValueError("session.json state_interval disagrees with manifest")


def verify(root: Path) -> dict:
    manifest_path = root / "TRANSCRIPT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported transcript schema: {manifest.get('schema')}")
    if manifest.get("status") != "complete":
        raise ValueError(f"transcript is not complete: {manifest.get('status')}")
    interval = _state_interval(manifest)

    seen = set()
    for entry in manifest.get("files", []):
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path: {relative}")
        if relative in seen:
            raise ValueError(f"duplicate manifest path: {relative}")
        seen.add(relative)
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing captured file: {relative}")
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"integrity check failed: {relative}")
    if Path("events.jsonl") not in seen:
        raise ValueError("manifest does not cover events.jsonl")
    _cross_check_session_json(root, seen, manifest, interval)

    events_path = root / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    if len(events) != manifest.get("event_count"):
        raise ValueError("event count does not match manifest")
    if [event["sequence"] for event in events] != list(range(len(events))):
        raise ValueError("event sequence is not contiguous")

    names = [event["event"] for event in events]
    required = {"hello", "hello_ack", "forward", "forward_result",
                "backward", "backward_result", "optimizer_step", "step_ack",
                "state_snapshot", "close"}
    missing = sorted(required.difference(names))
    if missing:
        raise ValueError("transcript is incomplete; missing " + ", ".join(missing))
    expected_directions = {
        "hello": "received", "hello_ack": "sent",
        "forward": "received", "forward_result": "sent",
        "backward": "received", "backward_result": "sent",
        "optimizer_step": "received", "step_ack": "sent", "close": "received",
    }
    for event in events:
        expected = expected_directions.get(event["event"])
        if expected is not None and event.get("direction") != expected:
            raise ValueError(f"{event['event']} has wrong direction")
    # Identity pairing, not counting. Counting alone accepts a transcript whose
    # responses carry a different mb_id than their requests: the index builder then
    # rejects it, after the expensive capture has already been taken.
    for request, response in (("forward", "forward_result"),
                              ("backward", "backward_result"),
                              ("optimizer_step", "step_ack")):
        if names.count(request) != names.count(response):
            raise ValueError(f"unpaired {request}/{response} events")
        pending: dict[tuple, int] = {}
        for event in events:
            if event["event"] not in (request, response):
                continue
            key = (event.get("step"), event.get("mb_id"))
            if event["event"] == request:
                if key in pending:
                    raise ValueError(
                        f"duplicate in-flight {request} for step/mb_id {key}")
                pending[key] = event["sequence"]
            else:
                if key not in pending:
                    raise ValueError(
                        f"{response} with no matching {request} for step/mb_id {key}")
                if event["sequence"] < pending[key]:
                    raise ValueError(
                        f"{response} precedes its {request} for step/mb_id {key}")
                del pending[key]
        if pending:
            raise ValueError(
                f"{len(pending)} {request} events never answered: "
                f"{sorted(pending)[:3]}")
    initial = [event for event in events
               if event["event"] == "state_snapshot"
               and event.get("reason") == "initial"]
    if len(initial) != 1:
        raise ValueError("transcript must contain exactly one initial state snapshot")
    steps = [event for event in events if event["event"] == "optimizer_step"]
    snapshots = [event for event in events
                 if event["event"] == "state_snapshot"
                 and event.get("reason") == "after_optimizer_step"]
    # With --transcript-state-interval N the recorder persists a post-step
    # snapshot only when (step + 1) % N == 0; the linkage check must expect
    # exactly those steps, no more, no less.
    expected_state_steps = [int(event["step"]) + 1 for event in steps
                            if not (int(event["step"]) + 1) % interval]
    observed_state_steps = [int(event["step"]) for event in snapshots]
    if observed_state_steps != expected_state_steps:
        raise ValueError("every optimizer update must have one matching "
                         "post-step remote state snapshot")
    return {"session_id": manifest["session_id"], "events": len(events),
            "optimizer_steps": len(steps), "state_interval": interval,
            "state_snapshots": len(snapshots) + 1,
            "files_verified": len(seen)}


def verify_collection(root: Path) -> dict:
    manifest_path = root / "COLLECTION_MANIFEST.json"
    collection = json.loads(manifest_path.read_text())
    if collection.get("schema") != COLLECTION_SCHEMA:
        raise ValueError(f"unsupported collection schema: {collection.get('schema')}")
    sessions = collection.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("collection manifest has no sessions")
    listed = set()
    reports = []
    for entry in sessions:
        relative = Path(entry["manifest_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe collection path: {relative}")
        if relative in listed:
            raise ValueError(f"duplicate collection path: {relative}")
        listed.add(relative)
        manifest = root / relative
        if not manifest.is_file() or digest(manifest) != entry["manifest_sha256"]:
            raise ValueError(f"collection integrity check failed: {relative}")
        session_root = manifest.parent
        try:
            report = verify(session_root)
        except ValueError as error:
            raise ValueError(
                f"session {entry['session_id']} failed verification ({error}); "
                "quarantine it with --collection <root> "
                f"--quarantine {entry['session_id']}") from error
        if report["session_id"] != entry["session_id"]:
            raise ValueError(f"collection session identity mismatch: {relative}")
        reports.append(report)
    discovered = {path / "TRANSCRIPT_MANIFEST.json"
                  for path in root.glob("session_*") if path.is_dir()}
    expected = {root / path for path in listed}
    if discovered != expected:
        unlisted = sorted(path.parent.name for path in discovered - expected)
        absent = sorted(path.parent.name for path in expected - discovered)
        raise ValueError(
            "collection does not enumerate every session directory "
            f"(on disk but unlisted: {unlisted}; listed but absent: {absent}); "
            "quarantine crashed sessions with --collection <root> "
            "--quarantine <session>")
    return {"sessions": len(reports),
            "events": sum(report["events"] for report in reports),
            "optimizer_steps": sum(report["optimizer_steps"] for report in reports),
            "files_verified": sum(report["files_verified"] for report in reports)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript",
                        help="session_<uuid> directory created by the cloud server")
    parser.add_argument("--collection",
                        help="root directory containing COLLECTION_MANIFEST.json")
    parser.add_argument("--quarantine", metavar="SESSION",
                        help="move this crashed or incomplete session (directory "
                             "name session_<id> or bare session id) from "
                             "--collection into <collection>/quarantine/ and "
                             "remove it from the collection manifest, so "
                             "verify_collection can pass")
    parser.add_argument("--quarantine-reason",
                        help="audit reason recorded in quarantine/QUARANTINE_LOG.jsonl "
                             "(requires --quarantine)")
    args = parser.parse_args(argv)
    if args.quarantine:
        if args.transcript or not args.collection:
            parser.error("--quarantine requires --collection "
                         "and excludes --transcript")
        destination = quarantine_session(
            args.collection, args.quarantine,
            reason=args.quarantine_reason or "not specified")
        print(json.dumps({"quarantined": str(destination)}, sort_keys=True))
        return 0
    if args.quarantine_reason:
        parser.error("--quarantine-reason requires --quarantine")
    if bool(args.transcript) == bool(args.collection):
        parser.error("exactly one of --transcript or --collection is required")
    report = (verify(Path(args.transcript)) if args.transcript
              else verify_collection(Path(args.collection)))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
