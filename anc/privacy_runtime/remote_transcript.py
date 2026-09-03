"""Authenticated-at-rest capture of a compromised latent-cloud session.

The recorder runs on the remote node.  It deliberately records only values
that node receives or creates: websocket messages, response messages, and its
own model/optimizer checkpoints.  It never accepts trusted-side labels or
pre-release tensors.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path


SCHEMA = "dtraining.remote_transcript.v1"
COLLECTION_SCHEMA = "dtraining.remote_transcript_collection.v1"
QUARANTINE_SCHEMA = "dtraining.remote_transcript_quarantine.v1"

_QUARANTINE_DIRNAME = "quarantine"
_QUARANTINE_LOG = "QUARANTINE_LOG.jsonl"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class RemoteTranscript:
    """Append-only capture for one cloud websocket session."""

    def __init__(self, root: str | Path, session_id: str,
                 state_interval: int = 1):
        if state_interval <= 0:
            raise ValueError("state_interval must be positive")
        self.collection_root = Path(root)
        self.root = self.collection_root / f"session_{session_id}"
        self.root.mkdir(parents=True, exist_ok=False)
        self.session_id = session_id
        self.state_interval = state_interval
        self.events_path = self.root / "events.jsonl"
        self._entries: list[dict] = []
        self._sequence = 0
        self._closed = False
        self._record_bytes(
            "session.json",
            _json_bytes({"schema": SCHEMA, "session_id": session_id,
                         "started_at_utc": _now_utc(),
                         "state_interval": state_interval}),
            kind="session")

    def _record_bytes(self, relative: str, data: bytes, *, kind: str,
                      metadata: dict | None = None) -> dict:
        path = self.root / relative
        _atomic_write(path, data)
        entry = {"path": relative, "kind": kind, "bytes": len(data),
                 "sha256": _sha256(data)}
        if metadata:
            entry.update(metadata)
        self._entries.append(entry)
        return entry

    def _event(self, event: dict) -> None:
        event = {"sequence": self._sequence,
                 "recorded_at_utc": _now_utc(),
                 "recorded_monotonic_ns": time.monotonic_ns(), **event}
        self._sequence += 1
        line = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def record_text(self, direction: str, message: str, *, event: str,
                    step: int | None = None, mb_id: int | None = None,
                    parsed: dict | None = None) -> None:
        relative = f"messages/{self._sequence:08d}.json"
        entry = self._record_bytes(relative, message.encode(), kind="message",
                                   metadata={"direction": direction})
        self._event({"event": event, "direction": direction,
                     "transport": "text", "step": step, "mb_id": mb_id,
                     "parsed": parsed, "file": entry})

    def record_wire(self, direction: str, message: bytes, *, event: str,
                    header: dict, step: int, mb_id: int) -> None:
        relative = f"messages/{self._sequence:08d}.bin"
        entry = self._record_bytes(relative, message, kind="message",
                                   metadata={"direction": direction})
        self._event({"event": event, "direction": direction,
                     "transport": "binary", "step": step, "mb_id": mb_id,
                     "header": header, "file": entry})

    def record_state(self, state: dict, *, step: int, reason: str) -> None:
        """Persist a remote-only torch checkpoint and include it in the ledger."""
        if reason != "initial" and step % self.state_interval:
            return
        import torch

        state_dir = self.root / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=state_dir, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            torch.save(state, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            self.record_state_bytes(temporary.read_bytes(), step=step,
                                    reason=reason)
        finally:
            if temporary.exists():
                temporary.unlink()

    def record_state_bytes(self, data: bytes, *, step: int, reason: str) -> None:
        """Record a serialized remote-state checkpoint.

        `record_state` supplies torch serialization in production. Keeping the
        byte-level primitive separate makes integrity checks testable on hosts
        that intentionally do not install torch.
        """
        relative = f"state/{step:08d}_{reason}.pt"
        entry = self._record_bytes(relative, data, kind="state",
                                   metadata={"step": step, "reason": reason})
        self._event({"event": "state_snapshot", "step": step,
                     "reason": reason, "file": entry})

    def record_error(self, error: str) -> None:
        self._event({"event": "server_error", "error": error})

    def finalize(self, status: str) -> Path:
        if self._closed:
            return self.root / "TRANSCRIPT_MANIFEST.json"
        if self.events_path.exists():
            data = self.events_path.read_bytes()
            self._entries.append({"path": "events.jsonl", "kind": "events",
                                  "bytes": len(data), "sha256": _sha256(data)})
        manifest = {"schema": SCHEMA, "session_id": self.session_id,
                    "status": status, "closed_at_utc": _now_utc(),
                    "event_count": self._sequence,
                    "state_interval": self.state_interval,
                    "files": self._entries}
        path = self.root / "TRANSCRIPT_MANIFEST.json"
        _atomic_write(path, _json_bytes(manifest))
        self._update_collection(path, manifest)
        self._closed = True
        return path

    def _update_collection(self, session_manifest_path: Path,
                           session_manifest: dict) -> None:
        """Atomically add this completed or failed session to the root ledger."""
        import fcntl

        collection_path = self.collection_root / "COLLECTION_MANIFEST.json"
        lock_path = self.collection_root / ".collection_manifest.lock"
        self.collection_root.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if collection_path.exists():
                    collection = json.loads(collection_path.read_text())
                    if collection.get("schema") != COLLECTION_SCHEMA:
                        raise ValueError("existing collection manifest has an "
                                         "unsupported schema")
                else:
                    collection = {"schema": COLLECTION_SCHEMA, "sessions": []}
                relative = session_manifest_path.relative_to(self.collection_root)
                entry = {
                    "session_id": self.session_id,
                    "manifest_path": str(relative),
                    "manifest_sha256": _sha256(session_manifest_path.read_bytes()),
                    "status": session_manifest["status"],
                    "event_count": session_manifest["event_count"],
                }
                sessions = {item["session_id"]: item
                            for item in collection.get("sessions", [])}
                sessions[self.session_id] = entry
                collection = {
                    "schema": COLLECTION_SCHEMA,
                    "updated_at_utc": _now_utc(),
                    "sessions": [sessions[session_id]
                                 for session_id in sorted(sessions)],
                }
                _atomic_write(collection_path, _json_bytes(collection))
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def record_complete_session(root: str | Path, session_id: str, *,
                            steps: int = 1,
                            mb_ids: tuple[int, ...] = (0,),
                            state_interval: int = 1) -> Path:
    """Record a well-formed complete session, as latent_cloud_server.py does.

    Fixture support for the transcript verifier (experiment W3.1): drives the
    real recorder through the exact event sequence the cloud server emits
    for a cleanly closed session, so verify() and verify_collection() can be
    exercised without torch or a GPU.  Post-step state snapshots are gated
    by `state_interval` exactly as RemoteTranscript.record_state gates them.
    """
    transcript = RemoteTranscript(root, session_id,
                                  state_interval=state_interval)
    hello = json.dumps({"op": "hello", "protocol": "latent-native-v5"})
    transcript.record_text("received", hello, event="hello",
                           parsed={"op": "hello"})
    transcript.record_text("sent", json.dumps({"op": "hello_ack"}),
                           event="hello_ack", parsed={"op": "hello_ack"})
    transcript.record_state_bytes(b"state-initial", step=0, reason="initial")
    for step in range(steps):
        for mb_id in mb_ids:
            _record_step_frames(transcript, step, mb_id)
        transcript.record_text("received", json.dumps({"op": "optimizer_step"}),
                               event="optimizer_step", step=step,
                               parsed={"op": "optimizer_step"})
        # step_ack echoes the request's step: the verifier pairs events by
        # (step, mb_id) identity, like every other request/response pair.
        transcript.record_text("sent", json.dumps({"op": "step_ack"}),
                               event="step_ack", step=step,
                               parsed={"op": "step_ack"})
        if not (step + 1) % state_interval:
            transcript.record_state_bytes(f"state-{step + 1}".encode(),
                                           step=step + 1,
                                           reason="after_optimizer_step")
    transcript.record_text("received", json.dumps({"op": "close"}),
                           event="close", step=steps, parsed={"op": "close"})
    return transcript.finalize("complete")


def _record_step_frames(transcript: RemoteTranscript, step: int,
                        mb_id: int) -> None:
    """One forward/backward exchange, as the cloud server records it."""
    header = {"op": "forward", "mb_id": mb_id, "shape": [1, 1, 1]}
    transcript.record_wire("received", b"forward-frame", event="forward",
                           header=header, step=step, mb_id=mb_id)
    result = {"op": "forward_result", "mb_id": mb_id, "shape": [1, 1, 1]}
    transcript.record_wire("sent", b"forward-result", event="forward_result",
                           header=result, step=step, mb_id=mb_id)
    backward = {"op": "backward", "mb_id": mb_id, "shape": [1, 1, 1]}
    transcript.record_wire("received", b"backward-frame", event="backward",
                           header=backward, step=step, mb_id=mb_id)
    gradient = {"op": "backward_result", "mb_id": mb_id, "shape": [1, 1, 1]}
    transcript.record_wire("sent", b"backward-result",
                           event="backward_result", header=gradient,
                           step=step, mb_id=mb_id)


def quarantine_session(collection_root: str | Path, session: str, *,
                       reason: str) -> Path:
    """Move one crashed or incomplete session out of the collection ledger.

    A crashed session stays on disk and stays listed in the collection
    manifest, so verify() rejects it and verify_collection() rejects both
    keeping it and deleting it.  Quarantining moves the session directory to
    <root>/quarantine/, removes its entry from COLLECTION_MANIFEST.json
    under the collection lock, and appends an audit record to
    <root>/quarantine/QUARANTINE_LOG.jsonl.  `session` is the session
    directory name (session_<id>) or the bare session id.
    """
    import fcntl

    root = Path(collection_root)
    name = session if session.startswith("session_") else f"session_{session}"
    source = root / name
    destination = root / _QUARANTINE_DIRNAME / name
    if destination.is_dir() and not source.exists():
        return destination  # already quarantined; a re-run is a no-op
    if destination.exists():
        raise ValueError(f"quarantine destination already exists: {destination}")
    if not source.is_dir():
        raise ValueError(f"no such session directory: {source}")

    with (root / ".collection_manifest.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            # Delist before moving: a crash in between leaves an unlisted
            # directory on disk, which a re-run of this call cleans up.
            removed = _delist_session(root, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            _append_quarantine_log(root, name=name, reason=reason,
                                   removed=removed)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return destination


def _delist_session(root: Path, name: str) -> list[dict]:
    """Drop every collection entry that points at session directory `name`."""
    collection_path = root / "COLLECTION_MANIFEST.json"
    if not collection_path.exists():
        return []
    collection = json.loads(collection_path.read_text())
    if collection.get("schema") != COLLECTION_SCHEMA:
        raise ValueError("existing collection manifest has an "
                         "unsupported schema")
    session_id = name[len("session_"):]
    removed: list[dict] = []
    kept: list[dict] = []
    for entry in collection.get("sessions", []):
        target = removed if _entry_matches(entry, name, session_id) else kept
        target.append(entry)
    if not removed:
        return []
    _atomic_write(collection_path, _json_bytes(
        {"schema": COLLECTION_SCHEMA, "updated_at_utc": _now_utc(),
         "sessions": kept}))
    return removed


def _entry_matches(entry: dict, name: str, session_id: str) -> bool:
    return (entry.get("session_id") == session_id
            or Path(str(entry.get("manifest_path", ""))).parent.name == name)


def _append_quarantine_log(root: Path, *, name: str, reason: str,
                           removed: list[dict]) -> None:
    """Append the audit record for one quarantined session."""
    record = {"schema": QUARANTINE_SCHEMA,
              "quarantined_at_utc": _now_utc(),
              "session": name, "session_id": name[len("session_"):],
              "reason": reason, "removed_entries": removed}
    path = root / _QUARANTINE_DIRNAME / _QUARANTINE_LOG
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_verifier():
    """Load bin/verify_remote_transcript.py without making bin/ a package."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "bin" / "verify_remote_transcript.py"
    spec = importlib.util.spec_from_file_location("verify_remote_transcript",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _self_test_recorder(directory: str) -> bool:
    """Recorder smoke check: raw files, manifest paths, collection entry."""
    transcript = RemoteTranscript(directory, "test", state_interval=1)
    transcript.record_text("received", '{"op":"hello"}', event="hello",
                           parsed={"op": "hello"})
    transcript.record_wire("received", b"frame", event="forward",
                           header={"op": "forward", "mb_id": 0},
                           step=0, mb_id=0)
    transcript.record_state_bytes(b"initial", step=0, reason="initial")
    manifest = json.loads(transcript.finalize("complete").read_text())
    paths = {entry["path"] for entry in manifest["files"]}
    collection = json.loads(
        (Path(directory) / "COLLECTION_MANIFEST.json").read_text())
    ok = (manifest["schema"] == SCHEMA and manifest["event_count"] == 3
          and manifest["state_interval"] == 1
          and collection["schema"] == COLLECTION_SCHEMA
          and collection["sessions"][0]["session_id"] == "test"
          and {"session.json", "events.jsonl", "messages/00000000.json",
               "messages/00000001.bin", "state/00000000_initial.pt"}.issubset(paths))
    print(f"  {'ok  ' if ok else 'FAIL'} recorder smoke test")
    return ok


def _self_test_verify_single(verifier, directory: str) -> bool:
    """W3.1 single-session fixture: a real recorded session passes verify()."""
    manifest_path = record_complete_session(directory, "verify_single",
                                            steps=2, mb_ids=(0, 1))
    try:
        report = verifier.verify(manifest_path.parent)
    except Exception as error:  # noqa: BLE001 - the reason is the result
        print(f"  FAIL single-session verify fixture "
              f"({type(error).__name__}: {str(error)[:70]})")
        return False
    ok = (report["session_id"] == "verify_single"
          and report["optimizer_steps"] == 2
          and report["state_snapshots"] == 3)
    print(f"  {'ok  ' if ok else 'FAIL'} single-session verify fixture: "
          f"{report['events']} events, {report['files_verified']} files")
    return ok


def _self_test_verify_collection(verifier, directory: str) -> bool:
    """W3.1 multi-session fixture: two sessions pass verify_collection()."""
    record_complete_session(directory, "multi_a", steps=1)
    record_complete_session(directory, "multi_b", steps=2,
                            state_interval=2)
    try:
        report = verifier.verify_collection(Path(directory))
    except Exception as error:  # noqa: BLE001 - the reason is the result
        print(f"  FAIL multi-session verify fixture "
              f"({type(error).__name__}: {str(error)[:70]})")
        return False
    ok = (report["sessions"] == 2 and report["optimizer_steps"] == 3)
    print(f"  {'ok  ' if ok else 'FAIL'} multi-session verify fixture: "
          f"{report['sessions']} sessions, {report['events']} events")
    return ok


def self_test() -> int:
    import tempfile

    verifier = _load_verifier()
    with tempfile.TemporaryDirectory() as recorder_dir:
        checks = [_self_test_recorder(recorder_dir)]
    with tempfile.TemporaryDirectory() as single_dir:
        checks.append(_self_test_verify_single(verifier, single_dir))
    with tempfile.TemporaryDirectory() as collection_dir:
        checks.append(_self_test_verify_collection(verifier, collection_dir))
    ok = all(checks)
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(self_test() if "--self-test" in sys.argv else 2)
