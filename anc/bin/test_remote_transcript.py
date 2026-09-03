#!/usr/bin/env python3
"""Fixtures for the transcript verifier (experiment W3.1).

The verifier is the gate deciding whether an expensive capture is usable; these
fixtures exercise it on single- and multi-session transcripts, which
`remote_transcript.self_test()` does not (it writes a three-event session and
never calls `verify()`).

The hand-built set below carries the load-bearing negative case: a transcript whose
responses carry a different `mb_id` than their requests. Comparing event *counts*
only accepts such a transcript as complete; the index builder then rejects it,
after the capture has already been paid for.

Three further fixture groups:
- real-recorder fixtures: sessions produced by the actual RemoteTranscript class
  verify, single- and multi-session. Building these exposed that the cloud server
  recorded step_ack with the post-increment step, so no real transcript could pair
  optimizer_step/step_ack by (step, mb_id) identity;
- state-interval fixtures: with --transcript-state-interval N the recorder persists
  post-step snapshots only every N steps; those transcripts verify when the manifest
  declares N, and a snapshot thinned without the declaration is rejected;
- quarantine fixtures: a crashed session is moved to <root>/quarantine/ and delisted,
  so verify_collection passes on the survivors instead of failing on a session that
  can be neither kept nor deleted.

Pure Python; no network, no GPU. torch is optional -- only the positive branch of the
recorder's state gate additionally runs where torch is installed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from privacy_runtime.remote_transcript import (  # noqa: E402
    RemoteTranscript, SCHEMA, quarantine_session, record_complete_session)

sys.path.insert(0, str(ROOT / "bin"))
sys.path.insert(0, str(ROOT / "split-training"))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "verify_remote_transcript", ROOT / "bin" / "verify_remote_transcript.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)

from latent_cloud_server import LatentServer, build_parser  # noqa: E402


def _write_session(root: Path, events: list[dict], status: str = "complete",
                   *, manifest_interval: int = 1,
                   session_interval: int = 1) -> Path:
    """Materialise a session directory whose manifest digests are self-consistent.

    `manifest_interval` and `session_interval` are separate so a fixture can
    prove the verifier rejects a manifest that disagrees with session.json.
    """
    root.mkdir(parents=True, exist_ok=True)
    payload = root / "payload.bin"
    payload.write_bytes(b"opaque wire bytes")
    (root / "session.json").write_text(json.dumps(
        {"schema": SCHEMA, "session_id": "fixture",
         "state_interval": session_interval}, sort_keys=True))

    for index, event in enumerate(events):
        event.setdefault("sequence", index)

    (root / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events))

    files = []
    for name in ("payload.bin", "session.json", "events.jsonl"):
        blob = (root / name).read_bytes()
        files.append({"path": name, "bytes": len(blob),
                      "sha256": hashlib.sha256(blob).hexdigest()})

    (root / "TRANSCRIPT_MANIFEST.json").write_text(json.dumps({
        "schema": SCHEMA, "status": status, "session_id": "fixture",
        "event_count": len(events), "state_interval": manifest_interval,
        "files": files}, indent=1))
    return root


def _events(pairs: list[tuple[int, int, int, int]]) -> list[dict]:
    """pairs: (request_step, request_mb, response_step, response_mb)."""
    out = [{"event": "hello", "direction": "received"},
           {"event": "hello_ack", "direction": "sent"},
           {"event": "state_snapshot", "step": 0, "reason": "initial"}]
    for rq_step, rq_mb, rs_step, rs_mb in pairs:
        out += [
            {"event": "forward", "direction": "received", "step": rq_step, "mb_id": rq_mb},
            {"event": "forward_result", "direction": "sent", "step": rs_step, "mb_id": rs_mb},
            {"event": "backward", "direction": "received", "step": rq_step, "mb_id": rq_mb},
            {"event": "backward_result", "direction": "sent", "step": rs_step, "mb_id": rs_mb},
        ]
    out += [{"event": "optimizer_step", "direction": "received", "step": 0, "mb_id": 0},
            {"event": "step_ack", "direction": "sent", "step": 0, "mb_id": 0},
            {"event": "state_snapshot", "step": 1, "reason": "after_optimizer_step"},
            {"event": "close", "direction": "received"}]
    return out


def _outcome(label: str, thunk, should_pass: bool) -> bool:
    try:
        thunk()
        passed = should_pass
        detail = "accepted"
    except Exception as error:  # noqa: BLE001 - the rejection reason is the result
        passed = not should_pass
        detail = f"rejected ({type(error).__name__}: {str(error)[:60]})"
    print(f"  {'ok  ' if passed else 'FAIL'} {label}: {detail}")
    return passed


def _expect(label: str, root: Path, should_pass: bool) -> bool:
    return _outcome(label, lambda: verifier.verify(root), should_pass)


def _check(label: str, condition: bool, detail: str = "") -> bool:
    suffix = f": {detail}" if detail else ""
    print(f"  {'ok  ' if condition else 'FAIL'} {label}{suffix}")
    return condition


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _argparse_rejects_zero() -> bool:
    """argparse exits with SystemExit, which _outcome's Exception net misses."""
    import contextlib
    import io

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            build_parser().parse_args(["--transcript-state-interval", "0"])
        rejected, detail = False, "accepted"
    except SystemExit:
        rejected, detail = True, "rejected by argparse"
    return _check("non-positive intervals are rejected at the CLI",
                  rejected, detail)


def _hand_built_pairing_fixtures(work: Path) -> list[bool]:
    print("== hand-built pairing fixtures ==")
    return [
        _expect(
            "well-formed single session is accepted",
            _write_session(work / "good", _events([(0, 0, 0, 0), (0, 1, 0, 1)])),
            should_pass=True),
        _expect(
            "MISMATCHED mb_id is rejected (the defect W3.1 exists for)",
            _write_session(work / "mb_mismatch", _events([(0, 0, 0, 1)])),
            should_pass=False),
        _expect(
            "mismatched step is rejected",
            _write_session(work / "step_mismatch", _events([(0, 0, 7, 0)])),
            should_pass=False),
        _expect(
            "duplicate in-flight request is rejected",
            _write_session(work / "duplicate", _duplicate_events()),
            should_pass=False),
        _expect(
            "unanswered request is rejected",
            _write_session(work / "unanswered", _unanswered_events()),
            should_pass=False),
        _expect(
            "response preceding its request is rejected",
            _write_session(work / "reordered", _reordered_events()),
            should_pass=False),
        _expect(
            "incomplete session is rejected",
            _write_session(work / "incomplete", _events([(0, 0, 0, 0)]),
                           status="failed"), should_pass=False),
        _expect(
            "tampered payload is rejected",
            _tampered_session(work / "tampered"), should_pass=False),
    ]


def _duplicate_events() -> list[dict]:
    duplicate = _events([(0, 0, 0, 0)])
    duplicate.insert(4, {"event": "forward", "direction": "received",
                         "step": 0, "mb_id": 0})
    return duplicate


def _unanswered_events() -> list[dict]:
    unanswered = [event for event in _events([(0, 0, 0, 0)])
                  if event["event"] != "forward_result"]
    unanswered.append({"event": "forward_result", "direction": "sent",
                       "step": 0, "mb_id": 9})
    return unanswered


def _reordered_events() -> list[dict]:
    reordered = _events([(0, 0, 0, 0)])
    reordered[3], reordered[4] = reordered[4], reordered[3]
    return reordered


def _tampered_session(root: Path) -> Path:
    session = _write_session(root, _events([(0, 0, 0, 0)]))
    (root / "payload.bin").write_bytes(b"different bytes entirely")
    return session


def _real_recorder_fixtures(work: Path) -> list[bool]:
    """Requirement 1: sessions from the real recorder verify, single and multi."""
    print("== real-recorder fixtures ==")
    single = work / "real_single"
    manifest = record_complete_session(single, "real_single", steps=2,
                                       mb_ids=(0, 1))
    checks = [_expect("real recorder session passes verify()",
                      manifest.parent, should_pass=True)]
    report = verifier.verify(manifest.parent)
    checks.append(_check(
        "verify() reports the recorded coverage",
        report["session_id"] == "real_single"
        and report["optimizer_steps"] == 2
        and report["state_snapshots"] == 3 and report["state_interval"] == 1,
        f"{report['events']} events, {report['files_verified']} files"))

    multi = work / "real_multi"
    record_complete_session(multi, "multi_a", steps=1)
    record_complete_session(multi, "multi_b", steps=2)
    try:
        report = verifier.verify_collection(multi)
        passed = (report["sessions"] == 2
                  and report["optimizer_steps"] == 3)
        detail = (f"{report['sessions']} sessions, "
                  f"{report['optimizer_steps']} optimizer steps")
    except Exception as error:  # noqa: BLE001 - the rejection reason is the result
        passed = False
        detail = f"{type(error).__name__}: {str(error)[:60]}"
    print(f"  {'ok  ' if passed else 'FAIL'} "
          f"real recorder collection passes verify_collection(): {detail}")
    checks.append(passed)
    return checks


def _state_interval_fixtures(work: Path) -> list[bool]:
    """Requirement 2a: thinned snapshots verify when the interval is declared."""
    print("== state-interval fixtures ==")
    thinned = work / "thinned"
    manifest = record_complete_session(thinned, "thinned", steps=5,
                                       state_interval=3)
    state_files = sorted(path.name
                         for path in (manifest.parent / "state").iterdir())
    checks = [
        _check("interval=3 over 5 steps persists only the initial state "
               "and step 3",
               state_files == ["00000000_initial.pt",
                               "00000003_after_optimizer_step.pt"],
               ", ".join(state_files)),
        _expect("thinned session passes verify() on the declared interval",
                manifest.parent, should_pass=True),
    ]
    report = verifier.verify(manifest.parent)
    checks.append(_check(
        "verify() reports the declared interval",
        report["state_interval"] == 3 and report["state_snapshots"] == 2))

    undeclared = _events([(0, 0, 0, 0)])
    undeclared[-1:-1] = [
        {"event": "optimizer_step", "direction": "received",
         "step": 1, "mb_id": 0},
        {"event": "step_ack", "direction": "sent", "step": 1, "mb_id": 0},
    ]
    checks.append(_expect(
        "a post-step snapshot thinned without the declaration is rejected",
        _write_session(work / "undeclared", undeclared), should_pass=False))

    disagree = [event for event in _events([(0, 0, 0, 0)])
                if not (event["event"] == "state_snapshot"
                        and event.get("reason") == "after_optimizer_step")]
    checks.append(_expect(
        "session.json disagreeing with the manifest interval is rejected",
        _write_session(work / "disagree", disagree, manifest_interval=2,
                       session_interval=1), should_pass=False))
    return checks


def _server_flag_fixtures(work: Path) -> list[bool]:
    """Requirement 2b: --transcript-state-interval exists and reaches the gate."""
    print("== server flag fixtures ==")
    checks = [
        _check("--transcript-state-interval exists and defaults to 1",
               build_parser().parse_args([]).transcript_state_interval == 1),
        _check("--transcript-state-interval parses its value",
               build_parser().parse_args(
                   ["--transcript-state-interval", "4"]
               ).transcript_state_interval == 4),
    ]
    checks.append(_argparse_rejects_zero())

    server = LatentServer(build_parser().parse_args(
        ["--transcript-dir", str(work / "flagdir"),
         "--transcript-state-interval", "4"]))
    transcript = server.transcript("flagprobe")
    checks.append(_check(
        "the flag reaches the recorder",
        transcript.state_interval == 4
        and (work / "flagdir" / "session_flagprobe").is_dir()))

    gated_root = work / "flagdir" / "session_gated"
    gated = RemoteTranscript(work / "flagdir", "gated", state_interval=4)
    gated.record_state({"probe": 0}, step=1, reason="after_optimizer_step")
    checks.append(_check(
        "gated-out state steps write nothing (before torch even loads)",
        not (gated_root / "state").exists()))
    if _torch_available():
        gated.record_state({"probe": 1}, step=4,
                           reason="after_optimizer_step")
        written = sorted(path.name for path in (gated_root / "state").iterdir())
        checks.append(_check("in-interval state steps are persisted",
                             written == ["00000004_after_optimizer_step.pt"]))
    else:
        print("  --   torch absent; the positive state-gate branch runs "
              "only where torch is installed")
    return checks


def _record_crashed_session(root: Path, session_id: str) -> Path:
    """The server's crash path: some events, then finalize('incomplete')."""
    transcript = RemoteTranscript(root, session_id)
    transcript.record_text("received", '{"op":"hello"}', event="hello",
                           parsed={"op": "hello"})
    return transcript.finalize("incomplete")


def _quarantine_fixtures(work: Path) -> list[bool]:
    """Requirement 3: a crashed session is quarantinable, not a dead end."""
    print("== quarantine fixtures ==")
    crashed_root = work / "crashed_collection"
    record_complete_session(crashed_root, "good", steps=1)
    _record_crashed_session(crashed_root, "crashed")
    checks = [_outcome(
        "collection containing a crashed session is rejected",
        lambda: verifier.verify_collection(crashed_root), should_pass=False)]

    destination = quarantine_session(crashed_root, "crashed",
                                     reason="fixture: simulated crash")
    collection = json.loads(
        (crashed_root / "COLLECTION_MANIFEST.json").read_text())
    log = [json.loads(line) for line
           in (crashed_root / "quarantine" / "QUARANTINE_LOG.jsonl")
           .read_text().splitlines()]
    checks += [
        _check("quarantine moves the session directory, manifest intact",
               destination == crashed_root / "quarantine" / "session_crashed"
               and (destination / "TRANSCRIPT_MANIFEST.json").is_file()),
        _check("quarantine delists the session",
               [entry["session_id"] for entry in collection["sessions"]]
               == ["good"]),
        _check("quarantine appends an audit record",
               len(log) == 1 and log[0]["session_id"] == "crashed"
               and log[0]["reason"] == "fixture: simulated crash"),
        _outcome("verify_collection passes on the survivors",
                 lambda: verifier.verify_collection(crashed_root),
                 should_pass=True),
        _check("re-quarantining is a no-op",
               quarantine_session(crashed_root, "crashed",
                                  reason="re-run") == destination),
    ]

    orphan_root = work / "orphan_collection"
    record_complete_session(orphan_root, "good", steps=1)
    RemoteTranscript(orphan_root, "orphan").record_text(
        "received", '{"op":"hello"}', event="hello", parsed={"op": "hello"})
    checks.append(_outcome(
        "collection with an unlisted crashed directory is rejected",
        lambda: verifier.verify_collection(orphan_root), should_pass=False))
    quarantine_session(orphan_root, "session_orphan",
                       reason="fixture: crashed before finalize")
    checks.append(_outcome(
        "verify_collection passes after quarantining the orphan",
        lambda: verifier.verify_collection(orphan_root), should_pass=True))

    checks.append(_outcome(
        "quarantine refuses an unknown session",
        lambda: quarantine_session(crashed_root, "nosuch",
                                   reason="fixture"), should_pass=False))
    return checks


def _cli(argv: list[str]) -> tuple[bool, str]:
    """Run the verifier CLI in-process; return (ok, output or failure reason)."""
    import contextlib
    import io

    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), \
                contextlib.redirect_stderr(io.StringIO()):
            verifier.main(argv)
        return True, output.getvalue().strip()
    except SystemExit as error:
        return False, f"usage error (exit {error.code})"
    except Exception as error:  # noqa: BLE001 - the rejection is the result
        return False, f"{type(error).__name__}: {str(error)[:60]}"


def _cli_fixtures(work: Path) -> list[bool]:
    print("== CLI fixtures ==")
    root = work / "cli_collection"
    record_complete_session(root, "good", steps=1)
    _record_crashed_session(root, "crashed")
    before, _ = _cli(["--collection", str(root)])
    code, _ = _cli(["--collection", str(root), "--quarantine", "crashed",
                    "--quarantine-reason", "fixture: CLI"])
    after, _ = _cli(["--collection", str(root)])
    return [
        _check("--collection fails before the quarantine", not before),
        _check("--quarantine exits 0 and moves the session", code),
        _check("the collection verifies after the CLI quarantine", after),
        _check("--quarantine without --collection is a usage error",
               not _cli(["--quarantine", "crashed"])[0]),
    ]


def self_test() -> int:
    work = Path(tempfile.mkdtemp(prefix="transcript-fixtures-"))
    try:
        checks = (_hand_built_pairing_fixtures(work)
                  + _real_recorder_fixtures(work)
                  + _state_interval_fixtures(work)
                  + _server_flag_fixtures(work)
                  + _quarantine_fixtures(work)
                  + _cli_fixtures(work))
        ok = all(checks)
        print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
        return 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(self_test())
