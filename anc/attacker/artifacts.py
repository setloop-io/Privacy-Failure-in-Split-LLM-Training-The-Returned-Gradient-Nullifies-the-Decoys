#!/usr/bin/env python3
"""Artifact + journal conventions for the attacker framework.

Every attack writes ONE JSON artifact with the repo's standard top-level
fields (cf. rotation_lifetime.py / e8_robustness.py):

    {"schema", "config", "threat_model", "provenance", "results": [],
     "summary": [], ...}

plus a crash-safe per-cell JSONL journal at <output>.jsonl appended after
EACH completed cell (pattern: rotation_lifetime._append_jsonl) so a late
hard crash cannot take completed cells down with the process. Failed cells
are journaled with an "error" field instead of raising (rotation_lifetime
solve hardening).
"""

import json
import os
import socket
import subprocess
import time


def light_provenance():
    """Hostname + git commit + container tag, in the spirit of
    trained_inversion.make_provenance but dependency-free and never
    raising (attacks run on heterogeneous hosts)."""
    prov = {"hostname": socket.gethostname(),
            "container_image": os.environ.get("CONTAINER_IMAGE", "unknown"),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime())}
    try:
        prov["dtraining_commit"] = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
             "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5).stdout.strip() \
            or "unknown"
    except Exception:
        prov["dtraining_commit"] = "unknown"
    return prov


def make_artifact(schema, config, threat_model, interpretation=None):
    """The standard artifact skeleton."""
    out = {"schema": schema,
           "config": config,
           "threat_model": threat_model,
           "provenance": light_provenance(),
           "results": [],
           "summary": []}
    if interpretation is not None:
        out["interpretation"] = interpretation
    return out


def append_jsonl(output, record):
    """Crash-safe per-cell journal: <output>.jsonl appended immediately
    after each completed (or failed-with-error) cell."""
    if not output:
        return
    with open(output + ".jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


def write_artifact(output, out):
    """Final artifact write (pretty JSON, atomic-ish via .tmp rename)."""
    if not output:
        return
    tmp = output + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, output)
    print(f"[artifact] wrote {output} (+ {output}.jsonl journal)")


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import tempfile
    print("artifact/journal conventions (pure python):")
    out = make_artifact("dtraining.attacker.selftest.v1", {"x": 1},
                        "self-test threat model")
    check("artifact carries the standard top-level fields",
          all(k in out for k in
              ("schema", "config", "threat_model", "provenance",
               "results", "summary")))
    check("provenance has hostname/commit/timestamp",
          all(k in out["provenance"] for k in
              ("hostname", "dtraining_commit", "timestamp_utc")))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "art.json")
        append_jsonl(p, {"cell": 1, "top1": 3.2})
        append_jsonl(p, {"cell": 2, "error": "error: boom", "top1": None})
        lines = open(p + ".jsonl").read().strip().splitlines()
        check("journal appends one JSON record per line", len(lines) == 2)
        check("error cells are journaled, not raised",
              json.loads(lines[1])["error"] == "error: boom")
        write_artifact(p, out)
        check("artifact file written and parses",
              json.load(open(p))["schema"] == "dtraining.attacker.selftest.v1")
        check("no .tmp left behind",
              not os.path.exists(p + ".tmp"))

    print("SELF-TEST " + ("PASSED" if ok else 'FAILED'))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
