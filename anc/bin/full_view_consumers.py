#!/usr/bin/env python3
"""CLI over attacker/full_view.py's consumers: the three reads of a complete
remote transcript (final-window, per-step, accumulated) plus the structure-only
coverage report. The attack-framework entry point is
`python -m attacker --attack full-history`; this script is the direct driver.

Payload tensors are not part of the index; they live in the raw transcript
collection. The w34 raw transcript payloads are not included in this release
(only the verified collection and index under
paper-data/collected/diagnostic/w34_complete/), so payload mode is exercised
on synthetic fixtures only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacker.full_view import (INDEX_SCHEMA, PAYLOAD_KINDS, VIEWS,
                                check_index_coverage, view_final_window)


def load_index(path: Path) -> dict:
    index = json.loads(Path(path).read_text())
    if index.get("schema") != INDEX_SCHEMA:
        raise ValueError(f"{path} is not a {INDEX_SCHEMA} index")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--coverage", action="store_true",
                        help="structure-only coverage report (no payloads)")
    parser.add_argument("--view", choices=sorted(VIEWS),
                        help="emit the payload-reference view")
    parser.add_argument("--last-steps", type=int, default=512)
    parser.add_argument("--output")
    args = parser.parse_args()

    index = load_index(Path(args.index))
    if args.coverage:
        report = check_index_coverage(index)
    elif args.view:
        view = (view_final_window(index, args.last_steps)
                if args.view == "final-window" else VIEWS[args.view](index))
        missing = []
        for session in view["sessions"]:
            for frame in session["frames"]:
                for kind in PAYLOAD_KINDS:
                    path = Path(args.index).parent / frame[kind]["path"]
                    if not path.is_file():
                        missing.append(str(path))
        report = {"view": view["view"], "payloads_missing": len(missing),
                  "payload_mode": "blocked" if missing else "ready",
                  "note": "payload tensors live in the raw transcript "
                          "collection; absent payloads mean structure-only "
                          "(the w34 raw transcript is on poseidon)"}
        if not missing:
            report["view_ref"] = view
    else:
        parser.error("one of --coverage / --view is required")
    data = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(data)
    print(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
