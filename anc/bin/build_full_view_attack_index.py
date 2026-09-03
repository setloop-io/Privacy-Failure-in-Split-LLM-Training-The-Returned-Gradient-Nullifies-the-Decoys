#!/usr/bin/env python3
"""Build a label-free attack index from a verified complete-view collection."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from attacker.full_view import build_index


def write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True,
                        help="complete-view root with COLLECTION_MANIFEST.json")
    parser.add_argument("--output", required=True,
                        help="label-free JSON index for full-view attacks")
    args = parser.parse_args()
    index = build_index(args.collection)
    write_atomic(Path(args.output), index)
    print(json.dumps({"sessions": len(index["sessions"]),
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
