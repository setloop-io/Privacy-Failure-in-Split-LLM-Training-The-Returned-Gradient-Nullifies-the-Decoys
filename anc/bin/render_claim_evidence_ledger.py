#!/usr/bin/env python3
"""Validate and render the Paper 1 claim/evidence ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"id", "statement", "adversary_view", "configuration", "seeds",
            "artifacts", "statistical_unit", "positive_control", "status",
            "provenance", "limitations"}
PROVENANCE = {"code_commit", "dirty_state", "tokenizer", "corpus",
              "corpus_split"}


def load_and_validate(path: Path) -> dict:
    ledger = json.loads(path.read_text())
    if ledger.get("schema") != "dtraining.claim_evidence_ledger.v1":
        raise ValueError("unsupported claim ledger schema")
    allowed = set(ledger.get("policy", {}).get("allowed_statuses", []))
    if not allowed:
        raise ValueError("ledger declares no allowed statuses")
    ids = set()
    for claim in ledger.get("claims", []):
        missing = REQUIRED.difference(claim)
        if missing:
            raise ValueError(f"claim {claim.get('id')}: missing {sorted(missing)}")
        if claim["id"] in ids:
            raise ValueError(f"duplicate claim id: {claim['id']}")
        ids.add(claim["id"])
        if claim["status"] not in allowed:
            raise ValueError(f"claim {claim['id']}: invalid status")
        if set(claim["provenance"]) != PROVENANCE:
            raise ValueError(f"claim {claim['id']}: invalid provenance fields")
        if not isinstance(claim["artifacts"], list) or not claim["artifacts"]:
            raise ValueError(f"claim {claim['id']}: artifacts must be nonempty")
        for artifact in claim["artifacts"]:
            if not (ROOT / artifact).is_file():
                raise ValueError(f"claim {claim['id']}: missing artifact {artifact}")
    return ledger


def render(ledger: dict) -> str:
    lines = ["# Claim Evidence Ledger", "", ledger["policy"]["rule"], "",
             "| Claim | Adversary view | Configuration / seeds | Artifact | Unit / control | Status |",
             "| --- | --- | --- | --- | --- | --- |"]
    for claim in ledger["claims"]:
        compact = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
        config = compact(claim["configuration"]) + "; seeds=" + ",".join(
            str(seed) for seed in claim["seeds"]) if claim["seeds"] else compact(claim["configuration"])
        lines.append("| " + " | ".join([
            compact(claim["statement"]), compact(claim["adversary_view"]), config,
            "<br>".join(f"`{artifact}`" for artifact in claim["artifacts"]),
            compact(claim["statistical_unit"]) + "; " + compact(claim["positive_control"]),
            f"**{claim['status']}**",
        ]) + " |")
    lines += ["", "## Provenance", ""]
    for claim in ledger["claims"]:
        lines += [f"### `{claim['id']}`", ""]
        for key, value in claim["provenance"].items():
            lines.append(f"- `{key}`: {value}")
        for limitation in claim["limitations"]:
            lines.append(f"- Limitation: {limitation}")
        lines.append("")
    return "\n".join(lines)


def self_test() -> int:
    ledger = load_and_validate(ROOT / "paper-data" / "claim_evidence_ledger.json")
    text = render(ledger)
    ok = "| Claim |" in text and "**withdrawn**" in text
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path,
                        default=ROOT / "paper-data" / "claim_evidence_ledger.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "CLAIM_EVIDENCE_LEDGER.md")
    parser.add_argument("--check", action="store_true",
                        help="validate only; do not write the Markdown report")
    args = parser.parse_args()
    ledger = load_and_validate(args.ledger)
    if not args.check:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render(ledger))
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test() if "--self-test" in __import__("sys").argv else main())
