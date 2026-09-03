#!/usr/bin/env python3
"""Enforce the development plan's dispatch rule: every task has an acceptance
test, and the dependency graph is acyclic. (The plan document itself is not
included in this release.)

Checks:
  1. Every task ID in the section 5 register has a row in the section 5.1
     acceptance register, and vice versa.
  2. Every task ID referenced anywhere in the plan document is defined in the
     register.
  3. The declared dependency graph has no cycles.

Exit 0 on success. Prints a JSON report.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = re.compile(r"W\d+\.\d+[ab]?|F\.1")
ROW = re.compile(r"^\|\s*\*{0,2}(W\d+\.\d+[ab]?|F\.1)\*{0,2}\s*\|")


def section(text: str, start: str, end: str) -> str:
    lower = text.index(start)
    upper = text.index(end, lower)
    return text[lower:upper]


def row_ids(block: str) -> list[str]:
    return [match.group(1) for line in block.splitlines()
            if (match := ROW.match(line))]


def declared_edges(register: str) -> dict[str, set[str]]:
    """Dependencies stated in a 'Blocked by' column or a 'Blocked by:' sentence."""
    edges: dict[str, set[str]] = {}
    for line in register.splitlines():
        match = ROW.match(line)
        if not match:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        task = match.group(1)
        # A 'Blocked by' column is any cell that is only task IDs and separators.
        deps: set[str] = set()
        for cell in cells[1:]:
            stripped = re.sub(r"[\s,;*]|--+|and", "", cell)
            if stripped and TASK.fullmatch(stripped) or (
                    cell and set(TASK.findall(cell)) and
                    not re.sub(TASK, "", cell).strip(" ,;*-and")):
                deps |= set(TASK.findall(cell))
        edges.setdefault(task, set()).update(deps - {task})
    return edges


def find_cycle(edges: dict[str, set[str]]) -> list[str] | None:
    colour: dict[str, int] = {}
    stack: list[str] = []

    def walk(node: str) -> list[str] | None:
        colour[node] = 1
        stack.append(node)
        for neighbour in sorted(edges.get(node, ())):
            if colour.get(neighbour) == 1:
                return stack[stack.index(neighbour):] + [neighbour]
            if colour.get(neighbour, 0) == 0:
                if cycle := walk(neighbour):
                    return cycle
        colour[node] = 2
        stack.pop()
        return None

    for node in sorted(edges):
        if colour.get(node, 0) == 0:
            if cycle := walk(node):
                return cycle
    return None


def check(plan_path: Path) -> dict:
    text = plan_path.read_text()
    register = section(text, "## 5. Task register", "### 5.1 Acceptance register")
    acceptance = section(text, "### 5.1 Acceptance register", "## 6. Experiment queue")

    tasks = row_ids(register)
    accepted = row_ids(acceptance)
    referenced = set(TASK.findall(text))

    duplicates = sorted({t for t in tasks if tasks.count(t) > 1})
    missing_acceptance = sorted(set(tasks) - set(accepted))
    orphan_acceptance = sorted(set(accepted) - set(tasks))
    undefined = sorted(referenced - set(tasks))
    cycle = find_cycle(declared_edges(register))

    problems = []
    if duplicates:
        problems.append(f"duplicate task rows: {duplicates}")
    if missing_acceptance:
        problems.append(f"tasks with no acceptance test: {missing_acceptance}")
    if orphan_acceptance:
        problems.append(f"acceptance rows for undefined tasks: {orphan_acceptance}")
    if undefined:
        problems.append(f"referenced but not defined: {undefined}")
    if cycle:
        problems.append(f"dependency cycle: {' -> '.join(cycle)}")

    # Task states, generated from git history rather than hand-maintained.
    import subprocess
    try:
        log = subprocess.run(["git", "log", "--oneline", "--all"],
                             capture_output=True, text=True, check=False).stdout
    except Exception:
        log = ""
    touched = sorted({t for t in tasks if t in log})
    untouched = sorted(set(tasks) - set(touched))

    return {"tasks": len(tasks), "acceptance_rows": len(accepted),
            "dispatchable": not problems, "problems": problems,
            "states": {
                "mentioned_in_commits": len(touched),
                "never_mentioned": len(untouched),
                "sums_to_total": len(touched) + len(untouched) == len(tasks),
            },
            "never_mentioned_ids": untouched,
            "note": ("mentioned_in_commits is NOT the same as done. A task is done only "
                     "when its section 5.1 acceptance criterion is met.")}


def self_test() -> int:
    report = check(ROOT / "PLAN.md")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("SELF-TEST " + ("PASSED" if report["dispatchable"] else "FAILED"))
    return 0 if report["dispatchable"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "PLAN.md")
    args = parser.parse_args()
    report = check(args.plan)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["dispatchable"] else 1


if __name__ == "__main__":
    raise SystemExit(self_test() if "--self-test" in __import__("sys").argv
                     else main())
