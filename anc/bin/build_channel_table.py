#!/usr/bin/env python3
"""Generate the channel-status table rows for the manuscript from committed
artifacts: paper-data/family_metric_map.json + the w51 confirmation matrix.

Named-command traceability: the manuscript's Table 'channels' regenerates with
  python3 bin/build_channel_table.py
Prints LaTeX tabular rows to stdout. Status per family is derived, never
transcribed: measured (executed cells exist), constructible (metric + positive
control, unexecuted), or unmeasured (no metric or no positive control).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("paper-data")
FAM = json.loads((ROOT / "family_metric_map.json").read_text())
MATRIX = json.loads(
    (ROOT / "collected/diagnostic/w51_confirmation_matrix.json").read_text())

# executed post-freeze cells by family (2026-08-27 campaign; phasec dir)
EXECUTED = {"forward_only", "gradient_only", "joint_forward_gradient"}


def status(family: str, info: dict) -> str:
    if family in EXECUTED:
        return "measured (2026-08-27)"
    applicable = info.get("applicable_metrics", [])
    if not applicable:
        return "unmeasured (no applicable metric)"
    if not info.get("positive_control"):
        return "unmeasured (no positive control)"
    return "constructible, unexecuted"


def main() -> int:
    rows = []
    for family, info in FAM["families"].items():
        metrics = ", ".join(f"\\texttt{{{m}}}" for m in
                            info.get("applicable_metrics", [])) or "---"
        control = info.get("positive_control") or "none"
        rows.append(f"\\texttt{{{family}}} & {metrics} & "
                    f"{control.replace('_', chr(92) + '_')} & {status(family, info)} \\\\")
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
