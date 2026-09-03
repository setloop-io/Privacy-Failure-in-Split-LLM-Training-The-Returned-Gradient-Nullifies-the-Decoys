#!/usr/bin/env python3
"""Regenerate the manuscript figures from committed artifacts.

Every figure is drawn from a committed JSON under paper-data/; nothing is
hand-transcribed. Usage: python3 build_figures.py (from this directory).
Outputs PDFs into this directory and prints a sha256 manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
W56 = ROOT / "paper-data/collected/diagnostic/gradaudit/w56_gradaudit_summary.json"
W24 = ROOT / "paper-data/collected/diagnostic/w24_metric_sweep/w24_dose_response.json"
OUT = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 150,
})


def _load(path: Path) -> Any:
    assert path.is_file(), f"missing artifact: {path}"
    return json.loads(path.read_text())


def _cells() -> List[Dict[str, Any]]:
    data = _load(W56)
    assert data["schema"] == "dtraining.gradaudit_w56.v1", "unexpected W5.6 schema"
    return data["cells"]


def fig_w56_shape() -> Path:
    cells = {c["cell"]: c for c in _cells()}
    ladder = [
        ("split 13\n(12 delegated)", cells["gradaudit_ladder_s13_2k"]),
        ("split 17\n(8 delegated)", cells["gradaudit_ladder_s17_2k"]),
        ("split 19\n(6 delegated)", cells["gradaudit_ladder_s19_2k"]),
    ]
    width = [
        ("D=64\n(headline)", cells["gradaudit_a2b_40k"]),
        ("D=96", cells["gradaudit_d96_10k_4k"]),
        ("D=128", cells["gradaudit_d128_10k_4k"]),
    ]
    budget = [
        ("40k steps", cells["gradaudit_a2b_40k"]),
        ("100k steps", cells["gradaudit_a2c_100k"]),
    ]
    panels: List[Tuple[str, List[Tuple[str, Dict[str, Any]]]]] = [
        ("(a) depth ladder", ladder),
        ("(b) latent width", width),
        ("(c) training budget", budget),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for ax, (title, rows) in zip(axes, panels):
        labels = [r[0] for r in rows]
        real = [r[1]["grad_real"]["paired_pp"] for r in rows]
        lo = [r[1]["grad_real"]["ci95"][0] for r in rows]
        hi = [r[1]["grad_real"]["ci95"][1] for r in rows]
        errs = [[r - l for r, l in zip(real, lo)], [h - r for r, h in zip(real, hi)]]
        ax.errorbar(labels, real, yerr=errs, fmt="o-", capsize=3,
                    color="black", ecolor="gray", linewidth=1)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(title)
        ax.set_ylim(-0.7, 1.2)
        ax.set_ylabel("paired advantage (pp)" if title == "(a) depth ladder" else "")
    fig.tight_layout()
    out = OUT / "fig_w56_shape.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def _doses() -> List[Dict[str, Any]]:
    data = _load(W24)
    assert data["schema"] == "dtraining.w24_dose_response.v1", "unexpected W2.4 schema"
    return data["doses"]


def fig_w24_dose() -> Path:
    doses = [d for d in _doses()
             if d["mode"] == "coordinate" and d["amplitude"] == 1.0 and d["coverage"] > 0]
    assert doses, "no coverage sweep rows in artifact"
    assert all(d["tag"].startswith("coord_cov") for d in doses), "non-coverage row leaked in"
    doses.sort(key=lambda d: d["coverage"])
    cov = [d["coverage"] for d in doses]
    top1 = [d["token_top1"] for d in doses]
    rare = [d["rare_token_top1"] for d in doses]
    ce = [d["token_cross_entropy"] for d in doses]
    auc = [d["membership_auc"] for d in doses]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.2))
    ax1.plot(cov, top1, "o-", label="token_top1", color="black")
    ax1.plot(cov, rare, "s--", label="rare_token_top1", color="gray")
    ax1.set_xscale("log")
    ax1.axhline(1.0, color="red", linewidth=0.8, linestyle=":")
    ax1.annotate("+1.0 pp gate", (cov[0], 1.05), fontsize=6, color="red")
    ax1.set_xlabel("injected coverage (fraction of rows)")
    ax1.set_ylabel("recovery advantage (pp)")
    ax1.legend()
    ax2.plot(cov, ce, "o-", label="token_cross_entropy", color="black")
    ax2.plot(cov, auc, "s--", label="membership_auc (AUC-0.5)", color="gray")
    ax2.set_xscale("log")
    ax2.set_xlabel("injected coverage (fraction of rows)")
    ax2.set_ylabel("metric value (nats / AUC-0.5)")
    ax2.legend()
    fig.suptitle("dose-response per metric: the four curves disagree", y=1.02)
    fig.tight_layout()
    out = OUT / "fig_w24_dose.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    assert ROOT.is_dir(), "repository root not found"
    made = [fig_w56_shape(), fig_w24_dose()]
    for path in made:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path.name}  sha256={digest}")
    assert all(p.stat().st_size > 1000 for p in made), "figure output suspiciously small"


if __name__ == "__main__":
    main()
