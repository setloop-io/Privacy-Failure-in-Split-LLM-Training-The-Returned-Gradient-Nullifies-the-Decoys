#!/usr/bin/env python3
"""Regenerate all publication figures + TABLES.md from paper-data/results JSONs.

Usage:  python3 make_figures.py        (run from paper-data/, or anywhere)
Missing input files are tolerated with a warning (re-run as more data lands).
"""
import glob
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ---------------------------------------------------------------- style ----
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
})
MISSING = []

# ------------------------------------------------------- paper constants ---
# Reference values from the original paper (cited in captions).
PAPER_ACCEPT = {  # acceptance, n-gram size 3..5; only n3 + per-category at n5 given
    "7B":  {"n3": 1.21, "code": 1.43, "structured": 1.20, "creative": 1.12, "conversational": 1.17},
    "12B": {"n3": 1.21, "code": 1.57, "structured": 1.23, "creative": 1.06, "conversational": 1.12},
}
PAPER_TOKS_80MS = {"7B": {"sequential": 8.3, "lookahead": 9.3},
                   "12B": {"sequential": 8.0, "lookahead": 8.7}}
PAPER_INVERSION = {2: 58.8, 4: 44.3, 6: 44.8, 8: 34.8}  # top-1 % vs local layers
PAPER_DECOMP_7B = {"rtt": 77.4, "local": 26.2, "cloud": 15.9, "serialization": 1.0, "fixed": 42.9}

CATEGORIES = ["code", "structured", "creative", "conversational"]
MODELS = {  # label -> (rtt glob pattern, ablation dir, ablation glob)
    "7B":  ("rtt/rtt_mistral-7b-instruct_*ms.json",   "ablation_7b",  "ablation_*_80ms.json"),
    "12B": ("rtt/rtt_mistral-nemo-12b-instruct_*ms.json", "ablation_12b", "ablation_*_80ms.json"),
    "27B": ("rtt/rtt_qwen36-27b_*ms.json",            "ablation_27b", "ablation_*_80ms.json"),
}
ZERO_RTT = {"7B": "rtt/rtt_7b_0ms.json", "12B": "rtt/rtt_12b_0ms.json", "27B": "rtt/rtt_qwen36-27b_0ms.json"}


def load(path):
    """Load JSON, warning + None if missing."""
    if not os.path.exists(path):
        MISSING.append(os.path.relpath(path, RESULTS))
        warnings.warn(f"missing: {os.path.relpath(path, RESULTS)}")
        return None
    with open(path) as f:
        return json.load(f)


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"wrote figures/{name}.{{pdf,png}}")


# ------------------------------------------------------------ data load ----
def rtt_points(model):
    """Return {mode: {'rtt': [...], 'tok': [...], 'fixed': [...]}} sorted by RTT."""
    out = {}
    paths = sorted(glob.glob(os.path.join(RESULTS, MODELS[model][0])))
    z = os.path.join(RESULTS, ZERO_RTT[model])
    if os.path.exists(z) and z not in paths:
        paths.append(z)
    if not paths:
        MISSING.append(MODELS[model][0])
        warnings.warn(f"no RTT files for {model}")
    for p in paths:
        d = load(p)
        if not d or "summary" not in d:
            continue
        for mode in ("sequential", "lookahead"):
            s = d["summary"].get(mode)
            if not s:
                continue
            e = out.setdefault(mode, {"rtt": [], "tok": [], "fixed": []})
            e["rtt"].append(s["avg_measured_rtt_ms"])
            e["tok"].append(s["avg_measured_tok_s"])
            e["fixed"].append(s["avg_fixed_overhead_ms"])
    for mode, e in out.items():
        order = np.argsort(e["rtt"])
        for k in ("rtt", "tok", "fixed"):
            e[k] = [e[k][i] for i in order]
    return out


def ablation_accept(model, rtt="80ms"):
    """{category: n5_acceptance} from ablation summary_by_category."""
    pat = os.path.join(RESULTS, MODELS[model][1], MODELS[model][2].replace("80ms", rtt))
    paths = sorted(glob.glob(pat))
    if not paths and rtt != "0ms":
        pat = os.path.join(RESULTS, MODELS[model][1], MODELS[model][2].replace("80ms", "0ms"))
        paths = sorted(glob.glob(pat))
    if not paths:
        MISSING.append(f"{MODELS[model][1]} ablation")
        warnings.warn(f"no ablation for {model}")
        return {}
    d = load(paths[0])
    if not d:
        return {}
    return {c: d.get("summary_by_category", {}).get(c, {}) for c in CATEGORIES}


RTT = {m: rtt_points(m) for m in MODELS}
ABL = {m: ablation_accept(m) for m in MODELS}
DEF = load(os.path.join(RESULTS, "defense/defense_results.json"))
INV = load(os.path.join(RESULTS, "inversion_7b/inversion_results.json"))
TRANSPORT = {}
for t in ("direct", "tunnel"):
    for r in ("0ms", "80ms"):
        TRANSPORT[(t, r)] = load(os.path.join(RESULTS, f"transport/transport_{t}_{r}.json"))

# ------------------------------------------------------------ figure 1 -----
fig, ax = plt.subplots(figsize=(3.6, 2.8))
colors = {"7B": "tab:blue", "12B": "tab:orange", "27B": "tab:green"}
mstyles = {"sequential": "o-", "lookahead": "s-"}
for m in MODELS:
    for mode in ("sequential", "lookahead"):
        e = RTT[m].get(mode)
        if not e:
            continue
        lbl = f"{m} {'seq' if mode == 'sequential' else 'LA'}"
        ax.plot(e["rtt"], e["tok"], mstyles[mode], color=colors[m], ms=3.5, lw=1.2, label=lbl)
        fixed = float(np.mean(e["fixed"]))
        acc = e["tok"][0] * (e["rtt"][0] + fixed) / 1000.0  # effective tokens/step
        xs = np.linspace(max(min(e["rtt"]), 1), max(max(e["rtt"]), 120), 100)
        ax.plot(xs, acc * 1000.0 / (xs + fixed), "--", color=colors[m], lw=0.8, alpha=0.5)
for m, d in PAPER_TOKS_80MS.items():
    for mode, v in d.items():
        ax.plot(80, v, "*", color=colors[m], ms=10, mec="k", mew=0.5, zorder=5,
                label=f"paper {m} {'seq' if mode == 'sequential' else 'LA'} @80ms")
ax.set_xscale("log")
ax.set_xticks([2, 5, 10, 20, 40, 80, 100])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("measured RTT (ms)")
ax.set_ylabel("throughput (tok/s)")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1)
save(fig, "fig1_throughput_vs_rtt")

# ------------------------------------------------------------ figure 2 -----
series = ["paper 7B", "GB10 7B", "paper 12B", "GB10 12B", "GB10 27B"]
vals = {s: [] for s in series}
for c in CATEGORIES:
    vals["paper 7B"].append(PAPER_ACCEPT["7B"][c])
    vals["paper 12B"].append(PAPER_ACCEPT["12B"][c])
    vals["GB10 7B"].append(ABL.get("7B", {}).get(c, {}).get("n5_acceptance", np.nan))
    vals["GB10 12B"].append(ABL.get("12B", {}).get(c, {}).get("n5_acceptance", np.nan))
    vals["GB10 27B"].append(ABL.get("27B", {}).get(c, {}).get("n5_acceptance", np.nan))
fig, ax = plt.subplots(figsize=(3.6, 2.6))
x = np.arange(len(CATEGORIES))
w = 0.16
hatches = {"paper 7B": "//", "GB10 7B": "", "paper 12B": "//", "GB10 12B": "", "GB10 27B": ""}
scolors = {"paper 7B": "tab:blue", "GB10 7B": "tab:blue", "paper 12B": "tab:orange",
           "GB10 12B": "tab:orange", "GB10 27B": "tab:green"}
for i, s in enumerate(series):
    b = ax.bar(x + (i - 2) * w, vals[s], w, label=s, color=scolors[s],
               hatch=hatches[s], edgecolor="white" if not hatches[s] else "k", lw=0.3)
    for r, v in zip(b, vals[s]):
        if not np.isnan(v):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6, rotation=90)
ax.set_xticks(x)
ax.set_xticklabels(CATEGORIES, rotation=15, ha="right")
ax.set_ylabel("acceptance (tokens/step, n=5)")
ax.set_ylim(0, max(max(v) for v in vals.values() if not all(np.isnan(v))) * 1.35)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
save(fig, "fig2_acceptance_by_category")

# ------------------------------------------------------------ figure 3 -----
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 2.8))

# (a) top-1 vs split depth
axa.plot(list(PAPER_INVERSION), list(PAPER_INVERSION.values()), "k^--", ms=5, lw=1.2,
         label="paper MLP (~880 pairs)")
if INV:
    dk = sorted(k for k in INV if k.startswith("depth_layer"))
    xs = [INV[k]["split_after_layer"] for k in dk]
    ys = [INV[k]["top1_accuracy"] for k in dk]
    axa.plot(xs, ys, "o-", color="tab:blue", ms=5, lw=1.2,
             label="GB10 same-method (~880 pairs)")
if DEF:
    depths = sorted(int(d) for d in DEF["inversion"])
    ys = [DEF["inversion"][str(d)]["none"]["top1_mean"] for d in depths]
    es = [DEF["inversion"][str(d)]["none"]["top1_std"] for d in depths]
    axa.errorbar(depths, ys, yerr=es, fmt="s-", color="tab:red", ms=5, lw=1.2, capsize=3,
                 label="GB10 stronger attacker (~9.3K pairs)")
axa.set_xlabel("split depth (local layers)")
axa.set_ylabel("inversion top-1 (%)")
axa.set_title("(a) inversion vs depth")
axa.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)

# (b) defense grid
if DEF:
    depths = sorted(int(d) for d in DEF["inversion"])
    defenses = DEF["config"]["defenses"]
    x = np.arange(len(defenses))
    w = 0.26
    dcolors = {2: "tab:blue", 4: "tab:orange", 8: "tab:green"}
    for i, d in enumerate(depths):
        ys = [DEF["inversion"][str(d)].get(df, {}).get("top1_mean", np.nan) for df in defenses]
        es = [DEF["inversion"][str(d)].get(df, {}).get("top1_std", 0) for df in defenses]
        axb.bar(x + (i - 1) * w, ys, w, yerr=es, capsize=2, color=dcolors[d],
                label=f"depth {d}", error_kw={"lw": 0.8})
        base = DEF["inversion"][str(d)]["none"]["top1_mean"]
        axb.axhline(base, color=dcolors[d], ls=":", lw=0.9, alpha=0.8)
    axb.set_xticks(x)
    axb.set_xticklabels(defenses, rotation=30, ha="right")
    axb.set_ylabel("inversion top-1 (%)")
    axb.set_title("(b) defense grid")
    axb.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=3)
else:
    axb.text(0.5, 0.5, "defense_results.json missing", ha="center", transform=axb.transAxes)
save(fig, "fig3_inversion")

# ------------------------------------------------------------ figure 4 -----
fig, ax = plt.subplots(figsize=(3.6, 3.0))
if DEF:
    depths = sorted(int(d) for d in DEF["inversion"])
    defenses = DEF["config"]["defenses"]
    markers = {"none": "o", "int8": "s", "fp8": "^", "noise_0.5%": "v",
               "noise_1.0%": "D", "noise_2.0%": "P", "noise_5.0%": "X"}
    dcolors = {2: "tab:blue", 4: "tab:orange", 8: "tab:green"}
    for df in defenses:
        for d in depths:
            inv = DEF["inversion"][str(d)].get(df)
            if not inv:
                continue
            if df == "none":
                fid = 1.0  # no defense -> bit-identical continuation
            else:
                cats = DEF.get("fidelity", {}).get(str(d), {}).get(df, {})
                if not cats:
                    continue
                fid = float(np.mean([cats[c]["token_identity_rate"] for c in cats]))
            ax.scatter(fid, inv["top1_mean"], marker=markers.get(df, "o"),
                       color=dcolors[d], s=35, edgecolor="k", lw=0.4, zorder=4)
    # legends: defense = marker, depth = color
    for df, mk in markers.items():
        ax.scatter([], [], marker=mk, color="gray", edgecolor="k", lw=0.4, label=df)
    for d, c in dcolors.items():
        ax.scatter([], [], marker="o", color=c, edgecolor="k", lw=0.4, label=f"depth {d}")
    ax.annotate("degenerate frontier:\ndefenses preserve attack rate\nwhile destroying fidelity",
                xy=(0.1, 60), xytext=(0.35, 40), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("mean token-identity rate (fidelity)")
    ax.set_ylabel("inversion top-1 (%)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1)
else:
    ax.text(0.5, 0.5, "defense_results.json missing", ha="center", transform=ax.transAxes)
save(fig, "fig4_fidelity")

# ------------------------------------------------------------ figure 5 -----
fig, ax = plt.subplots(figsize=(3.6, 2.6))
groups = [("direct", "0ms"), ("direct", "80ms"), ("tunnel", "0ms"), ("tunnel", "80ms")]
x = np.arange(len(groups))
w = 0.36
for i, mode in enumerate(("sequential", "lookahead")):
    ys, lbls = [], []
    for g in groups:
        d = TRANSPORT.get(g)
        v = d["summary"][mode]["avg_measured_tok_s"] if d else np.nan
        ys.append(v)
        lbls.append("" if np.isnan(v) else f"{v:.1f}")
    b = ax.bar(x + (i - 0.5) * w, ys, w, color="tab:blue" if i == 0 else "tab:orange",
               label="sequential" if i == 0 else "lookahead")
    for r, t in zip(b, lbls):
        if t:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.1, t,
                    ha="center", va="bottom", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([f"{t}\n{r}" for t, r in groups])
ax.set_ylabel("throughput (tok/s)")
ax.set_ylim(0, max(y for y in ys if not np.isnan(y)) * 1.2 if any(not np.isnan(y) for y in ys) else 1)
ax.legend(frameon=False)
save(fig, "fig5_transport")

# ------------------------------------------------------------ TABLES.md ----
L = []
L.append("# Reproduced results — underlying numbers\n")
L.append("Generated by `make_figures.py` from `paper-data/results/`. "
         "'paper' columns are reference values from the original paper.\n")

L.append("\n## Fig 1 — throughput vs RTT (measured tok/s)\n")
for m in MODELS:
    L.append(f"\n### {m}\n")
    L.append("| measured RTT (ms) | seq tok/s | LA tok/s | fixed overhead seq (ms) | fixed overhead LA (ms) |")
    L.append("|---|---|---|---|---|")
    rows = {}
    for mode in ("sequential", "lookahead"):
        for r, t, f in zip(RTT[m].get(mode, {}).get("rtt", []),
                           RTT[m].get(mode, {}).get("tok", []),
                           RTT[m].get(mode, {}).get("fixed", [])):
            rows.setdefault(round(r, 1), {})[mode] = (t, f)
    for r in sorted(rows):
        seq = rows[r].get("sequential", (None, None))
        la = rows[r].get("lookahead", (None, None))
        L.append(f"| {r} | {seq[0] if seq[0] is not None else '—'} | {la[0] if la[0] is not None else '—'} "
                 f"| {seq[1] if seq[1] is not None else '—'} | {la[1] if la[1] is not None else '—'} |")
L.append(f"\nPaper @80ms: 7B seq {PAPER_TOKS_80MS['7B']['sequential']} / LA {PAPER_TOKS_80MS['7B']['lookahead']}; "
         f"12B seq {PAPER_TOKS_80MS['12B']['sequential']} / LA {PAPER_TOKS_80MS['12B']['lookahead']} tok/s.\n")

L.append("\n## Fig 2 — acceptance by category (n=5)\n")
L.append("| category | paper 7B | GB10 7B | paper 12B | GB10 12B | GB10 27B |")
L.append("|---|---|---|---|---|---|")
for ci, c in enumerate(CATEGORIES):
    row = [c] + [("—" if np.isnan(vals[s][ci]) else f"{vals[s][ci]:.2f}") for s in series]
    L.append("| " + " | ".join(row) + " |")

L.append("\n## Fig 3a — inversion top-1 (%) vs split depth\n")
L.append("| depth | paper MLP (~880 pr) | GB10 same-method (~880 pr) | GB10 strong attacker (~9.3K pr) |")
L.append("|---|---|---|---|")
depths_all = sorted(set(PAPER_INVERSION) |
                    ({INV[k]['split_after_layer'] for k in INV if k.startswith('depth_layer')} if INV else set()) |
                    ({int(d) for d in DEF['inversion']} if DEF else set()))
for d in depths_all:
    pap = PAPER_INVERSION.get(d, "—")
    inv = next((f"{INV[k]['top1_accuracy']:.2f}" for k in INV if k.startswith("depth_layer")
                and INV[k]["split_after_layer"] == d), "—") if INV else "—"
    de = (f"{DEF['inversion'][str(d)]['none']['top1_mean']:.2f}±{DEF['inversion'][str(d)]['none']['top1_std']:.2f}"
          if DEF and str(d) in DEF["inversion"] else "—")
    L.append(f"| {d} | {pap} | {inv} | {de} |")
if INV:
    L.append("\nAdditional attacker variants in inversion_results.json (top-1 %): "
             + "; ".join(f"{k} = {INV[k]['top1_accuracy']}" for k in INV
                         if not k.startswith("depth_layer")) + "\n")

L.append("\n## Fig 3b / 4 — defense grid\n")
if DEF:
    L.append("\n### Inversion top-1 (%, mean±std)\n")
    L.append("| defense | " + " | ".join(f"depth {d}" for d in sorted(int(x) for x in DEF["inversion"])) + " |")
    L.append("|---|" + "---|" * len(DEF["inversion"]))
    for df in DEF["config"]["defenses"]:
        row = [df]
        for d in sorted(int(x) for x in DEF["inversion"]):
            e = DEF["inversion"][str(d)].get(df)
            row.append(f"{e['top1_mean']:.2f}±{e['top1_std']:.2f}" if e else "—")
        L.append("| " + " | ".join(row) + " |")
    L.append("\n### Fidelity — mean token-identity rate over categories\n")
    L.append("| defense | " + " | ".join(f"depth {d}" for d in sorted(int(x) for x in DEF["fidelity"])) + " |")
    L.append("|---|" + "---|" * len(DEF["fidelity"]))
    for df in DEF["config"]["defenses"]:
        if df == "none":
            row = [df] + ["1.0000 (no defense)"] * len(DEF["fidelity"])
        else:
            row = [df]
            for d in sorted(int(x) for x in DEF["fidelity"]):
                cats = DEF["fidelity"][str(d)].get(df, {})
                row.append(f"{np.mean([cats[c]['token_identity_rate'] for c in cats]):.4f}" if cats else "—")
        L.append("| " + " | ".join(row) + " |")

L.append("\n## Fig 5 — transport: direct vs tunnel (tok/s)\n")
L.append("| transport | RTT | seq tok/s | LA tok/s |")
L.append("|---|---|---|---|")
for (t, r), d in TRANSPORT.items():
    if d:
        L.append(f"| {t} | {r} | {d['summary']['sequential']['avg_measured_tok_s']} "
                 f"| {d['summary']['lookahead']['avg_measured_tok_s']} |")

L.append("\n## Latency decomposition @80ms, 7B sequential: GB10 vs paper (ms/step)\n")
L.append("| component | GB10 | paper |")
L.append("|---|---|---|")
d80 = load(os.path.join(RESULTS, "rtt/rtt_mistral-7b-instruct_80ms.json"))
if d80:
    s = d80["summary"]["sequential"]
    L.append(f"| network RTT | {s['avg_measured_rtt_ms']} | {PAPER_DECOMP_7B['rtt']} |")
    L.append(f"| local compute | {s['avg_local_compute_ms']} | {PAPER_DECOMP_7B['local']} |")
    L.append(f"| cloud compute | {s['avg_cloud_compute_ms']} | {PAPER_DECOMP_7B['cloud']} |")
    ser = s["avg_fixed_overhead_ms"] - s["avg_local_compute_ms"] - s["avg_cloud_compute_ms"]
    L.append(f"| serialization+other (fixed−local−cloud) | {ser:.1f} | {PAPER_DECOMP_7B['serialization']} |")
    L.append(f"| fixed overhead total | {s['avg_fixed_overhead_ms']} | {PAPER_DECOMP_7B['fixed']} |")

with open(os.path.join(HERE, "TABLES.md"), "w") as f:
    f.write("\n".join(L) + "\n")
print("wrote TABLES.md")

if MISSING:
    print("\nWARNING — missing inputs (re-run when they land):")
    for m in MISSING:
        print("  -", m)
