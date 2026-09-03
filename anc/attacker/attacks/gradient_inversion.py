#!/usr/bin/env python3
"""gradient-inversion (training mode) — DLG++ boundary inversion.

Framework form of split-training/gradient_inversion.py (weak DLG) and
split-training/dlgpp.py (DLG++: multi-restart, cosine LR, L2+cosine
matching on boundary activation AND gradient, token-level refinement,
optional LM seq-prior).

Two paths:
  --toy          self-contained synthetic DLG on a random linear boundary:
                 victim tokens -> random embed -> random linear head;
                 attacker observes h* and g* = dL/dh and recovers the
                 tokens by continuous optimization + nearest-embedding
                 snap (the dlgpp.py loop in miniature).
  --dlgpp "--model M --corpus-file C ..."
                 runs split-training/dlgpp.py unmodified as a subprocess
                 with the given args (the full model-driven attack) and
                 normalizes its JSON into the framework
                 artifact. This is how real DLG++ runs are driven.

Usage:
    python -m attacker --mode training --attack gradient-inversion --help
    python -m attacker --mode training --attack gradient-inversion --toy \
        --quick --output /tmp/dlg.json
    python -m attacker --mode training --attack gradient-inversion \
        --dlgpp "--model /models/qwen3-0.6b --toy --quick" --output d.json
"""

import argparse
import json
import os
import shlex
import subprocess
import sys

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .. import artifacts
from .common import add_common_args, require_torch

EXPERIMENT_ID = "dlgpp_gradient_inversion"
MODES = ("training",)
REQUIRES_LABELS = False
DESCRIPTION = ("DLG++ optimization attacker on the training boundary "
               "(h*, g* = dL/dh) — multi-restart, L2+cosine match, "
               "nearest-embedding refinement")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--dlgpp", default=None, metavar='"ARGS"',
                    help="delegate to split-training/dlgpp.py with these "
                         "args (quoted string); real model-driven runs")
    ap.add_argument("--seq-len", type=int, default=16,
                    help="toy victim sequence length")
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=300,
                    help="optimization rounds per restart")
    return ap


def toy_dlg(args, seed):
    """Synthetic DLG on a random linear boundary. Returns dict of
    metrics."""
    g = torch.Generator().manual_seed(args.seed + seed)
    H, V, T = args.hidden, 64, args.seq_len
    embed = torch.randn(V, H, generator=g)
    head = torch.randn(H, H, generator=g) / (H ** 0.5)
    true_ids = torch.randint(0, V, (T,), generator=g)
    h_star = (embed[true_ids] @ head).detach()           # boundary acts
    # boundary gradient for a next-token regression loss on the head out
    target = torch.randn(T, H, generator=g)
    g_star = 2.0 * (h_star - target)                      # dL/dh (linear)

    def objs(z):
        h_hat = z @ head
        g_hat = 2.0 * (h_hat - target)
        l_h = ((h_hat - h_star) ** 2).sum()
        l_g = ((g_hat - g_star) ** 2).sum()
        return l_h + l_g

    best_z, best_l = None, None
    for r in range(args.restarts):
        z = torch.randn(T, H, generator=torch.Generator().manual_seed(
            args.seed + 900 * seed + r), requires_grad=True)
        opt = torch.optim.Adam([z], lr=0.05)
        for step in range(args.rounds):
            opt.zero_grad()
            loss = objs(z)
            loss.backward()
            opt.step()
        if best_l is None or loss.item() < best_l:
            best_l, best_z = loss.item(), z.detach()
    d = torch.cdist(best_z.float(), embed.float())
    rec_ids = d.argmin(1)
    acc = (rec_ids == true_ids).float().mean().item()
    return {"token_recovery": round(100.0 * acc, 4),
            "random_baseline": round(100.0 / V, 4),
            "final_objective": round(best_l, 6),
            "restarts": args.restarts, "rounds": args.rounds}


def run(args):
    if args.dlgpp:
        here = os.path.dirname(os.path.abspath(__file__))
        script = os.path.join(here, "..", "..", "split-training",
                              "dlgpp.py")
        if not os.path.exists(script):
            raise SystemExit(f"dlgpp.py not found at {script}")
        cmd = [sys.executable, script] + shlex.split(args.dlgpp)
        if "--output" not in args.dlgpp:
            if not args.output:
                raise SystemExit("pass --output (or include --output in "
                                 "--dlgpp args)")
            cmd += ["--output", args.output + ".dlgpp.json"]
        print(f"[dlgpp] delegating: {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            return rc
        dlg_out = cmd[cmd.index("--output") + 1]
        out = artifacts.make_artifact(
            "dtraining.attacker.dlgpp.v1",
            {"attack": EXPERIMENT_ID, "delegated": True,
             "dlgpp_args": args.dlgpp, "mode": args.mode},
            "semi-honest cloud observes boundary activation h* and "
            "gradient g* = dL/dh at the split depth (DLG++ A2 attacker, "
            "delegated to split-training/dlgpp.py).")
        with open(dlg_out) as f:
            out["results"] = [json.load(f)]
        artifacts.write_artifact(args.output, out)
        return 0
    require_torch(EXPERIMENT_ID)
    if not args.toy:
        raise SystemExit(f"[{EXPERIMENT_ID}] pass --toy, or --dlgpp for "
                         "the model-driven delegation path.")
    if args.quick:
        args.seeds, args.restarts, args.rounds = [0, 1], 2, 100
        args.seq_len = 8
    out = artifacts.make_artifact(
        "dtraining.attacker.dlgpp.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "hidden": args.hidden, "seq_len": args.seq_len,
         "restarts": args.restarts, "rounds": args.rounds,
         "seeds": args.seeds},
        "semi-honest cloud observes boundary activation h* and gradient "
        "g* = dL/dh at the split depth; optimization inversion "
        "(multi-restart, L2 match on h and g, nearest-embedding snap).",
        interpretation="toy path is a machinery check; real numbers come "
                       "from the --dlgpp delegation (split-training/"
                       "dlgpp.py, depths 1/4/8, Qwen3-0.6B).")
    for seed in args.seeds:
        rec = {"experiment": EXPERIMENT_ID, "seed": seed,
               **toy_dlg(args, seed)}
        artifacts.append_jsonl(args.output, rec)
        out["results"].append(rec)
        print(f"[dlgpp-toy] seed={seed}: token recovery="
              f"{rec['token_recovery']}% (random "
              f"{rec['random_baseline']}%)")
    out["summary"].append({
        "experiment": EXPERIMENT_ID,
        "token_recovery_mean": sum(r["token_recovery"]
                                   for r in out["results"])
        / max(1, len(out["results"]))})
    artifacts.write_artifact(args.output, out)
    return 0
