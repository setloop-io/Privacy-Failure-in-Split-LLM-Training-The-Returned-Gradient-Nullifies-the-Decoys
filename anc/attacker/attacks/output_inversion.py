#!/usr/bin/env python3
"""output-inversion (inference mode) — serving/output-side inversion.

Framework form of split-training/output_inversion.py (train a decoder on
OUTPUT-side features — the cloud's final-layer hidden states / logits —
to recover the served tokens) and the sequence-aware SipIt line
(split-training/sipit_inversion.py): under exact arithmetic the hidden
states are injective in the input, so left-to-right nearest-neighbor
verification recovers the sequence; the framework scores the trained-
decoder surrogate.

Two paths:
  --toy      synthetic output features: token -> random output embedding
             through a fixed random map; NN decode (machinery check).
  --output-inversion-script "--model M ..."
             runs split-training/output_inversion.py
             unmodified for model-driven runs.

Usage:
    python -m attacker --mode inference --attack output-inversion --help
    python -m attacker --mode inference --attack output-inversion --toy \
        --quick --output /tmp/oinv.json
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
from .common import add_common_args, nn_mean_decode, require_torch

EXPERIMENT_ID = "output_inversion"
MODES = ("inference",)
REQUIRES_LABELS = False
DESCRIPTION = ("serving/output-side inversion: decoder on the cloud's "
               "output features (SipIt-style injectivity line + trained "
               "decoder) — inference surface only")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--output-inversion-script", default=None,
                    metavar='"ARGS"',
                    help="delegate to split-training/output_inversion.py "
                         "with these args (quoted string)")
    return ap


def run(args):
    if args.output_inversion_script:
        here = os.path.dirname(os.path.abspath(__file__))
        script = os.path.join(here, "..", "..", "split-training",
                              "output_inversion.py")
        if not os.path.exists(script):
            raise SystemExit(f"output_inversion.py not found at {script}")
        cmd = [sys.executable, script] + shlex.split(
            args.output_inversion_script)
        if "--output" not in args.output_inversion_script:
            if not args.output:
                raise SystemExit("pass --output (or include --output in "
                                 "the delegation args)")
            cmd += ["--output", args.output + ".oinv.json"]
        print(f"[oinv] delegating: {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            return rc
        del_out = cmd[cmd.index("--output") + 1]
        out = artifacts.make_artifact(
            "dtraining.attacker.output_inversion.v1",
            {"attack": EXPERIMENT_ID, "delegated": True,
             "script_args": args.output_inversion_script,
             "mode": args.mode},
            "semi-honest cloud in split inference observes its own output "
            "features; decoder recovers the served tokens (delegated to "
            "split-training/output_inversion.py).")
        with open(del_out) as f:
            out["results"] = [json.load(f)]
        artifacts.write_artifact(args.output, out)
        return 0
    require_torch(EXPERIMENT_ID)
    if not args.toy:
        raise SystemExit(f"[{EXPERIMENT_ID}] pass --toy, or "
                         "--output-inversion-script for delegation.")
    if args.quick:
        args.seeds = [0, 1]
    out = artifacts.make_artifact(
        "dtraining.attacker.output_inversion.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "hidden": args.hidden, "seeds": args.seeds},
        "semi-honest cloud in split inference observes its own OUTPUT "
        "features (final hidden states); nearest-mean decoder recovers "
        "the served tokens. Toy path is a machinery check for the "
        "artifact/journal wiring; real numbers come from the delegation "
        "path or split-training/output_inversion.py directly.",
        interpretation="the output side is the weaker surface for the "
                       "defender: output features are a near-invertible "
                       "function of the tokens (SipIt injectivity), so "
                       "this attacker needs no boundary labels at all.")
    for seed in args.seeds:
        g = torch.Generator().manual_seed(args.seed + seed)
        V, H = 64, args.hidden
        tok_emb = torch.randn(V, H, generator=g)
        tok_emb = tok_emb / tok_emb.norm(dim=1, keepdim=True)
        out_map = torch.randn(H, H, generator=g) / (H ** 0.5)
        ref_tok = torch.randint(0, V, (2048,), generator=g)
        vic_tok = torch.randint(0, V, (512,), generator=g)
        ref_f = 4.0 * tok_emb[ref_tok] @ out_map \
            + 0.1 * torch.randn(2048, H, generator=g)
        vic_f = 4.0 * tok_emb[vic_tok] @ out_map \
            + 0.1 * torch.randn(512, H, generator=g)
        top1 = nn_mean_decode(vic_f, ref_f, ref_tok, vic_tok)
        rec = {"experiment": EXPERIMENT_ID, "seed": seed,
               "output_decode_top1": top1,
               "random_baseline": round(100.0 / V, 4)}
        artifacts.append_jsonl(args.output, rec)
        out["results"].append(rec)
        print(f"[oinv-toy] seed={seed}: output decode top-1={top1}%")
    artifacts.write_artifact(args.output, out)
    return 0
