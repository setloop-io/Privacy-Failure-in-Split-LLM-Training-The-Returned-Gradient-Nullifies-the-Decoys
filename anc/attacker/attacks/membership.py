#!/usr/bin/env python3
"""membership (E-A4, training mode) — membership/property inference on
captured training-boundary features.

Framework form of split-training/membership_inference.py: attacker fits a
logistic probe on captured boundary features (document-disjoint split,
3 attacker seeds, document-level bootstrap). The math module is pure
python and is imported UNCHANGED from split-training/ (that script
stays the single source of truth); this module adds the framework's
artifact/journal conventions and a synthetic fixture generator.

Input JSONL (same schema as the split-training script):
  {"document_id": str, "condition": "split_ft"|"fedavg",
   "membership": 0|1, "property": 0|1, "features": [float, ...]}

Usage:
    python -m attacker --mode training --attack membership --help
    python -m attacker --mode training --attack membership --toy --quick \
        --output /tmp/ea4.json
    python -m attacker --mode training --attack membership \
        --features run1.jsonl run2.jsonl --output ea4.json
"""

import argparse
import json
import os
import sys
import tempfile

from .. import artifacts
from .common import add_common_args

EXPERIMENT_ID = "ea4_membership"
MODES = ("training",)
REQUIRES_LABELS = False
DESCRIPTION = ("E-A4: logistic-probe membership/property inference on "
               "captured training-boundary features (wraps split-training/"
               "membership_inference.py unchanged)")


def _load_mi():
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.abspath(os.path.join(here, "..", "..", "split-training"))
    if cand not in sys.path:
        sys.path.insert(0, cand)
    import membership_inference as mi
    return mi


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--features", nargs="+", default=None,
                    help="E-A4 feature JSONL files (real captures; see "
                         "split-training/capture_ea4_features.py)")
    return ap


def make_toy_features(path, n_docs=120, n_feat=16, seed=0, effect=0.8):
    """Synthetic E-A4 records: members' features shifted by `effect` along
    a fixed direction — a detectable membership signal. Both conditions
    are emitted (membership_inference.run requires both)."""
    import random
    rng = random.Random(seed)
    direction = [rng.gauss(0, 1) for _ in range(n_feat)]
    with open(path, "w") as f:
        for cond in ("split_ft", "fedavg"):
            for i in range(n_docs):
                m = i % 2
                feat = [rng.gauss(0, 1) + effect * m * direction[j]
                        for j in range(n_feat)]
                f.write(json.dumps({
                    "document_id": f"{cond}-doc{i:04d}",
                    "condition": cond,
                    "membership": m, "property": (i // 2) % 2,
                    "features": feat}) + "\n")
    return path


def run(args):
    mi = _load_mi()
    if args.quick:
        args.seeds = [0]
    if args.toy:
        tmp = tempfile.mkdtemp(prefix="ea4_toy_")
        paths = [make_toy_features(os.path.join(tmp, "toy.jsonl"),
                                   seed=args.seed)]
    elif args.features:
        paths = args.features
    else:
        raise SystemExit(f"[{EXPERIMENT_ID}] pass --toy or --features "
                         "FILE.jsonl ...")
    out = artifacts.make_artifact(
        "dtraining.attacker.ea4.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": args.toy,
         "features": paths, "seeds": args.seeds},
        "honest-but-curious cloud with captured training-boundary features "
        "per training document; fits document-disjoint logistic probes for "
        "membership and property (3 attacker seeds, document-level "
        "bootstrap) — split-training/membership_inference.py math, "
        "unchanged.")
    try:
        inner_out = (args.output + ".membership.json") if args.output \
            else os.path.join(tempfile.gettempdir(),
                              "ea4_membership_inner.json")
        mi.run(paths, inner_out, args.seeds)
        with open(inner_out) as fh:
            summary = json.load(fh)
    except Exception as e:
        artifacts.append_jsonl(args.output, {"experiment": EXPERIMENT_ID,
                                             "error": str(e)})
        raise
    out["results"] = [summary]
    out["summary"] = out["results"]
    artifacts.write_artifact(args.output, out)
    return 0
