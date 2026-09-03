#!/usr/bin/env python3
"""max-effort — pre-emption of the "attacker not optimized to completion"
objection.

Several committed defenses rest on attacks run with a handful of seeds and
one budget grid. This driver reruns a given attack with more solve seeds,
multiple independent initializations, and a swept budget grid, then reports
the DISTRIBUTION (min/p25/mean/p75/max across all reruns), not just the
mean — so a paper claim can be stated against an optimized attacker, not
the median one.

Implemented as an in-process driver over the factorized toy core of a
registered attack (currently: accumulation, ica-bss). Attacks without a
factorized core can be driven per-cell via `python -m attacker --attack X
...` and aggregated from the JSONL journals.

Usage:
    python -m attacker --mode training --attack max-effort --help
    python -m attacker --mode training --attack max-effort --toy --quick \
        --target accumulation --output /tmp/me.json
"""

import argparse
import statistics

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .. import artifacts
from .common import add_common_args, require_torch

EXPERIMENT_ID = "max_effort"
MODES = ("training", "inference")
REQUIRES_LABELS = "inherits the target attack"
DESCRIPTION = ("reruns a given attack with more solve seeds, multiple "
               "initializations, and swept budgets; reports the "
               "distribution (not just the mean) — the 'not optimized to "
               "completion' pre-emptor")

# factorized per-cell cores: name -> (callable(args, budget, init_seed)
#                                     -> top1 float)
_CORES = {}


def _core_accumulation(args, budget, init_seed):
    from ..synthetic import make_toy_world
    from ..solve_primitives import solve_w
    from .common import recovery_with_what_nn
    world = make_toy_world(hidden=args.hidden, n_public=4096, n_victim=512,
                           n_epochs=1, master_seed=args.seed + 1000,
                           seed=args.seed + init_seed)
    g = torch.Generator().manual_seed(args.seed + 911 * init_seed)
    order = torch.randperm(world["public_h"].shape[0], generator=g)
    idx = order[:budget]
    w_hat, tag = solve_w(world["public_h"][idx],
                         world["wire"](world["public_h"][idx], 0))
    if w_hat is None:
        return None
    return recovery_with_what_nn(world["wire"](world["victim_h"], 0), w_hat,
                                 world["public_h"], world["public_tok"],
                                 world["victim_tok"])


def _core_ica(args, budget, init_seed):
    from ..synthetic import make_toy_world
    from ..ica import fastica
    from .common import nn_mean_decode
    world = make_toy_world(hidden=args.hidden, n_public=budget + 512,
                           n_victim=512, n_epochs=1,
                           master_seed=args.seed + 1000,
                           seed=args.seed + init_seed)
    hw = world["wire"](world["public_h"][:budget], 0)
    fit = fastica(hw, seed=args.seed + 71 * init_seed)
    s_ref = (world["public_h"].double() - fit["mean"]) @ fit["unmixing"]
    s_vic = (world["victim_h"].double() - fit["mean"]) @ fit["unmixing"]
    return nn_mean_decode(s_vic.float(), s_ref.float(),
                          world["public_tok"], world["victim_tok"])


_CORES["accumulation"] = _core_accumulation
_CORES["ica-bss"] = _core_ica


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--target", default="accumulation",
                    help="attack to optimize (factorized cores: "
                         + ", ".join(sorted(_CORES)) + ")")
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[32, 64, 128, 256],
                    help="swept budget (target-specific: labeled pairs / "
                         "capture rows per cell)")
    ap.add_argument("--inits", type=int, default=8,
                    help="independent initializations per budget")
    return ap


def pct(vals, q):
    s = sorted(vals)
    if not s:
        return None
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[k]


def run(args):
    require_torch(EXPERIMENT_ID)
    if args.target not in _CORES:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] no factorized core for '{args.target}'. "
            f"Available: {', '.join(sorted(_CORES))}. For other attacks, "
            "drive them per-cell via `python -m attacker --attack X` and "
            "aggregate the .jsonl journals (see attacker/README.md).")
    if not args.toy:
        raise SystemExit(f"[{EXPERIMENT_ID}] --toy only (factorized toy "
                         "cores).")
    if args.quick:
        args.budgets = args.budgets[:3]
        args.inits = 4
    core = _CORES[args.target]
    out = artifacts.make_artifact(
        "dtraining.attacker.max_effort.v1",
        {"attack": EXPERIMENT_ID, "target": args.target, "mode": args.mode,
         "toy": True, "hidden": args.hidden, "budgets": args.budgets,
         "inits": args.inits, "seed": args.seed},
        "max-effort attacker: the target attack rerun with swept budgets "
        "and multiple independent initializations; reported as a "
        "DISTRIBUTION (min/p25/mean/p75/max), pre-empting the 'we did not "
        "optimize the attacker to completion' objection.",
        interpretation="Defense claims should be quoted against the p75/max "
                       "of this distribution, not the mean of a "
                       "single-seed run.")
    for B in args.budgets:
        vals, errors = [], 0
        for init in range(args.inits):
            try:
                v = core(args, B, init)
            except RuntimeError as e:
                artifacts.append_jsonl(args.output, {
                    "experiment": EXPERIMENT_ID, "target": args.target,
                    "budget": B, "init": init, "error": str(e)})
                errors += 1
                continue
            if v is None:
                errors += 1
                continue
            vals.append(v)
        dist = {"min": min(vals) if vals else None,
                "p25": pct(vals, 0.25),
                "mean": (statistics.fmean(vals) if vals else None),
                "p75": pct(vals, 0.75),
                "max": max(vals) if vals else None,
                "std": (statistics.pstdev(vals) if len(vals) > 1 else 0.0),
                "n_ok": len(vals), "n_error": errors}
        rec = {"experiment": EXPERIMENT_ID, "target": args.target,
               "budget": B, "distribution": dist, "all_runs": vals}
        artifacts.append_jsonl(args.output, rec)
        out["results"].append(rec)
        out["summary"].append(rec)
        print(f"[max-effort] target={args.target} budget={B}: "
              f"mean={dist['mean']} p75={dist['p75']} max={dist['max']} "
              f"(n={dist['n_ok']}, err={errors})")
    artifacts.write_artifact(args.output, out)
    return 0
