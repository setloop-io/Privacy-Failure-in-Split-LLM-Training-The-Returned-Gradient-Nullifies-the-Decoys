#!/usr/bin/env python3
"""leak-accumulation — persistent-APT attacker with a cross-epoch leak
budget that does NOT reset.

The E-R1a/E-R4 analyses bound the attacker per epoch — rotation resets the
accumulation window. But a persistent adversary (APT) who steals a few key
rows per epoch ACCUMULATES them across epochs: the per-epoch leak budget
does NOT reset just because the key rotated. This attack composites the
committed partial-row-leak result (e8_robustness attack 4: decoding with
leaked rows + orthogonal completion recovers the leaked coordinates
exactly) with the rotation schedule: each epoch t the attacker holds
leaked row sets L_0..L_t (a few fresh rows of each W_t) and decodes with
the per-epoch composite W_tilde_t.

Two arms:
  * per-epoch decode: recovery from W_tilde_t (leaked rows of W_t +
    random completion) — does a small per-epoch leak already clear the
    band at the tail (few leaked rows = few exact coordinates)?
  * cumulative exposure: sum of leaked coordinates over epochs — the
    quantity the per-epoch budget accounting misses.

Usage:
    python -m attacker --mode training --attack leak-accumulation --help
    python -m attacker --mode training --attack leak-accumulation --toy \
        --quick --output /tmp/leak.json
"""

import argparse

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .. import artifacts
from .common import (add_common_args, journal_error, recovery_with_what_nn,
                     require_torch)

EXPERIMENT_ID = "leak_accumulation"
MODES = ("training", "inference")
REQUIRES_LABELS = "insider row-leak (breach-case parametrization)"
DESCRIPTION = ("persistent APT: steals a few key rows per epoch and "
               "accumulates across epochs (per-epoch leak budget does NOT "
               "reset) — composite of the row-leak result + rotation")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--leak-per-epoch", type=int, nargs="+",
                    default=[1, 4, 16],
                    help="W rows stolen per epoch (budget sweep)")
    return ap


def leak_completion(w_true, row_idx, seed):
    """e8_robustness attack 4 construction: leaked rows kept EXACT, the
    remainder replaced by a random orthogonal completion of the leaked
    row space (Gram-Schmidt against the leaked rows)."""
    H = w_true.shape[0]
    g = torch.Generator().manual_seed(seed)
    leaked = w_true.double()[row_idx]
    rand = torch.randn(H, H, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(rand)
    basis = [leaked[i] / leaked[i].norm() for i in range(leaked.shape[0])]
    rows = list(basis)
    for i in range(H):
        v = q[:, i].clone()
        for b in rows:
            v = v - (v @ b) * b
        n = v.norm()
        if n > 1e-8 and len(rows) < H:
            rows.append(v / n)
        if len(rows) == H:
            break
    comp = torch.stack(rows)
    out = comp.clone()
    out[row_idx] = leaked  # exact leaked rows in their true positions
    return out


def run(args):
    require_torch(EXPERIMENT_ID)
    from ..synthetic import make_toy_world
    if not args.toy:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] real-breach driver is an integration point "
            "(feed leaked row indices per epoch); --toy exercises the full "
            "leak/completion/decode composite.")
    if args.quick:
        args.epochs = 4
        args.seeds = [0, 1]
        args.leak_per_epoch = args.leak_per_epoch[:2]
    out = artifacts.make_artifact(
        "dtraining.attacker.leak_accumulation.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "hidden": args.hidden, "epochs": args.epochs,
         "leak_per_epoch": args.leak_per_epoch, "seeds": args.seeds},
        "PERSISTENT (APT) insider: steals --leak-per-epoch rows of W_t per "
        "epoch and accumulates them ACROSS epochs (the per-epoch leak "
        "budget does NOT reset with rotation). Composite of the partial-"
        "row-leak break (e8_robustness attack 4) and the rotation "
        "schedule.",
        interpretation="The per-epoch budget accounting of E-R1a/E-R4 "
                       "does not bound this adversary: what matters is the "
                       "CUMULATIVE leaked coordinates and the per-epoch "
                       "decode achievable from the composite W_tilde_t.")
    for budget in args.leak_per_epoch:
        for seed in args.seeds:
            world = make_toy_world(hidden=args.hidden, n_public=1024,
                                   n_victim=512, n_epochs=args.epochs,
                                   master_seed=args.seed + 1000 * seed,
                                   seed=args.seed + seed)
            g = torch.Generator().manual_seed(args.seed + seed)
            cumulative = 0
            for t in range(args.epochs):
                row_idx = torch.randperm(args.hidden, generator=g)[:budget]
                try:
                    w_tilde = leak_completion(world["Ws"][t], row_idx,
                                              args.seed + seed * 31 + t)
                    top1 = recovery_with_what_nn(
                        world["wire"](world["victim_h"], t), w_tilde,
                        world["public_h"], world["public_tok"],
                        world["victim_tok"])
                except RuntimeError as e:
                    journal_error(args.output, EXPERIMENT_ID,
                                  {"budget": budget, "seed": seed,
                                   "epoch": t}, e)
                    continue
                cumulative += budget
                rec = {"experiment": EXPERIMENT_ID,
                       "leak_per_epoch": budget, "seed": seed, "epoch": t,
                       "leaked_rows_this_epoch": budget,
                       "cumulative_leaked_rows": cumulative,
                       "composite_decode_top1": top1}
                artifacts.append_jsonl(args.output, rec)
                out["results"].append(rec)
                print(f"[leak] budget={budget} seed={seed} epoch={t}: "
                      f"decode={top1:.2f}% (cumulative leak={cumulative} "
                      f"rows)")
        vals = [r for r in out["results"]
                if r["leak_per_epoch"] == budget]
        if vals:
            out["summary"].append({
                "experiment": EXPERIMENT_ID, "leak_per_epoch": budget,
                "decode_mean": sum(r["composite_decode_top1"]
                                   for r in vals) / len(vals),
                "final_cumulative_leaked_rows":
                    max(r["cumulative_leaked_rows"] for r in vals),
                "n_cells": len(vals)})
    artifacts.write_artifact(args.output, out)
    return 0
