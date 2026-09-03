#!/usr/bin/env python3
"""sharded (E-R5) — block-diagonal sharded secrets with staggered
rotation.

Framework form of split-training/rotation_lifetime.py E-R5: W block-diagonal with
s blocks. NEGATIVE CONTROL: static sharding WEAKENS per-block budgets
(0.1*h_b pairs suffice per block). The mechanism defense: staggered
rotation — block i rotates when (epoch + i*(period//s)) % period == 0 — so
no 2-epoch sliding window has all blocks on the same key and the composed
solve fails.

Usage:
    python -m attacker --mode training --attack sharded --help
    python -m attacker --mode training --attack sharded --toy --quick \
        --output /tmp/er5.json
"""

import argparse

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .. import artifacts
from ..solve_primitives import (block_diag_secret, solve_w,
                                stagger_rotating_blocks)
from .common import (add_common_args, journal_error, recovery_with_what_nn,
                     require_torch)

EXPERIMENT_ID = "er5_sharded"
MODES = ("training", "inference")
REQUIRES_LABELS = True
DESCRIPTION = ("E-R5: per-block solves against block-diagonal secrets; "
               "staggered rotation defeats the 2-epoch sliding-window "
               "composed solve")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, hidden=64)
    ap.add_argument("--shards", type=int, nargs="+", default=[1, 4],
                    help="block-diagonal shard counts s")
    ap.add_argument("--pairs-per-block", type=int, default=16,
                    help="labeled pairs per block (static arm)")
    ap.add_argument("--rotation-period", type=int, default=4)
    return ap


def run(args):
    require_torch(EXPERIMENT_ID)
    from ..synthetic import make_toy_world
    if not args.toy:
        raise SystemExit(f"[{EXPERIMENT_ID}] real-run driver: use "
                         "split-training/rotation_lifetime.py "
                         "--experiment er5; --toy exercises the attack "
                         "math.")
    if args.quick:
        args.seeds, args.solve_seeds = [0, 1], [0]
        args.shards = args.shards[:2]
    out = artifacts.make_artifact(
        "dtraining.attacker.er5.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "hidden": args.hidden, "shards": args.shards,
         "pairs_per_block": args.pairs_per_block,
         "rotation_period": args.rotation_period, "seeds": args.seeds},
        "oracle-labeled attacker against block-diagonal sharded secrets: "
        "static arm = per-block solves at a fraction of the full-W budget "
        "(negative control: sharding alone WEAKENS the budget); staggered "
        "arm = 2-epoch sliding-window composed solve, which must fail "
        "because no window has all blocks on the same key.",
        interpretation="static sharding is not a defense (per-block K50 "
                       "drops ~1/s); the staggered schedule is what "
                       "restores it.")
    for s in args.shards:
        if args.hidden % s:
            print(f"[er5] hidden={args.hidden} not divisible by s={s}, "
                  "skipping")
            continue
        hb = args.hidden // s
        for seed in args.seeds:
            world = make_toy_world(hidden=args.hidden, n_public=4096,
                                   n_victim=512, n_epochs=2,
                                   master_seed=args.seed + 1000 * seed,
                                   seed=args.seed + seed)
            # static arm: per-block solves
            w_static = block_diag_secret(args.hidden, s, args.seed + seed)
            hw = world["public_h"].double() @ w_static.double()
            recs = []
            for b in range(s):
                sl = slice(b * hb, (b + 1) * hb)
                n = min(args.pairs_per_block, world["public_h"].shape[0])
                w_hat_b, tag = solve_w(
                    world["public_h"][:n, sl].contiguous(),
                    hw[:n, sl].contiguous())
                if w_hat_b is None:
                    journal_error(args.output, EXPERIMENT_ID,
                                  {"shards": s, "seed": seed, "block": b},
                                  tag)
                    continue
                vrec = recovery_with_what_nn(
                    (world["victim_h"].double()
                     @ w_static.double())[:, sl].float(),
                    w_hat_b, world["public_h"][:, sl],
                    world["public_tok"], world["victim_tok"])
                recs.append(vrec)
            if recs:
                rec = {"experiment": EXPERIMENT_ID, "arm": "static",
                       "shards": s, "seed": seed,
                       "pairs_per_block": args.pairs_per_block,
                       "per_block_top1_mean": sum(recs) / len(recs)}
                artifacts.append_jsonl(args.output, rec)
                out["results"].append(rec)
                print(f"[er5] static s={s} seed={seed}: per-block mean="
                      f"{rec['per_block_top1_mean']:.2f}%")
            # staggered arm: 2-epoch window composed solve
            period = args.rotation_period
            w_e0 = block_diag_secret(args.hidden, s, args.seed + seed)
            rot = stagger_rotating_blocks(1, s, period)
            w_e1 = w_e0.clone()
            g = torch.Generator().manual_seed(args.seed + seed + 5)
            for b in rot:
                sl = slice(b * hb, (b + 1) * hb)
                q, r = torch.linalg.qr(torch.randn(hb, hb, generator=g))
                w_e1[:, sl] = w_e0[:, sl] @ (q * torch.sign(
                    torch.diagonal(r)))
            n = 4 * args.pairs_per_block
            h_pool = torch.cat([world["public_h"][:n],
                                world["public_h"][n:2 * n]])
            hw_pool = torch.cat([
                world["public_h"][:n].double() @ w_e0.double(),
                world["public_h"][n:2 * n].double() @ w_e1.double()])
            w_hat, tag = solve_w(h_pool, hw_pool)
            if w_hat is None:
                journal_error(args.output, EXPERIMENT_ID,
                              {"shards": s, "seed": seed,
                               "arm": "staggered"}, tag)
            else:
                # Per-block decode against epoch-0 wire rows: the composed
                # (pooled) solve must fail ON THE ROTATED BLOCKS (their keys
                # differ across the window) while unrotated blocks stay
                # recoverable — stated at block granularity.
                vw = world["victim_h"].double() @ w_e0.double()
                per_block = {}
                for b in range(s):
                    sl = slice(b * hb, (b + 1) * hb)
                    per_block[b] = recovery_with_what_nn(
                        vw[:, sl].float(), w_hat[sl, sl],
                        world["public_h"][:, sl], world["public_tok"],
                        world["victim_tok"])
                rot_vals = [per_block[b] for b in rot] or [None]
                unrot_vals = [v for b, v in per_block.items()
                              if b not in rot]
                rec = {"experiment": EXPERIMENT_ID, "arm": "staggered",
                       "shards": s, "seed": seed,
                       "rotated_blocks_in_window": rot,
                       "per_block_top1": {str(b): v for b, v in
                                          per_block.items()},
                       "rotated_block_top1_mean": (
                           sum(v for v in rot_vals if v is not None)
                           / max(1, len([v for v in rot_vals
                                         if v is not None]))),
                       "unrotated_block_top1_mean": (
                           sum(unrot_vals) / max(1, len(unrot_vals))),
                       "note": "rotated blocks must sit in the label-free "
                               "band: no 2-epoch window has all blocks on "
                               "the same key"}
                artifacts.append_jsonl(args.output, rec)
                out["results"].append(rec)
                print(f"[er5] staggered s={s} seed={seed}: rotated-block "
                      f"mean={rec['rotated_block_top1_mean']:.2f}% "
                      f"unrotated={rec['unrotated_block_top1_mean']:.2f}% "
                      f"(rotated {rot})")
    artifacts.write_artifact(args.output, out)
    return 0


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    got = {e: stagger_rotating_blocks(e, 4, 8) for e in range(16)}
    check("stagger map: no 2-epoch window sees all 4 blocks rotate",
          all(len(set(got[e]) | set(got[e + 1])) < 4 for e in range(15)))
    check("every block rotates exactly twice in 16 epochs",
          all(sum(i in got[e] for e in range(16)) == 2 for i in range(4)))
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1
