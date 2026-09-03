#!/usr/bin/env python3
"""stale-key (E-R4) — ratcheted keys / stale-key attack.

Framework form of split-training/rotation_lifetime.py E-R4: the attacker is
HANDED W_1..W_3 plus the derivation rule (not master_seed) and attacks
epoch 4 with (i) zero fresh pairs and (ii) E fresh pairs. Stale keys must
give no bootstrap: (i) sits in the label-free band with W-alignment
rel_err ~= sqrt(2) (random); (ii) matches the E-R1a curve.

Usage:
    python -m attacker --mode training --attack stale-key --help
    python -m attacker --mode training --attack stale-key --toy --quick \
        --output /tmp/er4.json
"""

import argparse

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .. import artifacts
from ..solve_primitives import (RANDOM_REL_ERR, ratchet_seed, solve_w,
                                w_rel_err)
from .common import (add_common_args, journal_error, recovery_with_what_nn,
                     require_torch)

EXPERIMENT_ID = "er4_stale_key"
MODES = ("training", "inference")
REQUIRES_LABELS = "insider (stale keys) + optional fresh labeled pairs"
DESCRIPTION = ("E-R4: handed stale keys W_1..W_3 + the derivation rule "
               "(not master_seed), attack epoch 4 with zero / E fresh "
               "pairs — stale keys must give no bootstrap")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--stale-epochs", type=int, default=3,
                    help="handed keys W_1..W_k")
    ap.add_argument("--target-epoch", type=int, default=4)
    ap.add_argument("--fresh-pair-budgets", type=int, nargs="+",
                    default=[64, 256],
                    help="E fresh labeled pairs in the target epoch")
    return ap


def run(args):
    require_torch(EXPERIMENT_ID)
    from ..synthetic import make_toy_world
    if not args.toy:
        raise SystemExit(f"[{EXPERIMENT_ID}] real-run driver: use "
                         "split-training/rotation_lifetime.py "
                         "--experiment er4; --toy exercises the attack "
                         "math.")
    if args.quick:
        args.seeds, args.solve_seeds = [0, 1], [0]
        args.fresh_pair_budgets = args.fresh_pair_budgets[:1]
    out = artifacts.make_artifact(
        "dtraining.attacker.er4.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "hidden": args.hidden, "stale_epochs": args.stale_epochs,
         "target_epoch": args.target_epoch,
         "fresh_pair_budgets": args.fresh_pair_budgets,
         "seeds": args.seeds},
        "insider HANDED stale keys W_1..W_k plus the derivation rule but "
        "not master_seed (the ratchet is one-way: stale keys + the rule "
        "do not yield future keys). Attacks the target epoch with zero "
        "fresh pairs and with E fresh pairs.",
        interpretation="zero-fresh-pair recovery must sit in the "
                       "label-free band with W rel_err ~= sqrt(2); "
                       "fresh-pair cells must match the E-R1a curve at "
                       "the same E (stale keys give zero bootstrap).")
    for seed in args.seeds:
        master = args.seed + 1000 * seed
        world = make_toy_world(hidden=args.hidden, n_public=4096,
                               n_victim=512,
                               n_epochs=args.target_epoch + 1,
                               master_seed=master, seed=args.seed + seed)
        t = args.target_epoch
        victim_wire = world["wire"](world["victim_h"], t)
        # (i) zero fresh pairs: score every stale key — for a one-way
        # ratchet even the closest stale key is random w.r.t. W_t.
        stale_rows = []
        for k in range(1, args.stale_epochs + 1):
            w_stale = world["Ws"][k]
            rel = w_rel_err(w_stale.double(), world["Ws"][t].double())
            top1 = recovery_with_what_nn(
                victim_wire, w_stale.double(), world["public_h"],
                world["public_tok"], world["victim_tok"])
            stale_rows.append({"stale_epoch": k, "w_rel_err": rel,
                               "top1": top1})
        rec = {"experiment": EXPERIMENT_ID, "seed": seed, "arm":
               "zero_fresh_pairs", "target_epoch": t,
               "random_rel_err_reference": RANDOM_REL_ERR,
               "stale_results": stale_rows,
               "best_stale_top1": max(r["top1"] for r in stale_rows)}
        artifacts.append_jsonl(args.output, rec)
        out["results"].append(rec)
        print(f"[er4] seed={seed} zero-fresh: best stale top1="
              f"{rec['best_stale_top1']:.2f}% (rel_err ~ "
              f"{stale_rows[-1]['w_rel_err']})")
        # (ii) E fresh pairs in epoch t (stale keys held but, by design,
        # useless for the solve)
        for E in args.fresh_pair_budgets:
            recs, err = [], None
            for ss in args.solve_seeds:
                g = torch.Generator().manual_seed(master + ss)
                order = torch.randperm(world["public_h"].shape[0],
                                       generator=g)
                idx = order[:E]
                w_hat, tag = solve_w(world["public_h"][idx],
                                     world["wire"](world["public_h"][idx],
                                                   t))
                if w_hat is None:
                    err = tag
                    continue
                recs.append(recovery_with_what_nn(
                    victim_wire, w_hat, world["public_h"],
                    world["public_tok"], world["victim_tok"]))
            cell = {"experiment": EXPERIMENT_ID, "seed": seed,
                    "arm": "fresh_pairs", "epoch_size": E,
                    "target_epoch": t}
            if not recs:
                journal_error(args.output, EXPERIMENT_ID, cell, err)
                continue
            rec = {**cell, "top1_mean": sum(recs) / len(recs),
                   "n_solves": len(recs),
                   "note": "stale keys held but unused by the solve — "
                           "must match E-R1a at the same E"}
            artifacts.append_jsonl(args.output, rec)
            out["results"].append(rec)
            print(f"[er4] seed={seed} E={E}: top1={rec['top1_mean']:.2f}%")
    artifacts.write_artifact(args.output, out)
    return 0


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # pure-python: the one-way property the attack relies on
    s = [ratchet_seed(999, t) for t in range(6)]
    check("ratchet chain is one-way-ordered (no trivial f(t)->t+1)",
          len(set(s)) == 6 and s[4] != ratchet_seed(s[3], 1))
    check("stale seed does not derive the next epoch's seed",
          ratchet_seed(999, 4) != ratchet_seed(ratchet_seed(999, 3), 1))
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1
