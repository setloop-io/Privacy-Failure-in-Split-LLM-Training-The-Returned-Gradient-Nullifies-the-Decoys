#!/usr/bin/env python3
"""known-prefix (E-R3) — scaffold randomization vs known-prefix
accumulation under a STATIC W.

Framework form of split-training/rotation_lifetime.py E-R3: sessions of an
S-token known prefix. Arm (a) fixed_scaffold = the committed break; arm
(b) jittered_scaffold = per-session pad insertion + template-slot
permutation so the PASSIVE-alignment attacker's assumed alignment is
wrong (the realignment-aware attacker is alignment-search / E-R8, a
separate module).

Usage:
    python -m attacker --mode inference --attack known-prefix --help
    python -m attacker --mode inference --attack known-prefix --toy \
        --quick --output /tmp/er3.json
"""

import argparse

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .. import artifacts
from ..solve_primitives import crossing_k, solve_w
from .common import (add_common_args, journal_error, recovery_with_what_nn,
                     require_torch)
from .alignment_search import jitter_rows

EXPERIMENT_ID = "er3_known_prefix"
MODES = ("training", "inference")
REQUIRES_LABELS = "partial (known system-prompt prefix)"
DESCRIPTION = ("E-R3: known-prefix accumulation vs fixed / jittered "
               "scaffold (static W); jittered arm must show no K90 "
               "crossing within the session budget")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--prefix-len", type=int, default=32,
                    help="tokens per known-prefix session (S)")
    ap.add_argument("--sessions", type=int, default=8,
                    help="sessions per accumulation curve")
    return ap


def run(args):
    require_torch(EXPERIMENT_ID)
    from ..synthetic import make_toy_world
    if not args.toy:
        raise SystemExit(f"[{EXPERIMENT_ID}] real-run driver: use "
                         "split-training/rotation_lifetime.py "
                         "--experiment er3 (model-driven); --toy exercises "
                         "the framework's attack math.")
    if args.quick:
        args.sessions, args.prefix_len = 4, 24
        args.seeds, args.solve_seeds = [0, 1], [0]
    out = artifacts.make_artifact(
        "dtraining.attacker.er3.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "hidden": args.hidden, "prefix_len": args.prefix_len,
         "sessions": args.sessions, "seeds": args.seeds,
         "solve_seeds": args.solve_seeds},
        "semi-honest cloud that KNOWS the system-prompt prefix (S tokens), "
        "static W. Arm fixed_scaffold: assumed alignment is correct (the "
        "committed break). Arm jittered_scaffold: per-session pad "
        "insertion + template-slot permutation breaks the passive "
        "attacker's assumed alignment.",
        interpretation="fixed_scaffold crosses the K90 threshold within a "
                       "few sessions; jittered_scaffold must not cross "
                       "within the session budget — against a PASSIVE-"
                       "alignment adversary (see alignment-search/E-R8 for "
                       "the realignment-aware one).")
    for arm in ("fixed_scaffold", "jittered_scaffold"):
        for seed in args.seeds:
            world = make_toy_world(hidden=args.hidden,
                                   n_public=max(4096, args.sessions
                                                * args.prefix_len + 512),
                                   n_victim=512, n_epochs=1,
                                   master_seed=args.seed + 1000 * seed,
                                   seed=args.seed + seed)
            w = world["Ws"][0]
            s_tok = args.prefix_len
            assumed, wire = [], []
            for si in range(args.sessions):
                h_sess = world["public_h"][si * s_tok:(si + 1) * s_tok]
                h_true = (h_sess if arm == "fixed_scaffold" else
                          jitter_rows(h_sess, pad_scale=1.0,
                                      seed=args.seed + 31337 + si))
                assumed.append(h_sess)
                wire.append(world["wire"](h_true, 0))
            h_assumed, h_wire_all = torch.cat(assumed), torch.cat(wire)
            victim_wire = world["wire"](world["victim_h"], 0)
            baseline = None
            curve = []
            for n_sess in range(1, args.sessions + 1):
                hh = h_assumed[:n_sess * s_tok]
                hw = h_wire_all[:min(n_sess * s_tok,
                                     h_wire_all.shape[0])]
                n = min(hh.shape[0], hw.shape[0])
                recs, err = [], None
                for ss in args.solve_seeds:
                    g = torch.Generator().manual_seed(
                        args.seed + 500000 + ss)
                    order = torch.randperm(n, generator=g)
                    w_hat, tag = solve_w(hh[:n][order], hw[:n][order])
                    if w_hat is None:
                        err = tag
                        continue
                    recs.append(recovery_with_what_nn(
                        victim_wire, w_hat, world["public_h"],
                        world["public_tok"], world["victim_tok"]))
                if not recs:
                    journal_error(args.output, EXPERIMENT_ID,
                                  {"arm": arm, "seed": seed,
                                   "sessions": n_sess}, err)
                    continue
                m = sum(recs) / len(recs)
                if baseline is None:
                    # undefended reference: decode with the TRUE W
                    baseline = recovery_with_what_nn(
                        victim_wire, w.double(), world["public_h"],
                        world["public_tok"], world["victim_tok"])
                curve.append({"K": n, "top1_mean": m})
                rec = {"experiment": EXPERIMENT_ID, "arm": arm,
                       "seed": seed, "sessions": n_sess, "K": n,
                       "top1_mean": round(m, 4)}
                artifacts.append_jsonl(args.output, rec)
                out["results"].append(rec)
                print(f"[er3] arm={arm} seed={seed} sessions={n_sess} "
                      f"(K={n}): top-1={m:.2f}%")
            thr = 0.9 * (baseline or 100.0)
            cross = crossing_k(curve, threshold=thr)
            out["summary"].append({
                "experiment": EXPERIMENT_ID, "arm": arm, "seed": seed,
                "baseline_top1": baseline, "k90_threshold": round(thr, 4),
                "sessions_to_k90": cross["k50_interpolated"],
                "crossing": cross})
    artifacts.write_artifact(args.output, out)
    return 0
