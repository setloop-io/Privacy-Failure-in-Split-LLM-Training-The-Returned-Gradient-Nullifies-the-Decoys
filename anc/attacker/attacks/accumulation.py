#!/usr/bin/env python3
"""accumulation (E-R1a) — per-epoch oracle accumulation budget.

Framework form of the E-R1a attack core of split-training/rotation_lifetime.py:
the attacker gathers E
labeled (h, h @ W_t) pairs from WITHIN one epoch, solves polar(lstsq) for
W_hat_t (attacker.solve_primitives.solve_w), and scores victim-token
recovery — plus the cross-epoch pooling control (pairs pooled across
differently-keyed epochs must sit in the label-free band).

Sources of pairs:
  --toy                     synthetic world (machinery check)
  --capture-dir DIR +       real wire captures grouped per epoch via
  --canonical-pt FILE       attacker.captures; FILE is a torch-saved dict
                            {"h": [N,H], "tok": [N]} of the attacker's
                            canonical replay rows aligned by row order
                            within (epoch, phase)

Usage:
    python -m attacker --mode training --attack accumulation --help
    python -m attacker --mode training --attack accumulation --toy \
        --quick --output /tmp/er1.json
"""

import argparse

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .. import artifacts, captures
from ..solve_primitives import (K50_THRESHOLD_PCT, crossing_k,
                                partition_pairs, solve_w)
from .common import (add_common_args, journal_error, recovery_with_what_nn,
                     require_torch)

EXPERIMENT_ID = "er1a_accumulation"
MODES = ("training", "inference")
REQUIRES_LABELS = True
DESCRIPTION = ("E-R1a: E labeled pairs from within ONE epoch -> polar "
               "lstsq solve -> victim recovery; cross-epoch pooling "
               "control must sit in the label-free band")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--epoch-sizes", type=int, nargs="+",
                    default=[64, 256, 1024],
                    help="labeled-pairs-per-epoch sweep (E)")
    ap.add_argument("--capture-dir", default=None)
    ap.add_argument("--canonical-pt", default=None,
                    help="torch-saved {\"h\": [N,H], \"tok\": [N]} canonical "
                         "replay rows (row-aligned to the captures within "
                         "each epoch)")
    ap.add_argument("--phase", default=None,
                    help="training captures: restrict to fwd or bwd")
    return ap


def align_canonical(epochs, sizes, canon_rows, canon_tok):
    """Row-align canonical replay rows to captures, per epoch.

    Returns ({epoch: rows}, {epoch: tokens}). Both come from the SAME offset
    window, so a victim's wire rows and its labels cannot be drawn from
    different epochs.

    Pure slicing, so it works on tensors or plain sequences and is testable
    without torch.
    """
    total = sum(sizes)
    if len(canon_rows) < total or len(canon_tok) < total:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] canonical replay has "
            f"{len(canon_rows)} rows / {len(canon_tok)} tokens but the "
            f"captures need {total}; slices would silently truncate and every "
            f"recovery would be scored against shifted labels")
    rows, toks, off = {}, {}, 0
    for ep, n in zip(epochs, sizes):
        rows[ep], toks[ep] = canon_rows[off:off + n], canon_tok[off:off + n]
        off += n
    return rows, toks


def solve_and_score(h_in, hw_in, victim, solve_seeds, seed):
    """Multi-solve-seed accumulation cell. Returns (list of top1, err)."""
    recs, err = [], None
    for ss in solve_seeds:
        g = torch.Generator().manual_seed(seed + 500000 + ss)
        order = torch.randperm(h_in.shape[0], generator=g)
        w_hat, tag = solve_w(h_in[order], hw_in[order])
        if w_hat is None:
            err = tag
            continue
        recs.append(recovery_with_what_nn(
            victim["h_wire"], w_hat, victim["ref_h"], victim["ref_tok"],
            victim["victim_tok"]))
    return recs, err


def run(args):
    require_torch(EXPERIMENT_ID)
    from ..synthetic import make_toy_world
    if args.quick:
        args.epoch_sizes = args.epoch_sizes[:2]
        args.seeds = [0, 1]
        args.solve_seeds = [0, 1]
    if not args.toy and not (args.capture_dir and args.canonical_pt):
        raise SystemExit(f"[{EXPERIMENT_ID}] pass --toy, or --capture-dir "
                         "+ --canonical-pt for real captures.")
    out = artifacts.make_artifact(
        "dtraining.attacker.er1a.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": args.toy,
         "capture_dir": args.capture_dir, "hidden": args.hidden,
         "epoch_sizes": args.epoch_sizes, "seeds": args.seeds,
         "solve_seeds": args.solve_seeds, "phase": args.phase},
        "oracle/self-labeled attacker under per-epoch rotation: E labeled "
        "(h, h @ W_t) pairs from WITHIN one epoch -> polar(lstsq) W solve "
        "-> victim-token recovery. Cross-epoch pooling control (pairs "
        "pooled across differently-keyed epochs solved as one W) must sit "
        "in the label-free band.",
        interpretation="recovery < 50% at epoch size E => the per-epoch "
                       "budget is safe at E; the K50 crossing is the "
                       "honest security parameter vs the static-W prior.")
    for seed in args.seeds:
        world = make_toy_world(hidden=args.hidden,
                               n_public=max(4096, 2 * max(args.epoch_sizes)
                                            + 512),
                               n_victim=512, n_epochs=5,
                               master_seed=args.seed + 1000 * seed,
                               seed=args.seed + seed)
        if args.capture_dir:
            records = captures.scan_captures(args.capture_dir, args.mode)
            by_ep = captures.load_epoch_rows(records, phase=args.phase)
            can = torch.load(args.canonical_pt, map_location="cpu")
            canon_rows, canon_tok = can["h"].float(), can["tok"]
            # row-align canonical rows to captures within each epoch
            epochs = sorted(by_ep, key=lambda e: (e is None, e))
            sizes = [by_ep[ep].shape[0] for ep in epochs]
            canon_by_ep, canon_tok_by_ep = align_canonical(
                epochs, sizes, canon_rows, canon_tok)
            wire_by_ep = by_ep
            ref_h, ref_tok = canon_rows, canon_tok
            # victim = held-out wire rows of the first epoch, labelled from
            # THAT epoch's canonical slice.
            ep0 = epochs[0]
            nv = min(512, wire_by_ep[ep0].shape[0] // 4)
            victim = {"h_wire": wire_by_ep[ep0][-nv:].double(),
                      "ref_h": ref_h, "ref_tok": ref_tok,
                      "victim_tok": canon_tok_by_ep[ep0][-nv:]}
        else:
            epochs = list(range(world["n_epochs"]))
            ref_h, ref_tok = world["public_h"], world["public_tok"]
            victim = {"h_wire": world["wire"](world["victim_h"], 0),
                      "ref_h": ref_h, "ref_tok": ref_tok,
                      "victim_tok": world["victim_tok"]}
        # Pairs the within-epoch solve can actually draw: on real captures
        # only epoch ep0 is read, so the budget check must use ep0's row
        # count, not the canonical grand total.
        n_avail = (canon_by_ep[epochs[0]].shape[0] if args.capture_dir
                   else ref_h.shape[0])
        curve = []
        for E in args.epoch_sizes:
            if E > n_avail // 2:
                print(f"[er1] E={E}: only {n_avail} pairs available, "
                      "skipping")
                continue
            if args.capture_dir:
                ep0 = epochs[0]
                h_in = canon_by_ep[ep0][:E]
                hw_in = wire_by_ep[ep0][:E].double()
            else:
                h_in = ref_h[:E]
                hw_in = world["wire"](h_in, 0)
            recs, err = solve_and_score(h_in, hw_in, victim,
                                        args.solve_seeds, args.seed + seed)
            # cross-epoch control: E pairs pooled over 4 epochs
            parts = partition_pairs(E, min(4, len(epochs)))
            hp, hwp = [], []
            off = E
            for t, n_t in zip(epochs, parts):
                if args.capture_dir:
                    hp.append(canon_by_ep[t][off:off + n_t])
                    hwp.append(wire_by_ep[t][off:off + n_t].double())
                else:
                    hp.append(ref_h[off:off + n_t])
                    hwp.append(world["wire"](ref_h[off:off + n_t], t))
                off += n_t
            pooled, perr = solve_and_score(torch.cat(hp), torch.cat(hwp),
                                           victim, args.solve_seeds,
                                           args.seed + seed + 77)
            cell = {"experiment": EXPERIMENT_ID, "epoch_size": E,
                    "seed": seed}
            if not recs:
                journal_error(args.output, EXPERIMENT_ID, cell,
                              err or perr or "no successful solve")
                continue
            m = sum(recs) / len(recs)
            mp = (sum(pooled) / len(pooled)) if pooled else None
            rec = {**cell, "within_epoch_top1": round(m, 4),
                   "pooled_cross_epoch_top1": mp,
                   "n_solves": len(recs)}
            artifacts.append_jsonl(args.output, rec)
            out["results"].append(rec)
            curve.append({"K": E, "top1_mean": m})
            print(f"[er1] E={E} seed={seed}: within={m:.2f}% "
                  f"pooled={mp}%")
        if curve:
            out["summary"].append({
                "experiment": EXPERIMENT_ID, "seed": seed,
                "max_safe_epoch": max(
                    (c["K"] for c in curve
                     if c["top1_mean"] < K50_THRESHOLD_PCT), default=None),
                "k50_crossing": crossing_k(curve)})
    artifacts.write_artifact(args.output, out)
    return 0


def self_test():
    """Row/label alignment. Pure python: align_canonical only slices."""
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("accumulation canonical alignment:")
    # Tokens equal the row index, so a label drawn from the wrong epoch is
    # immediately visible.
    epochs, sizes = [0, 1, 2], [4, 3, 2]
    rows = [f"r{i}" for i in range(9)]
    toks = list(range(9))
    by_ep, tok_by_ep = align_canonical(epochs, sizes, rows, toks)

    check("each epoch's rows and tokens come from the same window",
          all(by_ep[e] == [f"r{t}" for t in tok_by_ep[e]] for e in epochs))
    check("epoch 0 gets the first slice", tok_by_ep[0] == [0, 1, 2, 3])
    check("epoch 2 gets the last slice", tok_by_ep[2] == [7, 8])

    # The victim is the tail of epoch 0's wire rows, so its labels must be
    # the tail of epoch 0's tokens.
    nv = 2
    ep0 = epochs[0]
    check("victim labels are epoch 0's tail, not the corpus tail",
          tok_by_ep[ep0][-nv:] == [2, 3])
    check("the old grand-total slice would have taken epoch 2's tail",
          toks[sum(sizes) - nv:sum(sizes)] == [7, 8])
    check("so the two disagree whenever there is more than one epoch",
          tok_by_ep[ep0][-nv:] != toks[sum(sizes) - nv:sum(sizes)])

    single, single_tok = align_canonical([0], [4], rows, toks)
    check("with a single epoch the old and new slices agree, which is why "
          "this stayed invisible",
          single_tok[0][-nv:] == toks[4 - nv:4])

    short = False
    try:
        align_canonical(epochs, sizes, rows[:5], toks[:5])
    except SystemExit:
        short = True
    check("a canonical replay shorter than the captures fails loudly", short)

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1
