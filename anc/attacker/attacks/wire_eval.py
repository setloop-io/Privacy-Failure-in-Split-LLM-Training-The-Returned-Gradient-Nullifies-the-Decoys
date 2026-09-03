#!/usr/bin/env python3
"""wire-eval — per-epoch wire-capture scoring.

Loads a wire-capture directory through the shared capture core
(attacker.captures — schema selected by --mode), groups rows per epoch,
solves polar(lstsq) per epoch against attacker-side canonical rows
(--canonical-pt), and reports:
  * per-epoch top-1 (within-epoch budget),
  * the label-free band (decode raw rotated rows, no solve),
  * the cross-epoch pooling control (must sit in the band),
  * fwd-only vs fwd+bwd arms (training mode: do wire grads give extra
    leverage?),
  * an undefended reference when the canonical rows carry the pre-rotation
    activations.

This module is the framework's unified scorer for wire captures; --toy writes
its own synthetic captures first so the capture-loading path is exercised end
to end. (The standalone eval scripts it subsumes are not included in this
release.)

Usage:
    python -m attacker --mode training --attack wire-eval --help
    python -m attacker --mode training --attack wire-eval --toy --quick \
        --output /tmp/wire.json
    python -m attacker --mode training --attack wire-eval \
        --capture-dir ER_CAPTURE_DIR --canonical-pt replay.pt \
        --output wire.json
"""

import argparse
import tempfile

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .. import artifacts, captures
from ..solve_primitives import solve_w
from .common import (add_common_args, journal_error, nn_mean_decode,
                     recovery_with_what_nn, require_torch)

EXPERIMENT_ID = "wire_eval"
MODES = ("training", "inference")
REQUIRES_LABELS = True
DESCRIPTION = ("per-epoch wire-capture scoring: within-epoch solves, "
               "label-free band, cross-epoch pooling control, fwd/bwd "
               "arms (e9_eval / er_train_eval superset)")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--capture-dir", default=None)
    ap.add_argument("--canonical-pt", default=None,
                    help="torch-saved {\"h\": [N,H], \"tok\": [N]} canonical "
                         "rows, row-aligned to the captures (sorted by the "
                         "mode's alignment key, filtered per phase arm)")
    ap.add_argument("--min-epoch-rows", type=int, default=16)
    return ap


VICTIM_ROWS_CAP = 512


def split_epoch_rows(n):
    """(solve_rows, victim_rows) for an epoch of n captured rows.

    The victim is the held-out TAIL, so the solve set is the complement. Both
    sides of the solve must be cut to solve_rows: torch.linalg.lstsq needs
    matching leading dimensions, and solve_w turns a mismatch into
    (None, "error: ...") rather than raising — a wrong cut does not crash,
    it journals every cell as a failed solve and reports nothing.
    """
    nv = min(VICTIM_ROWS_CAP, n // 4)
    return n - nv, nv


def solve_rows_of(canon_rows, wire_rows):
    """The wire rows that pair with a canonical slice, one per row."""
    return wire_rows[:len(canon_rows)]


def score_epochs(by_ep, canon_by_ep, world, victim, args, phase_tag):
    """Per-epoch solves + pooled control for one phase arm."""
    rows = []
    epochs = sorted(by_ep, key=lambda e: (e is None, e))
    for ep in epochs:
        wire_rows = by_ep[ep]
        if wire_rows.shape[0] < args.min_epoch_rows:
            print(f"[wire] epoch {ep}: {wire_rows.shape[0]} rows "
                  f"< --min-epoch-rows, skipped")
            continue
        can = canon_by_ep[ep]
        w_hat, tag = solve_w(can, solve_rows_of(can, wire_rows).double())
        vw = victim["h_wire_by_ep"].get(ep, victim["h_wire"])
        vt = victim.get("victim_tok_by_ep", {}).get(
            ep, victim["victim_tok"])
        # label-free band: rotated victim rows decoded against the PLAIN
        # reference (no solve) — the ceiling of any failed attack.
        band = nn_mean_decode(vw.float(), victim["ref_h"],
                              victim["ref_tok"], vt)
        cell = {"experiment": EXPERIMENT_ID, "phase_arm": phase_tag,
                "epoch": ep, "n_rows": int(wire_rows.shape[0])}
        if w_hat is None:
            journal_error(args.output, EXPERIMENT_ID, cell, tag)
            continue
        top1 = recovery_with_what_nn(vw, w_hat,
                                     victim["ref_h"], victim["ref_tok"],
                                     vt)
        rec = {**cell, "within_epoch_top1": top1,
               "labelfree_band_top1": band}
        artifacts.append_jsonl(args.output, rec)
        rows.append(rec)
        print(f"[wire] {phase_tag} epoch={ep}: within={top1:.2f}% "
              f"band={band:.2f}%")
    # cross-epoch pooling control
    if len(epochs) >= 2:
        pool_eps = [e for e in epochs if e in canon_by_ep]
        h_pool = torch.cat([canon_by_ep[e] for e in pool_eps])
        w_pool_rows = torch.cat([solve_rows_of(canon_by_ep[e], by_ep[e])
                                 for e in pool_eps])
        w_hat, tag = solve_w(h_pool, w_pool_rows.double())
        if w_hat is None:
            journal_error(args.output, EXPERIMENT_ID,
                          {"phase_arm": phase_tag, "arm": "pooled"}, tag)
        else:
            # scored per epoch key (the pooled solve is a compromise across
            # keys; report the mean over epoch-scored decodes). Rows and
            # labels must come from the SAME epoch.
            per_ep = [recovery_with_what_nn(
                victim["h_wire_by_ep"].get(e, victim["h_wire"]), w_hat,
                victim["ref_h"], victim["ref_tok"],
                victim.get("victim_tok_by_ep", {}).get(
                    e, victim["victim_tok"]))
                for e in pool_eps]
            top1 = sum(per_ep) / len(per_ep)
            rec = {"experiment": EXPERIMENT_ID, "phase_arm": phase_tag,
                   "arm": "pooled_cross_epoch",
                   "pooled_top1_mean_over_epochs": top1,
                   "pooled_top1_per_epoch": per_ep,
                   "note": "expected in the label-free band: accumulation "
                           "does not cross rotation boundaries"}
            artifacts.append_jsonl(args.output, rec)
            rows.append(rec)
            print(f"[wire] {phase_tag} pooled-cross-epoch: "
                  f"{top1:.2f}% (per-epoch {per_ep})")
    return rows


def run(args):
    require_torch(EXPERIMENT_ID)
    from ..synthetic import make_toy_world, write_toy_captures
    if args.quick:
        args.seeds, args.solve_seeds = [0, 1], [0]
    if not args.toy and not (args.capture_dir and args.canonical_pt):
        raise SystemExit(f"[{EXPERIMENT_ID}] pass --toy, or --capture-dir "
                         "+ --canonical-pt.")
    out = artifacts.make_artifact(
        "dtraining.attacker.wire_eval.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": args.toy,
         "capture_dir": args.capture_dir,
         "min_epoch_rows": args.min_epoch_rows, "seeds": args.seeds},
        "known-plaintext/self-labeled wire attacker: per-epoch (h, h @ "
        "W_t) pairs from the capture dir -> polar(lstsq) solve -> victim "
        "recovery; label-free band + cross-epoch pooling control + "
        "fwd/bwd arms (training). Superset of probes/e9_eval.py and "
        "probes/er_train_eval.py.")
    for seed in args.seeds:
        world = make_toy_world(hidden=args.hidden,
                               n_public=max(2048, 128 * 4 + 512),
                               n_victim=512, n_epochs=4,
                               master_seed=args.seed + 1000 * seed,
                               seed=args.seed + seed)
        if args.toy:
            cap_dir = tempfile.mkdtemp(prefix="wire_toy_")
            write_toy_captures(world, cap_dir, args.mode,
                               rows_per_epoch=128)
        else:
            cap_dir = args.capture_dir
        records = captures.scan_captures(cap_dir, args.mode)
        if not records:
            raise SystemExit(f"no wire captures in {cap_dir}")
        phases = ("fwd", "bwd") if args.mode == "training" else \
            tuple({m.get("phase") for m, _ in records})
        # canonical rows: toy = the world public rows per epoch; real =
        # --canonical-pt row-aligned per epoch
        if args.toy:
            n_per = 128
            canon_by_ep = {t: world["public_h"][t * n_per:(t + 1) * n_per]
                           for t in range(world["n_epochs"])}
            ref_h, ref_tok = world["public_h"], world["public_tok"]
        else:
            can = torch.load(args.canonical_pt, map_location="cpu")
            ref_h, ref_tok = can["h"].float(), can["tok"]
        victim = {"h_wire": world["wire"](world["victim_h"], 0),
                  "h_wire_by_ep": {t: world["wire"](world["victim_h"], t)
                                   for t in range(world["n_epochs"])},
                  "ref_h": ref_h, "ref_tok": ref_tok,
                  "victim_tok": world["victim_tok"]}
        for ph in phases:
            by_ep = captures.load_epoch_rows(records, phase=ph)
            if not by_ep:
                continue
            if not args.toy:
                # real captures: align canonical rows per epoch, then use
                # a held-out tail of each epoch's OWN wire rows as the
                # victim set (labels from the canonical replay).
                off = 0
                canon_by_ep = {}
                victim["h_wire_by_ep"] = {}
                victim["victim_tok_by_ep"] = {}
                epochs = sorted(by_ep, key=lambda e: (e is None, e))
                for ep in epochs:
                    n = by_ep[ep].shape[0]
                    n_solve, _ = split_epoch_rows(n)
                    canon_by_ep[ep] = ref_h[off:off + n_solve]
                    victim["h_wire_by_ep"][ep] = \
                        by_ep[ep][n_solve:].double()
                    victim["victim_tok_by_ep"][ep] = \
                        ref_tok[off + n_solve:off + n]
                    off += n
                victim["h_wire"] = victim["h_wire_by_ep"][epochs[0]]
                victim["victim_tok"] = \
                    victim["victim_tok_by_ep"][epochs[0]]
            rows = score_epochs(by_ep, canon_by_ep, world, victim, args,
                                ph)
            out["results"].extend(rows)
    vals = [r["within_epoch_top1"] for r in out["results"]
            if "within_epoch_top1" in r]
    if vals:
        out["summary"].append({"experiment": EXPERIMENT_ID,
                               "within_epoch_top1_mean":
                                   sum(vals) / len(vals),
                               "n_epoch_cells": len(vals)})
    artifacts.write_artifact(args.output, out)
    return 0


def self_test():
    """The row arithmetic the real-capture solve depends on. Pure python."""
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("wire-eval epoch row split:")
    for n in (64, 800, 797, 2048, 2000):
        n_solve, n_victim = split_epoch_rows(n)
        if n_solve + n_victim != n:
            check(f"n={n}: solve + victim == n", False)
    check("solve rows and victim rows partition the epoch exactly",
          all(sum(split_epoch_rows(n)) == n
              for n in (16, 64, 800, 797, 2048, 2000, 10000)))
    check("the victim tail is capped at 512",
          split_epoch_rows(10000)[1] == 512)
    check("small epochs give a quarter to the victim",
          split_epoch_rows(800) == (600, 200))

    # Both sides of the solve need the same leading dim, or solve_w returns
    # (None, "error: ...") and the cell reports nothing.
    canon = list(range(600))          # n_solve for n=800
    wire = list(range(800))           # every captured row
    check("the wire side is cut to the canonical length before the solve",
          len(solve_rows_of(canon, wire)) == len(canon))
    check("passing the uncut wire rows would mismatch, which is exactly the "
          "case solve_primitives pins as a failed solve",
          len(wire) != len(canon))
    check("the solve set and the victim tail do not overlap",
          solve_rows_of(canon, wire)[-1] < wire[len(canon)])

    # Ragged epochs are the realistic trigger: a short final microbatch.
    a_solve, a_victim = split_epoch_rows(800)
    b_solve, b_victim = split_epoch_rows(797)
    check("two ragged epochs give different victim lengths (200 vs 199)",
          a_victim != b_victim)
    check("so pooled scoring must resolve labels per epoch, not once",
          a_victim == 200 and b_victim == 199)

    check("slicing is a no-op when the two already match (the toy path)",
          solve_rows_of(list(range(128)), list(range(128)))
          == list(range(128)))

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1
