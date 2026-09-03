#!/usr/bin/env python3
"""ica-bss — label-free higher-order attacker (ICA/BSS) on within-epoch
captures.

The committed label-free attacks (e8_robustness 1a/1b) fail because the
label-free problem is unidentifiable UP TO SECOND ORDER:
Q = cov(h')^{-1/2} R cov(h)^{1/2} matches the second moment for ANY
orthogonal R. That argument does NOT cover higher-order algorithms. This
attacker runs FastICA blind source separation on within-epoch wire
captures (h' = h @ W_t is an orthogonal mixture of the activation
coordinates — exactly the ICA identifiability setting) and decodes in the
recovered source space. It tests whether the ~0.5% label-free band is
robust or just untested against the right algorithm.

Dependency note: sklearn is NOT in the repo venvs, so this uses the
framework's own pure-torch FastICA (attacker/ica.py).

Usage:
    python -m attacker --mode training --attack ica-bss --help
    python -m attacker --mode training --attack ica-bss --toy --quick \
        --output /tmp/ica.json
"""

import argparse

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .. import artifacts
from ..ica import fastica
from .common import (add_common_args, journal_error, nn_mean_decode,
                     require_torch)

EXPERIMENT_ID = "ica_bss"
MODES = ("training", "inference")
REQUIRES_LABELS = False
DESCRIPTION = ("label-free ICA/BSS on within-epoch captures — tests "
               "whether the ~0.5% label-free band is robust or just "
               "untested vs the right algorithm")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--rows-per-epoch", type=int, default=4096,
                    help="wire rows fed to ICA per epoch")
    ap.add_argument("--ica-seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="ICA restarts (random orthogonal init)")
    ap.add_argument("--max-iter", type=int, default=200)
    return ap


def run(args):
    require_torch(EXPERIMENT_ID)
    from ..synthetic import make_toy_world
    if not args.toy:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] real-capture path: group wire captures per "
            "epoch (attacker.captures.load_epoch_rows) and feed rows here; "
            "--toy exercises the full ICA + decode math.")
    if args.quick:
        args.rows_per_epoch = 1024
        args.seeds = [0]
        args.ica_seeds = [0, 1]
    out = artifacts.make_artifact(
        "dtraining.attacker.ica_bss.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "hidden": args.hidden, "rows_per_epoch": args.rows_per_epoch,
         "ica_seeds": args.ica_seeds, "max_iter": args.max_iter,
         "seeds": args.seeds},
        "PASSIVE label-free observer of within-epoch wire rows h' = h @ "
        "W_t. Higher-order attacker: FastICA blind source separation on "
        "the rotated rows (orthogonal mixture identifiability), decode in "
        "the recovered source space. No (h', token) labels anywhere.",
        interpretation="Recovery above the label-free band means the band "
                       "is an artifact of second-order attackers and the "
                       "defense needs a higher-order label-free bound; "
                       "recovery in the band (expected for near-Gaussian "
                       "activations, on which ICA is unidentifiable) "
                       "strengthens the band claim.")
    for seed in args.seeds:
        world = make_toy_world(hidden=args.hidden,
                               n_public=args.rows_per_epoch + 512,
                               n_victim=512, n_epochs=2,
                               master_seed=args.seed + 1000 * seed,
                               seed=args.seed + seed)
        for epoch in range(2):
            h_wire_rows = world["wire"](
                world["public_h"][:args.rows_per_epoch], epoch)
            best = None
            for iseed in args.ica_seeds:
                try:
                    fit = fastica(h_wire_rows, max_iter=args.max_iter,
                                  seed=args.seed + 71 * iseed)
                except RuntimeError as e:
                    journal_error(args.output, EXPERIMENT_ID,
                                  {"seed": seed, "epoch": epoch,
                                   "ica_seed": iseed}, e)
                    continue
                # decode: map recovered sources of VICTIM and reference
                # rows through the same unmixing; NN in source space
                # (label-free: ref labels used ONLY for scoring)
                s_ref = (world["public_h"].double()
                         - fit["mean"]) @ fit["unmixing"]
                s_vic = (world["victim_h"].double()
                         - fit["mean"]) @ fit["unmixing"]
                top1 = nn_mean_decode(s_vic.float(), s_ref.float(),
                                      world["public_tok"],
                                      world["victim_tok"])
                if best is None or top1 > best[0]:
                    best = (top1, iseed, fit["converged"], fit["n_iter"])
            # label-free band for THIS world: decode rotated victim rows
            # against the PLAIN (unrotated) reference — the ceiling any
            # failed label-free attack sits in.
            band = nn_mean_decode(
                world["wire"](world["victim_h"], epoch).float(),
                world["public_h"], world["public_tok"],
                world["victim_tok"])
            rec = {"experiment": EXPERIMENT_ID, "seed": seed,
                   "epoch": epoch, "ica_top1_best": best[0],
                   "ica_seed_best": best[1], "converged": best[2],
                   "n_iter": best[3], "labelfree_band_top1": band,
                   "n_restarts": len(args.ica_seeds)}
            artifacts.append_jsonl(args.output, rec)
            out["results"].append(rec)
            print(f"[ica] seed={seed} epoch={epoch}: best-of-"
                  f"{len(args.ica_seeds)}={best[0]:.2f}% band={band:.2f}%")
    vals = [r["ica_top1_best"] for r in out["results"]]
    if vals:
        out["summary"].append({"experiment": EXPERIMENT_ID,
                               "ica_top1_mean": sum(vals) / len(vals),
                               "band_mean": sum(
                                   r["labelfree_band_top1"]
                                   for r in out["results"]) / len(vals),
                               "n_cells": len(vals)})
    artifacts.write_artifact(args.output, out)
    return 0
