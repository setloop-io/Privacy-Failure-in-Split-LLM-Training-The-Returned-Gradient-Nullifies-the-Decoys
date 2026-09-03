#!/usr/bin/env python3
"""subspace-joint — joint attacker across epochs over the stable low-dim
activation subspace.

Per-epoch rotation (E-R1a) bounds the attacker to the pairs it can harvest
WITHIN one epoch. But boundary activations live on a stable low-dim
subspace (the basis B barely moves across epochs — only the key rotates).
A joint attacker alternates minimization over

  (shared basis B, per-epoch keys W_t)  s.t.  h_t ≈ B c_t,  wire_t = h_t W_t

which needs FAR fewer labeled pairs per epoch than the independent
per-epoch solves: once B is estimated from pooled public activations, each
epoch's key is a k×k problem (k = subspace dim), not H×H. If it works,
max_safe_epoch drops and the B=128 budget choice needs revision.

Implemented: alternating minimization (B = top-k PCA of the pooled
canonical rows; per-epoch reduced solve for Q_t on the subspace with a
lifted W_hat_t = I - B B^T + B Q_t B^T), a pairs-per-epoch sweep, and the
head-to-head against the independent solve_w at the same budget.

Usage:
    python -m attacker --mode training --attack subspace-joint --help
    python -m attacker --mode training --attack subspace-joint --toy \
        --quick --output /tmp/sj.json
"""

import argparse

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .. import artifacts
from ..solve_primitives import polar, solve_w
from .common import (add_common_args, journal_error, recovery_with_what_nn,
                     require_torch)

EXPERIMENT_ID = "subspace_joint"
MODES = ("training", "inference")
REQUIRES_LABELS = True
DESCRIPTION = ("cross-epoch joint attacker: shared low-dim basis B + "
               "per-epoch keys W_t by alternating minimization — needs "
               "fewer pairs/epoch than independent solves; if it works, "
               "max_safe_epoch drops and B=128 needs revision")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--subspace-dims", type=int, nargs="+", default=[8, 16],
                    help="candidate subspace dims k")
    ap.add_argument("--pairs-per-epoch", type=int, nargs="+",
                    default=[4, 8, 16, 32],
                    help="labeled pairs available per epoch (sweep)")
    ap.add_argument("--epochs", type=int, default=4,
                    help="joint epochs t")
    ap.add_argument("--alt-iters", type=int, default=5,
                    help="alternating-minimization iterations")
    return ap


def joint_solve(canon_by_t, wire_by_t, k, n_iters):
    """Alternating minimization over shared basis B and per-epoch reduced
    keys Q_t. canon_by_t/wire_by_t: lists of [n_t, H] labeled pair matrices
    (same row order). Returns (Ws_hat list, diagnostics)."""
    H = canon_by_t[0].shape[1]
    pooled = torch.cat(canon_by_t).double()
    mu = pooled.mean(0)
    _, _, V = torch.pca_lowrank(pooled - mu, q=k)
    B = V  # [H, k]
    Qs = [torch.eye(k, dtype=torch.float64) for _ in canon_by_t]
    for _ in range(n_iters):
        # E-step: per-epoch reduced key given B:  min || (h B) Q - (wire B) ||
        for t in range(len(canon_by_t)):
            c = (canon_by_t[t].double() - mu) @ B
            z = (wire_by_t[t].double() - mu) @ B
            try:
                sol = torch.linalg.lstsq(c.contiguous(),
                                         z.contiguous()).solution
                Qs[t] = polar(sol)
            except RuntimeError as e:
                return None, {"error": "error: " + str(e).splitlines()[0]}
        # M-step: re-fit B on residual-aligned rows (canonical rows rotated
        # by the current per-epoch estimates, pooled)
        aligned_c, aligned_w = [], []
        for t in range(len(canon_by_t)):
            aligned_c.append(canon_by_t[t].double() - mu)
            aligned_w.append((wire_by_t[t].double() - mu)
                             @ (torch.eye(H, dtype=torch.float64)
                                - B @ B.T + B @ Qs[t] @ B.T).T)
        joint = torch.cat([torch.cat([a, b]) for a, b in
                           zip(aligned_c, aligned_w)])
        _, _, V = torch.pca_lowrank(joint, q=k)
        B = V
    Ws = [torch.eye(H, dtype=torch.float64) - B @ B.T + B @ Q @ B.T
          for Q in Qs]
    return Ws, {"k": k, "alt_iters": n_iters}


def run(args):
    require_torch(EXPERIMENT_ID)
    from ..synthetic import make_toy_world
    if not args.toy:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] real-capture driver is an integration "
            "point (feed per-epoch capture pairs from --capture-dir); "
            "--toy exercises the full alternating-minimization math.")
    if args.quick:
        args.epochs = 3
        args.seeds = [0, 1]
        args.solve_seeds = [0]
        args.pairs_per_epoch = args.pairs_per_epoch[:3]
        args.subspace_dims = args.subspace_dims[:1]
    out = artifacts.make_artifact(
        "dtraining.attacker.subspace_joint.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "toy": True,
         "hidden": args.hidden, "subspace_dims": args.subspace_dims,
         "pairs_per_epoch": args.pairs_per_epoch, "epochs": args.epochs,
         "alt_iters": args.alt_iters, "seeds": args.seeds},
        "oracle-labeled attacker under per-epoch rotation, but JOINT across "
        "epochs: exploits the stable low-dim activation subspace by "
        "alternating minimization over (shared basis B, per-epoch keys "
        "W_t), needing fewer labeled pairs per epoch than the independent "
        "per-epoch polar(lstsq) solves of E-R1a.",
        interpretation="If joint recovery crosses 50% at pairs-per-epoch "
                       "where the independent solve is still in the band, "
                       "max_safe_epoch drops and the B=128 budget needs "
                       "revision.")
    for k in args.subspace_dims:
        for E in args.pairs_per_epoch:
            per_seed_joint, per_seed_ind = [], []
            for seed in args.seeds:
                world = make_toy_world(hidden=args.hidden,
                                       n_public=max(4096, E * args.epochs
                                                    * len(args.solve_seeds)
                                                    + 512),
                                       n_victim=512, n_epochs=args.epochs,
                                       master_seed=args.seed + 1000 * seed,
                                       seed=args.seed + seed)
                joint_recs, ind_recs = [], []
                for ss in args.solve_seeds:
                    g = torch.Generator().manual_seed(args.seed + 5000 * seed
                                                      + ss)
                    order = torch.randperm(world["public_h"].shape[0],
                                           generator=g)
                    canon_by_t, wire_by_t = [], []
                    off = 0
                    for t in range(args.epochs):
                        idx = order[off:off + E]
                        off += E
                        canon_by_t.append(world["public_h"][idx])
                        wire_by_t.append(world["wire"](
                            world["public_h"][idx], t))
                    try:
                        Ws_hat, diag = joint_solve(canon_by_t, wire_by_t, k,
                                                   args.alt_iters)
                    except RuntimeError as e:
                        journal_error(args.output, EXPERIMENT_ID,
                                      {"k": k, "pairs_per_epoch": E,
                                       "seed": seed, "solve_seed": ss}, e)
                        continue
                    if Ws_hat is None:
                        journal_error(args.output, EXPERIMENT_ID,
                                      {"k": k, "pairs_per_epoch": E,
                                       "seed": seed, "solve_seed": ss},
                                      diag["error"])
                        continue
                    recs_t = [recovery_with_what_nn(
                        world["wire"](world["victim_h"], t), Ws_hat[t],
                        world["public_h"], world["public_tok"],
                        world["victim_tok"]) for t in range(args.epochs)]
                    joint_recs.append(sum(recs_t) / len(recs_t))
                    # independent baseline: same budget, plain solve_w
                    ind_t = []
                    for t in range(args.epochs):
                        w_hat, tag = solve_w(canon_by_t[t], wire_by_t[t])
                        if w_hat is None:
                            continue
                        ind_t.append(recovery_with_what_nn(
                            world["wire"](world["victim_h"], t), w_hat,
                            world["public_h"], world["public_tok"],
                            world["victim_tok"]))
                    if ind_t:
                        ind_recs.append(sum(ind_t) / len(ind_t))
                if not joint_recs:
                    continue
                jm = sum(joint_recs) / len(joint_recs)
                im = (sum(ind_recs) / len(ind_recs)) if ind_recs else None
                per_seed_joint.append(jm)
                per_seed_ind.append(im)
                rec = {"experiment": EXPERIMENT_ID, "k": k,
                       "pairs_per_epoch": E, "seed": seed,
                       "joint_top1_mean": round(jm, 4),
                       "independent_top1_mean": im,
                       "n_solve_seeds": len(joint_recs)}
                artifacts.append_jsonl(args.output, rec)
                out["results"].append(rec)
                print(f"[sj] k={k} E={E} seed={seed}: joint={jm:.2f}% "
                      f"independent={im}%")
            if per_seed_joint:
                out["summary"].append({
                    "experiment": EXPERIMENT_ID, "k": k,
                    "pairs_per_epoch": E,
                    "joint_mean": sum(per_seed_joint)
                    / len(per_seed_joint),
                    "independent_mean": (
                        sum(x for x in per_seed_ind if x is not None)
                        / max(1, len([x for x in per_seed_ind
                                      if x is not None]))),
                    "n_seeds": len(per_seed_joint)})
    artifacts.write_artifact(args.output, out)
    return 0
