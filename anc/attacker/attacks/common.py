"""Shared helpers for attack modules: CLI conventions, toy world setup,
decode surrogates, and per-cell error journaling."""

import argparse

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .. import artifacts
from ..solve_primitives import h_wire, ratchet_secret


def add_common_args(ap, hidden=64):
    """The flags every attack shares (repo conventions)."""
    ap.add_argument("--output", default=None,
                    help="artifact JSON path; <output>.jsonl is the per-cell "
                         "crash journal")
    ap.add_argument("--toy", action="store_true",
                    help="synthetic tensors machinery check (no model needed)")
    ap.add_argument("--hidden", type=int, default=hidden,
                    help="toy hidden dim H")
    ap.add_argument("--seed", type=int, default=1234,
                    help="base seed (master seed for toy ratchet chains)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="attack repetitions (decoder/init/draw seeds)")
    ap.add_argument("--solve-seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="independent pair-order/subsample solve repetitions")
    ap.add_argument("--quick", action="store_true",
                    help="shrink grids for a fast machinery check")
    return ap


def require_torch(attack_name):
    if torch is None:
        raise SystemExit(
            f"[{attack_name}] requires torch; install it or use a venv "
            f"(e.g. split-training/venv-cpu). --help/--self-test work "
            f"torch-less.")


def nn_mean_decode(h_rec, ref_h, ref_tok, victim_tok):
    """Trained-decoder surrogate for toy runs: per-token mean of the
    recovered reference rows, nearest mean per victim row, top-1 %.
    Real runs inject trained_inversion.evaluate_decoder instead."""
    vocab = int(max(ref_tok.max().item(), victim_tok.max().item())) + 1
    means = torch.stack([ref_h[ref_tok == v].mean(0)
                         if bool((ref_tok == v).any())
                         else torch.zeros(ref_h.shape[1])
                         for v in range(vocab)])
    d = torch.cdist(h_rec.float(), means.float())
    pred = d.argmin(1)
    return round(100.0 * (pred == victim_tok).float().mean().item(), 4)


def recovery_with_what_nn(h_star_wire, w_hat, ref_h, ref_tok, victim_tok):
    """recovery_with_what with the NN-mean decode: de-rotate and decode."""
    h_rec = (h_star_wire.double() @ w_hat.double().T).float()
    return nn_mean_decode(h_rec, ref_h, ref_tok, victim_tok)


def label_free_band(h_star, ref_h, ref_tok, victim_tok, seed):
    """Label-free band reference: decode rows rotated by a RANDOM W with no
    solve — the ceiling any failed accumulation attack sits in."""
    w = ratchet_secret(h_star.shape[1], seed, 999999)
    return recovery_with_what_nn(h_wire(h_star, w), torch.eye(
        h_star.shape[1], dtype=torch.float64), ref_h, ref_tok, victim_tok)


def journal_error(output, experiment, cell, err):
    """Per-cell error record instead of a crash (rotation_lifetime solve
    hardening): journal the failed cell with an error field and continue."""
    rec = {"experiment": experiment, "error": str(err), **cell}
    artifacts.append_jsonl(output, rec)
    print(f"[{experiment}] cell failed ({err}); recorded as error: {cell}")
    return rec
