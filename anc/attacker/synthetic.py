#!/usr/bin/env python3
"""Synthetic toy scenario generator (torch-only).

Builds a self-contained rotated-boundary world for --toy runs of the
attacks: hidden-dim H, per-epoch orthogonal W_t from the ratchet chain,
"public" activation rows (attacker's labeled reference), "victim" rows,
and optionally a wire-capture directory in the repo's sidecar schema so
the capture-loading path is exercised end to end.

Nothing here claims realism — it is the machinery check (--toy pattern
from rotation_lifetime.py) that proves artifact + journal + solve wiring.
"""

import json
import os

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .solve_primitives import ratchet_secret, h_wire


def make_toy_world(hidden=64, n_public=2048, n_victim=256, n_epochs=4,
                   master_seed=12345, seed=0, structured=True):
    """Returns dict with:
      public_h  [n_public, H]  attacker-side canonical activations
      public_tok [n_public]    their token labels (from a small vocab)
      victim_h  [n_victim, H], victim_tok
      Ws        [n_epochs, H, H] per-epoch secrets
      wire(t)   helper: rows under epoch t (fp64, like the defense seam)
    `structured=True` gives rows a low-dim + clustered structure (tokens
    live near per-token mean directions) so nearest-neighbor decodes and
    subspace attacks have signal; False = iid gaussian rows."""
    g = torch.Generator().manual_seed(seed)
    vocab = 32
    tok_dirs = torch.randn(vocab, hidden, generator=g)
    tok_dirs = tok_dirs / tok_dirs.norm(dim=1, keepdim=True)

    def rows(n, off):
        if structured:
            tok = torch.randint(0, vocab, (n,), generator=g)
            h = 4.0 * tok_dirs[tok] + torch.randn(n, hidden, generator=g)
        else:
            tok = torch.randint(0, vocab, (n,), generator=g)
            h = torch.randn(n, hidden, generator=g)
        return h.float(), tok

    public_h, public_tok = rows(n_public, 0)
    victim_h, victim_tok = rows(n_victim, 1)
    Ws = torch.stack([ratchet_secret(hidden, master_seed, t)
                      for t in range(n_epochs)])

    def wire(h, t):
        return h_wire(h, Ws[t])

    return {"hidden": hidden, "vocab": vocab, "n_epochs": n_epochs,
            "master_seed": master_seed, "seed": seed,
            "public_h": public_h, "public_tok": public_tok,
            "victim_h": victim_h, "victim_tok": victim_tok,
            "Ws": Ws, "wire": wire, "tok_dirs": tok_dirs}


def write_toy_captures(world, capture_dir, mode, rows_per_epoch=128,
                       phase=None):
    """Write wire_NNNN.{pt,json} captures of the public rows under their
    epoch keys, in the mode's sidecar schema. Returns the record list.
    In training mode writes BOTH fwd (h_in) and bwd (g_out) phases; the
    backward rows are the same canonical rows rotated (g_canonical @ W_t)
    as in er_train_eval's replay convention."""
    if phase is None:
        phases = ("fwd", "bwd") if mode == "training" else ("decode",)
    else:
        phases = (phase,)
    os.makedirs(capture_dir, exist_ok=True)
    n_per = min(rows_per_epoch, world["public_h"].shape[0]
                // world["n_epochs"])
    idx = 0
    records = []
    for t in range(world["n_epochs"]):
        h = world["public_h"][t * n_per:(t + 1) * n_per]
        hw = world["wire"](h, t).float()
        for ph in phases:
            pt = os.path.join(capture_dir, f"wire_{idx:04d}.pt")
            torch.save(hw, pt)
            if mode == "training":
                meta = {"session_id": "toy", "mb_id": t, "phase": ph,
                        "step": t, "epoch": t}
            else:
                meta = {"session_id": "toy", "request_seq": t,
                        "phase": ph, "position": t, "epoch": t}
            with open(pt[:-3] + ".json", "w") as f:
                json.dump(meta, f)
            records.append((meta, pt))
            idx += 1
    return records
