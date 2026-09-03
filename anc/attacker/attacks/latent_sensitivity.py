"""Compromise-fraction battery for the latent-native defense (v9 analysis).

Answers "how much compromise before privacy breaks" on a trusted release
bundle, WITHOUT touching the frozen latent_probe gate.  Four arms:

  capture       attacker captures only a fraction p of frames
  chaff-id      attacker heuristically discards the x% least-connected rows
                (mean |off-diag Gram| to frame-mates) as chaff
  gauge         attacker holds canonical (pre-gauge) rows for a fraction of
                blocks (requires --bundle-canonical-fraction on the runner;
                simulates a partial TLN side channel)
  known-pt      attacker knows the true labels of a fraction f of eval rows

Each cell trains the strongest latent_probe arm (invariant_only MLP) and
reports top-1 vs the majority control of the SAME eval subset.
"""

EXPERIMENT_ID = "latent_sensitivity"
MODES = ("training",)
REQUIRES_LABELS = True
DESCRIPTION = ("compromise-fraction battery: capture/chaff-id/gauge/known-pt "
               "sweeps on a latent release bundle")


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser


def _train_probe(train_x, train_y, eval_x, eval_y, classes, epochs, restarts,
                 batch_size, lr):
    """Train the invariant_only probe on whole frames; return best top-1.

    Rows with label < 0 (padding from the chaff-id arm) are masked out of
    the training loss; eval scoring covers rows with valid labels only, and
    cell() computes the majority control on the same kept subset."""
    import torch
    from torch import nn
    from privacy_runtime.latent_native import latent_invariants

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inv_dim = latent_invariants(train_x[:1]).shape[-1]

    def build():
        width = max(32, 2 * inv_dim)
        return nn.Sequential(
            nn.LayerNorm(inv_dim), nn.Linear(inv_dim, width), nn.GELU(),
            nn.Linear(width, classes)).to(device)

    valid_frames = (train_y >= 0).any(dim=1)
    train_x = train_x[valid_frames]
    train_y = train_y[valid_frames]
    # Any label below -1 (e.g. the -10 unknown-class marker from the
    # known-plaintext arm) is excluded from the CE loss like padding.
    train_y = torch.where(train_y < -1, torch.full_like(train_y, -1),
                          train_y)
    best = 0.0
    for restart in range(restarts):
        torch.manual_seed(1000 + restart)
        probe = build()
        opt = torch.optim.AdamW(probe.parameters(), lr=lr)
        for _ in range(epochs):
            perm = torch.randperm(train_x.shape[0])
            for start in range(0, train_x.shape[0], batch_size):
                idx = perm[start:start + batch_size]
                bx = train_x[idx].to(device)
                by = train_y[idx].to(device)
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(
                    probe(latent_invariants(bx)).flatten(0, 1), by.flatten(),
                    ignore_index=-1)
                loss.backward()
                opt.step()
        with torch.no_grad():
            pred = probe(latent_invariants(eval_x.to(device))).argmax(-1)
            pred = pred.cpu()
            valid = eval_y >= 0
            correct = int((pred[valid] == eval_y[valid]).sum())
            total = int(valid.sum())
        best = max(best, correct / max(1, total))
    return best


def _majority(train_y, eval_y):
    flat = train_y.reshape(-1)
    mode = int(torch_mode(flat))
    valid = eval_y.reshape(-1) >= 0
    kept = eval_y.reshape(-1)[valid]
    return float((kept == mode).float().mean())


def torch_mode(values):
    import torch
    return torch.bincount(values.reshape(-1)[values.reshape(-1) >= 0]).argmax()


def run(args):
    from .common import require_torch
    require_torch(EXPERIMENT_ID)
    import torch
    from .. import artifacts

    bundle = torch.load(args.bundle, map_location="cpu")
    train_x = bundle["train_wire"].float()
    train_y = bundle["train_tokens"].long()
    eval_x = bundle["eval_wire"].float()
    eval_y = bundle["eval_tokens"].long()
    classes = int(train_y.max().item()) + 1
    lookup = torch.full((classes,), -1, dtype=torch.long)
    lookup[torch.unique(train_y)] = torch.arange(
        len(torch.unique(train_y)))
    train_c = lookup[train_y]
    eval_c = lookup[eval_y]
    n_classes = len(torch.unique(train_y))

    # Filter eval to known-class rows: unknown-class tokens are remapped to
    # a guaranteed-wrong class (the latent_probe convention counts unknowns
    # as misses via the full denominator).
    eval_c = torch.where(eval_c < 0, torch.full_like(eval_c, -10), eval_c)

    results = []
    rng = torch.Generator().manual_seed(2024)

    def cell(arm, fraction, tx, ty, ex, ey):
        if tx.shape[0] < 2 or ey.numel() == 0:
            return
        majority = _majority(ty, ey)
        top1 = _train_probe(tx, ty, ex, ey, n_classes, args.epochs,
                            args.restarts, args.batch_size, args.lr)
        results.append({
            "arm": arm, "fraction": fraction,
            "top1_pct": 100.0 * top1,
            "majority_pct": 100.0 * majority,
            "excess_pp": 100.0 * (top1 - majority),
            "train_rows": int(ty.numel()), "eval_rows": int(ey.numel()),
        })

    # arm: capture fraction (attacker captures p% of train and eval blocks)
    for p in (0.1, 0.25, 0.5, 1.0):
        k_train = max(2, int(round(train_x.shape[0] * p)))
        k_eval = max(2, int(round(eval_x.shape[0] * p)))
        idx_t = torch.randperm(train_x.shape[0], generator=rng)[:k_train]
        idx_e = torch.randperm(eval_x.shape[0], generator=rng)[:k_eval]
        cell("capture", p, train_x[idx_t], train_c[idx_t],
             eval_x[idx_e], eval_c[idx_e])

    # arm: chaff identification (drop the x% least Gram-connected rows)
    def gram_connectivity(frames):
        unit = frames / frames.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        gram = (unit @ unit.transpose(-1, -2)).abs()
        seq = gram.shape[-1]
        eye = torch.eye(seq, dtype=torch.bool).unsqueeze(0)
        return gram.masked_fill(eye, 0.0).sum(-1) / max(1, seq - 1)

    conn_t = gram_connectivity(train_x)
    conn_e = gram_connectivity(eval_x)
    for x in (0.0, 0.25, 0.5, 0.9):
        def drop_lowest(fx, fy, conn):
            flat_conn = conn.reshape(-1)
            threshold = torch.quantile(flat_conn, x) if x > 0 else -1.0
            keep = (flat_conn > threshold).reshape(fy.shape)
            kept = [fx[b][keep[b]] for b in range(fx.shape[0])]
            kept_y = [fy[b][keep[b]] for b in range(fy.shape[0])]
            kept = [r for r in kept if r.shape[0] > 0]
            kept_y = [r for r in kept_y if r.shape[0] > 0]
            rows = max(r.shape[0] for r in kept)
            # pad rows back to a rectangular tensor with -1 labels
            import torch as _t
            out = _t.full((len(kept), rows, fx.shape[-1]), 0.0)
            out_y = _t.full((len(kept), rows), -1, dtype=_t.long)
            for i, (r, ry) in enumerate(zip(kept, kept_y)):
                out[i, :r.shape[0]] = r
                out_y[i, :r.shape[0]] = ry
            return out, out_y

        tx, ty = drop_lowest(train_x, train_c, conn_t)
        ex, ey = drop_lowest(eval_x, eval_c, conn_e)
        cell("chaff_id", x, tx, ty, ex, ey)

    # arm: gauge compromise (canonical rows for a fraction of blocks)
    if "canonical_train_wire" in bundle:
        canon_tx = bundle["canonical_train_wire"].float()
        canon_ty = lookup[bundle["canonical_train_tokens"].long()]
        canon_ex = bundle["canonical_eval_wire"].float()
        canon_ey = lookup[bundle["canonical_eval_tokens"].long()]
        canon_ey = torch.where(canon_ey < 0, torch.full_like(canon_ey, -10),
                               canon_ey)
        fraction = canon_ex.shape[0] / max(1, eval_x.shape[0])
        # compromised requests: attacker trains on canonical rows
        cell("gauge_canonical_views", fraction, canon_tx, canon_ty,
             canon_ex, canon_ey)
        # remaining (gauged) eval rows, attacker trained on canonical train
        cell("gauge_residual_gauged_eval", fraction, canon_tx, canon_ty,
             eval_x, eval_c)

    # arm: known-plaintext fraction (f% of eval rows join the train set)
    for f in (0.01, 0.05, 0.2):
        n_known = max(1, int(round(eval_x.shape[0] * f)))
        idx = torch.randperm(eval_x.shape[0], generator=rng)
        known, held = idx[:n_known], idx[n_known:]
        cell("known_pt", f,
             torch.cat([train_x, eval_x[known]]),
             torch.cat([train_c, eval_c[known]]),
             eval_x[held], eval_c[held])

    summary = {
        "arms": sorted(set(r["arm"] for r in results)),
        "worst_excess_pp": max((r["excess_pp"] for r in results), default=None),
    }
    artifact = artifacts.make_artifact(
        "dtraining.attacker.latent_sensitivity.v1",
        {"attack": EXPERIMENT_ID, "mode": "training",
         "epochs": args.epochs, "restarts": args.restarts},
        "Compromise-fraction sensitivity battery on a trusted latent release "
        "bundle; extends, does not replace, the frozen latent_probe gate.")
    artifact["results"].extend(results)
    artifact["summary"].append(summary)
    artifacts.write_artifact(args.output, artifact)
    print(json.dumps(summary, indent=1))
    return 0


import json  # noqa: E402
