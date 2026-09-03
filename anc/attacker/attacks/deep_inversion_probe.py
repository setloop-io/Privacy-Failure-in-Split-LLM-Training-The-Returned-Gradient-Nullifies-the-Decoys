#!/usr/bin/env python3
"""Deep Transformer inversion probe (positive-control attack).

Behavior-faithful port of the original cluster probe (same architecture,
training loop, and Wilson scoring); the four committed result artifacts in
paper-data/collected/diagnostic/deep_probe/ were produced by that tree.
This port adds two things the original lacked:

  --shuffled-null   runs the identical protocol with globally permuted labels
                    (the matched null every positive control needs)
  final-epoch accuracy reported alongside best-over-epochs: best-over-epochs
                    is selection on the evaluation set, so the paper quotes both

The architecture is a sequence-to-sequence Transformer encoder over each
released latent frame (Linear -> 4x TransformerEncoderLayer, norm_first ->
LayerNorm -> Linear head over observed token classes).
"""
from __future__ import annotations

import argparse
import math
from statistics import NormalDist

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None

from .. import artifacts
from .common import add_common_args, require_torch

EXPERIMENT_ID = "deep_inversion_probe"
MODES = ("training",)
REQUIRES_LABELS = True
DESCRIPTION = ("Deep Transformer inversion probe; built to pass the naked "
               "positive control (reads undefended D=1024 latents back to tokens)")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--bundle", help="trusted .pt bundle from the v5/v6 runner")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim-feedforward", type=int, default=2048)
    ap.add_argument("--shuffled-null", action="store_true",
                    help="also run the identical protocol with globally "
                         "permuted labels (the matched null)")
    return ap


def wilson_upper(correct, total, alpha, trials=1):
    if total <= 0:
        return float("nan")
    z = NormalDist().inv_cdf(1.0 - alpha / max(1, trials))
    p = correct / total
    den = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt(p * (1.0 - p) / total
                           + z * z / (4.0 * total * total))
    return (center + radius) / den


def _run_condition(train_x, train_y, eval_x, eval_y, known_eval, classes,
                   args, seed_offset):
    """The artifact's protocol: best-over-epochs selection (stated), with
    final-epoch accuracy reported beside it."""
    lookup = torch.full((int(max(classes.max(), eval_y.max())) + 1,), -1,
                        dtype=torch.long)
    lookup[classes] = torch.arange(classes.numel())
    train_classes = lookup[train_y]
    eval_classes = lookup[eval_y]
    d = train_x.shape[-1]

    class DeepInversionProbe(nn.Module):
        def __init__(self, d_model, num_classes):
            super().__init__()
            self.input_proj = nn.Linear(d, d_model) if d != d_model else nn.Identity()
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=args.heads,
                dim_feedforward=args.dim_feedforward,
                batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer,
                                                     num_layers=args.layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)

        def forward(self, x):
            return self.head(self.norm(self.transformer(self.input_proj(x))))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    total = eval_y.numel()
    all_scores, final_scores = [], []
    for restart in range(args.restarts):
        torch.manual_seed(args.seed + 1009 * restart + seed_offset)
        d_model = min(1024, max(512, d))
        model = DeepInversionProbe(d_model, classes.numel()).to(device)
        tx, ty = train_x.to(device), train_classes.to(device)
        ex, ey = eval_x.to(device), eval_classes.to(device)
        known = known_eval.reshape(-1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        best_eval_acc = 0.0
        final_acc = 0.0
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(tx.shape[0])
            block_batch = max(1, args.batch_size)
            for start in range(0, tx.shape[0], block_batch):
                idx = order[start:start + block_batch]
                optimizer.zero_grad()
                logits = model(tx[idx]).reshape(-1, classes.numel())
                loss = criterion(logits, ty[idx].reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                preds = model(ex).reshape(-1, classes.numel()).argmax(dim=-1)
                correct = int((preds[known] == ey.reshape(-1)[known]).sum())
                final_acc = correct / total
                best_eval_acc = max(best_eval_acc, final_acc)
        all_scores.append(best_eval_acc)
        final_scores.append(final_acc)
    return {"restart_best_scores": all_scores,
            "restart_final_scores": final_scores,
            "best_eval_acc": max(all_scores),
            "final_epoch_acc": final_scores[int(max(range(len(all_scores)),
                key=lambda i: all_scores[i]))] if all_scores else 0.0}


def run(args):
    require_torch(EXPERIMENT_ID)
    if args.toy:
        raise SystemExit("deep-inversion-probe requires a real trusted bundle")
    if not args.bundle:
        raise SystemExit("--bundle is required")

    bundle = torch.load(args.bundle, map_location="cpu")
    train_x = bundle["train_wire"].float()
    train_y = bundle["train_tokens"].long()
    eval_x = bundle["eval_wire"].float()
    eval_y = bundle["eval_tokens"].long()
    if train_x.ndim != 3 or eval_x.ndim != 3:
        raise ValueError("bundle views must be [blocks,tokens,D]")
    if train_x.shape[1:] != eval_x.shape[1:]:
        raise ValueError("train/eval latent block shapes differ")

    classes = torch.unique(train_y).sort().values
    known_eval = torch.isin(eval_y, classes)
    d = train_x.shape[-1]

    out = artifacts.make_artifact(
        "dtraining.attacker.deep_inversion_probe.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode, "latent_dim": d,
         "train_blocks": train_x.shape[0], "eval_blocks": eval_x.shape[0],
         "sequence_length": train_x.shape[1], "epochs": args.epochs,
         "batch_size": args.batch_size, "restarts": args.restarts,
         "layers": args.layers, "heads": args.heads},
        "Deep sequence-to-sequence Transformer inversion probe, built to pass "
        "the naked positive control. best_eval_acc selects over epochs on the "
        "evaluation set (stated, not hidden); final_epoch_acc is reported "
        "beside it.",
        interpretation="Positive control: the probe must break the naked "
                       "boundary; the defended reading is meaningful only "
                       "against this demonstrated sensitivity.")

    flat = train_y.reshape(-1)
    majority = int(torch.mode(flat).values)
    total = eval_y.numel()
    majority_acc = int((eval_y == majority).sum()) / total if total else 0.0

    real = _run_condition(train_x, train_y, eval_x, eval_y, known_eval,
                          classes, args, 0)
    best = real["best_eval_acc"]
    wu = wilson_upper(int(best * total), total, 0.05, args.restarts)
    result = {"condition": "real", "majority_acc": majority_acc,
              "wilson_upper": wu, "excess_pp": (wu - majority_acc) * 100,
              **real}
    out["results"].append(result)
    artifacts.append_jsonl(args.output, result) if args.output else None

    if args.shuffled_null:
        generator = torch.Generator().manual_seed(args.seed + 7001)
        shuffled = train_y.reshape(-1)[torch.randperm(train_y.numel(),
                                    generator=generator)]
        train_y_null = shuffled[:train_y.numel()].reshape(train_y.shape)
        null = _run_condition(train_x, train_y_null, eval_x, eval_y,
                              known_eval, classes, args, 50000)
        nbest = null["best_eval_acc"]
        null["condition"] = "shuffled_null"
        null["excess_pp"] = (wilson_upper(int(nbest * total), total, 0.05,
                                          args.restarts) - majority_acc) * 100
        out["results"].append(null)
        if args.output:
            artifacts.append_jsonl(args.output, null)

    artifacts.write_artifact(args.output, out)
    return 0


def self_test() -> int:
    """Torch-guarded: a synthetic near-identity bundle must read above floor."""
    if torch is None:
        print("  [SKIP] torch absent")
        return 0
    torch.manual_seed(0)
    frames, rows, d, vocab = 24, 8, 32, 12
    tokens = torch.randint(0, vocab, (frames, rows))
    codebook = torch.randn(vocab, d)
    wire = codebook[tokens] + 0.01 * torch.randn(frames, rows, d)
    bundle = {"train_wire": wire[:16], "train_tokens": tokens[:16],
              "eval_wire": wire[16:], "eval_tokens": tokens[16:]}

    args = build_parser().parse_args(["--bundle", "fixture", "--epochs", "4",
                                      "--restarts", "1", "--layers", "1",
                                      "--heads", "2", "--dim-feedforward", "64"])
    classes = torch.unique(bundle["train_tokens"]).sort().values
    known = torch.isin(bundle["eval_tokens"], classes)
    result = _run_condition(bundle["train_wire"], bundle["train_tokens"],
                            bundle["eval_wire"], bundle["eval_tokens"], known,
                            classes, args, 0)
    reads = result["best_eval_acc"] > 0.5
    has_final = "final_epoch_acc" in result
    ok = reads and has_final
    print(f"  [{'PASS' if ok else 'FAIL'}] near-identity fixture reads above "
          f"50% (got {result['best_eval_acc']:.2f}) with final-epoch reported")
    return 0 if ok else 1
