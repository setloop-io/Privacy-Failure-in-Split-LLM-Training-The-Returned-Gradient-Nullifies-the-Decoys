#!/usr/bin/env python3
"""Known-plaintext and invariant probes for independently gauged latents.

The real bundle is created on trusted TLN and contains block-preserving
released views plus token labels for train/evaluation partitions. It is a
red-team evaluation artifact, not data that production UCN receives. The
output contains aggregate scores only.
"""

import argparse
import math
from statistics import NormalDist

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .. import artifacts
from .common import add_common_args, require_torch

EXPERIMENT_ID = "latent_probe"
MODES = ("training",)
REQUIRES_LABELS = True
DESCRIPTION = ("real-capture coordinate and rotation-invariant token probes "
               "for independently rotated/permuted latent blocks")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--bundle", help="trusted .pt bundle from the v5/v6 runner")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dump-eval-predictions",
                    help="diagnostic only: .pt path for the per-arm eval "
                         "argmax predictions. Scoring is untouched; nothing "
                         "is collected unless this is set.")
    ap.add_argument("--dump-eval-probabilities",
                    help="diagnostic only: .pt path for the per-arm eval "
                         "per-class probabilities (float16, one softmax over "
                         "the observed classes per row). W2.3's "
                         "token_cross_entropy needs the distribution, not the "
                         "argmax. Implies --dump-eval-predictions. Scoring is "
                         "untouched; nothing is collected unless this is set.")
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


def run(args):
    require_torch(EXPERIMENT_ID)
    if args.toy:
        raise SystemExit("latent-probe requires a real trusted bundle")
    if not args.bundle:
        raise SystemExit("--bundle is required")
    from torch import nn
    from privacy_runtime.latent_native import latent_invariants

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
    lookup = torch.full((int(max(classes.max(), eval_y.max())) + 1,), -1,
                        dtype=torch.long)
    lookup[classes] = torch.arange(classes.numel())
    train_classes = lookup[train_y]
    eval_classes = lookup[eval_y]
    known_eval = eval_classes >= 0
    d = train_x.shape[-1]
    invariant_dim = latent_invariants(train_x[:1]).shape[-1]

    class Probe(nn.Module):
        def __init__(self, model_kind):
            super().__init__()
            self.model_kind = model_kind
            if model_kind in ("invariant_only", "invariant_graph"):
                width = max(32, 2 * invariant_dim)
                self.trunk = nn.Sequential(
                    nn.LayerNorm(invariant_dim),
                    nn.Linear(invariant_dim, width), nn.GELU())
            else:
                width = max(64, 2 * d)
                self.trunk = nn.Sequential(
                    nn.LayerNorm(d + invariant_dim),
                    nn.Linear(d + invariant_dim, width), nn.GELU())
            if model_kind == "invariant_graph":
                self.graph_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.LayerNorm(4 * width),
                        nn.Linear(4 * width, 2 * width), nn.GELU(),
                        nn.Linear(2 * width, width))
                    for _ in range(3)
                ])
            self.head = nn.Linear(width, classes.numel())

        def forward(self, blocks):
            inv = latent_invariants(blocks)
            value = (inv if self.model_kind in
                     ("invariant_only", "invariant_graph") else
                     torch.cat((blocks, inv), -1))
            hidden = self.trunk(value)
            if self.model_kind == "invariant_graph":
                unit = blocks.float() / blocks.float().norm(
                    dim=-1, keepdim=True).clamp_min(1e-8)
                graph = (unit @ unit.transpose(-1, -2)).square()
                seq = blocks.shape[-2]
                eye = torch.eye(seq, device=blocks.device,
                                dtype=torch.bool).unsqueeze(0)
                graph = graph.masked_fill(eye, 0.0)
                normalized = graph / graph.sum(
                    -1, keepdim=True).clamp_min(1e-8)
                for layer in self.graph_layers:
                    one = normalized @ hidden
                    two = normalized @ one
                    maximum = (normalized.unsqueeze(-1)
                               * hidden.unsqueeze(-3)).amax(-2)
                    hidden = hidden + layer(torch.cat(
                        (hidden, one, two, maximum), dim=-1))
            return self.head(hidden)

    out = artifacts.make_artifact(
        "dtraining.attacker.latent_probe.v1",
        {"attack": EXPERIMENT_ID, "mode": args.mode,
         "bundle_schema": bundle.get("schema"), "latent_dim": d,
         "train_blocks": train_x.shape[0], "eval_blocks": eval_x.shape[0],
         "sequence_length": train_x.shape[1], "epochs": args.epochs,
         "batch_size": args.batch_size, "restarts": args.restarts,
         "models": ["coordinate_plus_invariants", "invariant_only",
                    "invariant_graph"]},
        "Known-plaintext evaluator with labels confined to trusted TLN. "
        "Each block preserves exactly the released UCN view, including "
        "fresh coordinate rotation and token permutation. The invariant-only "
        "arm uses norms, sorted Gram similarities and observed row position; "
        "the graph arm attacks the complete sign-invariant Gram structure.",
        interpretation="Production privacy fails if the Bonferroni-adjusted "
        "upper confidence recovery exceeds the exact matched majority control "
        "by more than the declared margin.")

    flat_train_y = train_classes.reshape(-1)
    total = eval_y.numel()
    all_scores = []
    dump_path = getattr(args, "dump_eval_predictions", None)
    prob_dump_path = getattr(args, "dump_eval_probabilities", None)
    if prob_dump_path and not dump_path:
        dump_path = prob_dump_path
    dumped_arms = []
    model_names = ("coordinate_plus_invariants", "invariant_only",
                   "invariant_graph")
    for model_index, model_name in enumerate(model_names):
        for restart in range(args.restarts):
            torch.manual_seed(args.seed + 1009 * restart
                              + 100000 * model_index)
            model = Probe(model_name)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            order_generator = torch.Generator().manual_seed(
                args.seed + 7001 * restart)
            flat_count = train_x.shape[0] * train_x.shape[1]
            for _ in range(args.epochs):
                if model_name == "invariant_graph":
                    block_order = torch.randperm(
                        train_x.shape[0], generator=order_generator)
                    block_batch = max(1, args.batch_size // train_x.shape[1])
                    for start in range(0, train_x.shape[0], block_batch):
                        idx = block_order[start:start + block_batch]
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(train_x[idx])
                        loss = torch.nn.functional.cross_entropy(
                            logits.flatten(0, 1),
                            train_classes[idx].flatten())
                        loss.backward()
                        optimizer.step()
                else:
                    order = torch.randperm(
                        flat_count, generator=order_generator)
                    features = (latent_invariants(train_x)
                                if model_name == "invariant_only" else
                                torch.cat((train_x,
                                           latent_invariants(train_x)), -1))
                    features = features.reshape(flat_count, -1)
                    for start in range(0, flat_count, args.batch_size):
                        idx = order[start:start + args.batch_size]
                        optimizer.zero_grad(set_to_none=True)
                        logits = model.head(model.trunk(features[idx]))
                        loss = torch.nn.functional.cross_entropy(
                            logits, flat_train_y[idx])
                        loss.backward()
                        optimizer.step()
            with torch.no_grad():
                eval_logits = model(eval_x)
                prediction = eval_logits.argmax(-1)
                correct = int(((prediction == eval_classes) & known_eval).sum())
            if dump_path:
                entry = {"model": model_name, "restart": restart,
                         "prediction": prediction.clone()}
                if prob_dump_path:
                    entry["probabilities"] = torch.softmax(
                        eval_logits.float(), dim=-1).half()
                    # Train-side confidence: membership_auc's "seen in training"
                    # population.
                    train_logits = model(train_x)
                    train_probs = torch.softmax(train_logits.float(), dim=-1)
                    flat_probs = train_probs.max(-1).values.reshape(-1)
                    entry["train_confidence"] = flat_probs.half()
                dumped_arms.append(entry)
            score = 100.0 * correct / total
            upper = 100.0 * wilson_upper(
                correct, total, 0.05,
                trials=len(model_names) * args.restarts)
            rec = {"experiment": EXPERIMENT_ID, "model": model_name,
                   "restart": restart, "correct": correct, "total": total,
                   "top1_pct": score, "bonferroni_upper95_pct": upper}
            artifacts.append_jsonl(args.output, rec)
            out["results"].append(rec)
            all_scores.append(rec)

    train_flat_tokens = train_y.reshape(-1)
    majority = int(torch.mode(train_flat_tokens).values)
    majority_correct = int((eval_y == majority).sum())
    majority_pct = 100.0 * majority_correct / total
    best = max(all_scores, key=lambda record: record["top1_pct"])
    worst_upper = max(record["bonferroni_upper95_pct"]
                      for record in all_scores)
    out["summary"].append({
        "experiment": EXPERIMENT_ID,
        "best_probe_top1_pct": best["top1_pct"],
        "best_probe_model": best["model"],
        "best_probe_restart": best["restart"],
        "worst_bonferroni_upper95_pct": worst_upper,
        "label_free_majority_pct": majority_pct,
        "upper95_excess_over_majority_pp": worst_upper - majority_pct,
        "known_eval_fraction": float(known_eval.float().mean()),
    })
    if dump_path:
        torch.save({"schema": "dtraining.attacker.latent_probe_predictions.v1",
                    "bundle": args.bundle, "classes": classes,
                    "eval_tokens": eval_y, "known_eval": known_eval,
                    "has_probabilities": bool(prob_dump_path),
                    "arms": dumped_arms}, dump_path)
    artifacts.write_artifact(args.output, out)
    return 0


def self_test():
    values = [wilson_upper(0, 100, 0.05),
              wilson_upper(50, 100, 0.05),
              wilson_upper(100, 100, 0.05)]
    ok = 0.0 < values[0] < values[1] < values[2] <= 1.0
    print(f"  [{'PASS' if ok else 'FAIL'}] Wilson upper bound monotonic")
    return 0 if ok else 1
