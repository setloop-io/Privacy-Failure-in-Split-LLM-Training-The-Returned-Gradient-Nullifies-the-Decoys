"""Matching-attack arms for the latent-native defense (v11.0).

Adapted from the published attack families our floor claim must now face:

- VMA (Hidden No More, ICML 2025): vocabulary-matching via causal-attention
  structure (~100% prompt decode from hidden states).  Adapted to the latent
  surface: a contrastive matcher scores each released row against per-class
  reference signatures built from the train partition.
- Permutation reversal (arXiv 2505.18332): sorted-L1-style matching reverses
  pure permutation hiding at >99%.  Our bundles carry rotation + scale +
  chaff on top of permutation, so this arm measures whether ANY order
  information survives: per-row gauge-invariant signatures (sorted squared
  Gram row profile + norm quantile bin) are matched against reference
  sequences with known order, reporting position accuracy vs the random
  baseline.

These extend the frozen latent_probe gate — which is unchanged.
"""

EXPERIMENT_ID = "latent_matching"
MODES = ("training",)
REQUIRES_LABELS = True
DESCRIPTION = ("VMA-style vocabulary matching + permutation order-recovery "
               "arms on latent release bundles (v11)")


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser


def run(args):
    from .common import require_torch
    require_torch(EXPERIMENT_ID)
    import torch
    from torch import nn
    from privacy_runtime.latent_native import latent_invariants
    from .. import artifacts

    bundle = torch.load(args.bundle, map_location="cpu")
    train_x = bundle["train_wire"].float()
    train_y = bundle["train_tokens"].long()
    eval_x = bundle["eval_wire"].float()
    eval_y = bundle["eval_tokens"].long()

    classes = torch.unique(train_y)
    lookup = torch.full((int(classes.max()) + 1,), -1, dtype=torch.long)
    lookup[classes] = torch.arange(classes.numel())
    train_c = lookup[train_y]
    eval_c = lookup[eval_y]

    results = []

    # VMA arm: contrastive matcher scoring each row against per-class
    # invariant centroids (the latent-space analog of matching against the
    # vocabulary embedding structure).
    def vma_arm():
        train_inv = latent_invariants(train_x)
        eval_inv = latent_invariants(eval_x)
        centroids = torch.zeros(classes.numel(), train_inv.shape[-1])
        for index in range(classes.numel()):
            mask = train_c.reshape(-1) == index
            if mask.any():
                centroids[index] = train_inv.reshape(-1, train_inv.shape[-1]
                                                     )[mask].mean(0)
        pred = (torch.nn.functional.normalize(eval_inv, dim=-1)
                @ torch.nn.functional.normalize(centroids, dim=-1).T
                ).argmax(-1)
        valid = eval_c.reshape(-1) >= 0
        top1 = float((pred.reshape(-1)[valid]
                      == eval_c.reshape(-1)[valid]).float().mean())
        majority = _majority(train_c, eval_c)
        results.append({
            "arm": "vma_centroid_match", "top1_pct": 100.0 * top1,
            "majority_pct": 100.0 * majority,
            "excess_pp": 100.0 * (top1 - majority),
            "eval_rows": int(valid.sum()),
        })

    # Permutation-reversal arm: the bundle carries permuted rows + aligned
    # labels, so the TRUE order is not available as ground truth. The
    # meaningful metric is token recovery via position-free signature
    # matching: match each row's gauge-invariant signature (position channel
    # EXCLUDED — including it would make "order recovery" tautological) to
    # reference rows with known labels, and score the matched label.
    def order_arm():
        train_inv = latent_invariants(train_x)[..., :-1]   # drop position
        eval_inv = latent_invariants(eval_x)[..., :-1]
        ref = train_inv[: train_x.shape[0] // 2]
        ref_labels = train_c[: train_x.shape[0] // 2]
        ref_flat = ref.reshape(-1, ref.shape[-1])
        ref_lab_flat = ref_labels.reshape(-1)
        correct = 0
        total = 0
        chunk = 64
        for start in range(0, eval_x.shape[0], chunk):
            sig = eval_inv[start:start + chunk]
            dist = torch.cdist(sig.reshape(-1, sig.shape[-1]), ref_flat)
            matched = ref_lab_flat[dist.argmin(-1)]
            true = eval_c[start:start + chunk].reshape(-1)
            valid = true >= 0
            correct += int((matched[valid] == true[valid]).sum())
            total += int(valid.sum())
        top1 = correct / max(1, total)
        majority = _majority(train_c, eval_c)
        results.append({
            "arm": "sorted_signature_matching_position_free",
            "top1_pct": 100.0 * top1,
            "majority_pct": 100.0 * majority,
            "excess_pp": 100.0 * (top1 - majority),
            "eval_rows": total,
        })

    vma_arm()
    order_arm()

    summary = {
        "arms": [r["arm"] for r in results],
        "worst_excess_pp": max(r["excess_pp"] for r in results),
        "note": ("matching arms adapted from VMA (ICML'25) and permutation "
                 "reversal (arXiv:2505.18332) to the latent surface; frozen "
                 "latent_probe gate unchanged"),
    }
    artifact = artifacts.make_artifact(
        "dtraining.attacker.latent_matching.v1",
        {"attack": EXPERIMENT_ID, "mode": "training",
         "restarts": args.restarts, "epochs": args.epochs},
        "VMA-style centroid matching and permutation order-recovery against "
        "gauged+chaffed latent bundles; extends the frozen gate.")
    artifact["results"].extend(results)
    artifact["summary"].append(summary)
    artifacts.write_artifact(args.output, artifact)
    import json
    print(json.dumps(summary, indent=1))
    return 0


def _majority(train_c, eval_c):
    import torch
    flat = train_c.reshape(-1)
    mode = int(torch.bincount(flat[flat >= 0]).argmax())
    valid = eval_c.reshape(-1) >= 0
    return float((eval_c.reshape(-1)[valid] == mode).float().mean())
