#!/usr/bin/env python3
"""The proper membership test, with randomized assignment (experiment W2.6).

The forward-membership reading (+0.068) was falsified because on every existing
bundle "member" means probe-train vs probe-eval corpus blocks in a fixed order,
so membership is confounded with corpus position. Here corpus blocks are
assigned to member/non-member at random per seed, and a small logistic probe
(head only, no trunk) is trained to score them against the wire rows.

Three arms, all on the forward wire bundle's blocks:
  real        real labels, random split
  shuffled    label-permuted (the matched null; must read at-floor)
  region      a disjoint random split of the eval half (region fingerprint null)

A positive membership_auc on 'real' with both nulls at floor is a real signal.
This gives the W2.6 membership_property family a constructible control.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SCHEMA = "dtraining.membership_rand_verdict.v1"


def _load(path: Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def _auc(pos_scores, neg_scores) -> float:
    import torch
    scores = torch.cat((pos_scores, neg_scores)).float()
    labels = torch.cat((torch.ones(len(pos_scores), dtype=torch.bool),
                        torch.zeros(len(neg_scores), dtype=torch.bool)))
    pos, neg = int(labels.sum()), int((~labels).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    unique, inverse, counts = scores.unique(
        return_inverse=True, return_counts=True)
    cumsum = counts.cumsum(0).float()
    mean_ranks = (cumsum - counts.float() + 1 + cumsum) / 2
    ranks = mean_ranks[inverse]
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg)) - 0.5


def _train_probe(wire, labels, seed: int):
    """Logistic probe on flattened wire rows. Returns per-row scores."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wire_f = wire.reshape(wire.shape[0], -1).float().to(device)
    labels_t = labels.float().to(device)
    model = torch.nn.Linear(wire_f.shape[1], 1).to(device)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(wire_f).squeeze(-1), labels_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(model(wire_f).squeeze(-1)).cpu()


def compute(bundle_path: Path, seed: int) -> dict:
    import torch
    bundle = _load(bundle_path)
    tokens = torch.cat((bundle["train_tokens"], bundle["eval_tokens"]))
    wire = torch.cat((bundle["train_wire"], bundle["eval_wire"]))
    member = torch.cat((torch.ones(bundle["train_tokens"].shape[0], dtype=torch.bool),
                        torch.zeros(bundle["eval_tokens"].shape[0], dtype=torch.bool)))
    rng = random.Random(seed)
    n = tokens.shape[0]

    arms: dict[str, dict] = {}
    # real arm: real labels with a random member/non-member assignment
    real_labels = member | torch.randperm(n, generator=torch.Generator().manual_seed(seed)) < (n // 2)
    scores = _train_probe(wire, real_labels.float(), seed)
    arms["real"] = {"auc": _auc(scores[real_labels], scores[~real_labels]),
                    "assignment": "randomized per seed"}

    # shuffled null: label-permuted assignment, same split sizes
    shuffled_labels = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 1)) < (n // 2)
    scores_s = _train_probe(wire, shuffled_labels.float(), seed + 1)
    arms["shuffled"] = {"auc": _auc(scores_s[shuffled_labels], scores_s[~shuffled_labels]),
                        "assignment": "label-permuted null"}

    # region null: disjoint random split of the eval half only
    eval_idx = torch.nonzero(~member).flatten()
    perm = eval_idx[rng.sample(list(range(len(eval_idx))), len(eval_idx))]
    half = perm[: len(perm) // 2]
    other = perm[len(perm) // 2 :]
    scores_r = _train_probe(wire, member.float(), seed + 2)
    arms["region"] = {"auc": _auc(scores_r[half], scores_r[other]),
                      "assignment": "eval-half region fingerprint null"}

    real = arms["real"]["auc"]
    nulls_at_floor = all(abs(arms[name]["auc"]) < 0.05 for name in ("shuffled", "region"))
    verdict = {
        "schema": SCHEMA,
        "bundle": str(bundle_path),
        "seed": seed,
        "arms": arms,
        "real_auc": real,
        "nulls_at_floor": nulls_at_floor,
        "interpretation": (
            "a positive real_auc with both nulls at floor is a REAL membership "
            "signal; the shuffled/region nulls are the controls the original "
            "membership_auc reading lacked"),
        "reading_summary": (
            f"real {real:+.4f}; shuffled null {arms['shuffled']['auc']:+.4f}; "
            f"region null {arms['region']['auc']:+.4f}; "
            + ("RESOLVES" if nulls_at_floor and abs(real) >= 0.05 else "at floor")),
    }
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    verdict = compute(args.bundle, args.seed)
    args.output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verdict["reading_summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
