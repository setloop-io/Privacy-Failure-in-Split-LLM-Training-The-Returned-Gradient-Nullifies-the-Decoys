#!/usr/bin/env python3
"""Paired, cluster-aware attacker advantage (experiments W2.1a / W2.1b).

Replaces the max-over-arms Bonferroni-Wilson gate, which has three defects the
repository has measured:

  1. It compares an arm's upper confidence bound against a control point estimate,
     so a constant predictor scores a positive "excess" made entirely of confidence
     width (v9.2 headline: +0.00244 pp point, +0.29232 pp reported).
  2. Degenerate arms that emit one or two token classes sit exactly at the control
     and pin the maximum, hiding movement in the arms that can actually read the
     representation (the pedestal).
  3. It treats token rows as independent, which the evaluation protocol forbids.

This tool instead scores each arm against the SAME control on the SAME rows, pairs
row by row, clusters by frame, and bootstraps over clusters. An arm identical to the
control reads exactly 0.0 with zero variance, by construction.

Input: a dtraining.attacker.latent_probe_predictions.v1 dump written by
`attacker --attack latent-probe --dump-eval-predictions`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEGENERATE_MODAL_SHARE = 0.99


def _load(path: Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def majority_class(tokens, known) -> int:
    """The constant predictor: the most frequent scoreable eval token.

    This is the STRONGEST constant predictor on the evaluation rows, which makes the
    test conservative for claiming a leak: an arm must beat the best possible
    constant baseline, not the weaker train-derived one the published gate used.
    The trade-off is that a genuinely weak leak could be masked; that limitation is
    stated rather than tuned away. The selected class matches the published gate's
    (token 279, ' the'), so the two differ in rate, not in identity.
    """
    import torch
    values = tokens[known]
    return int(torch.bincount(values).argmax()) if values.numel() else -1


def cluster_bootstrap(per_cluster: list[float], draws: int, seed: int
                      ) -> tuple[float, float]:
    if len(per_cluster) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(per_cluster)
    means = []
    for _ in range(draws):
        means.append(sum(per_cluster[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * draws)], means[min(int(0.975 * draws), draws - 1)]


def score_arm(prediction, tokens, known, control: int, draws: int, seed: int,
              classes=None) -> dict:
    """Paired advantage of one arm over the constant control, clustered by frame.

    `prediction` holds INDICES INTO `classes`, not raw token ids. Comparing it to
    `tokens` directly reads ~0.04% instead of ~3.2% and makes every arm look
    catastrophically worse than the control.
    """
    if classes is not None:
        prediction = classes[prediction]
    correct_arm = (prediction == tokens) & known
    correct_ctl = (tokens == control) & known

    modal_share = 0.0
    values = prediction[known]
    if values.numel():
        import torch
        modal_share = float(torch.bincount(values).max()) / float(values.numel())

    per_cluster = []
    for frame in range(tokens.shape[0]):
        rows = int(known[frame].sum())
        if rows == 0:
            continue
        delta = int(correct_arm[frame].sum()) - int(correct_ctl[frame].sum())
        per_cluster.append(100.0 * delta / rows)

    advantage = sum(per_cluster) / len(per_cluster) if per_cluster else float("nan")
    low, high = cluster_bootstrap(per_cluster, draws, seed)
    return {
        "paired_advantage_pp": round(advantage, 6),
        "ci95_low_pp": round(low, 6), "ci95_high_pp": round(high, 6),
        "clusters": len(per_cluster),
        "modal_class_share": round(modal_share, 6),
        "degenerate": modal_share > DEGENERATE_MODAL_SHARE,
        "resolves": bool(per_cluster) and low > 0.0,
    }


def analyse(dump, draws: int, seed: int) -> dict:
    tokens, known = dump["eval_tokens"], dump["known_eval"]
    control = majority_class(tokens, known)
    arms = []
    for arm in dump["arms"]:
        result = score_arm(arm["prediction"], tokens, known, control, draws,
                           seed, dump.get("classes"))
        result.update(model=arm["model"], restart=arm["restart"])
        arms.append(result)

    eligible = [a for a in arms if not a["degenerate"]]
    best = max(eligible, key=lambda a: a["paired_advantage_pp"], default=None)
    return {
        "schema": "dtraining.paired_advantage.v1",
        "bundle": dump.get("bundle"),
        "control_class": control,
        "clusters": arms[0]["clusters"] if arms else 0,
        "arms_total": len(arms),
        "arms_degenerate_excluded": sum(a["degenerate"] for a in arms),
        "best_eligible": best,
        "verdict": ("resolves" if best and best["resolves"] else "at-floor"),
        "arms": arms,
    }


def self_test() -> int:
    """W2.1a acceptance: an arm identical to the control reads exactly 0.0."""
    import torch
    frames, rows = 64, 80
    tokens = torch.randint(0, 40, (frames, rows))
    known = torch.ones(frames, rows, dtype=torch.bool)
    control = majority_class(tokens, known)

    constant = score_arm(torch.full((frames, rows), control), tokens, known,
                         control, 200, 1)
    leaky = score_arm(tokens.clone(), tokens, known, control, 200, 1)

    # Guard the class-index mapping: with a non-identity `classes` table, an arm
    # predicting the correct INDEX must still resolve. Comparing raw indices to
    # token ids silently reads ~0 and makes every arm look far worse than control.
    classes = torch.arange(40).flip(0)
    shuffled_tokens = classes[torch.randint(0, 40, (frames, rows))]
    idx_of = {int(v): i for i, v in enumerate(classes)}
    oracle_idx = torch.tensor([[idx_of[int(x)] for x in row] for row in shuffled_tokens])
    mapped = score_arm(oracle_idx, shuffled_tokens, known,
                       majority_class(shuffled_tokens, known), 200, 1, classes)

    checks = {
        "constant_arm_reads_exactly_zero": constant["paired_advantage_pp"] == 0.0,
        "constant_arm_zero_variance": constant["ci95_low_pp"] == 0.0
                                      and constant["ci95_high_pp"] == 0.0,
        "constant_arm_flagged_degenerate": constant["degenerate"],
        "constant_arm_does_not_resolve": not constant["resolves"],
        "oracle_arm_resolves": leaky["resolves"],
        "oracle_arm_not_degenerate": not leaky["degenerate"],
        "class_index_mapping_applied": mapped["resolves"],
    }
    for name, passed in checks.items():
        print(f"  {'ok  ' if passed else 'FAIL'} {name}")
    ok = all(checks.values())
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = analyse(_load(args.dump), args.draws, args.seed)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test() if "--self-test" in __import__("sys").argv
                     else main())
