#!/usr/bin/env python3
"""Leakage metrics beyond token_top1, emitted in the summarizer's schema (experiment W2.3).

`evaluation_protocol.json` declares eight metrics; on its own `latent_probe.py` produces
`top1_pct` and `bonferroni_upper95_pct`, not records keyed
`metric: "token_top1"`, and nothing produced the `{arm, attack, metric, seed, cluster_id,
value}` records `summarize_complete_view_matrix.py` requires.

This module computes what a prediction dump can support and emits it in that schema. It
does **not** implement all eight, and it does not pretend to: `describe_gaps()` states
what each remaining metric needs and why the dump cannot supply it. A metric that cannot
be computed is reported as unavailable rather than approximated.

Computable from a `latent_probe_predictions.v1` dump:

  token_top1        per-cluster top-1 recovery, paired against the constant control
  rare_token_top1   the same, restricted to tokens below a declared frequency percentile
  token_cross_entropy  per-cluster cross-entropy advantage over the constant control,
                       when the dump carries per-row probabilities
                       (latent_probe --dump-eval-probabilities; dumps without them
                       report it unavailable, never approximated from the argmax)
  membership_auc    per-cluster AUC of the probe's confidence scoring train blocks
                       (seen in probe training) above eval blocks (held out) -- the
                       shadow-model membership test, with the constant control's AUC
                       as the paired baseline
  semantic_cosine   per-cluster cosine proximity between predicted and true token
                       embeddings in the PUBLIC base-model embedding space (the
                       adversary holds the public weights), minus the constant
                       control's cosine. Requires --embeddings; the named measure for
                       the semantic_recovery family.

Not computable from it, with the reason:

  sequence_exact_match  released rows carry a fresh per-request permutation shared
                       by both directions, so order never crosses the boundary; the
                       only order-carrying channel would be timing, declared unmeasured
  semantic_recovery     without --embeddings: needs the public base-model embedding table
  property_auc          needs property labels
  response_side_recovery needs response-side capture, which the protocol does not record
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SCHEMA = "dtraining.leakage_metrics.v1"
RARE_PERCENTILE = 50.0


def _load(path: Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_embeddings(path):
    """A [vocab, H] tensor from a .pt file (bare tensor or {'embeddings': ...})."""
    import torch
    if path is None:
        return None
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        value = value.get("embeddings")
    if not torch.is_tensor(value) or value.ndim != 2:
        raise ValueError("--embeddings must resolve to a [vocab, H] tensor")
    return value


def describe_gaps(has_probabilities: bool = False,
                  has_embeddings: bool = False) -> dict:
    gaps = {
        "sequence_exact_match": "Released rows carry a fresh per-request permutation "
                                "shared by both directions, so sequence order never "
                                "crosses the boundary -- even the joint and cross-Gram "
                                "views carry no order. The only order-carrying channel "
                                "would be timing, which is declared unmeasured.",
        "property_auc": "Needs property labels per row.",
        "response_side_recovery": "Needs response-side capture. The transcript protocol "
                                  "records the forward and backward wire, not the response.",
    }
    if not has_embeddings:
        gaps["semantic_recovery"] = (
            "Needs the public base-model embedding table (adversary-known). "
            "Pass --embeddings; the metric is semantic_cosine, arm minus the "
            "constant control's cosine.")
    if not has_probabilities:
        gaps = {"token_cross_entropy": "This dump carries argmax predictions only. "
                                       "Re-score the arm with latent_probe "
                                       "--dump-eval-probabilities.",
                "membership_auc": "Needs the same per-row probabilities (its confidence "
                                  "score is the argmax probability).",
                **gaps}
    return gaps


def rare_token_mask(tokens, known, percentile: float):
    """Rows whose true token is rarer than the given percentile of the observed
    frequency distribution. Frequencies come from the evaluation rows themselves, so
    'rare' is relative to this corpus slice and must be reported as such."""
    import torch
    counts = Counter(int(value) for value in tokens[known])
    if not counts:
        return torch.zeros_like(known)
    ordered = sorted(counts.values())
    cutoff = ordered[min(int(len(ordered) * percentile / 100.0), len(ordered) - 1)]
    rare = {token for token, count in counts.items() if count <= cutoff}
    flags = torch.zeros_like(known)
    for token in rare:
        flags |= tokens == token
    return flags & known


def per_cluster(prediction, tokens, mask, control: int) -> dict[str, float]:
    """Paired advantage over the constant control, one value per frame."""
    values: dict[str, float] = {}
    for frame in range(tokens.shape[0]):
        rows = int(mask[frame].sum())
        if rows == 0:
            continue
        arm = int(((prediction[frame] == tokens[frame]) & mask[frame]).sum())
        ctl = int(((tokens[frame] == control) & mask[frame]).sum())
        values[f"frame-{frame:05d}"] = 100.0 * (arm - ctl) / rows
    return values


def raw_recovery(prediction, tokens, mask) -> float:
    """Absolute top-1 recovery on a row subset, in percent.

    Needed because on a rare-token subset the constant control scores exactly zero by
    construction -- the control class is by definition frequent. Advantage over a
    trivially-zero control collapses to the raw rate and reads as "+0.0000", which looks
    like "no effect" when it in fact means "nothing recovered at all". Both are reported
    so the distinction is visible.
    """
    rows = int(mask.sum())
    if rows == 0:
        return float("nan")
    return 100.0 * int(((prediction == tokens) & mask).sum()) / rows


def semantic_cosine_per_cluster(prediction, tokens, mask, control_id: int,
                                embeddings) -> dict[str, float]:
    """Per-cluster semantic proximity of predictions to truth, minus the control.

    The measure: mean cosine similarity between the predicted token's embedding
    and the true token's embedding in the PUBLIC base model's embedding space
    (the adversary holds the public base weights, so this table is part of the
    adversary view), minus the same quantity for the constant control. A token
    recovery that lands on a synonym scores high even when top-1 is exact-zero;
    this is the harm-oriented granularity token_top1 cannot express. Positive
    advantage = the arm reads semantic content above the control. token ids are
    raw vocabulary ids here (predictions are already mapped through `classes`).
    """
    import torch
    unit = embeddings.float()
    unit = unit / unit.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    true_vec = unit[tokens]
    control_cos = (unit[control_id].unsqueeze(0) * true_vec).sum(-1)
    values: dict[str, float] = {}
    for frame in range(tokens.shape[0]):
        rows = int(mask[frame].sum())
        if rows == 0:
            continue
        pred_cos = (unit[prediction[frame]] * true_vec[frame]).sum(-1)
        advantage = (pred_cos - control_cos[frame])[mask[frame]]
        values[f"frame-{frame:05d}"] = float(advantage.mean())
    return values


def cross_entropy_per_cluster(probabilities, token_class_idx, known, control: int,
                              n_classes: int) -> dict[str, float]:
    """Per-cluster cross-entropy advantage over the constant control, in nats.

    token_class_idx: the true token's index into the dump's `classes` axis
    (NOT the raw token id -- probabilities are over observed classes, the same
    indexing discipline as predictions). The arm's value is the mean negative
    log-probability the probe assigns to the true class; the constant
    control's is -log(freq(control)) on every row. Advantage is control - arm,
    so a positive value means the probe reads the wire better than the
    constant control. Lower arm entropy = more leak.
    """
    import torch
    control_logp = -float(torch.log(
        (token_class_idx[known] == control).float().mean().clamp_min(1e-12)))
    values: dict[str, float] = {}
    for frame in range(token_class_idx.shape[0]):
        rows = int(known[frame].sum())
        if rows == 0:
            continue
        true_p = probabilities[frame][known[frame],
                                      token_class_idx[frame][known[frame]]]
        arm = float(-torch.log(true_p.float().clamp_min(1e-12)).mean())
        values[f"frame-{frame:05d}"] = control_logp - arm
    return values


def membership_auc_per_cluster(train_confidence, eval_confidence,
                               n_frames: int, rows_per_frame: int) -> dict[str, float]:
    """AUC of the probe's confidence scoring train blocks above eval blocks.

    confidence: per-row scalar score (the probe's probability for its argmax).
    One AUC per eval frame, that frame's held-out rows against all trained
    rows; the paired baseline is the constant control's AUC (exactly 0.5,
    since a label-free score carries no membership information). Ties get
    average ranks.
    """
    import torch
    values: dict[str, float] = {}
    for frame in range(n_frames):
        held = eval_confidence[frame * rows_per_frame:(frame + 1) * rows_per_frame]
        scores = torch.cat((train_confidence, held))
        labels = torch.cat((torch.ones(train_confidence.shape[0], dtype=torch.bool),
                            torch.zeros(held.shape[0], dtype=torch.bool)))
        pos, neg = int(labels.sum()), int((~labels).sum())
        if pos == 0 or neg == 0:
            continue
        unique, inverse, counts = scores.unique(
            return_inverse=True, return_counts=True)
        cumsum = counts.cumsum(0).float()
        mean_ranks = (cumsum - counts.float() + 1 + cumsum) / 2
        ranks = mean_ranks[inverse]
        auc = float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))
        values[f"frame-{frame:05d}"] = auc - 0.5
    return values


def compute(dump, arm: str, attack: str, seed: int, embeddings=None) -> dict:
    import torch
    tokens, known, classes = dump["eval_tokens"], dump["known_eval"], dump.get("classes")
    control = int(torch.bincount(tokens[known]).argmax())
    # Class-index discipline: probabilities are over the dump's `classes` axis,
    # so every probability lookup uses the token's class index, never the raw
    # token id. `control` is likewise a class index here.
    if classes is not None:
        lookup = torch.full((int(max(classes.max(), tokens.max())) + 1,), -1,
                            dtype=torch.long)
        lookup[classes] = torch.arange(classes.numel())
        token_class_idx = lookup[tokens]
    else:
        token_class_idx = tokens.clone()

    records, per_metric, raw, control_rate = [], {}, {}, {}
    rare = rare_token_mask(tokens, known, RARE_PERCENTILE)
    n_classes = int(classes.numel()) if classes is not None else int(tokens.max()) + 1
    has_probs = any(entry.get("probabilities") is not None for entry in dump["arms"])
    unavailable = describe_gaps(has_probs, embeddings is not None)

    for entry in dump["arms"]:
        prediction = entry["prediction"]
        if classes is not None:
            prediction = classes[prediction]
        label = f"{entry['model']}#{entry['restart']}"
        metrics = [("token_top1", known), ("rare_token_top1", rare)]
        if embeddings is not None:
            metrics.append(("semantic_cosine", known))

        probabilities = entry.get("probabilities")
        ce_values = None
        if probabilities is not None:
            ce_values = cross_entropy_per_cluster(
                probabilities.float(), token_class_idx, known, control, n_classes)
            metrics.append(("token_cross_entropy", known))

        auc_values = None
        train_confidence = entry.get("train_confidence")
        if probabilities is not None and train_confidence is not None:
            eval_confidence = probabilities.float().max(-1).values.reshape(-1)
            flat_train = train_confidence.float().reshape(-1)
            auc_values = membership_auc_per_cluster(
                flat_train, eval_confidence, tokens.shape[0], tokens.shape[1])
            metrics.append(("membership_auc", known))

        for metric, mask in metrics:
            if metric == "token_cross_entropy":
                values = ce_values or {}
            elif metric == "membership_auc":
                values = auc_values or {}
            elif metric == "semantic_cosine":
                values = semantic_cosine_per_cluster(prediction, tokens, mask,
                                                     control, embeddings)
            else:
                values = per_cluster(prediction, tokens, mask, control)
            if not values:
                continue
            per_metric.setdefault(metric, {})[label] = (
                sum(values.values()) / len(values))
            if metric in ("token_top1", "rare_token_top1"):
                raw.setdefault(metric, {})[label] = raw_recovery(
                    prediction, tokens, mask)
            if metric == "rare_token_top1":
                control_rate.setdefault(metric, 100.0 * int(
                    ((tokens == control) & mask).sum()) / max(int(mask.sum()), 1))
            for cluster, value in values.items():
                records.append({"arm": arm, "attack": attack, "metric": metric,
                                "seed": seed, "cluster_id": cluster,
                                "value": value, "probe": label})

    return {
        "schema": SCHEMA,
        "bundle": dump.get("bundle"),
        "control_class": control,
        "rare_token_percentile": RARE_PERCENTILE,
        "rare_rows": int(rare.sum()), "scoreable_rows": int(known.sum()),
        "metrics_computed": sorted(per_metric),
        "metrics_unavailable": unavailable,
        "mean_advantage_by_probe": per_metric,
        "raw_recovery_pct_by_probe": raw,
        "control_recovery_pct": control_rate,
        "degenerate_control_subsets": [m for m, r in control_rate.items() if r == 0.0],
        "reading_note": ("On a subset where control_recovery_pct is 0, the paired "
                         "advantage equals the raw recovery rate. A +0.0000 advantage "
                         "there means nothing was recovered, not that the arm matched "
                         "the control. membership_auc values are AUC - 0.5, so 0.0 is "
                         "the constant-control floor."),
        "records": records,
    }


def _membership_auc_over_frames(train_confidence, eval_confidence,
                                n_frames: int, rows_per_frame: int):
    """One AUC per eval frame: that frame's held-out rows against all trained rows."""
    values: dict[str, float] = {}
    for frame in range(n_frames):
        held = eval_confidence[frame * rows_per_frame:(frame + 1) * rows_per_frame]
        scores = torch.cat((train_confidence, held))
        labels = torch.cat((torch.ones(train_confidence.shape[0], dtype=torch.bool),
                            torch.zeros(held.shape[0], dtype=torch.bool)))
        pos, neg = int(labels.sum()), int((~labels).sum())
        if pos == 0 or neg == 0:
            continue
        unique, inverse, counts = scores.unique(
            return_inverse=True, return_counts=True)
        cumsum = counts.cumsum(0).float()
        mean_ranks = (cumsum - counts.float() + 1 + cumsum) / 2
        ranks = mean_ranks[inverse]
        auc = float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))
        values[f"frame-{frame:05d}"] = auc - 0.5
    return values


def _fixture_dump(with_probs: bool):
    import torch
    frames, rows = 32, 80
    generator = torch.Generator().manual_seed(7)
    tokens = torch.randint(0, 30, (frames, rows), generator=generator)
    known = torch.ones(frames, rows, dtype=torch.bool)
    control = int(torch.bincount(tokens[known]).argmax())
    classes = torch.arange(30)
    arms = [{"model": "constant", "restart": 0,
             "prediction": torch.full((frames, rows), control)},
            {"model": "oracle", "restart": 0, "prediction": tokens.clone()}]
    if with_probs:
        n = 30
        oracle_probs = torch.full((frames, rows, n), 1.0 / (n * (n - 1)))
        oracle_probs.scatter_(-1, tokens.unsqueeze(-1), 1.0 / n + (n - 1) / (n * (n - 1)))
        arms[1]["probabilities"] = oracle_probs
        arms[1]["train_confidence"] = torch.rand(frames * rows,
                                                 generator=torch.Generator().manual_seed(8))
    return {"eval_tokens": tokens, "known_eval": known, "classes": classes,
            "bundle": "fixture", "arms": arms}


def self_test() -> int:
    import torch
    report = compute(_fixture_dump(with_probs=False),
                     arm="defended", attack="forward_only", seed=42)

    required = {"arm", "attack", "metric", "seed", "cluster_id", "value"}
    checks = {
        "emits_summarizer_schema": all(required <= set(r) for r in report["records"]),
        "both_metrics_computed": report["metrics_computed"] == ["rare_token_top1",
                                                                "token_top1"],
        "constant_arm_reads_zero_on_token_top1":
            abs(report["mean_advantage_by_probe"]["token_top1"]["constant#0"]) < 1e-9,
        "oracle_arm_beats_control":
            report["mean_advantage_by_probe"]["token_top1"]["oracle#0"] > 0,
        "rare_subset_is_a_strict_subset":
            0 < report["rare_rows"] < report["scoreable_rows"],
        "six_metrics_declared_unavailable": len(report["metrics_unavailable"]) == 6,
        "raw_recovery_reported": "rare_token_top1" in report["raw_recovery_pct_by_probe"],
        "degenerate_control_flagged":
            "rare_token_top1" in report["degenerate_control_subsets"],
    }

    with_probs = compute(_fixture_dump(with_probs=True),
                         arm="defended", attack="forward_only", seed=42)
    checks["four_metrics_computed_with_probabilities"] = (
        with_probs["metrics_computed"] == ["membership_auc", "rare_token_top1",
                                           "token_cross_entropy", "token_top1"])
    checks["four_gaps_declared_with_probabilities"] = (
        len(with_probs["metrics_unavailable"]) == 4)
    checks["cross_entropy_oracle_positive"] = (
        with_probs["mean_advantage_by_probe"]["token_cross_entropy"]["oracle#0"] > 0)
    checks["membership_auc_clustered_per_frame"] = all(
        r["cluster_id"].startswith("frame-")
        for r in with_probs["records"] if r["metric"] == "membership_auc")

    # Regression: probabilities are over the `classes` axis, so a dump whose
    # classes are sparse raw token ids must not index out of bounds -- the
    # failure the first re-score hit on the real E1 dump.
    sparse = _fixture_dump(with_probs=True)
    sparse["classes"] = sparse["classes"] * 1000
    sparse["eval_tokens"] = sparse["eval_tokens"] * 1000
    sparse_report = compute(sparse, arm="defended", attack="forward_only", seed=42)
    checks["sparse_class_ids_do_not_index_oob"] = (
        "token_cross_entropy" in sparse_report["metrics_computed"])

    # semantic_cosine with a one-hot public embedding table: the oracle arm
    # reads cos=1 against truth, the constant arm exactly the control's 0.
    embeddings = torch.eye(30)
    with_emb = compute(_fixture_dump(with_probs=True), arm="defended",
                       attack="forward_only", seed=42, embeddings=embeddings)
    checks["semantic_cosine_computed"] = (
        "semantic_cosine" in with_emb["metrics_computed"])
    checks["semantic_oracle_positive"] = (
        with_emb["mean_advantage_by_probe"]["semantic_cosine"]["oracle#0"] > 0)
    checks["semantic_constant_zero"] = (
        abs(with_emb["mean_advantage_by_probe"]["semantic_cosine"]["constant#0"])
        < 1e-9)
    checks["three_gaps_with_embeddings"] = (
        len(with_emb["metrics_unavailable"]) == 3)

    for name, passed in checks.items():
        print(f"  {'ok  ' if passed else 'FAIL'} {name}")
    ok = all(checks.values())
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--attack", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--embeddings", type=Path,
                        help="public base-model embedding table (.pt [vocab, H]); "
                             "enables the semantic_cosine metric (arm minus the "
                             "constant control's cosine). The adversary holds the "
                             "public weights, so this table is in the adversary view.")
    parser.add_argument("--baseline-for",
                        help="name of this arm's matched null, e.g. grad_real_shuffled. "
                             "Without it summarize_complete_view_matrix.py falls back to "
                             "the global shuffled_label arm and, if none exists, emits no "
                             "contrast at all -- the cells still print but the arm-vs-null "
                             "comparison silently does not happen.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jsonl", type=Path,
                        help="write the records as JSONL for the summarizer")
    args = parser.parse_args()

    report = compute(_load(args.dump), args.arm, args.attack, args.seed,
                     embeddings=_load_embeddings(args.embeddings))
    if args.baseline_for:
        for record in report["records"]:
            record["baseline_for"] = args.baseline_for
    if args.jsonl:
        args.jsonl.write_text("".join(
            json.dumps({k: v for k, v in record.items() if k != "probe"}) + "\n"
            for record in report["records"]))
    summary = {k: v for k, v in report.items() if k != "records"}
    summary["record_count"] = len(report["records"])
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test() if "--self-test" in __import__("sys").argv
                     else main())
