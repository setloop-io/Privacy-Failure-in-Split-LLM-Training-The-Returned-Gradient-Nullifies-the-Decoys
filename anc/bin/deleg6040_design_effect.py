#!/usr/bin/env python3
"""Measure the design effect of the privacy metric's per-row sampling model.

The metric scores an attacker's top-1 recovery over n evaluation rows and
reports a Bonferroni-adjusted Wilson upper bound, which assumes the n rows are
independent Bernoulli trials. They are not: rows arrive as eval_blocks frames
of sequence_length rows each (32 real corpus rows plus 48 recycled chaff rows
at the v13 operating point); the invariant_graph arm makes every row's
prediction a function of its whole frame's Gram matrix; and the real rows of a
frame are one contiguous corpus block.

WHAT IS ESTIMATED. Write X_f for the number of rows frame f scores correct,
k for the number of frames and m for the rows per frame, so n = k*m. Under a
cluster model with exchangeable rows inside a frame and intra-cluster
correlation rho,

    Var(X_f) = m p (1-p) [1 + (m-1) rho],

so the variance of the overall rate p_hat is inflated by exactly
D = 1 + (m-1) rho over the binomial value the Wilson bound assumes. D is the
design effect, and re-evaluating every Wilson bound at effective sample size
n/D is the correction the audit's break-even table sweeps over an ASSUMED D.

TWO ESTIMATORS, both reported.

  * ICC_ANOVA -- the one-way random-effects (analysis of variance) estimator,
    rho = (MSB - MSW) / (MSB + (m-1) MSW). For a binary outcome with equal
    cluster sizes this is the standard consistent ICC estimator: it needs only
    within-frame exchangeability, and returns 0 in expectation under
    independence. D_ANOVA = 1 + (m-1) rho.

  * D_KISH -- the linearized survey design effect, the ratio of the
    between-cluster variance of p_hat to its binomial value,
    D = [k/((k-1) n^2) SUM (X_f - p m)^2] / [p(1-p)/n]. This is literally the
    quantity the Wilson correction needs, and it makes no equal-size
    assumption. It carries a k/(k-1) with-replacement inflation the ANOVA
    form does not: on k perfectly clustered frames it returns m*k/(k-1) where
    the ANOVA form returns exactly m. At k=256 that is +0.4%.

UNCERTAINTY. A nonparametric cluster bootstrap: frames are resampled with
replacement (the frame is the independent unit under the cluster model, so it
is the correct resampling unit) and both estimators recomputed, giving a
percentile interval on rho itself. A delete-one-frame jackknife standard
error is reported alongside as an independent nonparametric cross-check.
Neither assumes normality of a binary row.

REAL VERSUS CHAFF. Chaff rows are recycled inside frames and may carry the
correlation for reasons unrelated to the defense, so D is measured separately
on the 32 real rows and the 48 chaff rows of each frame. Which released row
is which is not recorded -- the release permutation never leaves the trusted
process -- but each frame's real LABEL MULTISET is a deterministic function of
the corpus, so the real-correct count of frame f is pinned to the interval
[SUM_v max(0, hit_v - chaff_v), SUM_v min(hit_v, real_v)] exactly as
bin/deleg6040_bundle_forensics.py derives it. Where the interval does not
collapse to a point, D is reported at both endpoints and labelled as two
admissible allocations, never as a bound on D.

NOTHING IS TRAINED HERE. The tool reads a per-row prediction dump written by
the frozen attacker under --dump-eval-predictions, and refuses to read it
unless the re-score that produced it reproduces the cell's committed
*_attacker.json arm counts exactly.

    # per bundle (needs torch, the corpus and the tokenizer)
    python3 bin/deleg6040_design_effect.py --dump D_pred.pt --bundle D.pt \
      --rescore-json D_rescore.json --recorded-attacker-json CELL_attacker.json \
      --run-json CELL.json --model <qwen3-0.6b dir> --corpus <corpus.txt> \
      --output CELL_design_effect.json

    # verdicts re-evaluated at the measured D, over all committed cells
    python3 bin/deleg6040_design_effect.py --sweep paper-data/collected \
      --measurements <dir of *_design_effect.json> --output deff_sweep.json

    python3 bin/deleg6040_design_effect.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deleg6040_bundle_forensics import frame_partition  # noqa: E402
from deleg6040_metric_stats_audit import (  # noqa: E402
    excess_and_cap_at, load_cell, worst_arm,
)

SCHEMA = "dtraining.deleg6040.design_effect.v1"
LEGACY_GATE_PP = 1.0
FLOOR_BUDGET_PP = 0.70
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260819
NULL_DRAWS = 2000
LADDER_NULL_DRAWS = 500
CLUSTER_SCALES = (1, 2, 4, 8, 16, 32, 64, 128)
CONSTANT_FAMILIES = ("invariant_only", "invariant_graph")


# estimators

def anova_rho(counts: np.ndarray, size: int) -> np.ndarray:
    """One-way random-effects ICC of a binary outcome, equal cluster sizes.

    `counts` holds successes per cluster; the last axis indexes clusters, so
    a (B, k) array returns B bootstrap replicates at once.  MSB and MSW are
    the between/within mean squares of the underlying 0/1 rows.
    """
    k = counts.shape[-1]
    if k < 2 or size < 2:
        raise ValueError(f"need >=2 clusters of >=2 rows, got {k}x{size}")
    p = counts / size
    centred = p - p.mean(-1, keepdims=True)
    msb = size * (centred ** 2).sum(-1) / (k - 1)
    msw = size * (p * (1.0 - p)).sum(-1) / (k * (size - 1))
    denominator = msb + (size - 1) * msw
    safe = np.where(denominator > 0.0, denominator, 1.0)
    return np.where(denominator > 0.0, (msb - msw) / safe, np.nan)


def kish_deff(counts: np.ndarray, size: int) -> np.ndarray:
    """Linearized design effect of the clustered rate (Kish).

    Var_cluster(p_hat) = k/((k-1) n^2) SUM (X_f - p m)^2 is the standard
    with-replacement cluster variance; dividing by p(1-p)/n gives the factor
    by which the Wilson bound's effective sample size must be reduced.
    """
    k = counts.shape[-1]
    n = float(k * size)
    p = counts.sum(-1) / n
    residual = counts - p[..., None] * size
    var_cluster = k * (residual ** 2).sum(-1) / ((k - 1) * n * n)
    var_srs = p * (1.0 - p) / n
    safe = np.where(var_srs > 0.0, var_srs, 1.0)
    return np.where(var_srs > 0.0, var_cluster / safe, np.nan)


def kish_deff_ragged(correct: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Kish design effect when clusters differ in size (a ratio estimator).

    p_hat = SUM A_f / SUM M_f, whose linearized cluster variance is
    k/((k-1) N^2) SUM (A_f - p M_f)^2.  Reduces to kish_deff() when every M_f
    is equal, so the unambiguous sub-populations are measured on the same
    definition as the full ones.
    """
    k = correct.shape[-1]
    n = sizes.sum(-1)
    p = correct.sum(-1) / np.where(n > 0, n, 1.0)
    residual = correct - p[..., None] * sizes
    var_cluster = k * (residual ** 2).sum(-1) / ((k - 1) * n * n)
    var_srs = p * (1.0 - p) / np.where(n > 0, n, 1.0)
    safe = np.where(var_srs > 0.0, var_srs, 1.0)
    return np.where(var_srs > 0.0, var_cluster / safe, np.nan)


def ragged_stats(correct: np.ndarray, sizes: np.ndarray, draws: int,
                 seed: int) -> dict[str, Any]:
    """Design effect and bootstrap interval for unequal-size clusters."""
    correct = np.asarray(correct, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    rng = np.random.default_rng(seed)
    k = correct.size
    chunk = max(1, 4_000_000 // max(1, k))
    parts: list[np.ndarray] = []
    done = 0
    while done < draws:
        take = min(chunk, draws - done)
        index = rng.integers(0, k, size=(take, k))
        parts.append(kish_deff_ragged(correct[index], sizes[index]))
        done += take
    boot = np.concatenate(parts)
    deff = float(kish_deff_ragged(correct, sizes))
    mean_size = float(sizes.mean())
    return {"clusters": k, "mean_cluster_size": mean_size,
            "rows": float(sizes.sum()), "correct": float(correct.sum()),
            "p_hat": float(correct.sum()) / max(1.0, float(sizes.sum())),
            "deff_kish": deff,
            "deff_kish_ci95": [float(v)
                               for v in np.nanpercentile(boot, [2.5, 97.5])],
            "implied_icc": (deff - 1.0) / (mean_size - 1.0)
            if mean_size > 1.0 else float("nan")}


def bootstrap_cluster(counts: np.ndarray, size: int, draws: int,
                      seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample whole frames with replacement; return (rho, D_kish) draws."""
    rng = np.random.default_rng(seed)
    k = counts.size
    chunk = max(1, 4_000_000 // max(1, k))
    rho_parts: list[np.ndarray] = []
    kish_parts: list[np.ndarray] = []
    done = 0
    while done < draws:
        take = min(chunk, draws - done)
        sample = counts[rng.integers(0, k, size=(take, k))]
        rho_parts.append(anova_rho(sample, size))
        kish_parts.append(kish_deff(sample, size))
        done += take
    return np.concatenate(rho_parts), np.concatenate(kish_parts)


def jackknife_se(counts: np.ndarray,
                 statistic: Callable[[np.ndarray], float]) -> float:
    """Delete-one-frame jackknife standard error of a cluster statistic."""
    k = counts.size
    values = np.array([statistic(np.delete(counts, index))
                       for index in range(k)], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return float("nan")
    return float(np.sqrt((finite.size - 1) / finite.size
                         * ((finite - finite.mean()) ** 2).sum()))


def permutation_null(total: int, clusters: int, size: int, draws: int,
                     seed: int) -> np.ndarray:
    """D_anova when the SAME successes are reallocated to frames at random.

    This is the exchangeable null the Wilson bound assumes, conditioned on the
    observed success total, so it gives the exact reference distribution of
    the estimator for this cell's n, k, m and p.  A measured D inside it is
    indistinguishable from independent rows; one above it is not.
    """
    rng = np.random.default_rng(seed)
    n = clusters * size
    base = np.zeros(n, dtype=np.int8)
    base[:total] = 1
    chunk = max(1, 8_000_000 // max(1, n))
    parts: list[np.ndarray] = []
    done = 0
    while done < draws:
        take = min(chunk, draws - done)
        block = rng.permuted(np.tile(base, (take, 1)), axis=1)
        counts = block.reshape(take, clusters, size).sum(-1).astype(float)
        parts.append(1.0 + (size - 1) * anova_rho(counts, size))
        done += take
    return np.concatenate(parts)


def lag1_autocorrelation(counts: np.ndarray) -> float:
    """Correlation between neighbouring frames' correct counts.

    The design effect corrects for dependence WITHIN a frame and assumes the
    frames themselves are independent, which is also what the cluster
    bootstrap resamples on.  Frames are consecutive corpus blocks, so this is
    the check that the assumption holds; a large value would mean the
    bootstrap interval is too narrow and D is not the whole correction.
    """
    if counts.size < 3 or counts.std() == 0.0:
        return float("nan")
    centred = counts - counts.mean()
    return float((centred[:-1] * centred[1:]).sum() / (centred ** 2).sum())


def cluster_point(counts: np.ndarray, size: int) -> dict[str, Any]:
    """The two point estimators and the population they were measured on."""
    rho = float(anova_rho(counts, size))
    n = int(counts.size) * size
    return {"clusters": int(counts.size), "cluster_size": size, "rows": n,
            "correct": int(counts.sum()), "p_hat": float(counts.sum()) / n,
            "icc_anova": rho, "deff_anova": 1.0 + (size - 1) * rho,
            "deff_kish": float(kish_deff(counts, size)),
            "frame_autocorrelation_lag1": lag1_autocorrelation(counts)}


def deff_mean(rows: np.ndarray) -> float:
    """Kish design effect of the mean of a per-row score that is not binary.

    `rows` is (clusters, cluster_size).  Var_srs uses the score's own row
    variance rather than p(1-p), so the same definition covers the paired
    difference d = 1{arm correct} - 1{control correct}, which takes values in
    {-1, 0, +1}.
    """
    k, size = rows.shape
    n = float(k * size)
    mean = float(rows.mean())
    totals = rows.sum(1)
    var_cluster = k * float(((totals - mean * size) ** 2).sum()) / ((k - 1)
                                                                    * n * n)
    var_srs = float(rows.var(ddof=1)) / n
    return var_cluster / var_srs if var_srs > 0.0 else float("nan")


def paired_ladder(arm: np.ndarray, control: np.ndarray,
                  groups: Sequence[int]) -> list[dict[str, Any]]:
    """Design effect of the ARM MINUS CONTROL difference, at each scale.

    The gate subtracts a control scored on the very same rows, so any row
    property both predictors share -- above all the drift in how often the
    modal token occurs -- is common mode.  If the long-range design effect is
    that drift, it cancels here and this stays near 1 while the one-sample
    ladder climbs.  This is the quantity a paired analysis would bound; the
    published statistic is not paired, so this diagnoses the cause rather
    than restating the published number.
    """
    difference = arm.astype(float) - control.astype(float)
    frames = difference.shape[0]
    out = []
    for group in groups:
        clusters = frames // group
        if clusters < 32:
            break
        merged = difference[:clusters * group].reshape(clusters, -1)
        out.append({"frames_per_cluster": int(group), "clusters": clusters,
                    "cluster_size": int(merged.shape[1]),
                    "deff_paired": deff_mean(merged)})
    return out


def cluster_scale_ladder(counts: np.ndarray, size: int,
                         groups: Sequence[int], null_draws: int = 0,
                         seed: int = 0) -> list[dict[str, Any]]:
    """The design effect when consecutive frames are merged into one cluster.

    D as measured on single frames corrects dependence WITHIN a frame only.
    Frames are consecutive corpus blocks, so any dependence that runs across
    frame boundaries would be invisible there and would also make the cluster
    bootstrap too narrow.  Merging g neighbouring frames into one cluster of
    g*m rows measures the two together: a D that stays flat as g grows means
    the dependence is confined to the frame; a D that climbs means it is not.
    """
    out = []
    for group in groups:
        clusters = counts.size // group
        if clusters < 32:
            break
        merged = counts[:clusters * group].reshape(clusters, group).sum(1)
        step = {"frames_per_cluster": int(group),
                **cluster_point(merged, size * group)}
        if null_draws > 0:
            null = permutation_null(step["correct"], clusters, size * group,
                                    null_draws, seed + group)
            step["deff_null_ci95"] = [float(v) for v in
                                      np.nanpercentile(null, [2.5, 97.5])]
            step["null_p_value"] = float(np.nanmean(
                null >= step["deff_anova"]))
        out.append(step)
    return out


def cluster_stats(counts: np.ndarray, size: int, draws: int, seed: int,
                  null_draws: int) -> dict[str, Any]:
    """Point estimates, bootstrap interval, jackknife SE and permutation null."""
    counts = np.asarray(counts, dtype=float)
    point = cluster_point(counts, size)
    boot_rho, boot_kish = bootstrap_cluster(counts, size, draws, seed)
    lo, hi = (float(v) for v in np.nanpercentile(boot_rho, [2.5, 97.5]))
    kish = [float(v) for v in np.nanpercentile(boot_kish, [2.5, 97.5])]
    null = permutation_null(point["correct"], point["clusters"], size,
                            null_draws, seed + 1)
    return {
        **point, "icc_ci95": [lo, hi],
        "icc_jackknife_se": jackknife_se(
            counts, lambda c: float(anova_rho(c, size))),
        "deff_anova_ci95": [1.0 + (size - 1) * lo, 1.0 + (size - 1) * hi],
        "deff_kish_ci95": kish,
        "deff_null_ci95": [float(v)
                           for v in np.nanpercentile(null, [2.5, 97.5])],
        "deff_null_median": float(np.nanmedian(null)),
        "null_p_value": float(np.nanmean(null >= point["deff_anova"])),
        "bootstrap_draws": int(draws), "bootstrap_seed": int(seed),
        "null_draws": int(null_draws),
    }


# per-row correctness

def load_predictions(dump_path: Path, bundle_path: Path) -> dict[str, Any]:
    """Per-arm per-row correctness exactly as the frozen attacker scores it."""
    import torch

    dump = torch.load(dump_path, map_location="cpu")
    bundle = torch.load(bundle_path, map_location="cpu")
    tokens = dump["eval_tokens"]
    if not torch.equal(tokens, bundle["eval_tokens"]):
        raise SystemExit("prediction dump does not match the bundle labels")
    classes, known = dump["classes"], dump["known_eval"]
    majority = int(torch.mode(bundle["train_tokens"].reshape(-1)).values)
    arms = []
    for arm in dump["arms"]:
        predicted = classes[arm["prediction"]]
        values, counts = torch.unique(predicted.reshape(-1),
                                      return_counts=True)
        arms.append({
            "model": arm["model"], "restart": arm["restart"],
            "correct": ((predicted == tokens) & known).numpy(),
            "predicted": predicted.numpy(),
            "distinct_tokens_predicted": int(values.numel()),
            "modal_token": int(values[int(counts.argmax())]),
            "modal_token_share": float(counts.max()) / predicted.numel(),
        })
    return {"labels": tokens.numpy(), "majority": majority, "arms": arms,
            "known": known.numpy(),
            "classes": set(int(v) for v in classes.tolist()),
            "control_correct": (tokens == majority).numpy()}


def verify_arm_counts(arms: Sequence[dict[str, Any]], rescore: Path,
                      recorded: Path) -> dict[str, Any]:
    """Refuse the dump unless the re-score reproduces the committed counts."""
    live = json.loads(rescore.read_text())
    published = json.loads(recorded.read_text())
    for label, source in (("rescore", live), ("committed", published)):
        counts = {(r["model"], r["restart"]): r["correct"]
                  for r in source["results"]}
        for arm in arms:
            key = (arm["model"], arm["restart"])
            got = int(arm["correct"].sum())
            if counts.get(key) != got:
                raise SystemExit(f"arm {key}: dump scores {got}, {label} "
                                 f"artifact records {counts.get(key)}")
    if live["summary"][0] != published["summary"][0]:
        raise SystemExit("re-score summary differs from the committed artifact")
    return {"rescored_summary": live["summary"][0],
            "committed_summary": published["summary"][0],
            "identical": True}


def real_labels_per_frame(model_dir: str, corpus: str, seq_len: int,
                          train_blocks: int,
                          eval_blocks: int) -> list[list[int]]:
    """Regenerate the real evaluation labels exactly as the runner did.

    bin/run_latent_native_v5_06b.py:252-255 takes the first
    train_blocks+eval_blocks disjoint (seq_len+1)-token blocks of the corpus
    and holds out the tail; each held block's labels are its shifted tokens.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    ids = tokenizer(Path(corpus).read_text(errors="replace"),
                    add_special_tokens=False)["input_ids"]
    width = seq_len + 1
    blocks = [ids[i:i + width]
              for i in range(0, len(ids) - width + 1, width)]
    held = blocks[:train_blocks + eval_blocks][train_blocks:]
    if len(held) != eval_blocks:
        raise SystemExit(f"corpus yields {len(held)} eval blocks, "
                         f"expected {eval_blocks}")
    return [block[1:] for block in held]


def real_correct_bounds(labels: np.ndarray, parts: list[tuple[Counter, Counter]],
                        correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame real-correct count, pinned to [lo, hi] by the label multisets.

    A frame's k rows carrying token v split into real_v and chaff_v; if k of
    them score, between max(0, k - chaff_v) and min(k, real_v) of the hits sat
    on real rows.  Equal endpoints mean the attribution is exact.
    """
    lo = np.zeros(len(parts), dtype=float)
    hi = np.zeros(len(parts), dtype=float)
    for index, (real, chaff) in enumerate(parts):
        hit = Counter(int(v) for v, ok in zip(labels[index], correct[index])
                      if ok)
        lo[index] = sum(max(0, n - chaff[v]) for v, n in hit.items())
        hi[index] = sum(min(n, real[v]) for v, n in hit.items())
    return lo, hi


def unambiguous_rows(labels: np.ndarray,
                     parts: list[tuple[Counter, Counter]],
                     correct: np.ndarray) -> dict[str, np.ndarray]:
    """Rows whose side is settled by the frame's own label multisets.

    A token carried only by real rows of a frame identifies every row of that
    frame carrying it as real, and vice versa; those rows need no allocation
    at all, so their per-frame sizes and correct counts are exact for EVERY
    arm, including the diverse ones whose lo/hi bracket is wide.  Cluster
    sizes then differ between frames, which ragged_stats() handles.

    READ THE RESULT WITH CARE.  Membership is a frame-level property -- a
    token is unambiguous in a frame only if that frame's chaff draw missed it
    -- so the sub-population's own rate swings between frames by construction
    and its design effect is inflated by the selection, not by the design.
    The control proves it: it never sees a frame at all, yet reads D ~ 2.4
    here against D ~ 1 on the rows the metric actually scores.  This is a
    diagnostic, never the real-row measurement.
    """
    out = {key: np.zeros(len(parts), dtype=float)
           for key in ("real_rows", "real_correct", "chaff_rows",
                       "chaff_correct")}
    for index, (real, chaff) in enumerate(parts):
        seen = Counter(int(v) for v in labels[index])
        hit = Counter(int(v) for v, ok in zip(labels[index], correct[index])
                      if ok)
        for value, count in seen.items():
            if chaff[value] == 0:
                out["real_rows"][index] += count
                out["real_correct"][index] += hit[value]
            elif real[value] == 0:
                out["chaff_rows"][index] += count
                out["chaff_correct"][index] += hit[value]
    return out


# one bundle

def arm_populations(correct: np.ndarray, parts: list[tuple[Counter, Counter]],
                    labels: np.ndarray, seq_len: int,
                    args: argparse.Namespace) -> dict[str, Any]:
    """Design effect on all rows, on real rows only, and on chaff rows only."""
    draws, seed, null = args.bootstrap, args.bootstrap_seed, args.null_draws
    rows_per_frame = correct.shape[1]
    total = correct.sum(1).astype(float)
    real_lo, real_hi = real_correct_bounds(labels, parts, correct)
    exact = bool(np.array_equal(real_lo, real_hi))
    chaff_size = rows_per_frame - seq_len
    out: dict[str, Any] = {
        "all_rows": cluster_stats(total, rows_per_frame, draws, seed, null),
        "real_attribution_exact": exact,
        "real_ambiguous_rows": float((real_hi - real_lo).sum()),
        "real_rows_lo": cluster_stats(real_lo, seq_len, draws, seed, null),
        "chaff_rows_lo": cluster_stats(total - real_hi, chaff_size, draws,
                                       seed, null),
    }
    if not exact:
        out["real_rows_hi"] = cluster_stats(real_hi, seq_len, draws, seed,
                                            null)
        out["chaff_rows_hi"] = cluster_stats(total - real_lo, chaff_size,
                                             draws, seed, null)
    unambiguous = unambiguous_rows(labels, parts, correct)
    out["real_rows_unambiguous"] = ragged_stats(
        unambiguous["real_correct"], unambiguous["real_rows"], draws, seed)
    out["chaff_rows_unambiguous"] = ragged_stats(
        unambiguous["chaff_correct"], unambiguous["chaff_rows"], draws, seed)
    out["cluster_scale_ladder"] = cluster_scale_ladder(
        total, rows_per_frame, CLUSTER_SCALES, args.ladder_null_draws,
        args.bootstrap_seed)
    out["frame_counts"] = {name: " ".join(str(int(v)) for v in series)
                           for name, series in
                           (("correct", total), ("real_correct_lo", real_lo),
                            ("real_correct_hi", real_hi))}
    return out


def arm_family(model: str, distinct: int) -> str:
    """Constant-predictor family versus diverse-predictor family."""
    return "near_constant" if model in CONSTANT_FAMILIES else "diverse"


def positive_controls(scored: dict[str, Any],
                      parts: list[tuple[Counter, Counter]], seq_len: int,
                      args: argparse.Namespace) -> dict[str, Any]:
    """Two row properties this design MUST cluster, measured the same way.

    A row is scoreable only if its token is in the attacker's train class set,
    and the 32 real rows of a frame are one contiguous corpus block, so rare
    tokens arrive in bursts.  If the estimator returned D ~ 1 here too it
    would be measuring nothing; these are the check that it can see clustering
    in this very data.  The real-row count is exact -- it reads the
    regenerated label multiset, not the released row order.
    """
    draws, seed, null = args.bootstrap, args.bootstrap_seed, args.null_draws
    known = scored["known"]
    classes = scored["classes"]
    known_real = np.array([sum(n for v, n in real.items() if v in classes)
                           for real, _ in parts], dtype=float)
    return {
        "scoreable_row_indicator_all_rows": cluster_stats(
            known.sum(1).astype(float), known.shape[1], draws, seed, null),
        "scoreable_row_indicator_real_rows": cluster_stats(
            known_real, seq_len, draws, seed, null)}


def measure_bundle(args: argparse.Namespace) -> dict[str, Any]:
    """Every arm plus the control, on one retained bundle."""
    scored = load_predictions(Path(args.dump), Path(args.bundle))
    reproduction = verify_arm_counts(scored["arms"], Path(args.rescore_json),
                                     Path(args.recorded_attacker_json))
    run = json.loads(Path(args.run_json).read_text())
    labels = scored["labels"]
    frames, rows_per_frame = labels.shape
    seq_len = rows_per_frame - int(run["chaff_tokens"])
    real = real_labels_per_frame(args.model, args.corpus, seq_len,
                                 int(run["train_blocks"]), frames)
    parts = [frame_partition([int(v) for v in observed], block)
             for observed, block in zip(labels, real)]
    arms = []
    for arm in scored["arms"]:
        arms.append({
            "model": arm["model"], "restart": arm["restart"],
            "family": arm_family(arm["model"],
                                 arm["distinct_tokens_predicted"]),
            "distinct_tokens_predicted": arm["distinct_tokens_predicted"],
            "modal_token": arm["modal_token"],
            "modal_token_share": arm["modal_token_share"],
            "paired_ladder": paired_ladder(arm["correct"],
                                           scored["control_correct"],
                                           CLUSTER_SCALES),
            "agrees_with_control_on_every_row": bool(
                (arm["correct"] == scored["control_correct"]).all()),
            **arm_populations(arm["correct"], parts, labels, seq_len, args)})
    control = {"token": scored["majority"], "family": "control",
               "model": "majority_control", "restart": 0,
               "distinct_tokens_predicted": 1,
               "modal_token": scored["majority"], "modal_token_share": 1.0,
               **arm_populations(scored["control_correct"], parts, labels,
                                 seq_len, args)}
    return {"schema": SCHEMA, "cell": Path(args.recorded_attacker_json).name
            .replace("_attacker.json", ""),
            "bundle": Path(args.bundle).name,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "frames": int(frames), "rows_per_frame": int(rows_per_frame),
            "real_per_frame": seq_len,
            "chaff_per_frame": int(run["chaff_tokens"]),
            "runner_train_blocks": int(run["train_blocks"]),
            "reproduction": reproduction, "arms": arms, "control": control,
            "positive_controls": positive_controls(scored, parts, seq_len,
                                                   args),
            "gate": gate_reading(Path(args.recorded_attacker_json), arms,
                                 control)}


def gate_reading(recorded: Path, arms: list[dict[str, Any]],
                 control: dict[str, Any]) -> dict[str, Any]:
    """The measured design effect of the arm the published gate actually reads."""
    cell = load_cell(recorded)
    arm, upper = worst_arm(cell)
    match = [a for a in arms
             if (a["model"], a["restart"]) == (arm["model"], arm["restart"])]
    if len(match) != 1:
        raise SystemExit(f"gate arm {arm} not present in the dump")
    gate = match[0]
    return {"model": arm["model"], "restart": arm["restart"],
            "correct": arm["correct"], "upper95_pct": upper,
            "family": gate["family"],
            "deff_anova": gate["all_rows"]["deff_anova"],
            "deff_anova_ci95": gate["all_rows"]["deff_anova_ci95"],
            "deff_kish": gate["all_rows"]["deff_kish"],
            "icc_anova": gate["all_rows"]["icc_anova"],
            "icc_ci95": gate["all_rows"]["icc_ci95"],
            "deff_null_ci95": gate["all_rows"]["deff_null_ci95"],
            "null_p_value": gate["all_rows"]["null_p_value"],
            "frame_autocorrelation_lag1":
                gate["all_rows"]["frame_autocorrelation_lag1"],
            "cluster_scale_ladder": gate["cluster_scale_ladder"],
            "paired_ladder": gate["paired_ladder"],
            "deff_widest_cluster": max(
                step["deff_anova"] for step in gate["cluster_scale_ladder"]),
            "deff_paired_widest_cluster": max(
                step["deff_paired"] for step in gate["paired_ladder"]),
            "agrees_with_control_on_every_row":
                gate["agrees_with_control_on_every_row"],
            "control_cluster_scale_ladder": control["cluster_scale_ladder"],
            "control_deff_widest_cluster": max(
                step["deff_anova"]
                for step in control["cluster_scale_ladder"]),
            "deff_real_rows_lo": gate["real_rows_lo"]["deff_anova"],
            "deff_chaff_rows_lo": gate["chaff_rows_lo"]["deff_anova"],
            "real_attribution_exact": gate["real_attribution_exact"],
            "control_deff_anova": control["all_rows"]["deff_anova"],
            "control_deff_anova_ci95": control["all_rows"]["deff_anova_ci95"],
            "control_icc_anova": control["all_rows"]["icc_anova"]}


# verdicts at measured D

def verdict_at(cell: dict[str, Any], correct: int, arm_deff: float,
               control_deff: float) -> dict[str, Any]:
    """Both gate legs re-evaluated at a measured design effect.

    The arm's bound moves to effective sample size n/arm_deff and the floor
    to n/control_deff, since the control is its own clustered estimate.  The
    estimator, the arm, the control and both legs are otherwise the committed
    ones, so arm_deff = control_deff = 1 reproduces the published numbers.
    """
    excess, _ = excess_and_cap_at(arm_deff, cell, correct, LEGACY_GATE_PP,
                                  FLOOR_BUDGET_PP)
    _, cap = excess_and_cap_at(control_deff, cell, correct, LEGACY_GATE_PP,
                               FLOOR_BUDGET_PP)
    return {"arm_deff": arm_deff, "control_deff": control_deff,
            "excess_pp": excess, "gate_cap_pp": cap,
            "verdict": "PASS" if excess <= cap else "FAIL"}


def reevaluate_cell(path: Path,
                    measured: dict[str, Any] | None) -> dict[str, Any]:
    """One committed cell's verdict, published and at its measured D."""
    cell = load_cell(path)
    arm, _ = worst_arm(cell)
    published = verdict_at(cell, arm["correct"], 1.0, 1.0)
    record = {"file": str(path), "n_rows": cell["n_rows"],
              "frames": cell["frames"], "rows_per_frame": cell["rows_per_frame"],
              "gate_arm": arm["model"], "gate_restart": arm["restart"],
              "published": published, "covered": measured is not None}
    if measured is None:
        return record
    gate = measured["gate"]
    record["measured"] = {
        "icc_anova": gate["icc_anova"], "icc_ci95": gate["icc_ci95"],
        "deff_anova": gate["deff_anova"],
        "deff_anova_ci95": gate["deff_anova_ci95"],
        "deff_kish": gate["deff_kish"],
        "deff_null_ci95": gate["deff_null_ci95"],
        "null_p_value": gate["null_p_value"],
        "deff_real_rows_lo": gate["deff_real_rows_lo"],
        "deff_chaff_rows_lo": gate["deff_chaff_rows_lo"],
        "frame_autocorrelation_lag1": gate["frame_autocorrelation_lag1"],
        "deff_widest_cluster": gate["deff_widest_cluster"],
        "deff_paired_widest_cluster": gate["deff_paired_widest_cluster"],
        "agrees_with_control_on_every_row":
            gate["agrees_with_control_on_every_row"],
        "control_deff_widest_cluster": gate["control_deff_widest_cluster"],
        "cluster_scale_ladder": gate["cluster_scale_ladder"],
        "paired_ladder": gate["paired_ladder"],
        "at_widest_cluster": verdict_at(cell, arm["correct"],
                                        gate["deff_widest_cluster"],
                                        gate["control_deff_widest_cluster"]),
        "family": gate["family"],
        "control_deff_anova": gate["control_deff_anova"],
        "at_point": verdict_at(cell, arm["correct"], gate["deff_anova"],
                               gate["control_deff_anova"]),
        "at_upper_ci": verdict_at(cell, arm["correct"],
                                  gate["deff_anova_ci95"][1],
                                  gate["control_deff_anova_ci95"][1]),
        "at_gate_deff_single": verdict_at(cell, arm["correct"],
                                          gate["deff_anova"],
                                          gate["deff_anova"])}
    return record


def extrapolate(record: dict[str, Any], icc: float) -> dict[str, Any]:
    """A covered cell's ICC carried to an uncovered cell of a different shape.

    D = 1 + (m-1) rho is a function of the frame size, so only rho travels;
    the design effect is rebuilt at the target cell's own m.  Labelled as an
    extrapolation everywhere it is used.
    """
    cell = load_cell(Path(record["file"]))
    arm, _ = worst_arm(cell)
    deff = 1.0 + (cell["rows_per_frame"] - 1) * icc
    return {"icc_applied": icc, "deff": deff,
            **verdict_at(cell, arm["correct"], deff, deff)}


# sweeping

def load_measurements(directory: Path) -> dict[str, dict[str, Any]]:
    """Every per-bundle measurement artifact, keyed by cell stem."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*_design_effect.json")):
        report = json.loads(path.read_text())
        if report.get("schema") != SCHEMA:
            raise SystemExit(f"{path}: not a {SCHEMA} artifact")
        out[report["cell"]] = report
    if not out:
        raise SystemExit(f"no *_design_effect.json under {directory}")
    return out


def spread(values: list[float], size: int) -> dict[str, Any]:
    """Across-cell spread of an ICC, and the design effect it implies at m."""
    array = np.array(values, dtype=float)
    mean = float(array.mean())
    half = (1.96 * float(array.std(ddof=1)) / np.sqrt(array.size)
            if array.size > 1 else float("nan"))
    return {"n": int(array.size), "icc_mean": mean,
            "icc_mean_ci95": [mean - half, mean + half],
            "icc_min": float(array.min()), "icc_median": float(np.median(array)),
            "icc_max": float(array.max()),
            "deff_at_mean": 1.0 + (size - 1) * mean,
            "deff_at_max": 1.0 + (size - 1) * float(array.max())}


def family_summary(measurements: dict[str, dict[str, Any]], population: str,
                   exact_only: bool = False) -> dict[str, Any]:
    """ICC by arm family across every covered cell, on one row population.

    exact_only keeps the arms whose real/chaff split needed no allocation, so
    the real-row and chaff-row figures are measurements rather than one
    admissible endpoint of a bracket.
    """
    buckets: dict[str, list[float]] = {}
    significant: dict[str, int] = {}
    size = 0
    for report in measurements.values():
        for arm in report["arms"] + [report["control"]]:
            if exact_only and not arm["real_attribution_exact"]:
                continue
            stats = arm[population]
            size = stats["cluster_size"]
            buckets.setdefault(arm["family"], []).append(stats["icc_anova"])
            significant[arm["family"]] = significant.get(arm["family"], 0) + (
                stats["null_p_value"] < 0.05)
    return {family: {**spread(values, size), "cluster_size": size,
                     "n_significant_at_5pct": significant[family]}
            for family, values in sorted(buckets.items())}


def family_summary_ragged(measurements: dict[str, dict[str, Any]],
                          population: str) -> dict[str, Any]:
    """Design effect by arm family on an exactly-attributed sub-population."""
    buckets: dict[str, list[float]] = {}
    sizes: dict[str, list[float]] = {}
    for report in measurements.values():
        for arm in report["arms"] + [report["control"]]:
            stats = arm[population]
            buckets.setdefault(arm["family"], []).append(stats["deff_kish"])
            sizes.setdefault(arm["family"], []).append(
                stats["mean_cluster_size"])
    out = {}
    for family, values in sorted(buckets.items()):
        array = np.array(values, dtype=float)
        out[family] = {"n": int(array.size), "deff_mean": float(array.mean()),
                       "deff_min": float(array.min()),
                       "deff_median": float(np.median(array)),
                       "deff_max": float(array.max()),
                       "mean_cluster_size": float(np.mean(sizes[family]))}
    return out


def pooled_icc(measurements: dict[str, dict[str, Any]],
               shape: int) -> dict[str, Any]:
    """Gate-arm ICC across covered cells of one frame size."""
    values = [m["gate"]["icc_anova"] for m in measurements.values()
              if m["rows_per_frame"] == shape]
    if not values:
        raise SystemExit(f"no covered cell with {shape} rows per frame")
    return {"rows_per_frame": shape, **spread(values, shape)}


def family_tables(measurements: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Every by-family view, on each row population.

    The `_exact` views keep only the arms whose real/chaff split needed no
    allocation; the `_lo` views keep every arm at one admissible endpoint and
    are a sensitivity, not a measurement.  The `_unambiguous` views are a
    diagnostic whose selection rule is frame-correlated -- see
    unambiguous_rows().
    """
    shapes = sorted({m["rows_per_frame"] for m in measurements.values()})
    return {
        "gate_arm_icc_by_frame_shape": [pooled_icc(measurements, s)
                                        for s in shapes],
        "family_all_rows": family_summary(measurements, "all_rows"),
        "family_real_rows_exact": family_summary(measurements, "real_rows_lo",
                                                 exact_only=True),
        "family_chaff_rows_exact": family_summary(measurements,
                                                  "chaff_rows_lo",
                                                  exact_only=True),
        "family_real_rows_lo": family_summary(measurements, "real_rows_lo"),
        "family_chaff_rows_lo": family_summary(measurements, "chaff_rows_lo"),
        "family_real_rows_unambiguous": family_summary_ragged(
            measurements, "real_rows_unambiguous"),
        "family_chaff_rows_unambiguous": family_summary_ragged(
            measurements, "chaff_rows_unambiguous")}


def verdict_counts(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """How many verdicts survive each correction, and which do not."""
    covered = [c for c in cells if c["covered"]]

    def passing(records: list[dict[str, Any]], reading: str) -> int:
        return sum(r["measured"][reading]["verdict"] == "PASS"
                   for r in records)

    return {
        "n_covered": len(covered),
        "n_published_pass": sum(c["published"]["verdict"] == "PASS"
                                for c in cells),
        "n_covered_pass_published": sum(
            c["published"]["verdict"] == "PASS" for c in covered),
        "n_covered_pass_measured": passing(covered, "at_point"),
        "n_covered_pass_measured_upper": passing(covered, "at_upper_ci"),
        "n_covered_pass_widest_cluster": passing(covered, "at_widest_cluster"),
        "flipped_to_fail": [
            c["file"] for c in covered
            if c["published"]["verdict"] == "PASS"
            and c["measured"]["at_point"]["verdict"] == "FAIL"]}


def sweep(root: Path, measurements: dict[str, dict[str, Any]],
          extrapolate_icc: float | None,
          exclude: Sequence[str] = ()) -> dict[str, Any]:
    """Every committed attacker artifact, re-judged at the measured D."""
    cells = []
    for path in sorted(root.rglob("*_attacker.json")):
        if any(token in str(path) for token in exclude):
            continue
        stem = path.name.replace("_attacker.json", "")
        record = reevaluate_cell(path, measurements.get(stem))
        if not record["covered"] and extrapolate_icc is not None:
            record["extrapolated"] = extrapolate(record, extrapolate_icc)
        cells.append(record)
    return {
        "schema": "dtraining.deleg6040.design_effect_sweep.v1",
        "root": str(root), "excluded": list(exclude),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_cells": len(cells), **verdict_counts(cells),
        **family_tables(measurements),
        "extrapolation_icc": extrapolate_icc,
        "n_extrapolated_pass": sum(
            c["extrapolated"]["verdict"] == "PASS" for c in cells
            if "extrapolated" in c) if extrapolate_icc is not None else None,
        "cells": cells}


def print_cell_row(cell: dict[str, Any]) -> None:
    """One covered cell: its measured D, its diagnostics, and four verdicts."""
    m, name = cell["measured"], Path(cell["file"]).name
    point, upper = m["at_point"], m["at_upper_ci"]
    arm = cell["gate_arm"][:16] + "/" + str(cell["gate_restart"])
    print(f"{name.replace('_attacker.json', ''):<38}{arm:>18}"
          f"{m['deff_anova']:>7.3f}{m['deff_anova_ci95'][0]:>7.3f}"
          f"{m['deff_anova_ci95'][1]:>7.3f}"
          f"{m['null_p_value']:>7.3f}"
          f"{m['frame_autocorrelation_lag1']:>7.3f}"
          f"{m['deff_real_rows_lo']:>7.3f}{m['deff_chaff_rows_lo']:>8.3f}"
          f"{m['deff_widest_cluster']:>8.3f}"
          f"{m['deff_paired_widest_cluster']:>8.3f}"
          f"{point['excess_pp']:>9.4f}{point['gate_cap_pp']:>7.4f}"
          f"{cell['published']['verdict']:>6}{point['verdict']:>6}"
          f"{upper['verdict']:>6}{m['at_widest_cluster']['verdict']:>7}")


def print_sweep(report: dict[str, Any]) -> None:
    """The covered cells, their measured D, and what it does to each verdict."""
    header = (f"{'cell':<38}{'gate arm':>18}{'D':>7}{'Dlo':>7}{'Dhi':>7}"
              f"{'p':>7}{'ac1':>7}{'Dreal':>7}{'Dchaff':>8}{'Dwide':>8}"
              f"{'Dpair':>8}{'excess':>9}{'cap':>7}{'pub':>6}{'meas':>6}"
              f"{'@hi':>6}{'@wide':>7}")
    print(header)
    print("-" * len(header))
    for cell in report["cells"]:
        if cell["covered"]:
            print_cell_row(cell)
    print(f"\ncells {report['n_cells']}  covered {report['n_covered']}  "
          f"published PASS {report['n_published_pass']}/{report['n_cells']}")
    print(f"covered cells passing: published {report['n_covered_pass_published']}"
          f"  at measured D {report['n_covered_pass_measured']}"
          f"  at upper 95% D {report['n_covered_pass_measured_upper']}"
          f"  at widest merged cluster {report['n_covered_pass_widest_cluster']}")
    print(f"verdicts flipped PASS->FAIL at the measured D: "
          f"{report['flipped_to_fail'] or 'none'}")
    for population in ("family_all_rows", "family_real_rows_exact",
                       "family_chaff_rows_exact", "family_real_rows_lo",
                       "family_chaff_rows_lo"):
        print(f"\n{population}")
        for family, stats in report[population].items():
            print(f"  {family:<15} arms={stats['n']:>3} "
                  f"ICC mean {stats['icc_mean']:+.5f} "
                  f"CI[{stats['icc_mean_ci95'][0]:+.5f},"
                  f"{stats['icc_mean_ci95'][1]:+.5f}] "
                  f"max {stats['icc_max']:+.5f}  "
                  f"D(m={stats['cluster_size']}) mean {stats['deff_at_mean']:.3f} "
                  f"max {stats['deff_at_max']:.3f}  "
                  f"p<0.05 in {stats['n_significant_at_5pct']}")
    for population in ("family_real_rows_unambiguous",
                       "family_chaff_rows_unambiguous"):
        print(f"\n{population} -- DIAGNOSTIC ONLY: membership is a frame-level "
              f"property, so\nthe selection inflates D (the frame-blind "
              f"control reads ~2.4 here against ~1 on scored rows)")
        for family, stats in report[population].items():
            print(f"  {family:<15} arms={stats['n']:>3} "
                  f"mean cluster {stats['mean_cluster_size']:>6.1f}  "
                  f"D_kish mean {stats['deff_mean']:.3f} "
                  f"median {stats['deff_median']:.3f} "
                  f"min {stats['deff_min']:.3f} max {stats['deff_max']:.3f}")


def verify_artifact(path: Path) -> int:
    """Recompute every design effect from the artifact's own frame counts.

    The .pt prediction dumps are gitignored and live only on the runner host,
    so each artifact carries the per-frame correct counts the estimators were
    fed, as whitespace-separated integers.  This re-derives ICC and both
    design effects from those counts alone and refuses any disagreement,
    which makes the measurement checkable from committed data with no bundle,
    no corpus and no torch.
    """
    report = json.loads(path.read_text())
    if report.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: not a {SCHEMA} artifact")
    populations = (("all_rows", "correct", report["rows_per_frame"]),
                   ("real_rows_lo", "real_correct_lo", report["real_per_frame"]),
                   ("real_rows_hi", "real_correct_hi", report["real_per_frame"]))
    failures = 0
    for arm in report["arms"] + [report["control"]]:
        for key, series, size in populations:
            if key not in arm:
                continue
            counts = np.array(arm["frame_counts"][series].split(),
                              dtype=float)
            rho = float(anova_rho(counts, size))
            for name, got in (("icc_anova", rho),
                              ("deff_anova", 1.0 + (size - 1) * rho),
                              ("deff_kish", float(kish_deff(counts, size)))):
                if abs(arm[key][name] - got) > 1e-9:
                    print(f"  [FAIL] {arm['model']}/{arm['restart']} {key} "
                          f"{name}: artifact {arm[key][name]} recomputed {got}")
                    failures += 1
    print(f"  [{'PASS' if not failures else 'FAIL'}] {report['cell']}: "
          f"{len(report['arms']) + 1} arms re-derived from frame counts")
    return 1 if failures else 0


# self-test

def _self_test_estimators() -> list[bool]:
    """Three analytic cluster patterns with hand-derivable design effects."""
    size, k = 8, 64
    perfect = np.array([size] * (k // 2) + [0] * (k // 2), dtype=float)
    rho = float(anova_rho(perfect, size))
    kish = float(kish_deff(perfect, size))
    uniform = np.full(k, 3.0)
    rho_uniform = float(anova_rho(uniform, size))
    checks = [
        abs(rho - 1.0) < 1e-12,
        abs(1.0 + (size - 1) * rho - size) < 1e-12,
        abs(kish - size * k / (k - 1.0)) < 1e-12,
        abs(rho_uniform + 1.0 / (size - 1)) < 1e-12,
        abs(float(kish_deff(uniform, size))) < 1e-12,
    ]
    equal = np.full(k, float(size))
    checks.append(abs(float(kish_deff_ragged(perfect, equal))
                      - float(kish_deff(perfect, size))) < 1e-12)
    long_range = cluster_scale_ladder(perfect, size, (1, 2))
    alternating = np.array([size, 0] * (k // 2), dtype=float)
    confined = cluster_scale_ladder(alternating, size, (1, 2))
    checks.append(abs(long_range[1]["deff_anova"] - 2 * size) < 1e-12)
    checks.append(abs(confined[0]["deff_anova"] - size) < 1e-12
                  and abs(confined[1]["deff_anova"]) < 1e-12)
    signs = np.where(np.arange(k) < k // 2, 1.0, -1.0)[:, None]
    clustered_score = np.repeat(signs, size, axis=1)
    balanced = np.tile(np.array([1.0, -1.0]), (k, size // 2))
    checks.append(abs(deff_mean(clustered_score)
                      - (k * size - 1.0) / (k - 1.0)) < 1e-9)
    checks.append(abs(deff_mean(balanced)) < 1e-12)
    names = ["perfect clustering: ICC = 1", "perfect clustering: D = m",
             "perfect clustering: D_kish = m k/(k-1)",
             "uniform frames: ICC = -1/(m-1)", "uniform frames: D_kish = 0",
             "ragged estimator reduces to the equal-size one",
             "cross-frame dependence: merged D climbs to 2m",
             "frame-confined dependence: merged D collapses to 0",
             "non-binary score, cluster-constant: D = (n-1)/(k-1)",
             "non-binary score, within-cluster balanced: D = 0"]
    for name, ok in zip(names, checks):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return checks


def _self_test_independence() -> list[bool]:
    """Independent rows must return D ~ 1, with the CI covering 1."""
    rng = np.random.default_rng(4242)
    size, k = 80, 256
    counts = rng.binomial(size, 0.05, size=k).astype(float)
    stats = cluster_stats(counts, size, 2000, 7, 500)
    checks = [abs(stats["deff_anova"] - 1.0) < 0.25,
              stats["icc_ci95"][0] <= 0.0 <= stats["icc_ci95"][1],
              abs(stats["deff_kish"] - stats["deff_anova"]) < 0.25,
              abs(stats["deff_null_median"] - 1.0) < 0.05,
              stats["null_p_value"] > 0.01]
    names = ["independent rows: D_anova within 0.25 of 1",
             "independent rows: ICC 95% interval covers 0",
             "independent rows: the two estimators agree",
             "exchangeable null centres on D = 1",
             "independent rows: null p-value is not significant"]
    for name, ok in zip(names, checks):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return checks


def _self_test_known_icc() -> list[bool]:
    """A beta-binomial with a set rho must be recovered inside its interval."""
    rng = np.random.default_rng(99)
    size, k, p, rho = 80, 4000, 0.06, 0.15
    scale = (1.0 - rho) / rho
    counts = rng.binomial(size, rng.beta(p * scale, (1 - p) * scale,
                                         size=k)).astype(float)
    stats = cluster_stats(counts, size, 2000, 11, 500)
    checks = [stats["icc_ci95"][0] <= rho <= stats["icc_ci95"][1],
              stats["null_p_value"] == 0.0]
    print(f"  [{'PASS' if checks[0] else 'FAIL'}] beta-binomial rho=0.15 "
          f"inside the bootstrap interval ({stats['icc_anova']:.4f})")
    print(f"  [{'PASS' if checks[1] else 'FAIL'}] beta-binomial rejected by "
          f"the exchangeable null (p={stats['null_p_value']:.3f})")
    return checks


def _self_test_real_bounds() -> list[bool]:
    """The forensics attribution bound, on one hand-checked frame.

    Rows a,a,a,b,c with a,a,b real: 'a' is 2 real + 1 chaff, 'b' real only,
    'c' chaff only.  Two 'a' rows and the 'c' row score, so the real share of
    the three hits is at least 2-1=1 and at most min(2,2)=2.
    """
    labels = np.array([[0, 0, 0, 1, 2]])
    correct = np.array([[True, True, False, False, True]])
    parts = [frame_partition([0, 0, 0, 1, 2], [0, 0, 1])]
    lo, hi = real_correct_bounds(labels, parts, correct)
    ok = (lo[0], hi[0]) == (1.0, 2.0)
    print(f"  [{'PASS' if ok else 'FAIL'}] real/chaff attribution bounds "
          f"[{lo[0]:.0f},{hi[0]:.0f}]")
    return [ok]


def self_test() -> int:
    checks = (_self_test_estimators() + _self_test_independence()
              + _self_test_known_icc() + _self_test_real_bounds())
    return 0 if all(checks) else 1


# main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", help="per-row prediction .pt from the attacker")
    ap.add_argument("--bundle", help="the retained bundle it was scored on")
    ap.add_argument("--rescore-json", help="the re-score that wrote the dump")
    ap.add_argument("--recorded-attacker-json",
                    help="the cell's committed *_attacker.json")
    ap.add_argument("--run-json", help="the cell's committed run *.json")
    ap.add_argument("--model", help="tokenizer/model directory")
    ap.add_argument("--corpus", help="corpus the cell was run on")
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_DRAWS)
    ap.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--null-draws", type=int, default=NULL_DRAWS,
                    help="permutation draws for the exchangeable-row null")
    ap.add_argument("--ladder-null-draws", type=int, default=LADDER_NULL_DRAWS,
                    help="permutation draws per cluster-scale rung")
    ap.add_argument("--sweep", help="re-judge every artifact under this root")
    ap.add_argument("--measurements", help="directory of *_design_effect.json")
    ap.add_argument("--extrapolate-icc", type=float,
                    help="apply this ICC to uncovered cells, labelled as an "
                         "extrapolation")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="skip artifact paths containing any of these "
                         "substrings (concurrent work under the same root)")
    ap.add_argument("--verify", nargs="+",
                    help="re-derive one or more measurement artifacts from "
                         "their own committed frame counts")
    ap.add_argument("--output")
    ap.add_argument("--self-test", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return self_test()
    if args.verify:
        return max(verify_artifact(Path(p)) for p in args.verify)
    if args.sweep:
        if not args.measurements:
            raise SystemExit("--sweep needs --measurements")
        report = sweep(Path(args.sweep),
                       load_measurements(Path(args.measurements)),
                       args.extrapolate_icc, args.exclude)
        print_sweep(report)
    else:
        for name in ("dump", "bundle", "rescore_json",
                     "recorded_attacker_json", "run_json", "model", "corpus"):
            if not getattr(args, name):
                raise SystemExit(f"--{name.replace('_', '-')} is required")
        report = measure_bundle(args)
        print(json.dumps(report["gate"], indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
