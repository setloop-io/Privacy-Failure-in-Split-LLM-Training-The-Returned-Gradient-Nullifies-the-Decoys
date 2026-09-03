#!/usr/bin/env python3
"""Turn a captured outbound backward wire into frozen-attacker bundles.

The capture comes from `bin/run_latent_native_v5_06b.py --grad-channel-bundle`
and holds, per recorded training step, the exact bytes that crossed the trust
boundary in BOTH directions plus the wire-order honest labels:

  wire     the released forward frame (TLN -> UCN, DP clip+noise applied
           inside tln.encode, privacy_runtime/latent_native.py:474)
  grad     the output gradient (TLN -> UCN, run_latent_native_v5_06b.py:660;
           no clip, no noise, not accounted when --wire-quant none)

Each emitted bundle uses the SAME schema the committed gate consumes, so it can
be scored by the unmodified nine-arm attacker:

  python3 -m attacker --attack latent-probe --bundle <arm>.pt --output <arm>.json

Arms:
  wire_all   forward frame, every wire row      (matched control for grad_all)
  grad_all   output gradient, every wire row
  grad_real  output gradient, rows its own zero support marks as real tokens
  wire_real  forward frame, de-chaffed by the gradient's zero support
  joint_real forward frame and output gradient of the same step, concatenated
  joint_real_scaled  as joint_real, gradient block lifted to the frame's row scale
  xgxt_real  cosine cross-Gram XG^T over the real rows (W4.2 cross-tensor
             invariant: rotation and both scale gauges cancel)
  xgxt_real_shuffled  xgxt_real with globally permuted labels (negative control)
  grad_shuffled  grad_all with globally permuted labels (negative control)
  grad_real_shuffled  grad_real with globally permuted labels (negative control)

Train/evaluation partitions are disjoint corpus blocks: block index
`step % train_blocks`, first half train, second half evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

BUNDLE_SCHEMA = "dtraining.latent_release_bundle.v1"
CAPTURE_SCHEMA = "dtraining.outbound_grad_channel.v1"


def load_capture(path: Path) -> dict:
    capture = torch.load(path, map_location="cpu")
    if capture.get("schema") != CAPTURE_SCHEMA:
        raise ValueError(f"{path} is not a {CAPTURE_SCHEMA} capture")
    for key in ("wire", "grad", "tokens", "is_real", "frame_steps"):
        if key not in capture:
            raise ValueError(f"capture is missing {key}")
    if capture["wire"].shape != capture["grad"].shape:
        raise ValueError("wire and grad frames must have identical shape")
    if capture["wire"].shape[:2] != capture["tokens"].shape:
        raise ValueError("labels must be one per wire row")
    return capture


def support_report(capture: dict) -> dict:
    """Does the gradient's zero pattern disclose which rows are real tokens?"""
    grad, is_real = capture["grad"], capture["is_real"]
    nonzero = grad.abs().sum(dim=-1) > 0
    frames, rows = nonzero.shape
    per_frame = nonzero.sum(dim=1)
    exact = int((nonzero == is_real).all(dim=1).sum())
    return {
        "frames": frames,
        "rows_per_frame": rows,
        "real_rows_per_frame": int(is_real[0].sum()),
        "nonzero_rows_per_frame_min": int(per_frame.min()),
        "nonzero_rows_per_frame_max": int(per_frame.max()),
        "row_agreement_zero_support_vs_real": float(
            (nonzero == is_real).float().mean()),
        "frames_with_exact_agreement": exact,
        "frames_with_exact_agreement_pct": 100.0 * exact / frames,
    }


def row_norms(frames: torch.Tensor, mask: torch.Tensor) -> dict:
    norms = frames.norm(dim=-1)[mask]
    quantiles = torch.quantile(norms, torch.tensor([0.5, 0.95, 1.0]))
    return {"rows": int(norms.numel()), "min": float(norms.min()),
            "median": float(quantiles[0]), "p95": float(quantiles[1]),
            "max": float(quantiles[2]), "mean": float(norms.mean())}


def magnitude_report(capture: dict) -> dict:
    """Row magnitudes in both outbound directions.

    The forward frame is clipped to clip_norm inside tln.encode before the
    calibrated noise is added; nothing bounds the output gradient, so the two
    distributions are not expected to be comparable.
    """
    grad, wire = capture["grad"], capture["wire"]
    live = grad.abs().sum(dim=-1) > 0
    everything = torch.ones_like(live)
    return {"outbound_forward_frame_rows": row_norms(wire, everything),
            "outbound_output_gradient_rows": row_norms(grad, live)}


def select_rows(frames: torch.Tensor, mask: torch.Tensor,
                keep: int) -> torch.Tensor:
    """Gather exactly `keep` masked rows per frame, fail-fast on a short frame."""
    if int(mask.sum(dim=1).min()) < keep:
        raise ValueError("a frame has fewer selected rows than requested")
    index = torch.stack([row.nonzero(as_tuple=True)[0][:keep] for row in mask])
    return torch.gather(
        frames, 1, index.unsqueeze(-1).expand(-1, -1, frames.shape[-1]))


def gather_labels(tokens: torch.Tensor, mask: torch.Tensor,
                  keep: int) -> torch.Tensor:
    index = torch.stack([row.nonzero(as_tuple=True)[0][:keep] for row in mask])
    return torch.gather(tokens, 1, index)


def split_indices(capture: dict,
                  unique_blocks: bool = False) -> tuple[torch.Tensor,
                                                        torch.Tensor]:
    """Disjoint corpus-block halves; optionally one frame per block.

    A ring buffer longer than one pass over the corpus records some blocks
    twice, which leaves the evaluation rows correlated in pairs. --unique-blocks
    keeps the last frame of each block so every evaluation row is a distinct
    corpus row.
    """
    blocks = capture["frame_steps"] % int(capture["train_blocks"])
    keep = torch.ones_like(blocks, dtype=torch.bool)
    if unique_blocks:
        last = {int(block): index for index, block in enumerate(blocks)}
        keep = torch.zeros_like(blocks, dtype=torch.bool)
        keep[torch.tensor(sorted(last.values()))] = True
    half = int(capture["train_blocks"]) // 2
    train = ((blocks < half) & keep).nonzero(as_tuple=True)[0]
    evaluation = ((blocks >= half) & keep).nonzero(as_tuple=True)[0]
    if train.numel() == 0 or evaluation.numel() == 0:
        raise ValueError("recorded window does not cover both block halves")
    return train, evaluation


def write_bundle(path: Path, views: torch.Tensor, labels: torch.Tensor,
                 train: torch.Tensor, evaluation: torch.Tensor) -> dict:
    bundle = {
        "schema": BUNDLE_SCHEMA,
        "train_wire": views[train].contiguous(),
        "train_tokens": labels[train].contiguous(),
        "eval_wire": views[evaluation].contiguous(),
        "eval_tokens": labels[evaluation].contiguous(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    return {"path": str(path),
            "train_blocks": list(bundle["train_wire"].shape),
            "eval_blocks": list(bundle["eval_wire"].shape),
            "eval_rows": int(bundle["eval_tokens"].numel())}


def equalising_scale(wire: torch.Tensor, grad: torch.Tensor) -> float:
    """Constant that lifts gradient rows to the frame's median row norm.

    Plain concatenation is useless here: the frame's rows are dominated by the
    DP noise (median norm ~2.96) while gradient rows sit near 0.01, so the
    rotation-invariant features of the concatenation are numerically the
    frame's own. A single positive constant equalises the two blocks' influence
    without altering any direction or any relative structure within a block.
    """
    return float(wire.norm(dim=-1).median() / grad.norm(dim=-1).median())


def cross_gram(wire: torch.Tensor, grad: torch.Tensor, mask: torch.Tensor,
               keep: int, unit: bool = True) -> torch.Tensor:
    """Cross-Gram XG^T over the rows the zero support marks as real.

    The forward frame and the output gradient of one step carry the SAME
    per-request rotation R (and independent permutation/scale gauges), so
    wire @ grad^T cancels R. With unit=True (cosine form) the row/column scale
    gauges cancel too: entry (i, j) is cos(h_i, g_j) up to permutation. With
    unit=False the scale gauges survive (s_i s'_j h_i . g_j) --- weaker
    invariance, but it keeps the magnitude channel that the cosine form
    discards. (The naked-cell reading, +0.39 pp at floor where grad_real
    reads +21.40 pp, shows the cosine form alone is too weak there.)
    Output shape [frames, keep, keep]: row i's feature is its alignment
    against every real gradient row of the same step.
    """
    real_w = select_rows(wire, mask, keep)
    real_g = select_rows(grad, mask, keep)
    if unit:
        real_w = real_w / real_w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        real_g = real_g / real_g.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return real_w @ real_g.transpose(-1, -2)


def shuffle(labels: torch.Tensor, seed: int) -> torch.Tensor:
    """Globally permuted labels: the negative control for an arm."""
    generator = torch.Generator().manual_seed(seed)
    flat = labels.reshape(-1)
    return flat[torch.randperm(flat.numel(),
                               generator=generator)].reshape(labels.shape)


def build_arms(capture: dict, outdir: Path, seed: int,
               unique_blocks: bool = False) -> dict:
    train, evaluation = split_indices(capture, unique_blocks)
    wire, grad, tokens = capture["wire"], capture["grad"], capture["tokens"]
    nonzero = grad.abs().sum(dim=-1) > 0
    keep = int(capture["is_real"][0].sum())
    real_grad = select_rows(grad, nonzero, keep)
    real_wire = select_rows(wire, nonzero, keep)
    real_tokens = gather_labels(tokens, nonzero, keep)
    shuffled = shuffle(tokens, seed)
    # The forward frame and the output gradient of one step carry the SAME
    # per-request rotation, so their joint view is what the compromised node
    # actually holds; concatenating them keeps every cross-tensor invariant.
    arms = {
        "wire_all": (wire, tokens),
        "grad_all": (grad, tokens),
        "grad_real": (real_grad, real_tokens),
        "wire_real": (real_wire, real_tokens),
        "joint_real": (torch.cat([real_wire, real_grad], dim=-1), real_tokens),
        "joint_real_scaled": (torch.cat(
            [real_wire, real_grad * equalising_scale(real_wire, real_grad)],
            dim=-1), real_tokens),
        # W4.2: the cross-tensor invariant. Rotation-cancelling XG^T over the
        # real rows, cosine (scale-cancelling) and raw (scale-keeping) forms
        # --- the joint view the frozen family's within-view arms cannot reach.
        "xgxt_real": (cross_gram(wire, grad, nonzero, keep), real_tokens),
        "xgxt_real_shuffled": (cross_gram(wire, grad, nonzero, keep),
                               shuffle(real_tokens, seed + 2)),
        "xgxt_raw_real": (cross_gram(wire, grad, nonzero, keep, unit=False),
                          real_tokens),
        "xgxt_raw_real_shuffled": (cross_gram(wire, grad, nonzero, keep,
                                              unit=False),
                                   shuffle(real_tokens, seed + 3)),
        "grad_shuffled": (grad, shuffled),
        "grad_real_shuffled": (real_grad, shuffle(real_tokens, seed + 1)),
    }
    return {name: write_bundle(outdir / f"{name}.pt", views, labels,
                               train, evaluation)
            for name, (views, labels) in arms.items()}


# Required of every capture. Missing one means the capture is malformed, so the
# lookup is deliberately a hard failure.
CONFIG_KEYS = ("split_after", "steps", "train_blocks", "seq_len",
               "chaff_tokens", "wire_quant", "remote_grad_clip",
               "noise_multiplier")
# Written only by runners carrying the outbound-gradient DP leg (experiment W1.2).
# Absent in every capture recorded before it, so these are looked up softly.
# Without them the report cannot say whether the capture it summarizes was
# protected -- which is exactly the provenance this project is about.
OPTIONAL_CONFIG_KEYS = ("outbound_grad_dp", "grad_clip_norm",
                        "grad_noise_multiplier")


def capture_config(capture: dict) -> dict:
    """Scalar run configuration, tensors dropped, for the bundle report."""
    def scalar(value):
        return None if torch.is_tensor(value) else value

    config = {key: scalar(capture[key]) for key in CONFIG_KEYS}
    for key in OPTIONAL_CONFIG_KEYS:
        if key in capture:
            config[key] = scalar(capture[key])
    return config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unique-blocks", action="store_true",
                    help="keep one frame per corpus block so evaluation rows "
                         "are not duplicated across the recorded window")
    args = ap.parse_args()

    capture = load_capture(Path(args.capture))
    report = {
        "schema": "dtraining.grad_channel_bundles.v1",
        "capture": str(args.capture),
        "config": capture_config(capture),
        "support_leak": support_report(capture),
        "magnitudes": magnitude_report(capture),
        "unique_blocks": bool(args.unique_blocks),
        "bundles": build_arms(capture, Path(args.outdir), args.seed,
                              args.unique_blocks),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2,
                                            sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def self_test() -> int:
    """Round-trip a synthetic capture whose gradient support marks real rows."""
    frames, rows, width, real = 8, 10, 4, 6
    is_real = torch.zeros(frames, rows, dtype=torch.bool)
    for index in range(frames):
        is_real[index, torch.randperm(rows)[:real]] = True
    grad = torch.randn(frames, rows, width) * is_real.unsqueeze(-1)
    capture = {"schema": CAPTURE_SCHEMA, "wire": torch.randn(frames, rows,
                                                             width),
               "grad": grad, "tokens": torch.arange(frames * rows).reshape(
                   frames, rows), "is_real": is_real,
               "frame_steps": torch.arange(frames), "train_blocks": frames,
               "seq_len": real, "chaff_tokens": rows - real,
               "wire_quant": "none", "remote_grad_clip": 1.0,
               "noise_multiplier": 0.35, "split_after": 14, "steps": frames}
    leak = support_report(capture)
    ok = leak["frames_with_exact_agreement"] == frames
    nonzero = grad.abs().sum(dim=-1) > 0
    picked = gather_labels(capture["tokens"], nonzero, real)
    ok = ok and bool((picked == capture["tokens"][is_real].reshape(
        frames, real)).all())
    print(f"  [{'PASS' if ok else 'FAIL'}] zero support recovers the real rows")

    # W4.2: the cross-Gram must be invariant to the shared per-request
    # rotation and to both scale gauges, and must preserve the alignment
    # signal (a correlated h/g pair reads strongest on its own row).
    torch.manual_seed(0)
    frames2, keep2, width2 = 4, 6, 8
    h = torch.randn(frames2, keep2, width2)
    # a real output gradient is dominated by its own row's activation, so the
    # fixture mixes near-diagonally: cos(h_i, g_i) must be large
    mix = torch.eye(width2) + 0.1 * torch.randn(width2, width2)
    g = h @ mix + 0.01 * torch.randn(frames2, keep2, width2)
    rotation = torch.linalg.qr(torch.randn(width2, width2))[0]
    scale_w = torch.exp(0.75 * torch.randn(frames2, keep2, 1))
    scale_g = torch.exp(0.75 * torch.randn(frames2, keep2, 1))
    wire_plain = torch.zeros(frames2, keep2, width2)
    grad_plain = torch.zeros(frames2, keep2, width2)
    wire_gauged = torch.zeros(frames2, keep2, width2)
    grad_gauged = torch.zeros(frames2, keep2, width2)
    mask2 = torch.ones(frames2, keep2, dtype=torch.bool)
    wire_plain, grad_plain = h, g
    wire_gauged, grad_gauged = (scale_w * h) @ rotation, (scale_g * g) @ rotation
    xg_plain = cross_gram(wire_plain, grad_plain, mask2, keep2)
    xg_gauged = cross_gram(wire_gauged, grad_gauged, mask2, keep2)
    invariant = torch.allclose(xg_plain, xg_gauged, atol=1e-5)
    # the signal survives: row i's own pair (i, i) beats the off-diagonal mean
    diag = xg_gauged.diagonal(dim1=-2, dim2=-1).mean()
    offdiag = (xg_gauged.sum(dim=(-1, -2))
               - xg_gauged.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
               ) / (keep2 * (keep2 - 1))
    signal = bool(diag > offdiag.mean() + 3 * offdiag.std())
    # the within-view Gram still carries R: it must NOT match the plain one
    within_carries_r = not torch.allclose(
        wire_gauged @ wire_gauged.transpose(-1, -2),
        wire_plain @ wire_plain.transpose(-1, -2), atol=1e-4)
    ok2 = invariant and signal and within_carries_r
    print(f"  [{'PASS' if ok2 else 'FAIL'}] cross-Gram cancels the shared "
          f"rotation and both scale gauges, keeping the alignment signal")
    # raw form: rotation still cancels, but the scale gauges must SURVIVE
    xr_plain = cross_gram(wire_plain, grad_plain, mask2, keep2, unit=False)
    xr_gauged = cross_gram(wire_gauged, grad_gauged, mask2, keep2, unit=False)
    rotation_cancels_raw = torch.allclose(
        xr_plain, cross_gram(wire_plain @ rotation, grad_plain @ rotation,
                             mask2, keep2, unit=False), atol=1e-5)
    scale_survives_raw = not torch.allclose(xr_plain, xr_gauged, atol=1e-4)
    ok3 = rotation_cancels_raw and scale_survives_raw
    print(f"  [{'PASS' if ok3 else 'FAIL'}] raw cross-Gram cancels the "
          f"rotation but keeps the scale gauges")
    ok = ok and ok2 and ok3
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(self_test() if "--self-test" in sys.argv else main())
