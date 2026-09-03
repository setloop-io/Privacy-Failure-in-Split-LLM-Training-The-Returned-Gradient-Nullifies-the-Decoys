#!/usr/bin/env python3
"""Why neither a bigger surrogate nor a wider channel moved the utility bound.

The boundary is a power-limited Gaussian channel, and it is already saturated.

From privacy_runtime/activation_dp.py:97-103, each released row is clipped to
L2 norm <= max_norm and then has iid Gaussian noise added per coordinate:

    clipped   = row * min(1, max_norm / ||row||)
    protected = clipped + N(0, (noise_multiplier * max_norm)^2) per coordinate

So, per released row:

    signal power = ||clipped||^2 <= max_norm^2        INDEPENDENT of D
    noise  power = D * (noise_multiplier*max_norm)^2  LINEAR in D

This is the wideband regime of an AWGN channel: adding coordinates does not add
signal, because the clip caps total signal power no matter how many dimensions
carry it. Capacity per row is

    C(D) = (D/2) * ln(1 + S / (D * sigma^2))       nats

which increases in D but converges to the power-limited asymptote

    C_inf = S / (2 * sigma^2)

and -- the part that matters for planning -- sigma = noise_multiplier*max_norm,
so S/(2 sigma^2) = max_norm^2 / (2 * noise_multiplier^2 * max_norm^2)
                 = 1 / (2 * noise_multiplier^2).

**max_norm cancels.** Raising the clip raises signal and noise together. The
asymptotic capacity of this boundary is a function of the noise multiplier
alone, and the noise multiplier is the privacy parameter.

The script prints the predicted capacity curve beside the measured surrogate
closure so the prediction can be checked rather than asserted.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

CLIP = 1.0
NOISE_MULT = 0.35


def capacity_nats(d: int, clip: float, noise_mult: float) -> float:
    signal = clip ** 2
    sigma_sq = (noise_mult * clip) ** 2
    return (d / 2.0) * math.log(1.0 + signal / (d * sigma_sq))


def asymptote_nats(noise_mult: float) -> float:
    return 1.0 / (2.0 * noise_mult ** 2)


def measure(path: Path) -> dict:
    run = json.loads(path.read_text())
    base = run["baseline_eval_loss"]
    gap = run["zero_cloud_eval_loss"] - base
    closed = run["zero_cloud_eval_loss"] - run["candidate_eval_loss"]
    return {
        "latent_dim": run["latent_dim"],
        "cloud_experts": run["cloud_experts"],
        "cloud_layers": run["cloud_layers"],
        "train_blocks": run["train_blocks"],
        "steps": run["steps"],
        "noise_multiplier": run["noise_multiplier"],
        "clip_norm": run["dp"]["parameters"]["forward_clip"],
        "dp_epsilon_composed": run["dp"]["epsilon"]["composed"],
        "fraction_closed": closed / gap,
        "residual_nats": run["candidate_eval_loss"] - base,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cells", nargs="+", required=True,
                    help="runner artifact basenames, without .json")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.dir)
    cells = [measure(root / f"{name}.json") for name in args.cells]

    asym = asymptote_nats(NOISE_MULT)
    print(f"Power-limited asymptote  C_inf = 1/(2*noise^2) "
          f"= {asym:.3f} nats/row  (noise_multiplier={NOISE_MULT}, "
          f"independent of clip_norm)")
    print()
    h = (f"{'D':>5}{'experts':>8}{'layers':>7}{'blocks':>7}{'steps':>7} | "
         f"{'C(D) nats':>10}{'% of C_inf':>11} | {'closure':>8}{'residual':>9} | "
         f"{'DP epsilon':>13}")
    print(h)
    print("-" * len(h))
    rows = []
    for c in cells:
        cap = capacity_nats(c["latent_dim"], c["clip_norm"], c["noise_multiplier"])
        rows.append({**c, "capacity_nats": cap, "pct_of_asymptote": 100 * cap / asym})
        print(f"{c['latent_dim']:>5}{c['cloud_experts']:>8}{c['cloud_layers']:>7}"
              f"{c['train_blocks']:>7}{c['steps']:>7} | {cap:>10.3f}"
              f"{100 * cap / asym:>10.1f}% | {100 * c['fraction_closed']:>7.1f}%"
              f"{c['residual_nats']:>+9.4f} | {c['dp_epsilon_composed']:>13.2f}")

    print()
    print("What raising the noise budget would buy, if privacy allowed it:")
    print(f"{'noise':>8}{'C_inf nats':>12}{'vs 0.35':>10}")
    for nm in (0.35, 0.30, 0.25, 0.20):
        a = asymptote_nats(nm)
        print(f"{nm:>8.2f}{a:>12.3f}{a / asym:>9.2f}x")

    dims = sorted({c["latent_dim"] for c in cells})
    # Compare epsilon only among cells that differ in D alone. Release count
    # scales with train_blocks and steps, and epsilon scales with releases, so
    # pooling cells with different windows would compare the wrong thing.
    same_window = [c for c in cells
                   if (c["train_blocks"], c["steps"]) ==
                   (cells[0]["train_blocks"], cells[0]["steps"])]
    eps = sorted({round(c["dp_epsilon_composed"], 6) for c in same_window})
    report = {
        "schema": "dtraining.deleg6040.channel_capacity.v1",
        "model": ("per-row AWGN, signal capped by the L2 clip and independent "
                  "of D, noise linear in D "
                  "(privacy_runtime/activation_dp.py:97-103)"),
        "asymptote_nats": asym,
        "asymptote_depends_only_on_noise_multiplier": True,
        "rows": rows,
        "dp_epsilon_identical_across_latent_dim": len(eps) == 1,
        "latent_dims_compared": dims,
        "epsilon_comparison_scope": "cells sharing train_blocks and steps",
        "n_cells_in_epsilon_comparison": len(same_window),
        "capacity_gain_64_to_128_pct": (
            100 * (capacity_nats(128, CLIP, NOISE_MULT)
                   / capacity_nats(64, CLIP, NOISE_MULT) - 1)
            if {64, 128} <= set(dims) else None),
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print()
    print(f"DP epsilon identical across the {len(same_window)} cells that share "
          f"a window and differ only in D: "
          f"{report['dp_epsilon_identical_across_latent_dim']} "
          f"-- the accountant records noise_multiplier and release count only, "
          f"never D, so widening the channel costs SNR and buys nothing in the "
          f"formal accounting either.")
    if report["capacity_gain_64_to_128_pct"] is not None:
        print(f"predicted capacity gain, D 64 -> 128: "
              f"{report['capacity_gain_64_to_128_pct']:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
