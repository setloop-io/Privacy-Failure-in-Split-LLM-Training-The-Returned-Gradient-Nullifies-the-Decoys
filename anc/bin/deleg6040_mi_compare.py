#!/usr/bin/env python3
"""AC2.3: is the boundary activation at layer 14 leakier than at layer 21?

The D1 result only isolates delegation share if moving the boundary earlier did
not simply hand the bottleneck a more token-recoverable hidden state. The
published per-layer curve says it did not -- the curve is not monotone in depth
and layer 14 sits *below* layer 21 -- but that curve was measured on a different
stack, so this re-measures it here.

The layer index means the same thing in both tools: the published per-layer
curve (its generator is not included in this release) probes
run_layer_stack(layers[: upto + 1], ...) and
bin/run_latent_native_v5_06b.py:361 builds the boundary activation with
run_layer_stack(layers[:args.split_after + 1], ...) -- the same expression. So
"layer N" here IS split_after = N, read directly with no interpolation.

An inversion (layer 14 above layer 21 on this stack) would put the confound back
and is reported as such rather than smoothed over.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

LAYERS = (2, 8, 14, 21)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--published", required=True)
    ap.add_argument("--measured", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pub = {r["layer"]: r for r in
           json.loads(Path(args.published).read_text())["results"]}
    new = {r["layer"]: r for r in
           json.loads(Path(args.measured).read_text())["results"]}

    rows = []
    for layer in LAYERS:
        rows.append({
            "layer": layer,
            "published_top1_pct": pub[layer]["top1_pct"],
            "measured_top1_pct": new[layer]["top1_pct"],
            "delta_pp": new[layer]["top1_pct"] - pub[layer]["top1_pct"],
            "measured_probe_ce_nats": new[layer]["probe_ce_nats"],
            "measured_mi_lower_bound_bits": new[layer]["mi_lower_bound_bits_per_token"],
        })

    pub_delta = pub[14]["top1_pct"] - pub[21]["top1_pct"]
    new_delta = new[14]["top1_pct"] - new[21]["top1_pct"]

    # token_prior_entropy_nats is a property of the corpus window, not of the
    # layer: it is identical at every layer within a run. Matching the published
    # value is the strongest single check that the same corpus window was used.
    entropy_match = all(
        abs(new[l]["token_prior_entropy_nats"]
            - pub[l]["token_prior_entropy_nats"]) < 1e-12 for l in LAYERS)

    report = {
        "schema": "dtraining.deleg6040.mi_compare.v1",
        "convention": ("mi_budget layer N == split_after N; both tools use "
                       "run_layer_stack(layers[: N + 1], ...) "
                       "(measure_mi_budget.py:83, run_latent_native_v5_06b.py:361)"),
        "rows": rows,
        "published_layer14_minus_layer21_pp": pub_delta,
        "measured_layer14_minus_layer21_pp": new_delta,
        "ordering_reproduces": (pub_delta < 0) == (new_delta < 0),
        "layer14_less_recoverable_than_layer21": new_delta < 0,
        "corpus_window_matches_published": entropy_match,
        "token_prior_entropy_nats": new[LAYERS[0]]["token_prior_entropy_nats"],
        "mi_lower_bound_vacuous_everywhere": all(
            r["measured_mi_lower_bound_bits"] == 0.0 for r in rows),
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")

    print(f"{'layer':>6} {'published %':>13} {'this stack %':>14} {'delta pp':>10}")
    for r in rows:
        print(f"{r['layer']:>6} {r['published_top1_pct']:>13.4f} "
              f"{r['measured_top1_pct']:>14.4f} {r['delta_pp']:>+10.4f}")
    print()
    print(f"published  layer14 - layer21 : {pub_delta:+.4f} pp")
    print(f"this stack layer14 - layer21 : {new_delta:+.4f} pp")
    print(f"ordering reproduces          : {report['ordering_reproduces']}")
    print(f"layer14 LESS recoverable     : "
          f"{report['layer14_less_recoverable_than_layer21']}")
    print(f"corpus window matches        : {entropy_match} "
          f"(prior entropy {report['token_prior_entropy_nats']:.12f} nats)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
