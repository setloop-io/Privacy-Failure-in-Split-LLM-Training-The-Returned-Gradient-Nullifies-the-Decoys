#!/usr/bin/env python3
"""E1: seed-level hierarchical estimate of the
gradient-channel leak over every committed seed.

Three provenance groups, twelve seeds total:
  original    -- the three dirty-tree seeds (RELAYED; archived source of record)
  packaged    -- the six W1.7 packaged-code seeds
  postfreeze  -- the three W5.7 post-freeze confirmation seeds

Model: normal-normal random-effects meta-analysis (DerSimonian-Laird) over the
per-seed paired advantages, with per-seed standard errors recovered from the
committed 95% cluster-bootstrap CIs. Reports the grand mean and CI, between-seed
variance tau^2, I^2 heterogeneity, per-group means, and leave-one-out influence
(the seed-42 out-of-band question). Pure statistics over committed artifacts;
no new runs.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("paper-data/collected/diagnostic")


def ci_to_se(ci):
    return (ci[1] - ci[0]) / (2 * 1.959964)


def load_seeds() -> list[dict]:
    seeds = []
    # original three (dirty tree, RELAYED): values from the E1 report table
    # (paired advantage + CI), cross-checked against e1_unprotected/.
    orig = [(0.7945, (0.554334, 1.045107)),
            (0.8064, (0.546, 1.059)),
            (0.7726, (0.523, 1.026))]
    for i, (est, ci) in enumerate(orig):
        seeds.append({"group": "original", "seed": f"orig_{i}",
                      "paired_pp": est, "se": ci_to_se(ci)})
    six = json.loads((ROOT / "e1_reproduction_w12/e1_packaged_sixseed_summary.json")
                     .read_text())["draws"]
    for d in six:
        arm = d["arms"]["grad_real"]
        seeds.append({"group": "packaged", "seed": f"s{d['seed']}",
                      "paired_pp": arm["paired_pp"], "se": ci_to_se(arm["ci95"])})
    w57 = json.loads((ROOT / "e1_confirmation/w57_confirmation_summary.json")
                     .read_text())["seeds"]
    for d in w57:
        arm = d["grad_real"]
        seeds.append({"group": "postfreeze", "seed": f"s{d['seed']}",
                      "paired_pp": arm["paired_pp"], "se": ci_to_se(arm["ci95"])})
    return seeds


def random_effects(seeds: list[dict]) -> dict:
    import statistics as st
    y = [s["paired_pp"] for s in seeds]
    v = [s["se"] ** 2 for s in seeds]
    w = [1.0 / x for x in v]
    fixed = sum(a * b for a, b in zip(w, y)) / sum(w)
    q = sum(a * (b - fixed) ** 2 for a, b in zip(w, y))
    df = len(y) - 1
    c = sum(w) - sum(x * x for x in w) / sum(w)
    tau2 = max(0.0, (q - df) / c)
    wr = [1.0 / (x + tau2) for x in v]
    mean = sum(a * b for a, b in zip(wr, y)) / sum(wr)
    se = (1.0 / sum(wr)) ** 0.5
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 else 0.0
    return {"mean_pp": mean, "ci95": [mean - 1.959964 * se, mean + 1.959964 * se],
            "tau2": tau2, "tau_pp": tau2 ** 0.5, "I2_pct": i2,
            "Q": q, "fixed_effect_mean_pp": fixed,
            "median_pp": st.median(y), "min_pp": min(y), "max_pp": max(y)}


def main() -> int:
    seeds = load_seeds()
    overall = random_effects(seeds)
    groups = {}
    for g in ("original", "packaged", "postfreeze"):
        sub = [s for s in seeds if s["group"] == g]
        groups[g] = {"n": len(sub), **random_effects(sub)}
    loo = {}
    for s in seeds:
        sub = [x for x in seeds if x is not s]
        loo[s["seed"]] = random_effects(sub)["mean_pp"]
    verified = [s for s in seeds if s["group"] != "original"]
    verified_only = {"n": len(verified),
                     "seeds": [s["seed"] for s in verified],
                     **random_effects(verified)}
    out = {
        "schema": "dtraining.e1_hierarchical_estimate.v2",
        "estimand": "grad_real paired token_top1 advantage over the constant "
                    "baseline, pp",
        "method": "normal-normal random effects (DerSimonian-Laird); per-seed SE "
                  "from committed 95% cluster-bootstrap CIs",
        "n_seeds": len(seeds),
        "seeds": seeds,
        "overall": overall,
        "verified_only": verified_only,
        "by_group": groups,
        "leave_one_out_mean_pp": loo,
        "provenance_note": "original group is RELAYED (dirty-tree source of "
                           "record); packaged and postfreeze are VERIFIED from "
                           "packaged code. 'verified_only' pools the nine "
                           "VERIFIED seeds and is the figure cited by the "
                           "manuscript; 'overall' retains all twelve for "
                           "continuity with earlier reports.",
    }
    outdir = ROOT / "e1_hierarchical"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "e1_hierarchical_estimate.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"n={len(seeds)} seeds; grand mean {overall['mean_pp']:.4f} pp "
          f"CI [{overall['ci95'][0]:.4f}, {overall['ci95'][1]:.4f}]; "
          f"tau {overall['tau_pp']:.4f}; I2 {overall['I2_pct']:.1f}%")
    v = verified_only
    print(f"verified-only n={v['n']}; mean {v['mean_pp']:.4f} pp "
          f"CI [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]; "
          f"tau {v['tau_pp']:.4f}; I2 {v['I2_pct']:.1f}%")
    for g, r in groups.items():
        print(f"  {g:<10} n={r['n']}  mean {r['mean_pp']:.4f}  "
              f"[{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]")
    print("leave-one-out spread:",
          f"[{min(loo.values()):.4f}, {max(loo.values()):.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
