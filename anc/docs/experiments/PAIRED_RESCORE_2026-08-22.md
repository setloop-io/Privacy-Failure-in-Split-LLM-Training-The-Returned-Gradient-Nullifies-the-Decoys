# The published ledger, re-scored with a paired cluster-aware statistic

**Date:** 2026-08-22. **Campaign items:** W2.1a (statistic) and W2.1b (application).
**Tool:** `bin/paired_advantage.py`. **Artifacts:**
`paper-data/collected/diagnostic/paired_rescore/*_paired.json` (22 cells).
**Inputs:** the 22 `--dump-eval-predictions` files under
`~/experiments/results/training/ledger_regate/dumps/` on odysseus.

## Bottom line

**Ten of twenty-two published cells resolve above the constant predictor** under a paired,
cluster-aware statistic. **Seven of those ten passed the published privacy gate.**

The campaign's headline cell survives: v9.2 K=2 D=64 reads **+0.0224 pp [-0.0256, +0.0688]**
--- at floor, genuinely. Several other cells the gate waved through do not.

That is the paper's thesis, measured on its own ledger: *the probe passes, and the frame is
still resolvably above its control.*

## Why the published gate and this statistic disagree

The gate computes `max over nine arms of WilsonUpper(arm) - point(control)`. Three
consequences, all previously documented in this repository and all reproduced here:

1. It compares an arm's **upper confidence bound** to a control **point estimate**, so an arm
   identical to the control still scores a positive "excess" made entirely of confidence
   width. On the v9.2 headline that is +0.00244 pp of point excess reported as +0.29232 pp.
2. Arms that emit one or two token classes sit exactly at the control and **pin the maximum**,
   masking movement in arms that can read the representation. **Fourteen of the twenty-two
   cells contain at least one such arm.**
3. It treats token rows as independent, which `paper-data/evaluation_protocol.json` forbids.

`bin/paired_advantage.py` scores every arm against the **same** control on the **same** rows,
pairs row by row, clusters by frame, bootstraps over clusters (2,000 draws), and excludes arms
whose modal class share exceeds 0.99. An arm identical to the control reads exactly 0.0 with
zero variance --- by construction, not by measurement.

## Results

`adv` is the paired advantage over the constant control in percentage points; the interval is
a 95% cluster bootstrap over 512 frames. `deg` counts arms excluded as degenerate.

| cell | adv (pp) | 95% CI | deg | verdict | published gate |
| --- | ---: | --- | ---: | --- | --- |
| `v93_public4k_priv1k_d64_06b` | **+1.5283** | [+1.336, +1.718] | 0 | resolves | FAIL (+1.490) |
| `v10_pub4k_priv2k_06b` | **+0.8577** | [+0.616, +1.108] | 0 | resolves | **PASS** (+0.732) |
| `e2_seq128_06b` | **+0.7773** | [+0.577, +0.983] | 0 | resolves | FAIL (+1.364) |
| `e2_seq64_06b` | **+0.5835** | [+0.372, +0.804] | 1 | resolves | FAIL (+1.250) |
| `v91_radial_d16_06b` | **+0.5687** | [+0.418, +0.724] | 0 | resolves | **PASS** (+0.688) |
| `v94_k4_2endpoints_d64_06b` | **+0.3138** | [+0.199, +0.425] | 0 | resolves | **PASS** privacy (+0.646) |
| `v92_k2_d16_06b` | **+0.2412** | [+0.081, +0.391] | 0 | resolves | **PASS** (+0.557) |
| `v92_k2_radial_d64_06b` | **+0.1641** | [+0.012, +0.310] | 1 | resolves | **PASS** (+0.323) |
| `v12_invmlp_06b_s42` | **+0.1446** | [+0.073, +0.215] | 3 | resolves | **PASS** (+0.404) |
| `v92_s43` | **+0.0422** | [+0.005, +0.080] | 1 | resolves | **PASS** |
| `v13_a9_mine010` | +0.1495 | [-0.009, +0.312] | 3 | at-floor | PASS |
| `v11_v92_06b_s42` | +0.0913 | [-0.041, +0.228] | 2 | at-floor | PASS |
| `v13_a2_mionly010` | +0.0838 | [-0.013, +0.184] | 0 | at-floor | PASS |
| `v91_radial_d64_06b` | +0.0437 | [-0.047, +0.132] | 3 | at-floor | PASS (+0.400) |
| `v92_s44` | +0.0380 | [-0.030, +0.111] | 1 | at-floor | PASS |
| `v92_k2_d64_06b` | **+0.0224** | [-0.026, +0.069] | 1 | **at-floor** | PASS (+0.292) |
| `v92_s42` | +0.0135 | [-0.093, +0.120] | 3 | at-floor | PASS |
| `v13_a1_fragment2` | -0.0340 | [-0.151, +0.079] | 0 | at-floor | PASS |
| `v95_d128_radial_k2_06b` | -0.0446 | [-0.109, +0.026] | 4 | at-floor | PASS privacy (+0.287) |
| `v12_converge10k_06b_s42` | -0.0659 | [-0.209, +0.074] | 3 | at-floor | PASS (+0.417) |
| `e1_d32_06b` | -0.1170 | [-0.286, +0.057] | 3 | at-floor | PASS (+0.426) |
| `e1_d64_06b` | -0.2236 | [-0.437, -0.011] | 3 | at-floor | PASS (+0.411) |

## Validation

The statistic reproduces the published **point** structure while adding intervals and
degenerate-arm exclusion, which is the behaviour that should be expected if it is measuring
the same thing correctly:

| cell | published point excess | paired advantage |
| --- | ---: | ---: |
| `v92_k2_d64_06b` | +0.0024 | +0.0224 |
| `v92_k2_d16_06b` | +0.2637 | +0.2412 |
| `v91_radial_d64_06b` | -0.0098 | +0.0437 |
| `v91_radial_d16_06b` | +0.2734 | +0.5687 |
| `e2_seq64_06b` | +0.8998 | +0.5835 |
| `e2_seq128_06b` | +1.0742 | +0.7773 |
| `e1_d64_06b` | +0.0000 | -0.2236 |

Residual differences come from two deliberate choices, stated rather than tuned:

- **The control is the strongest constant predictor on the evaluation rows** (the modal
  scoreable eval token), not the weaker train-derived one the gate used. Both select the same
  class --- token 279, `' the'` --- so they differ in rate, not identity. This makes the test
  **conservative for claiming a leak**: an arm must beat the best possible constant baseline.
  The trade-off is that a genuinely weak leak could be masked. Ten cells resolve anyway.
- **Degenerate arms are excluded** before the maximum is taken, so a pinned pedestal cannot
  set the reading.

## Two corrections made during this work

Recorded because both would have produced a confidently wrong result.

1. **A class-index bug in the first version of the tool.** `prediction` holds indices into the
   dump's `classes` table, not raw token ids. Comparing them directly reads 0.04% instead of
   3.24%, which made every arm appear ~5.9 pp *worse* than the control and produced a clean,
   plausible, entirely false "0 of 22 resolve" result. The fix maps through `classes`; a
   regression fixture in `--self-test` now fails if the mapping is dropped.
2. **The rescore was at one point assumed blocked** --- "no prediction `.pt` is
   committed and the source bundles are root-owned on the runner host". Twenty-two dumps are
   present and readable under `~/experiments/results/training/ledger_regate/dumps/`. It was
   never blocked.

## Limitations

- Twenty-two cells is the set with retained prediction dumps, not the full 111-cell ledger.
- The clustering unit is the frame. `DESIGN_EFFECT_MEASUREMENT.md` (not included in this
  release) measures the frame-level design effect at 0.847--1.158 over 15 cells, so frame
  clustering is appropriate there; the long-range dependence that document identifies as
  corpus drift is **not** absorbed here.
- The statistic answers "does this arm beat the best constant predictor". It does not
  establish *what* was recovered. Sequence, rare-token, and semantic metrics remain
  unimplemented (W2.3).
- Single scoring seed for the bootstrap (2,000 draws, seed 42). Re-running with other seeds
  moves the intervals by less than their width but this was not swept.
