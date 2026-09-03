# E1: the unprotected gradient channel on a forward-passing cell

**Date:** 2026-08-22. **Experiment:** E1 (campaign item W0.1).
**Cells:** `e1_unprot_a2b_split14` (run 2026-08-21, this work) and `gradfix_a2b_split14`
(run 2026-08-19, pre-existing).
**Artifacts:** `paper-data/collected/diagnostic/e1_unprotected/`,
`paper-data/collected/diagnostic/gradfix/`, `paper-data/collected/diagnostic/e1_matched_pair.json`.

## What was run, and why it is a matched pair

The threat-model audit left one question open: *does the gradient channel break a cell that
otherwise passes the gate?* It specified the experiment and did not run it, for time.

`gradfix_a2b_split14` looked at first like that experiment. It is not: it ran with
`--outbound-grad-dp clip_noise`, clip 0.01, noise 0.35 --- the outbound-gradient-DP
**fix**. Its null result measures the remediation working, not the exposure.

`e1_unprot_a2b_split14` supplies the missing half: the same a2b configuration, same seed,
same corpus, with `--outbound-grad-dp off` --- the setting the runner's own help text
describes as restoring "the unprotected, unaccounted backward wire every artifact before this
fix was produced with".

The setting is confirmed from the DP accounting, not from the command line:

| | E1 | gradfix |
| --- | --- | --- |
| `dp.releases` | `forward`, `return` | `forward`, **`gradient` (3,199,904)**, `return` |
| `dp.parameters` | no gradient entries | `gradient_clip 0.01`, `gradient_noise_multiplier 0.35` |

The cells otherwise track closely --- `train_seconds` 3,933.8 vs 4,205.1, `eval_loss_delta`
0.9185 vs 0.9358 --- so they differ in the intended variable and little else. Neither was
designed as half of a pair.

## Result

n = 65,536 evaluation rows per arm. `lift` is the point excess over the arm's own control;
`z` assumes independent rows and is shown only for comparability with published cells;
`paired` is the statistic of record (W2.1a), clustered by frame with a 2,000-draw bootstrap.

### E1 --- gradient unprotected

| arm | lift (pp) | z | Bonferroni u95 | paired (pp) | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `grad_real` | **+0.7080** | **+8.68** | +0.9342 | **+0.7945** | **resolves** |
| `grad_real_shuffled` (null) | -0.0168 | -0.20 | +0.1995 | -0.0169 | at-floor |
| `wire_real` | **+0.6210** | **+7.61** | +0.8455 | **+0.7555** | **resolves** |

### gradfix --- gradient protected

| arm | lift (pp) | z | Bonferroni u95 |
| --- | ---: | ---: | ---: |
| `grad_real` | +0.0809 | +0.99 | +0.2943 |
| `grad_real_shuffled` (null) | +0.0061 | +0.07 | +0.2203 |
| `wire_real` | +0.2960 | +3.63 | +0.5139 |

The shuffled-label control is at floor in both cells, so the E1 signal is
information-dependent rather than a property of the channel's structure.

## The mechanism: chaff works until the gradient says which rows are chaff

This is the part that matters, and it is measured, not inferred.

| | E1 (unprotected) | gradfix (protected) |
| --- | --- | --- |
| frames where the gradient's zero-support **exactly** matches the real/chaff partition | **4,096 of 4,096 (100%)** | **0 of 4,096 (0%)** |
| row-level agreement, zero-support vs real | **1.000** | 0.400 |
| non-zero gradient rows per 80-row frame | **exactly 32** | 80 |

A chaff row's gradient is identically zero, because `restore` truncates before the loss. With
the gradient unprotected, the compromised node reads the real/chaff partition **perfectly and
for free**, in every frame. Adding Gaussian noise makes every row non-zero and takes the
agreement to 0.400 --- which is exactly the base rate, 32/80. The fix closes the disclosure
completely.

So the defence and its defeat are both visible in one pair. The released frame as a whole ---
all 80 rows, chaff included --- reads `recovery_above_label_free_pct` = **-0.124 pp**, at
floor: chaff is doing its job. Restrict to the 32 real rows and the same frame reads
**+0.6210 pp at z = +7.61**. In E1 that restriction costs the attacker nothing, because the
gradient hands it over.

**A necessary caveat on `wire_real`.** Both cells build it using the *oracle* partition, so
the arm itself does not prove an attacker could construct it. What the `support_leak`
measurement adds is that in E1 the oracle is **redundant** --- the gradient reproduces the
partition at 100%. In gradfix it is not redundant: the partition sits at chance, so
gradfix's `wire_real` +0.2960 credits an attacker with knowledge it would not have, and
should not be read as an achievable attack.

## All seven arms, and where the gate breaks

Scored after the initial three. `grad_all` and `wire_all` cover all 80 released rows;
the `_real` arms cover the 32 real rows; the joint arms concatenate frame and gradient.

| arm | rows | lift (pp) | z | Bonferroni u95 | vs 1.0 gate | paired (pp) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `grad_all` | 163,840 | +0.2838 | +5.51 | +0.4201 | passes | --- |
| `joint_real` | 65,536 | +0.6134 | +7.52 | +0.8378 | passes | +0.7458 |
| `wire_real` | 65,536 | +0.6210 | +7.61 | +0.8455 | passes | +0.7555 |
| `grad_real` | 65,536 | +0.7080 | +8.68 | +0.9342 | passes | +0.7945 |
| **`joint_real_scaled`** | 65,536 | **+1.0315** | **+12.64** | **+1.2640** | **FAILS** | **+1.1672** |
| `grad_real_shuffled` (null) | 65,536 | -0.0168 | -0.20 | +0.1995 | passes | -0.0169 |

Three things follow.

**A naive attacker needs no cleverness at all.** `grad_all` takes the raw 80-row gradient
frame, runs the frozen probe on it, and resolves at z = +5.51 --- no partition inference, no
oracle, no rescaling. Restricting to the real rows via the disclosed zero-support raises it
2.5x, to z = +8.68, but the channel is readable before any of that.

**Scaled concatenation reaches past both single channels.** `joint_real_scaled` lifts the
gradient block to the frame's magnitude and reads +1.1672 pp paired, above `grad_real`
(+0.7945) and `wire_real` (+0.7555). Plain concatenation does not (+0.7458, *below*
`grad_real`), because the gradient's rows are ~0.3% of the frame's magnitude and the
rotation-invariant features of the raw concatenation are numerically the frame's own.

This **reverses the threat-model audit's conclusion** for this cell. That audit measured
`joint_real_scaled` on cells whose forward path already leaked and found it "does change the
per-arm counts but not their maximum", recording the joint view as needing a purpose-built
cross-Gram arm. On a forward-passing cell with the gradient unprotected, a one-line rescaling
is enough. The open question (campaign item W4.2) is accordingly how much further
`XG^T` goes, not whether the joint view is reachable.

**One arm breaks the pre-declared gate.** `joint_real_scaled` reads u95 **+1.2640 pp against
the 1.0 pp threshold** on a cell whose published forward reading is +0.4147 pp and passes.
The gate did not fail because its threshold was miscalibrated here --- it failed because the
channel was never scored.

## The gate would have passed it

E1's gradient arm reads a Bonferroni upper-95 excess of **+0.9342 pp, under the pre-declared
1.0 pp gate**, while carrying token information at z = +8.68 and a paired advantage of
+0.7945 pp [resolves].

The gate never scored the gradient --- that is the omitted-channel finding, and it is why the
channel went unmeasured for the whole campaign. The additional point here is narrower and
worse: **had the gradient been scored by the frozen attacker under the pre-declared
threshold, it would have passed.**

## What this does not establish

- **One seed, pre-freeze.** Exploratory. The confirmatory replication at >= 3
  post-freeze seeds is campaign item W5.7.
- **The joint result is one cell, one seed.** `joint_real_scaled` breaking the gate is the
  strongest reading in the campaign and the least replicated. Seeds 43 and 44 were
  subsequently run from packaged code; see `E1_REPRODUCTION_W17_2026-08-22.md`.
- **Token-identity signal, not reconstruction.** `token_top1` is the only implemented metric.
  Sequence, rare-token, and semantic recovery remain unimplemented (campaign item W2.3), so
  nothing here says what text was recovered.
- **The paired statistic clusters by frame.** `DESIGN_EFFECT_MEASUREMENT.md` (not included
  in this release) supports that at the frame level, but the long-range corpus drift it
  identifies is not absorbed.
- **`a2b` fails its own utility gate** (`utility_delta_le_0_35: false`,
  `eval_loss_delta` 0.9185). It passes the forward-privacy leg only. This is not a
  configuration anyone would ship, and the result should not be described as breaking a
  deployable operating point.
- **The fix is measured on one cell.** gradfix closes the disclosure completely here; whether
  clip 0.01 is the right operating point, and what it costs in utility, is not established.

## Correction

An earlier report from this session stated that E1's forward frame "reads -0.124 pp, at or
below its control", implying the forward path was clean. That conflated two different
measurements: `recovery_above_label_free_pct` is the in-run probe over the **full 80-row
released frame**, while the `wire_real` arm is the frozen attacker over the **32 real rows**.
The first is at floor; the second resolves at z = +7.61. Both numbers were correct; the
inference drawn from the first was not.
