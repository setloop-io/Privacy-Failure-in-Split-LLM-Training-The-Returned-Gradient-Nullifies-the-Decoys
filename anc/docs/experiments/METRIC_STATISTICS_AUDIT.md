# Metric statistics audit: the control estimator and the sampling model

**Date:** 2026-08-19.
**Scripts:** `bin/deleg6040_metric_stats_audit.py`, `bin/deleg6040_paper_control.py`.
**Artifacts:** `paper-data/collected/diagnostic/deleg_60_40/metric_stats_audit.json`,
`…/paper_control_eval_rows.json`.
**Scope:** all 111 attacker artifacts under `paper-data/collected`, re-derived from raw
`correct`/`total` counts, plus a byte-exact reconstruction of the campaign corpus used to
regenerate the evaluation labels. No training, no cloud server, no GPU, no remote host.
Nothing under `paper-data/collected/` was modified; two files were added.

Two defects raised by the external adversarial review
(finding 5; the review document is not included in this release) are worked here. The
review is right about both, and the arithmetic in `bin/deleg6040_verify_stats.py` remains
correct — what is wrong is the estimator it computes and the sampling model it assumes.

## Bottom line

- **Defect 1 is real in direction and exactly zero in magnitude everywhere it can be
  measured. No verdict changes.** The paper's control is never smaller than the code's, so
  the fix can only *lower* an excess, one pp of control for one pp of excess. Both legs of
  the gate are monotone in the control, so **no passing cell can become a failing one —
  101 of 111 verdicts are safe by algebra, before any measurement.** Only the 10 failing
  cells could move, and each needs a specific, computed control increase first.
- **Where it is measurable, the shift is +0 rows.** The campaign corpus was reconstructed
  byte-exactly (sha256 verified against `paper-data/corpus_manifest_original.json`) and
  the evaluation labels regenerated. The evaluation rows' most frequent token and the
  training rows' most frequent token are the same token — 279, `' the'` — so the paper's
  control and the code's control are the same number, **397 / 8,192 = 4.846191%**, and the
  excess shift is **0.000000 pp**. Cross-checked against all 26 committed artifacts that
  share that evaluation partition, on both the control count and `known_eval_fraction`.
- **That includes the published failure that was closest to flipping.** The v6-era
  $\sigma{=}0.30$ tripwire (`latent_v6_iso_seed43`, +1.0386 pp, cited in the draft
  manuscript at `main.tex:188` and `main.tex:317`; line numbers refer to
  `papers/arxiv-draft/main.tex`, not included in this release) needed only
  **4 extra rows out of 8,192** to become a pass. It gets
  none. Its FAIL verdict survives the paper's own definition.
- **For 85 of 111 cells the magnitude remains uncomputable, and no number here estimates
  it.** Those cells release CSPRNG-drawn chaff — between 27% and 60% of every frame —
  whose row identities were never stored, so their evaluation label multiset is not a
  function of the corpus and cannot be recovered from any committed artifact. Their
  verdicts would need a control shift of +0.25 to +1.35 pp to move.
- **Defect 2's two corrections pull in opposite directions, and the published number sits
  between them.** Pairing removes the arm/control covariance and generally *narrows* the
  interval; clustering removes the independence assumption and *widens* it. The published
  excess lies inside the admissible paired bracket in **111 of 111 cells**, so it is never
  outside what a correct paired analysis could produce — but it is not a bound on the
  difference either, and three passing cells fail at the worst admissible overlap.
- **The campaign's verdicts survive a design effect of about 2 and collapse between 3 and
  6.** 95 of 111 cells still pass at $D{=}2$, 70 at $D{=}3$, 51 at $D{=}4$, 16 at $D{=}6$,
  and **0 at the design effect the frame structure permits.** Nothing in the repository
  measures $D$. This is a bound, not a measurement.
- **The "statistical floor" is not a property of $(n, p_0, k)$.** It is a function of how
  far the arm and the control agree, which the metric never looks at. Two committed cells
  contain an arm that *is* the constant majority predictor; paired, their excess is
  **exactly 0.0000 pp with zero variance**, against a published floor of +0.4096 and
  +0.4100 pp. The second leg of the gate in `main.tex` Eq. (2) is calibrated on that
  quantity.

---

# Defect 1 — the control estimator contradicts the paper

## The two estimators

`main.tex:153` defines the control as *"the best constant predictor on the evaluation
rows"*. `attacker/attacks/latent_probe.py:215-217` computes something else:

```python
train_flat_tokens = train_y.reshape(-1)
majority = int(torch.mode(train_flat_tokens).values)
majority_correct = int((eval_y == majority).sum())
```

It takes the mode of the **train** partition and scores that one constant on the
evaluation rows. Write $L_v$ for the number of evaluation rows carrying token $v$, $m$ for
the train mode, $n$ for the row count and $U$ for the worst Bonferroni-adjusted Wilson
upper bound over the arms. Then

$$\text{excess}_\text{code} = U - 100\,L_m/n, \qquad
  \text{excess}_\text{paper} = U - 100\,\max_v L_v / n .$$

$U$ does not depend on the control, so **the excess moves by exactly minus the control
shift, one pp for one pp**. And $\max_v L_v \ge L_m$ by definition, so the code's control
is never larger and the code's excess is never smaller. The threshold report's own admission
(`DELEGATION_60_40_THRESHOLD.md:289`, not included in this release) is correct in direction.

## Why no passing cell can fail

The gate is $\text{excess} \le \min(1.0,\ \text{floor} + 0.70)$ with
$\text{floor}(C) = \mathrm{WilsonUpper}(C, n, z) - 100C/n$ for control count $C$. Written
out, the two legs are

$$\text{leg 1:}\ \ U - 100C/n \le 1.0, \qquad
  \text{leg 2:}\ \ U - \mathrm{WilsonUpper}(C, n, z) \le 0.70 .$$

The control term **cancels outright in leg 2** — raising the control lowers the excess and
raises the floor by the same amount, so leg 2 only ever compares $U$ against the Wilson
bound at the control count. Both legs are monotone non-increasing in $C$, so the passing
set is upward closed and a single threshold decides each cell. Raising the control can
therefore only turn FAIL into PASS. `smallest_control_passing()` computes that threshold;
across the sweep it exceeds the current control for exactly the 10 cells that currently
fail and for no other, which is the same statement measured rather than argued.

## What it would take to flip each failing cell

Every column is re-derived from the artifact's raw counts.

| excess pp | control | control to pass | rows | shift pp | recomputable | cell |
|---|---|---|---|---|---|---|
| +1.0386 | 397 | 401 | **+4** | **+0.0488** | **yes** | `invalid/latent_v6_rejected/latent_v6_iso_seed43` |
| +1.1159 | 397 | 407 | +10 | +0.1221 | **yes** | `invalid/latent_v6_rejected/latent_v6_defgraph_seed43` |
| +1.2500 | 1,301 | 1,373 | +72 | +0.2511 | no | `diagnostic/latent_v9/latent_v9_e2_seq64_06b` |
| +1.3643 | 2,212 | 2,388 | +176 | +0.3906 | no | `diagnostic/latent_v9/latent_v9_e2_seq128_06b` |
| +1.4899 | 2,124 | 2,326 | +202 | +0.4932 | no | `diagnostic/latent_v9/latent_v9_v93_public4k_priv1k_d64_06b` |
| +1.5528 | 1,092 | 1,206 | +114 | +0.5566 | no | `diagnostic/latent_reval/latent_reval_e5_pub4k_priv1k_mine` |
| +1.5770 | 1,086 | 1,205 | +119 | +0.5811 | no | `diagnostic/deleg_60_40/a4_sublayer_attn_split14` |
| +1.6676 | 13,964 | 16,811 | +2,847 | +0.8688 | no | `diagnostic/deleg_60_40/a2_window4096_split14` |
| +2.0904 | 1,019 | 1,243 | +224 | +1.0938 | no | `diagnostic/deleg_60_40/deleg_6040_split14` |
| +2.3437 | 1,101 | 1,377 | +276 | +1.3477 | no | `diagnostic/deleg_60_40/deleg_6040_ladder_split13` |

Five of the ten are verdicts published in the draft manuscript, matching its quoted
figures to four decimals: the $\sigma{=}0.30$ tripwire "+1.039" (`main.tex:188`, `main.tex:317`), E2
seq-64 "+1.250" and seq-128 "+1.364" (`main.tex:560-561`), v9.3/E5 "+1.490"
(`main.tex:567`), and the E5 re-evaluation carried as a bar in Fig. `fig:reval`
(`main.tex:936-941`). The remaining five are this campaign's own diagnostic cells and one
rejected v6 artifact.

**The first row was the exposure.** `main.tex:188` uses it to argue the metric "remains a
valid tripwire" — *"it did: noise $\sigma{=}0.30$ failed at $+1.039$ pp in the v6-era
configuration"* — and `main.tex:317` uses it to bracket the noise boundary. A control
larger by four rows in 8,192 would have turned that failure into a pass and cost the
argument one of its three supporting points. It is measured below, and it gets zero extra
rows.

## Where the paper's control is recoverable, and where it is not

The evaluation rows are not the corpus rows. Each released frame is `seq_len` real rows
concatenated with `chaff_tokens` recycled rows and permuted as one block
(`bin/run_latent_native_v5_06b.py:377-419`). The real labels are a deterministic function
of corpus, tokenizer and `seq_len` and are regenerable exactly — that is the method
`bin/deleg6040_bundle_forensics.py` already uses and validates. The chaff labels are not:
`chaff_sample` draws a fresh CSPRNG permutation per frame from `secrets.token_bytes(16)`
(`:295-304`), and the selection never leaves the trusted process. **So wherever chaff is
released, $\max_v L_v$ is unrecoverable in principle, not merely unavailable.**

Which cells release chaff is measured, not assumed. Chaff is redrawn on every run, so
independent cells sharing a frame shape must land on different control counts if any chaff
is in the released view. Grouping all 111 artifacts by `(eval_blocks, rows_per_frame)`:

| frames × rows | cells | distinct control counts | control | chaff-free |
|---|---|---|---|---|
| 256 × 32 | **26** | **1** | **397** | **yes** |
| 256 × 48 | 22 | 20 | 592–614 | no |
| 256 × 64 | 1 | 1 | 883 | no (single cell) |
| 256 × 80 | 37 | 31 | 1,019–1,101 | no |
| 256 × 112 | 3 | 3 | 1,245–1,373 | no |
| 256 × 176 | 3 | 3 | 2,165–2,212 | no |
| 512 × 80 | 15 | **15** | 2,074–2,145 | no |
| 1024 × 80 | 1 | 1 | 4,317 | no (single cell) |
| 4096 × 80 | 3 | 3 | 13,656–14,063 | no |

Twenty-six independently run cells, spanning v6 and v7, four seeds and 13 distinct
recorded hostnames, all report a control of exactly `4.8461914062%` = 397/8,192. Every other multi-cell group
spreads across nearly as many control values as it has cells, because each run redraws its
chaff. That identifies the 32-row group as the candidate; what settles it is the
regeneration itself, in the next section, which reproduces **two** statistics those 26
artifacts recorded independently — the control count 397 and the 5,765 rows inside the
train class set — from a chaff-free `seq_len` 32 partition and nothing else. A third
corroboration: `deleg_6040_*_forensics.json` measures the *real-row* component of the
control at **397 in all three of its cells**, over the same 256 evaluation blocks. The
numbers agree because they are the same 8,192 corpus rows.

So the paper's control is recoverable for **26 of 111 cells** — including the exposed
$\sigma{=}0.30$ failure — and for **85 of 111 it is not recoverable from any committed
artifact**, because between 27% and 60% of every scored row is chaff.

*Incidental:* three of the 26 (`latent_v7_E_chaff16`, `latent_v7_G_chaff16_int8`,
`latent_v7_H_chaff16_n045`) record `chaff_tokens: 16` in their run artifact while their
attacker bundle carries 32-row frames and the chaff-free control. Whatever chaff those
runs used during training, none of it reached the view the attacker was scored on. Noted
because it is why the group's declared chaff widths are mixed; it is not part of either
defect.

## The measurement: the shift is exactly zero

Regenerating the 26 cells' control needs the campaign corpus, which is not in the
repository and which the raw-corpus reconstruction recipe reports as *"could not be
obtained"* — that committed reconstruction is the **raw** WikiText-2 variant, a
different file (sha256
`1ac2aed3…`) already shown to change results materially. The original is pinned by
`paper-data/corpus_manifest_original.json`: 10,316,588 bytes, 14,284 lines, sha256
`78b6bfb9…`, the *tokenized* variant.

**It was obtained here, byte-exactly.** The recipe is in
`bin/deleg6040_paper_control.py`'s header and reproduces the manifest hash: the tokenized
`wikitext-2-v1` train split at the pinned dataset revision, lines stripped, lines of 200
characters or fewer dropped, joined with `\n` and **no trailing newline** — that last byte
is decisive, since with it the file hashes to something else. The manifest's own
"12,095 lines contain `<unk>`" is what identifies the 200-character threshold uniquely.

Three independent checks tie the regenerated labels to the committed campaign artifacts,
and the tool aborts on any of them:

| check | regenerated | committed | source |
|---|---|---|---|
| corpus sha256 | `78b6bfb9…` | `78b6bfb9…` | `corpus_manifest_original.json` |
| control count on the eval rows | 397 / 8,192 | 397 / 8,192 | all 26 chaff-free artifacts |
| eval rows inside the train class set | 5,765 | 5,765 | same artifacts' `known_eval_fraction` |
| control's real-row component | 397 | 397 | `deleg_6040_*_forensics.json`, 3 cells |

With the labels verified, both estimators are computed directly:

| estimator | token | correct | rate |
|---|---|---|---|
| code — mode of the **train** labels, scored on eval rows | 279 (`' the'`) | 397 | 4.846191% |
| paper — argmax over the **eval** label counts | 279 (`' the'`) | 397 | 4.846191% |

**They are the same estimator's output. The control shift is +0 rows, +0.000000 pp, and
the excess shift is 0.000000 pp on every one of the 26 cells.** The two definitions
disagree only when the evaluation partition's most frequent token differs from the
training partition's, and on this corpus it does not: `' the'` leads on both sides, 397 to
309 on the evaluation rows and 451 to 361 on the training rows.

**No verdict changes.** In particular the $\sigma{=}0.30$ v6-era failure at +1.0386 pp,
which needed only 4 extra rows, **keeps its FAIL verdict under the paper's own
definition**, and `main.tex:188` and `main.tex:317` stand on it.

## The 85 chaffed cells

Their control is still not computable, for the reason given above: chaff labels are not a
function of the corpus. What the corpus now pins exactly is both components that feed
them. The 37 cells at 256 frames of 80 rows draw their 8,192 real rows from the partition
measured above, and their 12,288 chaff rows from a pool that is the train label multiset:

| token | real eval rows | chaff pool | measured chaff share, 3 forensics cells |
|---|---|---|---|
| 279 `' the'` | **397** of 8,192 (4.846%) | **451** of 8,192 (5.505%) | 5.062%, 5.501%, 5.518% |
| 1154 `' ,'` | 309 of 8,192 (3.772%) | 361 of 8,192 (4.407%) | — |

The measured chaff shares of token 279 track its pool share to within 0.02 pp on two of
the three cells and 0.44 pp on the third. That is a side result worth recording: the
chaff-provenance assumption that the bundle forensics audit (not included in this release)
listed under "known limits" as
*"asserted from the code, not measured"* is now corroborated by measurement.

`' ,'` is the best-placed competitor on both components, so it carries the loosest
requirement of any token. Against that, the flip thresholds for the four failing cells of
this shape require it to take between **7.29% and 8.69% of the chaff rows** while holding
4.407% of the pool — a 1.65× to 1.97× over-representation, where the observed draws are
within 0.44 pp of the pool share. **That is an argument from measured quantities, not a
computation of the control, and it is not offered as one.** The exact position remains:
for these 85 cells the shift is unknown, is bounded below by zero, and would have to reach
+0.25 to +1.35 pp to move any verdict.

---

# Defect 2 — the intervals assume independence the design does not provide

## What the design actually is

Every attacker artifact records its own frame structure, and it is exact: for all 111
cells `eval_blocks × sequence_length == total`, with no exceptions. At the v13 operating
point that is 256 frames of 80 rows, 32 real and 48 chaff. Three features break the
independent-Bernoulli model the Wilson bound assumes:

- the `invariant_graph` arm computes a normalised Gram matrix over the whole frame and
  propagates through it, so every row's prediction is a function of all 80 rows of its
  frame (`attacker/attacks/latent_probe.py:115-135`);
- chaff rows are recycled train rows sampled from a shared 8,192-row pool
  (`run_latent_native_v5_06b.py:279-304`), so rows repeat across frames;
- the 32 real rows of a frame are one contiguous corpus block, so their labels are
  correlated by construction.

Separately, the control and the arm are scored on the **same rows**. Subtracting a point
estimate of the control from a one-sample upper bound on the arm is not a confidence bound
on their difference, whatever the row model.

## The paired comparison

Let $A$ be the arm's correct count, $M$ the control's, and $b$, $c$ the discordant counts
(arm right/control wrong, and the reverse). Then $b - c = A - M$ **exactly**, for every
cell, with no extra information. The paired difference $d_i \in \{-1, 0, +1\}$ has

$$\widehat{\mathrm{SE}} = \frac{\sqrt{(b+c) - (b-c)^2/n}}{n},$$

which depends on the discordant total $b + c = A + M - 2\,\text{both}$ — and "both", the
rows both predictors get right, is the one quantity the artifacts do not record. It is *bounded*, though:
$\text{both} \in [\max(0, A+M-n),\ \min(A, M)]$, tightened wherever a forensics artifact
records how many rows the arm spends on the control's own token. So the paired standard
error lies in an exactly computable interval.

For the campaign's worst cell, `deleg_6040_split14` (gate arm `invariant_graph` r2,
1,354/20,480 against a control of 1,019, overlap pinned to [25, 1019] by the committed
prediction concentration):

| quantity | value |
|---|---|
| paired point difference | +1.6357 pp |
| implied SE, published construction (arm alone, independent rows) | 0.1736 pp |
| paired SE, tightest admissible overlap | 0.0886 pp |
| paired SE, widest admissible overlap | 0.2351 pp |
| published excess | **+2.0904 pp** |
| paired upper bound, tightest overlap | +1.8608 pp |
| paired upper bound, widest overlap | +2.2326 pp |

The published number sits between the two paired bounds, and the cell fails the gate at
either end. That pattern is general: **the published excess lies inside the admissible
paired bracket in 111 of 111 cells.** The published statistic is therefore never outside
what a correct paired analysis could return — but it is not itself a bound on the
difference, and it is not conservative: it is neither an upper nor a lower end of the
bracket.

Three currently-passing cells fail only at the worst admissible overlap, all three
straddling the gate:

| published | paired bracket | cell |
|---|---|---|
| +0.8969 | [+0.3826, +1.1284] | `latent_v7_defense/latent_v7_F_int8` |
| +0.8613 | [+0.4560, +1.0578] | `latent_v7_defense/latent_v71_35b_s44_n035` |
| +0.8292 | [+0.4168, +1.0270] | `latent_v7_defense/latent_v7_35ba3b_wss_chaff16_seed42` |

Resolving them needs the per-arm agreement count, which the frozen attacker emits only
under `--dump-eval-predictions`.

## The floor is a property of agreement, not of $n$ and $p_0$

`main.tex:163-171` treats the statistical floor as a function of $(n, p_0, k)$ alone, and
Eq. (2)'s second leg budgets 0.70 pp on top of it. Under a paired analysis that is not
what the floor is. An arm scoring exactly the majority *count* can be anywhere from
identical to the control to maximally discordant with it:

- **identical**: $b = c = 0$, $d_i \equiv 0$, excess exactly **0.0000 pp** with zero
  variance, at any $n$ and any design effect;
- **maximally discordant**: $b = c = M$, giving $z \cdot 100\sqrt{2M}/n$ — at
  $n = 20{,}480$, $M = 1{,}073$, that is **+0.5743 pp**.

The published floor for that cell is +0.4096 pp, inside the range. And the first end is
not hypothetical. Two committed cells realise it exactly:

| cell | arm | correct | both | b+c | paired excess | published floor |
|---|---|---|---|---|---|---|
| `deleg_6040_conv10k_split14` | `invariant_only` r0 | 1,073 | 1,073 | 0 | **+0.0000 pp** | +0.4096 |
| `deleg_6040_ctrl_split21` | `invariant_only` r0, r1 | 1,075 | 1,075 | 0 | **+0.0000 pp** | +0.4100 |

Those arms emit a single token class over all 20,480 rows — they *are* the constant
majority predictor (see "what the winning arms actually predict" in the bundle forensics
audit, not included in this release). The
metric charges them +0.41 pp for being the control. That audit already
observed their raw top-1 excess is exactly zero on every population; the addition here is
that the +0.4096 pp is not a floor the sample size imposes, it is the cost of not pairing,
and it does not shrink with $n$ for any reason a paired analysis would recognise.

This matters beyond presentation: the 0.70 pp budget in Eq. (2) was fitted to the largest
floor-relative reading among passing cells (per the gate recalibration analysis, not
included in this release). If the floor is partly an artifact of the unpaired construction,
so is the budget calibrated on
top of it.

## The clustering correction, as a bound

The true design effect is not measurable from committed artifacts: it needs per-row
correctness grouped by frame, which exists only in the `--dump-eval-predictions` `.pt`
output. Two things can still be computed exactly.

**The extremal bound.** With clusters of size $m$ the design effect cannot exceed $m$.
`max_cluster_sum_squares()` computes the exact worst allocation the design permits —
packing the discordant rows into as few frames as possible — and at the v13 operating
point it inflates the gate arm's standard error by 8.89×, against the ceiling
$\sqrt{80} = 8.94$. Under that
bound every cell in the campaign fails, including the shuffled-label negative controls and
the cells whose gate arm is provably the constant majority predictor. **The extremal bound
is therefore true and useless**: it says only that clusters of 80 can destroy any
conclusion, not that they do.

**The break-even design effect**, which is the informative version. Relax independence and
nothing else: keep the estimator, the arm, the control and both gate legs exactly as
published, and re-evaluate every Wilson bound at effective sample size $n/D$. $D = 1$
reproduces the committed numbers exactly; the floor moves with it, so the second leg
self-corrects. Bisecting for the $D$ at which each cell's verdict turns:

| assumed design effect $D$ | cells passing (of 111) |
|---|---|
| 1.0 (as published) | 101 |
| 1.5 | 100 |
| 2 | 95 |
| 3 | 70 |
| 4 | 51 |
| 5 | 40 |
| 6 | 16 |
| 8 | 13 |
| 10 | 9 |
| 16 | 2 |
| each cell's own $m$ | **0** |

The median break-even $D$ over all 111 cells is 3.45; the most robust cell is
`a2c_window4096_100k_split14` at $D = 47.8$, which it earns from $n = 327{,}680$ rather
than from any property of the defense. Six of the 101 currently-passing cells need $D < 2$ — near-exact
independence — to keep their verdict.

**Read this as a sensitivity, not a verdict.** No cell is shown here to fail. What is
shown is that the campaign's privacy conclusions are stable against a design effect of
about 2, degrade sharply from 3 to 6, and are entirely lost at the design effect the frame
structure permits — and that nothing in the repository measures which regime the data is
in. For a per-row design in which one arm explicitly conditions on the whole frame, $D=1$
is the least likely value in the range.

---

# What this implies for the paper

Reported, not applied. Line references below are to the draft manuscript
(`papers/arxiv-draft/main.tex`), which is not included in this release and has since been
revised.

1. `main.tex:153` and `attacker/attacks/latent_probe.py:215-217` still have to be
   reconciled, but no number moves either way. The accurate wording is "the constant
   predictor given by the training partition's mode, scored on the evaluation rows", with
   a footnote that on this corpus the two estimators coincide exactly — the same token,
   the same 397 rows — for every cell where the evaluation rows are corpus rows. Claiming
   the paper's estimator without that footnote asserts something unverified for the 85
   chaffed cells.
2. No verdict needs revising for defect 1. The $\sigma{=}0.30$ v6-era failure at +1.039 pp
   was the exposure and it holds; `main.tex:188` and `main.tex:317` stand.
3. `main.tex:163-171` describes the floor as a function of $(n, p_0, k)$. It is also a
   function of arm/control agreement, and equals exactly zero for an arm that is the
   control. Eq. (2)'s 0.70 pp budget inherits that.
4. The metric's population should be stated. The bundle forensics audit (not included in
   this release) already established
   that only 28.2% of the 20,480 scored rows are held-out corpus content; the control is a
   property of that mixed population too.
5. Row independence should be stated as an assumption with its consequence. A single
   sentence naming the design effect the verdicts tolerate would be honest and is
   computable today: `--sweep` prints it per cell.

# Known limits

- **$\max_v L_v$ is measured for 26 cells and not computed for the other 85.** Nothing
  here estimates it. Every statement about the 85 is either a threshold ("this much would
  be needed"), a proven direction, or an argument explicitly labelled as one.
- **The corpus is a reconstruction, validated by hash and by three artifact
  cross-checks — it is not the original file.** The original lives on
  `gx10-odysseus.nord` and was never committed. Byte-identity is established by sha256
  against the manifest, and the tokenization is tied to the campaign by two counts the
  artifacts recorded independently (397 and 5,765). The tokenizer was pulled from
  `Qwen/Qwen3-0.6B` at current `main` rather than a pinned revision; the two count matches
  are the evidence that it tokenizes identically, not a pin.
- **The corpus is deliberately not added to the repository.** It is the campaign's private
  training corpus. `bin/deleg6040_paper_control.py` carries the recipe and verifies the
  hash, so a reader reproduces it rather than receiving it.
- **The design effect is bounded, never measured.** The break-even table is a sensitivity
  analysis over an assumed $D$. The extremal bound is exact but vacuous. The true $D$
  needs per-frame correctness.
- **The paired standard error is an interval, not a point**, except where a committed
  forensics artifact pins the agreement count — three cells of 111.
- **Selection across arms is handled the way the campaign handles it**, by a Bonferroni
  factor equal to the number of arms in the artifact. Whether Bonferroni is the right
  correction for nine highly correlated arms trained on the same data is a separate
  question and is not addressed here.
- **This audit changes no committed verdict.** It reports what would have to be true for
  one to change.

# Re-deriving every number

```
# the sweep behind every table above, and the committed artifact
python3 bin/deleg6040_metric_stats_audit.py --sweep paper-data/collected \
  --output paper-data/collected/diagnostic/deleg_60_40/metric_stats_audit.json

# defect 1's measurement: both control estimators on the regenerated eval rows.
# Needs the corpus (recipe and hash check in the script header) and a Qwen3-0.6B
# tokenizer directory; it aborts unless the corpus hashes to the manifest and the
# regenerated labels reproduce all 26 chaff-free artifacts.
python3 bin/deleg6040_paper_control.py --corpus <wikitext2_corpus.txt> \
  --model <qwen3-0.6b tokenizer dir> \
  --output paper-data/collected/diagnostic/deleg_60_40/paper_control_eval_rows.json

# the artifact-only checks from that tool, needing neither corpus nor tokenizer
python3 bin/deleg6040_paper_control.py --self-test

# one cell, with its per-arm paired bounds
python3 bin/deleg6040_metric_stats_audit.py \
  --artifact paper-data/collected/diagnostic/deleg_60_40/deleg_6040_split14_attacker.json

# what cannot be computed from committed artifacts, and the command that would close it
python3 bin/deleg6040_metric_stats_audit.py --help-limits

# arithmetic checks that touch no artifact
python3 bin/deleg6040_metric_stats_audit.py --self-test

# the unchanged published statistics these are measured against
python3 bin/deleg6040_gate_recalibrate.py --sweep paper-data/collected --output /tmp/gate.json
```

Column key for the sweep table: `excess`/`floor` are the published statistics re-derived
from raw counts; `gate` is the current verdict; `rows`/`shift pp` are defect 1's flip
threshold; `rec` is whether the paper's control is recomputable for that cell; `paired`
and `pairedLO` bracket the paired upper bound; `deff*` is the break-even design effect;
`m` is the rows per released frame, which is also the design-effect ceiling.
