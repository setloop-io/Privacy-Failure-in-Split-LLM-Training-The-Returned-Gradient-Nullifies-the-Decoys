# Gate sensitivity calibration: a leakage-injection dose-response curve

**Date:** 2026-08-19.
**Node:** `gx10-odysseus.nord` only. No training, no cloud server, no `tln`/`ucn`.
**Scripts:** `bin/deleg6040_gate_sensitivity.py`, `bin/deleg6040_gate_sens_sweep.sh`.
**Artifacts:** `paper-data/collected/diagnostic/deleg_60_40/gate_sens_*.json`.

An external adversarial review of the campaign (its finding 6; the review document is not
included in this release) states the open question this answers:

> One shuffled-label permutation tests false positives for these already-trained predictors.
> It does not measure false-negative sensitivity. What would settle this is a controlled
> leakage-injection curve through the full cloud view [...]

`BUNDLE_FORENSICS.md` (not included in this release) established the gate has no false
positives: shuffling the evaluation
labels drops every cell to its own statistical floor. It also established that every arm
which beats the label-free control emits 1-7 distinct token classes over 20,480 rows, and
that in the floor cell the winning arm emits exactly one class for 100% of rows — it *is*
the constant majority predictor. Nothing has ever measured what the gate does when token
information really is present in the released frame.

This document measures that. Known, quantified amounts of true token information are
written into the released rows of a bundle that currently reads at the floor, the frozen
nine-arm attacker is re-scored unmodified, and the gate is re-derived from raw counts.

## Bottom line

Thirty-two doses, each a full nine-arm re-score of the frozen attacker on an injected copy
of `deleg_6040_conv10k_split14_bundle.pt`.

- **The gate has a large false-negative region, and it is measurable.** A leak that writes
  the token into the released row *exactly and noiselessly* has to reach **6.0% of released
  rows — 1,229 of 20,480 — before the gate moves off its floor**. At 5.5% coverage (1,126
  rows) the gate still reads `at-floor`, `PASS`, excess +0.8036 pp. At 4.0% coverage (819
  rows) it reads **+0.4096 pp, binomial z 0.000, `at-floor`, `PASS` — digit for digit the
  undosed reading** — while the coordinate arm has already converted 377 of those rows into
  correct predictions.
- **The excess clears +1.0 pp between 5.5% and 6.0% coverage**, at +1.0660 pp. The
  `resolved` criterion (z > 2.5392) and the +1.0 pp gate cross at the same grid point, so
  tightening the gate to the finer criterion buys nothing here.
- **A rotation-faithful leak — one injected into the trusted latent, where a real encoder
  leak would live — needs between 40% and 50% of released rows.** At 40% coverage, 8,192
  released rows are exact codewords, recoverable at 100.00% by a decoder holding the frame
  secret, and the gate reads +0.4248 pp, `PASS`. At 25% and 30% coverage it reads
  **+0.4096 pp with z exactly 0.000 — bit-identical to the undosed reading.**
- **The cause is the constant-predictor pedestal, not the statistics.** The gate is
  `max over nine arms` of a Wilson upper bound against the control. Three of the nine arms
  (`invariant_only` r0/r1/r2) emit one or two token classes over all 20,480 rows and score
  exactly 1,073 — the control — and they stay pinned there through every coordinate-mode
  dose. The arm that *does* read the leak starts 494 rows *below* the control. The gate
  needs +117 rows above the control to fail; that arm needs +611 rows to supply them. The
  gate's usable sensitivity is therefore about five times worse than its nominal statistical
  resolution, and the gap is pure pedestal.
- **Each arm family detects only the leak that lives in its own representation.** In
  coordinate mode the coordinate arms climb from z = -15.5 to +532.5 while the invariant arms
  never move by more than 5 rows. In invariant mode the invariant arms climb from z = 0.000
  to +62.7 while the coordinate arms stay between z = -13.3 and -14.6. Neither family covers
  for the other, and the campaign's floor cell is decided entirely by the family that is
  blind to a coordinate-visible leak.
- **What the gate misses is token identity, not a marginal that failed to shift.** The
  injection preserves every row's norm to 4.8e-07 (float32 rounding) and never touches a
  label, so the control is fixed at 5.2392578125% at all 32 doses. Nothing was added except
  the token.

Blunt answer to the question that prompted this: **as a detector of token leakage the gate
is coarse, and its coarseness is dominated by an artefact of its own construction.** It is a
valid *bound* — nothing it flags is spurious, as the bundle forensics audit showed — but
"passes
at the floor" licenses far less than it appears to. The reading +0.4096 pp with z = 0.00 is
consistent with anything from no leakage at all to 819 released rows being exact, noiseless
copies of their token.

## The cell under test

`deleg_6040_conv10k_split14_bundle.pt` — split 14, 10,000 steps, 39.3% delegation. Its
recorded reading is the campaign's canonical "pass at the statistical floor":

| quantity | value |
|---|---|
| released rows scored | 20,480 (256 frames x 80 rows: 32 real + 48 chaff) |
| latent width D | 64 |
| label-free majority control | 5.2392578125% = 1,073 / 20,480 (token 279, ` the`) |
| best arm | `invariant_only` r0/r1/r2, 1,073 correct — exactly the control |
| Bonferroni-Wilson upper (z = 2.539185) | 5.648881% |
| excess over control | **+0.4096 pp** |
| statistical floor | +0.4096 pp — the excess *is* the floor, to 4 dp |
| best-arm binomial z | **0.00** |
| coordinate arms | 506 / 579 / 518 correct, i.e. z = -17.8 / -15.5 / -17.4 |

Release configuration for this cell (`bin/deleg6040_cell.sh`, confirmed in
`deleg_6040_conv10k_split14.json`): `secret_wire_rotation=True`,
`secret_token_permutation=True`, `secret_token_gauge=False`, `wire_quant=none`,
`chaff_tokens=48`, `latent_dim=64`. Each released frame is therefore
`(latent ++ chaff)[permutation] @ R_frame` for a fresh Haar `R_frame` drawn from the v2
CSPRNG inside the trusted process and never released.

## What the gate needs, in rows

Every threshold in the campaign reduces to a count of correct rows out of 20,480 against a
control of 1,073. Re-derived by
`python3 bin/deleg6040_verify_stats.py --baseline <artifact>`:

| criterion | best arm must reach | rows above the control |
|---|---|---|
| excess above the statistical floor | 1,074 | +1 |
| `resolved` (excess > floor **and** z > 2.539185) | 1,154 | +81 (0.396% of rows) |
| legacy gate FAIL (excess > +1.0 pp) | 1,190 | +117 (0.571% of rows) |
| floor-relative gate FAIL (excess > min(1.0, floor+0.70)) | 1,190 | +117 — the cap is 1.0 pp here |

Because `floor + 0.70 = 1.1096 > 1.0`, the floor-relative gate of the gate recalibration
analysis (not included in this release) reduces to the legacy 1.0 pp gate on this cell.
The two verdicts are identical at every point of this sweep and are reported as one.

The number that matters for what follows: **the arm that reads a coordinate-visible leak
starts 494 rows *below* the control** (579 against 1,073). It must recover 575 injected rows
before the reading is even `resolved`, and 611 before the gate fails.

## The injection

A fixed codebook assigns every token value one Haar-uniform unit direction `c_t` in R^64,
drawn from the same v2 CSPRNG stream the release path uses
(`privacy_runtime.ratchet_v2.derive_gaussian_tensor`). For a selected released row `x` with
token `t`:

```
x' = ||x|| * normalise( sqrt(1 - a^2) * x/||x||  +  a * c_t )
```

Two knobs, both exactly quantifiable:

- **coverage `rho`** — the fraction of released rows that carry the signal. Rows are the
  first `round(rho*N)` of one fixed CSPRNG permutation, so the injected row sets are
  **nested** across `rho`: the rows dosed at 2% are a superset of those dosed at 1%.
- **amplitude `a`** — the codeword's share of the row's unit direction. The injected power
  fraction is exactly `a^2`, so the token-correlated SNR is exactly `a^2/(1-a^2)`. At
  `a = 1` the row *is* its codeword scaled to the original norm: the token is recoverable
  from that row exactly, with zero residual, by a decoder that knows the codebook.

Both partitions are dosed. The attacker trains on `train_wire` and is scored on
`eval_wire`; a leak present only in the evaluation partition would be unlearnable and would
not be leakage in any operational sense. Labels are never touched, so
`label_free_majority_pct` is *invariant across the entire sweep* — 5.2392578125% at every
point — and every movement in the excess is attributable to the arms alone.

The blend is **norm-preserving by construction** (`max_row_norm_drift` is recorded at every
dose and is at the float32 rounding limit). This is deliberate: it decouples "the gate
detects token information" from "the gate detects a changed marginal distribution". The row
norm is the *first* feature `latent_invariants` returns (`latent_native.py:97`), so a leak
that also shifted the norm distribution would give the invariant arms a magnitude channel,
and any detection driven by it would say nothing about token recovery.

### Two frames of reference

The released frame is rotated by a per-frame secret. Where the injected direction lives
relative to that rotation decides what an attacker can do with it, so both are swept.

**`--mode coordinate`.** `c_t` is fixed in *released* coordinates and is identical in every
frame. This models a leak that bypasses or survives the wire rotation — an absent, stale or
reused rotation, or a side channel that writes into the released view. It is the *easiest
possible* leak for the frozen attacker: the coordinate arms see a single consistent code,
whose optimal decoder is `argmax_t <x, c_t>` — a linear map from R^64 to the class set, well
inside the reach of the arm's LayerNorm-Linear-GELU-Linear trunk. It is: at full coverage
and `a = 1` the coordinate arms recover 18,053 of the 18,053 scoreable rows, every one.

**`--mode invariant`.** `c_t` is carried through a fresh per-frame Haar rotation, exactly as
`release()` would carry a direction injected into the trusted latent *before* rotation. This
is not an approximation of pre-rotation injection; it is equal to it in distribution.
Writing `X` for a frame's trusted latents, `R` for its (unknown) release rotation and `C`
for the Haar-uniform codebook drawn independently of both, injecting `a*c_t` before release
yields the released pair `(XR, CR)`. Conditioned on `R`, `CR` is again Haar-uniform and
therefore independent of `R`, hence independent of `XR`; the same holds for `(XR, CR')` with
a fresh independent Haar `R'`. The two joint laws coincide, and they coincide jointly across
frames because `{CR_b}` and `{CR'_b}` are both i.i.d. Haar given `C`. Only the
rotation-invariant structure of the injection survives into a form an attacker can exploit
across frames: within a frame, rows sharing a token move toward the same direction, so their
pairwise cosine rises — which is exactly the channel `latent_invariants` reads.

### How much information was added, measured not asserted

Every dose records what a decoder can actually get back out of the released row, so a null
gate reading can never be confused with a leak that was never there:

- `codebook_decode` — nearest-codeword top-1 over the injected rows, using the codebook
  alone. This is what a compromised node holding the code could read. In `coordinate` mode
  at `a = 1` it is 100.00% by construction.
- `oracle_decode` — the same, after un-rotating each frame by its secret rotation. In
  `invariant` mode this is 100.00% at `a = 1` while `codebook_decode` sits at chance: the
  token is present in the released frame and is exactly recoverable *given the secret*, and
  not otherwise.
- `cosines` — mean within-frame cosine between injected rows that share a token against
  those that do not. This is the rotation-invariant signature, and it is the same in both
  modes by construction.
- `rows_injected_scoreable` — injected rows whose token is one the frozen attacker can emit
  at all (its class set is the 1,874 token values of the train partition). An injected row
  outside that set can never be scored correct however cleanly it leaks, so the budget the
  gate could even see is smaller than the budget injected.

## Dose-response

All 32 doses, printed by
`python3 bin/deleg6040_gate_sensitivity.py --table paper-data/collected/diagnostic/deleg_60_40/gate_sens_*.json`.
`inj` is injected released rows out of 20,480; `dec%`/`orc%` are nearest-codeword top-1 on
those rows without and with the frame secret; `coord d`/`inv d` are rows gained by the best
arm of each family over the undosed reading.

```
       mode coverage   ampl     inj    dec%    orc%    excess    floor    best z   coord z     inv z  coord d    inv d     class legacy
 coordinate   0.0000  1.000       0    0.00    0.00   +0.4096  +0.4096    +0.000   -15.492    +0.000       +0       +0  at-floor   PASS
 coordinate   0.0025  1.000      51  100.00  100.00   +0.4147  +0.4096    +0.031   -15.053    +0.031      +14       +1  at-floor   PASS
 coordinate   0.0050  1.000     102  100.00  100.00   +0.4147  +0.4096    +0.031   -14.771    +0.031      +23       +1  at-floor   PASS
 coordinate   0.0100  1.000     205  100.00  100.00   +0.4096  +0.4096    +0.000   -14.959    +0.000      +17       +0  at-floor   PASS
 coordinate   0.0200  1.000     410  100.00  100.00   +0.4096  +0.4096    +0.000   -11.948    +0.000     +113       +0  at-floor   PASS
 coordinate   0.0300  1.000     614  100.00  100.00   +0.4096  +0.4096    +0.000    -8.593    +0.000     +220       +0  at-floor   PASS
 coordinate   0.0400  1.000     819  100.00  100.00   +0.4096  +0.4096    +0.000    -3.669    +0.000     +377       +0  at-floor   PASS
 coordinate   0.0450  1.000     922  100.00  100.00   +0.4147  +0.4096    +0.031    -2.038    +0.031     +429       +1  at-floor   PASS
 coordinate   0.0500  1.000    1024  100.00  100.00   +0.4147  +0.4096    +0.031    -0.220    +0.031     +487       +1  at-floor   PASS
 coordinate   0.0550  1.000    1126  100.00  100.00   +0.8036  +0.4096    +2.446    +2.446    +0.031     +572       +1  at-floor   PASS
 coordinate   0.0600  1.000    1229  100.00  100.00   +1.0660  +0.4096    +4.077    +4.077    +0.031     +624       +1  resolved   FAIL
 coordinate   0.1000  1.000    2048  100.00  100.00   +3.9150  +0.4096   +21.858   +21.858    +0.157    +1191       +5  resolved   FAIL
 coordinate   0.3000  1.000    6144  100.00  100.00  +21.0496  +0.4096  +130.178  +130.178    +0.125    +4645       +4  resolved   FAIL
 coordinate   1.0000  1.000   20480  100.00  100.00  +83.4717  +0.4096  +532.505  +532.505   +62.972   +17474    +2008  resolved   FAIL
 coordinate   1.0000  0.050   20480    0.07    0.07   +0.4096  +0.4096    +0.000   -11.353    +0.000     +132       +0  at-floor   PASS
 coordinate   1.0000  0.100   20480    0.28    0.28   +0.7683  +0.4096    +2.227    +2.227    +0.031     +565       +1  at-floor   PASS
 coordinate   1.0000  0.110   20480    0.34    0.34   +1.3635  +0.4096    +5.927    +5.927    +0.031     +683       +1  resolved   FAIL
 coordinate   1.0000  0.120   20480    0.45    0.45   +1.8720  +0.4096    +9.095    +9.095    +0.063     +784       +2  resolved   FAIL
 coordinate   1.0000  0.130   20480    0.59    0.59   +2.5156  +0.4096   +13.109   +13.109    +0.031     +912       +1  resolved   FAIL
 coordinate   1.0000  0.150   20480    0.84    0.84   +4.0702  +0.4096   +22.831   +22.831    +0.031    +1222       +1  resolved   FAIL
 coordinate   1.0000  0.200   20480    2.40    2.40   +8.8895  +0.4096   +53.125   +53.125    +0.063    +2188       +2  resolved   FAIL
 coordinate   1.0000  0.300   20480   11.64   11.64  +23.0107  +0.4096  +142.660  +142.660    +0.031    +5043       +1  resolved   FAIL
 coordinate   1.0000  0.500   20480   77.90   77.90  +57.4305  +0.4096  +363.345  +363.345    +1.850   +12080      +59  resolved   FAIL
  invariant   0.0500  1.000    1024    0.10  100.00   +0.4349  +0.4096    +0.157   -13.611    +0.157      +60       +5  at-floor   PASS
  invariant   0.2500  1.000    5120    0.04  100.00   +0.4096  +0.4096    +0.000   -14.363    +0.000      +36       +0  at-floor   PASS
  invariant   0.3000  1.000    6144    0.07  100.00   +0.4096  +0.4096    +0.000   -14.583    +0.000      +29       +0  at-floor   PASS
  invariant   0.3500  1.000    7168    0.06  100.00   +0.4349  +0.4096    +0.157   -14.583    +0.157      +29       +5  at-floor   PASS
  invariant   0.4000  1.000    8192    0.05  100.00   +0.4248  +0.4096    +0.094   -14.395    +0.094      +35       +3  at-floor   PASS
  invariant   0.5000  1.000   10240    0.06  100.00   +1.1971  +0.4096    +4.892   -13.297    +4.892      +70     +156  resolved   FAIL
  invariant   1.0000  1.000   20480    0.03  100.00  +10.4003  +0.4096   +62.659    +8.373   +62.659     +761    +1998  resolved   FAIL
  invariant   1.0000  0.250   20480    0.02    5.89   +0.4248  +0.4096    +0.094   -14.551    +0.094      +30       +3  at-floor   PASS
  invariant   1.0000  0.500   20480    0.02   78.04   +0.5360  +0.4096    +0.784   -11.290    +0.784     +134      +25  at-floor   PASS
```

The floor-relative verdict is `PASS`/`FAIL` in lockstep with the legacy one at all 32 doses
and is omitted from the table above; the committed JSONs carry both.

### The three thresholds

| axis | last PASS | first FAIL | injected at the last PASS |
|---|---|---|---|
| coordinate leak, coverage at `a=1` | rho = 0.055, +0.8036 pp, z +2.446 | rho = 0.060, +1.0660 pp, z +4.077 | 1,126 rows (985 scoreable), each an exact codeword |
| coordinate leak, amplitude at full coverage | a = 0.10, +0.7683 pp, z +2.227 | a = 0.11, +1.3635 pp, z +5.927 | 20,480 rows at 1.00% injected power (SNR 0.01010) |
| rotation-faithful leak, coverage at `a=1` | rho = 0.40, +0.4248 pp, z +0.094 | rho = 0.50, +1.1971 pp, z +4.892 | 8,192 rows (7,217 scoreable), 100.00% oracle-decodable |

**The smallest injected leakage the gate flags is 1,229 of 20,480 released rows — 6.0% —
made exact, noiseless, coordinate-visible functions of their token.** On the amplitude axis
the smallest flagged dose is an injected power fraction of 1.21% (`a = 0.11`, SNR 0.01234)
applied to every released row. On the axis that models a real encoder leak, it is 50% of
released rows.

**The fraction of rows that must carry recoverable token identity before the excess clears
+1.0 pp is between 5.5% and 6.0%** for a coordinate-visible leak and **between 40% and 50%**
for a rotation-faithful one.

## Why the gate is blind below those doses

The zero-dose per-arm reading, reproduced bit-for-bit from the committed campaign artifact:

| arm | correct | z vs control | distinct classes predicted | modal share |
|---|---|---|---|---|
| `coordinate_plus_invariants` r0 | 506 | -17.782 | 1,216 | 0.212 |
| `coordinate_plus_invariants` r1 | **579** | **-15.492** | 1,203 | 0.376 |
| `coordinate_plus_invariants` r2 | 518 | -17.405 | 1,184 | 0.237 |
| `invariant_only` r0 | **1,073** | **+0.000** | **1** | **1.000** |
| `invariant_only` r1 | 1,073 | +0.000 | 2 | 0.999 |
| `invariant_only` r2 | 1,073 | +0.000 | 2 | 1.000 |
| `invariant_graph` r0/r1/r2 | 1,071 / 1,063 / 1,059 | -0.06 / -0.31 / -0.44 | 6 / 4 / 3 | 0.96 / 0.92 / 0.91 |

The gate takes the maximum Wilson upper bound over these nine. That maximum belongs to a
constant predictor, and a constant predictor is *immovable*: no injected content can change
what it emits, because it emits one class. So the reading stays at the floor until some
other arm overtakes 1,073 — and the only arm reading a coordinate-visible leak begins 494
rows short.

The coordinate arm's climb is smooth and starts immediately; the gate's response is a step:

| coverage | rows injected | coordinate arm | its z | gate excess |
|---|---|---|---|---|
| 0.0000 | 0 | 579 | -15.492 | +0.4096 |
| 0.0200 | 410 | 692 | -11.948 | +0.4096 |
| 0.0400 | 819 | 956 | -3.669 | +0.4096 |
| 0.0500 | 1,024 | 1,066 | -0.220 | +0.4147 |
| 0.0550 | 1,126 | 1,151 | +2.446 | +0.8036 |
| 0.0600 | 1,229 | 1,203 | +4.077 | **+1.0660** |

Everything left of the parity point at `rho ~ 0.05` is information the frozen attacker
demonstrably extracted and the gate statistic discarded. At `rho = 0.04` the arm recovers
377 of the 819 injected rows and the gate reading is unchanged to four decimal places.

At full coverage the same pedestal shows up on the amplitude axis: at `a = 0.05` — every
released row carrying a token-correlated component at 0.25% of its power — the coordinate
arm gains 132 rows and the gate reads +0.4096 pp, identical to undosed.

## Which arm family detects first

Each family sees only the leak in its own representation, and neither covers for the other.

| dose | coordinate family best z | invariant family best z | which family sets the gate |
|---|---|---|---|
| coordinate, rho 0.06 | **+4.077** | +0.031 | coordinate |
| coordinate, rho 1.0, a 0.5 | **+363.345** | +1.850 | coordinate |
| invariant, rho 0.50 | -13.297 | **+4.892** | invariant |
| invariant, rho 1.00 | +8.373 | **+62.659** | invariant |

In coordinate mode the invariant arms are inert: across every dose from `rho = 0.0025` to
`rho = 0.30` the best invariant arm moves by at most 5 rows out of 20,480, while the
coordinate arms go from 579 to 5,224. Only at the saturating dose (`rho = 1`, `a = 1`, every
row an exact codeword) do the invariant arms finally move, to z = +62.972 — and there
`invariant_only` is predicting 16 distinct classes rather than 1.

In invariant mode the reverse holds, and it holds *starkly*: at `rho = 0.5` the invariant
family gains 156 rows out of 9,001 scoreable injected rows — a 1.7% conversion — and that is
enough to fail the gate, while the coordinate arms sit at z = -13.297, no better than
undosed. The forensics finding that the invariant family wins on near-constant prediction is
confirmed here from the other direction: it is the family that responds to the
rotation-invariant channel, and it responds by discriminating a few more frequent classes,
not by recovering content.

The practical consequence: **the campaign's floor cell is decided by three constant
predictors, and its verdict is insensitive to everything the coordinate arms see until they
overtake 1,073 correct rows — a bar they start 494 rows below.**

## Is this a token-leakage detector or a distribution-shift detector?

The honest answer comes in two parts, and only one of them was measured here.

**Measured: it is not a sensitive detector of token leakage.** The injection changes no
marginal the gate could latch onto — row norms preserved to 4.8e-07, labels untouched,
control fixed at 5.2392578125% across all 32 doses — so the only thing added is token
identity. Under that constraint the gate needs 6.0% of released rows made *exactly*
recoverable before it moves, and 50% when the leak respects the wire rotation. Below those
doses it returns its clean-bundle reading to four decimal places.

**Not measured: whether gross distributional change is what does move it.** No dose in this
sweep perturbs the released frame's marginals, so this experiment cannot say directly what
the gate would do with one. What it can say is what the gate actually thresholds, and that
is not recovered content:
- The rotation-invariant signature of the leak is large and clean long before the gate
  reacts. At `rho = 1`, `a = 0.05` the mean within-frame cosine between rows sharing a token
  is +0.04140 against +0.00093 for rows that do not — a 44-fold separation over 16,940
  same-token pairs, present in every frame — and the gate reads its undosed value to four
  decimal places. At `a = 1` the same-token cosine is exactly
  1.000: rows sharing a token are *identical vectors*, the signature the runner's own chaff
  sampler is written to avoid ever creating (`chaff_sample`, "identical rows inside one frame
  would be trivially detectable (unit cosine) and must never occur"). Up to 40% coverage the
  gate does not react to it.
- What the gate thresholds is **arm ranking against a constant predictor**, and that comes
  apart from recovery in both directions. At `rho = 0.04` the coordinate arm recovers 377
  injected rows and the gate does not move at all. At `rho = 0.50` in invariant mode the
  invariant arm converts only 156 of 9,001 exactly-leaking rows — 1.7% — and the gate fails.
  The same verdict change is produced by two wildly different amounts of recovered content.

## Reproducing every number

All work runs inside `split-inference:spark` on `gx10-odysseus.nord`, on CPU — the frozen
attacker never touches the GPU. The bundle is retained and root-owned under
`~/experiments/results/training/deleg6040/bundles/` and is not committed (`*.pt` is
gitignored).

The seven sweeps were run from a staged checkout at `~/gate_sens` rather than from
`~/dtraining`, so that a long-running job's working tree was never modified. The staged copy
is this release's `attacker/`, `privacy_runtime/` and five `bin/deleg6040_*` files
(`gate_sensitivity.py`, `gate_sens_sweep.sh`, `gate_recalibrate.py`, `verify_stats.py`,
`bundle_forensics.py`);
`attacker/attacks/latent_probe.py` there has sha256
`cdb9d0a47af3ef8223d4e0af1bdf82994633e1d07a355c8ab5bacb2be6757e3b`, identical to this
release. Its only difference from the campaign's original copy is the diagnostic
`--dump-eval-predictions` flag added for the bundle forensics audit, which collects the
per-arm argmax and touches no scoring — proven again here by the zero dose reproducing
the recorded
artifact bit-for-bit. Substitute the repository checkout path for `$HOME/gate_sens` when
running from a full checkout.

```
docker run --rm --network host --ipc host \
  -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e CONTAINER_IMAGE=split-inference:spark \
  -v $HOME/experiments:/workspace/experiments -v $HOME/gate_sens:$HOME/gate_sens \
  -w $HOME/gate_sens split-inference:spark \
  bash bin/deleg6040_gate_sens_sweep.sh A     # then B C D E F G
```

Each sweep writes one JSON of dose points to
`/workspace/experiments/results/training/deleg6040/gate_sens/`. Sweep A's first dose is
`coverage 0`, which the tool asserts against the committed
`deleg_6040_conv10k_split14_attacker.json`; it aborts rather than continues if the undosed
re-score is not identical. Adding `-e REUSE=1` re-derives every report from the
frozen-attacker artifacts already on disk without scoring again.

One dose is roughly 4 minutes of wall clock on an idle 20-core host and about 40 core-minutes
of work; under the contention present for most of this session it ran at 8-13 minutes.

The dose-response table above is printed by

```
python3 bin/deleg6040_gate_sensitivity.py --table \
  paper-data/collected/diagnostic/deleg_60_40/gate_sens_*.json
```

and `python3 bin/deleg6040_gate_sensitivity.py --self-test` checks the injection algebra
(exact codeword at `a=1`, bit-identical wire at `a=0`, norm preservation, nested coverage).

Committed artifacts, all under `paper-data/collected/diagnostic/deleg_60_40/`:

| path | what |
|---|---|
| `gate_sens_A_coordinate_coverage.json` | 11 doses: coordinate leak, coverage sweep at `a=1`, including the zero dose |
| `gate_sens_B_coordinate_amplitude.json` | 6 doses: coordinate leak, amplitude sweep at full coverage |
| `gate_sens_C_invariant_coverage.json` | 4 doses: rotation-faithful leak, coverage sweep at `a=1` |
| `gate_sens_D_invariant_amplitude.json` | 2 doses: rotation-faithful leak, amplitude sweep at full coverage |
| `gate_sens_E_coordinate_refine.json` | 3 doses: coverage refinement across the coordinate threshold |
| `gate_sens_F_coordinate_amp_refine.json` | 3 doses: amplitude refinement across the coordinate threshold |
| `gate_sens_G_invariant_refine.json` | 3 doses: coverage refinement across the invariant threshold |
| `gate_sens_rescore/inj_*_attacker.json{,.jsonl}` | the 32 raw frozen-attacker artifacts and journals, one per dose |
| `gate_sens_rescore/gate_sens_bundle_digests.txt` | sha256 of every injected bundle and prediction dump |

## Integrity checks that were run

- **The zero dose reproduces the campaign artifact bit-for-bit.** The committed
  `gate_sens_rescore/inj_coordinate_rho0_a1_attacker.json` is equal to
  `deleg_6040_conv10k_split14_attacker.json` in its `results`, `summary` and `config`
  blocks: all nine arms, all counts, all Wilson bounds. The tool asserts four gate
  quantities against the recorded artifact before continuing and exits non-zero otherwise
  (`check_zero_dose`), and the assertion fired clean on both the scoring run and the
  `REUSE=1` re-derivation.
- **The reports were re-derived under the final tool with the scored bundles unchanged.**
  Every injected bundle's sha256 was snapshotted, all seven sweeps were re-run with
  `REUSE=1` (no attacker re-scoring), and all 64 `.pt` digests compared identical
  afterwards. The digests are committed at
  `gate_sens_rescore/gate_sens_bundle_digests.txt` and each sweep JSON carries the digest of
  the bundle its dose was scored on.
- **Scoring is deterministic under machine load.** Sweep E was run twice — once at load
  average ~29 with three other containers competing for the 20 cores, once at load average
  ~1 — and all three doses returned identical counts, excesses and z values.
- **The gate arithmetic is not this tool's.** Every excess, floor, z and verdict comes from
  `deleg6040_gate_recalibrate.recalibrate`, the campaign's own re-derivation from raw
  counts. Any committed dose can be re-gated independently:
  `python3 bin/deleg6040_gate_recalibrate.py --artifact paper-data/collected/diagnostic/deleg_60_40/gate_sens_rescore/inj_coordinate_rho0.06_a1_attacker.json`.

## Known limits

These bound what the numbers above license. They are limits of the *experiment*, not
caveats added after the fact.

- **The injected code is statistically independent of the model.** The codebook is
  drawn at random and has no relationship to where a token's latents actually sit. Real
  encoder leakage would more likely *strengthen structure already present*, and such a leak
  could be easier or harder to detect than this one. What is measured is the gate's response
  to newly added, independent token information — the only shape that can be dosed exactly.
- **At `a = 1` the injection replaces the row's own content rather than adding to it.** The
  amplitude knob interpolates at fixed norm; it does not superpose. So the high-amplitude
  end of the sweep is "the row is a codeword", not "the row is its old self plus a codeword".
  The low-amplitude end (`a <= 0.15`, where the thresholds sit) is close to additive:
  `sqrt(1-a^2) >= 0.989`.
- **The dose is applied to the retained post-training bundle, not through training.** The
  defence does not get to react to the leak, and neither does the minimax game that produced
  the encoder. This is deliberate — the question is the sensitivity of the *detector* with
  the defence held fixed — but it means the curve says nothing about whether such a leak
  could arise, or survive training, in this stack.
- **It also inherits every limit of the bundle itself.** As the external review's
  finding 1 records, the bundle is one post-training
  forward-latent view: no gradients, no training history, no protocol state. A gate
  sensitivity measured on that view is a sensitivity to leakage *in that view*.
- **The `invariant` mode's distributional equivalence is exact, its interpretation is not.**
  The construction is equal in law to injecting the codeword into the trusted latent before
  `release()`. But it injects a direction of *fixed relative magnitude* (`a` is a share of
  the row's own norm), which is the gauge-equivariant choice; an absolute-magnitude
  pre-rotation leak would interact differently with a token-scale gauge. This cell runs with
  `secret_token_gauge=False`, so the distinction does not arise here, and the result should
  not be carried to a gauged cell without re-running.
- **One bundle, one seed, one codebook.** All 32 doses use codebook seed 20260819 on the
  split-14 10k cell. The arm-level movement at very small doses is within probe-training
  variation and is not monotone: the best coordinate arm gains 23 rows at `rho = 0.005` and
  17 at `rho = 0.01`. Only the threshold locations, which sit far above that noise, are
  claimed.
- **Both partitions are dosed at the same coverage.** A leak affecting a fraction of rows
  gives the attacker proportionally few training examples of it, and part of the measured
  threshold is that learning cost rather than the gate statistic. The `coord d` column
  separates the two: it shows what the arm actually recovered at each dose, and the gate
  ignoring 377 recovered rows at `rho = 0.04` is a statement about the statistic alone.
- **Rows are dosed uniformly across real and chaff rows.** Chaff rows are recycled real
  latents carrying genuine token labels (`chaff_push`/`chaff_sample`), and the gate scores
  them, so injecting into them is injecting true token information into rows the gate counts.
  No real/chaff split of the injected budget was attempted; the bundle forensics audit (not
  included in this release) records the method.
- **The `resolved` and +1.0 pp criteria are not separated by this grid.** Both cross between
  `rho = 0.055` and `rho = 0.060`, and between `a = 0.10` and `a = 0.11`. A finer grid could
  separate them by a few tenths of a percent of rows; it would not change the conclusion.
- **The thresholds are for this bundle; the cell's own excess has run-to-run spread.**
  The metric-variance measurement (not included in this release) records sd 0.0481 pp on
  the excess across five byte-identical runs of this configuration (+0.4086 to +0.5210),
  because the DP draw and the gauge are redrawn per
  run. Every dose here re-scores one fixed retained bundle, so that spread is absent from the
  curve — but a different run of the same cell would start from a different undosed excess
  and could place the threshold somewhat differently. The step this sweep resolves the
  threshold with, +0.8036 to +1.0660 pp, is 2.5 times the 0.1054 pp that work gives as the
  difference needed to call two readings distinct.
- **The z values are not materially inflated by frame clustering.**
  The design-effect measurement (not included in this release) records the gate arm's
  frame-level design effect at 0.847-1.158 over 15 cells, so the Wilson and binomial
  treatment the gate applies — and that
  this document reports unchanged — is approximately calibrated at the frame. That work also
  finds the dependence that does exist is not at the frame.

## What this changes

Nothing about the verdicts already recorded: every cell that failed the gate still fails it,
and the bundle forensics audit's false-positive result stands. What changes is what a
**pass** means.

- The claim "this configuration passes the privacy gate at the statistical floor" should be
  read as "no arm of this attacker beat a constant predictor by more than 117 rows in
  20,480". It does not exclude a coordinate-visible leak affecting up to ~5.5% of released
  rows, nor a rotation-faithful one affecting up to ~40%.
- This cell's +0.4096 pp lands exactly on its own statistical floor, which is the middle of
  the region this calibration shows to be uninformative: the same reading is produced by a
  clean bundle and by one in which 819 rows are exact copies of their token. It remains a
  correct reading of the statistic; it is not evidence of an absence of leakage at the scale
  this experiment can inject.
- **This calibration measures one cell** (`deleg_6040_conv10k_split14`, n = 20,480), which
  has its own floor and its own arm counts; no other cell was re-run through this sweep.
  What transfers is the mechanism, not the number: the bundle forensics audit found the
  same constant-predictor arms setting the maximum in all three cells it examined, so the
  pedestal is expected wherever the winning arm's modal class share is near 1. Establishing
  the threshold for another cell requires re-running this sweep on its bundle.
- The pedestal is fixable, and cheaply, without touching the attacker: the constant-predictor
  arms could be excluded from the maximum (any arm whose modal class share exceeds, say,
  0.99 recovers nothing by construction), or each arm scored against *its own* undosed
  baseline rather than against the shared control. Either would have flagged the `rho = 0.02`
  dose — 410 exact rows, 113 recovered — that the present gate reads as identical to clean.
  This calibration does not propose a replacement gate; it measures the one in use.


