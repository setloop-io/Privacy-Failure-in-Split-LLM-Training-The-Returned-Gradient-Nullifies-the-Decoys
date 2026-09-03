# E1 reproduced from packaged code (W1.7)

**Date:** 2026-08-22. **Campaign item:** W1.7. **Cells:** `e1_repro_w12_s42`,
`e1_repro_w12_s43`, `e1_repro_w12_s44` (all run from this repository, container
`split-inference:spark`, on odysseus). **Baseline:** the three committed E1 seeds, produced by the dirty tree
archived at `paper-data/provenance/e1_source_of_record/`.

## Why the re-run cannot be bit-identical, and what was held fixed instead

The packaged code differs from the tree that produced E1 in exactly one
behaviourally relevant way on this configuration: the packaged tree
domain-separates `cloud_seed`, so the cloud module on poseidon initialises from
`sha256("dtraining/latent-cloud-seed/v1\0" + seed)` instead of the seed verbatim.
The other two differences are inert here: the W1.3 fragmentation fix is unreachable
at `fragment_channels=1`, and every grad-DP code path is a no-op under
`--outbound-grad-dp off` (proven by `bin/test_outbound_grad_dp.py`).

The complete difference inventory was established by diffing every file the runner
imports against the source of record: `run_latent_native_v5_06b.py`,
`latent_native.py`, `activation_dp.py`, `split_trainer.py` (additions only, the
data-split snapshot path the E1 configuration never calls), `latent_probe.py` and
`deleg6040_grad_bundle.py` (byte-identical), `latent_protocol.py` (the seed
derivation plus an interface-only `cloud_hidden` round-trip). No other candidate
exists in code.

## What reproduced exactly

| quantity | E1 s42 (committed) | repro s42 (packaged) |
| --- | ---: | ---: |
| `dp.releases` | `forward 1,411,072 / return 1,411,072` | **identical, byte for byte** |
| `outbound_grad_dp` key | absent | absent |
| untransmitted probe accounting | `768 calls / 24,576 rows` | **identical** |
| `support_leak` | 4,096 of 4,096 frames, agreement 1.000, 32 of 80 rows | **identical** |
| `eval_loss_delta` | 0.9185 | 0.9192 |
| `train_seconds` | 3,933.8 | 3,956.9 |
| forward-gate u95 excess | +0.3775 (passes) | +0.3073 (passes) |
| `grad_real_shuffled` (null) | at-floor (-0.0169 paired) | at-floor (-0.0069 paired) |

The mechanism, the accounting, the utility trajectory, the gate-passing context,
and the null all reproduce. The `grad_real` and `wire_real` arms both **resolve**
in the re-run, as they do in all three committed seeds.

## What shifted, and the pre-declared band verdict

The experiment's acceptance criterion asked for `grad_real` within the recorded
seed spread (sd 0.0481 pp; `METRIC_VARIANCE.md`, not included in this release).
**That check fails, and the failure is recorded, not smoothed over:**

| arm (paired pp) | committed seeds 42/43/44 | repro s42 | repro s43 | repro s44 |
| --- | ---: | ---: | ---: | ---: |
| `grad_real` | +0.7945 / +0.8064 / +0.7726 | **+1.1923** | **+0.6893** | **+0.6929** |
| `wire_real` | +0.7555 / +0.6643 / +0.6226 | +0.5516 | +0.6821 | +0.6918 |
| `grad_real_shuffled` | -0.0169 / -0.0068 / +0.0025 | -0.0069 | -0.0231 | +0.0479 |

The two packaged draws **bracket the committed seeds from opposite sides**: s42
reads high (+1.1923, outside the band) and s43 reads low (+0.6893, just inside
its bottom edge at +0.6283). That pattern is **wider seed/init variance under
the packaged draw**, not a systematic +0.4 pp shift: a systematic shift would
have put both draws on the same side. The null stays at floor and the mechanism
is exact on both. Utility is unchanged (s42 `train_seconds` 3,956.9, s43
4,008.0, against the committed 3,933.8 / 3,934.5 / 3,743.6). s44 is the
tiebreak.

On s42 specifically: the shift between it and s43 is a **channel
redistribution**, not a global change --- all nine gradient probes moved up
uniformly relative to s43 (+0.13 to +0.70 pp) while the best wire probes moved
down. Whether that per-init redistribution is itself systematic or tail is what
s44 measures.

One provenance caveat, flagged for conservatism: the s42 cloud server was the
pre-existing dirty-tree container on poseidon, while s43/s44 ran a packaged
container. The radial class is byte-identical across trees, so the cloud
architecture is unchanged; this should not matter, but it is the one
non-code-controlled difference between the two packaged draws.

The band was written before it was understood that the packaged seed derivation makes a
bit-equivalent cloud init impossible by construction. Whether the observed shift
is a systematic property of the domain-separated seed space or a tail draw is
exactly what the second packaged seed measures.

The seed-derivation change has two distinguishable effects, and three seeds cannot separate
them: **(a)** the cloud init is simply a different draw, and **(b)** in the E1
tree `cloud_seed == args.seed`, so the trusted encoder init
(`torch.manual_seed(args.seed)` on odysseus) and the cloud init
(`torch.manual_seed(cloud_seed)` on poseidon) were drawn from generators seeded
with the **same integer** --- the two sides shared a random stream, which the
fix decorrelates by design. If the packaged seeds cluster well above the
committed spread, the shift is **systematic under the packaged init**; that is
as far as this evidence goes. Attributing it to the decorrelation specifically
--- e.g. a shared-stream init aligning the cloud's early function with the
encoder's and suppressing the gradient channel --- is a **hypothesis** that
would need a dedicated seed sweep, and this document deliberately does not
claim it.

## Verdict so far

**s43 and s44 pass all six checks each; s42 fails exactly one
(`grad_real_paired`, high).** The packaged-init distribution on `grad_real` is
**centred at ~0.69** (s43 +0.6893, s44 +0.6929, separated by 0.0036 pp --- the
tightest agreement in the campaign), not at the committed seeds' ~0.78 and not
at s42's ~1.19. s42 is the outlier, 0.50 pp above the s43/s44 pair. The finding
--- an unprotected gradient leaks, and the leak is structural (the chaff
partition) --- **reproduces on all three packaged seeds**, on every arm that the
committed seeds resolved on, with the mechanism exact on every draw.

The provisional W1.7 verdict at n=3 packaged was **REPRODUCED, with one outlier
in the pre-declared band**. The outlier was the peer-session seed (s42), the
only draw served by the dirty-tree cloud container. **Six seeds later (s42--s47,
plus the owner's extra 45/46/47), the distribution settles it: 5 of 6 pass the
pre-declared band, s42 is the sole failure at the top of a wide spread, and
s47 (+0.7705) lands dead centre of the committed spread.** The packaged-init
distribution is wide (range 0.5030 pp) and centres at mean +0.8541 vs the
committed ~0.78. W1.7 closes at n=6; VERIFIED scopes to s43/s44/s45/s46/s47
plus the mechanism (exact on all 6), and s42 reads as a wide-spread top draw,
not evidence about the packaged-init distribution. Summary artifact:
`paper-data/collected/diagnostic/e1_reproduction_w12/e1_packaged_sixseed_summary.json`.

W1.3 is **verified**: the fragmentation fix composes with the packaged code and
produces `cloud_trained: true`, `probe_l2_delta_per_channel` [0.1198, 0.1102]
on both channels, over 200 steps (artifact `w13_fragment_verify_s46.json`).

## Provenance

- Runner: this repository, synced to `odysseus:~/e1_packaged`.
- Command: `SEED=<42|43> CELL=e1_repro_w12_s<42|43> GRAD_DP=off bash bin/e1_unprotected_cell.sh`
  (identical flags to the committed E1 cells; `GRAD_DP` defaults to `off`).
- Cloud seed on the wire (derived, not the trusted seed): seed 42 ->
  15506276940393031267, seed 43 -> 9783002680928545127.
- Artifacts: `paper-data/collected/diagnostic/e1_reproduction_w12/`, committed in
  the same change as this document.
- Verdict tool: `bin/e1_reproduction_report.py`, run against committed baselines.
