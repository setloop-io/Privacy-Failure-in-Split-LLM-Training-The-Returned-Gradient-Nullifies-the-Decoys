# Latent-native v10 — multi-segment delegation + corrected public pretraining

v10 attacks the delegation-share question directly: v9.2 delegated one
4-layer segment (~14% of the 0.6B); v10.0 delegates TWO segments through two
independently-protected boundaries.

## v10.0 — two-segment delegation

Layout (0.6B, 28 layers): prefix 0-15 | segment A 16-19 (UCN) | private
island 20-21 | segment B 22-25 (UCN, optionally a different node) | tail
26-27. Delegated share: 8/28 = 28.6% of layers.

Each boundary has its own private encoder/decoder, DP clip+noise, fresh
v2-stream gauges (rotation + permutation; no scale gauge — the v9.2 radial
winner configuration), and its own chaff pool.  Each segment is a separate
isolated cloud session (`--cloud-url-a` / `--cloud-url-b`).  Distillation
targets come from the frozen teacher path per segment; zero-cloud controls
are measured per segment.

Runner: `bin/run_latent_native_v10_2seg.py`.  Question under test: does
chained surrogate error accumulate past the utility gate, and does privacy
hold at both boundaries against the frozen attacker.

## v10.1 — E5 done right

The v9.3 failure (public pretraining + halved private phase) was diagnosed
as insufficient private minimax pressure.  v10.1 reruns it with a
full-length private adversarial phase: 4000 public + 2000 private steps on
the v9.2 winner.

Status: `launchable: false`.  The v10.1 cell passed the published gate
(+0.732 pp); under the paired cluster-aware statistic it resolves above the
constant control (+0.8577 pp, 95% CI [+0.616, +1.108]; see
`docs/experiments/PAIRED_RESCORE_2026-08-22.md`).  The campaign report and
raw artifacts are not included in this release.
