# Latent-native v12 — attacking the cloud-capacity ceiling

Every previous capacity axis saturated the Gram-kernel family: experts
(8→32), depth (2→11), width (16→128) all left cloud contribution flat
(~1.4–1.5 on the 0.6B).  v12 changes the function class.

| Stage | Content | Status |
|---|---|---|
| v12.0 | `invariant_mlp` cloud kind: per-row MLP over gauge-invariant features (norms + Gram-row statistics, position channel excluded) gating norm scale and Gram-message direction on top of the monomial skeleton — a learned nonlinear family.  Scale gauge OFF (norms are read), as with the radial kind; mechanism test covers rotation+permutation equivariance | complete (s42): published gate PASS (+0.404 pp); paired advantage +0.1446 pp [+0.073, +0.215] over the constant control |
| v12.1 | LM-head surrogate delegation (the largest single matmul in the step; output-side exposure → output-inversion arm required) | not run |
| v12.2 | 10k-step convergence cell (v9.2 winner + public pretraining) | complete (s42): published gate PASS (+0.417 pp); paired -0.0659 pp [-0.209, +0.074] (at floor) |

Reference cell to beat (0.6B, D=64, chaff-48, K=1): cloud non-bypass
1.488, loss delta +0.061, attacker excess +0.411 pp, ratio 1.81.

Final cells are re-scored under the paired statistic in
`docs/experiments/PAIRED_RESCORE_2026-08-22.md` (`v12_invmlp_06b_s42`,
`v12_converge10k_06b_s42`).

`launchable: false`.
