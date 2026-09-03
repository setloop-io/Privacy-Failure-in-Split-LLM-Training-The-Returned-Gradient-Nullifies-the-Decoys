# Latent-native v11 — external-attack validation program

v11 closes the gap the 2026 literature search flagged: our floor claim was
measured only against trained probes, while the field's strongest attacks
are *matching* algorithms.  Stages:

| Stage | Content | Status |
|---|---|---|
| v11.0 | VMA-style centroid matching + permutation order-recovery arms (`attacker/attacks/latent_matching.py`, adapted from Hidden No More (ICML'25) and arXiv:2505.18332) run against regenerated v9.2 bundles (0.6B + 35B) | **complete — both matching families fail** (0.02-0.03% / 1.5% vs ~5.9% majority control); frozen gate re-passes (+0.541/+0.285 pp) |
| v11.1 | Port the defense into the VFLAIR-LLM harness (KDD'25) for field-comparable numbers | not run (their defense API is tensor-transform hooks; ours is architectural — needs a custom module); differentiation handled in the paper's related-work section |
| v11.2 | MI-budget per-layer estimation (I(activation; prompt)) — the formal-guarantee path from UCLA 2606.11592 / Fraunhofer WACV'26 | **complete, honest negative**: CE-based lower bounds are vacuous at LLM vocabularies (bound = 0 while top-1 = 55-59%); documented with the reason |

The frozen `latent_probe` gate is re-scored on every bundle and remains the
binding gate; the new arms extend, never replace.

`launchable: false`.
