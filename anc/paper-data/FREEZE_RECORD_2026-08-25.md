# F.1 Freeze Record — 2026-08-25

**This is a recorded decision, not a silent one.**
Freeze commit: `222c4a75a8cdeda5ea9c8e99110693d95ceb217a` (`222c4a7`, the commit
that flips `evaluation_protocol.json` to `frozen`). Everything before this
commit is **exploratory**; everything after is **confirmatory**, and the
manuscript labels which.

## What is frozen

| Surface | State | Basis |
| --- | --- | --- |
| The defense | **Frozen** | `--outbound-grad-dp clip_noise`; the defense the confirmation arm runs |
| The protocol | **Frozen** | `evaluation_protocol.json` at this commit; the five required arms, nine attack families, four emitters |
| Per-metric thresholds | **Frozen** | from the W2.4 dose-response curves and the W2.6 family map, below |

## Pre-declared thresholds (frozen)

From `paper-data/family_metric_map.json` and the W2.4 calibration matrix:

| Metric | Threshold | Basis |
| --- | --- | --- |
| `token_top1` | **+1.0 pp** over the matched null | W2.4 sharp injection onset at coverage ~0.06-0.10, budget- and frame-invariant |
| `rare_token_top1` | **+1.0 pp** over the matched null, with `raw_recovery_pct_by_probe` reported alongside (degenerate-control flag) | most sensitive metric; on at coverage 0.04 |
| `token_cross_entropy` | **arm minus matched-null**, never raw | proper scoring rule; a confidently-wrong null inverts it (W2.3 s45 +21.1) |
| `membership_auc` | **diagnostic only** — no headline threshold | falsified as a channel (2026-08-24); the constant control is vacuous |

## What is frozen OUT (declared, not silent)

Per `evaluation_protocol.json` `open_requirements` and the W2.6 map, these have **no
representation-matched positive control** and are excluded from the confirmation
matrix's headline claims, recorded as unmeasured surfaces:

- `membership_property` AUC (needs randomized-assignment run)
- `property_auc` (no property labels in any capture)
- `semantic_recovery` (no similarity measure chosen)
- `response_side_recovery` (no response-side capture)
- `timing_metadata`, `active_perturbation` (no applicable metric)
- Seams S2/S4/S6 from `docs/THREAT_MAP.md` (no attack family)

These are **declared unmeasured**, not dropped. The confirmation matrix reports them
as such; the manuscript states the adversary view's boundary at the freeze.

## The confirmation rule from this point

- Thresholds are set **before** the defended confirmation arm runs (freeze_rules[0]).
- Adaptive red-team implementations stay separate from defense tuning (freeze_rules[1]).
- MINE / Donsker-Varadhan values are not privacy certificates (freeze_rules[2]).
- No complete-cloud claim until every required arm and control is recorded (freeze_rules[3]).
- Post-freeze confirmatory replication (W5.7) uses **genuinely post-freeze captures**,
  not re-scored pre-freeze runs.

## Validation

Validated by `bin/validate_evaluation_protocol.py` (frozen protocol must
state candidate_frozen).

## Re-freeze note (2026-09-02)

A documentation cleanup pass removed development-process references (plan/task
framing, delegation-workflow notes, stale paths to files not shipped in this
release) from Markdown documents and code comments. No data files, result JSONs,
protocols, or numeric values were modified. `SHA256SUMS.txt` was regenerated
over the cleaned tree; the frozen result surfaces listed above are unchanged.
