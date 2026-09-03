# W7.8 independent adversarial review round: verdicts and fixes applied

**Date:** 2026-08-27. **Scope:** an independent automated adversarial review of
the expanded manuscript and its bibliography. Every finding was validated
finding-by-finding against committed artifacts before any edit was applied.
This file records the fixes; no fix was applied unvalidated.

## Verdicts

| # | Finding | Verdict | Fix applied |
| --- | --- | --- | --- |
| 1 | depth monotonicity / gating attribution | PARTIAL — gate is bracketed 8–12 (not "11–12"), partition exact at every depth; language overstated | Abstract + §6 paragraph rewritten: token-recovery gate vs exact partition |
| 2 | "paired over matched null" misdescribes statistic | CONFIRMED — paired advantage is arm-over-constant-control; null is a parallel twin arm | §3 rule rewritten; §5.2 lead + table captions corrected |
| 3 | threshold semantics | CONFIRMED — +1.0 pp applies to arm-over-control; "order beyond threshold" was wrong (it straddles the gate) | §4.2 rewritten; tab:thresholds rows corrected |
| 4 | W24 calibration controls not shuffled-null | CONFIRMED — zero-dose source + constant control; nulls enter at scoring | tab:dose caption rewritten |
| 5 | CE has no numerical threshold | CONFIRMED — threshold_pp is a reading rule string | tab:thresholds row corrected |
| 6 | abstract bounds full-view | CONFIRMED — five declared-unmeasured families | Abstract bound recorded |
| 7 | committed vs cluster-only | CONFIRMED — fwdmem_scratch, red-team scorer, raw transcript not in repo | Abstract + §5.4 instrument-tracking sentence |
| 8 | hierarchical bootstrap over seeds | CONFIRMED — applies to arm-vs-null contrasts (e1_3seed_summary), not the per-seed CIs | §3 rewritten; §5.2 contrast sentence added (exploratory) |
| 9 | width D=128 / partition exact / single-seed | CONFIRMED — token recovery dies at 128; partition stays exact; all nine cells single-seed (seed 42) | Abstract + §6 intro + fig:shape caption |
| 10 | AUC range 0.058–0.159 | CONFIRMED — not a constant ~0.09 | §4.1 + fig:dose caption span stated |
| 11 | literature overclaim | CONFIRMED — bounded to the three audited evaluations | Abstract, §1, §5.3, §7.1 heading + body, Conclusion |
| 12 | four bib entries wrong | CONFIRMED — vs live arXiv/ACL | refs.bib authors/venues corrected |
| 13 | nine vs seven cells | CONFIRMED — table showed seven of nine | tab:shape now shows all nine; note corrected |
| 14 | authorship unresolved | CONFIRMED — marked as open in byline | \author now flags open authorship |

## Residual defects found during validation (all fixed)

- "10k to 100k" saturating text → 40k/100k, 2.5x
- s42 rounding 1.1923 (matches committed summary)
- tab:sixseed provenance bound to s43--s47; s48--s50 post-freeze captures
- rare_token "first resolving" → onset wording (0.04 reads +0.32 pp, below gate)

## Caveats

- The review's raw output was not committed; each finding was validated
  one-by-one against the committed artifacts and is recorded here.
- The four bib verifications come from live arXiv/ACL records, not committed
  artifacts; manuscript numbers trace to committed artifacts only.
