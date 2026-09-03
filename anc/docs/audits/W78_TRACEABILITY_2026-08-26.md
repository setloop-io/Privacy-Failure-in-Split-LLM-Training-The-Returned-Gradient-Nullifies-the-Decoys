# W7.8 traceability: manuscript numbers re-verified against artifacts

**Date:** 2026-08-26. **Campaign item:** W7.8.
Load-bearing manuscript numbers re-derived from committed artifacts before the
manuscript expansion. Every check below was run against the committed artifacts
it names.

## Verified (traceable and matching)

- Six-seed paired advantage: `+0.69..+1.19 pp`, mean `+0.8541` (e1_reproduction_w12).
- Support leak: 4,096/4,096 frames, row agreement 1.000, all six seeds and all
  three confirmation seeds.
- Post-freeze confirmation: `+0.6456 / +1.5008 / +0.9929`, nulls at floor.
- W5.6 nine cells: every cell of `w56_gradaudit_summary.json` re-derived
  (saturating across budgets, depth ladder, width cells).
- W2.4 onsets: `token_top1` between coverage 0.04 and 0.06, steep through 0.10;
  `rare_token_top1` from 0.04; CE flat until the leak dominates.
- Red team: its committed table matches the artifact values (nine seed cells).
- W3.4 manifest: checkpoint tarball sha256 re-computed, matches `MANIFEST.md`.
- Family map: five unmeasured families with reasons; frozen thresholds.
- Freeze: protocol `status: frozen`, freeze SHA `222c4a7` resolves in history.
- E1 archival hashes: `e1_runtime_sources.tar.gz` and dirty patch re-computed
  against the manifest. PASS.

## Corrections applied to the manuscript

Recorded, not quietly fixed. Each row: the defect, the artifact truth.

| # | Defect in the draft | Correct |
| --- | --- | --- |
| 1 | "10x the gradient exposure (100k vs 10k)" | The low-budget cell is **40k** steps (a2b); exposure ratio is **2.5x**; +0.9302 (40k) vs +0.9066 (100k). |
| 2 | null bound "|v| <= 0.07 pp" | s45's null is -0.070733; bound corrected to **<= 0.08 pp (and at floor, verdict-backed)**. |
| 3 | "membership_auc stays at the 0.5 floor" | Inj readings are AUC-0.5 in [0.058, 0.159], flat and dose-insensitive. The metric was **falsified as a channel** and is diagnostic only. |
| 4 | "resolves at >= 12 delegated layers (split 13/14)" | Per `split_trainer.py` partition math (resume_after=26): split 13 -> **12**, split 14 -> **11**, split 17 -> **8**, split 19 -> **6** delegated. Resolves at 12 and 11; at floor at 8 and 6. |
| 5 | "six seeds under the frozen instrument" | The six packaged seeds are **pre-freeze** (exploratory) per the W5.7 summary note; s42 was served by the pre-existing dirty-tree cloud container, s43--s47 packaged-to-packaged. Confirmatory = s48--s50 only. |
| 6 | forward-membership +0.068 falsification | A correction record is committed; the falsification bundle (`fwdmem_scratch`) lives on the cluster, not in the repo. Manuscript cites it as a recorded correction. |
| 7 | red team "<= 1e-6 pp" | Recorded in the committed W5.4 report; the red team's scorer and outputs are deliberately not committed, so the claim is **as recorded, not independently re-derivable from this repo**. |
| 8 | "180,636 events" | As recorded in the committed manifest; the raw transcript is on the cluster (owner's 100 MB policy). The committed attacker-side index is a different object (90,624 entries). |
| 9 | W8.4 differentiation missing | VFLAIR-LLM and MIXGUARD appear in no committed audit; differentiation written into the manuscript's related-work section. |

**The remaining inline numbers in the manuscript are drawn from the
verified list above only.**
