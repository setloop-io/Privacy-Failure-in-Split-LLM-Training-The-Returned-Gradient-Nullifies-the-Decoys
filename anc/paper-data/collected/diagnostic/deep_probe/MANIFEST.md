# Deep-inversion probe artifacts (positive-control contrast)

Produced 2026-08-25/26 on tln by `attacker/attacks/deep_inversion_probe.py`.
That module is included in this package at `attacker/attacks/deep_inversion_probe.py`
(registered as `deep-inversion`), behavior-faithful, with the matched shuffled
null and final-epoch accuracy added on top of the artifact protocol.

| file | condition | bundle | best acc | majority | Wilson u95 | excess |
| --- | --- | --- | --- | --- | --- | --- |
| `real_naked_split14_deepprobe.json` | naked D=1024 | real_naked split14 capture | 28.27% | 4.35% | 28.54% | **+24.19 pp** |
| `defended_d64_deep_probe.json` | defended D=64 (noise 0.35, chaff 48) | defended bundle | 5.29% | 5.27% | 5.56% | **+0.29 pp** |
| `defended_d64_deep_probe_r3.json` | defended D=64, 3 restarts | defended bundle | 5.29% | 5.27% | 5.64% | **+0.37 pp** |
| `defended_d1024_deep_probe.json` | defended D=1024 (dimension-matched) | defended bundle | 5.30% | 5.29% | 5.56% | **+0.27 pp** |

Protocol notes: 4-layer, 8-head Transformer encoder (norm_first), 50 epochs,
batch 128 blocks, AdamW lr 3e-4. `best_eval_acc` selects over epochs on the
evaluation set — a selection-on-test reading, stated not hidden; the ported
module also reports final-epoch accuracy and offers `--shuffled-null`.

The source bundles these ran on are the cluster's naked split-14 capture and
defended bundles (not packaged); an in-package validation of the ported module
against latent_isolation bundles (not included in this release) lives in
`outputs/deep_probe_validation_2026-08-27/`.
