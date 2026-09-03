# W3.4 complete capture -- artifact manifest

The raw transcript is too large for git (1.9 GB compressed, GitHub limit 100 MB).
Same policy as the .pt bundles: it lives on the cluster; the repo carries the
verification manifest and the hashes.

| artifact | where | sha256 |
| --- | --- | --- |
| cloud-side transcript (3 seeds, verified) | poseidon `~/experiments/results/training/w34_cloud_transcript_v2.tar.gz` | `0e7189401015a693ecb07c6dfe1e31484bcbceee8e7a95dc970b9040fe999fc4` |
| trusted-side checkpoints (3 seeds) | `w34_trusted_checkpoints.tar.gz` (in this dir, ~small) | `ddbd2f0e8a92fbdf6693e1b03d29be3e3505e9784d384e0eb041d73a8498cd2a` |

Verification: hardened verifier passes -- 3 sessions, 30,000 optimizer steps,
180,636 events, 180,642 files, state_interval 50. Raw collection on poseidon at
`~/experiments/results/training/w34_cloud_transcript_v2/` (root-owned).
