# W1.7 packaged repro — operational provenance

Contents (sha256 recorded 2026-08-22, source host odysseus):

| file | sha256 | what |
| --- | --- | --- |
| `w17_e1_repro_queue.sh` | 715b9963f01d3a06fa0c937b2d3d1210b257646edab726db6620c68e96ddaef2 | driver that produced `e1_repro_w12_s43` and `e1_repro_w12_s44` (16:59–20:10 UTC) |
| `w17_e1_repro_queue.log` | 6ea9111e208c72ef3b0b66c46f7c8a6a0828fb0ffcec521b0c9a3d745ab8f0d3 | its full stdout/stderr (not included in this release) |
| `w17_e1_extra_seeds.sh` | 0e90d0c4bf87edeb871b67a6104cf41b18559bd2a83505e9ca64534d759b21a2 | staged driver for seeds 45–47 |

Environment: container `split-inference:spark`, trusted side `~/dtraining-packaged`
(this repository @ 65a7aa8, rsynced 2026-08-22 ~16:50 UTC), cloud side poseidon
`latent-cloud` restarted from `~/dtraining-packaged` at ~16:57 UTC.

Cell `e1_repro_w12_s42` predates this manifest's driver (peer session, own
invocation); its cloud server was the pre-existing container with uncommitted
changes — the one non-code-controlled difference, byte-identity of the radial
class verified against `paper-data/provenance/e1_source_of_record/`.
