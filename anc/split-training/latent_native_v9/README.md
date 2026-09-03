# Latent-native v9 — progressive combination program (production track)

v9 is the progressive production-hardening program on top of v7/v7.1 and
the v8 MoE prototype. Each stage adds one mechanism to the running
combination and is kept only if privacy and runtime gates hold; a failed
stage is skipped (the next mechanism is tried on the previous successful
combination) and postmortemed.

Version map:

| Version | Combination | Status |
|---|---|---|
| v9.0 | v7.1 base (gauges, v2 keys, chaff-48, wss) + E1 width frontier + E2 sequence scaling | E1 PASS (D=32/64); E2 FAIL privacy (dropped) |
| v9.1 | v9.0(E1) + E3 radial+Gram hybrid experts (scale gauge off) | PASS at D=64 |
| v9.2 | v9.1 + E4 K=2 parallel channels | **PASS — final combination** |
| v9.3 | v9.2 + E5 public-data cloud pretraining | FAIL privacy (+1.49 pp); dropped |
| v9.4 | v9.2 + E6 K=4 across two endpoints | FAIL time gate (3.98x); dropped |
| v9.5 | any config + Byzantine verify (K=3 identical replicas across independent nodes) | PASS: 100% detection of an actively malicious node (which lied in its self-report), 0% false positives, utility intact under attack; CPU-node runtime is hardware-bound |

Stage gates (all must hold, frozen attacker unchanged):

- attacker upper-95 excess <= +1.0 pp over the label-free majority control
- utility loss delta <= +0.35
- eval time ratio <= 3x (target: stay near ~1.1-1.6x)
- cloud non-bypass (zero-cloud control worsens loss)
- GPU/CPU/VRAM utilization recorded per cell on both nodes
  (`latent_v9_<cell>_util.json`, peaks and means)

Artifacts (`paper-data/collected/diagnostic/latent_v9/`) and the analysis
and postmortems (`docs/experiments/LATENT_V9_PROGRESSIVE.md`) are not
included in this release.

`launchable: false`.
