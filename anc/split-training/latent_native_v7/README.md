# Latent-native v7 candidate

This directory records the hardened successor of the v6 candidate, evaluated
against a fully compromised UCN with the attacker code frozen. It is
deliberately `launchable: false`. There is no separate v7 code tree — as with
v6, all mechanisms are flag-driven in `bin/run_latent_native_v5_06b.py` and
`privacy_runtime/`; this directory is the version's declaration of record.

Compared with v6, v7 adds three defense-side changes:

1. **Chaff tokens** (`--chaff-tokens 16`): every released frame carries 16
   real latent rows recycled from earlier train blocks (CSPRNG-sampled
   without replacement, honest labels tracked, TLN drops them on restore).
   Within-frame Gram/order statistics — the only structure that survives the
   v6 gauges — are poisoned for a compromised host pooling wire captures.
2. **v2 portable key stream**: all gauge draws (coordinate rotation, token
   permutation, signed log-normal scale gauge, chaff selection) use
   `privacy_runtime/ratchet_v2.py` — fresh 128-bit CSPRNG master per draw,
   full 256-bit domain-separated key, SHA-256 counter-mode expansion,
   float64 Box-Muller/QR seam. Replaces the v1-style 63-bit torch-seed
   truncation; torch-build independence is CI-pinnable.
3. **Wire quantization available** (`--wire-quant int8`, fixed grid over the
   gauge clamp range) — off by default; measured to add no gain over chaff
   alone. The per-row absmax variant (`int8row`) is retained only as the
   documented counterexample: it is scale-invariant and strips the token
   gauge (the F_int8 postmortem is not included in this release).
4. **Encrypted transport**: the latent link runs TLS 1.3 (`wss://`,
   pinned local CA, hostname verification, plaintext refused; setup in
   `bin/latent_v7_tls_setup.sh`). Operational control only — no privacy
   credit against the compromised-UCN threat model.

The selected diagnostic keeps the v6 operating point (split-after 21,
resume-before 26, D=16, monomial cloud, noise 0.35, clip 1.0) plus chaff-16,
three 2,000-step seeds, two-node tln/ucn. Upper-95 attacker excess
0.554-0.576 pp (tightest of all tested arms), utility delta +0.088-0.090,
eval ratio 1.22-1.28x; the final configuration over the encrypted wss link
passes all gates on three seeds at +0.522-0.642 pp excess and ratio
1.50-1.58x (v2-key derivation overhead included, TLS cost negligible).

Findings behind the selection, including the statistical-floor analysis, the
silent-chaff-drop postmortem, and the absmax-quantization gauge interaction,
are not included in this release.

This is the best candidate for further production engineering, not a
production approval. The v6 blockers still stand (five-scalar cloud,
unverifiable malicious compute, unusable formal DP epsilon, diagnostic
scale, incomplete live-attacker/inference coverage); chaff's payoff against
the wire-capture attack family is measured by the v9 sensitivity battery and
the v11 matching runs on regenerated bundles.
