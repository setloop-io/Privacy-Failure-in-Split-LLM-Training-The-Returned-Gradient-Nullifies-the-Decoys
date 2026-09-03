# Latent-native v8 candidate (prototype)

v8 prototypes the answer to the v6/v7 "five-scalar cloud" blocker: an
**equivariant mixture-of-experts cloud** (`--cloud-kind monomial_moe`).
UCN's delegated capacity scales horizontally with expert count instead of
being fixed at five scalars, while the wire width and every gauge property
from v7 are unchanged.

Design (`MonomialMoELatentMiddle` in `privacy_runtime/latent_native.py`):

- Each of E experts per layer owns a distinct Gram kernel shape (quadratic
  temperature, signed linear weight, message gain) operating on the same
  D=16 gauged frames as v7.
- Routing is per-row softmax over a 3-feature vector of *squared* Gram
  statistics (off-diagonal mean / std / max), which is invariant to the
  coordinate, sign and scale gauges and covariant with row permutation — the
  router therefore respects every v7 gauge (verified by the mechanism test
  `5/monomial-moe-cloud-hides-token-scale-sign-order-and-coordinates`).
- Parameter count: `cloud_layers × E × 3 + 2 × cloud_layers × 4E + 1`
  (E=8, 2 layers ≈ 120 params; E=64 ≈ 900). Cloud FLOPs scale ~linearly in
  E (E Gram-message passes per layer), so untrusted GPUs are exercised
  proportionally to E at high request volume — VRAM stays trivial by design.

Protocol: `--cloud-experts` is pinned end-to-end (runner -> hello -> server
validation -> ack echo -> client fail-closed on mismatch). Session isolation,
D-only audit, TLS transport, and the frozen attacker evaluation are exactly
as in v7.

Status: prototype. First cells: `bin/latent_v8_sweep6.sh`.
`launchable: false` stands; the v7 blocker list applies, with "five
trainable scalars" now replaced by "expert-routed surrogate capacity
untested at production scale".
