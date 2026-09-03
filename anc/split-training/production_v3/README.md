# Production-v3 split-training privacy path

This directory isolates the post-K3-512 design. K3-512 is no longer a fixed
security target. Known-plaintext testing remains mandatory and adapts until the
recovery curve saturates.

The implementation provides a deeper trusted split, a learned irreversible
bottleneck with gradient-reversal adversarial training, formally accounted
forward/return Gaussian mechanisms, atomic four-direction traffic accounting,
CSPRNG request randomization, replay-resistant sampling, and finite-field
two-server additive sharing for linear operations.

The profile remains `launchable: false`. Additive sharing does not by itself
evaluate transformer nonlinearities. A real deployment requires audited MPC
for those operations and a second independently administered non-colluding
worker. Reconstructing shares on UCN is forbidden.

The local gate (`bin/test_production_v3_privacy.py`) is not included in this
release.
