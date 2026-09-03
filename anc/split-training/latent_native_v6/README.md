# Latent-native v6 candidate

This directory records the best empirical candidate found while evaluating v5
against a fully compromised UCN. It is deliberately `launchable: false`.

Compared with v5, TLN reduces the released width from D=128 to D=16 and
applies a fresh CSPRNG-derived coordinate rotation, token-row permutation, and
independent signed log-normal token gauge to every request. UCN executes only
a monomial-equivariant latent module, so it does not need any of those secrets.
TLN clips and noises both boundary directions and clips returned input
gradients. Each UCN connection starts with a fresh cloud model and optimizer.

The selected diagnostic uses split-after 21, resume-before 26, noise multiplier
0.35, three 2,000-step seeds, and 8,192 held-out token rows per seed. It passes
the strengthened coordinate, invariant, and full-Gram graph probes plus a
declared 0.5 return-perturbation control.

This is the best candidate for further production engineering, not a production
approval. The cloud module has only five learned scalar parameters; arbitrary
malicious computation cannot be verified; the formal activation-DP epsilon is
not useful; and neither convergence-scale training nor the complete live
attacker/inference suite has passed.
