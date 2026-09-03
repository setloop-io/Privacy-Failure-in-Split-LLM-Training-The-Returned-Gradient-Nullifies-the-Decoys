# Production E2E privacy experiment — split training

This directory is the isolated configuration package for an end-to-end split-training privacy experiment. It is intentionally fail-closed: the experiment must not run as a production claim while `contract.json` contains unresolved blockers. This v1 contract is superseded by `split-training/production_v3/contract.json` (first entry of the contract's blocker list).

The cross-mode implementation audit and paper rationale (`docs/experiments/E2E_PRIVACY_ARCHITECTURE_AUDIT.md`) are not included in this release.

## Security question

Can TLN train a useful model with compute delegated to a fully compromised UCN while keeping private training content below a predeclared attacker-recovery threshold, without claiming encryption or a TEE as the privacy mechanism?

TLN is trusted. UCN is malicious and may inspect host memory, process memory, VRAM, model weights, protocol state, traffic at its endpoint, and returned gradients. UCN may also deviate from the protocol. A mechanism is not a defense if UCN receives its secret or can recover it from public and transformed weights.

## Why the current design is not enough

- The E-R7/E-R9 transform is reversible. Supplying UCN with the seed reveals the transform directly.
- Supplying folded weights derived from public weights permits a single-snapshot pseudoinverse attack. Rotation changes the exposure window, not this fact.
- Local DP-SGD protects local parameter updates; it does not sanitize boundary activations or returned gradients.
- Current activation noise has no proven sensitivity bound or privacy accountant and therefore is not a DP guarantee.
- Prompt jitter, padding, sharding, and rotation may reduce convenient labels, but they do not remove information already present in a cloud-visible representation.
- Emulated attestation and transport encryption do not protect data from UCN itself.

## Required production boundary

The production arm must send only an irreversible, task-sufficient representation from TLN:

1. A private local encoder and learned bottleneck remain on TLN.
2. The bottleneck is trained adversarially against the repository attacker ensemble, with utility and leakage optimized together.
3. Per-example boundary clipping and calibrated Gaussian noise are applied on TLN, with a declared adjacency relation, sensitivity proof, composition accountant, and measured epsilon/delta.
4. The backward boundary is separately clipped/sanitized; forward-only protection is insufficient for training.
5. DP-SGD remains a separate local-update guarantee and must not be reported as activation privacy.
6. Prompt/template randomization is an auxiliary reduction of known-plaintext labels, never the primary privacy claim.
7. No transform seed, reversible key, canonical activation, raw tensor capture, or recoverable public/folded weight pair crosses into UCN.

An irreversible representation can change exact model semantics. The experiment therefore measures a privacy–utility frontier rather than requiring token-level identity with an unprotected run.

## Cells

The contract pre-registers five arms:

- `plain`: unprotected negative control.
- `reversible_fold`: current transform, retained as a negative control and regression baseline.
- `private_bottleneck`: TLN-only learned irreversible representation.
- `activation_ldp`: clipped/noised boundary with formal accounting.
- `private_bottleneck_ldp`: proposed production defense.

All arms use the same real corpus partition, model, split depth, optimizer budget, seeds, network profiles, and attacker effort. Five or more independent repetitions are mandatory. Smoke, toy, and dry-run results may validate machinery but cannot support the paper claim. Missing artifacts, an inactive live attacker, or a protocol downgrade invalidate the cell.

## Acceptance rule

The protected production arm passes only if all of the following hold:

- Every required passive, accumulation, cross-epoch, gradient, membership, blind-source, and active-cloud attack completed.
- The 95% confidence upper bound for recovery does not exceed the matched label-free control by more than 2 percentage points.
- No individual epoch/session exceeds that same predeclared bound.
- Mean held-out loss degradation versus the matched `plain` arm is at most 0.35 and the task-specific quality gate passes.
- The declared DP accountant validates the complete run, including composition across steps and both boundary directions.
- No secrets, raw tensors, checkpoints, or private text appear in collected evidence.

The contract checker (`bin/validate_e2e_privacy_contract.py`) is not included
in this release. The contract's recorded status is `not_ready`: a
`--require-ready` check fails until the blockers in the contract are
implemented and independently tested.
