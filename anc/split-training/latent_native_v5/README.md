# Latent-native v5 candidate

This isolated candidate replaces the reversible full-width v4 boundary. TLN
owns every 1024-dimensional operation. UCN receives a serialized transformer
whose input, state, and output are 128-dimensional latents; it contains no
canonical 128-to-1024 decoder.

TLN retains its private H-width residual while the request is in flight.
UCN computes only a D-width correction; TLN privately decodes and adds the
correction to that retained residual. This avoids reconstructing the entire
canonical residual stream from D dimensions and does not expose the bypass to
UCN.

The eight mechanisms are executable in `bin/test_latent_native_v5.py`. Passing
that gate establishes implementation mechanics, not privacy. Promotion remains
blocked until the full `/attacker` suite reaches saturation on fresh held-out
captures while utility and runtime gates pass. Obfuscation and dimensionality
reduction are not cryptographic confidentiality against a root-level UCN.

The intended training loop alternates: (1) optimize adaptive attackers on
detached released latents, then (2) optimize the private encoder and task model
through gradient reversal. The cloud middle is distilled to operate natively
in latent space instead of expanding a boundary activation back to model width.

| Change | Implemented mechanic | Remaining acceptance evidence |
|---|---|---|
| 1. Latent-native cloud | D-only transformer; shape/state audit | 0.6B distillation and quality |
| 2. Attacker in loop | alternating token/property/reconstruction adversaries | full `/attacker` saturation |
| 3. Width reduction | private 1024→128 encoder | dimensional ablation |
| 4. Latent clipping/noise | token-row bidirectional DP accountant | feasible privacy/utility budget |
| 5. Private boundary | encoder/decoder absent from UCN state | remote deployment manifest |
| 6. Request stochasticity | CSPRNG noise, nonce replay refusal | repeated-query attack |
| 7. Prompt randomization | slot permutation and dummy padding | tokenizer/quality integration |
| 8. Dense latent ratchet | fresh 128×128 transform per request | cross-session pooling attack |
