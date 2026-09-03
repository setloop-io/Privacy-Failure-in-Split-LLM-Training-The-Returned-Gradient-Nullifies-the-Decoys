# Complete remote transcript protocol

This protocol implements the complete-transcript capture for the latent-native v5 path
(campaign items W3.4/W3.5).
It creates a session bundle from the perspective of the compromised UCN
process, rather than regenerating a trusted-side post-training probe.

## Capture contract

Start `split-training/latent_cloud_server.py` on UCN with:

```bash
python3 split-training/latent_cloud_server.py \
  --latent-dim 64 --cloud-kind monomial_moe_radial \
  --cloud-experts 8 --cloud-layers 2 \
  --tls-cert /workspace/experiments/tls/ucn-server.crt \
  --tls-key /workspace/experiments/tls/ucn-server.key \
  --port 5025 \
  --transcript-dir /workspace/experiments/results/complete-view
```

Use the intended forward-gate-passing runner configuration unchanged, except
for pointing it at that server. The remote server creates one
`session_<uuid>/` directory per websocket connection. A valid single-channel
training session contains:

- every received and sent WebSocket message, stored as its exact raw bytes;
- decoded headers, request IDs, remote arrival timestamps, and remote timing
  metadata in `events.jsonl`;
- the cloud model and AdamW state at initialization and after every optimizer
  step;
- a final `TRANSCRIPT_MANIFEST.json` with SHA-256 and byte count for each
  message, checkpoint, and event log.

The transcript root maintains `COLLECTION_MANIFEST.json`, an atomically
updated, SHA-256-pinned list of every completed or failed cloud session. It is
the artifact to use for multi-channel runs.

The server only marks a transcript `complete` after the client sends its
normal `close` control message. A disconnect or server exception is sealed as
`incomplete` and must not be used as a full-view artifact.

## Verify before analysis

Run this independently, on the capture directory, before passing any files to
an attack implementation:

```bash
python3 bin/verify_remote_transcript.py \
  --transcript /workspace/experiments/results/complete-view/session_<uuid>
```

For multi-channel capture, verify the collection rather than cherry-picking a
single session:

```bash
python3 bin/verify_remote_transcript.py \
  --collection /workspace/experiments/results/complete-view
```

After verification, build the label-free full-view index consumed by attack
implementations. It pairs forward inputs, output gradients, returns, input
gradients, controls, timing, and remote state without joining any TLN-only
data:

```bash
python3 bin/build_full_view_attack_index.py \
  --collection /workspace/experiments/results/complete-view \
  --output /workspace/experiments/results/complete-view/attack-index.json
```

The verifier checks every recorded digest and size, event sequence continuity,
the complete forward/backward/control exchange, and initial plus post-update
remote-state snapshots. It reads no TLN memory, labels, pre-gauge latents,
or trusted secrets.

## Inclusion and exclusions

The bundle intentionally includes the exact views available to UCN:
`hello` metadata, forward inputs and outputs, outbound loss gradients,
returned input gradients, optimizer controls, state updates, and remote timing.
It intentionally excludes trusted-only values such as token labels, canonical
latents, gauge seeds, the private encoder/decoder, and trusted optimizer state.

For a multi-channel run, the full adversary view is the union of every session
listed in the verified collection. An attacker can then explicitly decide how
to pool those sessions.

## Status

A complete-view transcript has been captured and verified: 3 sessions, 30,000
optimizer steps, 180,636 events. The raw payload (1.9 GB compressed) remains on
the capture host under the repository's size policy; this release carries the
verification manifest and hashes at
`paper-data/collected/diagnostic/w34_complete/MANIFEST.md`, plus the collection
and verification implementation described above.
