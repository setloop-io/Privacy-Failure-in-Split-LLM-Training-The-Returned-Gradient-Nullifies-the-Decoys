# PROTOCOL — Moving CloudWorker onto the wire

How the in-process `CloudWorker` in `split_trainer.py` becomes a remote
`cloud_trainer_server.py` on ucn, reusing the binary WebSocket framing of
the split-inference system. **New server file; the inference server
(`cloud_server_kv.py`) is not included in this release.**

## 1. Framing (unchanged)

Same wire format as the inference system (`cloud_server_kv.py` /
`local_client_kv.py`, neither included in this release):

```
[4B header_len big-endian][JSON header UTF-8][raw tensor bytes]
```

One WS connection per training session (ws port **5003**; training uses a
different port than inference's 5001 so both can run on ucn at once).
HTTP health on **5002** (`/health` → `{"status":"ok","sessions":N}`), same
convention as inference's 5000.

## 2. Message types (new; keyed by header `"op"`)

| op | Direction | Header fields | Tensor payload |
|---|---|---|---|
| `hello` | local→cloud (text JSON) | `{op, model, split_after, resume_after, dtype, optimizer:{name,lr,weight_decay}, trainable, grad_compression:{kind,sigma/bits/k}?}` | — |
| `hello_ack` | cloud→cloud reply | `{op, session_id, n_cloud_layers, cloud_params}` | — |
| `forward_with_graph` | local→cloud | `{op, mb_id, hidden_shape, dtype, has_pos_emb}` | `hidden [B,T,d]` fp16/bf16, then optional `cos`,`sin` rotary halves |
| `forward_result` | cloud→local | `{op, mb_id, hidden_shape, dtype}` | `h_out [B,T,d]` |
| `backward_grad` | local→cloud | `{op, mb_id, grad_shape, dtype}` | `dL/d(h_out)` |
| `backward_result` | cloud→local | `{op, mb_id, grad_shape, dtype}` | `dL/d(h_in)` (grad wrt cloud input, so local finishes head backward) |
| `optimizer_step` | local→cloud (text JSON) | `{op}` (sent once per --grad-accum microbatches) | — |
| `step_ack` | cloud→local | `{op, mb_consumed}` | — |
| `close` | either | `{op, reason}` | — |

Why `backward_result` returns `dL/d(h_in)`: the cloud's graph includes its
input leaf, so input-grad is computed there for free; the local head's
backward is blocked on it. This is exactly what the in-process prototype
does (`CloudWorker.backward` returns `self._last_input.grad`).

## 3. Server-side state (per session)

```python
sessions[session_id] = {
    "layers": nn.ModuleList,          # middle layers only
    "optimizer": AdamW,               # configured by hello
    "pending": {mb_id: (input_leaf, output)},  # graphs held for backward
    "mb_since_step": int,
    "last_access": float,
}
```

- `forward_with_graph`: rebuild tensors (`torch.frombuffer(...).reshape(...)`),
  `input_leaf = hidden.detach().requires_grad_(True)`, run layers, stash
  `(input_leaf, output)` under `mb_id`, reply `forward_result`.
- `backward_grad`: pop `pending[mb_id]`, apply grad compression hook if
  configured, `torch.autograd.backward(output, grad)`, reply
  `backward_result` with `input_leaf.grad`.
- `optimizer_step`: `optimizer.step(); optimizer.zero_grad(set_to_none=True)`.
- Session timeout: `SESSION_TIMEOUT = 600`s sweep (same pattern as the
  inference server's session cleanup), freeing `pending` graphs.

## 4. Tensor dtypes / order

- Default dtype = the training dtype (`bf16` for training, `fp16` only for
  inference-comparability runs); declared in `hello`, echoed per-message.
- Tensor order in `forward_with_graph` payload: `hidden`, then `cos`, `sin`
  (when `has_pos_emb`). Attention mask is **not** sent: full fine-tuning
  uses causal-only attention with `attention_mask=None` (sdpa `is_causal`
  path), same as the prototype. If padding is ever introduced, add an
  optional 4th tensor + header flag.
- All tensors little-endian raw bytes, contiguous, shapes in header — same
  reconstruction code path as the inference server (not included in this
  release).

## 5. What changes vs the inference server

| inference server (not in this release) | cloud_trainer_server.py (training) |
|---|---|
| KV cache per session, `DynamicCache` | No KV cache (full-sequence training); instead `pending` autograd graphs per microbatch |
| `torch.no_grad()` / inference_mode | Gradients enabled; layers in `.train()` mode |
| Stateless weights (frozen model) | Optimizer state on server; weights mutate |
| One WS message per token step | One message pair per microbatch, K× less frequent |
| Hybrid-model patches (not included in this release) | Reuse verbatim for MoE experiment 6 — import the same layer-loading code |

Client side: a `RemoteCloudWorker` implementing the same four-method
interface (`forward/backward/zero_grad/step`) as `CloudWorker`, selected by
`--cloud-host ws://ucn:5003` (default `None` = in-process). Nothing else in
`split_trainer.py` changes — that was the point of the class boundary.

## 6. Failure/timeout semantics (deliberately simple)

- WS drop mid-step → abort the run, re-`hello` starts a fresh session from
  the last checkpoint. No graph recovery: at these model sizes a re-run of
  one optimizer step costs seconds.
- Checkpointing: local side owns the checkpoint (it can request cloud weights
  via a `get_state` op — v2; v1 checkpoints local params only and treats
  cloud weights as ephemeral per run, fine for throughput experiments, NOT
  for real fine-tunes — flag this before experiment 5's paper runs).

## 7. Response correlation and pipelined clients (M2b)

**No wire-format change was needed.** Every binary response already carries
the `mb_id` of its request, and the server keys `pending` graphs by `mb_id`
(popped on `backward_grad`), so:

- A pipelined client may have many requests in flight and may issue
  `backward_grad(mb_j)` while later forwards are still queued; the server
  processes frames in arrival order and is order-tolerant by construction.
- The client (`RemoteCloudWorker`) runs a mailbox dispatch loop:
  `_read_into_mailbox()` files each incoming frame under
  `(op, mb_id)`; `_wait(op, mb_id)` blocks until that exact response exists.
  Sync clients are unaffected — their strict request→response pattern is
  just the mailbox degenerate case (server handles both concurrently, one
  session each).
- `optimizer_step` is only ever sent when the client has drained every
  outstanding response, so `step_ack` needs no correlation id.
- Server error mid-flight: an `error` frame (or a dropped connection) raises
  `RuntimeError` in whatever `_wait` is blocked, aborting the run — v1 has
  no recovery, per §6.
- Test-only `--latency-ms` / `CLOUD_LATENCY_MS` on the server emulates WAN
  RTT faithfully: each frame is delayed in its own asyncio task (frames in
  flight are delayed concurrently — wire delay), while compute still runs
  serially. NOT a service-time sleep.

## 8. Operational semantics confirmed by loopback tests (M2)

- **Server weights persist across sessions** (they are the trained artifact):
  a second client sees the weights the first client trained. Exact-repro
  comparisons must restart the server (or accept the drift).
- **Wire dtype should match model dtype**: bf16 model + `--wire-dtype bf16`
  is bit-exact; fp16 wire perturbs boundary values ~2e-5 (measured step-0
  loss delta vs in-process); bf16 wire under an fp32 model perturbs ~1.0
  (8-bit mantissa) — don't do that.
- **Dead-client detection** is keepalive-based (ping 30s/timeout 60s): a
  SIGKILLed client's session (and its pending graphs) is reaped within ~90s,
  logged as `connection dropped ... (aborting run — v1 has no graph recovery)`.
