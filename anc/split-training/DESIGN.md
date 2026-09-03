# DESIGN — Pipeline-Split Fine-Tuning over WAN

Sequel to the split-inference study (not included in this release). Same two
DGX Spark
nodes (tln = local, ucn = cloud), same bond0 link with `tc netem` WAN
emulation, same trust model: the local node is trusted, the cloud is
honest-but-curious. What changes is the workload: **training**, so gradients
now cross the boundary in both directions and optimizer state exists on both
sides.

Guiding principle: simplest correct thing at every choice. This is a research
prototype for six experiments, not a serving system.

---

## 1. Trust boundary

| Stays local (trusted) | Cloud (honest-but-curious) |
|---|---|
| Embedding table | Middle decoder layers SA+1 .. RA−1 |
| Layers 0..SA and RA..end | Optimizer state for ITS layers only |
| Final norm + LM head | Boundary activations (fp16/bf16) |
| Data, labels, loss, sampling | Boundary gradients dL/d(activation) |
| Optimizer state for local params | |
| LoRA adapters (experiment 5) | |

The boundary tensors are exactly two per microbatch per direction:
- forward: `h = hidden_states` after layer SA, shape `[B, T, d]`
- backward: `dL/dh` (same shape)

Everything the cloud sees is a function of those two tensors. That is the
entire attack surface for experiments 2 and 3.

**Why this boundary and not alternatives:**
- *Sending logits or loss to the cloud* — leaks labels; rejected.
- *Cloud computes loss* — leaks labels; rejected.
- *Splitting inside a layer* — no privacy gain over a layer boundary, much
  more plumbing; rejected.

## 2. Forward protocol (requires_grad semantics)

Per microbatch:

1. Local: `h = head_fwd(input_ids)` (embed + layers 0..SA).
2. Local: `boundary = h.detach().requires_grad_(True)` — the boundary tensor
   is a **leaf** in the cloud's graph and the **sink** for the local graph.
3. Local → cloud: `boundary` (+ position embeddings / attention mask kwargs).
4. Cloud: runs its layers, **retains the autograd graph server-side**, returns
   `h_out`.
5. Local: `logits = lm_head(norm(tail_fwd(h_out)))`, CE loss against shifted
   labels.

`detach()` is the correctness crux — and it applies in **both** directions:
the cloud must not be inside the local autograd graph (it will be a network
hop), so `boundary_out = h.detach().requires_grad_(True)` goes to the cloud
and `boundary_in = h_out.detach().requires_grad_(True)` comes back. Local
tail backward ends at `boundary_in.grad`; the local head backward is a
separate explicit call seeded with the input-grad the cloud returns. The
prototype (`split_trainer.py`) implements exactly this in-process;
`CloudWorker.forward/backward` are the wire seam.

## 3. Backward protocol

The wire-accurate version uses **two detached boundary leaves** per
microbatch (`boundary_out` on the way in, `boundary_in` on the way back), so
there are three backward calls:

1. Local: `loss.backward()` — local tail grads + `boundary_in.grad`
   (= dL/d(cloud output)).
2. Local → cloud: `boundary_in.grad`. Cloud:
   `torch.autograd.backward(h_out, grad)` — cloud layer grads (stay on
   cloud) + `boundary_out.grad` (= dL/d(cloud input)), which it returns.
3. Local: `torch.autograd.backward(h_head, boundary_out.grad)` — local
   head + embedding grads.
4. After `--grad-accum` microbatches: cloud `optimizer.step()` on its own
   layers (its own AdamW, its own LR — exchanged once at session setup).
   Local steps its own optimizer independently.

No parameter gradients ever cross the wire. Per microbatch the wire carries
2 × `B·T·d` × 2 bytes = e.g. 1×512×2048 fp16 ≈ 4 MB — trivially small vs the
bond0 link; this is why split training is WAN-viable at all.

## 4. Schedules: synchronous vs 1F1B vs overlap

- **Milestone 1 (prototype): GPipe all-forward-all-backward (AFAB), sync
  boundary.** All K microbatches forward, then all K backward; optimizer
  steps once per step. Correct but serial on the wire: per microbatch the
  local node idles through two full RTTs (measured E1 @80ms: 0.456+0.465s
  cloud round-trips vs ~0.02s local compute).
- **Milestone 2b (implemented, `--pipeline overlap`): async boundary
  crossings ("training lookahead").** Same AFAB math, but sends and waits are
  decoupled and responses are correlated by `mb_id`:

  ```
  sync:     [head0][ cloud fwd0 ][tail0][ cloud bwd0 ][head-bwd0] [head1][ cloud fwd1 ]...
  overlap:  [head0][head1][head2][head3] ...  fwd requests stream ahead (window N)
                   \__ while waiting, do tail0/bwd0/... of already-answered mbs
  time ->   |A0 A1 A2 A3|C0 D0|C1 D1|...|F0 F1 F2 F3|   (A=head fwd, C=tail fwd+loss,
                                                         D=tail bwd, F=head bwd)
  ```

  The local node blocks only when the in-flight window is full AND the oldest
  response has not arrived. At most `--max-inflight N` forwards are
  outstanding, which bounds cloud pending-graph memory (each pending graph =
  one microbatch's middle-layer activations). Backward requests are issued
  immediately after each tail backward, so cloud graphs are freed early.
  `--max-inflight 1` degenerates exactly to the sync schedule (verified
  bit-exact). Optimizer steps are unchanged: once per K microbatches, so
  overlap is numerically identical to sync (verified bit-exact on loopback,
  20 steps).
  Cost vs sync: up to K local head-graphs retained per step (AFAB property),
  and cloud holds ≤N pending graphs instead of 1.
- **1F1B** (interleave one-forward-one-backward after warmup): same steady-
  state throughput as AFAB-overlap, K× lower activation memory. Still
  deferred — implement only if GB10 memory binds at 35B.
- **FedAvg-style local updates (experiment 4b)** is a *different axis*, not a
  schedule: data-parallel replicas of the full model, K local steps, then
  average weights. Implemented as a separate mode (`--mode fedavg`), since it
  has no pipeline boundary at all; it is the baseline we compare pipeline
  split against at each RTT.

## 5. Gradient compression hooks (experiment 3) — implemented: `gradient_inversion.py --phase e3`

Two seams, one already in production code:
- `CloudWorker.grad_hook` (in-process) / `grad_compression` in the WS hello
  (remote, currently noise only): applied to the boundary gradient before
  cloud backward — this is what TRAINING uses.
- `apply_defense()` in `gradient_inversion.py`: applied to the observed
  boundary gradient (and optionally the activation, `--defend-activation`)
  before the attacker sees them — this is what the ATTACK evaluation uses.

Three defense families, each one spec string:
- **DP noise**: `noise:σ`, σ ∈ {0.001, 0.01, 0.05, 0.1} × per-tensor RMS.
- **Quantization**: `quant:8`, `quant:4` (per-tensor affine) and `sign`
  (1-bit). Stochastic rounding deferred; v1 is deterministic round-to-nearest.
- **Top-k sparsification**: `topk:0.1`, `topk:0.01` keep-by-magnitude
  (no error feedback in v1 — the honest simple version).

Privacy metric: the E2 optimization attack against defended observations.
Utility metric: real split-training loss after `--utility-steps` with the
defense on the boundary gradient vs clean (same init, same data order).
Adaptive-attacker note: the attack optimizes THROUGH nothing — it inverts
the defended observation directly with a cosine objective (which is
deliberately scale/robustness-friendly); a stronger adaptive attacker
(e.g. straight-through inversion of the quantizer) is future work and one
more reason to report results as lower bounds. Fixed seeds, mean ± std CIs.

## 6. Inversion attack setup (experiment 2) — implemented: `gradient_inversion.py`

Attacker sits on the cloud and observes, per microbatch: boundary activations
`h*` (forward) and boundary gradients `g* = dL/dh` (which it computes and
returns). Implemented attack (single file, `--phase e2`):

1. **Gradient inversion (DLG/iDLG-style)**: the attacker replays the FULL
   model as a surrogate (public base checkpoint assumption — exact at
   fine-tune step 0, degrades with local drift, which `--train-steps`
   measures) and optimizes a dummy embedding sequence `z` (seq_len ≤ 32) to
   minimize `(1−cos(ĝ(z), g*)) + (1−cos(ĥ(z), h*))`, with ĝ via
   `create_graph=True` double-backward (math SDPA kernel required). Labels
   are self-generated pseudo-labels (current argmax reconstruction, detached)
   — causal-LM labels are the shifted inputs. Reconstruction = nearest
   embedding row of `z` per position (cosine).
2. Sweep split depth d ∈ {1, 4, 8} × training config {full FT, freeze-cloud,
   freeze-cloud+LoRA-local (E5 config)} × docs × 3 seeds; token accuracy +
   embedding cosine, mean ± std.
3. **Activation-only inversion** (MLP decoder from `inversion_experiment.py`,
   not included in this release) remains the inference-time comparison
   point; the optimization attack here subsumes it (activation term is in
   the objective).

Lower-bound framing for the paper: pseudo-labels are self-generated (no
oracle), the surrogate is exact only at step 0, and z is optimized
continuously then rounded — all three weaken the attack, so measured
leakage is a floor, not a ceiling.

## 7. FedAvg baseline (experiment 4b) — implemented: `fedavg_node.py`

FedAvg is the data-parallel analogue of the pipeline overlap schedule (§4):
both amortize one WAN crossing over K units of local work. Overlap amortizes
the RTT over K microbatches in flight; FedAvg amortizes a **weight-delta
exchange** over K local optimizer steps. Same question, other axis: how much
local work hides the network?

Mechanics: both nodes hold the full model, train on private disjoint shards
(doc-index parity split), and every K steps exchange flat fp32 weight deltas
against the last sync point (`avg = mean(deltas)`, both apply `ref + avg`),
over the same binary WS framing as the rest of the system (ws port 5004,
64 MB chunks, one persistent connection, `--sync-dtype fp32|bf16`).

**Break-even formula.** Per K-step round, FedAvg wastes `T_sync(K, RTT,
bytes)`; split-sync wastes `K_mb × 2 × RTT` per step (E1 measured), reduced
by overlap. FedAvg wins when

    T_sync  <  K × (t_step_split − t_step_local)

i.e. `K* ≈ T_sync / (t_step_split − t_step_local)` — every term measurable:
`T_sync` from the sync log, `t_step_split` from E1/E4 timings, `t_step_local`
from a no-network single-node run. Predicted `T_sync` for the 0.6B fp32
delta (2.4 GB): ~19 s on the hostile 1 Gbps profile (+2×RTT≈1 s), ~4 s at
the benign bond0 rate; halve with `--sync-dtype bf16` (1.2 GB → ~10 s
hostile). With `t_step_split − t_step_local ≈ 0.9 s` (80 ms profile, sync
schedule) that gives `K* ≈ 20` fp32 / `K* ≈ 10` bf16; overlap (E4) pushes
`t_step_split` toward `t_step_local` and inflates `K*` accordingly — that
interaction is the experiment.

**Privacy difference**: weights, not activations/gradients, cross the WAN.
The honest-but-curious peer sees weight deltas averaged over K local steps
of private data — the relevant literature is model-update/weight-delta
inversion (gradient leakage from aggregated updates), which is strictly
weaker per observation than raw boundary gradients (K-step averaging +
full-model aggregation), but spans the FULL parameter set rather than a
layer boundary. No activations or labels ever leave the node. This trades
the split-training threat surface for a federated one — a deliberate
contrast for the paper.

## 8. Experiment → CLI mapping (all single commands on tln)

| # | Experiment | Command sketch |
|---|---|---|
| 1 | Throughput vs RTT | sweep wrapper (`bin/run_rtt_sweep.sh` is not included in this release): for rtt in 0 5 20 40 80 450; `tc qdisc` set; `python split_trainer.py --steps 50 --grad-accum 8` (cloud remote via `cloud_trainer_server.py`) |
| 2 | Inversion vs depth | `python gradient_inversion.py --phase e2 --depths 1 4 8` |
| 3 | Defense frontier | `python gradient_inversion.py --phase e3 --defenses ...` |
| 4a | GPipe K vs RTT | `python split_trainer.py --grad-accum K` × RTT grid; break-even K where pipeline throughput ≈ 0-RTT throughput × (1−ε) |
| 4b | FedAvg baseline | `fedavg_node.py --role server` (ucn) + `--role client --cloud ws://ucn:5004` (tln), `--k-local K` × RTT grid |
| 5 | LoRA-local-only | `python split_trainer.py --freeze-cloud --lora-rank 16` (already in milestone-1 prototype) |
| 6 | MoE placement | `moe_placement_experiment.py` (not included in this release) `--experts-local {0,32,64,...}` on Qwen3.6-35B-A3B (40 layers, 256 experts/8 active): route-hot experts local, cold experts cloud; measure boundary traffic from actual routing histograms |

## 9. Model/milestone plan

1. **M1 (this prototype)**: in-process CloudWorker, Qwen3-0.6B, toy-model CPU
   smoke. Verified on Mac CPU (toy) + Spark GPU (full smoke).
2. **M2**: `cloud_trainer_server.py` on ucn — CloudWorker over the existing
   binary WS protocol (see PROTOCOL.md). Experiment 1, 4a.
3. **M3**: inversion + defense harnesses (experiments 2, 3) — pure local
   compute on one node first.
4. **M4**: FedAvg mode (4b), LoRA study (5), then MoE (6) once the 35B-A3B
   patched hybrid stack (patches not included in this release) is stable
   under training.

## 10. Decisions log (why, briefly)

- **fp16 vs bf16**: bf16 default for training (grad range); fp16 kept as a
  flag to match the inference setup for comparability. Loss computed in fp32
  (`.float()` on logits) to avoid CE underflow.
- **AdamW both sides, independent LRs**: simplest; a shared schedule is a
  config, not code.
- **No gradient checkpointing in v1**: 0.6B fits easily; add only when 35B
  MoE demands it.
- **New server file for training** (`cloud_trainer_server.py`), never patch
  the inference server (`cloud_server_kv.py`, not included in this release):
  the inference system was running paper experiments and had to keep
  serving.
- **position_embeddings computed locally, sent to cloud**: the cloud layers
  need (cos, sin); computing rotary embeddings locally avoids sending
  position_ids semantics across the wire and matches the inference server,
  which already ships position embeddings in its header.
