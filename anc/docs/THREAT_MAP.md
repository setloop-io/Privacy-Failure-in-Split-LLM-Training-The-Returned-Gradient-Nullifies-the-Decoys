> **HISTORICAL DRAFT.** This map was written against a branch not present
> in this repository; several files and line numbers it cites do not exist here
> (`cloud_server_kv.py` and `dlgpp.py`, among others, are not included in this
> release). It marks the
> backward-gradient seam S3 "covered" and states "the class is not the threat" --- which
> the day-0 experiment E1 directly contradicts.
>
> Retained only for the seam enumeration. Seams S2 (return tensor, "captured, read by
> nothing"), S4 (KV cache), and S6 (control metadata) have no attack family in
> `paper-data/evaluation_protocol.json`.

# Threat map — surfaces, not attacks

**Status:** historical draft. Derived by reading source on a branch not included in this
release. Nothing here was executed: this host
has no torch, so every coverage claim below means *this code path reads this seam*,
established by reading the module — never *it was run and produced a number*.

## Why this view exists

Two enumerations already exist and neither answers the benchmark's question.

| view | indexed by | lives in | answers |
|---|---|---|---|
| capability matrix | attack (17 modules) | `attacker/README.md` | what can I run? |
| attacker classes A1–A7 / B1–B4 | adversary capability | the project's distributed-training privacy writeup, §4 Table 8 (not included in this release) | who is in scope, and what does topology multiply? |
| **this map** | **seam** | here | **where does data cross, and what reads it there?** |

The third view is the one that shows gaps. An attack-indexed table can be complete
and still leave a seam untouched, because nothing in it is keyed to the seam. That
is what happened: one of eight seams is captured by a writer and read by zero
attacks, and the table gives no way to notice.

Vocabulary follows the project's existing taxonomy (A1–A7 attacker classes, B1–B4 labeling
sources). This map
does not introduce a competing taxonomy; it adds the axis those two lack.

## The seams

A seam is a place where data crosses to a party the deployment does not control.
Line references are the writer or the holder, verified by reading.

| id | seam | what crosses | writer / holder | captured as |
|---|---|---|---|---|
| S1 | forward boundary activation | `hidden`, rotated h·W | `split_trainer.py:313`, `cloud_trainer_server.py:319`, `cloud_server_kv.py:240`, `latent_cloud_server.py:166` | `fwd` / `prefill` / `decode` |
| S2 | return tensor, cloud → local | cloud-computed output | `latent_cloud_server.py:185` | `return` |
| S3 | backward gradient | dL/dh | `split_trainer.py:346`, `cloud_trainer_server.py:394` | `bwd` |
| S4 | KV cache resident at cloud | per-session cache | `cloud_server_kv.py` session cache | **not captured** |
| S5 | cloud-side weights + optimizer state | trainable params, `WEIGHT_VERSION` | `cloud_trainer_server.py:489`, `:552`, `:124` | **not captured** |
| S6 | control metadata | `hidden_shape`, `grad_shape`, `pe_shape`, `mb_id`, `step`, `block_indices`, `obf_epoch`, `dtype`, `session_id`, `timestamp` | sidecars + WS headers | in sidecars, never as signal |
| S7 | released latent bundle | `dtraining.latent_release_bundle.v1` | latent release path | n/a (bundle is the artifact) |
| S8 | raw-byte wire path | opaque bytes | `cloud_trainer_server.py:570` | `wire_*.bin`, `format: bytes` |

**S5 is structural and easy to miss.** The optimizer lives on the *cloud*
(`cloud_trainer_server.py:489` constructs it, `:552` steps it). The untrusted party
holds and mutates model state; it is not a stateless compute service. That is a
training-signal channel independent of every wire tensor.

## Coverage

```
SEAM COVERAGE                                        8 seams · 3 covered
  ■ S1  forward activation      wire-eval · accumulation · alignment-search
  □ S2  return tensor           ◀── CAPTURED, READ BY NOTHING
  ■ S3  backward gradient       wire-eval (bwd) · gradient-inversion (delegated)
  □ S4  KV cache                no attack; not captured
  □ S5  weights + optimizer     active-cloud, toy only; not captured
  □ S6  control metadata        block_indices read once; never as a channel
  ■ S7  released bundle         latent-probe · -sensitivity · -matching
  ⊘ S8  raw-byte path           declines loudly; defense-side fix needed

  ■ real driver reads it   □ gap   ⊘ blocked on the writer
```

### S1 — forward boundary activation. Covered.

`wire_eval.py:180` and `accumulation.py:113` reach it through `captures.py`;
`alignment_search.py:346` reads `cloud_*.json` directly with its own reader,
bypassing the shared core. Classes: A1, A2, A3 label-free; A5, A6 labeled.

Two caveats that belong to the instrument, not the defense:

- `accumulation.py:59` defaults `--phase` to `None`, and `group_by_epoch` skips the
  filter when phase is `None`. The default therefore pools **every** phase into one
  matrix. Against a capture directory containing `fwd`, `return`, and `bwd` rows,
  that is not an error — it is a wrong number.
- `cloud_server_kv.py:238` caps captures at 64 files (`if n < 64`), silently. An
  accumulation attack whose thesis is an evidence budget is bounded by the
  *defense's* capture hook, not by the attacker. Defense-side; not fixed in this release.

### S2 — return tensor. Captured by one writer, read by nothing.

`latent_cloud_server.py:185` writes `phase: "return"`. Then:

```
wire_eval.py:162
    phases = ("fwd", "bwd") if args.mode == "training" else \
        tuple({m.get("phase") for m, _ in records})
```

Training mode hardcodes the phase list; inference mode discovers it from the
records. `latent_cloud_server.py` is a **training** server — so the one mode with a
`return`-phase writer is the mode where the phase list cannot see it.
`git grep -c '"return"' attacker/` exits 1: no module anywhere in the framework
knows the phase exists. `alignment_search.py:346` filters to `fwd` only.

This is the seam an active/malicious cloud injects into. `active-cloud` documents
the `inject()` / `observe()` protocol (`active_cloud.py:230`) and has no live driver.
So the return path is unattacked from both directions — passively because no
attack selects the phase, actively because no harness is wired.

**What an attack would need:** nothing new in the capture format. Drop the hardcoded
tuple, and the existing `wire-eval` per-epoch scoring runs on `return` rows as-is.
The active leg needs the harness glue next to the server, which is defense-side code.

### S3 — backward gradient. Covered.

`wire_eval.py` bwd arm reads it; `gradient-inversion` wraps an external DLG-style
implementation (`dlgpp.py`, not included in this release)
(A4, label-free). The project's evaluation records A4 as weak even undefended (1.77% / 0.31% /
0.21% at depths 1/4/8), so coverage here is real but the class is not the threat.

### S4 — KV cache. No attack, no capture.

`git grep -il "kv_cache|past_key_value|kvcache" attacker/` returns **0 files**. The
cache is resident at the untrusted party for the life of a session and is the
surface the project's Table 8 names for disaggregated serving ("KV-cache side channels").

**What an attack would need:** a capture hook on cache state — which does not exist —
plus a threat statement about what a cache observer learns that a wire observer does
not. Currently unanswerable from artifacts.

### S5 — weights and optimizer state. Protocol only.

`active-cloud` in training mode is the FL malicious-server analogue: craft optimizer
updates that steer the local head to leak faster. The planner and analyzer are
implemented and toy-verified; the live driver is a documented integration point.
No capture hook exists for `WEIGHT_VERSION` or optimizer state.

The project's §4 scopes active/hijacking attackers **out** — "not addressed by any
confidentiality obfuscation." That scoping is correct for the defense paper and
wrong for a benchmark, whose whole output unit is "fails to an active cloud on
contact." The active axis has to be measurable for that sentence to mean anything.

### S6 — control metadata. One field read, never as a channel.

`block_indices` is read at `alignment_search.py:364-365` and nowhere else.
`timestamp` appears only in the schema declaration (`captures.py:47`) and in artifact
provenance — never used as a signal. Yet the headers carry `hidden_shape`,
`grad_shape`, `pe_shape`, `dtype`, and `session_id`, and the inference sidecar carries
wall-clock `timestamp`.

**What an attack would need:** a metadata-only adversary that never looks at a tensor —
shapes give sequence lengths and batch composition, `block_indices` gives the shard
layout, timestamps give a timing channel across `prefill`/`decode`. No module in the
framework treats metadata as observable. This is the cheapest uncovered surface.

### S7 — released latent bundle. Best covered.

`latent-probe`, `latent-sensitivity`, `latent-matching` all take real `--bundle`
inputs and have produced the v9–v13 campaign artifacts. The only seam where the
attack path and the evidence path are the same path.

### S8 — raw-byte wire path. Blocked on the writer.

`_er_capture_bytes` (`cloud_trainer_server.py:570`) emits `wire_*.bin` with
`format: bytes`. The sidecar carries no `hidden_shape` and no `dtype`, so rows cannot
be reconstructed. `captures.py` now declines with a message naming exactly
what the writer would have to add. Correct behaviour; the fix is defense-side.

## The finding this map produced

Key-set validation is not enough. The capture-schema fix makes the five writers' sidecars
*load*.
It cannot catch this:

| writer | `epoch` means | line |
|---|---|---|
| `split_trainer.py` | ratchet epoch (`header["obf_epoch"] = ep`) | `:575`, `:577` |
| `cloud_trainer_server.py` | ratchet epoch, passed through | `:319`, `:394` |
| `latent_cloud_server.py` | **`mb_id`** — the microbatch counter | `:92` |

Same field name, same type, incompatible semantics. Every validator passes, because
both writers emit the key. `group_by_epoch` then produces one "epoch" per microbatch
for latent captures, and any attack whose thesis is a **per-epoch budget** — which is
`accumulation`, the attack that produces the K50 unit — silently measures the wrong
window.

This drifted inside one repo, one team, both ends written in-house. The capture *schema*
had already drifted five ways; this is the same failure one level
deeper, where schema validation cannot reach. A portable capture contract needs typed,
versioned **semantics** per field, not key presence.

## What this implies for the declaration format

The map is the input to item 1, and it changes the shape of the answer. A deployment
declaration cannot be a flat list of claims; it has to be indexed by seam, because
that is the only index under which "no attack exists here" is expressible:

- **which seams the deployment exposes** — a pool of N pipeline stages has N copies
  of S1/S2/S3, and no field in the current sidecar says which seam a capture came from
- **what each seam's fields mean** — `epoch` above is the proof that names are not
  enough
- **which A-classes are claimed out of scope** — the project's writeup scopes active out; a
  benchmark must record that as a *declared exclusion* that shows up in the output,
  not as an assumption buried in prose

The harness then selects attacks per seam and reports, per seam, one of: a budget, a
declared exclusion, or "no attack exists." The third is a first-class result. Today it
is invisible, which is how S2 stayed unattacked.

## Reconciliation with Table 8

The project's Table 8 already marks the topology-level gaps honestly ("E8 per-seam untested
— flag as future work", "Not measured"). This map is the level below it and does not
contradict it:

| Table 8 topology | what this map adds |
|---|---|
| 2-node WAN split | S1/S3 covered; S2 captured-unread; S4/S5/S6 uncovered |
| Pool of N pipeline stages | no seam identifier exists in any sidecar — the N-seam case is not expressible in the capture format, before any attack question |
| Federated / many data owners | different surface; S5 is the closest analogue and is protocol-only |
| Disaggregated serving | S4 is its named surface and has zero coverage |
| Confidential GPUs | out of scope for a wire benchmark |

`disaggregat*` appears 4 times in the repo, all prose (related work, Table 8) — never
in code. A prior count recorded it as zero hits repo-wide; the
distinction that matters is prose-yes, code-no.

## Open questions

1. **S2 is a two-line change** (`wire_eval.py:162`) that would put the return path in
   scope for existing scoring. Defense-adjacent but attacker-side.
2. **S5 and the active axis.** The paper scopes active attackers out. The benchmark's
   headline sentence needs them in. Does the active leg become a measured axis, or does
   the benchmark declare it excluded and say so in every output?
3. ~~**The `epoch` semantic collision** is a defense-side writer fix
   (`latent_cloud_server.py:92`) or an attacker-side per-writer adapter. Which side?~~
   **Resolved: producer side.** The writer now emits the real
   optimizer step and `epoch: None`, since this server has no ratchet. The emitted key
   set is unchanged, so both the earlier capture contract and the stricter schema-validated
   one still accept it.

## Not verified here

- No real-capture attack was executed; this host has no torch.
- Coverage means "reads this seam in source," not "produced a number."
- S4 and S5 have no capture hook, so no artifact could confirm or refute their
  exploitability either way.
