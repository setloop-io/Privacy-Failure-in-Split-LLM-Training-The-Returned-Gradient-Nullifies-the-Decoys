# Threat-model scope audit: what actually crosses the boundary, and what was measured

Commissioned 2026-08-19 to test finding 1 of the external adversarial review of the
campaign (the review document is not included in this release),
which asserts that the privacy evaluation omits material parts of the compromised node's view.

**Bottom line.** The finding holds. The compromised node receives one output-gradient frame
per training step that gets no clip, no noise, no place in the DP accounting and no place in
any attacker bundle. Two things about it are now measured rather than argued: it discloses
the real/chaff partition of every training frame exactly (4,608 of 4,608 frames), and it
carries recoverable token information -- the project's own frozen nine-arm gate scores
$z = +14.98$ on 65,536 independent rows from the gradient alone, with a shuffled-label
control at $z = -0.99$. That measurement was taken in a configuration whose forward frame
also leaks, so it does not by itself overturn a gate-passing cell; §5.6 names the one further
run that would (subsequently run as experiment E1, 2026-08-22). The phrase "fully compromised
cloud node" was not supported by the pre-revision manuscript's training results, and §7 gives
the sentence that is.

The audit is in four parts: a line-by-line trace of every boundary crossing in the
latent-native v5 protocol (§1--§3); a quantification of what is unprotected (§4); a real
attack on real captured data from that channel (§5); and the narrower threat-model
sentence the evidence supports (§7).

Scope note. Everything below concerns the **latent-native v5 protocol**
(`bin/run_latent_native_v5_06b.py` + `privacy_runtime/latent_protocol.py` +
`split-training/latent_cloud_server.py`), which produced every delegation and v9.2--v13
privacy cell. The earlier ER split-trainer stack (`split-training/cloud_trainer_server.py`
and its train/eval probes, the latter not included in this release) is a different
protocol and is treated separately in §6.

Line numbers refer to the audited source tree; the same code paths ship in this release,
at possibly different line numbers.

---

## 1. Verdict on the reviewer's finding

**Finding 1 holds on all three of its mechanism claims, and understates the third.** The
omitted channel was then captured in situ and attacked with the project's own frozen gate:
it carries recoverable token information at $z = +14.98$ on 65,536 independent rows, with a
shuffled-label control at $z = -0.99$ (§5.4). It also discloses the real/chaff partition of
every training frame exactly, in 4,608 of 4,608 recorded frames (§5.3).

| Reviewer's claim | Verdict | Evidence |
| --- | --- | --- |
| TLN sends `cloud_k.grad` to UCN; with `wire_quant=none` it gets neither DP noise nor outbound clipping | **Confirmed** | `bin/run_latent_native_v5_06b.py:698-703`, `privacy_runtime/latent_protocol.py:147-150` |
| `remote_grad_clip` clips the gradient coming *back*, not the one going out | **Confirmed** | `bin/run_latent_native_v5_06b.py:470-476` applied at `:707` to `remote_input_grad`, the tensor returned by `chan.backward` |
| The "return DP" is applied to UCN's output *after* it reaches TLN, so it does not protect the outbound gradient, yet its releases are counted in `dp.rho.return` | **Confirmed** | `privacy_runtime/latent_native.py:477-479` inside `decode`, called at `bin/run_latent_native_v5_06b.py:639` on `trusted_return`, which is already restored from the cloud's reply |
| Bundles hold only a freshly regenerated post-training forward pass plus labels; they omit history, gradients, cloud weights, protocol state | **Confirmed, and stronger than stated** | `bin/run_latent_native_v5_06b.py:945-951`; and `train_wire` is not merely regenerated, it is **never transmitted at all** (§3) |
| This contradicts the draft manuscript's attacker-holds paragraph (`papers/arxiv-draft/main.tex:142-148`, not included in this release) | **Confirmed**, and a second, more specific contradiction exists at `main.tex:224` | §7 |

Where the reviewer is imprecise, in the project's favour:

- "neither DP noise nor outbound clipping" is right about the *mechanism*, but the outbound
  gradient is not entirely unshaped: it is the pullback of the loss through
  `protect_return`'s clipping Jacobian (`privacy_runtime/activation_dp.py:98-100`), so
  when that clip binds, gradient rows are attenuated. The additive Gaussian contributes
  **exactly zero** to the gradient (it is a constant with respect to the input), and no
  bound is enforced on the gradient's own norm. There is therefore no DP statement of any
  kind on this tensor; the reviewer's conclusion is correct even though the tensor is not
  raw in the strictest sense. Measured row norms are in §4.
- The accounting error is two-sided, not one-sided. `dp.releases.forward` also **over**-counts:
  it includes the post-training probe phase's `tln.encode` calls
  (`bin/run_latent_native_v5_06b.py:870`), which never reach the cloud. For
  `deleg_6040_conv10k_split14` that is 24,576 of 352,768 forward releases. Over-counting is
  conservative; the return-direction misattribution is not.
- The paper already declares the DP budget vacuous (`main.tex:1034-1039`,
  $\varepsilon\approx2.77\times10^{6}$), so the misattributed `rho.return` is not
  load-bearing for any privacy claim. It is still wrong, and it inflates a number the paper
  quotes.

Three further defects surfaced by the same trace, not in the review:

- **The chaff defense does not survive the backward wire.** The gradient of a chaff row is
  identically zero because `restore` truncates before the loss (`:459-461`), so the
  partition is disclosed exactly and for free (§5.3).
- **The trusted node's seed is disclosed to the compromised node**, and the private
  encoder/decoder initialisation is byte-reproducible from it (§4.3).
- **In fragmentation mode no gradient crosses at all, and the fragmented cloud never
  trains** (§4.4).

---

## 2. Every crossing of the trust boundary

Training step, single channel, `--wire-quant none` (every committed 60/40 cell:
`bin/deleg6040_cell.sh:82-101`).

| # | Item | Direction | Clipped | Noised | In attacker bundle | In `dp` accounting | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Released forward frame, 80 rows x D (32 real + 48 chaff), gauged and rotated | TLN -> UCN | yes, `clip_norm` | yes, `noise_multiplier * clip_norm` | **no** (§3) | yes, `forward` | `:558` encode, `:560` release, `:411-447` gauge, `latent_native.py:474`, `activation_dp.py:97-105` |
| 2 | **Output gradient `cloud_k.grad`, 80 rows x D, one per training step** | **TLN -> UCN** | **no** | **no** | **no** | **no** | `:698-703`, `latent_protocol.py:147-150` |
| 3 | Output gradient in Byzantine-verify mode (`mean_grad`, broadcast to every replica) | TLN -> UCN | no | no | no | no | `:682-689` |
| 4 | Redundant second forward frame in fragmentation mode | TLN -> UCN | as row 1 | as row 1 | no | no | `:661-677` (see §4.4) |
| 5 | `hello`: `latent_dim`, `wire_dtype`, `lr`, `active_delta`, `cloud_kind`, **`cloud_seed` = `--seed`**, `cloud_experts`, `cloud_layers` | TLN -> UCN | n/a | n/a | no | no | `latent_protocol.py:62-69`, seed passed at `:223` |
| 6 | Per-request header: `op`, `mb_id`, `shape` `[B,T,D]`, `training` | TLN -> UCN | n/a | n/a | no | no | `latent_protocol.py:104-106`, `:148-150` |
| 7 | `optimizer_step` control message (one per training step) | TLN -> UCN | n/a | n/a | no | no | `latent_protocol.py:158-163`, called at `:690`, `:709` |
| 8 | Cloud forward output | UCN -> TLN | on arrival, `return_clip` | on arrival, `return_noise` | no | **yes, `return`** | `latent_cloud_server.py:180-191`, `latent_native.py:477-479` |
| 9 | Cloud input-gradient `leaf.grad` | UCN -> TLN | on arrival, `remote_grad_clip` | no | no | no | `latent_cloud_server.py:193-201`, `:470-476` applied at `:707` |
| 10 | `hello_ack`: `session_id`, `latent_only_audit`, `max_parameter_dimension`, `state_shape_digest` | UCN -> TLN | n/a | n/a | no | recorded in the run artifact as `remote_protocol` | `latent_cloud_server.py:132-144` |

Evaluation pass (`training=False`, no backward): only row 1 crosses, coalesced 64 blocks per
request (`:797-798`, `latent_protocol.py:117-145`). Rows 2, 3, 7, 9 do not occur.

Rows 1, 8 are the only two the DP accountant knows about, and row 8 is not a release from
private data at all -- it is the cloud's own reply, noised by the trusted node after arrival.
The direction that genuinely carries private information outward on the backward path,
row 2, is invisible to the accountant.

**Arithmetic check on the accounting, `deleg_6040_conv10k_split14`.** `dp.releases.forward`
= 352,768 = 10,000 training steps x 32 rows + 256 eval blocks x 32 + 256 attack steps x 3
restarts x 32. `dp.releases.return` = 328,192 = 10,000 x 32 + 256 x 32. Both reproduce to
the unit, which confirms the accountant counts exactly the `encode`/`decode` call sites and
nothing else. The 800,000 gradient rows that crossed in the same run appear nowhere.

---

## 3. What the attacker bundle contains, and what it does not

`bin/run_latent_native_v5_06b.py:945-951`:

- `eval_wire` = `held_latents`, collected at `:820`/`:833` from the evaluation pass. These
  frames **were transmitted** (`:797-798`). Faithful.
- `train_wire` = `bundle_train_views`, collected at `:894` inside the post-training probe
  loop (`:853-895`). That loop calls `tln.encode` (`:870`) and `release` (`:879`) but
  never `chan.forward`. **These frames never crossed the boundary.** They are a
  same-distribution re-derivation from the frozen post-training encoder, not the released
  view. The reviewer said "freshly regenerated"; the sharper statement is "synthesised and
  never sent".
- `train_tokens`, `eval_tokens`: honest labels, correctly permuted into wire order
  (`:464-468`).
- `canonical_*`: only when `--bundle-canonical-fraction > 0`; 0.0 in every 60/40 cell.

Absent: the output gradients (row 2), the per-step training history, the cloud's evolving
weights and optimizer state, the protocol metadata of rows 5--7, and the cloud-side view of
rows 8--10.

`attacker/attacks/latent_probe.py:64-67` reads exactly the four tensors above and nothing
else, so the frozen nine-arm gate has never been exposed to any of it.

**Captures.** `main.tex:146` says "captures are the exact wire bytes" and that "wire-level
robustness is checked on captures". `split-training/latent_cloud_server.py:82-91` can indeed
capture `fwd` (`:166`), `return` (`:185`) and `bwd` (`:195`) phases, the last of which is
the outbound gradient verbatim. But `--capture-dir` is never passed by any latent-native v5 driver, and
`paper-data/collected/` contains no capture-derived artifact for this protocol: the only
capture-based schemas present (`dtraining.er.rotation.v1`, `dtraining.e9.rotation.v1`,
`er8_alignment_search`) belong to the ER stack. For the latent-native v5 protocol the
"captures" leg of the paper's evidence claim is empty.

---

## 4. Quantifying the unprotected channel

### 4.1 Shape and volume

One gradient frame per training step, shape `[1, seq_len + chaff_tokens, D]`, fp32, sent as
raw bytes (`latent_protocol.py:91-92`, `:150`). For the committed 60/40 cells
(`seq_len 32`, `chaff_tokens 48`, so 80 rows):

| Cell | steps | D | gradient rows sent | fp32 bytes |
| --- | --- | --- | --- | --- |
| `deleg_6040_split14` / ladder / ctrl (2k) | 2,000 | 64 | 160,000 | 41.0 MB |
| `deleg_6040_conv10k_split13/14`, `a1`, `a4_*` | 10,000 | 64 | 800,000 | 204.8 MB |
| `a3_d96_split14` | 10,000 | 96 | 800,000 | 307.2 MB |
| `a3_d128_split14` | 10,000 | 128 | 800,000 | 409.6 MB |
| `a2b_window4096_40k_split14` | 40,000 | 64 | 3,200,000 | 819.2 MB |
| `a2c_window4096_100k_split14` | 100,000 | 64 | 8,000,000 | 2,048.0 MB |

The count depends only on `steps`, `seq_len + chaff_tokens` and `D`; it does not depend on
the split point, the delegation share, or the noise multiplier.

### 4.2 What the tensor is, and what it structurally exposes

`cloud_k.grad` is the pullback of `total = distill + language + privacy + gram_flat`
(`:659`) through, in order: the truncating/un-permuting/un-rotating `restore`
(`:449-462`), `protect_return`'s clipping Jacobian, and the private decoder
(`latent_native.py:477-485`). Two structural consequences follow from the code alone:

1. **The chaff partition is disclosed exactly.** `restore` drops every row past `n_real`
   (`:459-461`), so the loss does not depend on the chaff rows and their gradient is
   identically zero. The compromised node reads the zero pattern and learns which 32 of the
   80 wire rows are real corpus tokens. Measured: §5.3.
2. **The gradient carries the same per-request rotation as the forward frame.** `restore`
   applies `R^T` on the forward path (`:450-452`), so its pullback right-multiplies by the
   same `R`. The adversary therefore holds two tensors per step under one shared gauge,
   and every cross-tensor inner product is a rotation invariant it can compute. The frozen
   nine-arm attacker never sees two tensors at once and cannot express this; §5.4 point 5
   shows that simply concatenating them does not reach it either.

### 4.3 Measured: outbound magnitudes, and the disclosed seed

Row norms measured on the real captured wire are tabulated in §5.3. The forward frame's rows
are bounded by construction at `clip_norm` before the noise is added; the gradient rows have
no bound of any kind, and their observed maximum (0.196 in cell B) is 1.6x their p95.

Separately: `latent_protocol.py:66` puts `cloud_seed` in the `hello` frame, and
`bin/run_latent_native_v5_06b.py:223` passes `args.seed` as that value -- the same integer
that seeds the trusted node's global torch RNG at `:207`, which initialises the private
encoder and decoder. Two independent container processes, same seed, same base weights,
same code path, produce a byte-identical trusted boundary module:

```
TLN_INIT_SHA256 45342a2060863ac42814a627f4a0dcf789e3be5d3655beeeb100a2588f5e687f
TLN_INIT_SHA256 45342a2060863ac42814a627f4a0dcf789e3be5d3655beeeb100a2588f5e687f
```

So an adversary holding the disclosed seed, the public base weights and the defense code
(all three granted by `main.tex:137-140`) can reconstruct the *initial* private
encoder/decoder exactly. Training then diverges under `secrets`-seeded noise
(`latent_native.py:101-106`), so this is an initialisation-time exposure, not a standing
one. Whether the disclosed initialisation plus the per-step gradient stream permits tracking
the encoder forward is **not measured here**; see §5.6 item 2.

Note this is not the failure mode `main.tex:237` prohibits ("seed secrecy on UCN is
prohibited as a privacy claim"). That rule is about not trusting the cloud with its own
secrets. This is the trusted node's own seed travelling to the cloud.

### 4.4 Fragmentation mode sends no gradient and never trains the cloud

With `--fragment-channels > 1`, `bin/run_latent_native_v5_06b.py:661-677` runs a *second*
forward pass after `total.backward()`, overwrites `cloud_paths` with fresh handles whose
`.grad` is `None`, assigns a `trusted_return` that is never read again, and returns. That
branch never calls `chan.backward` and never calls `chan.step`. Consequently, in every
fragmentation cell:

- no gradient crosses the boundary in the TLN -> UCN direction;
- the fragmented cloud modules never take an optimizer step, so they stay at their
  random initialisation for the whole run;
- `released.backward(...)` is never called on that path, so the trusted encoder receives no
  distillation or language-model gradient through the cloud at all;
- twice as many forward frames cross per step as the design intends.

`latent_v13_a1_fragment2.json` and `latent_v13_35b_v132_fragment.json` nevertheless record
`cloud_correction_improves_loss: true`, which is consistent with the trusted decoder --
not the cloud -- producing the improvement, the same substitution the review's finding 3
identifies. This defect is reported, not fixed: fixing it would change what those published
cells mean, which is a separate decision.

---

## 5. Measurement: a real attack on the real captured channel

### 5.1 Method

Nothing here is simulated or re-implemented. The instrument is the production runner, given
one opt-in flag that records the exact tensor passed to `RemoteLatentCloud.backward`
together with the matched forward frame and the wire-order honest labels:

- `bin/run_latent_native_v5_06b.py --grad-channel-bundle PATH --grad-channel-frames N`
  (`:136-147` argparse, `:478-498` recorder, `:700-702` call site, `:726-744` writer).
  The recorder is a ring buffer, so the retained window is the final N training steps --
  the most converged and therefore the most conservative choice. Every added line sits
  inside an `if grad_channel is not None` guard, consumes no RNG, and adds no key to the
  run artifact, so an invocation without the flag is unchanged.
- `bin/deleg6040_grad_bundle.py` turns the capture into bundles in the **existing**
  `dtraining.latent_release_bundle.v1` schema.
- Scoring is the **unmodified frozen gate**, `python3 -m attacker --attack latent-probe`:
  same nine arms (3 model classes x 3 restarts), same Bonferroni-adjusted Wilson upper
  bound, same majority control. Excess, floor, over-floor and best-arm binomial $z$ are
  then re-derived from the raw counts by the project's own
  `bin/deleg6040_gate_recalibrate.py`; all thirteen re-derivations matched their artifacts
  (`excess re-derived 13/13`).

Two cells were run, both against the live cloud server (`wss://poseidon.cluster:5025`,
D=64, `monomial_moe_radial`, 8 experts, 2 layers), both with every runner flag copied
verbatim from `bin/deleg6040_cell.sh`:

| | **Cell A** `grad_channel_10k_split14` | **Cell B** `grad_power_10k_w4096_split14` |
| --- | --- | --- |
| mirrors | `deleg_6040_conv10k_split14` -- the gate-**passing** 39.3% cell the paper leans on | the window of `a2_window4096_split14`, which already **fails** the gate at +1.6676 pp, $z{=}44.29$ |
| `--train-blocks` | 256 | 4,096 |
| `--steps` | 10,000 (39 passes over the window) | 10,000 (2.4 passes) |
| recorded frames | final 512 | final 4,096 |
| distinct corpus blocks in the window | 256, each recorded twice | 4,096, each recorded exactly once |
| `--eval-blocks` | 256 (as published) | 256 (`a2` used 4,096; reduced here, it only sets the runner's own utility pass) |
| independent evaluation rows, real-row arms | 4,096 (8,192 rows, each block twice) | 65,536 |

Cell A is the configuration that matters for the paper's claim but is capacity-limited: at
256 blocks it cannot yield more than about 4,096 independent real rows however long it runs.
Cell B trades that for eight times the independent sample, at the cost of landing in a
regime whose forward frame is already known to leak. Both are needed and neither is
sufficient alone.

Both ran from a separate tree (`~/dtraining_audit`) on the trusted node so that the
concurrent variance campaign, which re-reads `bin/run_latent_native_v5_06b.py` on every
repeat, was untouched. That tree is a byte-identical copy of `~/dtraining` for every file
this audit reads, verified by `sha256sum` before the instrumented runner was written into it.

Arms. Frames are split into disjoint corpus-block halves (first half train, second half
evaluation), and every arm is scored by the identical frozen attacker:

| Arm | View given to the attacker |
| --- | --- |
| `wire_all` | the released forward frame, all 80 rows |
| `grad_all` | the output gradient, all 80 rows |
| `wire_real` | forward frame, de-chaffed using the gradient's zero support |
| `grad_real` | output gradient, the 32 rows its own zero support marks real |
| `joint_real` | forward frame and gradient of the same step, concatenated |
| `joint_real_scaled` | as `joint_real`, gradient block lifted to the frame's row scale |
| `grad_shuffled`, `grad_real_shuffled` | the same views with globally permuted labels -- negative controls |

The comparison that carries the weight is arm-versus-arm **within one run**, on identical
frames, identical labels, identical attacker and identical sample size. `wire_real` and
`grad_real` score exactly the same rows.

### 5.2 Cell A reproduces the published cell

| | `deleg_6040_conv10k_split14` (published) | `grad_channel_10k_split14` (this audit) |
| --- | --- | --- |
| baseline eval loss | 4.977673412300646 | 4.977673412300646 |
| candidate eval loss | 6.252681 | 6.235358 |
| zero-cloud eval loss | 14.201871 | 14.182999 |
| eval loss delta | +1.27501 | +1.25768 |
| `dp.releases` forward / return | 352,768 / 328,192 | 352,768 / 328,192 |
| in-runner probe excess | -0.3125 pp | -0.2637 pp |
| frozen-gate excess, committed bundle | +0.4096 pp | +0.4103 pp |

Independent restart, independent CSPRNG noise, same numbers.

### 5.3 The chaff partition is disclosed exactly, with no attack at all

From the two `*_bundles.json` reports, over every recorded frame of both cells:

```
                                       cell A       cell B
frames                                    512         4096
rows_per_frame                             80           80
real_rows_per_frame                        32           32
nonzero_rows_per_frame  min / max       32/32        32/32
row_agreement_zero_support_vs_real        1.0          1.0
frames_with_exact_agreement          512 (100%)  4096 (100%)
```

In all 4,608 recorded frames, exactly 32 of 80 gradient rows are nonzero and they are
exactly the 32 real corpus rows. This is deterministic, not statistical: it follows from
`restore` truncating at `n_real` (`:459-461`), so the loss cannot depend on a chaff row and
its gradient is identically zero. `--chaff-tokens 48` therefore conceals nothing from an
adversary that reads the backward wire. `main.tex:220` describes UCN's view as
"80 rows/frame: 32 real + 48 chaff"; on the training path that partition is free.

Row magnitudes over the same frames:

| | rows | min | median | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: outbound forward frame | 40,960 | 2.0306 | 2.9574 | 3.3947 | 4.0361 |
| A: outbound output gradient | 16,384 | 0.000301 | 0.010261 | 0.028670 | 0.119079 |
| B: outbound forward frame | 327,680 | 1.9682 | 2.9573 | 3.4005 | 4.0737 |
| B: outbound output gradient | 131,072 | 0.000225 | 0.009461 | 0.026519 | 0.195864 |

The forward frame's median row norm of 2.957 is the DP noise, not the payload: clipping
bounds the payload at `clip_norm` = 1.0 and the added noise has expected row norm
$0.35\sqrt{64} = 2.8$. Nothing bounds the gradient's norms at all.

### 5.4 Frozen nine-arm attacker, every arm

`corr` is the best arm's correct count; `maj` is the majority control's count on the same
rows. `class` is the project's own classifier: `resolved` requires the best-arm binomial
$z$ to clear $z_{0.05/9} = 2.5392$. The shuffled arms show a different majority rate from
their unshuffled siblings because permuting the labels changes which class is the mode of
the train partition; each arm's control is computed on its own rows and labels.

| Cell | Arm | eval rows | maj % | best % | corr | maj | excess pp | floor pp | over-floor pp | ratio | $z$ | gate | class |
| :---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| A | committed bundle | 20,480 | 5.2588 | 5.2588 | 1,077 | 1,077 | +0.4103 | +0.4103 | +0.0000 | 1.000 | +0.000 | PASS | at-floor |
| A | `wire_all` | 20,480 | 5.7227 | 5.7324 | 1,174 | 1,172 | +0.4363 | +0.4262 | +0.0101 | 1.024 | +0.060 | PASS | at-floor |
| A | `grad_all` | 20,480 | 5.7227 | 5.7666 | 1,181 | 1,172 | +0.4716 | +0.4262 | +0.0454 | 1.107 | +0.271 | PASS | at-floor |
| A | `grad_shuffled` | 20,480 | 5.4102 | 5.4492 | 1,116 | 1,108 | +0.4560 | +0.4156 | +0.0404 | 1.097 | +0.247 | PASS | at-floor |
| A | `wire_real` | 8,192 | 6.0303 | 6.0547 | 496 | 494 | +0.7287 | +0.7030 | +0.0257 | 1.036 | +0.093 | PASS | at-floor |
| A | **`grad_real`** | 8,192 | 6.0303 | 6.3232 | 518 | 494 | **+1.0107** | +0.7030 | +0.3077 | 1.438 | +1.114 | **FAIL** | at-floor |
| A | `joint_real` | 8,192 | 6.0303 | 6.0547 | 496 | 494 | +0.7287 | +0.7030 | +0.0257 | 1.036 | +0.093 | PASS | at-floor |
| A | `joint_real_scaled` | 8,192 | 6.0303 | 6.0547 | 496 | 494 | +0.7287 | +0.7030 | +0.0257 | 1.036 | +0.093 | PASS | at-floor |
| B | committed bundle | 20,480 | 4.7070 | 5.9912 | 1,227 | 964 | +1.7193 | +0.3902 | +1.3290 | 4.406 | +8.677 | FAIL | resolved |
| B | `wire_real` | 65,536 | 4.5715 | 6.8634 | 4,498 | 2,996 | +2.5469 | +0.2117 | +2.3352 | 12.032 | +28.091 | FAIL | resolved |
| B | **`grad_real`** | 65,536 | 4.5715 | 5.7938 | 3,797 | 2,996 | **+1.4583** | +0.2117 | **+1.2467** | **6.889** | **+14.980** | **FAIL** | **resolved** |
| B | `grad_real_shuffled` | 65,536 | 4.7424 | 4.6600 | 3,054 | 3,108 | +0.1312 | +0.2153 | -0.0841 | 0.609 | -0.992 | PASS | at-floor |
| B | `wire_all` | 163,840 | 4.5984 | 7.5507 | 12,371 | 7,534 | +3.1197 | +0.1332 | +2.9865 | 23.423 | +57.054 | FAIL | resolved |

#### What this shows

**1. The output-gradient channel demonstrably carries recoverable token information.**
Cell B, `grad_real`: the attacker sees the gradient rows and nothing else -- no forward
frame, no cloud weights, no history -- and recovers 3,797 of 65,536 tokens against a
majority control of 2,996. That is +1.2467 pp over the statistical floor, 6.9x the floor,
best-arm binomial $z = +14.98$, against a Bonferroni threshold of 2.5392. The matched
negative control on the identical rows with globally permuted labels reads $z = -0.99$ and
passes the gate, so this is not a pipeline artifact: shuffling the labels destroys the
recovery completely. The channel is not information-free, and the frozen gate has never
been shown it.

**2. In cell B the gradient carries less than the forward frame, but independently.**
`wire_real` reaches $z = +28.09$ on the same rows. Cell B is a regime whose forward frame
already leaks -- its published sibling `a2_window4096_split14` fails at +1.6676 pp with
$z = +44.29$ -- so this cell cannot show the gradient breaking an otherwise-sound
configuration. What it shows is that the gradient is a second, independent leak of the same
tokens.

**3. In the gate-passing cell A the gradient reading fails the gate but is unresolved.**
`grad_real` breaches the pre-declared $\le$1.0 pp leg at +1.0107 pp, and its over-floor
reading of +0.3077 pp is twelve times the matched forward frame's +0.0257 pp on exactly the
same rows. But $z = +1.114$ is below threshold, so by the project's own classifier the
reading is at-floor: consistent with real recovery, consistent with noise. Cell A cannot be
pushed further -- 256 blocks admit at most about 4,096 independent real rows -- so
resolving it requires a different window (§5.6).

**4. The zero-support disclosure is what makes the gradient channel usable at all -- but it
does not help the forward attacker.** Raw per-row lift over the majority control:

| | `wire_all` (80 rows) | `wire_real` (32 rows) | `grad_all` (80 rows) | `grad_real` (32 rows) |
| --- | ---: | ---: | ---: | ---: |
| cell A | +0.0098 pp | +0.0244 pp | +0.0439 pp | +0.2930 pp |
| cell B | +2.9523 pp | +2.2919 pp | not scored | +1.2222 pp |

Restricting the gradient to its own disclosed real rows raises its reading by 6.7x in cell A
(+0.0439 -> +0.2930); over all 80 rows the gradient is at the floor and indistinguishable
from its shuffled-label control. Applying the same restriction to the *forward* frame does
not help: in cell B the chaffed frame recovers slightly **more** per row (+2.9523) than the
de-chaffed one (+2.2919). The reason is in the design: chaff rows are recycled *real* corpus
latents with honestly tracked labels (`:293-318`, and the comment at `:283-285`), not decoys,
so they are themselves recoverable and some of them repeat rows the probe trained on. Chaff
dilutes within-frame position and Gram structure; it does not conceal content. The
consequence for the threat model is narrower than it first appears but still real: the
backward wire's zero pattern is what makes the *gradient* channel usable, lifting it in
cell A from indistinguishable-from-shuffled to the only gate failure in that cell.
`grad_all` was not scored in cell B, so the same lift is measured only in cell A.

**5. The joint arms are inconclusive, for a reason worth recording.** Plain concatenation
(`joint_real`) reproduces `wire_real`'s correct-counts exactly on all six invariant-arm
restarts (494, 494, 496, 486, 379, 494), because the
gradient's rows are 0.3% of the frame's magnitude and the rotation-invariant features of the
concatenation are numerically the frame's own. Equalising the two blocks' scale
(`joint_real_scaled`, gradient lifted by the measured median-norm ratio
$2.9572/0.0102 = 290$) does change the per-arm counts (176, 169, 197, 494, 494, 492, 477,
496, 491) but not their maximum. Neither beats `grad_real` alone. The reason is structural: the frozen
attacker computes invariants of one vector, and the invariant that matters here is the
cross-Gram $XG^\top$ between the two tensors under their shared rotation (§4.2), which no
arm in the family computes. The joint view is untested, not clean.

### 5.5 What an adversary holding the gradients plus the frames would have

The compromised node in the deployed protocol holds, per the trace in §2, all of the
following that the frozen gate has never been shown:

| | Frozen gate's attacker | Deployed compromised node |
| --- | --- | --- |
| forward frames | 512 blocks (256 train synthesised + 256 eval transmitted), 40,960 rows | every transmitted frame: 10,256 blocks, 820,480 rows |
| output gradients | none | 10,000 frames, 800,000 rows, unclipped and unnoised |
| real/chaff partition | hidden (it must infer it) | free, exactly, every frame (§5.3) |
| tensors per frame under one shared gauge | one | two, sharing the same rotation `R` (§4.2) |
| private encoder/decoder at initialisation | none | reconstructible from the disclosed seed (§4.3) |
| cloud weights and their trajectory | none | its own, at every step |
| protocol metadata and timing | none | all of rows 5--7, 10 in §2 |
| ability to perturb and observe the response | none | granted by `main.tex:134-135` (active/malicious), never exercised against the gradient |

Three of these are demonstrated here rather than asserted: the chaff partition (exact, all
4,608 frames), the initialisation (byte-identical across processes), and the gradient
channel's token content (cell B, $z = +14.98$ with a clean negative control). The rest are
enumerated, not exploited.


### 5.6 What is still not measured

Named precisely, so nothing here is mistaken for a finished result:

1. **Whether the gradient channel breaks a cell that otherwise passes the gate.** This is
   the one question the audit leaves open, and it is the important one. Cell A is the
   gate-passing configuration but caps out near 4,096 independent real rows, where the
   gradient reading is +1.114 $\sigma$ -- suggestive, unresolved. Cell B has the power but
   its forward frame already leaks. **The experiment that settles it:** the `a2b`
   configuration -- `--train-blocks 4096 --steps 40000`, which passes the gate at +0.4147 pp
   -- run with `--grad-channel-frames 4096`. Its recorded window then covers 4,096 distinct
   blocks exactly once, giving 65,536 independent evaluation rows on a configuration whose
   forward frame passes. At that $n$ the floor is +0.21 pp, and cell A's observed raw lift of
   +0.293 pp would land at $z = 3.15$ -- above the 2.539 threshold -- so the design resolves
   the question in either direction rather than only confirming. Cost: roughly 75 minutes of
   training plus three scoring passes. **Subsequently run as experiment E1** (2026-08-22,
   `E1_GRADIENT_CHANNEL_2026-08-22.md`): on that configuration the gradient arm resolves at
   $z = +8.68$ and the scaled joint arm breaks the +1.0 pp gate.
2. **Encoder tracking from the disclosed initialisation.** §4.3 shows the adversary can
   reconstruct the encoder at step 0. Whether the per-step gradient stream lets it track the
   encoder through training -- and thereby invert the forward frame directly -- is untested.
   The experiment: reconstruct the trusted boundary module from the seed, replay the
   recorded gradient stream through a surrogate optimizer, and measure cosine similarity to
   the true encoder weights over training. It needs per-step weight checkpoints the runner
   does not record.
3. **The cross-tensor invariant.** §4.2 shows the forward frame and the gradient of one step
   share a rotation, so $XG^\top$ is a rotation invariant. No arm of the frozen family
   computes it, and §5.4 point 5 shows concatenation does not reach it. A purpose-built arm
   is needed.
4. **Per-step gradient inversion.** The classic split-learning attack -- solve for the input
   that produces an observed output gradient -- is not attempted. It is the natural strong
   attack against this channel and needs a dedicated implementation.
5. **The full-history attacker.** This audit records the final 512 (cell A) or 4,096
   (cell B) frames. The real adversary holds every step and every intermediate cloud weight.
6. **The other cells.** Two configurations were audited, both split 14, 10,000 steps, D=64.
   The 2k ladder rungs, the D=96/128 arms and the 40k/100k budget arms are untested on this
   channel; §4.1 shows the 100k cell exposes ten times more gradient rows.
7. **Byzantine-verify mode**, where the same gradient is broadcast to K replicas
   (`:682-689`), giving a colluding set K identical copies. Not exercised.

---

## 6. The ER stack, for fairness

The project has not ignored the backward direction everywhere.
The ER stack's train/eval probes (`probes/er_train_eval.py:250` and `:273-279`, not
included in this release) explicitly fold backward-phase captures into
the alignment solve and report `top1_fwd_bwd_pct` alongside `top1_fwd_only_pct`. That work
predates the latent-native v5 protocol and is a different defense (rotation-only, no learned
encoder). The gap this audit documents is specific to the latent-native v5 stack, where the
backward channel is neither captured, nor bundled, nor accounted, nor attacked.

---

## 7. What the evaluation licensed at audit time

Three statements in the pre-revision manuscript were contradicted by the code. Line numbers
in this section refer to that draft (`papers/arxiv-draft/main.tex`), which is not included
in this release; the manuscript has since been rewritten around this audit's findings.

| Location | Draft text | Status |
| --- | --- | --- |
| `main.tex:144-145` | bundles are "the exact views released to \ucn{} (the gauged $D$-width frames **and the returned gradients**, with honest labels confined to \tln{})" | False. No bundle contains any gradient (`:945-951`). `train_wire` was additionally never transmitted (§3). |
| `main.tex:224` | \ucn{} holds "Clipped, noised $D$-width gradients (clip 1.0, $\sigma{=}0.35$)" | False for the direction that matters. The gradient UCN *receives* is neither clipped nor noised (`:698-703`). Clip 1.0 is `--remote-grad-clip`, applied to the gradient UCN *sends* (`:470-476`, `:707`); $\sigma{=}0.35$ is the forward-frame noise. |
| `main.tex:146` | "**captures** are the exact wire bytes ... wire-level robustness is checked on captures" | Unsupported for this protocol. No latent-native v5 capture artifact exists (§3). |

The broad claim -- privacy "against a fully compromised cloud node" (`main.tex:33`, `:80`,
`:132`, `:1120`, `:1187`) -- is not what was measured, because the measured object is one
attacker family applied to one of the several tensors the compromised node holds.

The narrower claim the evidence supports, and the sentence the paper should use:

> Against an adversary restricted to the **released forward frames** -- the gauged
> $D$-width activation rows, with the per-step output gradients, the training history, the
> evolving cloud weights and the protocol metadata all withheld -- a nine-arm
> known-plaintext token-recovery attacker does not exceed the label-free majority control
> by more than the declared gate. This is a claim about one of the several tensors a
> compromised cloud node receives, not about that node's view. In the deployed training
> protocol the node additionally receives one output-gradient frame per step, unclipped,
> unnoised and unaccounted; that channel is measured in §5.4 to carry recoverable token
> content, and the forward-frame gate says nothing about it.

§5.4 measured that channel and found it is not empty: on 65,536 independent evaluation rows
the frozen gate recovers tokens from the output gradient alone at $z = +14.98$, 6.9x its
statistical floor, with a shuffled-label control at $z = -0.99$. That measurement was taken
in a configuration whose forward frame also leaks, so it does not by itself overturn any
gate-passing cell. What it does establish is that the omitted channel carries recoverable
token information, which removes "the gradient is uninformative" as an available defense of
the current scope.

Consequently the word \emph{fully} cannot stand in front of \emph{compromised} for the
training results. Two honest options:

1. **Restrict the claim to the forward frame** (accurate today, and the wording above).
2. **Restrict the claim to inference**, where no gradient crosses at all (`:797-798`,
   `training=False`), and state the training result as forward-frame-only.

Suggested replacement for the draft's attacker-holds paragraph (`main.tex:142-148`):

> \subsection{What the attacker holds}
> Evaluation artifacts are \textbf{bundles}: the gauged $D$-width frames released to
> \ucn{}, with honest labels confined to \tln{}. The evaluation-partition frames are the
> transmitted views; the training-partition frames are regenerated from the frozen
> post-training encoder under fresh gauges. Bundles do \emph{not} contain the per-step
> output gradients \tln{} sends to \ucn{} during training, the training history, the
> evolving cloud weights, or the protocol metadata. The frozen gate therefore measures
> resistance of the released forward frame, not of \ucn{}'s complete view; the
> output-gradient channel is characterized separately and is known to carry recoverable
> token content.

and for the corresponding row of Table~\ref{tab:threat} (`main.tex:223-224`):

> Private encoder $H{=}1024{\to}D{=}64$ and decoder $D{\to}H$ weights &
> Unclipped, unnoised $D$-width output gradients, one frame per training step \\

---

## 8. Reproduction and provenance

Trusted node `gx10-odysseus.nord`, cloud `wss://poseidon.cluster:5025`, image
`split-inference:spark`, corpus `wikitext2_corpus.txt`
(`sha256 78b6bfb9...`, asserted by `bin/deleg6040_precheck.sh`).

```sh
# 1. isolated tree, so the concurrent variance campaign's runner is untouched
cp -a ~/dtraining ~/dtraining_audit
# copy the instrumented runner and the two tools from this release into it

# 2. tool self-test
python3 bin/deleg6040_grad_bundle.py --self-test

# 3. cell A -- the gate-passing configuration
docker run --rm --gpus all --network host --ipc host \
  -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e CONTAINER_IMAGE=split-inference:spark \
  -e SPLIT_AFTER=14 -e STEPS=10000 -e FRAMES=512 \
  -e CELL=grad_channel_10k_split14 \
  -v $HOME/experiments:/workspace/experiments \
  -v $HOME/dtraining_audit:$HOME/dtraining_audit -w $HOME/dtraining_audit \
  split-inference:spark bash bin/deleg6040_grad_cell.sh

# 4. cell B -- the independent-sample configuration
docker run --rm --gpus all --network host --ipc host \
  -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e CONTAINER_IMAGE=split-inference:spark \
  -e SPLIT_AFTER=14 -e STEPS=10000 -e FRAMES=4096 \
  -e TRAIN_BLOCKS=4096 -e EVAL_BLOCKS=256 \
  -e CELL=grad_power_10k_w4096_split14 \
  -e ARMS="wire_all wire_real grad_real grad_real_shuffled" \
  -v $HOME/experiments:/workspace/experiments \
  -v $HOME/dtraining_audit:$HOME/dtraining_audit -w $HOME/dtraining_audit \
  split-inference:spark bash bin/deleg6040_grad_cell.sh

# 5. re-derive every excess, floor and z from the raw counts
python3 bin/deleg6040_gate_recalibrate.py \
  --sweep paper-data/collected/diagnostic/threat_model_scope
```

**Provenance, stated exactly, because the recorded commit is not sufficient.** The arm
artifacts carry `provenance.dtraining_commit = b1915da473bbb2ae2356a2bf9b0ac98e1d4462ef`
(the short form names the committed base-tree patch
`paper-data/provenance/odysseus_dtraining_b1915da.patch`) -- the same gap the external
review's finding 10 identifies
(`attacker/artifacts.py:32-39` records `git rev-parse HEAD` and nothing about the working
tree). The code that actually executed is that commit's tree with these files replaced:

| file | sha256 at time of execution | relation to this release |
| --- | --- | --- |
| `bin/run_latent_native_v5_06b.py` | `41287979d17f03d9a44a2537343554cd58b733ca83ed1139c5b21070b08e4638` | identical |
| `attacker/attacks/latent_probe.py` | `cdb9d0a47af3ef8223d4e0af1bdf82994633e1d07a355c8ab5bacb2be6757e3b` | identical |
| `bin/deleg6040_grad_bundle.py` | `803af720...` for cell A's first six arms, `06cff50b...` thereafter | `06cff50b...` is committed |
| `bin/deleg6040_grad_cell.sh` | `7f687356...` for cell A, `faded796...` by the end of cell B | `faded796...` is committed |

Everything the audit reads unmodified -- `privacy_runtime/latent_protocol.py`,
`privacy_runtime/latent_native.py`, `privacy_runtime/activation_dp.py`,
`split-training/latent_cloud_server.py`, `split-training/split_trainer.py`,
`bin/deleg6040_cell.sh` -- was verified byte-identical between this release and the executed
tree before the runs.

Three deviations, all recorded rather than smoothed over:

1. **The bundle tool gained arms between cell A's build and the end.** `803af720` built cell
   A's `wire_all`, `grad_all`, `wire_real`, `grad_real`, `joint_real` and `grad_shuffled`;
   `06cff50b` adds `grad_real_shuffled` and `joint_real_scaled` and refactors the
   label-shuffle into a helper that preserves its RNG stream exactly
   (`Generator().manual_seed(seed)` then one `randperm` over the flattened labels, as
   before). The six shared arms are built by identical code paths in both revisions. The
   retained cell A capture was rebuilt with `06cff50b` to produce `joint_real_scaled`, which
   also rewrote `grad_channel_10k_split14_bundles.json`; its `support_leak` and `magnitudes`
   blocks are functions of the capture alone and are unaffected.
2. **Three arms were scored outside the cell loop**, by invoking the loop's own
   `python3 -m attacker --attack latent-probe` command directly: cell A's `joint_real` and
   `joint_real_scaled`, and cell B's `wire_all`. Only the invocation differs.
3. **`bin/deleg6040_grad_cell.sh` was edited on disk while cell B was running.** bash reads
   a script lazily, so this could in principle have corrupted execution. It did not: the
   edit changed only the default `ARMS` line, cell B took its arm list from the environment,
   and the log
   (`grad_power_10k_w4096_split14.log:33,179,182,340,343,346,349`) shows the complete
   expected stage sequence -- cell header, runner done, arms built, the three requested arms
   each scored and each writing its artifact, cell complete -- with no shell error. The
   hazard was real and should not be repeated; the evidence says this run was unaffected.

Artifacts are collected under `paper-data/collected/diagnostic/threat_model_scope/`
(2 run artifacts, 2 bundle reports, 13 attacker artifacts, 2 logs). Raw tensor bundles
(`*_gradchannel.pt`, `bundles/*/*.pt`) are **not** committed: they hold released rows paired
with honest labels, the same reason the 60/40 bundles are excluded.
