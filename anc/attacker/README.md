# attacker/ — unified attacker framework

One framework for **every** attack methodology in this package. It subsumes
the earlier standalone scripts under `split-training/`
(`rotation_lifetime.py`, `dlgpp.py`, `e8_robustness.py`,
`membership_inference.py`, `output_inversion.py`, `sipit_inversion.py`,
`trained_inversion.py`; the older `probes/` scripts are not included in this
release) and adds six further attacker capabilities, described below.

Execution is parameterized by mode — the framework knows which surface it
operates on:

```bash
python -m attacker --list-attacks                    # full matrix
python -m attacker --list-attacks --mode training
python -m attacker --mode inference --attack wire-eval ...
python -m attacker --mode training  --attack accumulation ...
python -m attacker --self-test                       # torch-free fixtures
python -m attacker --attack alignment-search --help  # works torch-less
```

Mode selects the wire-capture sidecar schema (`attacker/captures.py`) and
the available attack set:

| mode | sidecar schema | adds |
|---|---|---|
| `training` | `{session_id, mb_id, phase, step, epoch}` (split-training/er_ratchet.py `SIDECAR_KEYS`) | gradient inversion, membership, active-update steering |
| `inference` | `{session_id, request_seq, phase, position, epoch}` (inference capture hook; captures without `epoch` normalize to `epoch=None`) | serving / output-side attacks |

## Threat-model taxonomy

Three orthogonal axes; every attack module declares its coordinates
(`MODES`, `REQUIRES_LABELS`) and restates them in its artifact's
`threat_model` field.

1. **Honest-but-curious vs active/malicious.** Most committed results
   assume a passive cloud that faithfully computes and only observes.
   `active-cloud` drops that assumption: the cloud *reacts* — crafted
   perturbations in returned tensors, crafted optimizer updates in
   training (FL malicious-server analogue).
2. **Labeled vs label-free.** Labeled = the attacker obtains (activation,
   known-token) pairs (oracle / chosen-plaintext / self-labeled replay) —
   strictly stronger than the default E8 model; the resulting curves are
   the honest security parameter, not a claim about the default model.
   Label-free = public text + public base weights only (`ica-bss`,
   `gradient-inversion`, `output-inversion`, the label-free band itself).
3. **Per-epoch vs cross-epoch.** Per-epoch rotation resets the honest
   accumulation window (E-R1a). Cross-epoch attackers refuse to reset:
   `subspace-joint` (shared basis), `leak-accumulation` (cumulative APT
   leak), `stale-key` (handed old keys). The cross-epoch pooling control
   (pooled pairs solved as one W) must sit in the label-free band.

## Capability matrix

| attack | modes | labels | breaks / tests | earlier implementation |
|---|---|---|---|---|
| `accumulation` [er1a] | both | labeled | per-epoch budget: E pairs → polar lstsq → recovery; K50 / max_safe_epoch | rotation_lifetime E-R1a |
| `known-prefix` [er3] | both | partial (scaffold) | fixed-scaffold break; jitter arm vs *passive* alignment | rotation_lifetime E-R3 |
| `stale-key` [er4] | both | insider + fresh pairs | stale keys give no bootstrap (rel_err ≈ √2) | rotation_lifetime E-R4 |
| `sharded` [er5] | both | labeled | static sharding weakens budgets; stagger saves it | rotation_lifetime E-R5 |
| `wire-eval` | both | labeled | per-epoch wire-capture scoring, fwd/bwd arms, pooled control | e9/er_train evaluators (not included in this release) |
| `gradient-inversion` | training | label-free | DLG++ optimization inversion of (h*, g*) | split-training/dlgpp.py, gradient_inversion.py |
| `membership` [ea4] | training | label-free | membership/property probes on boundary features | split-training/membership_inference.py |
| `output-inversion` | inference | label-free | output-side decode (SipIt injectivity line) | split-training/output_inversion.py, sipit_inversion.py |
| `latent-probe` | training | trusted evaluation labels | coordinate, per-token invariant, and full sign-invariant Gram-graph recovery with restart-adjusted confidence bounds | latent-native v5/v6 diagnostic probes |
| `latent-sensitivity` | training | trusted evaluation labels | **v9:** compromise-fraction battery — capture %, chaff-ID %, gauge-compromise %, known-plaintext % sweeps | v9 compromise battery |
| `latent-matching` | training | trusted evaluation labels | **v11:** VMA-style centroid vocabulary matching (Hidden No More, ICML'25) + position-free signature matching (permutation-reversal family, arXiv:2505.18332) | external attack families vs the v9.2 winner |
| `alignment-search` [er8] | both | partial (scaffold) | jitter arm vs a *realigning* attacker | — |
| `active-cloud` | both | label-free (active) | malicious-server probing/steering | — |
| `subspace-joint` | both | labeled | cross-epoch joint solve via stable subspace | — |
| `ica-bss` | both | label-free | higher-order (ICA) label-free attack | — |
| `leak-accumulation` | both | insider (rows) | APT cross-epoch leak accumulation | — |
| `max-effort` | both | inherits target | attacker-optimized distributions | — |

## The six added capabilities

### 1. alignment-search (E-R8) — approximate-alignment attacker vs jittered scaffold

E-R3's jitter arm is only validated against a **passive-alignment**
adversary — one that assumes wire row *i* is assumed prefix position *i*.
**The jittered arm's 0% could reflect that the attacker doesn't try to
realign.** This attacker performs edit-distance / DTW-style realignment
between captured boundary activations and candidate prefix positions
before the W solve, ICP-style: de-rotate with the current W_hat → realign
(`attacker/dtw.py`, pure-python, torch-free testable) → harvest matched
pairs → re-solve. The artifact reports both arms with the
passive-alignment baseline side by side, so a realignment-aware break of
the jittered arm is directly visible.

### 2. active-cloud — malicious-cloud reaction/probing attacks

Every committed break assumes honest-but-curious. A malicious cloud
injects crafted perturbations in the returned (rotated) tensors and
observes subsequent wire tensors; in training mode it crafts optimizer
updates steering the local head to leak W faster (the FL malicious-server
analogue — the server chooses the model the client trains on). The module
implements the offline **AttackPlanner** (probe direction/magnitude
schedule), **ObservationAnalyzer** (pairs → W candidates via the shared
solve), and a documented **ActiveCloudHarness** protocol (below) with a
synthetic in-process harness so the full plan/inject/observe/solve loop
runs end to end. **Integration point:** the live driver that wraps a
deployed `cloud_trainer_server.py` send path is *not* written here — it
belongs next to the server code:

```python
class ActiveCloudHarness:           # attacker/attacks/active_cloud.py
    def inject(self, delta): ...    # add delta (rotated space) to the
                                    # next tensor returned to the local node
    def observe(self): ...          # next wire tensor rows [n, H] after
                                    # the injection (fp32 CPU)
    def epoch(self): ...            # current ratchet epoch (None if off)
    def ground_truth_W(self): ...   # OPTIONAL, simulation only
    def close(self): ...
```

A live harness tees the server's receive path into `observe()` and wraps
its send path with `inject()`; the attack core needs nothing else.

### 3. subspace-joint — joint attacker across epochs

Per-epoch rotation bounds the attacker per epoch — but boundary
activations live on a **stable low-dim subspace**: the basis B barely
moves across epochs; only the key rotates. This attacker alternates
minimization over (shared basis B, per-epoch keys W_t) and needs far
fewer labeled pairs per epoch than the independent per-epoch solves (once
B is fixed, each epoch is a k×k problem, not H×H). **If it works,
max_safe_epoch drops and the B=128 budget choice needs revision.** The
module reports joint vs
independent recovery at the same pairs-per-epoch budget, head to head.

### 4. ica-bss — label-free higher-order attacker

The committed label-free failures (e8_robustness 1a/1b) are second-order
arguments: Q = cov(h')^{-1/2} R cov(h)^{1/2} matches the second moment
for *any* orthogonal R. That argument does not cover higher-order
algorithms. This attacker runs FastICA blind source separation on
within-epoch wire captures (h' = h @ W_t is an orthogonal mixture —
exactly the ICA identifiability setting) and decodes in the recovered
source space, testing **whether the ~0.5% label-free band is robust or
just untested against the right algorithm**. sklearn is not a dependency of
this package, so the framework carries its own pure-torch FastICA
(`attacker/ica.py`: whitening + tanh-negentropy fixed-point) and stays
self-contained. Expected result on real (near-Gaussian) activations: the
band holds, because ICA is unidentifiable on Gaussian sources — but now
it is *measured*, not assumed. (On the synthetic toy world, whose rows
are deliberately clustered and non-Gaussian, ICA does recover structure —
a machinery check, not a security claim.)

### 5. leak-accumulation — persistent APT model

E-R1a/E-R4 bound the attacker **per epoch**; rotation resets the
accumulation window. A persistent adversary who steals a few key rows per
epoch **accumulates them across epochs — the per-epoch leak budget does
NOT reset** just because the key rotated. Composite of the committed
row-leak result (e8_robustness attack 4: leaked rows + orthogonal
completion recovers the leaked coordinates exactly) with the rotation
schedule: per-epoch composite decode plus the *cumulative* leaked-rows
counter that per-epoch budget accounting misses.

### 6. max-effort — "we did not optimize the attacker to completion" pre-emptor

Several defense claims rest on attacks run with a handful of seeds and
one budget grid. This driver reruns a given attack with more solve seeds,
multiple independent initializations, and swept budgets, and reports the
**distribution** (min/p25/mean/p75/max over all reruns), not just the
mean. Factorized toy cores currently: `accumulation`, `ica-bss`
(`--target`). For any other attack, drive it per cell with
`python -m attacker --attack X ... --output cell_i.json` and aggregate
the `.jsonl` journals — every attack writes the same journal schema for
exactly this purpose. Defense claims should be quoted against the p75/max
of this distribution.

## Conventions

- **Torch-guarded imports**: `python -m attacker --help`,
  `--list-attacks`, and every attack's `--help` work on torch-less hosts.
  Torch is only touched inside `run()` / guarded self-test sections.
- **`--self-test`**: pure-python frozen fixtures per module
  (style: split-training/er_ratchet.py);
  `python -m attacker --self-test` aggregates all of them. Torch sections
  run when torch is present (any torch environment).
- **Solve hardening**: one shared solve module
  (`attacker/solve_primitives.py`) wraps `solve_w` / `polar` /
  `recovery_with_what` with a contiguous-fp64 discipline; solves never
  raise — failures return `(None, "error: ...")`
  and are journaled as per-cell error records (style: rotation_lifetime).
- **Artifacts**: every attack writes one JSON artifact with
  `schema` / `config` / `threat_model` / `provenance` / `results` /
  `summary`, plus a crash-safe per-cell JSONL journal at
  `<output>.jsonl` appended after each cell (pattern:
  `rotation_lifetime._append_jsonl`).

## Running each attack

All commands below are machinery-checked end to end (`--toy --quick`);
drop `--quick` for the full toy grids. Real-capture runs use
`--capture-dir` + `--canonical-pt` where supported, or the per-script
flags (`--dlgpp`, `--output-inversion-script`) for model-driven runs.

```bash
PY=python3   # any torch environment

# --- attacks subsuming the earlier standalone scripts ---
$PY -m attacker --mode training  --attack accumulation --toy --quick --output er1.json
$PY -m attacker --mode inference --attack known-prefix --toy --quick --output er3.json
$PY -m attacker --mode training  --attack stale-key --toy --quick --output er4.json
$PY -m attacker --mode training  --attack sharded --toy --quick --output er5.json
$PY -m attacker --mode training  --attack wire-eval --toy --quick --output wire.json
$PY -m attacker --mode training  --attack wire-eval \
      --capture-dir ER_CAPTURE_DIR --canonical-pt replay.pt --output wire.json
$PY -m attacker --mode training  --attack gradient-inversion --toy --quick --output dlg.json
$PY -m attacker --mode training  --attack gradient-inversion \
      --dlgpp "--model /models/qwen3-0.6b --corpus-file wiki.txt --seq-prior" --output dlg.json
$PY -m attacker --mode training  --attack membership --features run.jsonl --output ea4.json
$PY -m attacker --mode inference --attack output-inversion --toy --quick --output oinv.json
$PY -m attacker --mode inference --attack output-inversion \
      --output-inversion-script "--model /models/qwen3-0.6b" --output oinv.json

# --- the six added capabilities ---
$PY -m attacker --mode training  --attack alignment-search --toy --quick --output er8.json
$PY -m attacker --mode training  --attack active-cloud --toy --quick --output active.json
$PY -m attacker --mode training  --attack subspace-joint --toy --quick --output sj.json
$PY -m attacker --mode training  --attack ica-bss --toy --quick --output ica.json
$PY -m attacker --mode training  --attack leak-accumulation --toy --quick --output leak.json
$PY -m attacker --mode training  --attack max-effort --toy --quick \
      --target accumulation --output me.json

# --- v9/v11 latent arms (run against regenerated bundles) ---
$PY -m attacker --mode training  --attack latent-probe --bundle bundle.pt --output gate.json
$PY -m attacker --mode training  --attack latent-sensitivity --bundle bundle.pt --output sens.json
$PY -m attacker --mode training  --attack latent-matching --bundle bundle.pt --output match.json
```

## What is fully implemented vs integration-point-only

**Fully implemented and toy-verified:** the shared capture/pair-loading
core for both schemas; the shared solve primitives; the pure-python DTW /
edit-distance core; the pure-torch FastICA; the six added capabilities'
attack math end to end on synthetic tensors (including the
plan/inject/observe/solve loop of active-cloud over a synthetic harness);
framework paths of accumulation, known-prefix, stale-key, sharded,
wire-eval (synthetic captures through the real loading path),
membership (real math via split-training/membership_inference.py),
gradient-inversion (synthetic DLG + dlgpp.py driver),
output-inversion (synthetic decode + script driver); max-effort
factorized cores for accumulation and ica-bss.

**Integration points (documented, not silently stubbed):**

- `active-cloud` live server driver (harness glue next to the deployed
  servers — protocol above).
- Real-capture drivers for `alignment-search`, `subspace-joint`,
  `ica-bss`, `leak-accumulation`: each currently raises a descriptive
  error naming exactly what to feed it (per-epoch capture pairs via
  `--capture-dir`); the attack math itself is implemented and tested.
- `known-prefix` / `stale-key` / `sharded` model-driven runs: use
  `split-training/rotation_lifetime.py --experiment er3|er4|er5` (the
  framework's toy paths exercise the same attack math).
- `max-effort` targets beyond the two factorized cores: drive per cell
  via the CLI and aggregate journals.
- `latent-sensitivity` and `latent-matching` are fully implemented and
  run against real regenerated bundles (v9/v11 campaigns): the sensitivity
  battery's four compromise sweeps and the matching family's VMA-centroid
  and position-free signature arms.  Result JSONs are not included in this
  release.

## Efficiency design decision

**Separate attackers on a shared capture harness, parallel across hosts;
a composite red-team cell only as the final worst-case stress test.**
Each attack is an independent module reading the same capture directory
through `attacker/captures.py`, so N attacks × M hosts fan out
embarrassingly: one capture run feeds every attacker (captures are the
expensive part — model forward passes on the defense side; the attacks
are solves on cached rows). Per-cell JSONL journals make partial results
durable across host failures, and artifacts carry provenance for the
collection/ledger conventions. Only after the separate attackers are
characterized do you compose them (e.g. alignment-search harvesting pairs
*for* accumulation, max-effort wrapping the strongest cell) into a single
red-team run as the final worst-case number — composites are expensive,
harder to attribute, and unnecessary for the per-mechanism claims.

## Layout

```
attacker/
  __main__.py            CLI dispatcher (--mode/--attack/--self-test/--list-attacks)
  solve_primitives.py    shared solve core (solve_w/polar/recovery; contiguous-fp64)
  captures.py            wire-capture loading, both sidecar schemas, per-epoch grouping
  dtw.py                 pure-python DTW + edit alignment (torch-free)
  ica.py                 pure-torch FastICA (whitening + tanh negentropy)
  artifacts.py           artifact skeleton + crash-safe JSONL journal
  synthetic.py           toy rotated-boundary world + toy capture writer
  attacks/
    __init__.py          registry (name -> module, modes, labels)
    common.py            shared CLI flags, decode surrogates, error journaling
    accumulation.py      E-R1a per-epoch budget + cross-epoch control
    known_prefix.py      E-R3 fixed vs jittered scaffold
    stale_key.py         E-R4 stale-key / ratchet-chain attack
    sharded.py           E-R5 block-diagonal + staggered rotation
    wire_eval.py         per-epoch wire-capture scoring (e9/er_train superset)
    gradient_inversion.py DLG++ (synthetic + dlgpp.py driver)
    membership.py        E-A4 wrapper (membership_inference.py math)
    output_inversion.py  output-side inversion (+ script driver)
    alignment_search.py  E-R8 realigning attacker
    active_cloud.py      malicious-cloud planner/analyzer + harness protocol
    subspace_joint.py    cross-epoch joint (B, W_t) alternating solve
    ica_bss.py           label-free ICA/BSS attacker
    leak_accumulation.py APT cross-epoch leak accumulation
    max_effort.py        attacker-optimization distribution driver
```
