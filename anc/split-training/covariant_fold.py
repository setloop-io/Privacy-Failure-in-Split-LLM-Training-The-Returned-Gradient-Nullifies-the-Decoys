#!/usr/bin/env python3
"""E-R6 — the covariant weight fold: correct compute on rotated residuals.

The E8 defense rotates the boundary activation h -> h @ W (W secret
orthogonal). A plain cloud cannot consume hW: its weights expect the
CANONICAL residual stream. This experiment implements and costs the exact
"unwrap-inside-the-layer" fold that lets the cloud compute correctly on the
rotated stream WITHOUT holding W explicitly — it holds folded weights.

Derivation (exact for pre-RMSNorm transformer layers). The cloud receives
x = h @ W and must keep the residual stream rotated end-to-end:

  (a) INPUT consumers of the normed residual (q/k/v/gate/up projections):
      RMSNorm is row-scale invariant and W is orthogonal, so per row
      ||hW|| = ||h|| and RMSNorm(hW) = RMSNorm(h) @ W. The layer input is
      therefore n' = RMSNorm(hW) diag(g) = (RMSNorm(h) diag(g)) @
      [diag(g)^-1 W ... ] — concretely, feeding n' into the folded weight

          M_in = diag(g)^-1 @ W^T @ diag(g) @ W_in

      reproduces the CANONICAL pre-activation u @ W_in exactly, where g is
      the corresponding layernorm gain (input_layernorm for q/k/v,
      post_attention_layernorm for gate/up).
  (b) OUTPUT producers (o_proj, down_proj) get

          M_out = W_out @ W

      so the block output re-enters the residual stream in the ROTATED
      basis (y @ W), keeping the stream rotated across layers.

With torch.nn.Linear convention y = x @ P.T the stored weights become
P_in' = P_in @ diag(g) @ W @ diag(g)^-1  and  P_out' = W^T @ P_out.

WHAT THIS BUYS AND WHAT IT DOES NOT (read before citing):
  - Attention internals (RoPE, softmax, QK^T) operate on CANONICAL
    activations inside the layer — inherently. Cloud RAM during the epoch
    is the canonical-form sanctuary (the equivariance trilemma: trusted
    edge, enclave, or transformed weights). The fold moves trust to "cloud
    memory during the epoch" while keeping the WIRE and any logged boundary
    tensors rotated per epoch.
  - It removes PER-EPOCH KEY SHIPPING: the cloud can derive W_t from an
    8-byte ratchet seed and fold locally, instead of the trusted node
    pushing folded weights each epoch. Both variants are costed here.
  - Folded weights provide NO KEY SECRECY from anyone who can compare them
    with the public pretrained originals.  For a producer projection
    P_out' = W^T @ P_out, a full-rank public P_out gives
    W^T = P_out' @ pinv(P_out); stacked public projections give the same
    least-squares recovery when one projection is rank deficient.  This is a
    single-snapshot attack, so per-epoch ratcheting does not bound it.  The
    dense transform is outside ArrowMatch's direction-preserving premise, but
    it is broken by simpler linear algebra.  `probe_e9_equivariance.py`
    demonstrates recovery in float64 and after bf16 storage quantization.
  - It does NOT remove cloud trust. Its benefit is logistical (no repeated key
    transport), not cryptographic key hiding. See known_limitations.

Usage:
    python covariant_fold.py --help        # works without torch
    python covariant_fold.py --self-test   # torch-free fixtures
    python covariant_fold.py --toy --quick # CPU machinery check
    python covariant_fold.py --model <hf-model> --layers 2-37 --mode all --dtype fp16
"""

import argparse
import json
import math
import os
import sys
import time
from fractions import Fraction

# Guarded heavy imports: `--help`/`--self-test` must work on torch-less hosts.
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from split_trainer import _write_training_status
    from trained_inversion import make_provenance, seed_all
    from e8_obfuscation import make_secret
except ImportError:  # pragma: no cover - torch-less host
    _write_training_status = lambda **k: None
    make_provenance = seed_all = make_secret = None

# Weight matrices consumed/produced per pre-RMSNorm decoder layer.
CONSUMERS_INPUT_NORM = ("q_proj", "k_proj", "v_proj")     # use input_layernorm g
CONSUMERS_POST_NORM = ("gate_proj", "up_proj")            # use post_attention_layernorm g
PRODUCERS = ("o_proj", "down_proj")


# Pure-python helpers (torch-free; --self-test pins them with frozen fixtures)
def parse_layers(spec, n_layers):
    """"a-b" inclusive layer range, clamped/validated against n_layers."""
    a_s, _, b_s = spec.partition("-")
    a, b = int(a_s), int(b_s)
    if not (0 <= a <= b < n_layers):
        raise ValueError(f"--layers {spec} outside [0, {n_layers - 1}]")
    return a, b


def matmul_pure(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B)))
             for j in range(len(B[0]))] for i in range(len(A))]


def diag_pure(v):
    return [[v[i] if i == j else Fraction(0) for j in range(len(v))]
            for i in range(len(v))]


def transpose_pure(A):
    return [list(r) for r in zip(*A)]


def fold_in_pure(W, g, W_in):
    """M_in = diag(g)^-1 @ W^T @ diag(g) @ W_in, exact Fraction arithmetic."""
    dg = diag_pure(g)
    dgi = diag_pure([1 / x for x in g])
    return matmul_pure(dgi, matmul_pure(transpose_pure(W),
                                        matmul_pure(dg, W_in)))


def fold_out_pure(W, W_out):
    """M_out = W_out @ W, exact Fraction arithmetic."""
    return matmul_pure(W_out, W)


def profile_overhead_seconds(n_bytes, mbps):
    """Derived-not-measured transport time: n_bytes over mbps Mbit/s."""
    return n_bytes * 8 / (mbps * 1e6)


# Frozen --self-test fixtures: a 4-dim worked fold example. W is the
# (symmetric, orthogonal) Hadamard H4/2, g = (2,4,1,2), and W_in a small
# integer matrix. The expected matrices were computed once with exact
# Fraction arithmetic and frozen here so the pure-python fold wiring is
# pinned independently of any torch availability.
FIXTURE_W = [[Fraction(x, 2) for x in row] for row in
             [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]]
FIXTURE_G = [Fraction(2), Fraction(4), Fraction(1), Fraction(2)]
FIXTURE_W_IN = [[Fraction(x) for x in row] for row in
                [[1, 0, 2, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]]
FIXTURE_M_IN = [[Fraction(x) for x in row] for row in
                [[Fraction(3, 4), Fraction(5, 4), Fraction(3, 2), Fraction(3, 2)],
                 [Fraction(3, 8), Fraction(-3, 8), Fraction(1, 4), Fraction(-3, 4)],
                 [Fraction(1, 2), Fraction(3, 2), Fraction(1), Fraction(1)],
                 [Fraction(1, 4), Fraction(-5, 4), Fraction(3, 2), Fraction(-1, 2)]]]
FIXTURE_M_OUT = [[Fraction(x) for x in row] for row in
                 [[Fraction(3, 2), Fraction(3, 2), Fraction(-1, 2), Fraction(-1, 2)],
                  [Fraction(1), Fraction(-1), Fraction(0), Fraction(0)],
                  [Fraction(1), Fraction(0), Fraction(1), Fraction(0)],
                  [Fraction(1), Fraction(0), Fraction(-1), Fraction(0)]]]


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("4-dim worked fold example (exact Fraction arithmetic):")
    check("M_in == frozen fixture", fold_in_pure(FIXTURE_W, FIXTURE_G,
                                                 FIXTURE_W_IN) == FIXTURE_M_IN)
    check("M_out == frozen fixture",
          fold_out_pure(FIXTURE_W, FIXTURE_W_IN) == FIXTURE_M_OUT)
    # algebraic sanity: the fixture W must be orthogonal
    wwt = matmul_pure(FIXTURE_W, transpose_pure(FIXTURE_W))
    ident = diag_pure([Fraction(1)] * 4)
    check("fixture W is orthogonal (W W^T == I)", wwt == ident)

    print("--layers parsing/validation:")
    check("'2-37' on a 40-layer model -> (2, 37)",
          parse_layers("2-37", 40) == (2, 37))
    check("'0-0' is a single layer", parse_layers("0-0", 4) == (0, 0))
    for bad in ("5-2", "-1-3", "0-40", "38-40"):
        try:
            parse_layers(bad, 40)
            check(f"'{bad}' rejected", False)
        except ValueError:
            check(f"'{bad}' rejected", True)

    print("profile overhead arithmetic (derived-not-measured):")
    check("8 MiB over 1000 Mbps = 0.067108864 s",
          abs(profile_overhead_seconds(8 * 2**20, 1000)
              - 0.067108864) < 1e-9)
    check("halving the bandwidth doubles the time",
          abs(profile_overhead_seconds(1000, 500)
              - 2 * profile_overhead_seconds(1000, 1000)) < 1e-12)

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


# Toy pre-RMSNorm decoder (torch path). Mirrors the Mistral attribute layout
# (input_layernorm / self_attn.{q,k,v,o}_proj / post_attention_layernorm /
# mlp.{gate,up,down}_proj) so the SAME fold code runs on toy and 12B.
if torch is not None:
    class RMSNorm(torch.nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            ms = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(ms + self.eps) * self.weight


    class ToyAttention(torch.nn.Module):
        def __init__(self, dim, nheads=4):
            super().__init__()
            self.nheads = nheads
            self.q_proj = torch.nn.Linear(dim, dim, bias=False)
            self.k_proj = torch.nn.Linear(dim, dim, bias=False)
            self.v_proj = torch.nn.Linear(dim, dim, bias=False)
            self.o_proj = torch.nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            b, t, d = x.shape
            hd = d // self.nheads
            q = self.q_proj(x).view(b, t, self.nheads, hd).transpose(1, 2)
            k = self.k_proj(x).view(b, t, self.nheads, hd).transpose(1, 2)
            v = self.v_proj(x).view(b, t, self.nheads, hd).transpose(1, 2)
            mask = torch.triu(torch.ones(t, t, dtype=torch.bool,
                                         device=x.device), 1)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
            scores = scores.masked_fill(mask, float("-inf"))
            a = torch.softmax(scores, dim=-1) @ v
            return self.o_proj(a.transpose(1, 2).reshape(b, t, d))


    class ToyMLP(torch.nn.Module):
        def __init__(self, dim, mult=4):
            super().__init__()
            self.gate_proj = torch.nn.Linear(dim, mult * dim, bias=False)
            self.up_proj = torch.nn.Linear(dim, mult * dim, bias=False)
            self.down_proj = torch.nn.Linear(mult * dim, dim, bias=False)

        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x))
                                  * self.up_proj(x))


    class ToyDecoderLayer(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.input_layernorm = RMSNorm(dim)
            self.self_attn = ToyAttention(dim)
            self.post_attention_layernorm = RMSNorm(dim)
            self.mlp = ToyMLP(dim)

        def forward(self, x):
            x = x + self.self_attn(self.input_layernorm(x))
            return x + self.mlp(self.post_attention_layernorm(x))


    class ToyFoldModel(torch.nn.Module):
        def __init__(self, vocab=128, dim=64, n_layers=4):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(vocab, dim)
            self.layers = torch.nn.ModuleList(
                [ToyDecoderLayer(dim) for _ in range(n_layers)])
            self.norm = RMSNorm(dim)
            self.lm_head = torch.nn.Linear(dim, vocab, bias=False)

        def forward(self, ids):
            h = self.embed_tokens(ids)
            for layer in self.layers:
                h = layer(h)
            return self.lm_head(self.norm(h))


def _get_proj(layer, name):
    """Fetch q/k/v/o from self_attn, gate/up/down from mlp (Mistral + toy)."""
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, name):
        return getattr(layer.self_attn, name)
    if hasattr(layer, "mlp") and hasattr(layer.mlp, name):
        return getattr(layer.mlp, name)
    raise AttributeError(f"layer has no projection {name}")


def fold_layer(layer, W):
    """Rewrite `layer`'s weights in place so that feeding it x = h @ W
    produces the canonical internal activations and a ROTATED output.
    W: [H, H] orthogonal, fp32 CPU. Weights are folded in fp64 and cast back
    to their original dtype."""
    Wd = W.double().cpu()
    g_in = layer.input_layernorm.weight.detach().double().cpu()
    g_post = layer.post_attention_layernorm.weight.detach().double().cpu()

    def fold_consumer(proj, g):
        dg = torch.diag(g)
        dgi = torch.diag(1.0 / g)
        # P' = P @ diag(g) @ W @ diag(g)^-1  (stored-weight convention)
        p_new = proj.weight.detach().double().cpu() @ dg @ Wd @ dgi
        proj.weight.data = p_new.to(proj.weight.dtype).to(
            proj.weight.device)

    def fold_producer(proj):
        # P' = W^T @ P
        p_new = Wd.T @ proj.weight.detach().double().cpu()
        proj.weight.data = p_new.to(proj.weight.dtype).to(
            proj.weight.device)

    for name in CONSUMERS_INPUT_NORM:
        fold_consumer(_get_proj(layer, name), g_in)
    for name in CONSUMERS_POST_NORM:
        fold_consumer(_get_proj(layer, name), g_post)
    for name in PRODUCERS:
        fold_producer(_get_proj(layer, name))
    return layer


def unfold_grads_into(canonical_layers, folded_layers, W):
    """[ER] Accumulate folded-basis parameter grads onto the CANONICAL
    layers (+= into .grad), so the cloud's optimizer updates canonical
    weights and the folded copy stays an exact per-epoch reparametrization
    (loss-curve identity with the non-rotated baseline). In fp64 like the
    fold itself. The maps (verified to ~1e-6 against a canonical
    forward/backward):

      consumer P' = P @ M, M = diag(g) @ W @ diag(g)^-1
          => dL/dP = dL/dP' @ M^T
      producer  P' = W^T @ P
          => dL/dP = W @ dL/dP'
      norm gain g (feeds a consumer group): the folded graph's own dL/dg is
          w.r.t. the ROTATED normed stream and is NOT the canonical grad.
          With Z_c = dL/dP'_c @ diag(g)^-1 @ W^T = G_c^T r^c (G_c the grad at
          consumer c's canonical pre-activation, r^c the canonical normed
          input), the exact canonical gain grad is

              dL/dg = sum_c (Z_c * P_c).sum(dim=0)

      Any other param (biases) maps identity. Folded grads are cleared after
      the remap."""
    Wd = W.double().cpu()
    for can_layer, fol_layer in zip(canonical_layers, folded_layers):
        can_params = dict(can_layer.named_parameters())
        mapped = {}
        gain_grads = {}
        for norm_name, group in (("input_layernorm", CONSUMERS_INPUT_NORM),
                                 ("post_attention_layernorm",
                                  CONSUMERS_POST_NORM)):
            g = getattr(can_layer, norm_name).weight.detach().double().cpu()
            M = torch.diag(g) @ Wd @ torch.diag(1.0 / g)
            dgiWt = torch.diag(1.0 / g) @ Wd.T
            gg = None
            for name in group:
                fp = _get_proj(fol_layer, name).weight
                if fp.grad is None:
                    continue
                dP = fp.grad.detach().double().cpu()
                mapped[id(fp)] = dP @ M.T
                # exact canonical gain grad via Z = G^T r^c (see docstring)
                Z = dP @ dgiWt
                P = _get_proj(can_layer, name).weight.detach().double().cpu()
                contrib = (Z * P).sum(dim=0)
                gg = contrib if gg is None else gg + contrib
            if gg is not None:
                gain_grads[f"{norm_name}.weight"] = gg
        for name in PRODUCERS:
            fp = _get_proj(fol_layer, name).weight
            if fp.grad is not None:
                mapped[id(fp)] = Wd @ fp.grad.detach().double().cpu()
        for pname, p in fol_layer.named_parameters():
            if p.grad is None:
                continue
            g_can = gain_grads.get(pname)
            if g_can is None:
                g_can = mapped.get(id(p))
            if g_can is None:
                g_can = p.grad.detach().double().cpu()
            cp = can_params[pname]  # layer-local names match 1:1
            g_can = g_can.to(dtype=cp.dtype, device=cp.device)
            cp.grad = g_can if cp.grad is None else cp.grad + g_can
            p.grad = None


def unfold_check(layer_factory, hidden_dim, seed, device, dtype=None):
    """Mechanistic per-layer check: y_folded(hW) == y_canonical(h) @ W."""
    dtype = dtype or torch.float32
    seed_all(seed)
    layer = layer_factory().to(device)
    ref_layer = layer_factory().to(device)
    ref_layer.load_state_dict(layer.state_dict())
    W = make_secret(hidden_dim, seed)
    h = torch.randn(1, 8, hidden_dim, generator=torch.Generator().manual_seed(seed))
    h = h.to(device=device, dtype=dtype)
    with torch.no_grad():
        y_ref = ref_layer(h)
        fold_layer(layer, W)
        y_fold = layer((h.float().cpu() @ W).to(device=device, dtype=h.dtype))
    err = (y_fold.double() - y_ref.double() @ W.double().to(y_fold.device)
           ).abs().max().item()
    return err


def build_model(args):
    """(model_kind, layers-accessor, forward helpers). Toy or HF Mistral."""
    if args.toy:
        model = ToyFoldModel().to(args.device)
        return model, list(model.layers), model
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise RuntimeError("transformers not installed; use --toy for a CPU test")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(args.device).eval()
    return model, list(model.model.layers), model


def encode_prompt(args, text):
    if args.toy:
        ids = [ord(c) % 128 for c in text][:args.seq_len]
        return torch.tensor([ids], dtype=torch.long, device=args.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    return tok(text, return_tensors="pt").input_ids.to(args.device)


def exactness_run(args, model, layers):
    """Baseline greedy vs folded-middle greedy on rotated input, unwrapped
    with W^T before the tail.

    Order matters for memory (12B fp16 ~24GB): the baseline greedy is run
    FIRST on the UNFOLDED model and its token trajectory recorded; then the
    middle is folded in place and teacher-forced over the same trajectory.
    Per-step logits are compared at matching prefixes, so no deepcopy of the
    middle stack is ever held. Also returns the measured fold seconds (used
    by the cost report in --mode all)."""
    n_layers = len(layers)
    a, b = parse_layers(args.layers, n_layers)
    hidden_dim = (model.embed_tokens.weight.shape[1] if args.toy
                  else model.config.hidden_size)
    W = make_secret(hidden_dim, args.seed)
    ids = encode_prompt(args, "The history of the Roman Empire began")
    gen = args.gen_tokens

    core = None if args.toy else model.model
    rotary = None if args.toy else getattr(core, "rotary_emb", None)

    def lkws(h, pos_ids):
        kw = {"attention_mask": None, "position_ids": pos_ids}
        if rotary is not None:
            kw["position_embeddings"] = rotary(h, pos_ids)
        return kw

    def run_stack(stack, h, pos_ids):
        for lyr in stack:
            if args.toy:
                h = lyr(h)
            else:
                out = lyr(h, **lkws(h, pos_ids))
                h = out[0] if isinstance(out, tuple) else out
        return h

    def head_fwd(x_ids):
        h = (model.embed_tokens if args.toy else core.embed_tokens)(x_ids)
        pos = torch.arange(x_ids.shape[1], device=x_ids.device).unsqueeze(0)
        return run_stack(layers[:a], h, pos)

    def tail_fwd(h):
        pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
        h = run_stack(layers[b + 1:], h, pos)
        norm = model.norm if args.toy else core.norm
        lm_head = model.lm_head
        return lm_head(norm(h))

    def logits(x_ids, folded):
        """Last-position logits; with folded=True the boundary h is rotated
        by W, run through the (folded) middle, and unwrapped with W^T."""
        pos = torch.arange(x_ids.shape[1], device=x_ids.device).unsqueeze(0)
        h = head_fwd(x_ids)
        if folded:
            x = (h.float().cpu() @ W).to(device=h.device, dtype=h.dtype)
            x = run_stack(layers[a:b + 1], x, pos)
            h = (x.float().cpu() @ W.T).to(device=x.device, dtype=x.dtype)
        else:
            h = run_stack(layers[a:b + 1], h, pos)
        return tail_fwd(h)[:, -1].float()

    with torch.no_grad():
        # pass 1: baseline greedy on the UNFOLDED model
        base_ids = ids.clone()
        base_logits = []
        for _ in range(gen):
            lb = logits(base_ids, folded=False)
            base_logits.append(lb)
            base_ids = torch.cat([base_ids, lb.argmax(-1, keepdim=True)], 1)

        # fold in place (timed — this IS the per-epoch cloud-local cost)
        t0 = time.time()
        for lyr in layers[a:b + 1]:
            fold_layer(lyr, W)
        fold_seconds = time.time() - t0
        print(f"[fold] folded layers {a}-{b} in {fold_seconds:.1f}s")

        # pass 2: folded model teacher-forced over the baseline trajectory
        max_dlogit = 0.0
        n_match = 0
        for step in range(gen):
            lf = logits(base_ids[:, :ids.shape[1] + step], folded=True)
            max_dlogit = max(max_dlogit,
                             (base_logits[step] - lf).abs().max().item())
            n_match += int((lf.argmax(-1)
                            == base_ids[:, ids.shape[1] + step]).sum().item())
    print(f"[exactness] {gen} greedy tokens: max |dlogit| = "
          f"{max_dlogit:.3e}, token match {n_match}/{gen}")
    return {"layers": [a, b], "gen_tokens": gen,
            "max_abs_dlogit": max_dlogit, "greedy_token_matches": n_match,
            "greedy_token_match_frac": round(n_match / gen, 4),
            "fold_seconds_measured": round(fold_seconds, 4),
            "comparison": "teacher-forced over the baseline greedy "
                          "trajectory; per-step last-position logits",
            "tolerance_fp32": 1e-4,
            "within_fp32_tolerance": (max_dlogit < 1e-4
                                      if args.dtype == "fp32" or args.toy
                                      else None)}


def cost_run(args, model, layers, measured_fold_seconds=None):
    """Per-epoch fold cost: seconds (cloud-local variant) and bytes
    (trusted-node fold + push variant). If the exactness pass already folded
    and timed the middle, its measurement is reused; otherwise the fold is
    performed here (mutating the middle layers) purely for timing."""
    n_layers = len(layers)
    a, b = parse_layers(args.layers, n_layers)
    middle = layers[a:b + 1]
    n_bytes = sum(p.numel() for lyr in middle for p in lyr.parameters()) * 2
    if measured_fold_seconds is None:
        W = make_secret(model.embed_tokens.weight.shape[1] if args.toy
                        else model.config.hidden_size,
                        ratchet_seed_for_epoch(args.seed, epoch=0))
        t0 = time.time()
        for lyr in middle:
            fold_layer(lyr, W)
        measured_fold_seconds = time.time() - t0
    fold_seconds = measured_fold_seconds
    profiles = {"direct_0ms": args.bw_direct, "benign_20ms": args.bw_20ms,
                "benign_80ms": args.bw_80ms, "hostile": args.bw_hostile}
    overhead = {name: round(profile_overhead_seconds(n_bytes, mbps), 4)
                for name, mbps in profiles.items()}
    print(f"[cost] layers {a}-{b}: fold {fold_seconds:.1f}s on "
          f"{args.device}; folded weights {n_bytes / 2**30:.3f} GiB; "
          f"push overhead {overhead}")
    return {"layers": [a, b], "n_middle_layers": len(middle),
            "cloud_local_fold": {
                "fold_seconds": round(fold_seconds, 4),
                "bytes_shipped_per_epoch": 8,
                "note": "cloud derives W_t from the 8-byte ratchet seed and "
                        "folds locally; fold_seconds measured on "
                        f"{args.device}"},
            "trusted_fold_and_push": {
                "folded_weight_bytes_fp16": n_bytes,
                "push_seconds_by_profile_derived": overhead,
                "bandwidth_mbps_by_profile": profiles,
                "note": "DERIVED-NOT-MEASURED: bytes / profile bandwidth; "
                        "bandwidths from the repo transport profiles "
                        "(bin/set_wan.sh caps, noisy-neighbor contention)"}}


def ratchet_seed_for_epoch(master_seed, epoch):
    import hashlib
    return int.from_bytes(hashlib.sha256(
        f"{master_seed}:{epoch}".encode()).digest()[:8], "little")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.expanduser(
        "~/experiments/models/mistral-nemo-12b-instruct"),
                    help="HF model path (ignored with --toy)")
    ap.add_argument("--toy", action="store_true",
                    help="tiny built-in pre-RMSNorm model (CPU machinery "
                         "check only)")
    ap.add_argument("--layers", default="2-37",
                    help="middle-layer range to fold, inclusive (12B: 2-37)")
    ap.add_argument("--mode", choices=["exactness", "cost", "all"],
                    default="all")
    ap.add_argument("--gen-tokens", type=int, default=32,
                    help="greedy tokens for the exactness comparison")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"],
                    default="fp16",
                    help="12B fits 24GB in fp16; use fp32 for a near-exact "
                         "tolerance check")
    ap.add_argument("--device",
                    default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bw-direct", type=float, default=1000.0,
                    help="Mbps, direct profile (derived-not-measured)")
    ap.add_argument("--bw-20ms", type=float, default=1000.0,
                    help="Mbps, benign 20ms profile (derived-not-measured)")
    ap.add_argument("--bw-80ms", type=float, default=1000.0,
                    help="Mbps, benign 80ms profile (derived-not-measured)")
    ap.add_argument("--bw-hostile", type=float, default=600.0,
                    help="Mbps, hostile profile: 1Gbps cap minus ~400Mbps "
                         "noisy-neighbor contention (bin/noisy_neighbor.sh; "
                         "derived-not-measured)")
    ap.add_argument("--quick", action="store_true",
                    help="16 gen tokens (with --toy: full CPU check)")
    ap.add_argument("--output", default="covariant_fold.json")
    ap.add_argument("--self-test", action="store_true",
                    help="pure-python fixture checks; no torch needed")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test())

    if torch is None or make_secret is None:
        ap.error("torch/transformers not installed; install them or run --help only")

    if args.quick:
        args.gen_tokens = 16
    if args.toy and args.layers == "2-37":
        args.layers = "1-2"  # toy has 4 layers; clamp the 12B default

    seed_all(args.seed)
    _write_training_status(state="running", task="covariant_fold",
                           mode=args.mode, layers=args.layers,
                           started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    model, layers, _ = build_model(args)
    n_layers = len(layers)
    hidden_dim = (model.embed_tokens.weight.shape[1] if args.toy
                  else model.config.hidden_size)
    print(f"[model] {'toy' if args.toy else args.model}: {n_layers} layers, "
          f"hidden={hidden_dim}, device={args.device}")

    # per-layer mechanistic check (always; cheap)
    if args.toy:
        layer_factory = lambda: ToyDecoderLayer(hidden_dim)
        per_layer_err = unfold_check(layer_factory, hidden_dim, args.seed,
                                     args.device)
        print(f"[check] single-layer unfold err (fp32 toy): "
              f"{per_layer_err:.3e}")
    else:
        per_layer_err = None

    provenance = make_provenance(None, "none (no corpus; exactness prompt "
                                 "is fixed in-script)", 0, [],
                                 model_path=(None if args.toy else args.model))
    out = {
        "schema": "dtraining.covariant_fold.v1",
        "config": {"model": "toy" if args.toy else args.model,
                   "n_layers": n_layers, "layers": args.layers,
                   "mode": args.mode, "gen_tokens": args.gen_tokens,
                   "dtype": "fp32" if args.toy else args.dtype,
                   "device": args.device, "W_seed": args.seed,
                   "quick": args.quick},
        "threat_model": "honest-but-curious cloud holding FOLDED middle-layer "
                        "weights (never W explicitly); the wire and any logged "
                        "boundary tensors stay rotated per epoch under the E8 "
                        "ratchet. The cloud CAN derive canonical activations "
                        "inside its own RAM during the epoch — see "
                        "known_limitations.",
        "interpretation": "exactness: folded-model logits given rotated "
                          "input, unwrapped with W^T, must match baseline "
                          "greedy logits (fp32 near-exact, <1e-4; achieved "
                          "delta recorded) with exact greedy-token match "
                          "over >=32 tokens => the fold is a correct "
                          "reparametrization, not an approximation. cost: "
                          "cloud-local fold seconds/epoch vs trusted-node "
                          "fold+push bytes/epoch; push overhead is "
                          "derived-not-measured from profile bandwidths.",
        "provenance": provenance,
        "measurement_kind": "lab-harness: fold executed in-process on the "
                            "loaded checkpoint; transport seconds derived "
                            "from profile bandwidths, not measured",
        "evidence_status": "primary",
        "known_limitations": [
            "attention/MLP internals (RoPE, softmax, QK^T) are CANONICAL "
            "during compute: cloud RAM is the canonical-form sanctuary "
            "(equivariance trilemma: trusted edge, enclave, or transformed "
            "weights). The fold moves trust to 'cloud memory during the "
            "epoch'; it removes per-epoch key shipping, not cloud trust",
            "the 27B hybrid linear-attention layers (gated-deltanet conv "
            "paths) are NOT covered in v1 — dense pre-RMSNorm architectures "
            "only (Mistral-NeMo-12B verified; Llama-family should port "
            "directly but is unmeasured)",
            "the fold does not protect KV caches or logits, which are also "
            "canonical on the cloud",
            "bandwidth overheads are derived-not-measured (bytes / profile "
            "Mbps from bin/set_wan.sh caps and bin/noisy_neighbor.sh "
            "contention)"],
        "summary": {},
        "results": {},
    }

    if per_layer_err is not None:
        out["results"]["per_layer_unfold_check"] = {
            "max_abs_err": per_layer_err, "tolerance_fp32": 1e-4,
            "within_tolerance": per_layer_err < 1e-4}

    if args.mode in ("exactness", "all"):
        out["results"]["exactness"] = exactness_run(args, model, layers)
        out["summary"]["exactness_max_abs_dlogit"] = \
            out["results"]["exactness"]["max_abs_dlogit"]
        out["summary"]["exactness_token_match_frac"] = \
            out["results"]["exactness"]["greedy_token_match_frac"]
    if args.mode in ("cost", "all"):
        # in --mode all the exactness pass already folded and timed the
        # middle; reuse that measurement instead of a second (corrupting) fold
        measured = (out["results"].get("exactness", {}) or {}).get(
            "fold_seconds_measured")
        out["results"]["cost"] = cost_run(args, model, layers,
                                          measured_fold_seconds=measured)
        out["summary"]["fold_seconds"] = \
            out["results"]["cost"]["cloud_local_fold"]["fold_seconds"]
        out["summary"]["folded_weight_bytes_fp16"] = \
            out["results"]["cost"]["trusted_fold_and_push"
                                  ]["folded_weight_bytes_fp16"]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    _write_training_status(state="done", result_file=args.output)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
