#!/usr/bin/env python3
"""
E-D1: draft-model speculative decoding acceptance harness (single-node, in-process).

A small DRAFT model proposes K tokens per round; the TARGET verifies the whole
drafted block in ONE forward pass (single branch — safe for hybrid linear-attention
models like Qwen3.6 which cannot do multi-branch verification because linear state
pollutes sequentially). Greedy/argmax only, so the output is token-identical to
plain target decoding; we assert that against a sequential baseline.

This experiment is network-independent: it measures acceptance length. It
reports TWO rates and they are not the same number -- do not quote one for
the other:

  tokens_per_target_forward = tokens / (verification forwards + prefill)
      ACCEPTANCE QUALITY only. Ignores the state-rebuild replay.
  tokens_per_round_trip     = tokens / (the above + replay forwards)
      THE NETWORK-RELEVANT ONE. In the split deployment every target forward
      is one WAN round trip, and a PARTIAL accept forces a SECOND target
      forward inside the same round: the hybrid recurrent state is a single
      in-place summary tensor that cannot be rewound to a mid-block position,
      so the committed prefix must be re-forwarded to rebuild it (see
      speculative_generate). That replay is a second round trip, so it counts.

On the committed E-D1 artifact (paper-data/results-h100-2/training/
draft_spec_e_d1.json, 8 prompts x 80 tokens, pooled) tokens_per_target_forward
overstates the split-setting rate by 33% / 51% / 74% at K=2/4/8
(2.397 -> 1.798, 3.316 -> 2.199, 4.238 -> 2.443 tokens per round trip).
Both are kept: the first stays comparable with existing artifacts, the second
is the one to put in a latency model.

Heavy deps (torch / transformers) are imported lazily inside run paths so that
`--help` and `--self-test` work on machines without them.

Target hardware: H100 94GB. Qwen3.6-27B bf16 (~54GB) + Qwen3.5-2B draft fit
comfortably. transformers 5.14 / torch 2.11 on the image.

DRAFT TOKENIZER MUST EQUAL TARGET TOKENIZER (one token-id space both ways).
qwen3-0.6b is NOT compatible with qwen36 targets (re-indexed vocab); the
compatible small drafts are Qwen3.5-2B / Qwen3.5-4B (identical tokenizer.json).
The harness checks this at startup and aborts with a clear error.
"""

import argparse
import json
import os
import time

# Prompts carried over from the split-inference lookahead ablation
# (8 prompts / 4 categories).
PROMPTS = {
    "code": [
        "Write a Python function to implement binary search on a sorted array. Include docstring and type hints.",
        "Write a Python class for a linked list with insert, delete, and search methods.",
    ],
    "structured": [
        "List the top 10 largest countries by area with their capitals and populations in a formatted table.",
        "Explain the HTTP request lifecycle step by step, from DNS resolution to response rendering.",
    ],
    "creative": [
        "Write a short story about a robot that discovers it can dream. Make it emotional and surprising.",
        "Compose a poem about the ocean at midnight. Use vivid imagery and unexpected metaphors.",
    ],
    "conversational": [
        "What are the main differences between Python and Rust? When should I choose one over the other?",
        "Explain quantum computing to a 10-year-old. Use simple analogies they would understand.",
    ],
}

DEFAULT_TARGET = "/workspace/experiments/models/qwen36-27b"
DEFAULT_DRAFT = "/workspace/experiments/models/qwen35-2b"
DEFAULT_K = [2, 4, 8]
DEFAULT_MAX_TOKENS = 80
SEED = 0

QUICK_PROMPTS = [("code", PROMPTS["code"][0]),
                 ("conversational", PROMPTS["conversational"][0])]


# Pure acceptance logic (no torch) — CPU-verifiable via --self-test
def accept_prefix_len(target_preds, draft_tokens):
    """Longest prefix where target greedy prediction equals the drafted token.

    target_preds: target argmax at positions [committed, committed+1, ...]
                  (len == len(draft_tokens) + 1 possible predictions).
    Returns n accepted draft tokens (0..K).
    """
    n = 0
    for pred, d in zip(target_preds, draft_tokens):
        if pred != d:
            break
        n += 1
    return n


def correction_index(target_preds, draft_tokens, n_accepted):
    """Index into target_preds of the correction/bonus token position."""
    return min(n_accepted, len(target_preds) - 1)


def round_trip_metrics(tokens_generated: int, rounds: int,
                       replay_forwards: int) -> dict[str, float]:
    """Per-run target-side cost accounting. Pure python, no torch.

    Two denominators, deliberately both reported:

      target_forward_calls = rounds + 1 (prefill)
          Verification forwards only. Feeds tokens_per_target_forward, which
          measures acceptance quality and nothing else.
      round_trips          = target_forward_calls + replay_forwards
          Every target forward is one WAN round trip in the split deployment,
          and a partial accept costs a second one (the state-rebuild replay).
          Feeds tokens_per_round_trip -- the network-relevant rate.

    tokens_per_round_trip <= tokens_per_target_forward always, with equality
    iff replay_forwards == 0 (every round a full accept).

    NOTE target_forward_calls deliberately keeps its historical meaning
    (verification forwards only, replays excluded) so committed artifacts
    stay comparable; actual target forwards issued are the sum of it and
    target_replay_forwards.
    """
    target_forwards = rounds + 1
    round_trips = target_forwards + replay_forwards
    return {
        "target_forward_calls": target_forwards,
        "target_replay_forwards": replay_forwards,
        "tokens_per_target_forward":
            round(tokens_generated / target_forwards, 3) if rounds else 0,
        "tokens_per_round_trip":
            round(tokens_generated / round_trips, 3) if rounds else 0,
        "partial_accept_rate":
            round(replay_forwards / rounds, 4) if rounds else 0,
    }


def pooled_tokens_per_round_trip(recs: list) -> float:
    """Ratio of sums across runs -- NOT the mean of per-run ratios.

    The right aggregate for a cost rate: it weights every round trip equally
    instead of every prompt equally (short/low-accept prompts otherwise get
    the same vote as long ones). Quote this for network cost; the sibling
    mean_* fields stay for consistency with the pre-existing summary style.
    """
    trips = sum(r["target_forward_calls"] + r["target_replay_forwards"]
                for r in recs)
    toks = sum(r["tokens_generated"] for r in recs)
    return round(toks / trips, 3) if trips else 0


# DESIGN NOTE (documented only -- NOT implemented here; needs a server change)
#
# `commit n`: collapse the partial-accept replay back to one round trip. The
# cloud already holds everything the state rebuild needs (the verification
# block's input activations and the pre-block cache); it lacks only
# n_accepted, which the local side learns from the logits. Piggybacking
# {mb_id, n_accepted} as `commit_prev` on the NEXT round's verify_block
# header makes a partial round cost exactly ONE round trip, same as a full
# accept. Ordering is safe under PROTOCOL.md §7 (responses carry mb_id and
# frames are processed in arrival order).


# Cache snapshot / rollback for hybrid models (torch required)
def _cache_layers(cache):
    """Per-layer entries of a transformers Cache, across API variants."""
    for attr in ("layers", "caches"):
        layers = getattr(cache, attr, None)
        if layers is not None:
            return list(layers)
    return []


def snapshot_linear_states(cache):
    """Clone conv/recurrent state of linear-attention (gated delta-net) layers.

    transformers 5.14 reality (verified on h100-2, cache_utils.py): hybrid
    caches are DynamicCache with per-layer entries; linear layers are
    LinearAttentionLayer objects holding DICTS of tensors keyed by state idx:
      layer.conv_states = {0: tensor}, layer.recurrent_states = {0: tensor}
    (older/simpler variants use singular tensor attrs conv_state /
    recurrent_state -- handled too). cache.crop() on a hybrid cache RAISES
    unless activate_past_recording() was called before the block, and even
    then LinearAttentionLayer.crop(-n) only restores conv_states -- never
    recurrent_states. So we snapshot/restore both manually (Paper 1's
    checkpoint/restore approach, per verification block).
    """
    snap = {}
    for i, layer in enumerate(_cache_layers(cache)):
        st = {}
        for name in ("conv_states", "recurrent_states"):
            val = getattr(layer, name, None)
            if isinstance(val, dict):
                st[name] = {k: (v.clone() if hasattr(v, "clone") else v)
                            for k, v in val.items()}
        for name in ("conv_state", "recurrent_state"):
            val = getattr(layer, name, None)
            if val is not None and hasattr(val, "clone"):
                st[name] = val.clone()
        if st:
            snap[i] = st
    return snap


def rollback_cache(cache, committed_len, linear_snap=None):
    """Roll a cache back to committed_len tokens and restore linear state.

    1. Try cache.crop(committed_len).
    2. Verify resulting length; if crop is missing/unsupported/wrong
       (known crop(0) bug class in some versions), slice KV tensors manually.
    3. Restore snapshotted conv/recurrent state (crop does not touch these).
    """
    did_crop = False
    if hasattr(cache, "crop"):
        try:
            cache.crop(committed_len)
            did_crop = True
        except Exception:
            did_crop = False

    seq_len = None
    if hasattr(cache, "get_seq_length"):
        try:
            seq_len = cache.get_seq_length()
        except Exception:
            seq_len = None

    if not did_crop or (seq_len is not None and seq_len != committed_len):
        # Manual fallback: slice attention KV tensors per layer.
        for layer in _cache_layers(cache):
            for name in ("keys", "values", "key_cache", "value_cache"):
                t = getattr(layer, name, None)
                if t is not None and hasattr(t, "shape") and len(t.shape) == 4:
                    if t.shape[2] > committed_len:
                        setattr(layer, name, t[:, :, :committed_len].contiguous())

    if linear_snap:
        layers = _cache_layers(cache)
        for i, st in linear_snap.items():
            if i >= len(layers):
                continue
            for name, val in st.items():
                if isinstance(val, dict):
                    dst = getattr(layers[i], name, None)
                    if isinstance(dst, dict):
                        for k, v in val.items():
                            dst[k] = v.clone() if hasattr(v, "clone") else v
                else:
                    setattr(layers[i], name, val.clone())


def cache_seq_length(cache):
    if hasattr(cache, "get_seq_length"):
        try:
            return cache.get_seq_length()
        except Exception:
            return None
    return None


# Model loading + generation (torch required)
def load_model(path, torch, AutoModelForCausalLM, AutoTokenizer, device, dtype):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, device_map=device, attn_implementation="eager",
    )
    model.eval()
    return model, tok


def check_tokenizer_compat(tok_t, tok_d):
    """Draft and target MUST share one token-id space.

    Token-level speculation feeds draft-produced ids into the target's
    embedding and vice versa. Qwen3-0.6B (vocab 151936) vs Qwen3.6-27B
    (vocab 248320) re-index nearly every shared token (131331 of 131612
    shared strings map to different ids), so qwen3-0.6b cannot draft for
    qwen36 targets; use a same-tokenizer draft (Qwen3.5-2B/4B share
    qwen36's tokenizer.json byte-for-byte). Mismatched ids either assert
    in F.embedding (id >= vocab) or silently verify garbage.
    """
    vt, vd = tok_t.get_vocab(), tok_d.get_vocab()
    if vt != vd:
        shared = set(vt) & set(vd)
        mismatched = sum(1 for t in shared if vt[t] != vd[t])
        raise SystemExit(
            f"FATAL: draft/target tokenizers are not the same id space "
            f"(target vocab={len(vt)}, draft vocab={len(vd)}, "
            f"{mismatched}/{len(shared)} shared tokens re-indexed). "
            f"Use a draft with the target's tokenizer (e.g. Qwen3.5-2B "
            f"for Qwen3.6 targets)."
        )
    if tok_t.eos_token_id != tok_d.eos_token_id:
        raise SystemExit("FATAL: eos_token_id mismatch between tokenizers")
    print(f"[tokenizer] draft/target vocab identical ({len(vt)} tokens)")


def forward_logits(model, input_ids, cache):
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    return out.logits[0, -1], out.past_key_values


def sequential_baseline(model, tok, prompt, max_new_tokens, device, torch):
    """Plain greedy decode with KV cache. One forward per token."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        logits, cache = forward_logits(model, ids, None)
        gen = []
        t0 = time.time()
        forwards = 1
        for _ in range(max_new_tokens):
            nxt = int(logits.argmax())
            gen.append(nxt)
            if nxt == tok.eos_token_id:
                break
            logits, cache = forward_logits(
                model, torch.tensor([[nxt]], device=device), cache)
            forwards += 1
    dt = time.time() - t0
    return {
        "tokens": gen,
        "tokens_generated": len(gen),
        "forward_calls": forwards,
        "time_s": round(dt, 3),
        "tok_per_sec": round(len(gen) / dt, 2) if dt > 0 else 0,
    }


def speculative_generate(model_t, model_d, tok, prompt, K, max_new_tokens,
                         device, torch):
    """Greedy draft-target speculation.

    Round structure:
      - committed tokens end with correction/bonus token c (not yet in either
        cache; it is the first input of this round for both models).
      - draft: forward c, then greedily generate K tokens d1..dK (K forwards).
      - target: ONE forward over [c, d1..dK] -> K+1 greedy predictions.
      - accept longest prefix; commit accepted + 1 correction token.
      - rollback: recurrent state is a single in-place summary tensor and
        cannot be rewound to a mid-block position, so on PARTIAL accepts we
        roll back the whole block (KV slice + pre-block linear-state restore)
        and re-forward the committed tokens; full accepts need no rollback.
    """
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = ids.shape[1]

    rounds = []           # per-round dicts
    accept_by_pos = [0] * K
    rounds_at_pos = [0] * K  # rounds in which position i was still verifiable
    replay_forwards = 0      # extra forwards rebuilding state on partial accepts
    gen = []

    with torch.no_grad():
        logits_t, cache_t = forward_logits(model_t, ids, None)
        logits_d, cache_d = forward_logits(model_d, ids, None)
        layers_t = _cache_layers(cache_t)
        n_lin = sum(1 for l in layers_t if not hasattr(l, "keys"))
        snap0 = snapshot_linear_states(cache_t)
        print(f"  [cache] {type(cache_t).__name__}: {len(layers_t)} layers, "
              f"{n_lin} linear-attention, snapshot covers {len(snap0)}")
        if n_lin and not snap0:
            raise SystemExit(
                "FATAL: hybrid target cache but linear-state snapshot is "
                "empty -- rollback would corrupt state on partial accepts.")
        cur = int(logits_t.argmax())   # first token, target's own choice
        t0 = time.time()

        while len(gen) < max_new_tokens:
            gen.append(cur)
            if cur == tok.eos_token_id:
                break
            if len(gen) >= max_new_tokens:
                break

            # draft phase: feed cur, then generate K tokens greedily
            # Snapshot draft linear state too (Qwen3.5 drafts may be hybrid).
            snap_d = snapshot_linear_states(cache_d)
            dlogits, cache_d = forward_logits(
                model_d, torch.tensor([[cur]], device=device), cache_d)
            draft_tokens = []
            for _ in range(K):
                d = int(dlogits.argmax())
                draft_tokens.append(d)
                dlogits, cache_d = forward_logits(
                    model_d, torch.tensor([[d]], device=device), cache_d)

            # verify phase: snapshot, ONE target forward over the block
            committed_before = prompt_len + len(gen) - 1  # excl. cur (not cached)
            lin_snap = snapshot_linear_states(cache_t)
            block = torch.tensor([[cur] + draft_tokens], device=device)
            out = model_t(input_ids=block, past_key_values=cache_t,
                          use_cache=True)
            cache_t = out.past_key_values
            preds = [int(x) for x in out.logits[0].argmax(dim=-1)]  # K+1 preds

            n_acc = accept_prefix_len(preds, draft_tokens)
            corr = preds[correction_index(preds, draft_tokens, n_acc)]
            n_committed = n_acc + 1

            # Commit accepted draft tokens to the output (correction token
            # becomes cur and is appended at the top of the next round).
            accepted = draft_tokens[:n_acc]
            gen.extend(accepted)

            # EOS inside the accepted span ends generation there; anything
            # drafted after it is invalid.
            if tok.eos_token_id in accepted:
                gen = gen[:gen.index(tok.eos_token_id) + 1]
                break

            # rollback to committed length: recurrent state is a single
            # summary tensor (update_recurrent_state just copy_'s over it) and
            # cannot be rewound mid-block, so on a PARTIAL accept roll back
            # the WHOLE block (KV slice + linear-state restore to pre-block)
            # and re-forward [cur] + accepted, rebuilding KV and linear state
            # at exactly the committed point. Full accept: caches already
            # exactly right — do nothing.
            if n_acc < K:
                rollback_cache(cache_t, committed_before, lin_snap)
                replay = torch.tensor([[cur] + accepted], device=device)
                _, cache_t = forward_logits(model_t, replay, cache_t)
                replay_forwards += 1
                # draft cache: same rebuild (Qwen3.5 drafts are hybrid too);
                # correction token stays uncached, fed at next round start.
                rollback_cache(cache_d, committed_before, snap_d)
                _, cache_d = forward_logits(model_d, replay, cache_d)

            for i in range(min(K, len(draft_tokens))):
                rounds_at_pos[i] += 1
            for i in range(n_acc):
                accept_by_pos[i] += 1
            rounds.append({
                "drafted": len(draft_tokens),
                "accepted": n_acc,
                "committed": n_committed,
            })

            cur = corr

    dt = time.time() - t0
    gen = gen[:max_new_tokens]
    total_committed = sum(r["committed"] for r in rounds)
    total_rounds = len(rounds)
    return {
        "tokens": gen,
        "tokens_generated": len(gen),
        "rounds": total_rounds,
        # target_forward_calls / target_replay_forwards / tokens_per_target_
        # forward / tokens_per_round_trip / partial_accept_rate
        **round_trip_metrics(len(gen), total_rounds, replay_forwards),
        # Undercounts by replay_forwards: the partial-accept rebuild above
        # also re-forwards the draft cache. The draft runs LOCALLY in the
        # split deployment, so that costs compute, not a round trip; the
        # field keeps its historical definition for artifact comparability.
        "draft_forward_calls": sum(r["drafted"] for r in rounds) + total_rounds + 1,
        "total_accepted": total_committed - total_rounds,
        "total_committed": total_committed,
        "accept_rate_per_round": round(
            sum(r["accepted"] for r in rounds) /
            max(1, sum(r["drafted"] for r in rounds)), 4),
        "accept_by_position": [
            round(accept_by_pos[i] / rounds_at_pos[i], 4) if rounds_at_pos[i] else None
            for i in range(K)
        ],
        "time_s": round(dt, 3),
        "tok_per_sec": round(len(gen) / dt, 2) if dt > 0 else 0,
    }


# Experiment driver
def run_experiment(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bf16 on GPU: matches the qwen36 checkpoint dtype (config: bfloat16)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[load] target: {args.target}")
    model_t, tok = load_model(args.target, torch, AutoModelForCausalLM,
                              AutoTokenizer, device, dtype)
    print(f"[load] draft: {args.draft}")
    model_d, tok_d = load_model(args.draft, torch, AutoModelForCausalLM,
                                AutoTokenizer, device, dtype)
    check_tokenizer_compat(tok, tok_d)

    if args.quick:
        prompt_list = QUICK_PROMPTS
        k_values = [4]
        max_tokens = 32
    else:
        prompt_list = [(cat, p) for cat, ps in PROMPTS.items() for p in ps]
        k_values = args.k
        max_tokens = args.max_tokens

    results = {
        "experiment": "E-D1 draft-model speculation acceptance",
        "config": {
            "target": args.target,
            "draft": args.draft,
            "k_values": k_values,
            "max_tokens": max_tokens,
            "seed": SEED,
            "device": device,
            "dtype": str(dtype),
            "quick": args.quick,
        },
        "sequential_baseline": [],
        "speculation": [],
        "summary_by_k": {},
        "token_identity_check": [],
    }

    for category, prompt in prompt_list:
        print(f"\n[baseline] {category}: {prompt[:60]}...")
        base = sequential_baseline(model_t, tok, prompt, max_tokens, device, torch)
        base_rec = {k: v for k, v in base.items() if k != "tokens"}
        base_rec.update({"category": category,
                         "prompt_preview": prompt[:60] + "..."})
        results["sequential_baseline"].append(base_rec)
        print(f"  -> {base['tok_per_sec']} tok/s, {base['tokens_generated']} tokens")

        for K in k_values:
            print(f"[spec K={K}] {category}: {prompt[:60]}...")
            spec = speculative_generate(model_t, model_d, tok, prompt, K,
                                        max_tokens, device, torch)
            identical = spec["tokens"][:len(base["tokens"])] == base["tokens"] and \
                len(spec["tokens"]) == len(base["tokens"])
            # bf16 chunked-vs-recurrent numerics can flip greedy argmax at
            # near-tie positions (observed logit noise up to ~0.7); record
            # the identical prefix so a single tie-break doesn't read as
            # state corruption.
            prefix_len = next(
                (i for i, (a, b) in enumerate(zip(base["tokens"], spec["tokens"]))
                 if a != b), min(len(base["tokens"]), len(spec["tokens"])))
            results["token_identity_check"].append({
                "category": category, "k": K,
                "prompt_preview": prompt[:60] + "...",
                "identical_to_baseline": identical,
                "identical_prefix_len": prefix_len,
            })
            if not identical:
                print("  !! WARNING: speculative output != baseline (greedy "
                      "correctness violated)")
            spec_rec = {k: v for k, v in spec.items() if k != "tokens"}
            spec_rec.update({"category": category, "k": K,
                             "prompt_preview": prompt[:60] + "...",
                             "identical_to_baseline": identical})
            results["speculation"].append(spec_rec)
            print(f"  -> {spec['tokens_per_target_forward']} tok/target-fwd, "
                  f"{spec['tokens_per_round_trip']} tok/round-trip "
                  f"({spec['target_replay_forwards']} replays, "
                  f"partial={spec['partial_accept_rate']}), "
                  f"accept/round={spec['accept_rate_per_round']}, "
                  f"{spec['tok_per_sec']} tok/s")

    for K in k_values:
        recs = [r for r in results["speculation"] if r["k"] == K]
        if not recs:
            continue
        results["summary_by_k"][f"k{K}"] = {
            "mean_tokens_per_target_forward": round(
                sum(r["tokens_per_target_forward"] for r in recs) / len(recs), 3),
            "mean_tokens_per_round_trip": round(
                sum(r["tokens_per_round_trip"] for r in recs) / len(recs), 3),
            # ratio of sums, not mean of ratios -- the figure to quote for
            # network cost (see pooled_tokens_per_round_trip)
            "pooled_tokens_per_round_trip": pooled_tokens_per_round_trip(recs),
            "mean_accept_rate_per_round": round(
                sum(r["accept_rate_per_round"] for r in recs) / len(recs), 4),
            "mean_partial_accept_rate": round(
                sum(r["partial_accept_rate"] for r in recs) / len(recs), 4),
            "mean_tok_per_sec": round(
                sum(r["tok_per_sec"] for r in recs) / len(recs), 2),
            "total_accepted": sum(r["total_accepted"] for r in recs),
            "total_rounds": sum(r["rounds"] for r in recs),
            "total_replay_forwards": sum(
                r["target_replay_forwards"] for r in recs),
            "all_identical_to_baseline": all(
                r["identical_to_baseline"] for r in recs),
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {args.output}")
    print(json.dumps(results["summary_by_k"], indent=2))


# Self-test: pure-python, no torch/transformers required
class _FakeTensor:
    """Minimal tensor stand-in for cache-rollback logic checks (list-backed)."""
    def __init__(self, data):
        self.data = list(data)
        self.shape = (1, 1, len(self.data), 1)

    def clone(self):
        return _FakeTensor(self.data)

    def contiguous(self):
        return self

    def __getitem__(self, idx):
        # only supports t[:, :, :n] as produced by rollback_cache fallback
        return _FakeTensor(self.data[: idx[2].stop])


class _FakeLayer:
    def __init__(self, kv_len, linear=False):
        self.keys = _FakeTensor(range(kv_len))
        if linear:
            # mirror transformers 5.14 LinearAttentionLayer: dicts keyed by
            # state idx, plus legacy singular tensor attrs on older variants
            self.conv_states = {0: _FakeTensor([1.0] * 4)}
            self.recurrent_states = {0: _FakeTensor([2.0] * 4)}
            self.conv_state = _FakeTensor([1.0] * 4)
            self.recurrent_state = _FakeTensor([2.0] * 4)


class _FakeCache:
    def __init__(self, layers):
        self.layers = layers

    def crop(self, n):  # crop that only touches attention KV (transformers-style)
        for layer in self.layers:
            if hasattr(layer, "keys"):
                layer.keys = _FakeTensor(layer.keys.data[:n])

    def get_seq_length(self):
        return len(self.layers[0].keys.data)


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    print("accept_prefix_len:")
    check("all accepted", accept_prefix_len([5, 6, 7, 8], [5, 6, 7]) == 3)
    check("none accepted", accept_prefix_len([9, 6, 7], [5, 6, 7]) == 0)
    check("partial", accept_prefix_len([5, 6, 0, 0], [5, 6, 7]) == 2)
    check("correction idx", correction_index([5, 6, 0], [5, 6, 7], 2) == 2)

    print("cache rollback (hybrid: attention + linear layers):")
    cache = _FakeCache([_FakeLayer(10), _FakeLayer(10, linear=True)])
    snap = snapshot_linear_states(cache)
    check("snapshots only linear layer", list(snap.keys()) == [1])
    # simulate verification block: KV grows by 3, linear state mutates
    for layer in cache.layers:
        if hasattr(layer, "keys"):
            layer.keys = _FakeTensor(layer.keys.data + [99, 98, 97])
    cache.layers[1].recurrent_state = _FakeTensor([7.7] * 4)
    cache.layers[1].conv_state = _FakeTensor([8.8] * 4)
    cache.layers[1].recurrent_states[0] = _FakeTensor([7.7] * 4)
    cache.layers[1].conv_states[0] = _FakeTensor([8.8] * 4)
    rollback_cache(cache, 11, snap)
    check("KV cropped to committed len",
          cache.get_seq_length() == 11 and
          len(cache.layers[1].keys.data) == 11)
    check("recurrent state restored (legacy attr)",
          cache.layers[1].recurrent_state.data == [2.0] * 4)
    check("conv state restored (legacy attr)",
          cache.layers[1].conv_state.data == [1.0] * 4)
    check("recurrent_states dict restored (5.14 form)",
          cache.layers[1].recurrent_states[0].data == [2.0] * 4)
    check("conv_states dict restored (5.14 form)",
          cache.layers[1].conv_states[0].data == [1.0] * 4)

    print("cache rollback fallback (crop broken/no-op):")
    class BrokenCropCache(_FakeCache):
        def crop(self, n):
            pass  # no-op, like linear layers in Paper 1 / crop(0) bug
    cache2 = BrokenCropCache([_FakeLayer(10)])
    for layer in cache2.layers:
        layer.keys = _FakeTensor(layer.keys.data + [1, 2, 3])
    rollback_cache(cache2, 11, {})
    check("manual slice recovered length", cache2.get_seq_length() == 11)

    print("round-trip accounting (partial accepts cost a 2nd round trip):")
    # Real E-D1 record (k=2, code prompt): 80 tokens, 28 rounds, 2 replays.
    # The committed artifact reports tokens_per_target_forward == 2.759.
    m = round_trip_metrics(tokens_generated=80, rounds=28, replay_forwards=2)
    check("target_forward_calls = rounds + prefill",
          m["target_forward_calls"] == 29)
    check("tok/target-fwd = 80/29 = 2.759 (matches committed artifact)",
          m["tokens_per_target_forward"] == 2.759)
    check("tok/round-trip = 80/(29+2) = 2.581",
          m["tokens_per_round_trip"] == 2.581)
    check("round-trip rate is STRICTLY lower when replays > 0",
          m["tokens_per_round_trip"] < m["tokens_per_target_forward"])
    check("partial_accept_rate = 2/28 = 0.0714",
          m["partial_accept_rate"] == 0.0714)

    full = round_trip_metrics(tokens_generated=80, rounds=28, replay_forwards=0)
    check("zero replays -> the two rates are equal",
          full["tokens_per_round_trip"] == full["tokens_per_target_forward"]
          == 2.759)
    check("zero replays -> partial_accept_rate 0",
          full["partial_accept_rate"] == 0)
    empty = round_trip_metrics(tokens_generated=1, rounds=0, replay_forwards=0)
    check("zero rounds guarded (no ZeroDivisionError)",
          empty["tokens_per_round_trip"] == 0 and
          empty["partial_accept_rate"] == 0)

    # Aggregation: ratio of sums, not mean of ratios. E-D1 k=2 totals over the
    # 8 prompts are 640 tokens / 267 verification forwards / 89 replays.
    ed1_k2 = [{"tokens_generated": 640, "target_forward_calls": 267,
               "target_replay_forwards": 89}]
    check("pooled tok/round-trip on E-D1 k=2 totals = 640/356 = 1.798",
          pooled_tokens_per_round_trip(ed1_k2) == 1.798)
    skewed = [{"tokens_generated": 80, "target_forward_calls": 10,
               "target_replay_forwards": 0},
              {"tokens_generated": 80, "target_forward_calls": 80,
               "target_replay_forwards": 40}]
    mean_of_ratios = round(sum(
        r["tokens_generated"] /
        (r["target_forward_calls"] + r["target_replay_forwards"])
        for r in skewed) / len(skewed), 3)
    check("pooled = 160/130 = 1.231, differs from mean-of-ratios 4.333",
          pooled_tokens_per_round_trip(skewed) == 1.231 and
          mean_of_ratios == 4.333)
    check("pooled guards empty input", pooled_tokens_per_round_trip([]) == 0)

    print("prompt set:")
    flat = [(c, p) for c, ps in PROMPTS.items() for p in ps]
    check("8 prompts / 4 categories", len(flat) == 8 and len(PROMPTS) == 4)
    check("quick subset valid", len(QUICK_PROMPTS) == 2)

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(
        description="E-D1: draft-model speculative decoding acceptance harness "
                    "(single-node, in-process; greedy; hybrid-cache safe)")
    parser.add_argument("--target", default=DEFAULT_TARGET,
                        help="target model path (Qwen3.6-27B / 35B-A3B)")
    parser.add_argument("--draft", default=DEFAULT_DRAFT,
                        help="draft model path (e.g. Qwen3-0.6B)")
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_K,
                        help="draft lengths to sweep (default: 2 4 8)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--output",
                        default="experiment_data/draft_spec_e_d1.json")
    parser.add_argument("--quick", action="store_true",
                        help="2 prompts, K=4, 32 tokens (smoke test)")
    parser.add_argument("--self-test", action="store_true",
                        help="pure-python logic checks; no torch needed")
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(self_test())

    run_experiment(args)


if __name__ == "__main__":
    main()
