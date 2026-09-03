#!/usr/bin/env python3
"""E-R1a/E-R3/E-R4/E-R5 — SECRET LIFETIME of the rotated boundary defense.

The E8 defense (e8_obfuscation.py) hides the split boundary behind a secret
orthogonal W. The committed breaks are all ACCUMULATION attacks: oracle pairs
(K50 = 258/356/618 labeled pairs at depths 1/4/8, H=5120, bf16, 27B —
e8_robustness_27b_bf16_densek_20260807T230529Z.json), known fixed prefixes
(e8_known_prefix.py), partial row leaks. Every one of them needs pairs
collected under the SAME W. This suite measures how per-epoch rotation
(fresh W per epoch, ratchet-derived) shrinks the usable accumulation window:

  E-R1a  per-epoch accumulation budget: E labeled (h, hW_t) pairs from ONE
         epoch -> polar(lstsq) solve -> victim-token recovery. Plus the
         CROSS-EPOCH control: pairs pooled across 4 differently-keyed epochs
         must fail (recovery ~= label-free band), proving accumulation does
         not cross rotation boundaries. Reports max_safe_epoch vs the
         static-W K50 prior cited above.
  E-R3   scaffold randomization vs known-prefix: static W, sessions of an
         S=64-token prefix. Arm (a) fixed_scaffold = the committed break;
         arm (b) jittered_scaffold = per-session pad insertion + template-slot
         permutation so the attacker's assumed alignment is wrong and it
         harvests ~0 labelable pairs/session.
  E-R4   ratcheted keys / stale-key attack: attacker is HANDED W_1..W_3 plus
         the derivation rule (not master_seed) and attacks epoch 4 with
         (i) zero fresh pairs and (ii) E fresh pairs. Stale keys must give
         no bootstrap: (i) sits in the label-free band with W-alignment
         rel_err ~= sqrt(2) (random), (ii) matches the E-R1a curve.
  E-R5   sharded secrets: block-diagonal W with s blocks. NEGATIVE CONTROL:
         static sharding WEAKENS per-block budgets (0.1*h_b pairs suffice per
         block). The mechanism: staggered rotation — block i rotates when
         (epoch + i*(period//s)) % period == 0 — so no 2-epoch sliding
         window has all blocks on the same key and the composed solve fails.

Key schedule (E-R4/E-R5): seed_t = int.from_bytes(sha256(f"{master}:{t}")
.digest()[:8], "little"); W_t = make_secret(H, seed_t). The ratchet is
one-way: stale keys + the rule do not yield future keys without master_seed.

Usage:
    python rotation_lifetime.py --help        # works without torch
    python rotation_lifetime.py --self-test   # torch-free fixtures
    python rotation_lifetime.py --toy --quick # CPU machinery check
    python rotation_lifetime.py --experiment all --model <hf-model> --corpus-file <docs.txt>
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

# Guarded heavy imports: `--help`/`--self-test` must work on torch-less hosts.
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from split_trainer import (TEXT_SAMPLES, _write_training_status,
                               build_modules, make_layer_kwargs,
                               run_layer_stack)
    from trained_inversion import (collect_base_pairs, evaluate_decoder,
                                   make_provenance, mean_std, seed_all,
                                   split_at, train_decoder)
    from e8_obfuscation import make_secret
    from e8_robustness import boundary_acts, crossing_k, polar
except ImportError:  # pragma: no cover - torch-less host
    TEXT_SAMPLES = []
    _write_training_status = lambda **k: None
    build_modules = make_layer_kwargs = run_layer_stack = None
    collect_base_pairs = evaluate_decoder = mean_std = seed_all = None
    split_at = train_decoder = make_secret = make_provenance = None
    boundary_acts = crossing_k = polar = None

K50_THRESHOLD_PCT = 50.0
RANDOM_REL_ERR = math.sqrt(2.0)  # E||A-B||_F/||B||_F for independent orthogonals
K50_PRIOR = {1: 258, 4: 356, 8: 618}  # committed 27B dense-grid static-W K50
K50_PRIOR_ARTIFACT = "e8_robustness_27b_bf16_densek_20260807T230529Z.json"


# Pure-python helpers (torch-free; --self-test pins them with frozen fixtures)
def ratchet_seed(master_seed, epoch):
    """One-way per-epoch key derivation (see module docstring)."""
    return int.from_bytes(hashlib.sha256(
        f"{master_seed}:{epoch}".encode()).digest()[:8], "little")


def stagger_rotating_blocks(epoch, n_blocks, period):
    """Blocks rotating at `epoch` under the staggered schedule: block i
    rotates when (epoch + i*(period//n_blocks)) % period == 0."""
    step = period // n_blocks
    return [i for i in range(n_blocks) if (epoch + i * step) % period == 0]


def partition_pairs(n_pairs, n_epochs):
    """Split n_pairs as evenly as possible across n_epochs (first r get +1)."""
    base, rem = divmod(n_pairs, n_epochs)
    return [base + (1 if e < rem else 0) for e in range(n_epochs)]


def _append_jsonl(args, record):
    """Crash-safe per-cell result journal: appended immediately after each
    completed cell so a late hard crash (GB10 silent-kill phenomenon) cannot
    take completed cells down with the process."""
    if not getattr(args, "output", None):
        return
    with open(args.output + ".jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


# Frozen --self-test fixtures. The ratchet ints are precomputed sha256 chain
# outputs for master 12345, epochs 0..3; they pin the derivation byte-for-byte
# (endianness, truncation, formatting). The stagger fixture is the full
# rotation map for s=4, period=8, epochs 0..15.
FIXTURE_RATCHET_12345 = [10901005920735059415, 12207851219204689068,
                         14158799805508211343, 5173715490049009419]
FIXTURE_STAGGER_S4_P8 = {0: [0], 1: [], 2: [3], 3: [], 4: [2], 5: [],
                         6: [1], 7: [], 8: [0], 9: [], 10: [3], 11: [],
                         12: [2], 13: [], 14: [1], 15: []}
FIXTURE_CROSSING_CURVE = [(64, 12.0), (256, 38.0), (512, 61.0), (4096, 66.0)]


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("ratchet derivation pins the frozen sha256 chain (master 12345):")
    got = [ratchet_seed(12345, t) for t in range(4)]
    check(f"epochs 0..3 == {FIXTURE_RATCHET_12345}", got == FIXTURE_RATCHET_12345)
    check("derivation is one-way-ordered: epoch 1 != f(epoch 0) trivially",
          len(set(got)) == 4)
    check("different master -> different chain",
          ratchet_seed(12346, 0) != got[0])

    print("stagger schedule (s=4, period=8, epochs 0..15):")
    got = {e: stagger_rotating_blocks(e, 4, 8) for e in range(16)}
    check("full 16-epoch map matches the frozen fixture",
          got == FIXTURE_STAGGER_S4_P8)
    check("every block rotates exactly twice in 16 epochs",
          all(sum(i in got[e] for e in range(16)) == 2 for i in range(4)))
    check("no epoch rotates more than one block (staggering property)",
          all(len(got[e]) <= 1 for e in range(16)))
    check("no 2-epoch window sees all 4 blocks rotate",
          all(len(set(got[e]) | set(got[e + 1])) < 4 for e in range(15)))

    print("epoch-pair partitioning arithmetic:")
    check("256 over 4 epochs = [64,64,64,64]",
          partition_pairs(256, 4) == [64, 64, 64, 64])
    check("250 over 4 = [63,63,62,62] (first remainder epochs get +1)",
          partition_pairs(250, 4) == [63, 63, 62, 62])
    check("partition always sums to the request",
          sum(partition_pairs(4096, 4)) == 4096
          and sum(partition_pairs(1, 4)) == 1)
    check("more epochs than pairs yields zeros, not negatives",
          partition_pairs(2, 4) == [1, 1, 0, 0])

    print("crossing interpolation on a frozen synthetic curve:")
    curve = [{"K": k, "top1_mean": t} for k, t in FIXTURE_CROSSING_CURVE]
    cross = crossing_k(curve, threshold=K50_THRESHOLD_PCT)
    check("K50 interpolates linearly between 256 and 512",
          cross["k50_method"] == "linear_in_K"
          and cross["k50_bracket"] == [256, 512])
    # 256 + (50-38)/(61-38) * (512-256) = 389.5652...
    check("K50 = 389.5652 +- 0.01",
          cross["k50_interpolated"] is not None
          and abs(cross["k50_interpolated"] - 389.5652) <= 0.01)
    flat = crossing_k([{"K": k, "top1_mean": 3.0} for k, _, _ in
                       [(64, 0, 0), (4096, 0, 0)]], threshold=K50_THRESHOLD_PCT)
    check("a curve in the label-free band never brackets 50%",
          flat["k50_method"] == "not_bracketed"
          and flat["k50_interpolated"] is None)

    if torch is not None:
        print("solve error-path / polar fallback (torch present):")
        # near-duplicate pair rows -> ill-conditioned lstsq; must not raise
        g = torch.Generator().manual_seed(0)
        w_true = torch.linalg.qr(torch.randn(16, 16, generator=g))[0]
        h = torch.randn(64, 16, generator=g)
        h[1] = h[0] * (1 + 1e-14)
        w_hat, solver = solve_w(h, h @ w_true)
        check("ill-conditioned solve returns a result, not a raise",
              w_hat is not None)
        check("solver tag is recorded", isinstance(solver, str))
        if w_hat is not None:
            ident = w_hat.double().T @ w_hat.double()
            check("W_hat stays ~orthogonal",
                  bool((ident - torch.eye(16, dtype=torch.float64))
                       .abs().max() < 1e-6))
        # eigh polar fallback directly: orthogonality on a generic matrix
        q = _polar_eigh(torch.randn(16, 16, generator=g).double())
        check("eigh polar fallback yields an orthogonal factor",
              bool((q.T @ q - torch.eye(16, dtype=torch.float64))
                   .abs().max() < 1e-8))
        # mismatched pair shapes -> lstsq RuntimeError -> error record
        w_bad, info = solve_w(torch.randn(8, 4), torch.randn(9, 4))
        check("a failing solve returns (None, 'error: ...'), never raises",
              w_bad is None and isinstance(info, str)
              and info.startswith("error:"))

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


# Torch helpers
def ratchet_secret(hidden_dim, master_seed, epoch):
    """Per-epoch W from the ratchet chain (fp32 CPU, like make_secret)."""
    return make_secret(hidden_dim, ratchet_seed(master_seed, epoch))


def block_diag_secret(hidden_dim, n_blocks, seed):
    """Block-diagonal orthogonal W: independent seeded QR per diagonal block."""
    assert hidden_dim % n_blocks == 0
    hb = hidden_dim // n_blocks
    w = torch.zeros(hidden_dim, hidden_dim)
    for i in range(n_blocks):
        w[i * hb:(i + 1) * hb, i * hb:(i + 1) * hb] = make_secret(
            hb, ratchet_seed(seed, i))
    return w


def w_rel_err(a, b):
    """Relative Frobenius error ||a-b||/||b||; ~= sqrt(2) for unrelated Ws."""
    return round(((a - b).norm() / b.norm()).item(), 6)


def _polar_eigh(m, eps=1e-12):
    """Polar factor via eigendecomposition of m^T m: polar = m (m^T m)^{-1/2}.
    Fallback for LAPACK gesdd non-convergence in torch.linalg.svd; eigh uses
    syevd, an independent LAPACK code path. Tiny eigenvalues are floored."""
    evals, evecs = torch.linalg.eigh(m.double().T @ m.double())
    evals = evals.clamp_min(eps)
    return m.double() @ (evecs * evals.rsqrt()) @ evecs.T


def solve_w(h_pairs, hw_pairs):
    """W_hat = polar factor of the least-squares solution (e8_robustness
    attack 3), returned as (w_hat, solver_tag).

    NEVER raises: a linalg failure (torch._C._LinAlgError — a RuntimeError
    subclass — or a plain linalg RuntimeError) returns
    (None, "error: <msg>") so the caller journals the cell as failed and
    continues to the next one. One ill-conditioned cell must not kill a
    multi-hour run (an er3 fixed_scaffold session died on "linalg.svd:
    failed to converge"). e8_robustness.polar itself
    is not modified here, so its other callers are unaffected."""
    try:
        # Materialize BOTH operands as contiguous fp64 copies before the
        # solve: .double() is a NO-OP on an already-fp64 input, so a caller
        # passing a strided column slice (er5's block_solve feeds
        # pool[:, block_slice] views of the big pair matrices) would hand
        # LAPACK a non-contiguous view into a large live allocation — the
        # same solve-side memory access the GB10 memory-stomp guards
        # materialize away elsewhere in this file.
        sol = torch.linalg.lstsq(h_pairs.double().contiguous(),
                                 hw_pairs.double().contiguous()).solution
        try:
            return polar(sol), "lstsq+svd"
        except RuntimeError:
            # gesdd failed to converge on an ill-conditioned pair matrix;
            # retry just the polar via eigh of sol^T sol. Chosen over
            # propagating so the run survives; fallback cells are tagged
            # (solver_tag) and logged for downstream filtering.
            print("[solve_w] polar SVD failed; retrying polar via eigh "
                  "of sol^T sol")
            return _polar_eigh(sol), "lstsq+eigh_polar_fallback"
    except RuntimeError as e:
        return None, "error: " + str(e).splitlines()[0]


def recovery_with_what(decoder, h_wire, w_hat, victim_tok, device):
    """Decode wire features de-obfuscated with W_hat; top-1 %."""
    h_rec = (h_wire.double() @ w_hat.double().T).float()
    return evaluate_decoder(decoder, h_rec, victim_tok, device)[0]


def jitter_prefix(ids, pad_id, seed):
    """Per-session scaffold randomization: insert k in {1..8} random pad
    tokens at random positions, then permute 4 template slots. The attacker
    sees the WIRE tensors of this jittered prefix but computes its h on the
    un-jittered one, so its assumed alignment is wrong."""
    g = torch.Generator().manual_seed(seed)
    ids = list(ids)
    k = int(torch.randint(1, 9, (1,), generator=g).item())  # 1..8 pads
    positions = torch.randperm(len(ids) + k, generator=g)[:k].tolist()
    for pos in sorted(positions):
        ids.insert(min(pos, len(ids)), pad_id)
    n = len(ids)
    q = max(1, n // 4)
    slots = [ids[i * q:(i + 1) * q] for i in range(3)] + [ids[3 * q:]]
    order = torch.randperm(4, generator=g).tolist()
    out = [t for i in order for t in slots[i]]
    return out


# Experiment context: model + corpus + per-(depth,seed) decoders, built once.
class Ctx:
    def __init__(self, args):
        embed, layers, norm, lm_head, rotary, encode = build_modules(args)
        self.embed, self.layers, self.norm = embed, layers, norm
        self.lm_head, self.rotary, self.encode = lm_head, rotary, encode
        self.n_layers = len(layers)
        self.vocab_size = (lm_head.weight.shape[0] if not args.toy
                           else embed.weight.shape[0])
        self.hidden_dim = embed.weight.shape[1]
        self.random_top1 = round(100.0 / self.vocab_size, 4)
        print(f"[model] {'toy' if args.toy else args.model}: {self.n_layers} "
              f"layers, hidden={self.hidden_dim}, vocab={self.vocab_size}, "
              f"device={args.device}")

        if args.corpus_file:
            corpus_source = "corpus_file"
            with open(args.corpus_file) as f:
                docs = [l.strip() for l in f if len(l.strip()) > 500]  # real docs only — wikitext short lines are formatting artifacts
        else:
            corpus_source = "TEXT_SAMPLES"
            docs = list(TEXT_SAMPLES)
        if len(docs) < args.victim_docs + 4:
            raise ValueError(f"corpus too small: {len(docs)} docs")
        self.victim_docs = docs[-args.victim_docs:]
        self.attack_docs = docs[:-args.victim_docs]
        self.provenance = make_provenance(
            args.corpus_file, corpus_source, len(docs),
            range(len(docs) - args.victim_docs, len(docs)),
            model_path=getattr(args, "model", None))
        n_val = max(1, int(round(args.val_frac * len(self.attack_docs))))
        self.val_docs = self.attack_docs[:n_val]
        self.train_docs = self.attack_docs[n_val:]
        print(f"[data] {len(self.train_docs)} attack-train, {n_val} "
              f"attack-val, {len(self.victim_docs)} victim (held out)")

        self.victim_ids, victim_tokens = [], []
        for doc in self.victim_docs:
            b = encode([doc], args.seq_len)
            if b:
                self.victim_ids.append(b[0])
                victim_tokens.append(b[0][:-1])
        if not self.victim_ids:
            raise ValueError("no victim doc long enough to yield a block")
        self.victim_tok = torch.cat(victim_tokens)
        self.per_depth = {}  # depth -> dict of cached tensors/decoders

    def depth_ctx(self, depth, args, need_decoder=True):
        """Collect public/victim boundary activations and train decoders for
        one split depth (cached)."""
        if depth in self.per_depth:
            return self.per_depth[depth]
        head, middle, tail, sa, ra = split_at(self.layers, depth, self.n_layers)
        if sa != depth:
            print(f"[split] depth {depth} clamped to sa={sa} "
                  f"({self.n_layers} layers)")
        t0 = time.time()
        tr_h, _, tr_tok = collect_base_pairs(
            self.embed, head, middle, tail, self.norm, self.lm_head,
            self.rotary, self.encode, self.train_docs, args, with_grad=False)
        va_h, _, va_tok = collect_base_pairs(
            self.embed, head, middle, tail, self.norm, self.lm_head,
            self.rotary, self.encode, self.val_docs, args, with_grad=False)
        h_star = boundary_acts(self.embed, head, self.rotary,
                               self.victim_ids, args)
        decoders = {}
        if need_decoder:
            for seed in args.seeds:
                seed_all(args.seed + seed)
                decoders[seed] = train_decoder(
                    tr_h, tr_tok, va_h, va_tok, tr_h.shape[1],
                    self.vocab_size, args, f"rot_d{sa}_seed{seed}")
        base = {s: evaluate_decoder(decoders[s], h_star, self.victim_tok,
                                    args.device)[0] for s in decoders}
        print(f"[collect] depth={sa}: train={tr_h.shape[0]} "
              f"val={va_h.shape[0]} pairs ({time.time() - t0:.1f}s); "
              f"undefended baseline " +
              ", ".join(f"seed{s}={base[s]:.2f}%" for s in base))
        dc = {"sa": sa, "tr_h": tr_h, "tr_tok": tr_tok, "h_star": h_star,
              "decoders": decoders, "baseline": base}
        self.per_depth[depth] = dc
        return dc

    def band_top1(self, dc, seed, args, w_seed_offset=999999):
        """Label-free band reference: decoder applied to h_star @ W_rand with
        NO solve — the ceiling any failed accumulation attack sits in."""
        w = make_secret(self.hidden_dim, args.seed + w_seed_offset + seed)
        return evaluate_decoder(dc["decoders"][seed],
                                (h_wire(dc["h_star"], w)).float(),
                                self.victim_tok, args.device)[0]


def h_wire(h, w):
    """The wire tensor: h @ W computed in fp64 (mirrors the fp32 fp-exact
    seam of the deployed defense closely enough for rel-err purposes)."""
    return h.double() @ w.double()


# E-R1a — per-epoch accumulation budget + cross-epoch pooling control.
def run_er1(ctx, args, out):
    print("\n=== E-R1a: per-epoch accumulation budget ===")
    n_cross_epochs = 4
    for depth in args.depths:
        dc = ctx.depth_ctx(depth, args)
        sa = dc["sa"]
        cells = []
        for E in args.epoch_sizes:
            per_seed, pooled_seed, band_seed = [], [], []
            for seed in args.seeds:
                master = args.seed + 1000 * seed
                dec = dc["decoders"][seed]
                n_avail = dc["tr_h"].shape[0]
                if E > n_avail:
                    print(f"[er1] depth={sa} E={E} seed={seed}: only "
                          f"{n_avail} pairs available, skipping")
                    continue
                # within one epoch t=0: E labeled pairs under ONE W
                g = torch.Generator().manual_seed(master)
                order = torch.randperm(n_avail, generator=g)
                # Materialize EVERY index-derived gather NOW, before any fp64
                # lstsq/SVD solve: on the GB10 the large CPU fp64 solve was
                # observed to scribble fp64 bit patterns over live int64
                # index storage (27B E=4096 crash: index 4586158996040624336
                # = fp64 0.0416...). If no index tensor is alive across a
                # solve, there is nothing to corrupt.
                parts = partition_pairs(E, n_cross_epochs)
                idx_pool, off = [], E  # disjoint from the within-epoch slice
                for n_t in parts:
                    idx_pool.append(order[off:off + n_t].clone())
                    off += n_t
                w0 = ratchet_secret(ctx.hidden_dim, master, 0)
                idx = order[:E]
                h_in = dc["tr_h"][idx]
                hw_in = h_wire(dc["tr_h"][idx], w0)
                # cross-epoch control: E/4 pairs from each of 4 epochs,
                # pooled and solved as if one W
                hp, hwp = [], []
                for t, n_t in enumerate(parts):
                    w_t = ratchet_secret(ctx.hidden_dim, master, t)
                    idx_t = idx_pool[t]
                    assert int(idx_t.min()) >= 0 and \
                        int(idx_t.max()) < n_avail, \
                        "corrupted index slice (memory stomp during solve?)"
                    hp.append(dc["tr_h"][idx_t])
                    hwp.append(h_wire(dc["tr_h"][idx_t], w_t))
                h_pool = torch.cat(hp)
                hw_pool = torch.cat(hwp)
                w_hat, sv_within = solve_w(h_in, hw_in)
                w_pool, sv_pool = solve_w(h_pool, hw_pool)
                if w_hat is None or w_pool is None:
                    err = sv_within if w_hat is None else sv_pool
                    print(f"[er1] depth={sa} E={E} seed={seed}: solve "
                          f"failed ({err}); cell recorded as error")
                    _append_jsonl(args, {"experiment": "er1", "depth": sa,
                                         "epoch_size": E, "seed": seed,
                                         "error": err,
                                         "within_epoch_top1": None,
                                         "pooled_cross_epoch_top1": None,
                                         "labelfree_band_top1": None})
                    continue
                rec = recovery_with_what(dec, h_wire(dc["h_star"], w0),
                                         w_hat, ctx.victim_tok, args.device)
                rec_pool = recovery_with_what(dec, h_wire(dc["h_star"], w0),
                                              w_pool, ctx.victim_tok,
                                              args.device)
                band = ctx.band_top1(dc, seed, args)
                per_seed.append(rec)
                pooled_seed.append(rec_pool)
                band_seed.append(band)
                print(f"[er1] depth={sa} E={E} seed={seed}: "
                      f"within-epoch={rec:.2f}% pooled-4-epoch={rec_pool:.2f}% "
                      f"band={band:.2f}%")
                _append_jsonl(args, {"experiment": "er1", "depth": sa,
                                     "epoch_size": E, "seed": seed,
                                     "within_epoch_top1": rec,
                                     "pooled_cross_epoch_top1": rec_pool,
                                     "labelfree_band_top1": band})
                _write_training_status(state="running", phase="er1",
                                       depth=sa, epoch_size=E, seed=seed,
                                       top1=rec)
            if not per_seed:
                continue
            m, s = mean_std(per_seed)
            mp, sp = mean_std(pooled_seed)
            mb, sb = mean_std(band_seed)
            cells.append({"depth": sa, "epoch_size": E,
                          "recovery_mean": m, "recovery_std": s,
                          "pooled_cross_epoch_mean": mp,
                          "pooled_cross_epoch_std": sp,
                          "labelfree_band_mean": mb, "labelfree_band_std": sb,
                          "n_seeds": len(per_seed)})
            out["results"].append({"experiment": "er1", **cells[-1]})
        curve = [{"K": c["epoch_size"], "top1_mean": c["recovery_mean"]}
                 for c in cells]
        cross = crossing_k(curve, threshold=K50_THRESHOLD_PCT)
        safe = [c["epoch_size"] for c in cells
                if c["recovery_mean"] < K50_THRESHOLD_PCT]
        summary = {"experiment": "er1", "depth": sa,
                   "max_safe_epoch": max(safe) if safe else None,
                   "k50_crossing": cross,
                   "k50_prior_static_w": K50_PRIOR.get(sa),
                   "k50_prior_artifact": K50_PRIOR_ARTIFACT,
                   "cells": cells}
        out["summary"].append(summary)
        print(f"[er1] depth={sa}: max_safe_epoch={summary['max_safe_epoch']} "
              f"(static-W K50 prior {K50_PRIOR.get(sa)}), "
              f"measured crossing {cross['k50_method']}")


# E-R3 — scaffold randomization vs known-prefix accumulation (static W).
def run_er3(ctx, args, out):
    print("\n=== E-R3: scaffold randomization vs known-prefix ===")
    s_tok = args.prefix_len
    for depth in args.depths:
        dc = ctx.depth_ctx(depth, args)
        sa = dc["sa"]
        head = split_at(ctx.layers, depth, ctx.n_layers)[0]
        # session prefixes: first S tokens of public docs, cycling
        prefix_ids = []
        for doc in ctx.train_docs:
            b = ctx.encode([doc], max(args.seq_len, s_tok))
            if b and b[0].shape[0] >= s_tok:
                prefix_ids.append(b[0][:s_tok])
            if len(prefix_ids) >= args.sessions:
                break
        if len(prefix_ids) < args.sessions:
            print(f"[er3] depth={sa}: only {len(prefix_ids)} usable session "
                  f"prefixes (< {args.sessions}); skipping")
            continue

        def head_acts(ids):
            with torch.no_grad():
                x = ids.unsqueeze(0).to(args.device)
                pos = torch.arange(x.shape[1], device=args.device).unsqueeze(0)
                hidden = ctx.embed(x)
                lk = make_layer_kwargs(ctx.rotary, hidden, pos, args)
                return run_layer_stack(head, hidden, lk)[0].float().cpu()

        # Precompute per-session tensors ONCE per depth: the TRUE (possibly
        # jittered) prefix activations — what the wire carries — and the
        # attacker's ASSUMED activations. Wire side per seed is h_true @ W.
        sess = []
        for si in range(args.sessions):
            assumed = prefix_ids[si % len(prefix_ids)]
            jit = jitter_prefix(assumed, pad_id=0, seed=args.seed + 31337 + si)
            true_ids = torch.tensor(jit[:s_tok], dtype=torch.long)
            if true_ids.shape[0] < s_tok:
                true_ids = torch.cat([true_ids, torch.zeros(
                    s_tok - true_ids.shape[0], dtype=torch.long)])
            sess.append({"fixed_true": head_acts(assumed),
                         "fixed_assumed": head_acts(assumed),
                         "jit_true": head_acts(true_ids),
                         "jit_assumed": head_acts(assumed)})
        print(f"[er3] depth={sa}: {args.sessions} sessions x S={s_tok} "
              f"prefix activations cached")

        for arm in ("fixed_scaffold", "jittered_scaffold"):
            key = "fixed" if arm == "fixed_scaffold" else "jit"
            h_assumed_all = torch.cat([s[f"{key}_assumed"] for s in sess])
            h_true_all = torch.cat([s[f"{key}_true"] for s in sess])
            per_seed_curves = []
            for seed in args.seeds:
                w = make_secret(ctx.hidden_dim, args.seed + seed)
                dec = dc["decoders"][seed]
                baseline = dc["baseline"][seed]
                thr = 0.9 * baseline
                hw_all = h_wire(h_true_all, w)
                h_vic_wire = h_wire(dc["h_star"], w)
                curve = []
                for n_sess in range(1, args.sessions + 1):
                    hh = h_assumed_all[:n_sess * s_tok]
                    hw = hw_all[:n_sess * s_tok]
                    recs, solve_err = [], None
                    for solve_seed in args.solve_seeds:
                        g = torch.Generator().manual_seed(
                            args.seed + 500000 + solve_seed)
                        order = torch.randperm(hh.shape[0], generator=g)
                        w_hat, solver = solve_w(hh[order], hw[order])
                        if w_hat is None:
                            solve_err = solver  # "error: ..." from solve_w
                            continue
                        recs.append(recovery_with_what(
                            dec, h_vic_wire, w_hat, ctx.victim_tok,
                            args.device))
                    if not recs:
                        # Every solve in the cell failed (e.g. LAPACK SVD
                        # non-convergence): journal the failed cell with an
                        # error field and continue — do not kill the run.
                        print(f"[er3] depth={sa} arm={arm} seed={seed} "
                              f"sessions={n_sess} (K={n_sess * s_tok}): "
                              f"SOLVE FAILED ({solve_err}); cell recorded "
                              f"as error")
                        _append_jsonl(args, {"experiment": "er3", "depth": sa,
                                             "arm": arm, "seed": seed,
                                             "sessions": n_sess,
                                             "K": n_sess * s_tok,
                                             "error": solve_err,
                                             "top1_mean": None})
                        continue
                    m, _ = mean_std(recs)
                    curve.append({"K": n_sess * s_tok, "top1_mean": m})
                    print(f"[er3] depth={sa} arm={arm} seed={seed} "
                          f"sessions={n_sess} (K={n_sess * s_tok}): "
                          f"top-1={m:.2f}%")
                    _append_jsonl(args, {"experiment": "er3", "depth": sa,
                                         "arm": arm, "seed": seed,
                                         "sessions": n_sess,
                                         "K": n_sess * s_tok,
                                         "top1_mean": m})
                cross = crossing_k(curve, threshold=thr)
                k_cross = cross["k50_interpolated"]
                per_seed_curves.append({
                    "seed": seed, "baseline_top1": baseline,
                    "k90_threshold": round(thr, 4),
                    "sessions_to_k90": (math.ceil(k_cross / s_tok)
                                        if k_cross is not None else None),
                    "crossing": cross, "curve": curve})
            sess_list = [c["sessions_to_k90"] for c in per_seed_curves
                         if c["sessions_to_k90"] is not None]
            out["summary"].append({
                "experiment": "er3", "depth": sa, "arm": arm,
                "prefix_len": s_tok, "n_sessions": args.sessions,
                "labelable_pairs_per_session": (
                    s_tok if arm == "fixed_scaffold" else 0),
                "sessions_to_k90_mean": (
                    round(sum(sess_list) / len(sess_list), 4)
                    if sess_list else None),
                "sessions_to_k90_per_seed": per_seed_curves,
                "final_top1_mean": round(
                    sum(c["curve"][-1]["top1_mean"]
                        for c in per_seed_curves) / len(per_seed_curves), 4)})
            out["results"].extend(
                {"experiment": "er3", "depth": sa, "arm": arm, **c}
                for c in per_seed_curves)
            print(f"[er3] depth={sa} arm={arm}: sessions-to-K90 "
                  f"{sess_list or 'no crossing within ' + str(args.sessions)}")


# E-R4 — ratcheted keys: the stale-key (insider-with-old-key) attack.
def run_er4(ctx, args, out):
    print("\n=== E-R4: ratcheted keys / stale-key attack ===")
    epochs = [1, 2, 3, 4]
    for depth in args.depths:
        dc = ctx.depth_ctx(depth, args)
        sa = dc["sa"]
        for seed in args.seeds:
            master = args.seed + 7000 + 1000 * seed
            ws = {t: ratchet_secret(ctx.hidden_dim, master, t)
                  for t in epochs}
            dec = dc["decoders"][seed]
            h_w4 = h_wire(dc["h_star"], ws[4])
            # pairwise consecutive-epoch alignment (expected ~= sqrt(2))
            pairwise = {f"{t}-{t + 1}": w_rel_err(ws[t], ws[t + 1])
                        for t in epochs[:-1]}
            # (i) zero fresh pairs: best stale key is the most recent, W_3
            stale = {}
            for t in (1, 2, 3):
                rec = recovery_with_what(dec, h_w4, ws[t], ctx.victim_tok,
                                         args.device)
                stale[f"W_{t}"] = {"top1": rec,
                                   "rel_err_to_W4": w_rel_err(ws[t], ws[4])}
            band = ctx.band_top1(dc, seed, args)
            # (ii) E fresh pairs from epoch 4 (matches the E-R1a cell at E)
            fresh = {}
            n_avail = dc["tr_h"].shape[0]
            g = torch.Generator().manual_seed(master)
            order = torch.randperm(n_avail, generator=g)
            for E in args.stale_epoch_sizes:
                if E > n_avail:
                    continue
                idx = order[:E].clone()
                assert int(idx.min()) >= 0 and int(idx.max()) < n_avail, \
                    "corrupted index slice (memory stomp during solve?)"
                w_hat, solver = solve_w(dc["tr_h"][idx],
                                        h_wire(dc["tr_h"][idx], ws[4]))
                if w_hat is None:
                    fresh[str(E)] = {"top1": None, "rel_err_to_W4": None,
                                     "error": solver}
                    continue
                fresh[str(E)] = {
                    "top1": recovery_with_what(dec, h_w4, w_hat,
                                               ctx.victim_tok, args.device),
                    "rel_err_to_W4": w_rel_err(w_hat, ws[4])}
            entry = {"experiment": "er4", "depth": sa, "seed": seed,
                     "pairwise_consecutive_rel_err": pairwise,
                     "random_rel_err_reference": round(RANDOM_REL_ERR, 4),
                     "stale_only": stale,
                     "stale_only_best_top1": max(v["top1"]
                                                 for v in stale.values()),
                     "labelfree_band_top1": band,
                     "fresh_pairs_epoch4": fresh}
            out["results"].append(entry)
            print(f"[er4] depth={sa} seed={seed}: stale-best="
                  f"{entry['stale_only_best_top1']:.2f}% (band {band:.2f}%), "
                  f"fresh " + ", ".join(
                      f"E={k}:{v['top1']:.2f}%" if v["top1"] is not None
                      else f"E={k}:solve-error" for k, v in fresh.items()))
            _append_jsonl(args, entry)
        rows = [r for r in out["results"]
                if r["experiment"] == "er4" and r["depth"] == sa]
        out["summary"].append({
            "experiment": "er4", "depth": sa,
            "stale_only_best_top1_mean": round(
                sum(r["stale_only_best_top1"] for r in rows) / len(rows), 4),
            "labelfree_band_mean": round(
                sum(r["labelfree_band_top1"] for r in rows) / len(rows), 4),
            "pairwise_rel_err_mean": round(
                sum(v for r in rows
                    for v in r["pairwise_consecutive_rel_err"].values())
                / (3 * len(rows)), 4),
            "fresh_pairs_epoch4": {
                k: (lambda xs: round(sum(xs) / len(xs), 4) if xs else None)(
                    [r["fresh_pairs_epoch4"][k]["top1"] for r in rows
                     if r["fresh_pairs_epoch4"].get(k, {}).get("top1")
                     is not None])
                for k in [str(E) for E in args.stale_epoch_sizes]},
            "n_seeds": len(rows)})


# E-R5 — sharded secrets: static-shard negative control vs staggered rotation.
def run_er5(ctx, args, out):
    print("\n=== E-R5: sharded secrets with staggered rotation ===")
    period = args.rotation_period
    window = 2
    E = args.shard_epoch_pairs
    for depth in args.depths:
        dc = ctx.depth_ctx(depth, args)
        sa = dc["sa"]
        for seed in args.seeds:
            master = args.seed + 11000 + 1000 * seed
            dec = dc["decoders"][seed]
            for s in args.shards:
                if ctx.hidden_dim % s:
                    print(f"[er5] hidden={ctx.hidden_dim} not divisible by "
                          f"s={s}, skipping")
                    continue
                hb = ctx.hidden_dim // s
                # per-block pair matrices (columns of block b)
                n_avail = dc["tr_h"].shape[0]
                g = torch.Generator().manual_seed(master + s)
                order = torch.randperm(n_avail, generator=g)

                def block_solve(h_pool, hw_pool):
                    """Solve each diagonal block independently, compose.
                    Returns (w_hat, errs, failed_blocks): a block whose solve
                    failed leaves zeros in w_hat and None in errs, and the
                    caller records the cell as an error instead of dying."""
                    w_hat = torch.zeros(ctx.hidden_dim, ctx.hidden_dim,
                                        dtype=torch.float64)
                    errs, failed = [], []
                    for b in range(s):
                        sl = slice(b * hb, (b + 1) * hb)
                        wb, solver = solve_w(h_pool[:, sl], hw_pool[:, sl])
                        if wb is None:
                            failed.append(b)
                            errs.append(None)
                            continue
                        w_hat[sl, sl] = wb
                        errs.append(wb)
                    return w_hat, errs, failed

                # negative control: static sharded W, 2-epoch pool
                w_static = block_diag_secret(ctx.hidden_dim, s, master)
                idx = order[:2 * E].clone() if 2 * E <= n_avail else \
                    order.clone()
                assert int(idx.min()) >= 0 and int(idx.max()) < n_avail, \
                    "corrupted index slice (memory stomp during solve?)"
                hp = dc["tr_h"][idx]
                hwp = h_wire(hp, w_static)
                w_hat_s, errs_s, failed_s = block_solve(hp, hwp)
                per_block_err = [None if errs_s[b] is None else round(
                    w_rel_err(errs_s[b].float(),
                              w_static[b * hb:(b + 1) * hb,
                                       b * hb:(b + 1) * hb].double()), 6)
                    for b in range(s)]
                rec_static = None if failed_s else recovery_with_what(
                    dec, h_wire(dc["h_star"], w_static), w_hat_s.float(),
                    ctx.victim_tok, args.device)
                if failed_s:
                    print(f"[er5] depth={sa} seed={seed} s={s}: static "
                          f"block solve failed for blocks {failed_s}; "
                          f"cell recorded as error")
                # free the large fp64 static pool before the epoch loop; at
                # depth 8 three depth-contexts are cached and every MB of
                # headroom matters on the GB10 unified memory
                del hp, hwp, w_hat_s

                # staggered rotation, sliding 2-epoch window
                def epoch_key(block, epoch):
                    # block i has rotated once per schedule hit up to `epoch`
                    n_rot = sum(block in stagger_rotating_blocks(e, s, period)
                                for e in range(epoch + 1))
                    return ratchet_seed(master, block * 1000 + n_rot)

                # Materialize the per-epoch pair gather ONCE, before any of
                # the epoch loop's fp64 solves — same GB10 memory-stomp
                # discipline as er1: both window epochs reuse the SAME E
                # pairs (order[:E] below), so no int64 index tensor needs to
                # be read after a solve has run. Clone, bounds-check, gather
                # here; `order` is dead afterwards.
                idx_e = order[:E].clone() if E <= n_avail else order.clone()
                assert int(idx_e.min()) >= 0 and int(idx_e.max()) < n_avail, \
                    "corrupted index slice (memory stomp during solve?)"
                h_E = dc["tr_h"][idx_e]
                del order

                per_epoch = []
                for e in range(1, period + 1):
                    hp_e, hwp_e = [], []
                    for ep in (e - 1, e):  # sliding window of 2 epochs
                        w_ep = torch.zeros(ctx.hidden_dim, ctx.hidden_dim)
                        for b in range(s):
                            sl = slice(b * hb, (b + 1) * hb)
                            w_ep[sl, sl] = make_secret(
                                hb, epoch_key(b, ep))
                        hp_e.append(h_E)
                        hwp_e.append(h_wire(h_E, w_ep))
                        if ep == e:
                            w_cur = w_ep
                    # blocks whose KEY actually differs across the window —
                    # their pooled lstsq mixes two keys and must fail
                    rotated_in_window = [b for b in range(s)
                                         if epoch_key(b, e - 1)
                                         != epoch_key(b, e)]
                    w_hat_e, errs_e, failed_e = block_solve(
                        torch.cat(hp_e), torch.cat(hwp_e))
                    if failed_e:
                        err = f"block solve failed: blocks {failed_e}"
                        print(f"[er5] depth={sa} seed={seed} s={s} "
                              f"epoch={e}: SOLVE FAILED ({err}); cell "
                              f"recorded as error")
                        per_epoch.append({
                            "epoch": e, "composed_top1": None,
                            "error": err,
                            "per_block_rel_err": [None] * s,
                            "blocks_rotated_in_window": rotated_in_window})
                        _append_jsonl(args, {"experiment": "er5", "depth": sa,
                                             "seed": seed, "n_blocks": s,
                                             "epoch": e, "error": err,
                                             "composed_top1": None,
                                             "blocks_rotated_in_window":
                                             rotated_in_window})
                        continue
                    rec_e = recovery_with_what(
                        dec, h_wire(dc["h_star"], w_cur), w_hat_e.float(),
                        ctx.victim_tok, args.device)
                    per_block_err_e = [round(w_rel_err(
                        errs_e[b].float(),
                        w_cur[b * hb:(b + 1) * hb, b * hb:(b + 1) * hb]
                        .double()), 6) for b in range(s)]
                    per_epoch.append({"epoch": e, "composed_top1": rec_e,
                                      "per_block_rel_err": per_block_err_e,
                                      "blocks_rotated_in_window":
                                      rotated_in_window})
                    print(f"[er5] depth={sa} seed={seed} s={s} epoch={e}: "
                          f"composed={rec_e:.2f}% (window rotated blocks "
                          f"{rotated_in_window})")
                    _append_jsonl(args, {"experiment": "er5", "depth": sa,
                                         "seed": seed, "n_blocks": s,
                                         "epoch": e,
                                         "composed_top1": rec_e,
                                         "blocks_rotated_in_window":
                                         rotated_in_window})
                band = ctx.band_top1(dc, seed, args)
                entry = {"experiment": "er5", "depth": sa, "seed": seed,
                         "n_blocks": s, "block_size": hb,
                         "epoch_pairs": E, "window_epochs": window,
                         "rotation_period": period,
                         "static_sharded": {
                             "composed_top1": rec_static,
                             "per_block_rel_err": per_block_err,
                             "per_block_budget_pairs": round(0.1 * hb, 1),
                             "pairs_in_window": int(idx.shape[0])},
                         "staggered": {"per_epoch": per_epoch,
                                       "composed_top1_mean": (lambda xs:
                                           round(sum(xs) / len(xs), 4)
                                           if xs else None)(
                                           [p["composed_top1"]
                                            for p in per_epoch
                                            if p["composed_top1"]
                                            is not None])},
                         "labelfree_band_top1": band}
                out["results"].append(entry)
                _append_jsonl(args, entry)
                rs = (f"{rec_static:.2f}%" if rec_static is not None
                      else "solve-error")
                ms = entry["staggered"]["composed_top1_mean"]
                ms = f"{ms:.2f}%" if ms is not None else "n/a"
                print(f"[er5] depth={sa} seed={seed} s={s}: static-sharded "
                      f"composed={rs} (NEGATIVE CONTROL — small "
                      f"blocks are easier), staggered mean={ms} "
                      f"(band {band:.2f}%)")
        for s in args.shards:
            rows = [r for r in out["results"]
                    if r["experiment"] == "er5" and r["depth"] == sa
                    and r["n_blocks"] == s]
            if not rows:
                continue
            out["summary"].append({
                "experiment": "er5", "depth": sa, "n_blocks": s,
                "static_sharded_composed_mean": (lambda xs: round(
                    sum(xs) / len(xs), 4) if xs else None)(
                    [r["static_sharded"]["composed_top1"] for r in rows
                     if r["static_sharded"]["composed_top1"] is not None]),
                "staggered_composed_mean": (lambda xs: round(
                    sum(xs) / len(xs), 4) if xs else None)(
                    [r["staggered"]["composed_top1_mean"] for r in rows
                     if r["staggered"]["composed_top1_mean"] is not None]),
                "labelfree_band_mean": round(
                    sum(r["labelfree_band_top1"] for r in rows) / len(rows), 4),
                "n_seeds": len(rows)})


EXPERIMENTS = {"er1": run_er1, "er3": run_er3, "er4": run_er4, "er5": run_er5}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=list(EXPERIMENTS) + ["all"],
                    default="all")
    ap.add_argument("--model", default=os.path.expanduser(
        "~/experiments/models/qwen3-0.6b"), help="HF model path (ignored with --toy)")
    ap.add_argument("--toy", action="store_true",
                    help="tiny random built-in model (CPU machinery check only; "
                         "depths clamp to the toy's 4 layers, hidden=64)")
    ap.add_argument("--corpus-file", default=None,
                    help="public text, one document per line; the LAST "
                         "--victim-docs documents are the victim eval docs")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--solve-seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="independent pair-order/subsample solve repetitions")
    ap.add_argument("--victim-docs", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--max-pairs", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=30,
                    help="decoder training epochs")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epoch-sizes", type=int, nargs="+",
                    default=[64, 256, 1024, 4096],
                    help="E-R1a: labeled-pairs-per-epoch sweep")
    ap.add_argument("--prefix-len", type=int, default=64,
                    help="E-R3: tokens per known-prefix session (S)")
    ap.add_argument("--sessions", type=int, default=16,
                    help="E-R3: sessions per accumulation curve")
    ap.add_argument("--stale-epoch-sizes", type=int, nargs="+",
                    default=[256, 1024],
                    help="E-R4: fresh-pair budgets for epoch 4")
    ap.add_argument("--shards", type=int, nargs="+", default=[1, 4, 16],
                    help="E-R5: block-diagonal shard counts")
    ap.add_argument("--rotation-period", type=int, default=8,
                    help="E-R5: epochs per full rotation period")
    ap.add_argument("--shard-epoch-pairs", type=int, default=1024,
                    help="E-R5: labeled pairs harvested per epoch (E)")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device",
                    default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    ap.add_argument("--attn-impl", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="depth 1, 1 seed, small grids, 2 victim docs, "
                         "5 decoder epochs, seq 16 (<=5 min CPU on --toy)")
    ap.add_argument("--output", default="rotation_lifetime.json")
    ap.add_argument("--self-test", action="store_true",
                    help="pure-python fixture checks; no torch needed")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test())

    if torch is None or build_modules is None:
        ap.error("torch/transformers not installed; install them or run --help only")

    if args.quick:
        args.depths = [1]
        args.seeds = [0]
        args.solve_seeds = [0, 1]
        args.victim_docs = 2
        args.epochs = 5
        args.seq_len = 16
        args.max_pairs = 2000
        args.epoch_sizes = [64, 256]
        args.prefix_len = 16
        args.sessions = 4
        args.stale_epoch_sizes = [64, 256]
        args.shards = [1, 4]
        args.rotation_period = 4
        args.shard_epoch_pairs = 64

    seed_all(args.seed)
    _write_training_status(state="running", task="rotation_lifetime",
                           experiments=args.experiment, depths=args.depths,
                           started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    ctx = Ctx(args)
    out = {
        "schema": "dtraining.rotation_lifetime.v1",
        "config": {"model": "toy" if args.toy else args.model,
                   "n_layers": ctx.n_layers, "experiment": args.experiment,
                   "depths": args.depths, "seeds": args.seeds,
                   "solve_seeds": args.solve_seeds,
                   "victim_docs": args.victim_docs, "seq_len": args.seq_len,
                   "epoch_sizes": args.epoch_sizes,
                   "prefix_len": args.prefix_len, "sessions": args.sessions,
                   "stale_epoch_sizes": args.stale_epoch_sizes,
                   "shards": args.shards,
                   "rotation_period": args.rotation_period,
                   "shard_epoch_pairs": args.shard_epoch_pairs,
                   "dtype": args.dtype, "device": args.device,
                   "master_seed_base": args.seed, "quick": args.quick},
        "threat_model": "honest-but-curious cloud under PER-EPOCH rotation "
                        "of the E8 boundary secret W (fresh ratchet-derived W "
                        "each epoch). Attacker capabilities under test: "
                        "(E-R1a) E labeled (h, hW) pairs from WITHIN one "
                        "epoch (oracle serving/self-labeled traffic); "
                        "(E-R3) knowledge of a fixed system-prompt prefix, "
                        "static W, with/without scaffold jitter; (E-R4) an "
                        "insider HANDED stale keys W_1..W_3 plus the "
                        "derivation rule but not master_seed; (E-R5) labeled "
                        "pairs against block-diagonal sharded secrets with a "
                        "2-epoch sliding window. W itself is never leaked in "
                        "the current epoch.",
        "interpretation": "E-R1a: recovery below 50% at epoch size E => the "
                          "per-epoch budget is safe at E; max_safe_epoch is "
                          "compared against the static-W K50 prior "
                          f"{K50_PRIOR} (depths 1/4/8, 27B bf16, "
                          f"{K50_PRIOR_ARTIFACT}). The pooled-cross-epoch "
                          "control must sit in the label-free band (<=3.7% "
                          "on real models): accumulation does not cross "
                          "rotation boundaries. E-R3: fixed_scaffold "
                          "reproduces the known break (~4 sessions at S=64, "
                          "K90 ~= 232-240 pairs); jittered_scaffold must "
                          "show no K90 crossing within 16 sessions. E-R4: "
                          "stale-only recovery ~= band with W rel_err ~= "
                          "sqrt(2) (random); fresh-pair cells must match "
                          "E-R1a at the same E (stale keys give zero "
                          "bootstrap). E-R5: static sharding is the NEGATIVE "
                          "CONTROL — per-block budgets shrink to 0.1*h_b, so "
                          "s=16 is easier to break than s=1; staggering "
                          "restores the defense because no 2-epoch window "
                          "has all blocks on one key.",
        "provenance": ctx.provenance,
        "random_baseline_top1_pct": ctx.random_top1,
        "measurement_kind": "lab-harness simulation: attacks executed "
                            "in-process against collected boundary "
                            "activations; no live wire",
        "evidence_status": "primary",
        "known_limitations": [
            "labeled-pair attacker is an upper bound (oracle serving); real "
            "self-labeled traffic yields noisier pairs",
            "rotation cadence is per-epoch; sub-epoch accumulation is the "
            "binding constraint, so epoch sizes must stay below "
            "max_safe_epoch with margin",
            "E-R3 jitter models pad-insertion + slot permutation only; a "
            "stronger attacker with approximate alignment search is not "
            "covered",
            "E-R5 sharding WITHOUT staggering strictly weakens per-block "
            "budgets (negative control) — shard counts must not be deployed "
            "without the staggered schedule",
            "historical artifacts are immutable; the K50 prior quoted here "
            f"comes from {K50_PRIOR_ARTIFACT}"],
        "summary": [],
        "results": [],
    }

    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for name in names:
        EXPERIMENTS[name](ctx, args, out)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:  # crash-safe per experiment
            json.dump(out, f, indent=2)

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    _write_training_status(state="done", result_file=args.output)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
