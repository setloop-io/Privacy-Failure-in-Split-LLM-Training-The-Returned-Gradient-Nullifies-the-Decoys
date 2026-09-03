#!/usr/bin/env python3
"""SipIt-style exact sequential inversion attack (A3) on split boundaries.

Replication of Nikolaou et al. 2025, "Language Models are Injective and
Hence Invertible" (SipIt): transformer hidden states are injective in the
input tokens, so an observer of the boundary hidden-state sequence can
recover the exact input LEFT-TO-RIGHT by exploiting causality — the hidden
state h_t depends only on tokens 0..t.

Attack setting (split learning / split inference): the cloud observes the
boundary activations H = h_0..h_{T-1} after the victim's first d LOCAL
layers of a PUBLIC, UNMODIFIED model. The attacker possesses those exact
layers (embedding + layers 0..d) — e.g. Qwen3-0.6B from the Hub — and runs:

    for t in 0..T-1:
        for each candidate token x (shortlisted, see below):
            compute h_t(prefix + x) = f_d(embed(prefix ++ [x]))[-1]
        commit x* = argmin || h_t(prefix + x) - h_obs[t] ||   (L2)
        prefix += [x*]

Under exact arithmetic this is exact (injectivity + causal isolation).
In bf16/fp16 the committed prefix drifts from the victim's true prefix, so
the recomputed h_t accumulates floating-point error — this script QUANTIFIES
that (--dtypes bf16 fp32, --depths 1 4 8).

Candidate shortlist — v3 hybrid "MLP proposes, SipIt verifies":
  1. Dictionary: h_0 depends ONLY on token 0, so f_d(embed(x)) for the whole
     vocab is precomputed once (the "dictionary" D). Position 0 is an exact
     full-vocab nearest-neighbor lookup.
  2. Geometric proxy for t >= 1 (v2): R in-context anchor tokens calibrate a
     nearest-anchor local linearization of the dictionary,
         proxy(x) = min_r || h_meas(prefix + a_r) + D[x] - D[a_r] - h_obs[t] ||
     scored via the quadratic expansion ||D[x]||^2 + 2 D[x].off + ||off||^2
     (no [V,H] temporaries).
  3. Trained-MLP proposer (v3): the v1/v2 geometric proxies COLLAPSE at
     boundary depth >= 4 — measured oracle recall@4096 ~ 0-25%, mean proxy
     rank ~65k/152k (see diagnostics): attention mixes the prefix into the
     last position, so dictionary-space linearization is rank-deficient
     there. v3 therefore adds the trained_inversion.InversionDecoder MLP,
     trained per (depth, dtype) cell on (h_i, token_i) pairs from PUBLIC
     corpus text through the SAME public head (document-disjoint from the
     victim docs), and takes its top-M logits per position. The candidate
     set is union(proxy top-K, MLP top-M); the EXACT batched rescore then
     certifies which candidate is right. The MLP only proposes — every
     committed token is exactly verified against the observed boundary,
     so the attack's exactness guarantee is untouched.
  4. Adaptive K: an ORACLE diagnostic measures union recall@K of the true
     token (true prefix, sampled positions) on a K-ladder before each cell
     and picks the smallest K reaching --recall-target (cap --shortlist-cap).

THREAT-MODEL CAVEAT (implemented honestly, repeated in the JSON output):
this attack requires the attacker to possess the EXACT local-layer weights.
It applies to public-model deployments only. It does NOT apply against
E8-style basis obfuscation (unknown local transform) or genuinely private
(fine-tuned/secret) local weights — for those settings it is a ceiling, not
a demonstrated break.

Usage:
    python sipit_inversion.py --help        # works without torch
    python sipit_inversion.py --toy --quick # CPU machinery check
    python sipit_inversion.py --model <hf-model> --corpus-file <docs.txt> --depths 1 4 8
"""

import argparse
import json
import os
import sys
import time

# Guarded heavy imports: `--help` must work on torch-less hosts.
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from split_trainer import (TEXT_SAMPLES, _write_training_status,
                               build_modules, make_layer_kwargs,
                               run_layer_stack)
    from trained_inversion import (seed_all, mean_std, train_decoder,
                                   evaluate_decoder, make_provenance)
except ImportError:  # pragma: no cover - torch-less host
    TEXT_SAMPLES = []
    _write_training_status = lambda **k: None
    build_modules = make_layer_kwargs = run_layer_stack = seed_all = None
    mean_std = train_decoder = evaluate_decoder = make_provenance = None


# Head forward: embed + layers 0..d (inclusive). We slice the layer list
# directly rather than reusing trained_inversion.split_at: its clamp
# (n_layers - 3) leaves room for cloud+tail stages in TRAINING — an
# inversion attacker needs only the head, so depth d may go to n_layers - 1.
def head_forward(head, embed, rotary, ids, args):
    """ids [B, T] -> boundary hidden [B, T, hidden] after layers 0..d."""
    hidden = embed(ids)
    position_ids = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    lk = make_layer_kwargs(rotary, hidden, position_ids, args)
    return run_layer_stack(head, hidden, lk)


def build_dictionary(head, embed, rotary, vocab_size, args, chunk=256):
    """D[x] = f_d(embed(x)) for a length-1 sequence — the exact h_0 for
    token x. One batched pass over the vocab; returns ([V,H] float32 on
    args.device, row norms [V]) for quadratic-expansion scoring."""
    outs = []
    with torch.no_grad():
        for i in range(0, vocab_size, chunk):
            ids = torch.arange(i, min(i + chunk, vocab_size),
                               device=args.device).unsqueeze(1)
            outs.append(head_forward(head, embed, rotary, ids, args)[:, 0]
                        .float())
    D = torch.cat(outs)
    return D, D.pow(2).sum(dim=-1)


def make_anchors(vocab_size, n_refs, device):
    """R anchor tokens for the calibrated proxy, spread evenly over the vocab
    (geometry-agnostic; the anchor FORWARD supplies the in-context info)."""
    R = max(1, min(n_refs, vocab_size))
    return torch.linspace(0, vocab_size - 1, R).long().unique().to(device)


def proxy_scores(prefix, anchors, target, D, D_norm2, head, embed, rotary,
                 args):
    """Nearest-anchor calibrated proxy distances [vocab] for the next token
    after `prefix`, against observed boundary vector `target` ([H], device,
    float32). One batched anchor forward [R, len(prefix)+1, hidden].

    For anchor a_r with measured h_r = f_d(prefix ++ [a_r])[-1]:
        score_r(x) = || h_r + D[x] - D[a_r] - target ||^2
                   = ||D[x]||^2 + 2 D[x].off_r + ||off_r||^2
    with off_r = h_r - D[a_r] - target. Returns sqrt(min_r score_r)."""
    batch = torch.tensor([prefix + [int(a)] for a in anchors],
                         device=args.device)
    h_meas = head_forward(head, embed, rotary, batch, args)[:, -1].float()  # [R,H]
    best = None
    for r in range(anchors.shape[0]):
        a = int(anchors[r])
        off = h_meas[r] - D[a] - target
        s = D_norm2 + 2.0 * (D @ off) + float(off @ off)
        best = s if best is None else torch.minimum(best, s)
    return best.clamp_min(0).sqrt()


# Trained-MLP proposer: per-position pairs (h_i, token_i) from PUBLIC docs
# through the SAME public head (document-disjoint from victim docs).
def collect_head_pairs(head, embed, rotary, encode, docs, args, max_pairs):
    hs, toks, n = [], [], 0
    with torch.no_grad():
        for doc in docs:
            for block in encode([doc], args.seq_len):
                ids = block[:-1].unsqueeze(0).to(args.device)
                h = head_forward(head, embed, rotary, ids, args)[0].float()
                hs.append(h.cpu())
                toks.append(ids[0].cpu())
                n += ids.shape[1]
                if n >= max_pairs:
                    break
            if n >= max_pairs:
                break
    return torch.cat(hs), torch.cat(toks)


def candidate_union(proxy_d, k, decoder, mlp_k, target):
    """Union of proxy top-k and MLP top-mlp_k candidate token ids."""
    sets = [proxy_d.topk(min(k, proxy_d.shape[0]), largest=False).indices]
    if decoder is not None and mlp_k > 0:
        logits = decoder(target.unsqueeze(0))[0]
        sets.append(logits.topk(min(mlp_k, logits.shape[0])).indices)
    return torch.unique(torch.cat(sets))


# Oracle shortlist diagnostic: with the TRUE prefix, is the true token in the
# candidate union? -> recall@K per ladder K. Distinguishes "shortlist too
# small" (recall rises with K) from "proposer broken" (recall flat at depth).
def measure_recall(head, embed, rotary, D, D_norm2, anchors, decoder, mlp_k,
                   toks, obs, ks, args, n_positions):
    T = len(toks)
    n_pos = min(n_positions, max(T - 1, 1))
    positions = sorted({max(1, min(T - 1, int(round(p))))
                        for p in torch.linspace(1, max(T - 1, 1), n_pos).tolist()})
    hits = {k: 0 for k in ks}
    ranks = []
    with torch.no_grad():
        for t in positions:
            true_tok = toks[t]
            target = obs[t].to(args.device)
            proxy_d = proxy_scores(toks[:t], anchors, target, D, D_norm2,
                                   head, embed, rotary, args)
            rank = int((proxy_d < proxy_d[true_tok]).sum().item())
            ranks.append(rank)
            for k in ks:
                cand = candidate_union(proxy_d, k, decoder, mlp_k, target)
                if int((cand == true_tok).sum()):
                    hits[k] += 1
    n = max(len(positions), 1)
    recall = {str(k): round(hits[k] / n, 4) for k in ks}
    mean_rank = round(sum(ranks) / n, 1)
    return recall, mean_rank, positions


def choose_k(recall, ks, target):
    for k in ks:
        if recall[str(k)] >= target:
            return k
    return ks[-1]


# SipIt core: sequential left-to-right recovery of one sequence.
def recover_sequence(obs_H, head, embed, rotary, D, D_norm2, anchors,
                     decoder, mlp_k, shortlist_k, args):
    """obs_H [T, hidden] (float32 CPU) observed boundary sequence.
    Returns (recovered ids, stats)."""
    T = obs_H.shape[0]
    device = args.device
    vocab = D.shape[0]
    k = min(shortlist_k, vocab)
    recovered = []
    fwd_calls = 0
    t_start = time.perf_counter()

    with torch.no_grad():
        for t in range(T):
            target = obs_H[t].to(device)
            if t == 0:
                # Exact full-vocab scoring via the dictionary: D[x] IS h_0(x).
                d = torch.linalg.norm(D - target, dim=-1)
                recovered.append(int(d.argmin().item()))
                continue
            # candidate union: calibrated proxy top-k U MLP top-mlp_k
            proxy_d = proxy_scores(recovered, anchors, target, D, D_norm2,
                                   head, embed, rotary, args)
            fwd_calls += 1
            cand = candidate_union(proxy_d, k, decoder, mlp_k, target)
            # exact batched rescore: one forward [|cand|, t+1, hidden]
            batch = torch.stack([torch.tensor(recovered + [int(c)],
                                              device=device) for c in cand])
            h_cand = head_forward(head, embed, rotary, batch, args)[:, -1].float()
            fwd_calls += 1
            d = torch.linalg.norm(h_cand - target, dim=-1)
            recovered.append(int(cand[int(d.argmin().item())].item()))
            del batch, h_cand

    elapsed = time.perf_counter() - t_start
    # Verification: recompute the full boundary from the recovered prefix.
    with torch.no_grad():
        ids = torch.tensor([recovered], device=device)
        h_rec = head_forward(head, embed, rotary, ids, args)[0].float().cpu()
    rel_l2 = (torch.linalg.norm(h_rec - obs_H, dim=-1)
              / obs_H.norm(dim=-1).clamp_min(1e-12))
    stats = {
        "elapsed_s": round(elapsed, 3),
        "tokens_per_s": round(T / max(elapsed, 1e-9), 2),
        "forward_calls": fwd_calls,
        "verify_rel_l2_mean": round(float(rel_l2.mean()), 6),
        "verify_rel_l2_max": round(float(rel_l2.max()), 6),
    }
    return recovered, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.expanduser(
        "~/experiments/models/qwen3-0.6b"), help="HF model path (ignored with --toy)")
    ap.add_argument("--toy", action="store_true",
                    help="tiny random built-in model (CPU machinery check only)")
    ap.add_argument("--corpus-file", default=None,
                    help="public text, one document per line; victim docs are "
                         "held out from the END; the rest trains the MLP proposer")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 4, 8],
                    help="split depths d (local layers 0..d observed)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dtypes", nargs="+", default=["bf16"],
                    choices=["bf16", "fp16", "fp32"],
                    help="model/boundary dtypes to test fp error accumulation")
    ap.add_argument("--victim-docs", type=int, default=3,
                    help="victim sequences attacked per (depth, seed, dtype)")
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="tokens recovered per victim sequence")
    ap.add_argument("--shortlist", type=int, default=100,
                    help="base proxy shortlist size K (t>=1); the oracle recall "
                         "diagnostic may raise it up to --shortlist-cap")
    ap.add_argument("--shortlist-cap", type=int, default=4096,
                    help="hard cap for the adaptive proxy shortlist size")
    ap.add_argument("--mlp-shortlist", type=int, default=256,
                    help="top-M candidates from the trained MLP proposer, "
                         "unioned with the proxy shortlist (0 disables the MLP)")
    ap.add_argument("--mlp-pairs", type=int, default=20000,
                    help="cap on (h, token) pairs for MLP proposer training")
    ap.add_argument("--epochs", type=int, default=10,
                    help="MLP proposer training epochs per cell")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--recall-target", type=float, default=0.99,
                    help="oracle recall@K target for adaptive K selection")
    ap.add_argument("--recall-positions", type=int, default=12,
                    help="positions sampled per cell in the recall diagnostic")
    ap.add_argument("--proxy-refs", type=int, default=8,
                    help="R anchor tokens for the calibrated shortlist proxy")
    ap.add_argument("--seq-len", type=int, default=64,
                    help="encoding block length (>= max-tokens)")
    ap.add_argument("--device",
                    default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    ap.add_argument("--attn-impl", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--quick", action="store_true",
                    help="depth 1, 1 seed, 1 victim doc, 32 tokens (<=5 min CPU on --toy)")
    ap.add_argument("--output", default="sipit_inversion_results.json")
    args = ap.parse_args()

    if torch is None or build_modules is None:
        ap.error("torch/transformers not installed; install them or run --help only")

    if args.quick:
        args.depths = [1]
        args.seeds = [0]
        args.victim_docs = 1
        args.max_tokens = 32
        args.seq_len = 33
        args.dtypes = ["bf16"]
        args.epochs = 3
        args.mlp_pairs = 2000

    _write_training_status(state="running", task="sipit_inversion",
                           depths=args.depths, seeds=args.seeds,
                           dtypes=args.dtypes,
                           started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    # victim docs from the END of the corpus
    # --corpus-file REPLACES TEXT_SAMPLES (no mixing, so results are
    # attributable to one source).
    if args.corpus_file:
        corpus_source = "corpus_file"
        with open(args.corpus_file) as f:
            docs = [l.strip() for l in f if len(l.strip()) > 500]
    else:
        corpus_source = "TEXT_SAMPLES"
        docs = list(TEXT_SAMPLES)
    if len(docs) < args.victim_docs + 4:
        raise ValueError(f"corpus too small: {len(docs)} docs; need victim-docs "
                         f"({args.victim_docs}) + at least 4 MLP-training docs")
    victim_docs = docs[-args.victim_docs:]
    attack_docs = docs[:-args.victim_docs]
    provenance = make_provenance(
        args.corpus_file, corpus_source, len(docs),
        range(len(docs) - args.victim_docs, len(docs)),
        model_path=None if args.toy else args.model)
    n_val = max(1, int(round(0.15 * len(attack_docs))))
    mlp_val_docs, mlp_train_docs = attack_docs[:n_val], attack_docs[n_val:]
    print(f"[data] {len(docs)} docs: {len(mlp_train_docs)} MLP-train, "
          f"{n_val} MLP-val, {len(victim_docs)} victim (held out from end)")

    results = []
    summary = []
    diagnostics = []

    for dtype in args.dtypes:
        args.dtype = dtype  # build_modules reads args.dtype
        embed, layers, norm, lm_head, rotary, encode = build_modules(args)
        n_layers = len(layers)
        vocab_size = embed.weight.shape[0]
        print(f"[model] {'toy' if args.toy else args.model}: {n_layers} layers, "
              f"vocab={vocab_size}, dtype={dtype}, device={args.device}")

        # victim token blocks: encode each victim doc, take one block; a seeded
        # RNG picks the contiguous max-tokens window attacked (per-seed variety).
        victim_blocks = []
        for doc in victim_docs:
            blocks = encode([doc], args.seq_len)
            if blocks:
                victim_blocks.append(blocks[0][:-1])  # input positions only
        if not victim_blocks:
            raise ValueError("no victim doc long enough to yield a block")

        anchors = make_anchors(vocab_size, args.proxy_refs, args.device)

        for depth in args.depths:
            d = max(0, min(depth, n_layers - 1))
            if d != depth:
                print(f"[split] depth {depth} clamped to {d} ({n_layers} layers)")
            head = list(layers[: d + 1])

            t0 = time.time()
            D, D_norm2 = build_dictionary(head, embed, rotary, vocab_size, args)
            print(f"[dict] depth={d}: dictionary [{D.shape[0]}x"
                  f"{D.shape[1]}] in {time.time() - t0:.1f}s")

            # train the MLP proposer for this cell (public docs only)
            decoder = None
            if args.mlp_shortlist > 0:
                t0 = time.time()
                tr_h, tr_tok = collect_head_pairs(
                    head, embed, rotary, encode, mlp_train_docs, args,
                    args.mlp_pairs)
                va_h, va_tok = collect_head_pairs(
                    head, embed, rotary, encode, mlp_val_docs, args,
                    max(1000, args.mlp_pairs // 4))
                print(f"[mlp] depth={d} dtype={dtype}: train={tr_h.shape[0]} "
                      f"val={va_h.shape[0]} pairs ({time.time() - t0:.1f}s)")
                seed_all(args.seeds[0])
                decoder = train_decoder(
                    tr_h, tr_tok, va_h, va_tok, tr_h.shape[1], vocab_size,
                    args, f"sipit_mlp_d{d}_{dtype}")
                vtop1, vtop5 = evaluate_decoder(decoder, va_h, va_tok,
                                                args.device)
                print(f"[mlp] depth={d} dtype={dtype}: proposer val "
                      f"top-1={vtop1:.2f}% top-5={vtop5:.2f}%")
                diagnostics.append({"depth": d, "dtype": dtype, "phase": "mlp",
                                    "val_top1": vtop1, "val_top5": vtop5})
                decoder.eval()
                for p in decoder.parameters():
                    p.requires_grad_(False)
                del tr_h, tr_tok, va_h, va_tok

            # K ladder for the oracle recall diagnostic (deduped, <= vocab).
            ks = sorted({min(v, vocab_size) for v in
                         (args.shortlist, 256, 1024, args.shortlist_cap)})

            for seed in args.seeds:
                seed_all(seed)
                import random as _random
                rng = _random.Random(seed)
                for vi, block in enumerate(victim_blocks):
                    toks = block.tolist()
                    if len(toks) > args.max_tokens:
                        start = rng.randrange(0, len(toks) - args.max_tokens + 1)
                        toks = toks[start:start + args.max_tokens]
                    T = len(toks)
                    ids = torch.tensor([toks], device=args.device)
                    with torch.no_grad():
                        obs = head_forward(head, embed, rotary, ids,
                                           args)[0].float().cpu()

                    # oracle shortlist-recall diagnostic + adaptive K
                    recall, mean_rank, positions = measure_recall(
                        head, embed, rotary, D, D_norm2, anchors, decoder,
                        args.mlp_shortlist, toks, obs, ks, args,
                        args.recall_positions)
                    k_chosen = choose_k(recall, ks, args.recall_target)
                    diagnostics.append({
                        "depth": d, "dtype": dtype, "seed": seed,
                        "victim_idx": vi, "phase": "recall",
                        "proxy_refs": int(anchors.shape[0]),
                        "mlp_shortlist": args.mlp_shortlist,
                        "recall_at_k": recall, "mean_proxy_rank": mean_rank,
                        "n_positions": len(positions), "k_chosen": k_chosen})
                    print(f"[recall] depth={d} dtype={dtype} seed={seed} "
                          f"victim={vi}: "
                          + " ".join(f"@{k}={recall[str(k)]:.2f}" for k in ks)
                          + f" mean_rank={mean_rank} -> K={k_chosen}")

                    rec, stats = recover_sequence(
                        obs, head, embed, rotary, D, D_norm2, anchors,
                        decoder, args.mlp_shortlist, k_chosen, args)
                    correct = [int(a == b) for a, b in zip(rec, toks)]
                    acc = 100.0 * sum(correct) / T
                    first_err = next((i for i, c in enumerate(correct) if not c), None)
                    row = {"depth": d, "dtype": dtype, "seed": seed,
                           "victim_idx": vi, "n_tokens": T,
                           "k_used": k_chosen,
                           "exact_recovery_rate": round(acc, 2),
                           "first_error_pos": first_err,
                           "per_position_correct": correct,
                           **stats}
                    results.append(row)
                    print(f"[attack] depth={d} dtype={dtype} seed={seed} "
                          f"victim={vi}: acc={acc:.2f}% first_err={first_err} "
                          f"{stats['tokens_per_s']} tok/s "
                          f"verifyL2={stats['verify_rel_l2_max']:.2e}")
                    _write_training_status(state="running", phase="attack",
                                           depth=d, dtype=dtype, seed=seed,
                                           acc=acc)
            del D, D_norm2, decoder

        del embed, layers, norm, lm_head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # aggregate over seeds/victims per (depth, dtype)
    cells = sorted({(r["depth"], r["dtype"]) for r in results})
    for depth, dtype in cells:
        rows = [r for r in results if r["depth"] == depth and r["dtype"] == dtype]
        accs = [r["exact_recovery_rate"] for r in rows]
        speeds = [r["tokens_per_s"] for r in rows]
        m, s = mean_std(accs)
        sm, ss = mean_std(speeds)
        L = min(len(r["per_position_correct"]) for r in rows)
        per_pos = [round(100.0 * sum(r["per_position_correct"][i] for r in rows)
                         / len(rows), 2) for i in range(L)]
        diags = [x for x in diagnostics if x.get("phase") == "recall"
                 and x["depth"] == depth and x["dtype"] == dtype]
        recall_mean = {}
        for kstr in diags[0]["recall_at_k"] if diags else {}:
            vals = [x["recall_at_k"][kstr] for x in diags]
            recall_mean[kstr] = round(sum(vals) / len(vals), 4)
        summary.append({"depth": depth, "dtype": dtype,
                        "recovery_mean": m, "recovery_std": s,
                        "tokens_per_s_mean": sm, "tokens_per_s_std": ss,
                        "recall_at_k_mean": recall_mean,
                        "per_position_acc_mean": per_pos,
                        "n_runs": len(rows)})
        print(f"[summary] depth={depth} dtype={dtype}: "
              f"recovery={m:.2f}+-{s:.2f}% speed={sm:.2f} tok/s "
              f"recall={recall_mean}")

    out = {
        "config": {"model": "toy" if args.toy else args.model,
                   "depths": args.depths, "seeds": args.seeds,
                   "dtypes": args.dtypes, "victim_docs": args.victim_docs,
                   "max_tokens": args.max_tokens, "shortlist": args.shortlist,
                   "shortlist_cap": args.shortlist_cap,
                   "mlp_shortlist": args.mlp_shortlist,
                   "mlp_pairs": args.mlp_pairs, "epochs": args.epochs,
                   "recall_target": args.recall_target,
                   "proxy_refs": args.proxy_refs,
                   "seq_len": args.seq_len, "device": args.device,
                   "quick": args.quick},
        "provenance": provenance,
        "threat_model": (
            "semi-honest cloud observes the boundary hidden-state sequence "
            "after the victim's first d local layers of a PUBLIC, UNMODIFIED "
            "model and possesses those exact layers (embedding + 0..d). "
            "SipIt-style sequential inversion (Nikolaou et al. 2025): exact "
            "under exact arithmetic; bf16/fp16 prefix drift degrades it with "
            "depth — measured here. The MLP proposer is trained only on "
            "PUBLIC text through the SAME public head (document-disjoint "
            "from victims) — no victim-side information. Does NOT apply "
            "against E8-style basis obfuscation (unknown local transform) or "
            "genuinely private local weights; for those settings this is a "
            "ceiling for public-weight deployments only."),
        "algorithm": (
            "v3 hybrid 'MLP proposes, SipIt verifies': per position t the "
            "candidate set is union(anchor-calibrated dictionary proxy top-K, "
            "trained InversionDecoder MLP top-M on h_obs[t]); one exact "
            "batched rescore [n_cand, t+1, hidden] certifies argmin L2; "
            "commit and extend prefix. K adaptive per cell via an oracle "
            "recall@K diagnostic. Position 0: exact full-vocab NN in the "
            "dictionary D[x] = f_d(embed(x)). History: v1 single-reference "
            "additive proxy recall@100 ~= 5% at depth >= 4; v2 anchor "
            "calibration did NOT rescue depth >= 4 (recall@4096 ~ 0-25%, "
            "mean proxy rank ~65k/152k) — the geometric proxy is "
            "rank-deficient at depth, motivating the learned proposer."),
        "random_baseline_recovery_pct": None,  # exact-match baseline ~ 1/vocab
        "diagnostics": diagnostics,
        "summary": summary,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    _write_training_status(state="done", result_file=args.output)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
