#!/usr/bin/env python3
"""E8 — secret linear obfuscation of the split-training boundary.

Defense experiment for the trained MLP inversion attacker (68-76% token
recovery at depths 1-8, trained_inversion.py). The local node holds a
SECRET orthogonal matrix W in R^{H x H} (QR of a seeded randn, fp32) and
transforms the boundary activation before it crosses the trust boundary:

    local head:  h' = h @ W            (obfuscation, "transmitted")
    cloud:       middle layers run on h'  ... in THIS experiment the cloud
                 is in-process, so the tail recovers h = h' @ W^T before
                 the tail layers (W^{-1} = W^T, orthogonal)

Threat-model framing (read before interpreting):
  - DEFAULT: the attacker knows the base model and public text but NOT W.
    Its decoder is trained on plain h* pairs and evaluated on h' = h* @ W.
    Expectation: collapse toward the random baseline (W scrambles the
    per-dimension geometry the decoder keyes on).
  - BREACH CASE (a measurement, not the default assumption): if W leaks
    (insider, side channel), the attacker computes h' @ W^T = h and the
    same decoder returns to baseline recovery. We quantify both so the
    paper can say exactly how much the defense hinges on W's secrecy.
  - Unsupervised adaptation from unlabeled h' vectors is OUT OF SCOPE
    (noted as future work).

Utility check: full greedy decode (64 tokens, 2 prompts) with and without
the obfuscation round-trip must produce IDENTICAL tokens; the round-trip
is done in fp32 (h @ W @ W^T, then cast back to model dtype) and we report
the max elementwise deviation of the round-trip. NOTE: with --dtype bf16
the cast back introduces ~1e-3 relative rounding, which CAN flip a greedy
tie — run --dtype fp32 for an exact-identity utility check.

Also measured: per-step overhead of the two matmuls at H=1024
(expected sub-ms on GPU).

Usage:
    python e8_obfuscation.py --help        # works without torch
    python e8_obfuscation.py --toy --quick # CPU machinery check
    python e8_obfuscation.py --model <hf-model> --corpus-file <docs.txt> --output e8.json
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
# Unguarded: add_numerics_args runs while the parser is built (before the
# torch check), so a None fallback would break torch-less `--help`.
from trained_inversion import (MATMUL_PRECISION_CHOICES,  # noqa: E402
                               add_numerics_args, apply_numerics)
try:
    from split_trainer import (TEXT_SAMPLES, _write_training_status,
                               build_modules, make_layer_kwargs,
                               run_layer_stack)
    from trained_inversion import (collect_base_pairs, evaluate_decoder,
                                   make_provenance, mean_std, seed_all,
                                   split_at, train_decoder)
except ImportError:  # pragma: no cover - torch-less host
    TEXT_SAMPLES = []
    _write_training_status = lambda **k: None
    build_modules = make_layer_kwargs = run_layer_stack = None
    collect_base_pairs = evaluate_decoder = mean_std = seed_all = None
    split_at = train_decoder = make_provenance = None

UTILITY_PROMPTS = [
    "The history of the Roman Empire began",
    "def fibonacci(n):",
]


def make_secret(hidden_dim, seed):
    """Seeded orthogonal W via QR of randn (fp32, CPU)."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(hidden_dim, hidden_dim, generator=g))
    return q  # q @ q.T == I


def greedy_decode(embed, head, middle, tail, norm, lm_head, rotary,
                  prompt_ids, args, max_new, W=None):
    """Greedy continuation through the module stages. With W, applies the
    full obfuscation round-trip at the boundary in fp32 (as the local node
    would: h @ W out, @ W^T back in) and casts back to the model dtype."""
    ids = prompt_ids.unsqueeze(0).to(args.device)
    devs, mags = [], []
    with torch.no_grad():
        for _ in range(max_new):
            position_ids = torch.arange(ids.shape[1],
                                        device=args.device).unsqueeze(0)
            hidden = embed(ids)
            lk = make_layer_kwargs(rotary, hidden, position_ids, args)
            h = run_layer_stack(head, hidden, lk)
            if W is not None:
                rt = h.float() @ W @ W.T
                devs.append((rt - h.float()).abs().max().item())
                # max|h| over the SAME steps, so dev/mag is a true relative
                # error; deep-stack activations run ~1000x larger than depth
                # 1, so absolute deviation is not comparable across depths.
                mags.append(h.float().abs().max().item())
                h = rt.to(h.dtype)
            h = run_layer_stack(middle, h, lk)
            h = run_layer_stack(tail, h, lk)
            logits = lm_head(norm(h))
            nxt = logits[0, -1].float().argmax().view(1, 1)
            ids = torch.cat([ids, nxt], dim=1)
    return (ids[0, prompt_ids.shape[0]:].cpu(),
            (max(devs) if devs else None), (max(mags) if mags else None))


def matmul_overhead_ms(hidden_dim, seq_len, args, reps=100):
    """Mean ms per boundary matmul ([1, seq, H] @ [H, H], fp32)."""
    x = torch.randn(1, seq_len, hidden_dim, device=args.device)
    W = make_secret(hidden_dim, args.seed).to(args.device)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    # warmup
    for _ in range(5):
        _ = x @ W
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        y = x @ W
        z = y @ W.T
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    return round(dt / 2 * 1000, 4)  # per matmul, ms


def overhead_by_precision(hidden_dim, seq_len, args):
    """Time the boundary matmul under EVERY fp32 mode, restoring the run's.

    Overhead differs by hidden dimension, not dtype; measuring every mode in
    one run at one H makes the actual precision cost readable
    (overhead_matmul_hidden_dim below records H).
    """
    was = torch.get_float32_matmul_precision()
    out = {}
    try:
        for p in MATMUL_PRECISION_CHOICES:
            torch.set_float32_matmul_precision(p)
            out[p] = matmul_overhead_ms(hidden_dim, seq_len, args)
    finally:
        torch.set_float32_matmul_precision(was)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.expanduser(
        "~/experiments/models/qwen3-0.6b"), help="HF model path (ignored with --toy)")
    ap.add_argument("--toy", action="store_true",
                    help="tiny random built-in model (CPU machinery check only; "
                         "depths clamp to the toy's 4 layers)")
    ap.add_argument("--corpus-file", default=None,
                    help="public attack-training text, one document per line; "
                         "the LAST --victim-docs documents are the victim eval "
                         "docs (never used for attack training)")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--victim-docs", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--max-pairs", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gen-tokens", type=int, default=64,
                    help="greedy utility-check length")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                    help="use fp32 for an exact token-identity utility check")
    ap.add_argument("--device",
                    default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    ap.add_argument("--attn-impl", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--seed", type=int, default=42)
    add_numerics_args(ap)
    ap.add_argument("--quick", action="store_true",
                    help="depth 1, 1 seed, 16 gen tokens, 2 victim docs, "
                         "5 epochs, seq 16")
    ap.add_argument("--output", default="e8_obfuscation.json")
    args = ap.parse_args()

    if torch is None or build_modules is None:
        ap.error("torch/transformers not installed; install them or run --help only")

    numerics = apply_numerics(args)
    print(f"[numerics] {numerics}")

    if args.quick:
        args.depths = [1]
        args.seeds = [0]
        args.victim_docs = 2
        args.epochs = 5
        args.seq_len = 16
        args.gen_tokens = 16
        args.max_pairs = 2000

    seed_all(args.seed)
    _write_training_status(state="running", task="e8_obfuscation",
                           depths=args.depths,
                           started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    embed, layers, norm, lm_head, rotary, encode = build_modules(args)
    n_layers = len(layers)
    vocab_size = lm_head.weight.shape[0] if not args.toy else embed.weight.shape[0]
    hidden_dim = embed.weight.shape[1]
    print(f"[model] {'toy' if args.toy else args.model}: {n_layers} layers, "
          f"hidden={hidden_dim}, vocab={vocab_size}, device={args.device}")

    # secret W
    W = make_secret(hidden_dim, args.seed).to(args.device)
    ortho_err = (W @ W.T - torch.eye(hidden_dim, device=args.device)
                 ).abs().max().item()
    print(f"[secret] W {hidden_dim}x{hidden_dim} orthogonal, "
          f"max |W W^T - I| = {ortho_err:.2e}")

    # victim docs from the END of the corpus
    # --corpus-file REPLACES TEXT_SAMPLES (no mixing, so results are
    # attributable to one source).
    if args.corpus_file:
        corpus_source = "corpus_file"
        with open(args.corpus_file) as f:
            docs = [l.strip() for l in f if len(l.strip()) > 500]  # real docs only — wikitext short lines are formatting artifacts
    else:
        corpus_source = "TEXT_SAMPLES"
        docs = list(TEXT_SAMPLES)
    if len(docs) < args.victim_docs + 4:
        raise ValueError(f"corpus too small: {len(docs)} docs")
    victim_docs = docs[-args.victim_docs:]
    attack_docs = docs[:-args.victim_docs]
    provenance = make_provenance(
        args.corpus_file, corpus_source, len(docs),
        range(len(docs) - args.victim_docs, len(docs)), model_path=getattr(args, 'model', None))
    n_val = max(1, int(round(args.val_frac * len(attack_docs))))
    val_docs, train_docs_pool = attack_docs[:n_val], attack_docs[n_val:]
    print(f"[data] {len(train_docs_pool)} attack-train, {n_val} attack-val, "
          f"{len(victim_docs)} victim (held out from end)")

    victim_ids, victim_tokens = [], []
    for doc in victim_docs:
        b = encode([doc], args.seq_len)
        if b:
            victim_ids.append(b[0])
            victim_tokens.append(b[0][:-1])
    if not victim_ids:
        raise ValueError("no victim doc long enough to yield a block")
    victim_tok = torch.cat(victim_tokens)

    # utility: greedy with vs without round-trip
    print(f"[utility] greedy identity check ({args.gen_tokens} tokens, "
          f"{len(UTILITY_PROMPTS)} prompts, dtype={args.dtype})")
    utility = []
    for depth in args.depths:
        head, middle, tail, sa, ra = split_at(layers, depth, n_layers)
        for pi, prompt in enumerate(UTILITY_PROMPTS):
            ptext = prompt
            while not encode([ptext], args.seq_len):
                ptext = ptext + " " + prompt  # short prompt: repeat to fill a block
            ids = encode([ptext], args.seq_len)[0][:-1]
            seed_all(args.seed)
            base_ids, _, _ = greedy_decode(embed, head, middle, tail, norm,
                                           lm_head, rotary, ids, args,
                                           args.gen_tokens, W=None)
            seed_all(args.seed)
            obf_ids, max_dev, max_h = greedy_decode(embed, head, middle, tail,
                                                    norm, lm_head, rotary, ids,
                                                    args, args.gen_tokens, W=W)
            n = min(len(base_ids), len(obf_ids))
            identity = (base_ids[:n] == obf_ids[:n]).float().mean().item()
            rel_dev = max_dev / max_h if max_h else None
            utility.append({"depth": sa, "prompt_idx": pi,
                            "token_identity": round(identity, 4),
                            "tokens_compared": n,
                            "max_roundtrip_deviation": max_dev,
                            "max_abs_h": max_h,
                            "max_roundtrip_deviation_relative": rel_dev})
            print(f"  depth={sa} prompt={pi}: identity={identity:.4f} "
                  f"max_dev={max_dev:.3e} max|h|={max_h:.3e} "
                  f"rel={rel_dev:.3e}")
            _write_training_status(state="running", phase="utility",
                                   depth=sa, token_identity=round(identity, 4))

    overhead_ms = matmul_overhead_ms(hidden_dim, args.seq_len, args)
    print(f"[overhead] boundary matmul: {overhead_ms} ms per matmul "
          f"([1,{args.seq_len},{hidden_dim}] @ [{hidden_dim},{hidden_dim}] fp32, "
          f"2 per step)")
    overhead_by_prec = overhead_by_precision(hidden_dim, args.seq_len, args)
    print(f"[overhead] by fp32 matmul mode (H={hidden_dim}): {overhead_by_prec}")

    # attack: base decoder on plain / obfuscated / breached features
    results, summary = [], []
    for depth in args.depths:
        head, middle, tail, sa, ra = split_at(layers, depth, n_layers)
        if sa != depth:
            print(f"[split] depth {depth} clamped to sa={sa} ({n_layers} layers)")

        t0 = time.time()
        tr_h, _, tr_tok = collect_base_pairs(
            embed, head, middle, tail, norm, lm_head, rotary, encode,
            train_docs_pool, args, with_grad=False)
        va_h, _, va_tok = collect_base_pairs(
            embed, head, middle, tail, norm, lm_head, rotary, encode,
            val_docs, args, with_grad=False)
        print(f"[collect] depth={sa}: train={tr_h.shape[0]} "
              f"val={va_h.shape[0]} pairs ({time.time() - t0:.1f}s)")

        # victim boundary activations (no fine-tuning in E8: base model)
        hs = []
        with torch.no_grad():
            for ids in victim_ids:
                x = ids[:-1].unsqueeze(0).to(args.device)
                position_ids = torch.arange(x.shape[1],
                                            device=args.device).unsqueeze(0)
                hidden = embed(x)
                lk = make_layer_kwargs(rotary, hidden, position_ids, args)
                hs.append(run_layer_stack(head, hidden, lk)[0].float().cpu())
        h_star = torch.cat(hs)
        feat_sets = {
            "baseline_no_defense": h_star,
            "obfuscated": h_star @ W.cpu(),
            "breach_W_leaked": (h_star @ W.cpu()) @ W.cpu().T,
        }

        for seed in args.seeds:
            seed_all(args.seed + seed)
            dec = train_decoder(tr_h, tr_tok, va_h, va_tok, tr_h.shape[1],
                                vocab_size, args,
                                f"e8_d{sa}_seed{seed}")
            for fname, fx in feat_sets.items():
                top1, top5 = evaluate_decoder(dec, fx, victim_tok, args.device)
                results.append({"depth": sa, "setting": fname, "seed": seed,
                                "top1": top1, "top5": top5})
                print(f"[eval] depth={sa} {fname} seed={seed}: "
                      f"top-1={top1:.2f}% top-5={top5:.2f}%")
        for fname in feat_sets:
            t1s = [r["top1"] for r in results
                   if r["depth"] == sa and r["setting"] == fname]
            t5s = [r["top5"] for r in results
                   if r["depth"] == sa and r["setting"] == fname]
            m1, s1 = mean_std(t1s)
            m5, s5 = mean_std(t5s)
            summary.append({"depth": sa, "setting": fname,
                            "top1_mean": m1, "top1_std": s1,
                            "top5_mean": m5, "top5_std": s5,
                            "n_seeds": len(t1s)})

    out = {
        # measurement_kind is part of the evidence contract; without it an
        # artifact cannot be classified measured-vs-simulated automatically.
        "config": {"measurement_kind": "measured",
                   "model": "toy" if args.toy else args.model,
                   "n_layers": n_layers, "depths": args.depths,
                   "seeds": args.seeds, "victim_docs": args.victim_docs,
                   "seq_len": args.seq_len, "max_pairs": args.max_pairs,
                   "epochs": args.epochs, "gen_tokens": args.gen_tokens,
                   "dtype": args.dtype, "device": args.device,
                   "W_seed": args.seed, "W_ortho_err": ortho_err,
                   "quick": args.quick,
                   "val_frac": args.val_frac, "batch_size": args.batch_size,
                   "attn_impl": args.attn_impl,
                   "matmul_precision_requested": args.matmul_precision,
                   **numerics},
        "threat_model": "default: attacker has public text + public base "
                        "weights, NOT W; decoder trained on plain h* and "
                        "evaluated on h'=h*@W. 'breach_W_leaked' is a "
                        "MEASUREMENT of the insider-leak case (h'@W^T=h), "
                        "not the default assumption. Unsupervised adaptation "
                        "from unlabeled h' is out of scope (future work).",
        "interpretation": "obfuscated ~= random baseline => defense works "
                          "while W stays secret; breach ~= baseline_no_defense "
                          "=> security rests entirely on W's secrecy. Utility: "
                          "token_identity should be 1.0 (fp32 round-trip; bf16 "
                          "model cast may flip greedy ties — rerun with BOTH "
                          "--dtype fp32 and --matmul-precision highest to "
                          "verify exactness; --dtype fp32 alone still runs "
                          "TF32 matmuls and is not an exactness check).",
        "evidence_status": "supporting",
        "known_limitations": [
            "top1/top5 are exact-token identity under greedy decode; they say "
            "nothing about partial-sequence, embedding-level or sampled "
            "recovery, so obfuscated=0.0 is not a proof of non-invertibility.",
            "overhead_ms_per_matmul and its by-precision map are single "
            "unreplicated timings at one hidden dim and seq_len (both "
            "recorded). The gap between precision modes is smaller than the "
            "spread between runs, so they do not rank the modes.",
            "The defense arm assumes W stays secret; breach_W_leaked measures "
            "the insider-leak case and is not the default threat model.",
        ],
        "random_baseline_top1_pct": round(100.0 / vocab_size, 4),
        "provenance": provenance,
        "utility": utility,
        "overhead_ms_per_matmul": overhead_ms,
        "overhead_ms_per_matmul_by_precision": overhead_by_prec,
        "overhead_matmul_hidden_dim": hidden_dim,
        "overhead_matmul_seq_len": args.seq_len,
        "summary": summary,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    _write_training_status(state="done", result_file=args.output)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
