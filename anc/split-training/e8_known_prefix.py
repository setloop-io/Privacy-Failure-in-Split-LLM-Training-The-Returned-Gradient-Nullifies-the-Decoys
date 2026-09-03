#!/usr/bin/env python3
"""E8 known-prefix break: the REALISTIC Procrustes attack on secret W.

e8_robustness.py attack 3 measured an ORACLE break of the E8 obfuscation:
given K exact (h, hW) pairs the attacker recovers W by SVD of H^T (HW),
full break at K ~= hidden. That attacker is unrealistically strong — the
cloud never gets oracle pairs. This script measures the realistic version:

    User sessions almost always start with a KNOWN system prompt (or the
    cloud, being also a client of the service, can inject chosen prompts).
    Every session whose first S boundary tensors correspond to
    attacker-known text yields S exact pairs: the attacker computes h
    locally through the PUBLIC model head and observes hW on the wire.

Threat model: the attacker KNOWS the system prompt (very realistic) and the
model head is public; W is a fresh random orthogonal per deployment, hidden
from the attacker. Pairs accumulate across sessions until K >= hidden +
margin. We report, per (depth, wire_dtype, prefix length S):

  - sessions-to-full-break = ceil(hidden / S) (analytic) and the measured
    K (=> sessions) where victim-token recovery crosses 90% of the
    undefended baseline;
  - the recovery-vs-K curve: the standard MLP decoder (trained once on
    plain public h) applied to de-obfuscated features hW @ W_hat^T, with
    W_hat = polar factor of lstsq(H, HW) from the first K accumulated
    pairs, K swept on a log grid from S to 2*hidden;
  - the estimation error of W_hat vs K (relative Frobenius +
    misalignment angle) so the curve has a mechanistic x-axis;
  - wire-dtype mismatch: the attacker's h comes from ITS OWN copy in the
    model dtype (bf16 default) while the wire tensor crossed in
    --wire-dtype. split_trainer.py's wire default is "same as --dtype"
    (exact round-trip), so bf16/bf16 is the deployed case; fp16/fp32
    quantify how much cross-dtype noise degrades the break.

Usage:
    python e8_known_prefix.py --help        # works without torch
    python e8_known_prefix.py --toy --quick # CPU machinery check
    python e8_known_prefix.py --model <hf-model> --corpus-file <docs.txt> --output e8kp.json
"""

import argparse
import json
import math
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
    from trained_inversion import (collect_base_pairs, evaluate_decoder,
                                   make_provenance, mean_std, seed_all,
                                   split_at, train_decoder)
    from e8_obfuscation import make_secret
    from e8_robustness import (boundary_acts, crossing_k,
                               nested_repetition_stats, polar)
except ImportError:  # pragma: no cover - torch-less host
    TEXT_SAMPLES = []
    _write_training_status = lambda **k: None
    build_modules = make_layer_kwargs = run_layer_stack = None
    collect_base_pairs = evaluate_decoder = mean_std = seed_all = None
    split_at = train_decoder = make_secret = make_provenance = None
    boundary_acts = crossing_k = nested_repetition_stats = polar = None

WIRE_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def log_k_grid(s, hidden, n_points=6):
    """K sweep: log-spaced from S to 2*hidden, always including hidden."""
    lo, hi = max(1, s), 2 * hidden
    pts = {round(lo * (hi / lo) ** (j / (n_points - 1)))
           for j in range(n_points)}
    pts.add(hidden)
    return sorted(p for p in pts if lo <= p <= hi)


def w_estimation_error(w_hat, w_true):
    """Relative Frobenius error plus a misalignment angle: W_hat and W are
    both orthogonal, so tr(W_hat^T W) / H in [-1, 1] is 1 iff W_hat == W;
    arccos of it is a coarse mean rotation angle between the two frames."""
    rel_fro = ((w_hat - w_true).norm() / w_true.norm()).item()
    h = w_true.shape[0]
    cos = (torch.einsum("ij,ij->", w_hat, w_true) / h).clamp(-1.0, 1.0).item()
    return round(rel_fro, 6), round(math.degrees(math.acos(cos)), 4)


def prefix_pairs(embed, head, rotary, encode, docs, s_tokens, n_pairs_needed,
                 args):
    """Simulate sessions: session i starts with a known/chosen prefix of
    S tokens (first S tokens of public doc i, cycling). The attacker
    computes h through its OWN copy of the public head (model dtype) and
    observes hW on the wire. Returns attacker-side h [N, H] fp32 and the
    number of sessions used."""
    hs = []
    n = 0
    n_sessions = 0
    with torch.no_grad():
        for i in range(len(docs)):
            blocks = encode([docs[i]], max(args.seq_len, s_tokens))
            if not blocks:
                continue
            ids = blocks[0][:s_tokens].unsqueeze(0).to(args.device)
            if ids.shape[1] < s_tokens:
                continue
            position_ids = torch.arange(ids.shape[1],
                                        device=args.device).unsqueeze(0)
            hidden = embed(ids)
            lk = make_layer_kwargs(rotary, hidden, position_ids, args)
            h = run_layer_stack(head, hidden, lk)
            hs.append(h[0].float().cpu())
            n += ids.shape[1]
            n_sessions += 1
            if n >= n_pairs_needed:
                break
    if n < n_pairs_needed:
        raise ValueError(f"corpus yields only {n} prefix pairs "
                         f"({n_sessions} sessions x {s_tokens}); need "
                         f"{n_pairs_needed} — enlarge corpus or reduce K")
    return torch.cat(hs)[:n_pairs_needed], n_sessions


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.expanduser(
        "~/experiments/models/qwen3-0.6b"), help="HF model path (ignored with --toy)")
    ap.add_argument("--toy", action="store_true",
                    help="tiny random built-in model (CPU machinery check only; "
                         "depths clamp to the toy's 4 layers, hidden=64)")
    ap.add_argument("--corpus-file", default=None,
                    help="public text, one document per line; prefixes and "
                         "decoder training come from this pool; the LAST "
                         "--victim-docs documents are the victim eval docs")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 4, 8],
                    help="input-side split depths d (boundary after d local "
                         "layers, trained_inversion convention)")
    ap.add_argument("--prefix-lengths", type=int, nargs="+",
                    default=[32, 64, 128, 256],
                    help="known-prefix lengths S (tokens per session)")
    ap.add_argument("--wire-dtype", nargs="+", default=["bf16"],
                    choices=["bf16", "fp16", "fp32"],
                    help="dtype(s) of boundary tensors on the wire; the "
                         "attacker's copy computes in --dtype (split_trainer "
                         "default: wire matches model dtype => bf16)")
    ap.add_argument("--k-points", type=int, default=6,
                    help="log-grid points per recovery-vs-K curve")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--solve-seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="independent matched-pair order/subsample seeds; W_hat "
                         "is recomputed for every solve seed and decoder seeds "
                         "remain nested within each solve")
    ap.add_argument("--victim-docs", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--max-pairs", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device",
                    default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    ap.add_argument("--attn-impl", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="depth 1, 1 seed, S={32}, wire bf16, 2 victim docs, "
                         "5 epochs, seq 32 (<=5 min CPU on --toy)")
    ap.add_argument("--output", default="e8_known_prefix.json")
    args = ap.parse_args()

    if torch is None or build_modules is None:
        ap.error("torch/transformers not installed; install them or run --help only")

    if args.quick:
        args.depths = [1]
        args.prefix_lengths = [32]
        args.wire_dtype = ["bf16"]
        args.seeds = [0]
        args.solve_seeds = [0, 1]
        args.victim_docs = 2
        args.epochs = 5
        args.seq_len = 32
        args.max_pairs = 2000
        args.k_points = 4

    seed_all(args.seed)
    _write_training_status(state="running", task="e8_known_prefix",
                           depths=args.depths,
                           prefix_lengths=args.prefix_lengths,
                           started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    embed, layers, norm, lm_head, rotary, encode = build_modules(args)
    n_layers = len(layers)
    vocab_size = (lm_head.weight.shape[0] if not args.toy
                  else embed.weight.shape[0])
    hidden_dim = embed.weight.shape[1]
    random_top1 = round(100.0 / vocab_size, 4)
    print(f"[model] {'toy' if args.toy else args.model}: {n_layers} layers, "
          f"hidden={hidden_dim}, vocab={vocab_size}, device={args.device}")

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

    results = []
    summary = []

    out = {
        "config": {"model": "toy" if args.toy else args.model,
                   "n_layers": n_layers, "depths": args.depths,
                   "prefix_lengths": args.prefix_lengths,
                   "wire_dtype": args.wire_dtype, "k_points": args.k_points,
                   "seeds": args.seeds, "solve_seeds": args.solve_seeds,
                   "victim_docs": args.victim_docs,
                   "seq_len": args.seq_len, "max_pairs": args.max_pairs,
                   "epochs": args.epochs, "model_dtype": args.dtype,
                   "device": args.device, "W_seed": args.seed,
                   "quick": args.quick},
        "threat_model": "semi-honest cloud that KNOWS the system prompt "
                        "(very realistic: sessions start with a fixed known "
                        "prefix, or the cloud is itself a client and injects "
                        "chosen prompts) and whose model head is PUBLIC. Each "
                        "session's first S boundary tensors give S exact "
                        "(h, hW) pairs; W is a fresh random orthogonal per "
                        "deployment, hidden from the attacker. The attacker "
                        "computes h with its own copy in the model dtype "
                        f"({args.dtype}) while the wire crosses in "
                        "--wire-dtype. No oracle, no victim labels.",
        "interpretation": "recovery-vs-K flat at random until K ~ hidden, "
                          "then a sharp rise to the undefended baseline => "
                          "the realistic known-prefix attack breaks E8 with "
                          "ceil(hidden/S) sessions, no oracle needed — "
                          "per-session W rotation is MANDATORY, and even it "
                          "only helps if S < hidden. wire_dtype != model "
                          "dtype degrading the curve => cast noise is a "
                          "(weak) free defense. W_hat rel-Frobenius -> 0 as "
                          "K -> hidden is the mechanistic cause.",
        "random_baseline_top1_pct": random_top1,
        "provenance": provenance,
        "summary": summary,
        "results": results,
    }
    def dump_out():
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)

    for depth in args.depths:
        head, middle, tail, sa, ra = split_at(layers, depth, n_layers)
        if sa != depth:
            print(f"[split] depth {depth} clamped to sa={sa} ({n_layers} layers)")
        # fresh secret W per deployment (hidden from the attacker)
        W = make_secret(hidden_dim, args.seed).double()

        # decoder on plain public h (trained once per depth/seed)
        t0 = time.time()
        tr_h, _, tr_tok = collect_base_pairs(
            embed, head, middle, tail, norm, lm_head, rotary, encode,
            train_docs_pool, args, with_grad=False)
        va_h, _, va_tok = collect_base_pairs(
            embed, head, middle, tail, norm, lm_head, rotary, encode,
            val_docs, args, with_grad=False)
        h_star = boundary_acts(embed, head, rotary, victim_ids, args)
        print(f"[collect] depth={sa}: train={tr_h.shape[0]} val={va_h.shape[0]} "
              f"victim-positions={h_star.shape[0]} ({time.time() - t0:.1f}s)")

        decoders = {}
        for seed in args.seeds:
            seed_all(args.seed + seed)
            decoders[seed] = train_decoder(
                tr_h, tr_tok, va_h, va_tok, tr_h.shape[1], vocab_size, args,
                f"e8kp_d{sa}_seed{seed}")
        base_t1s, base_t5s = zip(*[evaluate_decoder(
            decoders[s], h_star, victim_tok, args.device) for s in args.seeds])
        bm1, bs1 = mean_std(list(base_t1s))
        bm5, bs5 = mean_std(list(base_t5s))
        summary.append({"depth": sa, "setting": "baseline_no_defense",
                        "top1_mean": bm1, "top1_std": bs1,
                        "top5_mean": bm5, "top5_std": bs5,
                        "n_seeds": len(base_t1s)})
        print(f"[eval] depth={sa} baseline_no_defense: "
              f"top-1={bm1:.2f}+-{bs1:.2f}%")

        for wd_name in args.wire_dtype:
            wire_dtype = getattr(torch, WIRE_DTYPES[wd_name])
            # victim traffic as seen on the wire, cast at the seam
            h_prime_vic = (h_star.double() @ W).to(wire_dtype).double()

            for s_tok in args.prefix_lengths:
                s_eff = min(s_tok, max(args.seq_len, s_tok))
                k_grid = log_k_grid(s_eff, hidden_dim, args.k_points)
                k_max = k_grid[-1]
                t0 = time.time()
                h_att, n_sess = prefix_pairs(
                    embed, head, rotary, encode, train_docs_pool, s_eff,
                    k_max, args)
                h_att = h_att.double()  # attacker-side h (model-dtype compute, fp64 for lstsq)
                # wire side: local node's hW, cast to the wire dtype
                h_wire = (h_att @ W).to(wire_dtype).double()
                print(f"[pairs] depth={sa} wire={wd_name} S={s_eff}: "
                      f"{h_att.shape[0]} pairs from {n_sess} sessions "
                      f"({time.time() - t0:.1f}s); K grid {k_grid}")

                curve = []
                for K in k_grid:
                    solve_repetitions = []
                    rels, angles = [], []
                    for solve_seed in args.solve_seeds:
                        g = torch.Generator().manual_seed(
                            args.seed + 100000 + solve_seed)
                        order = torch.randperm(h_att.shape[0], generator=g)
                        hh = h_att[order[:K]]
                        hw = h_wire[order[:K]]
                        sol = torch.linalg.lstsq(hh, hw)
                        w_hat = polar(sol.solution)
                        rel_fro, angle = w_estimation_error(w_hat, W)
                        h_rec = (h_prime_vic @ w_hat.T).float()
                        scores = [evaluate_decoder(
                            decoders[s], h_rec, victim_tok, args.device)
                            for s in args.seeds]
                        solve_repetitions.append({
                            "solve_seed": solve_seed,
                            "decoder_seeds": list(args.seeds),
                            "top1": [x[0] for x in scores],
                            "top5": [x[1] for x in scores],
                            "W_rel_fro": rel_fro,
                            "W_misalign_deg": angle,
                        })
                        rels.append(rel_fro); angles.append(angle)
                    top1_var = nested_repetition_stats(
                        solve_repetitions, "top1")
                    top5_var = nested_repetition_stats(
                        solve_repetitions, "top5")
                    # the shared helper names the outer unit "probe";
                    # rename to solve terminology for this harness
                    for d in (top1_var, top5_var):
                        d["between_solve_std"] = d.pop("between_probe_std")
                        d["n_solve_repetitions"] = d.pop("n_probe_repetitions")
                    m1 = top1_var["grand_mean"]
                    m5 = top5_var["grand_mean"]
                    entry = {"depth": sa, "wire_dtype": wd_name,
                             "prefix_len": s_eff, "K": K,
                             "sessions": math.ceil(K / s_eff),
                             "W_rel_fro_mean": round(sum(rels) / len(rels), 6),
                             "W_misalign_deg_mean": round(sum(angles) / len(angles), 4),
                             "top1_mean": m1,
                             "top5_mean": m5,
                             "top1_variance_decomposition": top1_var,
                             "top5_variance_decomposition": top5_var,
                             "solve_repetitions": solve_repetitions}
                    results.append(entry)
                    curve.append(entry)
                    print(f"[eval] depth={sa} wire={wd_name} S={s_eff} K={K} "
                          f"({entry['sessions']} sessions): top-1={m1:.2f}% "
                          f"|W_hat-W|/|W|={entry['W_rel_fro_mean']:.4f}")

                # sessions-to-full-break: analytic, and measured (K where
                # top-1 reaches 90% of the undefended baseline)
                analytic = math.ceil(hidden_dim / s_eff)
                thr = 0.9 * bm1
                crossing = crossing_k(curve, threshold=thr)
                k_cross = crossing["k50_interpolated"]
                summary.append({
                    "depth": sa, "setting": "known_prefix_break",
                    "wire_dtype": wd_name, "prefix_len": s_eff,
                    "sessions_to_full_break_analytic": analytic,
                    "sessions_to_full_break_interpolated": (
                        math.ceil(k_cross / s_eff) if k_cross is not None else None),
                    "K_at_90pct_interpolated": k_cross,
                    "K_at_90pct_crossing": crossing,
                    "baseline_top1_mean": bm1,
                    "final_K": curve[-1]["K"],
                    "final_top1_mean": curve[-1]["top1_mean"],
                    "final_W_rel_fro_mean": curve[-1]["W_rel_fro_mean"]})
                print(f"[summary] depth={sa} wire={wd_name} S={s_eff}: "
                      f"sessions-to-break analytic={analytic} "
                      f"interpolated={math.ceil(k_cross / s_eff) if k_cross is not None else crossing['k50_method']}")
                dump_out()  # crash-safe per completed curve

        dump_out()

    dump_out()
    _write_training_status(state="done", result_file=args.output)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
