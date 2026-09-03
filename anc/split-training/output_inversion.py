#!/usr/bin/env python3
"""Output-side inversion attack: recovering the GENERATED answer.

Every privacy experiment so far attacks the INPUT-side boundary (activation
after the victim's first d local layers; trained_inversion.py). The OUTPUT
side is unmeasured: in split inference the cloud's middle stack returns an
activation to the trusted tail (last k local layers + norm + lm_head). At
decode step t the tensor crossing back toward the tail at that cloud/tail
interface encodes what the model is about to emit — so a semi-honest cloud
can try to recover the generated ANSWER, not just the prompt.

Attack: for tail depth k (boundary at layer L-k, i.e. features = output of
layer L-k-1), train the standard InversionDecoder MLP on

    feature: boundary hidden state at position t   (teacher-forced)
    label:   token at position t+1                  (the next emitted token)

collected from PUBLIC corpus docs through the PUBLIC base model — the cloud
can generate arbitrary text with the public head+middle and read its own
returning boundary, so these pairs are free. Evaluate top-1/top-5 on the
held-out victim docs (END-of-corpus convention, document-disjoint), seeds
[0,1,2], mean +/- std.

Obfuscated variant (does E8 protect the output side too?): the tail holds a
secret orthogonal W (e8_obfuscation.make_secret) and the returning boundary
crosses as h' = h @ W. Four settings per (k, seed):
  - baseline_no_defense:  decoder on plain h, eval on plain h.
  - obfuscated_passive:   decoder trained on plain public h, applied to h'
                          (the E8 default threat model; expect collapse).
  - obfuscated_retrained: decoder retrained on (h @ W, next-token) pairs.
                          This is NOT free for a passive attacker (it needs
                          labels for obfuscated features) but IS free for a
                          cloud that logs the emitted tokens of its own
                          serving traffic — the realistic semi-honest API
                          operator. W is invertible and the MLP learns
                          through it, so expect ~baseline: obfuscation buys
                          nothing once output-side pairs are labeled.
  - breach_W_leaked:      W leaked; h' @ W^T = h returns to baseline
                          (sanity control, mirrors e8_obfuscation).

Usage:
    python output_inversion.py --help        # works without torch
    python output_inversion.py --toy --quick # CPU machinery check
    python output_inversion.py --model <hf-model> --corpus-file <docs.txt> --output outinv.json
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
    from trained_inversion import (evaluate_decoder, make_provenance,
                                   mean_std, seed_all, train_decoder)
    from e8_obfuscation import make_secret
except ImportError:  # pragma: no cover - torch-less host
    TEXT_SAMPLES = []
    _write_training_status = lambda **k: None
    build_modules = make_layer_kwargs = run_layer_stack = None
    evaluate_decoder = make_provenance = mean_std = seed_all = None
    train_decoder = make_secret = None


def clamp_tail_depth(k, n_layers):
    """Tail depth k = layers from the END; boundary at layer L-k (features =
    output of layer L-k-1). Need k >= 1 (a tail exists) and k <= L-1 (at
    least one layer in front of the boundary). Same clamp+warn style as
    split_at."""
    kc = max(1, min(k, n_layers - 1))
    return kc


# (feature, next-token) pair collection on the PUBLIC BASE model. Unlike
# trained_inversion.collect_base_pairs (input-side: h[t] -> token[t]), the
# output side predicts the NEXT token: h[t] -> token[t+1], teacher-forced.
def collect_output_pairs(embed, prefix, rotary, encode, docs, args):
    feats, labels = [], []
    n_pairs = 0
    with torch.no_grad():
        for doc in docs:
            for block in encode([doc], args.seq_len):
                ids = block[:-1].unsqueeze(0).to(args.device)
                lab = block[1:]
                position_ids = torch.arange(ids.shape[1],
                                            device=args.device).unsqueeze(0)
                hidden = embed(ids)
                lk = make_layer_kwargs(rotary, hidden, position_ids, args)
                h = run_layer_stack(prefix, hidden, lk)
                feats.append(h[0].float().cpu())
                labels.append(lab.cpu())
                n_pairs += lab.shape[0]
                if n_pairs >= args.max_pairs:
                    break
            if n_pairs >= args.max_pairs:
                break
    return torch.cat(feats), torch.cat(labels)


def victim_output_feats(embed, prefix, rotary, encode, victim_docs, args):
    """Boundary features + next-token labels for the held-out victim docs
    (one block per doc, truncate not chunk-mix — trained_inversion conv.)."""
    feats, labels = [], []
    with torch.no_grad():
        for doc in victim_docs:
            blocks = encode([doc], args.seq_len)
            if not blocks:
                continue
            block = blocks[0]
            ids = block[:-1].unsqueeze(0).to(args.device)
            position_ids = torch.arange(ids.shape[1],
                                        device=args.device).unsqueeze(0)
            hidden = embed(ids)
            lk = make_layer_kwargs(rotary, hidden, position_ids, args)
            h = run_layer_stack(prefix, hidden, lk)
            feats.append(h[0].float().cpu())
            labels.append(block[1:])
    if not feats:
        raise ValueError("no victim doc long enough to yield a block")
    return torch.cat(feats), torch.cat(labels)


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
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8],
                    help="TAIL depths k, in units of LAYERS FROM THE END of "
                         "the model: the boundary attacked is at layer L-k "
                         "(features = output of layer L-k-1, what the cloud's "
                         "last layer hands back to the trusted tail)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--victim-docs", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of attack docs held out (document-disjoint) "
                         "for best-epoch selection")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--max-pairs", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="unused here (no fine-tuning); kept for convention parity")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device",
                    default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    ap.add_argument("--attn-impl", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="depth 2, 1 seed, 2 victim docs, 5 epochs, seq 16 "
                         "(<=5 min CPU on --toy)")
    ap.add_argument("--output", default="output_inversion_results.json")
    args = ap.parse_args()

    if torch is None or build_modules is None:
        ap.error("torch/transformers not installed; install them or run --help only")

    if args.quick:
        args.depths = [2]
        args.seeds = [0]
        args.victim_docs = 2
        args.epochs = 5
        args.seq_len = 16
        args.max_pairs = 2000

    seed_all(args.seed)
    _write_training_status(state="running", task="output_inversion",
                           depths=args.depths,
                           started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    embed, layers, norm, lm_head, rotary, encode = build_modules(args)
    n_layers = len(layers)
    vocab_size = (lm_head.weight.shape[0] if not args.toy
                  else embed.weight.shape[0])
    hidden_dim = embed.weight.shape[1]
    print(f"[model] {'toy' if args.toy else args.model}: {n_layers} layers, "
          f"hidden={hidden_dim}, vocab={vocab_size}, device={args.device}")

    # secret W (E8 output-side obfuscation)
    W = make_secret(hidden_dim, args.seed)  # fp32 CPU; the defense secret
    ortho_err = (W @ W.T - torch.eye(hidden_dim)).abs().max().item()
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
        raise ValueError(f"corpus too small: {len(docs)} docs; need victim-docs "
                         f"({args.victim_docs}) + at least 4 attack docs")
    victim_docs = docs[-args.victim_docs:]
    attack_docs = docs[:-args.victim_docs]
    provenance = make_provenance(
        args.corpus_file, corpus_source, len(docs),
        range(len(docs) - args.victim_docs, len(docs)), model_path=getattr(args, 'model', None))
    # document-disjoint val split from the attack pool
    n_val = max(1, int(round(args.val_frac * len(attack_docs))))
    val_docs, train_docs_pool = attack_docs[:n_val], attack_docs[n_val:]
    print(f"[data] {len(docs)} docs: {len(train_docs_pool)} attack-train, "
          f"{n_val} attack-val, {len(victim_docs)} victim (held out from end)")

    results = []
    summary = []

    # dumped after EVERY depth: full runs are long; a crash must not lose
    # completed cells (trained_inversion dump_out pattern)
    out = {
        "config": {"model": "toy" if args.toy else args.model,
                   "n_layers": n_layers, "depths_from_end": args.depths,
                   "seeds": args.seeds, "victim_docs": args.victim_docs,
                   "val_frac": args.val_frac, "seq_len": args.seq_len,
                   "max_pairs": args.max_pairs, "epochs": args.epochs,
                   "dtype": args.dtype, "device": args.device,
                   "W_seed": args.seed, "W_ortho_err": ortho_err,
                   "quick": args.quick},
        "threat_model": "semi-honest cloud in split inference observes the "
                        "RETURNING boundary activation at the cloud/tail "
                        "interface (output of layer L-k-1) at every decode "
                        "step; it trains an MLP next-token decoder on PUBLIC "
                        "text through the PUBLIC base model (teacher-forced; "
                        "document-disjoint; victim docs held out from the END "
                        "of the corpus). 'obfuscated_passive': tail applies "
                        "secret orthogonal W, attacker has no labels for h'=h@W. "
                        "'obfuscated_retrained': stronger cloud that logs the "
                        "emitted tokens of its own serving traffic and so can "
                        "label obfuscated features. 'breach_W_leaked': W "
                        "leaked (insider) — sanity control, must return to "
                        "baseline.",
        "interpretation": "baseline high => the GENERATED ANSWER leaks from "
                          "the returning boundary even though token ids never "
                          "leave the trusted tail (output-side leakage, not "
                          "just prompt leakage). obfuscated_passive ~random "
                          "while baseline is high => E8 must be applied on "
                          "BOTH wire directions (input AND output) — a "
                          "one-direction defense leaves the answer exposed. "
                          "obfuscated_retrained ~baseline => obfuscation buys "
                          "nothing against a cloud that can label its own "
                          "output-side traffic. breach ~= baseline => the "
                          "machinery is correct (W round-trip is exact).",
        "random_baseline_top1_pct": round(100.0 / vocab_size, 4),
        "provenance": provenance,
        "summary": summary,
        "results": results,
    }
    def dump_out():
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)

    for k in args.depths:
        kc = clamp_tail_depth(k, n_layers)
        if kc != k:
            print(f"[split] tail depth {k} clamped to {kc} ({n_layers} layers)")
        boundary = n_layers - kc  # boundary layer index; features = output of boundary-1
        prefix = torch.nn.ModuleList(list(layers[:boundary]))
        print(f"[split] tail depth k={kc}: boundary at layer {boundary} "
              f"(prefix 0..{boundary - 1}, tail {boundary}..{n_layers - 1})")

        # attack training pairs from the PUBLIC BASE model
        t0 = time.time()
        tr_h, tr_tok = collect_output_pairs(embed, prefix, rotary, encode,
                                            train_docs_pool, args)
        va_h, va_tok = collect_output_pairs(embed, prefix, rotary, encode,
                                            val_docs, args)
        vic_h, vic_tok = victim_output_feats(embed, prefix, rotary, encode,
                                             victim_docs, args)
        print(f"[collect] k={kc}: train={tr_h.shape[0]} val={va_h.shape[0]} "
              f"victim={vic_h.shape[0]} pairs ({time.time() - t0:.1f}s)")

        vic_prime = vic_h @ W  # what crosses the wire under E8 obfuscation

        for seed in args.seeds:
            seed_all(args.seed + seed)
            # decoder on plain h (the free public attacker)
            dec_plain = train_decoder(
                tr_h, tr_tok, va_h, va_tok, tr_h.shape[1], vocab_size, args,
                f"outinv_k{kc}_plain_seed{seed}")
            # decoder retrained on obfuscated features (cloud that logs the
            # emitted tokens of its own traffic can label h' too)
            seed_all(args.seed + seed)
            dec_obf = train_decoder(
                tr_h @ W, tr_tok, va_h @ W, va_tok, tr_h.shape[1], vocab_size,
                args, f"outinv_k{kc}_obf_seed{seed}")

            feat_sets = {
                "baseline_no_defense": (dec_plain, vic_h),
                "obfuscated_passive": (dec_plain, vic_prime),
                "obfuscated_retrained": (dec_obf, vic_prime),
                "breach_W_leaked": (dec_plain, vic_prime @ W.T),
            }
            for fname, (dec, fx) in feat_sets.items():
                top1, top5 = evaluate_decoder(dec, fx, vic_tok, args.device)
                results.append({"tail_depth": kc, "boundary_layer": boundary,
                                "setting": fname, "seed": seed,
                                "top1": top1, "top5": top5})
                print(f"[eval] k={kc} {fname} seed={seed}: "
                      f"top-1={top1:.2f}% top-5={top5:.2f}%")
            del dec_plain, dec_obf

        for fname in ("baseline_no_defense", "obfuscated_passive",
                      "obfuscated_retrained", "breach_W_leaked"):
            t1s = [r["top1"] for r in results
                   if r["tail_depth"] == kc and r["setting"] == fname]
            t5s = [r["top5"] for r in results
                   if r["tail_depth"] == kc and r["setting"] == fname]
            m1, s1 = mean_std(t1s)
            m5, s5 = mean_std(t5s)
            summary.append({"tail_depth": kc, "boundary_layer": boundary,
                            "setting": fname,
                            "top1_mean": m1, "top1_std": s1,
                            "top5_mean": m5, "top5_std": s5,
                            "n_seeds": len(t1s)})
            print(f"[summary] k={kc} {fname}: top-1={m1:.2f}+-{s1:.2f}% "
                  f"top-5={m5:.2f}+-{s5:.2f}%")

        dump_out()  # crash-safe: every completed depth is on disk

    dump_out()
    _write_training_status(state="done", result_file=args.output)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
