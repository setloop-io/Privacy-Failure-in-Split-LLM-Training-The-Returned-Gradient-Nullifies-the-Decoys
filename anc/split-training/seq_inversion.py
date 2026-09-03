#!/usr/bin/env python3
"""A1: sequence-aware inversion attack on split boundaries.

trained_inversion.py's attacker is POSITION-INDEPENDENT: an MLP maps each
boundary activation h*[i] -> token[i] in isolation. Natural text is highly
redundant, so a sequence-aware attacker should be substantially stronger:
once a few anchor tokens are right, a language-model prior disambiguates the
rest. This script quantifies that uplift.

Two-stage decode (the required path — cheap and strong):
  1. Per-position MLP (InversionDecoder, the same architecture as
     trained_inversion.py)
     trained on (h*, token) pairs collected from PUBLIC text through the
     PUBLIC base model's first d layers. Gives top-k candidates + log-probs
     per position (a lattice).
  2. SEQUENCE RESCORING: beam-search the lattice (width --beam) for the top
     candidate sequences under the MLP scores, then rescore each full
     candidate with the base model ITSELF as the LM prior
     (sum of next-token log-probs of the candidate token sequence) and pick
     argmax( mlp_logprob + --lm-weight * lm_logprob ). The base model is
     public, so this scorer is free to the attacker.

Optional --joint: a small attention decoder (2-layer transformer encoder
over the boundary-activation sequence, positions cross-attending) trained
end-to-end on (h*_seq, token_seq) pairs, decoded with the same top-k + LM
rescore. Ablates "learned sequence context" vs "MLP + LM prior".

Metrics (per victim sequence, then averaged; vs the position-independent
MLP-argmax baseline computed in the SAME run — the uplift IS the result):
  - token top-1 exact accuracy
  - exact-match span rate: fraction of positions inside correct contiguous
    runs of length >= --min-span
  - longest correct subsequence: longest contiguous correct run / seq_len
  - full-sequence exact-match rate

Honest scope: attack training pairs come from the public base model; victim
docs are the LAST --victim-docs docs of the corpus and never enter the
attack training/val pools (document-disjoint). 3 seeds by default.

Usage:
    python seq_inversion.py --help        # works without torch
    python seq_inversion.py --toy --quick # CPU machinery check
    python seq_inversion.py --model <hf-model> --corpus-file <docs.txt> --depths 1 4 8
"""

import argparse
import json
import os
import sys
import time

# Guarded heavy imports: `--help` must work on torch-less hosts.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from split_trainer import (TEXT_SAMPLES, _write_training_status,
                               build_modules, make_layer_kwargs,
                               run_layer_stack)
except ImportError:  # pragma: no cover - torch-less
    TEXT_SAMPLES = []
    _write_training_status = lambda **k: None
    build_modules = make_layer_kwargs = run_layer_stack = None

try:
    # Reuse the position-independent attacker's collect/train machinery
    # verbatim (same decoder architecture, same pair collection).
    from trained_inversion import (InversionDecoder, collect_base_pairs,
                                   evaluate_decoder, make_provenance,
                                   mean_std, seed_all, split_at,
                                   train_decoder)
except ImportError:  # pragma: no cover - torch-less
    InversionDecoder = None
    collect_base_pairs = evaluate_decoder = seed_all = split_at = None
    train_decoder = None
    mean_std = make_provenance = None


# Optional joint attention decoder: 2-layer transformer encoder over the
# boundary-activation sequence (positions cross-attend), per-position head.
if nn is not None:

    class SeqAttnDecoder(nn.Module):
        def __init__(self, in_dim, vocab_size, d_model=256, nheads=4,
                     n_layers=2, max_len=1024):
            super().__init__()
            self.proj = nn.Linear(in_dim, d_model)
            self.pos = nn.Embedding(max_len, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model, nheads, 4 * d_model, batch_first=True,
                norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(enc_layer, n_layers)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, h):  # h: [B, L, in_dim] -> logits [B, L, vocab]
            x = self.proj(h) + self.pos.weight[: h.shape[1]].unsqueeze(0)
            return self.head(self.enc(x))


# Per-block boundary collection (victims + joint training). Same forward path
# as trained_inversion.collect_base_pairs but keeps the block structure
# instead of flattening to one (feature, token) pool.
def collect_boundary_sequences(embed, head, rotary, encode, docs, args,
                               max_blocks=None):
    h_seqs, tok_seqs = [], []
    for doc in docs:
        for block in encode([doc], args.seq_len):
            ids = block[:-1].unsqueeze(0).to(args.device)
            position_ids = torch.arange(ids.shape[1],
                                        device=args.device).unsqueeze(0)
            with torch.no_grad():
                hidden = embed(ids)
                lk = make_layer_kwargs(rotary, hidden, position_ids, args)
                boundary = run_layer_stack(head, hidden, lk)
            h_seqs.append(boundary[0].float().cpu())
            tok_seqs.append(ids[0].cpu())
            if max_blocks is not None and len(h_seqs) >= max_blocks:
                return h_seqs, tok_seqs
    return h_seqs, tok_seqs


# LM prior: score a full candidate token sequence with the base model itself
# (sum of next-token log-probs). The base checkpoint is public knowledge.
def lm_sequence_logprob(ids_1d, embed, layers, norm, lm_head, rotary, args):
    ids = ids_1d.unsqueeze(0).to(args.device)
    position_ids = torch.arange(ids.shape[1], device=args.device).unsqueeze(0)
    with torch.no_grad():
        hidden = embed(ids)
        lk = make_layer_kwargs(rotary, hidden, position_ids, args)
        out = run_layer_stack(layers, hidden, lk)
        logits = lm_head(norm(out)).float()
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    return logp.gather(-1, ids[:, 1:].unsqueeze(-1)).sum().item()


# Two-stage decode of ONE boundary-activation sequence.
def decode_sequence(decoder, h_seq, embed, layers, norm, lm_head, rotary,
                    args):
    """Returns (baseline_ids, rescored_ids, n_lm_calls). baseline = per-position
    argmax (position-independent attacker); rescored = beam over the top-k
    lattice + LM rescore."""
    with torch.no_grad():
        logits = decoder(h_seq.float().to(args.device))
        logp = F.log_softmax(logits.float(), dim=-1)  # [L, V]
    baseline = logp.argmax(dim=-1).cpu()
    k = min(args.top_k, logp.shape[-1])
    topv, topi = logp.topk(k, dim=-1)  # [L, k]
    topv, topi = topv.cpu(), topi.cpu()
    L = topi.shape[0]

    # Beam over the lattice scored by MLP log-probs. Scores are additive and
    # position-independent, so this enumerates the top-`beam` joint combos.
    beam = [([], 0.0)]
    for pos in range(L):
        cand = []
        for seq_tok, score in beam:
            for j in range(k):
                cand.append((seq_tok + [topi[pos, j].item()],
                             score + topv[pos, j].item()))
        cand.sort(key=lambda x: x[1], reverse=True)
        beam = cand[: args.beam]

    # Rescore full candidates with the LM prior; pick argmax combined.
    best_ids, best_score = None, float("-inf")
    for seq_tok, mlp_score in beam:
        ids = torch.tensor(seq_tok, dtype=torch.long)
        lm_score = lm_sequence_logprob(ids, embed, layers, norm, lm_head,
                                       rotary, args)
        score = mlp_score + args.lm_weight * lm_score
        if score > best_score:
            best_score, best_ids = score, ids
    return baseline, best_ids, len(beam)


# Sequence-level metrics.
def correct_runs(pred, gold):
    """Lengths of contiguous correct runs (per-position exact match)."""
    runs, cur = [], 0
    for p, g in zip(pred.tolist(), gold.tolist()):
        if p == g:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def sequence_metrics(pred, gold, min_span):
    L = gold.shape[0]
    runs = correct_runs(pred, gold)
    return {
        "token_acc": 100.0 * (pred == gold).float().mean().item(),
        "span_acc": 100.0 * sum(r for r in runs if r >= min_span) / L,
        "longest_run_frac": (max(runs) / L) if runs else 0.0,
        "exact": 1.0 if (pred == gold).all().item() else 0.0,
    }


def evaluate_attacker(decoder, h_seqs, tok_seqs, embed, layers, norm,
                      lm_head, rotary, args):
    base_agg, res_agg = [], []
    for h_seq, tok in zip(h_seqs, tok_seqs):
        baseline, rescored, _ = decode_sequence(
            decoder, h_seq, embed, layers, norm, lm_head, rotary, args)
        base_agg.append(sequence_metrics(baseline, tok, args.min_span))
        res_agg.append(sequence_metrics(rescored, tok, args.min_span))
    def agg(ms):
        return {k: round(sum(m[k] for m in ms) / max(len(ms), 1), 4)
                for k in ms[0]}
    return agg(base_agg), agg(res_agg)


# Joint attention-decoder training (optional --joint path).
def train_joint_decoder(h_seqs, tok_seqs, val_h, val_tok, in_dim, vocab_size,
                        args, tag):
    device = args.device
    decoder = SeqAttnDecoder(in_dim, vocab_size).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()
    n = len(h_seqs)
    best_val, best_state = -1.0, None
    for epoch in range(args.epochs):
        decoder.train()
        perm = torch.randperm(n)
        tot = 0.0
        nb = 0
        # Full-vocab sequence logits scale as B * L * V (>100 GiB at the MLP
        # batch size for Qwen's 151,936-token vocab), so the joint attacker
        # has its own sequence batch size. Optimizer batching only; examples,
        # labels, architecture, and evaluation protocol are unchanged.
        for i in range(0, n, args.joint_batch_size):
            idx = perm[i: i + args.joint_batch_size]
            bx = torch.stack([h_seqs[j] for j in idx]).float().to(device)
            by = torch.stack([tok_seqs[j] for j in idx]).to(device)
            logits = decoder(bx)
            loss = crit(logits.reshape(-1, vocab_size), by.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        # val top-1 (per-position argmax) for best-epoch selection
        decoder.eval()
        correct = total = 0
        with torch.no_grad():
            for i in range(0, len(val_h), args.joint_batch_size):
                vx = torch.stack(val_h[i:i + args.joint_batch_size]).float().to(device)
                vy = torch.stack(val_tok[i:i + args.joint_batch_size]).to(device)
                preds = decoder(vx).argmax(dim=-1)
                correct += (preds == vy).sum().item()
                total += vy.numel()
        vtop1 = round(100.0 * correct / max(total, 1), 2)
        _write_training_status(phase="train", state="running", run_id=tag,
                               epoch=epoch + 1, epochs=args.epochs,
                               loss=round(tot / max(nb, 1), 4), top1=vtop1,
                               metric_name="val_top1", metric_value=vtop1)
        if vtop1 > best_val:
            best_val = vtop1
            best_state = {k: v.detach().clone()
                          for k, v in decoder.state_dict().items()}
    if best_state is not None:
        decoder.load_state_dict(best_state)
    return decoder


def joint_topk_logits(decoder, h_seq, args):
    """Per-position logits so the joint decoder can share decode_sequence's
    lattice machinery (wraps [L,H] -> [L,V])."""
    with torch.no_grad():
        return decoder(h_seq.float().unsqueeze(0).to(args.device))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.expanduser(
        "~/experiments/models/qwen3-0.6b"), help="HF model path (ignored with --toy)")
    ap.add_argument("--toy", action="store_true",
                    help="tiny random built-in model (CPU machinery check only; "
                         "depths are clamped to the toy's 4 layers)")
    ap.add_argument("--corpus-file", default=None,
                    help="public attack-training text, one document per line; "
                         "the LAST --victim-docs documents are held out as the "
                         "victim's private data and never enter attack training")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 4, 8],
                    help="split depths (local layers 0..d)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--victim-docs", type=int, default=8,
                    help="documents held out from the END of the corpus as the "
                         "victim's private data")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of attack docs held out (document-disjoint) "
                         "for best-epoch selection")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--max-pairs", type=int, default=20000,
                    help="cap on (feature, token) pairs per attack split")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--joint-batch-size", type=int, default=8,
                    help="sequence batches for the joint full-vocabulary "
                         "decoder; kept separate from the position-wise MLP "
                         "batch to bound B*L*V logits memory")
    ap.add_argument("--top-k", type=int, default=8,
                    help="lattice width: candidates per position")
    ap.add_argument("--beam", type=int, default=8,
                    help="beam width for lattice search (4-8 recommended)")
    ap.add_argument("--lm-weight", type=float, default=1.0,
                    help="weight of the base-model LM prior in rescoring")
    ap.add_argument("--min-span", type=int, default=3,
                    help="minimum correct-run length counted in span_acc")
    ap.add_argument("--joint", action="store_true",
                    help="also train the 2-layer attention decoder ablation")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device",
                    default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    ap.add_argument("--attn-impl", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="depth 1, 1 seed, 2 victim docs, 5 epochs, seq 16 "
                         "(<=5 min CPU on --toy)")
    ap.add_argument("--output", default="seq_inversion_results.json")
    args = ap.parse_args()

    if args.joint_batch_size < 1:
        ap.error("--joint-batch-size must be at least 1")

    if torch is None or build_modules is None or collect_base_pairs is None:
        ap.error("torch/transformers not installed; install them or run --help only")

    if args.quick:
        args.depths = [1]
        args.seeds = [0]
        args.victim_docs = 2
        args.epochs = 5
        args.seq_len = 16
        args.max_pairs = 2000
        args.beam = 4

    seed_all(args.seed)
    _write_training_status(state="running", task="seq_inversion",
                           depths=args.depths, joint=args.joint,
                           started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    embed, layers, norm, lm_head, rotary, encode = build_modules(args)
    n_layers = len(layers)
    vocab_size = (lm_head.weight.shape[0] if not args.toy
                  else embed.weight.shape[0])
    hidden_dim = embed.weight.shape[1]
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
        raise ValueError(f"corpus too small: {len(docs)} docs; need victim-docs "
                         f"({args.victim_docs}) + at least 4 attack docs")
    victim_docs = docs[-args.victim_docs:]
    attack_docs = docs[:-args.victim_docs]
    provenance = make_provenance(
        args.corpus_file, corpus_source, len(docs),
        range(len(docs) - args.victim_docs, len(docs)), model_path=getattr(args, 'model', None))
    n_val = max(1, int(round(args.val_frac * len(attack_docs))))
    val_docs, train_docs_pool = attack_docs[:n_val], attack_docs[n_val:]
    print(f"[data] {len(docs)} docs: {len(train_docs_pool)} attack-train, "
          f"{n_val} attack-val, {len(victim_docs)} victim (held out from end)")

    results = []
    summary = []

    for depth in args.depths:
        head, _, _, sa, _ = split_at(layers, depth, n_layers)
        if sa != depth:
            print(f"[split] depth {depth} clamped to sa={sa} ({n_layers} layers)")

        # MLP attack-training pairs from the PUBLIC BASE model
        t0 = time.time()
        print(f"[collect] depth={sa}: base-model pairs...")
        tr_h, _, tr_tok = collect_base_pairs(
            embed, head, None, None, norm, lm_head, rotary, encode,
            train_docs_pool, args, with_grad=False)
        va_h, _, va_tok = collect_base_pairs(
            embed, head, None, None, norm, lm_head, rotary, encode,
            val_docs, args, with_grad=False)
        print(f"[collect] train={tr_h.shape[0]} val={va_h.shape[0]} pairs "
              f"({time.time() - t0:.1f}s)")

        # victim boundary sequences (per block, base model)
        v_h, v_tok = collect_boundary_sequences(
            embed, head, rotary, encode, victim_docs, args)
        print(f"[data] {len(v_h)} victim sequences x {args.seq_len} positions")

        # optional joint-decoder training sequences (per block)
        j_tr_h = j_tr_tok = j_va_h = j_va_tok = None
        if args.joint:
            j_tr_h, j_tr_tok = collect_boundary_sequences(
                embed, head, rotary, encode, train_docs_pool, args,
                max_blocks=max(4, args.max_pairs // args.seq_len))
            j_va_h, j_va_tok = collect_boundary_sequences(
                embed, head, rotary, encode, val_docs, args)

        attackers = {}  # (kind, seed) -> (decoder, logit_fn)
        for seed in args.seeds:
            seed_all(args.seed + seed)
            tag = f"seq_inv_mlp_d{sa}_seed{seed}"
            dec = train_decoder(tr_h, tr_tok, va_h, va_tok,
                                hidden_dim, vocab_size, args, tag)
            vtop1, vtop5 = evaluate_decoder(dec, va_h, va_tok, args.device)
            print(f"[attack] depth={sa} mlp seed={seed}: val top-1="
                  f"{vtop1:.2f}% top-5={vtop5:.2f}%")
            results.append({"phase": "attack_train", "depth": sa,
                            "attacker": "mlp", "seed": seed,
                            "val_top1": vtop1, "val_top5": vtop5})
            attackers[("mlp", seed)] = (
                dec, lambda h, d=dec: d(h.float().to(args.device)))

            if args.joint:
                seed_all(args.seed + seed)
                jtag = f"seq_inv_joint_d{sa}_seed{seed}"
                jdec = train_joint_decoder(j_tr_h, j_tr_tok, j_va_h, j_va_tok,
                                           hidden_dim, vocab_size, args, jtag)
                attackers[("joint", seed)] = (
                    jdec, lambda h, d=jdec: joint_topk_logits(d, h, args))

        # victim evaluation: baseline vs LM-rescored, per seed
        for kind in (["mlp", "joint"] if args.joint else ["mlp"]):
            per_seed = []
            for seed in args.seeds:
                _, logit_fn = attackers[(kind, seed)]

                # decode_sequence expects a module; pass a shim with __call__
                class _Shim:
                    def __call__(self, h):
                        return logit_fn(h)
                base_m, res_m = evaluate_attacker(
                    _Shim(), v_h, v_tok, embed, layers, norm, lm_head,
                    rotary, args)
                uplift = {k: round(res_m[k] - base_m[k], 4)
                          for k in base_m}
                per_seed.append((base_m, res_m, uplift))
                results.append({"phase": "attack_eval", "depth": sa,
                                "attacker": kind, "seed": seed,
                                "baseline": base_m, "rescored": res_m,
                                "uplift": uplift})
                print(f"[eval] depth={sa} {kind} seed={seed}: token_acc "
                      f"{base_m['token_acc']:.2f} -> {res_m['token_acc']:.2f}% "
                      f"(+{uplift['token_acc']:.2f}), longest_run "
                      f"{base_m['longest_run_frac']:.3f} -> "
                      f"{res_m['longest_run_frac']:.3f}")

            cell = {"depth": sa, "attacker": kind, "n_seeds": len(per_seed)}
            for name, idx in [("baseline", 0), ("rescored", 1), ("uplift", 2)]:
                for key in per_seed[0][0]:
                    m, s = mean_std([p[idx][key] for p in per_seed])
                    cell[f"{name}_{key}_mean"] = m
                    cell[f"{name}_{key}_std"] = s
            summary.append(cell)
            print(f"[summary] depth={sa} {kind}: token_acc "
                  f"{cell['baseline_token_acc_mean']:.2f} -> "
                  f"{cell['rescored_token_acc_mean']:.2f}% "
                  f"(uplift {cell['uplift_token_acc_mean']:+.2f}+-"
                  f"{cell['uplift_token_acc_std']:.2f})")

    out = {
        "config": {"model": "toy" if args.toy else args.model,
                   "n_layers": n_layers, "depths": args.depths,
                   "seeds": args.seeds, "victim_docs": args.victim_docs,
                   "val_frac": args.val_frac, "seq_len": args.seq_len,
                   "max_pairs": args.max_pairs, "epochs": args.epochs,
                   "batch_size": args.batch_size,
                   "joint_batch_size": args.joint_batch_size,
                   "top_k": args.top_k, "beam": args.beam,
                   "lm_weight": args.lm_weight, "min_span": args.min_span,
                   "joint": args.joint, "dtype": args.dtype,
                   "device": args.device, "quick": args.quick},
        "threat_model": "semi-honest cloud observes the boundary activation "
                        "sequence h* per microbatch; attacker trains a "
                        "per-position MLP on PUBLIC text through the PUBLIC "
                        "base model (document-disjoint; victim docs held out "
                        "from the END of the corpus) and rescores the top-k "
                        "lattice with the base model itself as LM prior",
        "interpretation": "uplift = sequence-aware (top-k lattice + beam + "
                          "base-model LM rescore) minus position-independent "
                          "MLP argmax, same decoder and same data. Positive "
                          "uplift quantifies how much sequence context adds "
                          "to the trained-inversion attack.",
        "random_baseline_top1_pct": round(100.0 / vocab_size, 4),
        "provenance": provenance,
        "summary": summary,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    _write_training_status(state="done", result_file=args.output)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
