#!/usr/bin/env python3
"""E7 — recovery vs amount of PRIVATE local-stage fine-tuning.

Follow-up to trained_inversion.py. That experiment showed the public-weights
trained MLP attacker recovers 68-76% of tokens at split depths 1-8, FLAT
across fine-tune steps 0->100 — i.e., ordinary fine-tuning drift did not
protect. E7 asks the sharper question: how much PRIVATE fine-tuning of the
local stage is needed before the public-weights attacker fails?

Setup:
  - Victim: split depth d in {1, 4, 8} (local layers 0..d). Local stage =
    embed_tokens + layers 0..d + tail layers + norm + lm_head.
  - Private adaptation: fine-tune ONLY the local stage (middle frozen) for
    T in {0, 10, 100, 1000} steps on a private shard (the last 20% of
    --corpus-file documents, held out from all attack data), AdamW lr 1e-5
    (gentle; the point is weight DRIFT, not task learning), seq 256.
  - Base attacker: InversionDecoder MLP trained ONCE on (boundary h*, token)
    pairs from PUBLIC text through the PUBLIC base model (weights never
    adapted), applied to the victim's boundary activations after T steps.
  - Adaptive attacker (upper bound): the same decoder RETRAINED on pairs
    collected through the T-step-adapted local stage (attacker knows the
    adapted weights). Boundary h* depends only on embed + head, so adaptive
    pair collection needs only the adapted local head.

The headline is the curve: recovery(T) per depth, base vs adaptive, with a
held-out utility loss per checkpoint (attack-pool docs never fine-tuned on)
so privacy gains can be read against utility cost.

Honest-scope notes: per-position h*[i] -> token[i] prediction (same
convention as trained_inversion.py); private FT uses the victim's own
private shard, so capture docs are the LAST --victim-docs of that shard
(never trained on); seeds drive decoder init/shuffle only, private FT is
deterministic per depth.

Usage:
    python e7_private_ft.py --help        # works without torch
    python e7_private_ft.py --toy --quick # CPU machinery check
    python e7_private_ft.py --model <hf-model> --corpus-file <docs.txt> --output e7.json
"""

import argparse
import json
import os
import sys
import time

# Guarded heavy imports: `--help` must work on torch-less hosts.
try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from split_trainer import (TEXT_SAMPLES, _write_training_status,
                               build_modules, make_layer_kwargs,
                               run_layer_stack, unique_params)
    from trained_inversion import (collect_base_pairs, evaluate_decoder,
                                   make_provenance, mean_std, seed_all,
                                   split_at, train_decoder)
except ImportError:  # pragma: no cover - torch-less host
    TEXT_SAMPLES = []
    _write_training_status = lambda **k: None
    build_modules = make_layer_kwargs = run_layer_stack = unique_params = None
    collect_base_pairs = evaluate_decoder = mean_std = seed_all = None
    split_at = train_decoder = make_provenance = None


# Private local-stage fine-tune with boundary capture at checkpoints.
# Plain full-model autograd with the middle layers frozen (no CloudWorker
# needed in-process; the trust story is about WHO owns the weights, not the
# wire). h* capture needs only embed+head, so it runs under no_grad.
def private_finetune_and_capture(embed, head, middle, tail, norm, lm_head,
                                 rotary, encode, ft_docs, capture_ids,
                                 checkpoints, args, eval_ids=None):
    """Fine-tune local stage only; capture victim boundary h* after each T
    in `checkpoints` (0 = at init). Returns ({T: h*}, {T: heldout_loss})."""
    checkpoints = sorted(set(checkpoints))
    for p in middle.parameters():
        p.requires_grad_(False)
    local_params = unique_params((embed, head, tail, norm, lm_head))
    opt = torch.optim.AdamW(local_params, lr=args.ft_lr)

    blocks = []
    for doc in ft_docs:
        blocks.extend(encode([doc], args.ft_seq_len))
    if not blocks:
        raise ValueError("no private-FT blocks; enlarge private shard or "
                         "shrink --ft-seq-len")

    def capture():
        hs = []
        with torch.no_grad():
            for ids in capture_ids:
                ids = ids[:-1].unsqueeze(0).to(args.device)
                position_ids = torch.arange(ids.shape[1],
                                            device=args.device).unsqueeze(0)
                hidden = embed(ids)
                lk = make_layer_kwargs(rotary, hidden, position_ids, args)
                h = run_layer_stack(head, hidden, lk)
                hs.append(h.detach()[0].float().cpu())
        return torch.cat(hs)

    def heldout_loss():
        """Utility co-metric: CE loss on docs never seen in fine-tuning.
        (Must NOT be the FT docs themselves — that would measure memorization.)"""
        if not eval_ids:
            raise ValueError("empty held-out eval set — refusing to fall back to FT docs")
        eval_set = eval_ids
        losses = []
        with torch.no_grad():
            for ids in eval_set:
                ids = ids.to(args.device)
                input_ids, labels = ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0)
                position_ids = torch.arange(input_ids.shape[1],
                                            device=args.device).unsqueeze(0)
                hidden = embed(input_ids)
                lk = make_layer_kwargs(rotary, hidden, position_ids, args)
                out = run_layer_stack(head, hidden, lk)
                out = run_layer_stack(middle, out, lk)
                out = run_layer_stack(tail, out, lk)
                logits = lm_head(norm(out))
                losses.append(F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    labels.reshape(-1)).item())
        return sum(losses) / len(losses)

    captured = {}
    utility = {}
    if 0 in checkpoints:
        captured[0] = capture()
        utility[0] = heldout_loss()
        print(f"    [private-ft] step 0 utility_loss={utility[0]:.4f}")
    max_step = checkpoints[-1]
    for step in range(1, max_step + 1):
        opt.zero_grad(set_to_none=True)
        ids = blocks[(step - 1) % len(blocks)].unsqueeze(0).to(args.device)
        input_ids, labels = ids[:, :-1], ids[:, 1:]
        position_ids = torch.arange(input_ids.shape[1],
                                    device=args.device).unsqueeze(0)
        hidden = embed(input_ids)
        lk = make_layer_kwargs(rotary, hidden, position_ids, args)
        h = run_layer_stack(head, hidden, lk)
        h = run_layer_stack(middle, h, lk)
        h = run_layer_stack(tail, h, lk)
        logits = lm_head(norm(h))
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                               labels.reshape(-1))
        loss.backward()
        opt.step()
        if step in checkpoints:
            captured[step] = capture()
            utility[step] = heldout_loss()
            print(f"    [private-ft] step {step}/{max_step} "
                  f"loss={loss.item():.4f} utility={utility[step]:.4f} "
                  f"(captured boundary)")
        elif step % 50 == 0 or step == max_step:
            print(f"    [private-ft] step {step}/{max_step} "
                  f"loss={loss.item():.4f}")
        if step % 10 == 0:
            _write_training_status(state="running", phase="private_ft",
                                   step=step, max_step=max_step,
                                   loss=round(loss.item(), 4))
    return captured, utility


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
                    help="public+private text, one document per line; the LAST "
                         "20%% is the victim's private shard (never used for "
                         "attack training)")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--train-steps-list", type=int, nargs="+",
                    default=[0, 10, 100, 1000],
                    help="private local-stage FT steps at which the boundary "
                         "is attacked")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--victim-docs", type=int, default=4,
                    help="docs held out from the END of the private shard as "
                         "capture/eval docs (never fine-tuned on)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seq-len", type=int, default=64,
                    help="block length for attack pairs and victim capture")
    ap.add_argument("--ft-seq-len", type=int, default=256,
                    help="block length for the private fine-tune")
    ap.add_argument("--ft-lr", type=float, default=1e-5,
                    help="private FT LR (gentle: drift, not task learning)")
    ap.add_argument("--max-pairs", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device",
                    default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    ap.add_argument("--attn-impl", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="depth 1, T in {0,100}, 1 seed, 1 victim doc, "
                         "5 epochs, seq 16, ft-seq 32")
    ap.add_argument("--output", default="e7_private_ft.json")
    args = ap.parse_args()

    if torch is None or build_modules is None:
        ap.error("torch/transformers not installed; install them or run --help only")

    if args.quick:
        args.depths = [1]
        args.train_steps_list = [0, 100]
        args.seeds = [0]
        args.victim_docs = 1
        args.epochs = 5
        args.seq_len = 16
        args.ft_seq_len = 32
        args.max_pairs = 2000

    seed_all(args.seed)
    _write_training_status(state="running", task="e7_private_ft",
                           depths=args.depths,
                           train_steps=args.train_steps_list,
                           started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    embed, layers, norm, lm_head, rotary, encode = build_modules(args)
    n_layers = len(layers)
    vocab_size = lm_head.weight.shape[0] if not args.toy else embed.weight.shape[0]
    hidden_dim = embed.weight.shape[1]
    print(f"[model] {'toy' if args.toy else args.model}: {n_layers} layers, "
          f"hidden={hidden_dim}, vocab={vocab_size}, device={args.device}")

    # private shard = last 20% of corpus
    # --corpus-file REPLACES TEXT_SAMPLES (no mixing, so results are
    # attributable to one source).
    if args.corpus_file:
        corpus_source = "corpus_file"
        with open(args.corpus_file) as f:
            docs = [l.strip() for l in f if len(l.strip()) > 500]  # real docs only — wikitext short lines are formatting artifacts
    else:
        corpus_source = "TEXT_SAMPLES"
        docs = list(TEXT_SAMPLES)
    n_priv = max(args.victim_docs + 1, int(round(0.20 * len(docs))))
    if len(docs) < n_priv + 4:
        raise ValueError(f"corpus too small: {len(docs)} docs; need private "
                         f"shard ({n_priv}) + at least 4 attack docs")
    private_shard = docs[-n_priv:]
    attack_docs = docs[:-n_priv]
    capture_docs = private_shard[-args.victim_docs:]   # eval only, never trained
    ft_docs = private_shard[:-args.victim_docs]
    n_val = max(1, int(round(args.val_frac * len(attack_docs))))
    val_docs, train_docs_pool = attack_docs[:n_val], attack_docs[n_val:]
    print(f"[data] {len(docs)} docs: {len(train_docs_pool)} attack-train, "
          f"{n_val} attack-val, {len(ft_docs)} private-FT, "
          f"{len(capture_docs)} private-capture (held out from end)")

    capture_ids, victim_tokens = [], []
    for doc in capture_docs:
        b = encode([doc], args.seq_len)
        if b:
            capture_ids.append(b[0])
            victim_tokens.append(b[0][:-1])
    if not capture_ids:
        raise ValueError("no capture doc long enough to yield a block")
    victim_tok = torch.cat(victim_tokens)
    print(f"[data] {len(capture_ids)} capture blocks, {victim_tok.shape[0]} "
          f"attacked positions (seq_len={args.seq_len})")

    # held-out utility set: attack-pool docs, never fine-tuned on (the private
    # FT pool is the private shard) — measures generalization, not memorization
    eval_ids = []
    for doc in val_docs:
        if len(eval_ids) >= 3:
            break
        b = encode([doc], args.seq_len)
        if b:
            eval_ids.append(b[0])
    if not eval_ids:
        raise ValueError("no held-out eval blocks: val docs too short for seq_len")

    provenance = make_provenance(
        args.corpus_file, corpus_source, len(docs),
        range(len(docs) - args.victim_docs, len(docs)),
        model_path=getattr(args, 'model', None), docs=docs)

    import copy
    results, summary = [], []

    for depth in args.depths:
        head, middle, tail, sa, ra = split_at(layers, depth, n_layers)
        if sa != depth:
            print(f"[split] depth {depth} clamped to sa={sa} ({n_layers} layers)")

        # base attacker: pairs through the PUBLIC base model, once
        t0 = time.time()
        tr_h, _, tr_tok = collect_base_pairs(
            embed, head, middle, tail, norm, lm_head, rotary, encode,
            train_docs_pool, args, with_grad=False)
        va_h, _, va_tok = collect_base_pairs(
            embed, head, middle, tail, norm, lm_head, rotary, encode,
            val_docs, args, with_grad=False)
        print(f"[collect] depth={sa}: base pairs train={tr_h.shape[0]} "
              f"val={va_h.shape[0]} ({time.time() - t0:.1f}s)")

        base_decoders = {}
        for seed in args.seeds:
            seed_all(args.seed + seed)
            dec = train_decoder(tr_h, tr_tok, va_h, va_tok, tr_h.shape[1],
                                vocab_size, args,
                                f"e7_base_d{sa}_seed{seed}")
            base_decoders[seed] = dec
            v1, v5 = evaluate_decoder(dec, va_h, va_tok, args.device)
            print(f"[attack] depth={sa} base seed={seed}: "
                  f"val top-1={v1:.2f}% top-5={v5:.2f}%")

        # private local-stage FT on fresh weights; capture at each T
        if args.toy:
            w_embed, w_layers = copy.deepcopy(embed), copy.deepcopy(layers)
            w_norm, w_lm_head = copy.deepcopy(norm), copy.deepcopy(lm_head)
        else:
            a = argparse.Namespace(**vars(args))
            w_embed, w_layers, w_norm, w_lm_head, _, _ = build_modules(a)
        w_head, w_middle, w_tail, sa2, _ = split_at(w_layers, depth, n_layers)
        print(f"[private-ft] depth={sa}: {max(args.train_steps_list)} steps, "
              f"captures at {args.train_steps_list}")
        captured, utility = private_finetune_and_capture(
            w_embed, w_head, w_middle, w_tail, w_norm, w_lm_head, rotary,
            encode, ft_docs, capture_ids, args.train_steps_list, args,
            eval_ids=eval_ids)

        for T in sorted(captured):
            h_star = captured[T]
            step_utility = utility.get(T)
            t1s, t5s = [], []
            for seed in args.seeds:
                top1, top5 = evaluate_decoder(base_decoders[seed], h_star,
                                              victim_tok, args.device)
                t1s.append(top1)
                t5s.append(top5)
                results.append({"depth": sa, "T": T, "attacker": "base",
                                "seed": seed, "top1": top1, "top5": top5,
                                "utility_loss": step_utility})
            m1, s1 = mean_std(t1s)
            m5, s5 = mean_std(t5s)
            summary.append({"depth": sa, "T": T, "attacker": "base",
                            "top1_mean": m1, "top1_std": s1,
                            "top5_mean": m5, "top5_std": s5,
                            "utility_loss": step_utility,
                            "n_seeds": len(t1s)})
            print(f"[eval] depth={sa} T={T} base: "
                  f"top-1={m1:.2f}+-{s1:.2f}% top-5={m5:.2f}+-{s5:.2f}% "
                  f"utility={step_utility:.3f}")

            # adaptive attacker at T (knows adapted weights)
            if T == 0:
                # adapted == base at T=0: reuse, exact consistency
                ad_t1s, ad_t5s = list(t1s), list(t5s)
                for seed in args.seeds:
                    results.append({"depth": sa, "T": T, "attacker": "adaptive",
                                    "seed": seed, "top1": t1s[args.seeds.index(seed)],
                                    "top5": t5s[args.seeds.index(seed)],
                                    "utility_loss": step_utility,
                                    "note": "T=0: identical to base"})
            else:
                atr_h, _, atr_tok = collect_base_pairs(
                    w_embed, w_head, w_middle, w_tail, w_norm, w_lm_head,
                    rotary, encode, train_docs_pool, args, with_grad=False)
                ava_h, _, ava_tok = collect_base_pairs(
                    w_embed, w_head, w_middle, w_tail, w_norm, w_lm_head,
                    rotary, encode, val_docs, args, with_grad=False)
                ad_t1s, ad_t5s = [], []
                for seed in args.seeds:
                    seed_all(args.seed + seed)
                    dec = train_decoder(atr_h, atr_tok, ava_h, ava_tok,
                                        atr_h.shape[1], vocab_size, args,
                                        f"e7_adaptive_d{sa}_T{T}_seed{seed}")
                    top1, top5 = evaluate_decoder(dec, h_star, victim_tok,
                                                  args.device)
                    ad_t1s.append(top1)
                    ad_t5s.append(top5)
                    results.append({"depth": sa, "T": T, "attacker": "adaptive",
                                    "seed": seed, "top1": top1, "top5": top5,
                                    "utility_loss": step_utility})
            m1, s1 = mean_std(ad_t1s)
            m5, s5 = mean_std(ad_t5s)
            summary.append({"depth": sa, "T": T, "attacker": "adaptive",
                            "top1_mean": m1, "top1_std": s1,
                            "top5_mean": m5, "top5_std": s5,
                            "utility_loss": step_utility,
                            "n_seeds": len(ad_t1s)})
            print(f"[eval] depth={sa} T={T} adaptive: "
                  f"top-1={m1:.2f}+-{s1:.2f}% top-5={m5:.2f}+-{s5:.2f}%")

    out = {
        "config": {"model": "toy" if args.toy else args.model,
                   "n_layers": n_layers, "depths": args.depths,
                   "train_steps_list": args.train_steps_list,
                   "seeds": args.seeds, "victim_docs": args.victim_docs,
                   "private_shard_frac": 0.20, "private_shard_docs": n_priv,
                   "ft_lr": args.ft_lr, "ft_seq_len": args.ft_seq_len,
                   "seq_len": args.seq_len, "max_pairs": args.max_pairs,
                   "epochs": args.epochs, "dtype": args.dtype,
                   "device": args.device, "quick": args.quick},
        "threat_model": "base attacker = public text + public base weights "
                        "only; adaptive attacker = knows the victim's adapted "
                        "local-stage weights (upper bound). Private shard is "
                        "the last 20% of the corpus, never used for attack "
                        "training; capture docs held out from its end.",
        "interpretation": "headline curve recovery(T) per depth: base attacker "
                          "falling with T => private local-stage drift defends; "
                          "flat => even 1000 private steps do not protect. "
                          "adaptive ~= flat/high confirms the ceiling is weight "
                          "secrecy, not the attack.",
        "random_baseline_top1_pct": round(100.0 / vocab_size, 4),
        "provenance": provenance,
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
