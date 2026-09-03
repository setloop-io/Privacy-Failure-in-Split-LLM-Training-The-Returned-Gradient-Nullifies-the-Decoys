#!/usr/bin/env python3
"""DLG++: strengthened optimization attacker (A2) for split-TRAINING boundaries.

Stronger sibling of gradient_inversion.py's deliberately-weak DLG/iDLG attack
(<=15% token recovery at depth 1 on Qwen3-0.6B). Same threat model — the
semi-honest cloud observes the boundary activation h* and gradient g* = dL/dh
at split depth d — but with the optimization upgrades the literature says
matter:

  1. Multi-restart (default 8 restarts, keep best final objective) with a
     cosine LR schedule + linear warmup per restart.
  2. Better objective: L2 AND cosine matching on both the boundary gradient
     and the boundary activation, plus an optional total-variation prior on
     the continuous embeddings (--tv-weight).
  3. Longer optimization (default 1000 rounds) with per-restart early stop
     (--patience rounds without objective improvement).
  4. Token-level refinement: after the continuous phase, coordinate descent
     over the top-k nearest-embedding candidates per position (2 passes).
  5. Optional --seq-prior: rescore refined candidates with the base model as
     an LM prior (avg logprob term — the A1 trick from the split-inference
     study).

Comparison targets: (a) the weak DLG in gradient_inversion.py and (b) the
trained MLP decoder in trained_inversion.py — this script measures A2 on the
same boundary at depths {1, 4, 8}, seq 64, Qwen3-0.6B.

Honest-scope notes (results remain a LOWER BOUND on privacy risk):
  - surrogate = public base checkpoint (exact at fine-tune step 0);
  - pseudo-labels are self-generated (true labels unknown);
  - single microbatches, seq_len <= 64.

Usage:
  python dlgpp.py --help        # works without torch
  python dlgpp.py --toy --quick # CPU machinery check (<=5 min)
  python dlgpp.py --model <hf-model> --corpus-file <docs.txt> --seq-prior --output dlgpp.json
"""

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext

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
    from trained_inversion import seed_all, split_at, mean_std, make_provenance
    from gradient_inversion import (capture_boundary, make_surrogate,
                                    run_split_training)
except ImportError:  # pragma: no cover - torch-less: imports still parse
    TEXT_SAMPLES = []
    _write_training_status = lambda **k: None
    seed_all = split_at = mean_std = None
    make_provenance = None
    build_modules = make_layer_kwargs = run_layer_stack = None
    capture_boundary = make_surrogate = run_split_training = None


# Double-backward needs the math attention kernel (flash/efficient SDPA has
# no second derivative). sdpa_kernel CMs are single-use -> fresh per call.
def _math_ctx_fn():
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return lambda: sdpa_kernel([SDPBackend.MATH])
    except Exception:  # pragma: no cover - older torch
        return nullcontext


def nearest_tokens(emb_seq, embed_w):
    """Cosine nearest neighbour over vocab rows: (1,T,H) -> (1,T)."""
    H = emb_seq.shape[-1]
    zn = F.normalize(emb_seq.float().reshape(-1, H), dim=-1)
    en = F.normalize(embed_w, dim=-1)
    return (zn @ en.T).argmax(-1).reshape(1, -1)


def topk_tokens(emb_seq, embed_w, k):
    """Top-k nearest vocab rows per position: (1,T,H) -> (T, k)."""
    H = emb_seq.shape[-1]
    zn = F.normalize(emb_seq.float().reshape(-1, H), dim=-1)
    en = F.normalize(embed_w, dim=-1)
    sims = zn @ en.T
    return sims.topk(min(k, sims.shape[-1]), dim=-1).indices


def surrogate_objs(embed, head, cloud, tail, norm, lm_head, rotary, z,
                   position_ids, pseudo, args):
    """Forward the surrogate on continuous embeddings z; return (h_hat, g_hat).
    g_hat = d(pseudo-label CE)/dh_hat with create_graph for double backward."""
    lk = make_layer_kwargs(rotary, z, position_ids, args)
    h_hat = run_layer_stack(head, z, lk)
    out = run_layer_stack(cloud, h_hat, lk)
    out = run_layer_stack(tail, out, lk)
    logits = lm_head(norm(out))
    loss = F.cross_entropy(logits.float()[:, :-1].reshape(-1, logits.shape[-1]),
                           pseudo[:, 1:].reshape(-1))
    g_hat = torch.autograd.grad(loss, h_hat, create_graph=True)[0]
    return h_hat, g_hat


def match_objective(h_hat, g_hat, h_star, g_star, z, args):
    """L2 + cosine on gradient AND activation + optional TV prior on z."""
    cos = (1 - F.cosine_similarity(g_hat.flatten(), g_star.flatten(), dim=0)
           + 1 - F.cosine_similarity(h_hat.flatten(), h_star.flatten(), dim=0))
    l2 = (((g_hat - g_star).pow(2).sum() / g_star.pow(2).sum().clamp_min(1e-12))
          + ((h_hat - h_star).pow(2).sum() / h_star.pow(2).sum().clamp_min(1e-12)))
    obj = cos + args.l2_weight * l2
    if args.tv_weight > 0:
        obj = obj + args.tv_weight * (z[:, 1:] - z[:, :-1]).pow(2).mean()
    return obj


def warmup_cosine(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    import math
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


# Phase 1: continuous optimization (one restart).
def continuous_restart(surr, head, cloud, tail, rotary, h_star, g_star,
                       embed_w, args, seed, seq_len):
    torch.manual_seed(seed)
    embed, norm, lm_head = surr["embed"], surr["norm"], surr["lm_head"]
    H = surr["embed_dim"]
    z = (torch.randn(1, seq_len, H, device=args.device)
         * args.init_scale).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=args.attack_lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: warmup_cosine(s, args.rounds,
                                     int(args.warmup_frac * args.rounds)))
    position_ids = torch.arange(seq_len, device=args.device).unsqueeze(0)
    math_ctx = _math_ctx_fn()

    best_obj, best_z, stale = float("inf"), None, 0
    for r in range(args.rounds):
        with math_ctx():
            pseudo = nearest_tokens(z.detach(), embed_w)
            h_hat, g_hat = surrogate_objs(embed, head, cloud, tail, norm,
                                          lm_head, rotary, z, position_ids,
                                          pseudo, args)
            obj = match_objective(h_hat, g_hat, h_star, g_star, z, args)
        opt.zero_grad()
        obj.backward()
        opt.step()
        sched.step()
        v = obj.item()
        if v < best_obj - 1e-6:
            best_obj, best_z, stale = v, z.detach().clone(), 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    return best_obj, best_z


# Phase 2: discrete token-level refinement (coordinate descent).
def discrete_objective(embed, head, cloud, tail, norm, lm_head, rotary, ids,
                       h_star, g_star, args, prior_w=0.0):
    """Match objective evaluated on the DISCRETE sequence ids (no double
    backward — g_hat needs only first-order grad). Optionally minus the base
    LM's avg logprob of ids (--seq-prior rescoring)."""
    position_ids = torch.arange(ids.shape[1], device=args.device).unsqueeze(0)
    z = embed(ids)
    lk = make_layer_kwargs(rotary, z, position_ids, args)
    # boundary leaf: surrogate params are frozen, so h_hat needs explicit grad
    h_hat = run_layer_stack(head, z, lk).detach().requires_grad_(True)
    out = run_layer_stack(cloud, h_hat, lk)
    out = run_layer_stack(tail, out, lk)
    logits = lm_head(norm(out))
    loss = F.cross_entropy(logits.float()[:, :-1].reshape(-1, logits.shape[-1]),
                           ids[:, 1:].reshape(-1))
    g_hat = torch.autograd.grad(loss, h_hat)[0]
    with torch.no_grad():
        obj = match_objective(h_hat, g_hat, h_star, g_star, z, args).item()
    if prior_w > 0:
        with torch.no_grad():
            lp = F.cross_entropy(
                logits.float()[:, :-1].reshape(-1, logits.shape[-1]),
                ids[:, 1:].reshape(-1), reduction="mean")
        obj = obj + prior_w * lp.item()
    return obj


def refine_tokens(surr, head, cloud, tail, rotary, z_best, h_star, g_star,
                  embed_w, args):
    """Coordinate descent over top-k nearest-embedding candidates per
    position, args.passes passes. Returns (best_ids, best_obj)."""
    embed, norm, lm_head = surr["embed"], surr["norm"], surr["lm_head"]
    ids = nearest_tokens(z_best, embed_w).to(args.device)
    cands = topk_tokens(z_best, embed_w, args.topk).to(args.device)
    T = ids.shape[1]
    best = discrete_objective(embed, head, cloud, tail, norm, lm_head, rotary,
                              ids, h_star, g_star, args)
    for _ in range(args.passes):
        improved = False
        for i in range(T):
            cur = ids[0, i].item()
            for c in cands[i].tolist():
                if c == cur:
                    continue
                trial = ids.clone()
                trial[0, i] = c
                o = discrete_objective(embed, head, cloud, tail, norm, lm_head,
                                       rotary, trial, h_star, g_star, args)
                if o < best - 1e-9:
                    best, ids, improved = o, trial, True
                    cur = c
        if not improved:
            break
    return ids, best


def seq_prior_rescore(surr, head, cloud, tail, rotary, ids, z_best, h_star,
                      g_star, embed_w, args):
    """A1 trick: redo coordinate descent with obj + prior_weight * NLL(seq)
    so the LM prior breaks ties between near-equal attack candidates."""
    embed, norm, lm_head = surr["embed"], surr["norm"], surr["lm_head"]
    cands = topk_tokens(z_best, embed_w, args.topk).to(args.device)
    T = ids.shape[1]
    best = discrete_objective(embed, head, cloud, tail, norm, lm_head, rotary,
                              ids, h_star, g_star, args,
                              prior_w=args.prior_weight)
    for _ in range(args.passes):
        improved = False
        for i in range(T):
            cur = ids[0, i].item()
            for c in cands[i].tolist():
                if c == cur:
                    continue
                trial = ids.clone()
                trial[0, i] = c
                o = discrete_objective(embed, head, cloud, tail, norm, lm_head,
                                       rotary, trial, h_star, g_star, args,
                                       prior_w=args.prior_weight)
                if o < best - 1e-9:
                    best, ids, improved = o, trial, True
                    cur = c
        if not improved:
            break
    return ids


# Full attack on one boundary observation.
def attack(surr, depth_sa, rotary, h_star, g_star, true_ids, args, seed):
    t0 = time.time()
    layers = surr["layers"]
    head, cloud, tail, _, _ = split_at(layers, depth_sa, len(layers))
    embed_w = surr["embed"].weight.detach().float()
    seq_len = true_ids.shape[1]

    best_obj, best_z = float("inf"), None
    for r in range(args.restarts):
        o, z = continuous_restart(surr, head, cloud, tail, rotary, h_star,
                                  g_star, embed_w, args, seed * 1000 + r,
                                  seq_len)
        if o < best_obj:
            best_obj, best_z = o, z
    cont_ids = nearest_tokens(best_z, embed_w)

    ref_ids, ref_obj = refine_tokens(surr, head, cloud, tail, rotary, best_z,
                                     h_star, g_star, embed_w, args)
    final_ids = ref_ids
    if args.seq_prior:
        final_ids = seq_prior_rescore(surr, head, cloud, tail, rotary,
                                      ref_ids, best_z, h_star, g_star,
                                      embed_w, args)

    true = true_ids.squeeze(0).cpu()
    H = surr["embed_dim"]

    def acc(pred_ids):
        return (pred_ids.squeeze(0).cpu() == true).float().mean().item()

    true_emb = F.normalize(embed_w[true_ids.squeeze(0)].reshape(-1, H), dim=-1).cpu()
    rec_emb = F.normalize(best_z.float().reshape(-1, H).cpu(), dim=-1)
    emb_cos = (true_emb * rec_emb).sum(-1).mean().item()
    return {"token_acc_continuous": acc(cont_ids),
            "token_acc": acc(final_ids),
            "token_acc_refined": acc(ref_ids),
            "emb_cos": emb_cos,
            "final_obj_continuous": best_obj,
            "final_obj": ref_obj,
            "restarts": args.restarts,
            "attack_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.expanduser(
        "~/experiments/models/qwen3-0.6b"), help="HF model path (ignored with --toy)")
    ap.add_argument("--toy", action="store_true",
                    help="tiny random built-in model (CPU machinery check only)")
    ap.add_argument("--corpus-file", default=None,
                    help="victim text, one document per line (REPLACES "
                         "TEXT_SAMPLES when given)")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 4, 8],
                    help="split depths (local layers 0..d)")
    ap.add_argument("--docs", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--train-steps-list", type=int, nargs="+", default=[0],
                    help="victim fine-tuning drift points; a fresh victim is "
                         "trained from the public base for each point while "
                         "the attacker surrogate remains the public base")
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="victim split-training learning rate for drift points")
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--seq-len", type=int, default=64,
                    help="attacked sequence length (32-64 recommended)")
    ap.add_argument("--rounds", type=int, default=1000,
                    help="continuous optimization rounds per restart")
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--patience", type=int, default=100,
                    help="per-restart early stop (rounds without improvement)")
    ap.add_argument("--attack-lr", type=float, default=0.05)
    ap.add_argument("--warmup-frac", type=float, default=0.05,
                    help="fraction of rounds for linear LR warmup")
    ap.add_argument("--init-scale", type=float, default=0.02)
    ap.add_argument("--l2-weight", type=float, default=1.0,
                    help="weight of the relative-L2 terms (cosine terms = 1)")
    ap.add_argument("--tv-weight", type=float, default=0.0,
                    help="total-variation prior weight on z (0 = off)")
    ap.add_argument("--topk", type=int, default=8,
                    help="candidates per position for token-level refinement")
    ap.add_argument("--passes", type=int, default=2,
                    help="coordinate-descent passes in the discrete phase")
    ap.add_argument("--seq-prior", action="store_true",
                    help="rescore refined tokens with the base LM prior (A1 trick)")
    ap.add_argument("--prior-weight", type=float, default=0.1,
                    help="weight of the LM NLL term in --seq-prior rescoring")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device",
                    default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    ap.add_argument("--attn-impl", choices=["sdpa", "eager"], default="eager",
                    help="eager default: double-backward through sdpa kernels "
                         "is not guaranteed; the attack needs create_graph")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="depth 1, 1 doc, 1 seed, seq 8, 100 rounds, "
                         "2 restarts (<=5 min CPU on --toy)")
    ap.add_argument("--output", default="dlgpp_results.json")
    args = ap.parse_args()

    if torch is None or build_modules is None:
        ap.error("torch/transformers not installed; install them or run --help only")
    args.defend_activation = False  # capture_boundary signature compat

    if args.quick:
        args.depths = [1]
        args.docs = 1
        args.seeds = [0]
        args.seq_len = 8
        args.rounds = 100
        args.restarts = 2
        args.patience = 30
        args.train_steps_list = [0, 1]

    seed_all(args.seed)
    _write_training_status(state="running", task="dlgpp",
                           depths=args.depths, rounds=args.rounds,
                           restarts=args.restarts,
                           started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    embed, layers, norm, lm_head, rotary, encode = build_modules(args)
    n_layers = len(layers)
    print(f"[model] {'toy' if args.toy else args.model}: {n_layers} layers, "
          f"device={args.device}")

    # Surrogate = public base checkpoint in fp32 (attack stability).
    surrogate = make_surrogate(args, n_layers, (embed, layers, norm, lm_head))

    # --corpus-file REPLACES TEXT_SAMPLES (no mixing, so results are
    # attributable to one source).
    if args.corpus_file:
        corpus_source = "corpus_file"
        with open(args.corpus_file) as f:
            texts = [l.strip() for l in f if l.strip()]
    else:
        corpus_source = "TEXT_SAMPLES"
        texts = list(TEXT_SAMPLES)
    # one block per doc, exactly seq_len+1 tokens (truncate, not chunk-mix)
    blocks = []
    doc_indices = []
    for ti, t in enumerate(texts):
        b = encode([t], args.seq_len)
        if b:
            blocks.append(b[0])
            doc_indices.append(ti)
        if len(blocks) >= args.docs:
            break
    if not blocks:
        raise ValueError("no doc long enough to yield a block")
    provenance = make_provenance(args.corpus_file, corpus_source,
                                 len(texts), doc_indices, model_path=getattr(args, 'model', None))
    print(f"[data] {len(blocks)} attack docs (seq_len={args.seq_len}), "
          f"depths={args.depths}, seeds={args.seeds}, "
          f"restarts={args.restarts}, rounds={args.rounds}, "
          f"seq_prior={args.seq_prior}")

    train_blocks = encode(texts, args.seq_len)
    results, summary = [], []
    for depth in args.depths:
        for train_steps in args.train_steps_list:
            # Fresh victim per drift point. The attacker surrogate above is
            # deliberately held at the public step-0 checkpoint.
            if args.toy:
                victim = make_surrogate(
                    args, n_layers, (embed, layers, norm, lm_head), freeze=False)
                w_embed, w_layers = victim["embed"], victim["layers"]
                w_norm, w_lm_head, w_rotary = victim["norm"], victim["lm_head"], rotary
            else:
                fresh = argparse.Namespace(**vars(args))
                w_embed, w_layers, w_norm, w_lm_head, w_rotary, _ = build_modules(fresh)
            head, cloud_l, tail, sa, ra = split_at(w_layers, depth, n_layers)
            if sa != depth:
                print(f"[split] depth {depth} clamped to sa={sa} ({n_layers} layers)")
            if train_steps:
                run_split_training(
                    w_embed, head, cloud_l, tail, w_norm, w_lm_head,
                    w_rotary, train_blocks, args, ("none", None),
                    train_steps, "full")
            accs, coss = [], []
            for di, ids in enumerate(blocks):
                ids = ids.to(args.device)
                input_ids, labels = ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0)
                h_star, g_star, loss0 = capture_boundary(
                    w_embed, head, cloud_l, tail, w_norm, w_lm_head,
                    w_rotary, input_ids, labels, args, ("none", None))
                for seed in args.seeds:
                    _write_training_status(state="running", phase="attack",
                                           depth=sa, doc=di, seed=seed,
                                           train_steps=train_steps)
                    m = attack(surrogate, sa, w_rotary, h_star, g_star,
                               input_ids, args, seed)
                    m.update({"depth": sa, "doc": di, "seed": seed,
                              "train_steps": train_steps,
                              "microbatch_loss": loss0})
                    results.append(m)
                    accs.append(m["token_acc"]); coss.append(m["emb_cos"])
                    print(f"[a2] depth={sa} train_steps={train_steps} "
                          f"doc={di} seed={seed} acc={m['token_acc']:.3f} "
                          f"(cont={m['token_acc_continuous']:.3f}) "
                          f"cos={m['emb_cos']:.3f} obj={m['final_obj']:.4f} "
                          f"({m['attack_s']:.0f}s)")
            am, asd = mean_std(accs); cm, csd = mean_std(coss)
            summary.append({"depth": sa, "train_steps": train_steps,
                            "token_acc_mean": am, "token_acc_std": asd,
                            "emb_cos_mean": cm, "emb_cos_std": csd,
                            "n": len(accs)})
            print(f"[summary] depth={sa} train_steps={train_steps}: "
                  f"acc={am:.3f}+-{asd:.3f} cos={cm:.3f}+-{csd:.3f}")
            del w_embed, w_layers, w_norm, w_lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    out = {
        "config": {"model": "toy" if args.toy else args.model,
                   "measurement_kind": "measured",
                   "n_layers": n_layers, "depths": args.depths,
                   "docs": args.docs, "seeds": args.seeds,
                   "train_steps_list": args.train_steps_list,
                   "victim_lr": args.lr,
                   "seq_len": args.seq_len, "rounds": args.rounds,
                   "restarts": args.restarts, "patience": args.patience,
                   "attack_lr": args.attack_lr, "l2_weight": args.l2_weight,
                   "tv_weight": args.tv_weight, "topk": args.topk,
                   "passes": args.passes, "seq_prior": args.seq_prior,
                   "prior_weight": args.prior_weight,
                   "dtype": args.dtype, "device": args.device,
                   "quick": args.quick},
        "attacker": "A2 DLG++ (multi-restart, cosine LR + warmup, L2+cosine "
                    "on grad+activation, optional TV prior, 1000 rounds with "
                    "early stop, top-k token-level refinement, optional LM "
                    "seq-prior rescoring)",
        "comparison_baselines": {
            "weak_dlg": "gradient_inversion.py (cosine-only, 1 restart, "
                        "200 rounds, lr 0.05, no refinement)",
            "trained_mlp": "trained_inversion.py InversionDecoder (A_act / "
                           "B_act+grad)"},
        "threat_model": "semi-honest cloud observes boundary activation h* and "
                        "gradient g*=dL/dh per microbatch; attacker uses the "
                        "public base checkpoint as surrogate + self-generated "
                        "pseudo-labels (results are a LOWER BOUND on privacy "
                        "risk)",
        "random_baseline_top1_pct": round(100.0 / surrogate["embed"].weight.shape[0], 6),
        "resolution_floor_top1_pct": round(100.0 / args.seq_len, 6),
        "evidence_status": "supporting",
        "known_limitations": [
            "Recovery below resolution_floor_top1_pct is below one token per attacked sequence, not proof of zero leakage.",
            "The public-base surrogate is exact only at train_steps=0; later points intentionally measure victim drift."
        ],
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
