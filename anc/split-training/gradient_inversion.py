#!/usr/bin/env python3
"""Gradient inversion attacks + defense frontier for split TRAINING (E2/E3).

Follow-on to the split-inference inversion study. Threat model: the
semi-honest cloud observes, per microbatch, the boundary ACTIVATION h*
(input to its
first layer) and the boundary GRADIENT g* = dL/dh (which it computes and
returns to the local node). The attacker reconstructs the local node's input
tokens DLG/iDLG-style by optimizing a dummy embedding sequence z whose
surrogate boundary activation/gradient match the observations:

    min_z  (1 - cos(g_hat(z), g*)) + (1 - cos(h_hat(z), h*))

where the surrogate is the PUBLIC base checkpoint (embed + all layers):
g_hat = dL_hat/dh_hat with pseudo-labels = current argmax reconstruction of z
(detached; causal-LM labels are the shifted inputs, so they come along with
the sequence). Reconstruction = per-position nearest embedding row of z.

Honest-scope notes (report as a LOWER BOUND on privacy risk):
  - the surrogate assumes local stages are at/near the base checkpoint; this
    is exact at fine-tune step 0 and degrades as local params drift
    (--train-steps > 0 measures exactly that effect);
  - pseudo-labels are self-generated (true labels unknown) — weakens the
    attack vs an oracle;
  - attacks run on single microbatches of seq_len <= 32.

E2: token-recovery vs split depth x training config (full FT / freeze-cloud /
freeze-cloud+LoRA-local). E3: defense x depth inversion accuracy + utility
(loss after N defended split-training steps vs clean).

Usage:
  python gradient_inversion.py --help        # works without torch
  python gradient_inversion.py --toy --quick # CPU check (~1 min)
  python gradient_inversion.py --model <hf-model> --corpus-file <docs.txt>  # full grid
"""

import argparse
import copy
import gc
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
    from split_trainer import (TEXT_SAMPLES, CloudWorker, ToyCausalLM,
                               _write_training_status, apply_lora,
                               build_modules, make_layer_kwargs,
                               run_layer_stack, unique_params)
except ImportError:  # pragma: no cover - torch-less: split_trainer still imports
    TEXT_SAMPLES = []
    CloudWorker = None
    ToyCausalLM = None
    _write_training_status = lambda **k: None
    apply_lora = build_modules = make_layer_kwargs = run_layer_stack = None
    unique_params = None


# Defenses (E3). Specs: "none" | "noise:<sigma_frac_of_RMS>" |
# "quant:<bits>" | "sign" | "topk:<keep_frac>"
def parse_defense(spec):
    if spec == "none":
        return ("none", None)
    if spec == "sign":
        return ("sign", None)
    kind, _, val = spec.partition(":")
    if kind not in ("noise", "quant", "topk") or not val:
        raise ValueError(f"bad defense spec {spec!r}")
    return (kind, float(val))


def apply_defense(t, parsed):
    """Defend a boundary tensor (activation or gradient)."""
    kind, val = parsed
    if kind == "none":
        return t
    if kind == "noise":
        rms = t.pow(2).mean().sqrt()
        return t + torch.randn_like(t) * (val * rms)
    if kind == "quant":
        bits = int(val)
        t_min, t_max = t.min(), t.max()
        scale = (t_max - t_min).clamp_min(1e-12) / (2 ** bits - 1)
        return (torch.round((t - t_min) / scale) * scale + t_min)
    if kind == "sign":
        return t.sign()
    if kind == "topk":
        k = max(1, int(val * t.numel()))
        thresh = t.abs().flatten().kthvalue(t.numel() - k + 1).values
        return t * (t.abs() >= thresh)
    raise ValueError(kind)


# Stages: reuse split_trainer's loader, then split like split_trainer.train.
def build_stages(args, dtype_override=None):
    a = argparse.Namespace(**vars(args))
    if dtype_override is not None:
        a.dtype = dtype_override
    embed, layers, norm, lm_head, rotary, encode = build_modules(a)
    n = len(layers)
    return embed, layers, norm, lm_head, rotary, encode, n


def split_at(layers, depth, n_layers):
    """Local head = 0..depth inclusive; cloud = depth+1..ra-1; tail = ra.."""
    sa = max(0, min(depth, n_layers - 3))
    ra = min(max(n_layers - 2, sa + 2), n_layers - 1)
    return (nn.ModuleList(list(layers[: sa + 1])),
            nn.ModuleList(list(layers[sa + 1: ra])),
            nn.ModuleList(list(layers[ra:])), sa, ra)


def capture_boundary(embed, head, cloud, tail, norm, lm_head, rotary,
                     input_ids, labels, args, defense):
    """One real split microbatch; returns (h*, g*) as the cloud sees them
    (defense applied to the gradient, and to the activation with
    --defend-activation)."""
    position_ids = torch.arange(input_ids.shape[1], device=args.device).unsqueeze(0)
    hidden = embed(input_ids)
    lk = make_layer_kwargs(rotary, hidden, position_ids, args)
    boundary = run_layer_stack(head, hidden, lk).detach().requires_grad_(True)
    out = run_layer_stack(cloud, boundary, lk)
    out = run_layer_stack(tail, out, lk)
    logits = lm_head(norm(out))
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                           labels.reshape(-1))
    g_star = torch.autograd.grad(loss, boundary)[0].detach()
    h_star = boundary.detach()
    g_seen = apply_defense(g_star, defense)
    h_seen = apply_defense(h_star, defense) if args.defend_activation else h_star
    return h_seen.float(), g_seen.float(), loss.item()


# Mini split trainer: --train-steps of real split training before the attack
# (and the E3 utility measurement). Same boundary-leaf protocol as
# split_trainer.train, in-process, optional grad defense on the wire.
def run_split_training(embed, head, cloud_layers, tail, norm, lm_head, rotary,
                       blocks, args, grad_defense, steps, train_config):
    cloud = CloudWorker(cloud_layers, lr=args.lr,
                        trainable=(train_config != "lora-local"
                                   and train_config != "freeze-cloud"),
                        grad_hook=(lambda g: apply_defense(g, grad_defense)))
    if train_config == "lora-local":
        apply_lora(nn.ModuleList([*head, *tail]), args.lora_rank, args.lora_alpha)
    local_mods = [embed, *head, *tail, norm, lm_head]
    local_params = unique_params(local_mods)
    opt = torch.optim.AdamW(local_params, lr=args.lr)
    losses = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        cloud.zero_grad()
        ids = blocks[step % len(blocks)].unsqueeze(0).to(args.device)
        input_ids, labels = ids[:, :-1], ids[:, 1:]
        position_ids = torch.arange(input_ids.shape[1], device=args.device).unsqueeze(0)
        hidden = embed(input_ids)
        lk = make_layer_kwargs(rotary, hidden, position_ids, args)
        head_out = run_layer_stack(head, hidden, lk)
        boundary = head_out.detach().requires_grad_(True)
        cloud_out = cloud.forward(boundary, lk)
        boundary_in = cloud_out.detach().requires_grad_(True)
        out = run_layer_stack(tail, boundary_in, lk)
        logits = lm_head(norm(out))
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                               labels.reshape(-1))
        loss.backward()
        grad_input = cloud.backward(boundary_in.grad)
        torch.autograd.backward(head_out, grad_tensors=grad_input)
        opt.step()
        cloud.step()
        losses.append(loss.item())
    return losses


# DLG/iDLG-style attack against the boundary observation.
def attack(surr, depth_sa, rotary, h_star, g_star, true_ids, args, seed):
    """Optimize dummy embeddings z (1, T, H) so the surrogate's boundary
    activation/gradient match the observations. Returns metrics dict."""
    torch.manual_seed(seed)
    embed, layers, norm, lm_head = (surr["embed"], surr["layers"],
                                    surr["norm"], surr["lm_head"])
    head, cloud, tail, _, _ = split_at(layers, depth_sa, len(layers))
    H = surr["embed_dim"]
    seq_len = true_ids.shape[1]
    z = (torch.randn(1, seq_len, H, device=args.device) * 0.02).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=args.attack_lr)
    position_ids = torch.arange(seq_len, device=args.device).unsqueeze(0)
    embed_w = embed.weight.detach().float()  # (V, H)

    def nearest_tokens(emb_seq):
        # cosine nearest neighbour over vocab rows: (1,T,H) -> (1,T)
        zn = F.normalize(emb_seq.float().reshape(-1, H), dim=-1)
        en = F.normalize(embed_w, dim=-1)
        return (zn @ en.T).argmax(-1).reshape(1, -1)

    # Double-backward (grad-of-grad through the surrogate) requires the math
    # attention kernel: flash/efficient SDPA has no second derivative.
    # (sdpa_kernel CMs are single-use -> fresh one per round.)
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        def math_ctx():
            return sdpa_kernel([SDPBackend.MATH])
    except Exception:  # pragma: no cover - older torch
        def math_ctx():
            return nullcontext()

    best = {"obj": float("inf"), "z": None}
    for r in range(args.rounds):
        with math_ctx():
            lk = make_layer_kwargs(rotary, z, position_ids, args)
            h_hat = run_layer_stack(head, z, lk)
            out = run_layer_stack(cloud, h_hat, lk)
            out = run_layer_stack(tail, out, lk)
            logits = lm_head(norm(out))
            # pseudo-labels: current argmax reconstruction, detached (labels are
            # the shifted inputs in a causal LM, so they ride along with z)
            pseudo = nearest_tokens(z.detach())
            loss = F.cross_entropy(logits.float()[:, :-1].reshape(-1, logits.shape[-1]),
                                   pseudo[:, 1:].reshape(-1))
            g_hat = torch.autograd.grad(loss, h_hat, create_graph=True)[0]
            obj = (1 - F.cosine_similarity(g_hat.flatten(), g_star.flatten(), dim=0)
                   + 1 - F.cosine_similarity(h_hat.flatten(), h_star.flatten(), dim=0))
        opt.zero_grad()
        obj.backward()
        opt.step()
        if obj.item() < best["obj"]:
            best = {"obj": obj.item(), "z": z.detach().clone()}

    z_best = best["z"]
    pred = nearest_tokens(z_best).squeeze(0).cpu()
    true = true_ids.squeeze(0).cpu()
    token_acc = (pred == true).float().mean().item()
    true_emb = F.normalize(embed_w[true_ids.squeeze(0)].reshape(-1, H), dim=-1).cpu()
    rec_emb = F.normalize(z_best.float().reshape(-1, H).cpu(), dim=-1)
    emb_cos = (true_emb * rec_emb).sum(-1).mean().item()
    return {"token_acc": token_acc, "emb_cos": emb_cos,
            "final_obj": best["obj"], "rounds": args.rounds}


def make_surrogate(args, n_layers, ref_modules, freeze=True):
    """Public-base surrogate in fp32 (attack stability). For --toy: a
    deepcopy of the pre-training state. For HF: a second fp32 load."""
    if args.toy:
        embed, layers, norm, lm_head = ref_modules
        surr = {"embed": copy.deepcopy(embed), "layers": copy.deepcopy(layers),
                "norm": copy.deepcopy(norm), "lm_head": copy.deepcopy(lm_head)}
    else:
        a = argparse.Namespace(**vars(args))
        a.dtype = "fp32"
        embed, layers, norm, lm_head, _, _ = build_modules(a)
        surr = {"embed": embed, "layers": nn.ModuleList(list(layers)),
                "norm": norm, "lm_head": lm_head}
    if freeze:
        for m in surr.values():
            if isinstance(m, nn.Module):
                for p in m.parameters():
                    p.requires_grad_(False)
    surr["embed_dim"] = surr["embed"].weight.shape[1]
    return surr


def mean_std(vals):
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
    return m, v ** 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.expanduser(
        "~/experiments/models/qwen3-0.6b"))
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--phase", choices=["e2", "e3", "all"], default="all")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 4, 8],
                    help="split depths (local layers 0..d)")
    ap.add_argument("--configs", nargs="+",
                    default=["full", "freeze-cloud", "lora-local"])
    ap.add_argument("--defenses", nargs="+",
                    default=["none", "noise:0.001", "noise:0.01", "noise:0.05",
                             "noise:0.1", "quant:8", "quant:4", "sign",
                             "topk:0.1", "topk:0.01"])
    ap.add_argument("--defend-activation", action="store_true",
                    help="also defend the boundary activation (default: grad only)")
    ap.add_argument("--docs", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seq-len", type=int, default=32,
                    help="attacked sequence length (16-32 recommended)")
    ap.add_argument("--rounds", type=int, default=200,
                    help="attack optimization rounds per (doc, seed)")
    ap.add_argument("--attack-lr", type=float, default=0.05)
    ap.add_argument("--train-steps", type=int, default=10,
                    help="real split-training steps before the attack (0 = at init)")
    ap.add_argument("--utility-steps", type=int, default=20,
                    help="E3 utility: defended vs clean loss after N steps")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device",
                    default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    ap.add_argument("--attn-impl", choices=["sdpa", "eager"], default="eager",
                    help="eager default: double-backward through sdpa kernels "
                         "is not guaranteed; the attack needs create_graph")
    ap.add_argument("--corpus-file", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="depth 1, none+noise:0.05+quant:4, 1 doc, 1 seed, "
                         "seq 8, rounds 100, train-steps 2, utility-steps 5")
    ap.add_argument("--output", default="gradient_inversion_results.json")
    args = ap.parse_args()

    if torch is None:
        ap.error("torch is not installed; install it or run --help only")

    if args.quick:
        args.depths = [1]
        args.defenses = ["none", "noise:0.05", "quant:4"]
        args.docs = 1
        args.seeds = [0]
        args.seq_len = 8
        args.rounds = 100
        args.train_steps = 2
        args.utility_steps = 5
        args.configs = ["full", "freeze-cloud", "lora-local"]

    torch.manual_seed(args.seed)
    _write_training_status(state="running", task="gradient_inversion",
                           phase=args.phase, depths=args.depths,
                           started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    embed, layers, norm, lm_head, rotary, encode, n_layers = build_stages(args)
    print(f"[model] {n_layers} layers, dtype={args.dtype}, device={args.device}")

    # Surrogate snapshot BEFORE any training (= public base assumption).
    surrogate = make_surrogate(args, n_layers, (embed, layers, norm, lm_head))

    texts = list(TEXT_SAMPLES)
    if args.corpus_file:
        with open(args.corpus_file) as f:
            texts.extend(l.strip() for l in f if l.strip())
    docs = texts[: args.docs]
    # one block per doc, exactly seq_len+1 tokens (truncate, not chunk-mix);
    # skip docs too short to yield a block
    blocks = []
    for t in texts:
        b = encode([t], args.seq_len)
        if b:
            blocks.append(b[0])
        if len(blocks) >= args.docs:
            break
    # longer blocks for the training/utility phases
    train_blocks = encode(texts, args.seq_len)
    print(f"[data] {len(docs)} attack docs, {len(train_blocks)} training blocks "
          f"(seq_len={args.seq_len})")

    results, summary, utility = [], [], []

    # ---- E2: inversion vs depth x training config (defense = none)
    if args.phase in ("e2", "all"):
        for depth in args.depths:
            for cfg in args.configs:
                # fresh stage copies per (depth, cfg): training mutates weights
                if args.toy:
                    base = make_surrogate(args, n_layers,
                                          (embed, layers, norm, lm_head),
                                          freeze=False)
                    w_embed, w_layers = base["embed"], base["layers"]
                    w_norm, w_head = base["norm"], base["lm_head"]
                else:
                    # reload to undo prior training mutations
                    a = argparse.Namespace(**vars(args))
                    w_embed, w_layers, w_norm, w_head, w_rotary, _ = build_modules(a)
                head, cloud_l, tail, sa, ra = split_at(w_layers, depth, n_layers)
                if args.train_steps > 0:
                    run_split_training(w_embed, head, cloud_l, tail, w_norm,
                                       w_head, rotary, train_blocks, args,
                                       ("none", None), args.train_steps, cfg)
                accs, coss = [], []
                for di, ids in enumerate(blocks):
                    ids = ids.to(args.device)
                    input_ids, labels = ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0)
                    h_star, g_star, loss0 = capture_boundary(
                        w_embed, head, cloud_l, tail, w_norm, w_head, rotary,
                        input_ids, labels, args, ("none", None))
                    for seed in args.seeds:
                        t0 = time.time()
                        m = attack(surrogate, sa, rotary, h_star, g_star,
                                   input_ids, args, seed)
                        m.update({"phase": "e2", "depth": sa, "train_config": cfg,
                                  "defense": "none", "doc": di, "seed": seed,
                                  "train_steps": args.train_steps,
                                  "microbatch_loss": loss0,
                                  "attack_s": round(time.time() - t0, 1)})
                        results.append(m)
                        accs.append(m["token_acc"])
                        coss.append(m["emb_cos"])
                        print(f"[e2] depth={sa} cfg={cfg} doc={di} seed={seed} "
                              f"acc={m['token_acc']:.3f} cos={m['emb_cos']:.3f}")
                am, asd = mean_std(accs)
                cm, csd = mean_std(coss)
                summary.append({"phase": "e2", "depth": sa, "train_config": cfg,
                                "token_acc_mean": am, "token_acc_std": asd,
                                "emb_cos_mean": cm, "emb_cos_std": csd,
                                "n": len(accs)})
                # free per-cell model copies (CUDA cache retains them otherwise)
                del w_embed, w_layers, w_norm, w_head
                if not args.toy:
                    del w_rotary
                gc.collect()
                torch.cuda.empty_cache()

    # ---- E3: defense frontier (inversion) + utility
    if args.phase in ("e3", "all"):
        for depth in args.depths:
            head, cloud_l, tail, sa, ra = split_at(layers, depth, n_layers)
            for spec in args.defenses:
                defense = parse_defense(spec)
                accs = []
                for di, ids in enumerate(blocks):
                    ids = ids.to(args.device)
                    input_ids, labels = ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0)
                    h_star, g_star, loss0 = capture_boundary(
                        embed, head, cloud_l, tail, norm, lm_head, rotary,
                        input_ids, labels, args, defense)
                    for seed in args.seeds:
                        m = attack(surrogate, sa, rotary, h_star, g_star,
                                   input_ids, args, seed)
                        results.append({"phase": "e3", "depth": sa,
                                        "defense": spec, "doc": di, "seed": seed,
                                        **{k: m[k] for k in
                                           ("token_acc", "emb_cos", "final_obj")}})
                        accs.append(m["token_acc"])
                        print(f"[e3] depth={sa} defense={spec} doc={di} "
                              f"seed={seed} acc={m['token_acc']:.3f}")
                am, asd = mean_std(accs)
                summary.append({"phase": "e3", "depth": sa, "defense": spec,
                                "token_acc_mean": am, "token_acc_std": asd,
                                "n": len(accs)})

        # utility: defended vs clean split-training loss curve
        for spec in args.defenses:
            defense = parse_defense(spec)
            if args.toy:
                base = make_surrogate(args, n_layers, (embed, layers, norm, lm_head),
                                      freeze=False)
                u_embed, u_layers, u_norm, u_head = (base["embed"], base["layers"],
                                                     base["norm"], base["lm_head"])
            else:
                a = argparse.Namespace(**vars(args))
                u_embed, u_layers, u_norm, u_head, u_rotary, _ = build_modules(a)
            head, cloud_l, tail, sa, ra = split_at(u_layers, args.depths[0], n_layers)
            losses = run_split_training(u_embed, head, cloud_l, tail, u_norm,
                                        u_head, rotary, train_blocks, args,
                                        defense, args.utility_steps, "full")
            utility.append({"defense": spec, "depth": sa,
                            "steps": args.utility_steps,
                            "first_loss": losses[0], "final_loss": losses[-1],
                            "curve": losses})
            print(f"[utility] defense={spec}: {losses[0]:.4f} -> {losses[-1]:.4f}")
            del u_embed, u_layers, u_norm, u_head
            if not args.toy:
                del u_rotary
            gc.collect()
            torch.cuda.empty_cache()

    out = {
        "config": {"model": "toy" if args.toy else args.model,
                   "n_layers": n_layers, "depths": args.depths,
                   "configs": args.configs, "defenses": args.defenses,
                   "docs": args.docs, "seeds": args.seeds,
                   "seq_len": args.seq_len, "rounds": args.rounds,
                   "train_steps": args.train_steps,
                   "utility_steps": args.utility_steps,
                   "dtype": args.dtype, "device": args.device,
                   "quick": args.quick},
        "threat_model": "semi-honest cloud observes boundary activation h* and "
                        "gradient dL/dh; attacker uses the public base "
                        "checkpoint as surrogate + self-generated pseudo-labels "
                        "(results are a LOWER BOUND on privacy risk)",
        "summary": summary,
        "utility": utility,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    _write_training_status(state="done", result_file=args.output)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
