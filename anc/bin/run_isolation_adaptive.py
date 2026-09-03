#!/usr/bin/env python3
"""Isolation experiment: adaptive self-monitoring defense on the v13 stack.

Runs training on the two-node split with the full v13 defense stack and watches
the attacker in real time: every --eval-interval steps a quick fresh probe
scores held-out recovery (trajectory logged per window); recovery excess over
the majority control beyond --tripwire-pct (default 5.0) escalates defense
actions (raise boundary noise +0.05, add chaff +16, reset the cloud session);
utility delta beyond --utility-gate (default 0.35) widens the boundary to the
next width in --width-ladder (64 -> 128 -> 256) with a fresh encoder/session,
continuing the step budget. A postmortem compares the best attacker's decoded
text against the source (token accuracy, word exact-match, normalized
Levenshtein similarity). The attacker never sees inputs, only the released
(gauged) frames.
"""

from __future__ import annotations

import argparse
import json
import math
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "split-training"))

from privacy_runtime.latent_native import (
    LatentPrivacyConfig, assert_ucn_latent_only, attacker_loss,
    build_latent_native_split, defender_privacy_loss, random_orthogonal,
)
from privacy_runtime.latent_protocol import RemoteLatentCloud
from privacy_runtime.ratchet_v2 import derive_permutation
from split_trainer import make_layer_kwargs, run_layer_stack


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def levenshtein_ratio(a: str, b: str) -> float:
    """Normalized edit similarity in [0,1] (1 = identical)."""
    if not a and not b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i]
        for j in range(1, lb + 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1,
                           prev[j - 1] + (a[i - 1] != b[j - 1])))
        prev = cur
    return 1.0 - prev[-1] / max(la, lb)


def load_blocks(tokenizer, corpus_path: Path, seq_len: int, max_blocks: int):
    text = corpus_path.read_text(errors="replace")
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    width = seq_len + 1
    blocks = [ids[i:i + width] for i in range(0, len(ids) - width + 1, width)]
    return blocks[:max_blocks]


def main() -> int:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cloud-url", required=True)
    ap.add_argument("--cloud-tls-ca", required=True)
    ap.add_argument("--cloud-kind", default="monomial_moe_radial")
    ap.add_argument("--cloud-experts", type=int, default=8)
    ap.add_argument("--cloud-layers", type=int, default=2)
    ap.add_argument("--cloud-hidden", type=int, default=0,
                    help="internal latent width for the invariant_mlp_deep "
                         "cloud kind (0 = unset)")
    ap.add_argument("--utility-reference", choices=("skip", "full"),
                    default="skip",
                    help="utility baseline: skip-the-middle (small-surrogate "
                         "cells) or the intact full model (50/50 cells)")
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--views", type=int, default=1,
                    help="K complementary sparse views (item 5): the encoder "
                         "emits K*latent_dim, view k = dims [k*D:(k+1)*D] "
                         "goes to its own cloud under its own rotation; one "
                         "shared row permutation so the collusion (union) "
                         "probe is evaluated conservatively")
    ap.add_argument("--mixup-lambda", type=float, default=0.0,
                    help="secret mixup (item 6): real rows released as "
                         "lam*z + (1-lam)*z_decoy with decoys recycled from "
                         "the chaff pool; TLN corrects the return via an "
                         "extra decoy forward (approximate: the cloud is "
                         "nonlinear).  0 disables")
    ap.add_argument("--width-ladder", default="64,128,256",
                    help="widths tried on utility-gate breach")
    ap.add_argument("--noise-multiplier", type=float, default=0.35)
    ap.add_argument("--clip-norm", type=float, default=1.0)
    ap.add_argument("--split-after", type=int, default=21)
    ap.add_argument("--resume-after", type=int, default=26)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--train-blocks", type=int, default=256)
    ap.add_argument("--eval-blocks", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--attack-steps", type=int, default=64)
    ap.add_argument("--postmortem-views", type=int, default=128,
                    help="distinct train views (x2 passes) for the final "
                         "best-effort postmortem probe; raise to strengthen "
                         "the instrument of record for privacy claims")
    ap.add_argument("--inference-blocks", type=int, default=0,
                    help="if > 0, run an inference-mode evaluation at phase "
                         "end: composed-vs-intact perplexity on N held-out "
                         "blocks (forward-only) and greedy generation samples")
    ap.add_argument("--inference-samples", type=int, default=3,
                    help="greedy generation samples in the inference eval")
    ap.add_argument("--save-bundle", default=None,
                    help="at phase end, dump the released wire frames + "
                         "aligned token labels (train/eval) as a .pt bundle "
                         "in the v5/v6 format the proven attacker/ arms read")
    ap.add_argument("--bundle-eval-blocks", type=int, default=64,
                    help="held-out blocks in the bundle's eval split; larger "
                         "n shrinks the Wilson/Bonferroni width for "
                         "decisive attacker readings")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--adversary-strength", type=float, default=1.0)
    ap.add_argument("--chaff-tokens", type=int, default=48)
    ap.add_argument("--remote-grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--tripwire-pct", type=float, default=5.0,
                    help="recovery excess over majority that triggers actions")
    ap.add_argument("--utility-gate", type=float, default=0.35)
    ap.add_argument("--utility-grace-steps", type=int, default=0,
                    help="per-phase steps before the utility gate arms; a "
                         "fresh width starts untrained, so judging it at the "
                         "first window measures the transient, not the "
                         "ceiling")
    ap.add_argument("--mine-beta", type=float, default=0.0,
                    help="if > 0, train a MINE statistics network alongside "
                         "the attackers and add beta * I_hat(z; x_embed) to "
                         "the defender loss; the estimate is logged per "
                         "window as the run's explicit MI budget")
    ap.add_argument("--backward-probe", action="store_true",
                    help="also train the window probe on the RETURNED "
                         "boundary gradients (the backward attack surface): "
                         "training frames are cached from the training loop "
                         "(no extra cloud traffic); held-out gradient frames "
                         "for scoring are generated once per window")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-ungauged", action="store_true",
                    help="start with rotation/permutation OFF (broken "
                         "posture); the controller's first tripwire action "
                         "enables them, exercising the full response loop")
    args = ap.parse_args()

    ladder = [int(v) for v in args.width_ladder.split(",")]
    if args.latent_dim not in ladder:
        ladder.insert(0, args.latent_dim)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    def connect_views(base_url, width):
        """K view clouds on consecutive ports (K=1: the single connection)."""
        host, port = base_url.rsplit(":", 1)
        return [RemoteLatentCloud(host + ":" + str(int(port) + k), width,
                                  args.lr, 0.0, args.cloud_kind, args.seed,
                                  tls_ca=args.cloud_tls_ca,
                                  cloud_experts=args.cloud_experts,
                                  cloud_layers=args.cloud_layers,
                                  cloud_hidden=args.cloud_hidden)
                for k in range(args.views)]

    remotes = connect_views(args.cloud_url, args.latent_dim)

    # Loader note: this plain CPU-then-.to(device) path transiently approaches
    # ~1.6x the weights on unified-memory hosts — gated by bin/spark_preflight.sh.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    core = model.model
    layers = core.layers
    hidden_dim = getattr(model.config, "hidden_size", None) \
        or model.config.text_config.hidden_size
    hidden_dim = int(hidden_dim)
    blocks = load_blocks(tokenizer, Path(args.corpus), args.seq_len,
                         args.train_blocks + args.eval_blocks)
    train_blocks = blocks[:args.train_blocks]
    eval_blocks = blocks[args.train_blocks:]

    attack_class_tokens = sorted(set(
        token for block in train_blocks for token in block[1:]))
    attack_lookup = torch.full((int(model.config.vocab_size),), -1,
                               dtype=torch.long, device=device)
    attack_lookup[torch.tensor(attack_class_tokens, device=device)] = (
        torch.arange(len(attack_class_tokens), device=device))

    def attack_classes(labels):
        return attack_lookup[labels]

    def tensors(block):
        ids = torch.tensor(block, dtype=torch.long, device=device)[None]
        return ids[:, :-1], ids[:, 1:]

    def base_states(input_ids):
        positions = torch.arange(input_ids.shape[1], device=device)[None]
        with torch.no_grad():
            hidden = core.embed_tokens(input_ids)
            kwargs = make_layer_kwargs(getattr(core, "rotary_emb", None),
                                       hidden, positions,
                                       type("Args", (), {"attn_impl": "sdpa"})())
            prefix = run_layer_stack(layers[:args.split_after + 1], hidden,
                                     kwargs)
            middle = run_layer_stack(
                layers[args.split_after + 1:args.resume_after], prefix, kwargs)
        return prefix.float(), middle.float(), positions

    def tail_loss(middle_float, positions, labels):
        hidden = middle_float.to(dtype)
        kwargs = make_layer_kwargs(getattr(core, "rotary_emb", None), hidden,
                                   positions,
                                   type("Args", (), {"attn_impl": "sdpa"})())
        hidden = run_layer_stack(layers[args.resume_after:], hidden, kwargs)
        logits = model.lm_head(core.norm(hidden)).float()
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten())

    log = []
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def write_artifact(postmortem=None):
        """Persist after every milestone event: a host wedge mid-run must
        not lose the trajectory (tln memory-wedge, 2026-08-19)."""
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(
            {"schema": "dtraining.isolation_adaptive.v1",
             "args": vars(args), "log": log,
             "postmortem": postmortem}, indent=1))
        tmp.rename(out_path)

    def emit(event, **fields):
        record = {"event": event, "time": time.time(), **fields}
        log.append(record)
        print(json.dumps(record), flush=True)
        if event in ("baseline_loss", "split_plan", "window", "action",
                     "phase_change", "done"):
            write_artifact()

    # phase runner (per width)
    full_ref = {}  # lazily-computed frozen full-model loss (utility-reference)

    def run_phase(width, remote_conns, step_budget, global_step0):
        view_w = width
        total_w = args.views * width
        cfg = LatentPrivacyConfig(
            hidden_dim=hidden_dim, latent_dim=total_w,
            cloud_layers=args.cloud_layers, cloud_heads=4,
            clip_norm=args.clip_norm, noise_multiplier=args.noise_multiplier,
            adversary_strength=args.adversary_strength,
            cloud_kind=args.cloud_kind, cloud_experts=args.cloud_experts,
            cloud_hidden=args.cloud_hidden)
        cfg_view = LatentPrivacyConfig(
            hidden_dim=hidden_dim, latent_dim=view_w,
            cloud_layers=args.cloud_layers, cloud_heads=4,
            clip_norm=args.clip_norm, noise_multiplier=args.noise_multiplier,
            adversary_strength=args.adversary_strength,
            cloud_kind=args.cloud_kind, cloud_experts=args.cloud_experts,
            cloud_hidden=args.cloud_hidden)
        tln, ucn, attackers = build_latent_native_split(
            cfg, vocab_size=len(attack_class_tokens))
        assert_ucn_latent_only(ucn, total_w, hidden_dim,
                                 allowed_internal_width=args.cloud_hidden
                                 or None)
        tln.to(device); attackers.to(device)
        attacker_opt = torch.optim.AdamW(attackers.parameters(), lr=args.lr)
        defender_opt = torch.optim.AdamW(tln.parameters(), lr=args.lr)

        # MINE statistics network (cell 9): estimates I(z ; x_embed) in nats
        # from the released (post-noise, pre-gauge) latents — the gauge is
        # bijective given the key, so the pre-gauge estimate is the canonical
        # leakage quantity. Logged per window as the run's MI budget; with
        # --mine-beta > 0 it is also an explicit MI penalty in the defender loss.
        mine_net = mine_opt = None
        if args.mine_beta > 0:
            mine_net = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim + total_w, 512), torch.nn.GELU(),
                torch.nn.Linear(512, 512), torch.nn.GELU(),
                torch.nn.Linear(512, 1)).to(device)
            mine_opt = torch.optim.AdamW(mine_net.parameters(), lr=args.lr)

        def mine_estimate(x_rows, z_rows):
            order = torch.randperm(z_rows.shape[0], device=z_rows.device)
            joint = torch.cat([x_rows, z_rows], dim=-1)
            marginal = torch.cat([x_rows, z_rows[order]], dim=-1)
            t_joint = mine_net(joint).squeeze(-1)
            t_marginal = mine_net(marginal).squeeze(-1)
            return (t_joint.mean()
                    - (torch.logsumexp(t_marginal, dim=0)
                       - math.log(t_marginal.shape[0])))
        noise = {"value": args.noise_multiplier}
        chaff_n = {"value": args.chaff_tokens}
        gauges = {"on": not args.start_ungauged}
        chaff_pool = None
        bw_cache = []  # (returned gradient, aligned token classes) per step

        def chaff_push(latent, labels, hidden):
            nonlocal chaff_pool
            rows = torch.cat([latent.detach().reshape(-1, latent.shape[-1])
                              .float().cpu(),
                              labels.detach().reshape(-1, 1).float().cpu(),
                              hidden.detach().reshape(-1, hidden.shape[-1])
                              .float().cpu()], dim=1)
            chaff_pool = rows if chaff_pool is None else torch.cat(
                [chaff_pool, rows], dim=0)
            chaff_pool = chaff_pool[-8192:]

        def chaff_sample():
            if chaff_pool is None or chaff_pool.shape[0] < chaff_n["value"]:
                return None
            index = derive_permutation(secrets.token_bytes(16), 0,
                                       chaff_pool.shape[0])[:chaff_n["value"]]
            rows = chaff_pool[index].to(device)
            return (rows[None, :, :total_w],
                    rows[None, :, total_w].long(),
                    rows[None, :, total_w + 1:])

        def release(latent, chaff=None):
            """Mixup (item 6) -> chaff -> shared row permutation -> K
            independent per-view rotations -> views [K, B, R, view_w].

            One permutation is shared across views so the collusion (union)
            probe gets row-aligned frames for free — conservative (it can
            only help the attacker). Rotations stay per-view: a single cloud
            never sees another view's basis.
            """
            value = latent
            n_real = latent.shape[-2]
            decoy_views = None
            if args.mixup_lambda > 0 and chaff_pool is not None \
                    and chaff_pool.shape[0] >= n_real:
                index = derive_permutation(secrets.token_bytes(16), 0,
                                           chaff_pool.shape[0])[:n_real]
                decoy = chaff_pool[index, :total_w].to(
                    device=device, dtype=value.dtype)[None]
                lam = args.mixup_lambda
                mixed = lam * value + (1.0 - lam) * decoy
                # decoy views under the SAME view rotations (identity perm)
                # so the return can be corrected row-wise on TLN
                dv = decoy.reshape(decoy.shape[0], decoy.shape[1],
                                   args.views, view_w).permute(2, 0, 1, 3)
                decoy_views = dv  # rotations applied below, after transform
                value = mixed
            if chaff is not None:
                value = torch.cat([value, chaff[0].to(value.dtype)], dim=-2)
            if gauges["on"]:
                permutation = derive_permutation(
                    secrets.token_bytes(16), 0, value.shape[-2]).to(
                        latent.device)
                inverse = torch.argsort(permutation)
                value = value[:, permutation]
                rotations = torch.stack(
                    [random_orthogonal(value, view_w)
                     for _ in range(args.views)])
            else:
                rotations = torch.eye(view_w, device=value.device,
                                      dtype=value.dtype).expand(
                                          args.views, -1, -1).contiguous()
                permutation = torch.arange(value.shape[-2],
                                           device=value.device)
                inverse = permutation
            if (chaff is not None
                    and value.shape[-2] != n_real + chaff[0].shape[-2]):
                raise RuntimeError("released frame lost chaff rows")
            views = value.reshape(value.shape[0], value.shape[1],
                                  args.views, view_w).permute(2, 0, 1, 3)
            views = torch.einsum("kbrw,kwv->kbrv", views, rotations)
            if decoy_views is not None:
                decoy_views = torch.einsum("kbrw,kwv->kbrv", decoy_views,
                                           rotations)
            return views, {"rotations": rotations, "inverse": inverse,
                           "n_real": n_real, "decoy_views": decoy_views}

        def restore(outs, meta):
            """[K, B, R, view_w] gauged cloud outputs -> [B, T, total_w]
            real rows in canonical order (ungauge + shared unpermute)."""
            value = torch.einsum("kbrw,kwv->kbrv", outs,
                                     meta["rotations"].transpose(-1, -2))
            value = value[:, :, meta["inverse"]]
            value = value[:, :, :meta["n_real"]]
            return value.permute(1, 2, 0, 3).reshape(
                outs.shape[1], meta["n_real"], total_w)

        def restore_decoy(outs, meta):
            """Decoy forwards carry no row permutation — ungauge only."""
            value = torch.einsum("kbrw,kwv->kbrv", outs,
                                     meta["rotations"].transpose(-1, -2))
            return value.permute(1, 2, 0, 3).reshape(
                outs.shape[1], outs.shape[2], total_w)

        def cloud_forward(views, training):
            outs, mbs = [], []
            for k, conn in enumerate(remote_conns):
                out, mb, _ = conn.forward(views[k].contiguous(),
                                          training=training)
                outs.append(out)
                mbs.append(mb)
            return outs, mbs  # leaf list: callers stack; grads land on leaves

        def cloud_backward(mbs, grad_views):
            grads = []
            for k, conn in enumerate(remote_conns):
                grads.append(conn.backward(
                    mbs[k], grad_views[k].contiguous()))
            return torch.stack(grads)

        def clip_remote_gradient(value):
            if not torch.isfinite(value).all():
                raise RuntimeError("UCN returned a non-finite gradient")
            flat = value.float().reshape(-1, value.shape[-1])
            norms = flat.norm(2, dim=-1, keepdim=True)
            scales = (args.remote_grad_clip
                      / norms.clamp_min(1e-12)).clamp(max=1.0)
            return (flat * scales).reshape_as(value).to(value.dtype)

        def as_union(views):
            """[K, B, R, W] -> [B, R, total_w] row-aligned collusion view."""
            return views.permute(1, 2, 0, 3).reshape(
                views.shape[1], views.shape[2], total_w)

        def quick_probe_excess():
            """Fresh small probes: the union (collusion) attacker on the
            row-aligned concatenation of all views, plus one per-view
            attacker (what a single compromised cloud can do alone).
            Returns (union_recovery, max_view_recovery, majority)."""
            _, _, union_probe = build_latent_native_split(
                cfg, len(attack_class_tokens))
            union_probe = union_probe.to(device)
            union_opt = torch.optim.AdamW(union_probe.parameters(),
                                          lr=args.lr)
            view_probe = view_opt = None
            if args.views > 1:
                _, _, view_probe = build_latent_native_split(
                    cfg_view, len(attack_class_tokens))
                view_probe = view_probe.to(device)
                view_opt = torch.optim.AdamW(view_probe.parameters(),
                                             lr=args.lr)
            for attack_step in range(args.attack_steps):
                input_ids, labels = tensors(
                    train_blocks[attack_step % len(train_blocks)])
                prefix, _, _ = base_states(input_ids)
                with torch.no_grad():
                    latent, _ = tln.encode(prefix, secrets.token_hex(16))
                    chaff = chaff_sample()
                    rel, meta = release(latent, chaff)
                    if chaff is not None:
                        lab = torch.cat([labels, chaff[1]], dim=1)
                        hid = torch.cat([prefix, chaff[2]], dim=1)
                    else:
                        lab = labels
                        hid = prefix
                    perm = torch.argsort(meta["inverse"])
                    lab = lab[:, perm]
                    hid = hid[:, perm]
                    union = as_union(rel)
                props = torch.tensor([attack_step % 2], device=device)
                union_opt.zero_grad(set_to_none=True)
                loss_u, _ = attacker_loss(
                    union_probe, union, attack_classes(lab), props, hid)
                loss_u.backward()
                union_opt.step()
                if view_probe is not None:
                    for k in range(args.views):
                        view_opt.zero_grad(set_to_none=True)
                        loss_v, _ = attacker_loss(
                            view_probe, rel[k], attack_classes(lab), props,
                            hid)
                        loss_v.backward()
                        view_opt.step()
            correct_u = 0
            correct_v = [0] * args.views
            total = 0
            with torch.no_grad():
                for block in eval_blocks[:16]:
                    input_ids, labels = tensors(block)
                    prefix, _, _ = base_states(input_ids)
                    latent, _ = tln.encode(prefix, secrets.token_hex(16))
                    chaff = chaff_sample()
                    rel, meta = release(latent, chaff)
                    lab = (torch.cat([labels, chaff[1]], dim=1)
                           if chaff is not None else labels)
                    lab = lab[:, torch.argsort(meta["inverse"])]
                    cls = attack_classes(lab)
                    out_u = union_probe(as_union(rel))["token"].argmax(-1)
                    correct_u += int(((out_u == cls) & (cls >= 0)).sum())
                    for k in range(args.views):
                        if view_probe is None:
                            break
                        out_v = view_probe(rel[k])["token"].argmax(-1)
                        correct_v[k] += int(((out_v == cls)
                                             & (cls >= 0)).sum())
                    total += int((cls >= 0).sum())
            best_view = max(correct_v) if view_probe is not None else correct_u
            all_train = [t for block in train_blocks for t in block[1:]]
            majority_tok = max(set(all_train), key=all_train.count)
            majority = sum(t == majority_tok for block in eval_blocks[:16]
                           for t in block[1:])
            total_rows = sum(len(b) - 1 for b in eval_blocks[:16])
            return (100.0 * correct_u / max(1, total),
                    100.0 * best_view / max(1, total),
                    100.0 * majority / max(1, total_rows))

        def quick_utility():
            cand, base = [], []
            if args.utility_reference == "full" and "value" not in full_ref:
                losses = []
                for block in eval_blocks[:8]:
                    input_ids, labels = tensors(block)
                    with torch.no_grad():
                        logits = model(input_ids).logits.float()
                        losses.append(float(F.cross_entropy(
                            logits.flatten(0, 1), labels.flatten()).item()))
                full_ref["value"] = mean(losses)
            for block in eval_blocks[:8]:
                input_ids, labels = tensors(block)
                prefix, _, positions = base_states(input_ids)
                with torch.no_grad():
                    if args.utility_reference == "full":
                        base.append(full_ref["value"])
                    else:
                        base.append(float(tail_loss(prefix, positions,
                                                    labels).item()))
                    latent, _ = tln.encode(prefix, secrets.token_hex(16))
                    rel, meta = release(latent, chaff_sample())
                    outs, _ = cloud_forward(rel, training=False)
                    ret = restore(torch.stack(outs), meta)
                    if meta["decoy_views"] is not None:
                        dec_outs, _ = cloud_forward(meta["decoy_views"],
                                                    training=False)
                        lam = args.mixup_lambda
                        ret = (ret - (1.0 - lam) * restore_decoy(
                            torch.stack(dec_outs), meta)) / lam
                    hat, _ = tln.decode(ret, secrets.token_hex(16),
                                          residual=prefix)
                    cand.append(float(tail_loss(hat, positions,
                                                labels).item()))
            return mean(cand) - mean(base)

        def quick_mine():
            """MINE estimate on held-out blocks with the current statistics
            network — the logged MI budget, in nats per boundary row."""
            vals = []
            with torch.no_grad():
                for block in eval_blocks[:4]:
                    input_ids, _ = tensors(block)
                    prefix, _, _ = base_states(input_ids)
                    latent, _ = tln.encode(prefix, secrets.token_hex(16))
                    x = core.embed_tokens(input_ids).reshape(
                        -1, hidden_dim).float()
                    z = latent.reshape(-1, total_w).float()
                    vals.append(float(mine_estimate(x, z)))
            return mean(vals)

        def backward_probe_excess():
            """Fresh probe on the RETURNED boundary gradients (the backward
            attack surface).  Training frames come from the training loop's
            own cache (no extra cloud traffic); held-out gradient frames are
            generated once per window through the normal defended path."""
            if len(bw_cache) < 8:
                return None
            _, _, probe = build_latent_native_split(
                cfg, len(attack_class_tokens))
            probe = probe.to(device)
            probe_opt = torch.optim.AdamW(probe.parameters(), lr=args.lr)
            for attack_step in range(args.attack_steps):
                grad_cpu, cls_cpu = bw_cache[attack_step % len(bw_cache)]
                grad = grad_cpu.to(device)
                cls = cls_cpu.to(device)
                probe_opt.zero_grad(set_to_none=True)
                loss_p = F.cross_entropy(
                    probe(grad)["token"].flatten(0, 1), cls.flatten(),
                    ignore_index=-1)
                loss_p.backward()
                probe_opt.step()
            correct = total = 0
            for block in eval_blocks[:16]:
                input_ids, labels = tensors(block)
                prefix, middle_target, positions = base_states(input_ids)
                latent, _ = tln.encode(prefix, secrets.token_hex(16))
                chaff = chaff_sample()
                rel, meta = release(latent, chaff)
                outs, mbs = cloud_forward(rel.detach(), training=True)
                stacked = torch.stack(outs)
                trusted_return = restore(stacked, meta)
                if meta["decoy_views"] is not None:
                    dec_outs, _ = cloud_forward(meta["decoy_views"],
                                                training=False)
                    lam = args.mixup_lambda
                    trusted_return = (trusted_return - (1.0 - lam)
                                      * restore_decoy(
                                          torch.stack(dec_outs), meta)) / lam
                restored, _ = tln.decode(trusted_return,
                                           secrets.token_hex(16),
                                           residual=prefix)
                loss_ev = F.mse_loss(
                    F.layer_norm(restored, (hidden_dim,)),
                    F.layer_norm(middle_target, (hidden_dim,))) \
                    + tail_loss(restored, positions, labels)
                # populates the per-view output leaves' .grad; tln params
                # also accumulate here, cleared by the next step's zero_grad
                loss_ev.backward()
                grad_ev = cloud_backward(
                    mbs, torch.stack([leaf.grad for leaf in outs]))
                with torch.no_grad():
                    out = probe(as_union(grad_ev.detach()))["token"].argmax(-1)
                    lab = (torch.cat([labels, chaff[1]], dim=1)
                           if chaff is not None else labels)
                    lab = lab[:, torch.argsort(meta["inverse"])]
                    cls = attack_classes(lab)
                    correct += int(((out == cls) & (cls >= 0)).sum())
                    total += int((cls >= 0).sum())
            all_train = [t for block in train_blocks for t in block[1:]]
            majority_tok = max(set(all_train), key=all_train.count)
            majority = sum(t == majority_tok for block in eval_blocks[:16]
                           for t in block[1:])
            return (100.0 * correct / max(1, total),
                    100.0 * majority / max(
                        1, sum(len(b) - 1 for b in eval_blocks[:16])))

        phase_outcome = "completed"
        global_step = global_step0
        remaining = step_budget

        def _count_params(modules):
            seen, total = set(), 0
            for module in modules:
                for parameter in module.parameters():
                    if id(parameter) not in seen:
                        seen.add(id(parameter))
                        total += parameter.numel()
            return total

        prefix_params = _count_params([*layers[:args.split_after + 1]])
        tail_params = _count_params([*layers[args.resume_after:]])
        dense_params = _count_params([core.embed_tokens, core.norm,
                                      model.lm_head])
        segment_params = _count_params(
            [layers[args.split_after + 1:args.resume_after]])
        # MoE: only the routed+shared experts touched per token burn FLOPs —
        # apply the active fraction to ALL layer stacks symmetrically (the
        # embedding and LM head are dense and always fully active).
        text_cfg = getattr(model.config, "text_config", model.config)
        n_exp = getattr(text_cfg, "num_experts", None)
        if n_exp:
            per_tok = getattr(text_cfg, "num_experts_per_tok", 1)
            shared = getattr(text_cfg, "shared_expert_intermediate_size", 0)
            moe_int = getattr(text_cfg, "moe_intermediate_size", 0)
            active_frac = min(1.0, (per_tok * moe_int + shared)
                              / (n_exp * moe_int + shared))
        else:
            active_frac = 1.0
        # training FLOPs/token ~ 6 x active params (matmul-dominated, seq=32)
        gf_tln = 6 * ((prefix_params + tail_params) * active_frac
                        + dense_params) / 1e9
        gf_segment = 6 * segment_params * active_frac / 1e9
        gf_cloud = 1.79 * (args.cloud_layers / 28) \
            * (max(args.cloud_hidden, 64) / 2048) ** 2  # deep-kind calibration
        emit("split_plan", width=view_w, views=args.views,
             boundary_after_layer=args.split_after,
             covered_layers=args.resume_after - args.split_after - 1,
             n_layers=len(layers), active_frac=round(active_frac, 4),
             tln_params=int(prefix_params + tail_params + dense_params),
             segment_params=int(segment_params),
             gflop_token_tln=round(gf_tln, 3),
             gflop_token_segment=round(gf_segment, 3),
             gflop_token_cloud=round(gf_cloud, 3),
             cloud_share_pct=round(100 * gf_cloud
                                   / (gf_tln + gf_cloud), 1))
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        while remaining > 0:
            window = min(args.eval_interval, remaining)
            for _ in range(window):
                input_ids, labels = tensors(
                    train_blocks[global_step % len(train_blocks)])
                prefix, middle_target, positions = base_states(input_ids)
                properties = torch.tensor([global_step % 2], device=device)
                nonce = secrets.token_hex(16)
                latent, _ = tln.encode(prefix, nonce)
                chaff = chaff_sample()
                released, meta = release(latent, chaff)
                chaff_push(latent, labels, prefix)
                if chaff is not None:
                    att_labels = torch.cat([labels, chaff[1]], dim=1)
                    att_hidden = torch.cat([prefix.detach(), chaff[2]], dim=1)
                else:
                    att_labels = labels
                    att_hidden = prefix.detach()
                perm = torch.argsort(meta["inverse"])
                att_labels = att_labels[:, perm]
                att_hidden = att_hidden[:, perm]
                att_cls = attack_classes(att_labels)
                for _ in range(3):
                    attacker_opt.zero_grad(set_to_none=True)
                    loss_a, _ = attacker_loss(
                        attackers, as_union(released).detach(), att_cls,
                        properties, att_hidden)
                    loss_a.backward()
                    attacker_opt.step()
                if mine_net is not None:
                    emb_rows = core.embed_tokens(input_ids).detach().reshape(
                        -1, hidden_dim).float()
                    z_rows = latent.detach().reshape(-1, total_w).float()
                    for _ in range(2):
                        mine_opt.zero_grad(set_to_none=True)
                        mine_loss = -mine_estimate(emb_rows, z_rows)
                        mine_loss.backward()
                        mine_opt.step()
                defender_opt.zero_grad(set_to_none=True)
                for parameter in attackers.parameters():
                    parameter.requires_grad_(False)
                outs, mbs = cloud_forward(released.detach(), training=True)
                stacked = torch.stack(outs)
                trusted_return = restore(stacked, meta)
                if meta["decoy_views"] is not None:
                    dec_outs, _ = cloud_forward(meta["decoy_views"],
                                                training=False)
                    lam = args.mixup_lambda
                    trusted_return = (trusted_return - (1.0 - lam)
                                      * restore_decoy(
                                          torch.stack(dec_outs), meta)) / lam
                restored, _ = tln.decode(trusted_return, nonce,
                                           residual=prefix)
                distill = F.mse_loss(
                    F.layer_norm(restored, (hidden_dim,)),
                    F.layer_norm(middle_target, (hidden_dim,))) \
                    + 0.001 * F.mse_loss(restored, middle_target)
                language = tail_loss(restored, positions, labels)
                if global_step >= args.warmup_steps:
                    privacy = defender_privacy_loss(
                        attackers, as_union(released), att_cls, properties,
                        att_hidden, cfg.adversary_strength)
                else:
                    privacy = latent.sum() * 0.0
                total = distill + language + privacy
                if mine_net is not None:
                    for parameter in mine_net.parameters():
                        parameter.requires_grad_(False)
                    z_live = latent.reshape(-1, total_w).float()
                    total = total + args.mine_beta * mine_estimate(
                        emb_rows, z_live)
                total.backward(retain_graph=True)
                if mine_net is not None:
                    for parameter in mine_net.parameters():
                        parameter.requires_grad_(True)
                grad_views = torch.stack([leaf.grad for leaf in outs])
                if not torch.isfinite(grad_views).all():
                    raise RuntimeError(
                        "local graph produced a non-finite cloud gradient "
                        f"(width={view_w} views={args.views} "
                        f"step={global_step} loss={float(total):.4f})")
                grad_in = cloud_backward(mbs, grad_views)
                if args.backward_probe:
                    bw_cache.append((as_union(grad_in.detach()).float().cpu(),
                                     att_cls.detach().cpu()))
                    del bw_cache[:-64]
                released.backward(clip_remote_gradient(grad_in))
                for conn in remote_conns:
                    conn.step()
                defender_opt.step()
                for parameter in attackers.parameters():
                    parameter.requires_grad_(True)
                global_step += 1
            remaining -= window

            recovery_u, recovery_v, majority = quick_probe_excess()
            excess = recovery_u - majority  # controller: union (strongest)
            util_delta = quick_utility()
            mine_nats = quick_mine() if mine_net is not None else None
            bw = backward_probe_excess() if args.backward_probe else None
            emit("window", step=global_step, width=view_w,
                 recovery_pct=round(recovery_u, 3), majority_pct=round(majority, 3),
                 excess_pp=round(excess, 3), utility_delta=round(util_delta, 4),
                 view_recovery_pct=None if args.views == 1 else round(
                     recovery_v, 3),
                 view_excess_pp=None if args.views == 1 else round(
                     recovery_v - majority, 3),
                 views=args.views, mixup=args.mixup_lambda,
                 mine_nats=None if mine_nats is None else round(mine_nats, 4),
                 backward_recovery_pct=None if bw is None else round(bw[0], 3),
                 backward_excess_pp=None if bw is None else round(
                     bw[0] - bw[1], 3),
                 tln_peak_gb=None if device != "cuda" else round(
                     torch.cuda.max_memory_allocated() / 1e9, 2),
                 noise=noise["value"], chaff=chaff_n["value"])

            if excess > args.tripwire_pct:
                if not gauges["on"]:
                    gauges["on"] = True
                    emit("action", reason="tripwire", action="enable_gauges",
                         step=global_step)
                elif noise["value"] < 0.60:
                    noise["value"] = round(noise["value"] + 0.05, 3)
                    tln.dp.forward_noise = noise["value"]
                    tln.dp.return_noise = noise["value"]
                    emit("action", reason="tripwire", action="noise_up",
                         new_noise=noise["value"], step=global_step)
                elif chaff_n["value"] < 96:
                    chaff_n["value"] += 16
                    emit("action", reason="tripwire", action="chaff_up",
                         new_chaff=chaff_n["value"], step=global_step)
                else:
                    emit("action", reason="tripwire",
                         action="session_reset_requested", step=global_step)
                    phase_outcome = "session_reset"
                    break
            if util_delta > args.utility_gate and (
                    global_step - global_step0) >= args.utility_grace_steps:
                emit("action", reason="utility_gate", action="widen_width",
                     from_width=width, step=global_step)
                phase_outcome = "widen"
                break

        def composed_next_logits(ids):
            """One full defended forward pass for the current sequence."""
            positions = torch.arange(ids.shape[1], device=device)[None]
            hidden = core.embed_tokens(ids)
            kwargs = make_layer_kwargs(getattr(core, "rotary_emb", None),
                                       hidden, positions,
                                       type("Args", (), {"attn_impl": "sdpa"})())
            prefix = run_layer_stack(layers[:args.split_after + 1], hidden,
                                     kwargs)
            latent, _ = tln.encode(prefix.float(), secrets.token_hex(16))
            rel, meta = release(latent, chaff_sample())
            outs, _ = cloud_forward(rel, training=False)
            ret = restore(torch.stack(outs), meta)
            if meta["decoy_views"] is not None:
                dec_outs, _ = cloud_forward(meta["decoy_views"],
                                            training=False)
                lam = args.mixup_lambda
                ret = (ret - (1.0 - lam) * restore_decoy(
                    torch.stack(dec_outs), meta)) / lam
            hat, _ = tln.decode(ret, secrets.token_hex(16),
                                  residual=prefix.float())
            hidden = hat.to(dtype)
            kwargs = make_layer_kwargs(getattr(core, "rotary_emb", None),
                                       hidden, positions,
                                       type("Args", (), {"attn_impl": "sdpa"})())
            hidden = run_layer_stack(layers[args.resume_after:], hidden,
                                     kwargs)
            return model.lm_head(core.norm(hidden)).float()

        def inference_eval():
            if args.inference_blocks <= 0:
                return None
            cand, base = [], []
            for block in eval_blocks[:args.inference_blocks]:
                input_ids, labels = tensors(block)
                with torch.no_grad():
                    logits = model(input_ids).logits.float()
                    base.append(float(F.cross_entropy(
                        logits.flatten(0, 1), labels.flatten()).item()))
                    cand.append(float(F.cross_entropy(
                        composed_next_logits(input_ids).flatten(0, 1),
                        labels.flatten()).item()))
            samples = []
            for block in eval_blocks[:args.inference_samples]:
                ids = torch.tensor(block[:len(block) // 2],
                                   dtype=torch.long, device=device)[None]
                cur_c, cur_b = ids.clone(), ids.clone()
                with torch.no_grad():
                    for _ in range(24):
                        cur_c = torch.cat([cur_c, composed_next_logits(
                            cur_c)[:, -1:].argmax(-1)], dim=1)
                        cur_b = torch.cat([cur_b, model(cur_b).logits[
                            :, -1:].argmax(-1)], dim=1)
                samples.append({
                    "prompt": tokenizer.decode(ids[0].cpu().tolist())[-60:],
                    "composed": tokenizer.decode(
                        cur_c[0, ids.shape[1]:].cpu().tolist())[:160],
                    "intact": tokenizer.decode(
                        cur_b[0, ids.shape[1]:].cpu().tolist())[:160]})
            # cand/base are mean cross-entropy in NATS, not perplexity; emitting
            # them as PPL understates the gap (5.23/4.86 nats reads 1.08x; the
            # true perplexities e^5.23/e^4.86 = 186.8/129.0 give 1.44x).
            nll_composed, nll_intact = mean(cand), mean(base)
            return {"nll_composed_nats": round(nll_composed, 4),
                    "nll_intact_nats": round(nll_intact, 4),
                    "perplexity_composed": round(math.exp(nll_composed), 4),
                    "perplexity_intact": round(math.exp(nll_intact), 4),
                    "delta": round(nll_composed - nll_intact, 4),
                    "delta_unit": "nats",
                    "samples": samples}

        if args.inference_blocks > 0:
            emit("inference_eval", **inference_eval())

        if args.save_bundle:
            # Wire-fidelity bundle for the proven attacker arms (v5/v6 format):
            # released frames exactly as the cloud saw them, labels aligned to
            # the (possibly permuted) frame rows.
            train_wire, train_tokens, eval_wire, eval_tokens = [], [], [], []
            with torch.no_grad():
                for blocks, out_x, out_y in (
                        (train_blocks[:256], train_wire, train_tokens),
                        (eval_blocks[:args.bundle_eval_blocks],
                         eval_wire, eval_tokens)):
                    for block in blocks:
                        input_ids, labels = tensors(block)
                        prefix, _, _ = base_states(input_ids)
                        latent, _ = tln.encode(prefix, secrets.token_hex(16))
                        chaff = chaff_sample()
                        rel, meta = release(latent, chaff)
                        lab = (torch.cat([labels, chaff[1]], dim=1)
                               if chaff is not None else labels)
                        lab = lab[:, torch.argsort(meta["inverse"])]
                        out_x.append(as_union(rel)[0].float().cpu())
                        out_y.append(lab[0].cpu())
            torch.save({
                "train_wire": torch.stack(train_wire),
                "train_tokens": torch.stack(train_tokens),
                "eval_wire": torch.stack(eval_wire),
                "eval_tokens": torch.stack(eval_tokens),
                "meta": {"width": view_w, "views": args.views,
                         "noise": noise["value"], "chaff": chaff_n["value"],
                         "gauges": gauges["on"], "mixup": args.mixup_lambda,
                         "split_after": args.split_after,
                         "resume_after": args.resume_after,
                         "seed": args.seed, "steps_run": global_step},
            }, args.save_bundle)
            emit("bundle_saved", path=args.save_bundle,
                 train_rows=int(train_wire[0].shape[0]) * len(train_wire))
        return {"outcome": phase_outcome, "global_step": global_step}

    # Baseline eval once
    baseline_losses = []
    with torch.no_grad():
        for block in eval_blocks[:8]:
            input_ids, labels = tensors(block)
            logits = model(input_ids).logits.float()
            baseline_losses.append(float(F.cross_entropy(
                logits.flatten(0, 1), labels.flatten()).item()))
    emit("baseline_loss", value=round(mean(baseline_losses), 4))

    global_step = 0
    remaining = args.steps
    final_width = ladder[0]
    current_url = args.cloud_url
    for width in ladder:
        final_width = width
        result = run_phase(width, remotes, remaining, global_step)
        global_step = result["global_step"]
        remaining = args.steps - global_step
        if result["outcome"] == "widen" and width != ladder[-1]:
            next_width = ladder[ladder.index(width) + 1]
            emit("phase_change", from_width=width, to_width=next_width)
            for conn in remotes:
                conn.close()
            # advance from the CURRENT base port by the view count — the
            # next width's K view servers sit on the next K ports
            base, port = current_url.rsplit(":", 1)
            current_url = base + ":" + str(int(port) + args.views)
            remotes = connect_views(current_url, next_width)
            continue
        break

    emit("done", total_steps=global_step)

    # postmortem: attacker reconstruction vs source
    # Train the best-effort probe on the train views, decode eval blocks, and
    # compare decoded text against source. The attacker sees only the released
    # (gauged) frames — never the input. The probe is built at the FINAL
    # operating width and attacks the UNION (collusion) frame; mixup is
    # omitted here, which can only help the attacker — the readings are an
    # upper bound on what the mixed-frame attacker achieves.
    emit("postmortem_start")
    post_total_w = args.views * final_width
    cfg = LatentPrivacyConfig(
        hidden_dim=hidden_dim, latent_dim=post_total_w,
        cloud_layers=args.cloud_layers, cloud_heads=4,
        clip_norm=args.clip_norm, noise_multiplier=args.noise_multiplier,
        adversary_strength=args.adversary_strength,
        cloud_kind=args.cloud_kind, cloud_experts=args.cloud_experts,
        cloud_hidden=args.cloud_hidden)
    tln_f, _, probe = build_latent_native_split(cfg, len(attack_class_tokens))
    tln_f.to(device); probe = probe.to(device)
    probe_opt = torch.optim.AdamW(probe.parameters(), lr=args.lr)

    def release_f(latent):
        permutation = derive_permutation(
            secrets.token_bytes(16), 0, latent.shape[-2]).to(latent.device)
        value = latent[:, permutation]
        rotations = torch.stack(
            [random_orthogonal(value, final_width)
             for _ in range(args.views)])
        views = torch.einsum(
            "kbrw,kwv->kbrv",
            value.reshape(value.shape[0], value.shape[1], args.views,
                          final_width).permute(2, 0, 1, 3), rotations)
        union = views.permute(1, 2, 0, 3).reshape(
            value.shape[0], value.shape[1], post_total_w)
        return union, permutation

    train_view_count = min(args.postmortem_views, len(train_blocks))
    for attack_step in range(train_view_count * 2):
        input_ids, labels = tensors(train_blocks[attack_step % train_view_count])
        prefix, _, _ = base_states(input_ids)
        with torch.no_grad():
            latent, _ = tln_f.encode(prefix, secrets.token_hex(16))
            rel, perm = release_f(latent)
            lab = labels[:, perm]
        props = torch.tensor([attack_step % 2], device=device)
        probe_opt.zero_grad(set_to_none=True)
        loss_p, _ = attacker_loss(probe, rel, attack_classes(lab), props,
                                  prefix)
        loss_p.backward()
        probe_opt.step()

    class_to_token = torch.tensor(attack_class_tokens, device=device)
    samples = []
    total_correct = total_rows = 0
    sims = []
    with torch.no_grad():
        for block_index, block in enumerate(eval_blocks[:8]):
            input_ids, labels = tensors(block)
            prefix, _, _ = base_states(input_ids)
            latent, _ = tln_f.encode(prefix, secrets.token_hex(16))
            rel, perm = release_f(latent)
            pred_cls = probe(rel)["token"].argmax(-1)
            pred_tokens = class_to_token[pred_cls.clamp_min(0)].reshape(-1)
            true_tokens = labels[:, perm].reshape(-1)
            correct = int((pred_tokens == true_tokens).sum())
            total_correct += correct
            total_rows += true_tokens.numel()
            src_text = tokenizer.decode(true_tokens.cpu().tolist())
            att_text = tokenizer.decode(pred_tokens.cpu().tolist())
            sims.append(levenshtein_ratio(src_text, att_text))
            if block_index < 4:
                src_words = src_text.split()
                att_words = att_text.split()
                pairs = [(s, a) for s, a in zip(src_words, att_words)][:12]
                samples.append({"block": block_index,
                                "source_excerpt": src_text[:120],
                                "attacker_excerpt": att_text[:120],
                                "word_pairs": pairs,
                                "levenshtein": round(sims[-1], 4)})
    postmortem = {
        "attacker_token_accuracy_pct": round(100.0 * total_correct
                                             / max(1, total_rows), 3),
        "mean_levenshtein": round(mean(sims), 4),
        "samples": samples,
        "interpretation": ("token accuracy near the majority control and low "
                           "string similarity mean the attacker's reads are "
                           "wrong constructions (e.g. 'today' -> 'yadto'); "
                           "high values would mean a real reconstruction"),
    }
    emit("postmortem", **{k: v for k, v in postmortem.items()
                          if k != "samples"})
    write_artifact(postmortem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
