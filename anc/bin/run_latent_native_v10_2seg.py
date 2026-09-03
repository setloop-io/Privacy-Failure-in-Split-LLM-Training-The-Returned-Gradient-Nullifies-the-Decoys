#!/usr/bin/env python3
"""v10 two-segment delegation: two independently-protected boundaries.

Architecture (Qwen3-0.6B defaults): prefix layers 0..A -> surrogate segment A
on UCN -> private island -> surrogate segment B on UCN (possibly a
different node) -> tail.  Each boundary has its own private encoder/decoder,
DP clip+noise, fresh v2-stream gauges, and chaff pool; each segment has its
own isolated cloud session (and can live on a different host).  The frozen
attacker and the v7/v9 gate definitions are unchanged; the bundle carries
rows from both boundaries.

Goal: raise UCN's share of the architecture (two 4-layer segments = 8/28
layers on the 0.6B) while measuring whether chained surrogate error
accumulates past the utility gate.
"""

from __future__ import annotations

import argparse
import json
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
from privacy_runtime.ratchet_v2 import (
    derive_gaussian_tensor, derive_permutation, derive_signs,
)
from split_trainer import make_layer_kwargs, run_layer_stack


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def load_blocks(tokenizer, corpus_path: Path, seq_len: int, max_blocks: int):
    text = corpus_path.read_text(errors="replace")
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    width = seq_len + 1
    blocks = [ids[i:i + width] for i in range(0, len(ids) - width + 1, width)]
    if len(blocks) < 8:
        raise RuntimeError("corpus must provide at least eight disjoint blocks")
    return blocks[:max_blocks]


def main() -> int:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--noise-multiplier", type=float, default=0.35)
    ap.add_argument("--clip-norm", type=float, default=1.0)
    ap.add_argument("--split-after-a", type=int, default=15)
    ap.add_argument("--resume-after-a", type=int, default=20)
    ap.add_argument("--split-after-b", type=int, default=21)
    ap.add_argument("--resume-after-b", type=int, default=26)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--train-blocks", type=int, default=256)
    ap.add_argument("--eval-blocks", type=int, default=256)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--attack-steps", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--adversary-strength", type=float, default=1.0)
    ap.add_argument("--cloud-kind", default="monomial_moe_radial")
    ap.add_argument("--cloud-experts", type=int, default=8)
    ap.add_argument("--cloud-layers", type=int, default=2)
    ap.add_argument("--cloud-url-a", required=True)
    ap.add_argument("--cloud-url-b", required=True)
    ap.add_argument("--cloud-tls-ca", required=True)
    ap.add_argument("--chaff-tokens", type=int, default=48)
    ap.add_argument("--remote-grad-clip", type=float, default=1.0)
    ap.add_argument("--probe-restarts", type=int, default=3)
    ap.add_argument("--attacker-updates", type=int, default=3)
    ap.add_argument("--attacker-bundle")
    ap.add_argument("--seed", type=int, default=42)
    # Accepted for shared stage scripts; both mechanisms are unconditional
    # in the v10 two-segment design, and the scale gauge is not part of it.
    ap.add_argument("--secret-wire-rotation", action="store_true",
                    help="accepted for compatibility; always on in v10")
    ap.add_argument("--secret-token-permutation", action="store_true",
                    help="accepted for compatibility; always on in v10")
    ap.add_argument("--token-scale-sigma", type=float, default=0.75,
                    help="accepted for compatibility; unused in v10")
    args = ap.parse_args()
    if not (0 <= args.split_after_a < args.resume_after_a <= args.split_after_b
            < args.resume_after_b):
        raise ValueError("invalid two-segment layout")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    remote_a = RemoteLatentCloud(args.cloud_url_a, args.latent_dim, args.lr,
                                 0.0, args.cloud_kind, args.seed,
                                 tls_ca=args.cloud_tls_ca,
                                 cloud_experts=args.cloud_experts,
                                 cloud_layers=args.cloud_layers)
    remote_b = RemoteLatentCloud(args.cloud_url_b, args.latent_dim, args.lr,
                                 0.0, args.cloud_kind, args.seed + 1000,
                                 tls_ca=args.cloud_tls_ca,
                                 cloud_experts=args.cloud_experts,
                                 cloud_layers=args.cloud_layers)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    core = model.model
    layers = core.layers
    if args.resume_after_b > len(layers):
        raise ValueError("resume_after_b beyond model depth")
    hidden_dim = getattr(model.config, "hidden_size", None)
    if hidden_dim is None:
        hidden_dim = model.config.text_config.hidden_size
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

    cfg = LatentPrivacyConfig(
        hidden_dim=hidden_dim, latent_dim=args.latent_dim,
        cloud_layers=args.cloud_layers, cloud_heads=4,
        clip_norm=args.clip_norm, noise_multiplier=args.noise_multiplier,
        adversary_strength=args.adversary_strength,
        cloud_kind=args.cloud_kind, cloud_experts=args.cloud_experts)
    # Two independent boundary pairs; one embedded attacker set applied to
    # both views (same D).
    tln_a, ucn_a, attackers = build_latent_native_split(
        cfg, vocab_size=len(attack_class_tokens))
    tln_b, _, _ = build_latent_native_split(
        cfg, vocab_size=len(attack_class_tokens))
    assert_ucn_latent_only(ucn_a, args.latent_dim, hidden_dim)
    tln_a.to(device); tln_b.to(device); attackers.to(device)
    remote_tampered_frames = 0
    attacker_opt = torch.optim.AdamW(attackers.parameters(), lr=args.lr)
    defender_opt = torch.optim.AdamW(
        list(tln_a.parameters()) + list(tln_b.parameters()), lr=args.lr)

    def tensors(block):
        ids = torch.tensor(block, dtype=torch.long, device=device)[None]
        return ids[:, :-1], ids[:, 1:]

    def run_layers(first, last, hidden, positions, no_grad=True):
        segment = layers[first:last]
        kwargs = make_layer_kwargs(getattr(core, "rotary_emb", None), hidden,
                                   positions,
                                   type("Args", (), {"attn_impl": "sdpa"})())
        if no_grad:
            with torch.no_grad():
                return run_layer_stack(segment, hidden, kwargs)
        return run_layer_stack(segment, hidden, kwargs)

    # Per-request v2-stream gauges (rotation + permutation; no scale gauge
    # with the radial cloud kind — the v9.2 winner's configuration).
    def release(latent, chaff=None):
        value = latent
        n_real = latent.shape[-2]
        if chaff is not None:
            value = torch.cat(
                [value, chaff[0].to(device=value.device, dtype=value.dtype)],
                dim=-2)
        permutation = derive_permutation(
            secrets.token_bytes(16), 0, value.shape[-2]).to(latent.device)
        inverse = torch.argsort(permutation)
        value = value[:, permutation]
        transform = random_orthogonal(value, args.latent_dim)
        value = value @ transform
        if (chaff is not None
                and value.shape[-2] != n_real + chaff[0].shape[-2]):
            raise RuntimeError("released frame silently lost chaff rows")
        return value, {"rotation": transform, "permutation": permutation,
                       "inverse_permutation": inverse, "n_real": n_real}

    def restore(value, meta):
        value = value @ meta["rotation"].transpose(-1, -2)
        value = value[:, meta["inverse_permutation"]]
        if value.shape[-2] > meta["n_real"]:
            value = value[:, :meta["n_real"]]
        return value

    # Per-boundary chaff pools (each boundary's chaff comes from its own
    # encoder's earlier rows, with labels and hidden states tracked).
    chaff_pools = {"a": None, "b": None}

    def chaff_push(which, latent, labels, hidden):
        if args.chaff_tokens <= 0:
            return
        rows = torch.cat([
            latent.detach().reshape(-1, latent.shape[-1]).float().cpu(),
            labels.detach().reshape(-1, 1).float().cpu(),
            hidden.detach().reshape(-1, hidden.shape[-1]).float().cpu(),
        ], dim=1)
        pool = chaff_pools[which]
        pool = rows if pool is None else torch.cat([pool, rows], dim=0)
        chaff_pools[which] = pool[-8192:]

    def chaff_sample(which):
        pool = chaff_pools[which]
        if args.chaff_tokens <= 0 or pool is None \
                or pool.shape[0] < args.chaff_tokens:
            return None
        index = derive_permutation(secrets.token_bytes(16), 0,
                                   pool.shape[0])[:args.chaff_tokens]
        rows = pool[index].to(device)
        return (rows[None, :, :args.latent_dim],
                rows[None, :, args.latent_dim].long(),
                rows[None, :, args.latent_dim + 1:])

    def clip_remote_gradient(value):
        if not torch.isfinite(value).all():
            raise RuntimeError("UCN returned a non-finite gradient")
        flat = value.float().reshape(-1, value.shape[-1])
        norms = flat.norm(2, dim=-1, keepdim=True)
        scales = (args.remote_grad_clip / norms.clamp_min(1e-12)).clamp(max=1.0)
        return (flat * scales).reshape_as(value).to(value.dtype)

    baseline_losses = []
    if device == "cuda":
        torch.cuda.synchronize()
    baseline_started = time.perf_counter()
    with torch.no_grad():
        for block in eval_blocks:
            input_ids, labels = tensors(block)
            logits = model(input_ids).logits.float()
            baseline_losses.append(float(F.cross_entropy(
                logits.flatten(0, 1), labels.flatten()).item()))
    if device == "cuda":
        torch.cuda.synchronize()
    baseline_eval_seconds = time.perf_counter() - baseline_started

    train_metrics = []
    started = time.perf_counter()
    for step in range(args.steps):
        input_ids, labels = tensors(train_blocks[step % len(train_blocks)])
        positions = torch.arange(input_ids.shape[1], device=device)[None]
        with torch.no_grad():
            hidden = core.embed_tokens(input_ids)
            prefix = run_layers(0, args.split_after_a + 1, hidden,
                                positions).float()
            teacher_a = run_layers(args.split_after_a + 1, args.resume_after_a,
                                   prefix.to(dtype), positions)
            island_true = run_layers(args.resume_after_a,
                                     args.split_after_b + 1, teacher_a,
                                     positions)
            teacher_a = teacher_a.float()
            teacher_b = run_layers(args.split_after_b + 1,
                                   args.resume_after_b, island_true,
                                   positions).float()
        properties = torch.tensor([step % 2], device=device)
        nonce_a = secrets.token_hex(16)
        nonce_b = secrets.token_hex(16)

        # Boundary A release
        latent_a, _ = tln_a.encode(prefix, nonce_a)
        chaff_a = chaff_sample("a")
        released_a, meta_a = release(latent_a, chaff_a)
        chaff_push("a", latent_a, labels, prefix)
        if chaff_a is not None:
            attack_labels_a = torch.cat([labels, chaff_a[1]], dim=1)
            attack_hidden_a = torch.cat([prefix.detach(), chaff_a[2]], dim=1)
        else:
            attack_labels_a = labels
            attack_hidden_a = prefix.detach()
        attack_labels_a = attack_labels_a[:, meta_a["permutation"]]
        attack_hidden_a = attack_hidden_a[:, meta_a["permutation"]]
        attack_labels_cls_a = attack_classes(attack_labels_a)

        # Attacker updates on the A view (attackers trainable, view detached)
        for _ in range(args.attacker_updates):
            attacker_opt.zero_grad(set_to_none=True)
            loss_a, _ = attacker_loss(
                attackers, released_a.detach(), attack_labels_cls_a,
                properties, attack_hidden_a)
            loss_a.backward()
            attacker_opt.step()

        defender_opt.zero_grad(set_to_none=True)
        for parameter in attackers.parameters():
            parameter.requires_grad_(False)

        # Segment A on UCN, then the private island WITH gradients
        cloud_a, mb_a, meta_wire_a = remote_a.forward(
            released_a.detach(), training=True)
        remote_tampered_frames += int(bool(meta_wire_a.get("tampered")))
        hat_a, _ = tln_a.decode(restore(cloud_a, meta_a), nonce_a,
                                  residual=prefix)
        island = run_layers(args.resume_after_a, args.split_after_b + 1,
                            hat_a.to(dtype), positions, no_grad=False)

        # Boundary B release (encoder input carries the island's graph)
        latent_b, _ = tln_b.encode(island.float(), nonce_b)
        chaff_b = chaff_sample("b")
        released_b, meta_b = release(latent_b, chaff_b)
        chaff_push("b", latent_b, labels, island)
        if chaff_b is not None:
            attack_labels_b = torch.cat([labels, chaff_b[1]], dim=1)
            attack_hidden_b = torch.cat([island.detach().float(), chaff_b[2]], dim=1)
        else:
            attack_labels_b = labels
            attack_hidden_b = island.detach()
        attack_labels_b = attack_labels_b[:, meta_b["permutation"]]
        attack_hidden_b = attack_hidden_b[:, meta_b["permutation"]]
        attack_labels_cls_b = attack_classes(attack_labels_b)

        # Attacker updates on the B view: unfreeze, update on the detached
        # view, re-freeze (the defender graph is untouched — views detached).
        for parameter in attackers.parameters():
            parameter.requires_grad_(True)
        for _ in range(args.attacker_updates):
            attacker_opt.zero_grad(set_to_none=True)
            loss_b, _ = attacker_loss(
                attackers, released_b.detach(), attack_labels_cls_b,
                properties, attack_hidden_b)
            loss_b.backward()
            attacker_opt.step()
        for parameter in attackers.parameters():
            parameter.requires_grad_(False)

        cloud_b, mb_b, meta_wire_b = remote_b.forward(
            released_b.detach(), training=True)
        remote_tampered_frames += int(bool(meta_wire_b.get("tampered")))
        hat_b, _ = tln_b.decode(restore(cloud_b, meta_b), nonce_b,
                                  residual=island.float())

        # Tail with true LM loss
        tail_hidden = run_layers(args.resume_after_b, len(layers),
                                 hat_b.to(dtype), positions, no_grad=False)
        logits = model.lm_head(core.norm(tail_hidden.to(dtype))).float()
        language = F.cross_entropy(logits.flatten(0, 1), labels.flatten())

        distill_a = (F.mse_loss(F.layer_norm(hat_a, (hidden_dim,)),
                                F.layer_norm(teacher_a, (hidden_dim,)))
                     + 0.001 * F.mse_loss(hat_a, teacher_a))
        distill_b = (F.mse_loss(F.layer_norm(hat_b, (hidden_dim,)),
                                F.layer_norm(teacher_b, (hidden_dim,)))
                     + 0.001 * F.mse_loss(hat_b, teacher_b))
        if step >= args.warmup_steps:
            privacy = (defender_privacy_loss(
                attackers, released_a, attack_labels_cls_a, properties,
                attack_hidden_a, cfg.adversary_strength)
                + defender_privacy_loss(
                    attackers, released_b, attack_labels_cls_b, properties,
                    attack_hidden_b, cfg.adversary_strength))
        else:
            privacy = latent_a.sum() * 0.0

        total = distill_a + distill_b + language + privacy
        total.backward(retain_graph=True)
        if not torch.isfinite(cloud_a.grad).all() \
                or not torch.isfinite(cloud_b.grad).all():
            raise RuntimeError("non-finite gradient at a cloud output")
        grad_a = clip_remote_gradient(
            remote_a.backward(mb_a, cloud_a.grad))
        grad_b = clip_remote_gradient(
            remote_b.backward(mb_b, cloud_b.grad))
        released_a.backward(grad_a, retain_graph=True)
        released_b.backward(grad_b)
        remote_a.step()
        remote_b.step()
        defender_opt.step()
        for parameter in attackers.parameters():
            parameter.requires_grad_(True)
        train_metrics.append({
            "step": step, "attacker_loss": float((loss_a + loss_b).item()),
            "distill_a": float(distill_a.item()),
            "distill_b": float(distill_b.item()),
            "language_loss": float(language.item())})
        for parameter in attackers.parameters():
            parameter.requires_grad_(True)
        train_metrics.append({
            "step": step, "attacker_loss": float((loss_a + loss_b).item()),
            "distill_a": float(distill_a.item()),
            "distill_b": float(distill_b.item()),
            "language_loss": float(language.item())})

    if device == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started

    # Evaluation: protected path, zero-cloud controls per segment, held-out
    # views for the probe/bundle.
    candidate_losses, zero_a_losses, zero_b_losses = [], [], []
    held_latents, held_labels = [], []
    if device == "cuda":
        torch.cuda.synchronize()
    candidate_started = time.perf_counter()
    with torch.no_grad():
        for block in eval_blocks:
            input_ids, labels = tensors(block)
            positions = torch.arange(input_ids.shape[1], device=device)[None]
            hidden = core.embed_tokens(input_ids)
            prefix = run_layers(0, args.split_after_a + 1, hidden,
                                positions).float()

            def protected_path(zero_a=False, zero_b=False):
                latent_a, _ = tln_a.encode(prefix, secrets.token_hex(16))
                released_a, meta_a = release(latent_a, chaff_sample("a"))
                cloud_a = (remote_a.forward(released_a, training=False)[0]
                           if not zero_a else torch.zeros_like(released_a))
                hat_a, _ = tln_a.decode(restore(cloud_a, meta_a),
                                          secrets.token_hex(16),
                                          residual=prefix)
                island = run_layers(args.resume_after_a,
                                    args.split_after_b + 1, hat_a.to(dtype),
                                    positions)
                latent_b, _ = tln_b.encode(island.float(),
                                             secrets.token_hex(16))
                released_b, meta_b = release(latent_b, chaff_sample("b"))
                cloud_b = (remote_b.forward(released_b, training=False)[0]
                           if not zero_b else torch.zeros_like(released_b))
                hat_b, _ = tln_b.decode(restore(cloud_b, meta_b),
                                          secrets.token_hex(16),
                                          residual=island.float())
                tail_hidden = run_layers(args.resume_after_b, len(layers),
                                         hat_b.to(dtype), positions)
                logits = model.lm_head(
                    core.norm(tail_hidden.to(dtype))).float()
                return float(F.cross_entropy(logits.flatten(0, 1),
                                             labels.flatten()).item())

            candidate_losses.append(protected_path())
            zero_a_losses.append(protected_path(zero_a=True))
            zero_b_losses.append(protected_path(zero_b=True))

            # Held views for the probe/bundle (both boundaries, chaff labels
            # tracked honestly like the v9 runner)
            latent_a, _ = tln_a.encode(prefix, secrets.token_hex(16))
            chaff_a = chaff_sample("a")
            rel_a, meta_a = release(latent_a, chaff_a)
            lab_a = (torch.cat([labels, chaff_a[1]], dim=1)
                     if chaff_a is not None else labels)
            lab_a = lab_a[:, meta_a["permutation"]]
            held_latents.append(rel_a.detach())
            held_labels.append(lab_a.detach())
            island = run_layers(args.resume_after_a, args.split_after_b + 1,
                                prefix.to(dtype), positions)
            latent_b, _ = tln_b.encode(island.float(),
                                         secrets.token_hex(16))
            chaff_b = chaff_sample("b")
            rel_b, meta_b = release(latent_b, chaff_b)
            lab_b = (torch.cat([labels, chaff_b[1]], dim=1)
                     if chaff_b is not None else labels)
            lab_b = lab_b[:, meta_b["permutation"]]
            held_latents.append(rel_b.detach())
            held_labels.append(lab_b.detach())
    if device == "cuda":
        torch.cuda.synchronize()
    candidate_eval_seconds = time.perf_counter() - candidate_started

    # Fresh post-training probes on both boundaries' views.
    probe_recoveries = []
    bundle_train_views, bundle_train_labels = [], []
    total_rows = sum(value.numel() for value in held_labels)
    for probe_restart in range(args.probe_restarts):
        torch.manual_seed(args.seed + 10000 + probe_restart)
        _, _, probe = build_latent_native_split(
            cfg, len(attack_class_tokens))
        probe = probe.to(device)
        probe_opt = torch.optim.AdamW(probe.parameters(), lr=args.lr)
        for attack_step in range(args.attack_steps):
            input_ids, labels = tensors(
                train_blocks[attack_step % len(train_blocks)])
            positions = torch.arange(input_ids.shape[1], device=device)[None]
            with torch.no_grad():
                hidden = core.embed_tokens(input_ids)
                prefix = run_layers(0, args.split_after_a + 1, hidden,
                                    positions).float()
                latent_a, _ = tln_a.encode(prefix, secrets.token_hex(16))
                views = []
                chaff_a = chaff_sample("a")
                rel_a, meta_a = release(latent_a, chaff_a)
                lab_a = (torch.cat([labels, chaff_a[1]], dim=1)
                         if chaff_a is not None else labels)
                lab_a = lab_a[:, meta_a["permutation"]]
                hid_a = (torch.cat([prefix.float(), chaff_a[2]], dim=1)
                         if chaff_a is not None else prefix)
                views.append((rel_a, lab_a,
                              hid_a[:, meta_a["permutation"]]))
                island = run_layers(args.resume_after_a,
                                    args.split_after_b + 1,
                                    prefix.to(dtype), positions)
                latent_b, _ = tln_b.encode(island.float(),
                                             secrets.token_hex(16))
                chaff_b = chaff_sample("b")
                rel_b, meta_b = release(latent_b, chaff_b)
                lab_b = (torch.cat([labels, chaff_b[1]], dim=1)
                         if chaff_b is not None else labels)
                lab_b = lab_b[:, meta_b["permutation"]]
                hid_b = (torch.cat([island.float(), chaff_b[2]], dim=1)
                         if chaff_b is not None else island)
                views.append((rel_b, lab_b,
                              hid_b[:, meta_b["permutation"]]))
                if probe_restart == 0 and attack_step < len(train_blocks):
                    for view in views:
                        bundle_train_views.append(view[0].detach().cpu())
                        bundle_train_labels.append(view[1].detach().cpu())
            props = torch.tensor([attack_step % 2], device=device)
            probe_opt.zero_grad(set_to_none=True)
            probe_loss = latent_a.sum() * 0.0
            for rel, lab, hidden_target in views:
                probe_loss = probe_loss + attacker_loss(
                    probe, rel, attack_classes(lab), props,
                    hidden_target)[0]
            probe_loss.backward()
            probe_opt.step()
        correct = 0
        with torch.no_grad():
            for view, lab in zip(held_latents, held_labels):
                outputs = probe(view)
                eval_attack_labels = attack_classes(lab)
                correct += int(((outputs["token"].argmax(-1)
                                 == eval_attack_labels)
                                & (eval_attack_labels >= 0)).sum().item())
        probe_recoveries.append(100.0 * correct / total_rows)

    train_token_pool = (torch.cat(bundle_train_labels).reshape(-1).tolist()
                        if bundle_train_labels else
                        [t for block in train_blocks for t in block[1:]])
    held_token_pool = [int(token) for value in held_labels
                       for token in value.reshape(-1).cpu().tolist()]
    majority = max(set(train_token_pool), key=train_token_pool.count)
    majority_correct = sum(token == majority for token in held_token_pool)
    recovery_pct = max(probe_recoveries)
    label_free_pct = 100.0 * majority_correct / total_rows

    if args.attacker_bundle:
        bundle_path = Path(args.attacker_bundle)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "dtraining.latent_release_bundle.v1",
            "train_wire": torch.cat(bundle_train_views, dim=0),
            "train_tokens": torch.cat(bundle_train_labels, dim=0),
            "eval_wire": torch.cat([v.cpu() for v in held_latents], dim=0),
            "eval_tokens": torch.cat([v.cpu() for v in held_labels], dim=0),
        }, bundle_path)

    baseline = mean(baseline_losses)
    candidate = mean(candidate_losses)
    zero_a = mean(zero_a_losses)
    zero_b = mean(zero_b_losses)
    result = {
        "schema": "dtraining.latent_native_v10_2seg.v1",
        "status": "completed",
        "model": str(args.model), "device": device, "seed": args.seed,
        "layout": {"prefix": [0, args.split_after_a],
                   "segment_a": [args.split_after_a + 1, args.resume_after_a - 1],
                   "island": [args.resume_after_a, args.split_after_b],
                   "segment_b": [args.split_after_b + 1, args.resume_after_b - 1],
                   "tail": [args.resume_after_b, len(layers) - 1]},
        "delegated_layer_share": (
            (args.resume_after_a - args.split_after_a - 1)
            + (args.resume_after_b - args.split_after_b - 1)) / len(layers),
        "sequence_length": args.seq_len, "steps": args.steps,
        "hidden_dim": hidden_dim, "latent_dim": args.latent_dim,
        "noise_multiplier": args.noise_multiplier,
        "cloud_kind": args.cloud_kind, "cloud_experts": args.cloud_experts,
        "chaff_tokens": args.chaff_tokens,
        "remote_protocol": {"a": remote_a.audit, "b": remote_b.audit},
        "remote_tampered_frames": remote_tampered_frames,
        "transport_tls": True,
        "baseline_eval_loss": baseline, "candidate_eval_loss": candidate,
        "eval_loss_delta": candidate - baseline,
        "zero_a_eval_loss": zero_a, "zero_b_eval_loss": zero_b,
        "cloud_a_improvement": zero_a - candidate,
        "cloud_b_improvement": zero_b - candidate,
        "baseline_eval_seconds": baseline_eval_seconds,
        "candidate_eval_seconds": candidate_eval_seconds,
        "eval_time_ratio": candidate_eval_seconds / baseline_eval_seconds,
        "probe_recovery_pct": probe_recoveries,
        "token_recovery_pct": recovery_pct,
        "label_free_majority_pct": label_free_pct,
        "recovery_above_label_free_pct": recovery_pct - label_free_pct,
        "train_seconds": train_seconds,
        "mean_step_seconds": train_seconds / max(1, args.steps),
        "training_tail": train_metrics[-3:],
        "gates": {
            "utility_delta_le_0_35": candidate - baseline <= 0.35,
            "privacy_above_band_le_1pct": recovery_pct - label_free_pct <= 1.0,
            "eval_time_ratio_le_3": (candidate_eval_seconds
                                     / baseline_eval_seconds <= 3.0),
            "cloud_latent_only": all(
                bool(remote_a.audit["latent_only_audit"])
                and bool(remote_b.audit["latent_only_audit"]) for _ in [1]),
            "cloud_a_improves_loss": zero_a - candidate > 0.01,
            "cloud_b_improves_loss": zero_b - candidate > 0.01,
        },
        "limitations": [
            "two-segment two-node latent protocol exercised",
            "short training run; not convergence or paper evidence",
            "fresh probe is a diagnostic subset, not the complete /attacker suite",
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
