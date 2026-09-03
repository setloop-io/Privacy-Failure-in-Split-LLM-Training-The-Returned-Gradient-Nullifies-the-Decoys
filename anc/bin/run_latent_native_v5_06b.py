#!/usr/bin/env python3
"""Qwen3-0.6B pilot for the latent-native v5 privacy candidate.

Single-process scientific harness: real frozen Qwen prefix/middle/tail with the
middle path replaced by the trainable D-only cloud module, training adaptive
token/property/reconstruction attackers. Emits aggregate JSON only; a pass here
does not replace the later two-node protocol and full /attacker evaluation.
"""
from __future__ import annotations
import argparse
import collections
import json
import math
import secrets
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "split-training"))

from privacy_runtime.latent_native import (
    LatentPrivacyConfig, assert_ucn_latent_only, attacker_loss,
    build_latent_native_split, defender_privacy_loss, random_orthogonal,
)
from privacy_runtime.mine import (build_mine_stats, mine_estimate,
    mine_loss_for_training)
from privacy_runtime.latent_protocol import RemoteLatentCloud
from privacy_runtime.trusted_checkpoint import TrustedBoundaryRecorder
from privacy_runtime.ratchet_v2 import (
    derive_gaussian_tensor, derive_permutation, derive_signs,
)
from split_trainer import (make_layer_kwargs, run_layer_stack,
                           run_sublayer_stack)

# A4: sublayers the TRUSTED node keeps when --delegate-sublayer delegates the other.
TRUSTED_SUBLAYERS = {"mlp": ("attn",), "attn": ("mlp",)}

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
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--noise-multiplier", type=float, default=8.0)
    ap.add_argument("--clip-norm", type=float, default=1.0)
    ap.add_argument("--split-after", type=int, default=15)
    ap.add_argument("--resume-after", type=int, default=26)
    ap.add_argument("--delegate-sublayer", choices=("full", "mlp", "attn"),
                    default="full",
                    help="A4: which residual sublayer of each delegated layer "
                         "goes to the untrusted surrogate. 'full' (default) "
                         "delegates whole layers and is the behaviour every "
                         "committed artifact was produced with. 'mlp' keeps "
                         "every delegated layer's attention sublayer on the "
                         "trusted node and hands off after it; 'attn' is the "
                         "mirror image")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--train-blocks", type=int, default=16)
    ap.add_argument("--eval-blocks", type=int, default=4)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup-steps", type=int, default=4,
                    help="task-only steps before encoder adversarial pressure")
    ap.add_argument("--attack-steps", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--adversary-strength", type=float, default=0.25)
    ap.add_argument("--cloud-kind",
                    choices=("transformer", "equivariant", "monomial",
                             "monomial_moe", "monomial_moe_radial",
                             "invariant_mlp"),
                    default="transformer")
    ap.add_argument("--cloud-experts", type=int, default=1,
                    help="expert count for the monomial_moe cloud")
    ap.add_argument("--cloud-channels", type=int, default=1,
                    help="E4: K parallel independently-gauged cloud channels "
                         "per block (fresh session+seed each); TLN averages "
                         "the restored returns before decode")
    ap.add_argument("--cloud-urls",
                    help="E6: comma-separated cloud URLs; channels are "
                         "distributed round-robin across nodes (defaults to "
                         "--cloud-url for all channels)")
    ap.add_argument("--cloud-layers", type=int, default=2,
                    help="message-passing depth of the cloud module")
    ap.add_argument("--secret-wire-rotation", action="store_true",
                    help="fresh TLN-only orthogonal D rotation per request; requires equivariant cloud")
    ap.add_argument("--secret-token-permutation", action="store_true",
                    help="fresh TLN-only token-row permutation per request; requires equivariant cloud")
    ap.add_argument("--secret-token-gauge", action="store_true",
                    help="fresh signed log-normal token scales; requires monomial cloud")
    ap.add_argument("--token-scale-sigma", type=float, default=0.75)
    ap.add_argument("--chaff-tokens", type=int, default=0,
                    help="recycled real latent rows from earlier train blocks "
                         "appended per released frame to poison Gram/position "
                         "leakage; labels are tracked honestly")
    ap.add_argument("--wire-quant", choices=("none", "bf16", "int8", "int8row"),
                    default="none",
                    help="quantize released forward frames and TLN->UCN "
                         "output gradients (straight-through on the forward)")
    ap.add_argument("--gram-flatten", type=float, default=0.0,
                    help="weight on an encoder-side regularizer that pulls "
                         "each released frame's sorted unit-Gram profile "
                         "toward a running corpus-average profile (attacks "
                         "the gauge-invariant channel directly)")
    ap.add_argument("--remote-grad-clip", type=float, default=1.0,
                    help="TLN-side per-token clipping of UCN-returned input gradients")
    ap.add_argument("--outbound-grad-dp", choices=("clip_noise", "off"),
                    default="clip_noise",
                    help="issue #105: protect the output gradient TLN sends "
                         "to UCN on the same footing as the released "
                         "forward frame (per-row clip to --grad-clip-norm "
                         "plus calibrated Gaussian noise, counted as "
                         "dp.releases.gradient). 'off' restores the "
                         "unprotected, unaccounted backward wire every "
                         "artifact before this fix was produced with")
    ap.add_argument("--grad-clip-norm", type=float, default=0.01,
                    help="per-row L2 bound on the outbound output gradient. "
                         "Set near the median row norm, the DP-SGD "
                         "convention: epsilon does not depend on it, utility "
                         "does. The measured medians on this stack are "
                         "0.0103 (10k steps / 256 blocks) and 0.0043 (40k "
                         "steps / 4096 blocks), so re-calibrate per "
                         "configuration and read back outbound_grad_dp in "
                         "the artifact")
    ap.add_argument("--grad-noise-multiplier", type=float,
                    help="noise multiplier for the outbound output gradient "
                         "(defaults to --noise-multiplier, so the backward "
                         "wire carries the same sigma as the forward frame)")
    ap.add_argument("--dp-account-untransmitted", action="store_true",
                    help="count the post-training probe phase's encode calls "
                         "as DP releases even though they never reach UCN. "
                         "Wrong, and on only to reproduce artifacts written "
                         "before issue #105")
    ap.add_argument("--bundle-canonical-fraction", type=float, default=0.0,
                    help="sensitivity analysis ONLY: also store the pre-gauge "
                         "canonical latent rows for this fraction of blocks, "
                         "simulating an attacker who has compromised the "
                         "gauges for that share of requests (TLN side "
                         "channel blast radius). Bundle must be deleted "
                         "immediately after scoring")
    ap.add_argument("--probe-restarts", type=int, default=3)
    ap.add_argument("--attacker-updates", type=int, default=1,
                    help="adaptive attacker optimizer updates per defender step")
    ap.add_argument("--attacker-bundle",
                    help="temporary trusted .pt bundle for python -m attacker --attack latent-probe")
    ap.add_argument("--grad-channel-bundle",
                    help="threat-model scope audit ONLY: record the raw "
                         "TLN->UCN output gradients (the outbound backward "
                         "wire, run_latent_native_v5_06b.py:660) together with "
                         "the matched forward frame and wire-order labels. "
                         "Inert unless passed; changes no RNG draw and no "
                         "reported key")
    ap.add_argument("--grad-channel-frames", type=int, default=512,
                    help="how many of the FINAL training frames "
                         "--grad-channel-bundle retains (a ring buffer, so the "
                         "recorded window is the most-converged one)")
    ap.add_argument("--trusted-checkpoint", metavar="DIR",
                    help="W3.3: record the trusted boundary module for the "
                         "transcript -- one full state_dict snapshot at "
                         "session start plus one sha256 usage-trace entry per "
                         "training step (the protected-encode/protected-decode "
                         "calls) appended to a single sidecar file, so the "
                         "transcript can prove which trusted module state "
                         "produced each wire frame. Inert unless passed")
    ap.add_argument("--cloud-url",
                    help="remote latent-only UCN server, e.g. ws://ucn:5013")
    ap.add_argument("--cloud-tls-ca",
                    help="pinned PEM certificate/CA for wss:// cloud links; "
                         "trusts only this anchor, hostname verification on")
    ap.add_argument("--active-cloud-delta", type=float, default=0.0,
                    help="declared malicious-return perturbation for the active control")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fragment-channels", type=int, default=1,
                    help="v13/A1 PrivDFS-style fragmentation: split each "
                         "gauged frame's channels across K nodes; no node "
                         "sees a reconstruction-sufficient projection")
    ap.add_argument("--mine-penalty", type=float, default=0.0,
                    help="v13/A9: weight on the MINE DV-bound MI penalty "
                         "(encoder is trained to minimize estimated "
                         "I(latent; token))")
    ap.add_argument("--public-corpus",
                    help="E5: public corpus for cloud pretraining phase "
                         "(no privacy requirement; disjoint from the private "
                         "train/eval split)")
    ap.add_argument("--public-steps", type=int, default=0,
                    help="E5: pretraining steps on the public corpus before "
                         "the private training phase")
    ap.add_argument("--byzantine-verify", action="store_true",
                    help="broadcast each frame (same gauges) to all channels "
                         "and compare restored returns on TLN; replicas are "
                         "seeded identically and receive identical (mean) "
                         "gradients to stay in lockstep")
    ap.add_argument("--byzantine-threshold", type=float, default=0.02,
                    help="relative deviation from the group mean above which "
                         "a frame is flagged tampered (calibrate on an "
                         "honest run first)")
    args = ap.parse_args()
    if args.noise_multiplier <= 0:
        raise ValueError("noise multiplier must be positive")
    grad_dp_on = args.outbound_grad_dp != "off"
    grad_noise_multiplier = (args.grad_noise_multiplier
                             if args.grad_noise_multiplier is not None
                             else args.noise_multiplier)
    if grad_dp_on and (args.grad_clip_norm <= 0 or grad_noise_multiplier <= 0):
        raise ValueError("outbound gradient clip and noise must be positive")
    if ((args.secret_wire_rotation or args.secret_token_permutation)
            and args.cloud_kind not in ("equivariant", "monomial",
                                        "monomial_moe",
                                        "monomial_moe_radial",
                                        "invariant_mlp")):
        raise ValueError("secret wire transforms require an equivariant cloud")
    if (args.secret_token_gauge
            and args.cloud_kind not in ("monomial", "monomial_moe")):
        raise ValueError("secret token gauge requires the monomial cloud; "
                         "the radial MoE cloud reads norms and is "
                         "intentionally incompatible with the scale gauge")
    if args.cloud_kind == "monomial_moe" and args.cloud_experts < 2:
        raise ValueError("monomial_moe requires --cloud-experts >= 2")
    if args.fragment_channels < 1 or (args.fragment_channels > 1
                                      and args.cloud_channels
                                      < args.fragment_channels):
        raise ValueError("--fragment-channels requires at least that many "
                         "--cloud-channels")
    if args.fragment_channels > 1 and args.latent_dim % args.fragment_channels:
        raise ValueError("latent dim must divide across fragments")
    if args.cloud_channels < 1 or (args.cloud_channels > 1
                                   and not args.cloud_url
                                   and not args.cloud_urls):
        raise ValueError("--cloud-channels > 1 requires --cloud-url(s)")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # E6: channels distributed round-robin across --cloud-urls. Fragment width
    # is computed BEFORE any connection so the first channel is correct too.
    channel_urls = [u.strip() for u in
                    (args.cloud_urls.split(",") if args.cloud_urls
                     else [args.cloud_url]) if u and u.strip()]
    channel_latent_dim = (args.latent_dim // args.fragment_channels
                          if args.fragment_channels > 1 else args.latent_dim)
    # Connect before the heavy model/corpus load: connecting after minutes of
    # startup intermittently killed the TLS upgrade (E4 postmortem).
    remote = (RemoteLatentCloud(channel_urls[0], channel_latent_dim,
                                args.lr, args.active_cloud_delta,
                                args.cloud_kind, args.seed,
                                tls_ca=args.cloud_tls_ca,
                                cloud_experts=args.cloud_experts,
                                cloud_layers=args.cloud_layers)
              if channel_urls else None)
    channels = [remote] if remote else []
    for channel_index in range(1, args.cloud_channels):
        channels.append(RemoteLatentCloud(
            channel_urls[channel_index % len(channel_urls)],
            channel_latent_dim, args.lr,
            args.active_cloud_delta, args.cloud_kind,
            # Byzantine replicas must be identical: same seed on every node.
            args.seed if args.byzantine_verify
            else args.seed + 1000 * channel_index,
            tls_ca=args.cloud_tls_ca,
            cloud_experts=args.cloud_experts, cloud_layers=args.cloud_layers))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    core = model.model
    layers = core.layers
    if not (0 <= args.split_after < args.resume_after <= len(layers)):
        raise ValueError("invalid private/cloud layer split")
    hidden_dim = getattr(model.config, "hidden_size", None)
    if hidden_dim is None:
        hidden_dim = model.config.text_config.hidden_size
    hidden_dim = int(hidden_dim)
    blocks = load_blocks(tokenizer, Path(args.corpus), args.seq_len,
                         args.train_blocks + args.eval_blocks)
    train_blocks = blocks[:args.train_blocks]
    eval_blocks = blocks[args.train_blocks:]
    if len(eval_blocks) < args.eval_blocks:
        raise RuntimeError("insufficient disjoint evaluation blocks")

    attack_class_tokens = sorted(set(
        token for block in train_blocks for token in block[1:]))
    attack_lookup = torch.full((int(model.config.vocab_size),), -1,
                               dtype=torch.long, device=device)
    attack_lookup[torch.tensor(attack_class_tokens, device=device)] = torch.arange(
        len(attack_class_tokens), device=device)

    def attack_classes(labels):
        return attack_lookup[labels]

    cfg = LatentPrivacyConfig(
        hidden_dim=hidden_dim, latent_dim=args.latent_dim,
        cloud_layers=args.cloud_layers,
        cloud_heads=4, clip_norm=args.clip_norm,
        noise_multiplier=args.noise_multiplier,
        adversary_strength=args.adversary_strength,
        cloud_kind=args.cloud_kind, cloud_experts=args.cloud_experts,
        gradient_clip_norm=args.grad_clip_norm if grad_dp_on else None,
        gradient_noise_multiplier=(grad_noise_multiplier if grad_dp_on
                                   else None))
    tln, ucn, attackers = build_latent_native_split(
        cfg, vocab_size=len(attack_class_tokens))
    assert_ucn_latent_only(ucn, args.latent_dim, hidden_dim)
    tln.to(device); ucn.to(device); attackers.to(device)
    # W3.3: record the trusted boundary module for the transcript (the cloud
    # side is snapshotted by the server every optimizer step). Inert unless
    # passed; consumes no RNG draws.
    trusted_recorder = None
    if args.trusted_checkpoint:
        trusted_recorder = TrustedBoundaryRecorder(args.trusted_checkpoint)
        trusted_recorder.snapshot(tln, meta={
            "seed": args.seed, "config": asdict(cfg),
            "steps": args.steps, "public_steps": args.public_steps,
            "cloud_channels": args.cloud_channels,
            "fragment_channels": args.fragment_channels,
            "cloud_session_ids": [chan.session_id for chan in channels]})
    remote_tampered_frames = 0
    attacker_opt = torch.optim.AdamW(attackers.parameters(), lr=args.lr)
    mine_stats = (build_mine_stats(args.latent_dim,
                                   int(model.config.vocab_size)).to(device)
                  if args.mine_penalty > 0 else None)
    mine_opt = (torch.optim.AdamW(mine_stats.parameters(), lr=args.lr)
                if mine_stats is not None else None)
    defender_opt = torch.optim.AdamW(
        (list(tln.parameters()) if remote else
         list(tln.parameters()) + list(ucn.parameters())), lr=args.lr)

    def tensors(block):
        ids = torch.tensor(block, dtype=torch.long, device=device)[None]
        return ids[:, :-1], ids[:, 1:]

    # Chaff pool: detached rows recycled from earlier train blocks — real
    # corpus latents with honestly tracked labels; poisons within-frame
    # Gram/order statistics without fabricating decoy content.
    chaff_pool = {"latent": None, "labels": None, "hidden": None}
    # Sensitivity analysis only: selection RNG for canonical-row capture
    # (see --bundle-canonical-fraction).
    canonical_gen = torch.Generator().manual_seed(args.seed + 777)
    canonical_train = {"wire": [], "tokens": []}
    canonical_eval = {"wire": [], "tokens": []}

    def chaff_push(latent, labels, hidden):
        if args.chaff_tokens <= 0:
            return
        rows = latent.detach().reshape(-1, latent.shape[-1]).float().cpu()
        row_labels = labels.detach().reshape(-1).cpu()
        row_hidden = hidden.detach().reshape(-1, hidden.shape[-1]).float().cpu()
        for key, value in (("latent", rows), ("labels", row_labels),
                           ("hidden", row_hidden)):
            pool = chaff_pool[key]
            pool = value if pool is None else torch.cat([pool, value], dim=0)
            chaff_pool[key] = pool[-8192:]

    def chaff_sample():
        if args.chaff_tokens <= 0 or chaff_pool["labels"] is None:
            return None
        pool_size = int(chaff_pool["labels"].shape[0])
        if pool_size < args.chaff_tokens:
            return None
        # Without replacement: identical rows in one frame would be trivially
        # detectable (unit cosine). Drawn from a fresh v2 CSPRNG master; the
        # selection never leaves TLN.
        index = derive_permutation(
            secrets.token_bytes(16), 0, pool_size)[:args.chaff_tokens]
        return (chaff_pool["latent"][index].to(device)[None],
                chaff_pool["labels"][index].to(device)[None],
                chaff_pool["hidden"][index].to(device)[None])

    def quantize_wire(value):
        """Straight-through wire quantization of the released view.

        int8 uses a FIXED grid over [-5, 5] (the token-gauge clamp range
        times clip norm); a per-row absmax grid would be scale-invariant and
        silently strip the token-scale gauge (F_int8 cell postmortem).
        """
        if args.wire_quant == "bf16":
            quantized = value.to(torch.bfloat16).to(value.dtype)
        elif args.wire_quant == "int8":
            scale = 5.0 * args.clip_norm
            quantized = (torch.round(value / scale * 127.0).clamp(-127, 127)
                         / 127.0 * scale)
        elif args.wire_quant == "int8row":
            # Per-row absmax grid: SCALE-INVARIANT, strips the token gauge.
            # Retained only to replicate the F_int8 postmortem finding.
            absmax = value.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
            quantized = torch.round(value / absmax * 127.0) / 127.0 * absmax
        else:
            return value
        return value + (quantized - value).detach()

    # Gram-flattening state: a detached EMA of the corpus-average sorted
    # unit-Gram profile — exactly the gauge-invariant channel the
    # invariant/graph attacker arms read.
    gram_target = {"value": None}

    def gram_profile(frame):
        unit = frame.float() / frame.float().norm(
            dim=-1, keepdim=True).clamp_min(1e-8)
        gram = unit @ unit.transpose(-1, -2)
        seq = frame.shape[-2]
        eye = torch.eye(seq, device=frame.device, dtype=torch.bool)
        offdiag = gram.masked_fill(eye.unsqueeze(0), float("nan"))
        # Sorted per-row similarities, flattened: [B, seq, seq-1] -> [B, -1]
        return offdiag.sort(dim=-1, descending=True).values[..., :seq - 1,
               ].reshape(frame.shape[0], -1)

    def gram_flatten_loss(frame):
        profile = gram_profile(frame)
        detached = profile.detach().mean(dim=0)
        if gram_target["value"] is None:
            gram_target["value"] = detached
            return frame.sum() * 0.0
        gram_target["value"] = 0.99 * gram_target["value"] + 0.01 * detached
        return F.mse_loss(profile, gram_target["value"].expand_as(profile))

    def base_states(input_ids, include_teacher=True):
        positions = torch.arange(input_ids.shape[1], device=device)[None]
        with torch.no_grad():
            hidden = core.embed_tokens(input_ids)
            kwargs = make_layer_kwargs(getattr(core, "rotary_emb", None),
                                       hidden, positions,
                                       type("Args", (), {"attn_impl": "sdpa"})())
            prefix = run_layer_stack(layers[:args.split_after + 1], hidden,
                                     kwargs)
            delegated = layers[args.split_after + 1:args.resume_after]
            # A4: with a sublayer mode the trusted node also runs the sublayer
            # it kept and hands off from there; the teacher stays the true
            # delegated-span output, so a perfect surrogate still recovers the
            # undefended model. 'full' returns `prefix` itself, untouched.
            handoff = (prefix if args.delegate_sublayer == "full"
                       else run_sublayer_stack(
                           delegated, prefix, kwargs,
                           TRUSTED_SUBLAYERS[args.delegate_sublayer]))
            middle = (run_layer_stack(delegated, prefix, kwargs)
                      if include_teacher else None)
        return handoff.float(), (middle.float() if middle is not None else None), positions

    def tail_loss(middle_float, positions, labels):
        hidden = middle_float.to(dtype)
        kwargs = make_layer_kwargs(getattr(core, "rotary_emb", None), hidden,
                                   positions,
                                   type("Args", (), {"attn_impl": "sdpa"})())
        hidden = run_layer_stack(layers[args.resume_after:], hidden, kwargs)
        logits = model.lm_head(core.norm(hidden)).float()
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten())

    def release(latent, chaff=None):
        value = latent
        n_real = latent.shape[-2]
        if chaff is not None:
            value = torch.cat(
                [value, chaff[0].to(device=value.device, dtype=value.dtype)],
                dim=-2)
        permutation = inverse = None
        if args.secret_token_permutation:
            permutation = derive_permutation(
                secrets.token_bytes(16), 0, value.shape[-2]).to(latent.device)
            inverse = torch.argsort(permutation)
            value = value[:, permutation]
        token_gauge = None
        if args.secret_token_gauge:
            gauge_master = secrets.token_bytes(16)
            log_scale = derive_gaussian_tensor(
                gauge_master, 0,
                (value.shape[0], value.shape[-2], 1)).to(
                    value.device) * args.token_scale_sigma
            scale = log_scale.exp().clamp(0.2, 5.0)
            signs = derive_signs(
                gauge_master, 1, scale.shape).to(value.device)
            token_gauge = (scale * signs).to(value.dtype)
            value = value * token_gauge
        transform = (random_orthogonal(value, args.latent_dim)
                     if args.secret_wire_rotation else None)
        if transform is not None:
            value = value @ transform
        value = quantize_wire(value)
        if (chaff is not None
                and value.shape[-2] != n_real + chaff[0].shape[-2]):
            raise RuntimeError("released frame silently lost chaff rows")
        return value, {"rotation": transform, "permutation": permutation,
                       "token_gauge": token_gauge,
                       "inverse_permutation": inverse,
                       "n_real": n_real}

    def restore(value, release_meta):
        transform = release_meta["rotation"]
        if transform is not None:
            value = value @ transform.transpose(-1, -2)
        token_gauge = release_meta["token_gauge"]
        if token_gauge is not None:
            value = value / token_gauge
        inverse = release_meta["inverse_permutation"]
        if inverse is not None:
            value = value[:, inverse]
        n_real = release_meta.get("n_real")
        if n_real is not None and value.shape[-2] > n_real:
            value = value[:, :n_real]
        return value

    def attack_targets(labels, hidden, release_meta):
        permutation = release_meta["permutation"]
        if permutation is None:
            return labels, hidden
        return labels[:, permutation], hidden[:, permutation]

    def clip_remote_gradient(value):
        if not torch.isfinite(value).all():
            raise RuntimeError("UCN returned a non-finite gradient")
        flat = value.float().reshape(-1, value.shape[-1])
        norms = flat.norm(2, dim=-1, keepdim=True)
        scales = (args.remote_grad_clip / norms.clamp_min(1e-12)).clamp(max=1.0)
        return (flat * scales).reshape_as(value).to(value.dtype)

    # Issue #105: what the outbound backward wire cost, read back from the
    # protection itself so a run reports whether its clip was calibrated.
    grad_dp = {"frames": 0, "rows": 0, "clip_scale_sum": 0.0,
               "max_preclip_norm": 0.0}

    def outbound_gradient(gradient, nonce: str):
        """The exact tensor that leaves TLN on the backward path.

        Protection first, wire format last, mirroring the forward path
        (encode -> gauge -> quantize). Every wire row is clipped and noised,
        chaff rows included, so no zero support discloses the real/decoy
        partition. With the leg off this is the pre-fix expression
        (quantize_wire is identity when --wire-quant is none).
        """
        if not grad_dp_on:
            return quantize_wire(gradient)
        protected, meta = tln.protect_gradient(gradient, nonce)
        grad_dp["frames"] += 1
        grad_dp["rows"] += int(meta["token_releases"])
        grad_dp["clip_scale_sum"] += float(meta["mean_clip_scale"])
        grad_dp["max_preclip_norm"] = max(grad_dp["max_preclip_norm"],
                                          float(meta["max_preclip_norm"]))
        return quantize_wire(protected)

    def wire_real_mask(release_meta, n_wire):
        """Wire-order mask, True where the row is a real corpus token.

        Audit helper: reads only trusted-side release metadata (exactly as
        attack_targets does for labels) to measure whether the outbound
        gradient's zero pattern discloses the chaff partition.
        """
        mask = torch.zeros(n_wire, dtype=torch.bool, device=device)
        mask[:int(release_meta["n_real"])] = True
        permutation = release_meta["permutation"]
        return (mask if permutation is None else mask[permutation]).cpu()

    def record_grad_channel(wire, grad, labels, release_meta, step):
        """Retain one frame of the real outbound backward wire (audit only)."""
        grad_channel.append({
            "wire": wire.detach().float().cpu(),
            "grad": grad.detach().float().cpu(),
            "tokens": labels.detach().cpu(),
            "is_real": wire_real_mask(release_meta, wire.shape[-2]),
            "step": int(step),
        })

    # Byzantine verification state (verify mode): one gauged frame broadcast
    # to identically-seeded replicas; TLN compares restored returns, uses
    # the coordinate-wise median, and counts deviations.
    byz_verified = 0
    byz_flagged = 0
    byz_max_dev = 0.0

    def byzantine_combine(stacked):
        """stacked [K,B,T,D] restored returns -> (median, worst_rel_dev)."""
        nonlocal byz_verified, byz_flagged, byz_max_dev
        group_mean = stacked.mean(dim=0, keepdim=True)
        deviation = ((stacked - group_mean).flatten(1).norm(2, dim=1)
                     / group_mean.flatten(1).norm(2).clamp_min(1e-8))
        worst = float(deviation.max())
        byz_max_dev = max(byz_max_dev, worst)
        byz_verified += 1
        if worst > args.byzantine_threshold:
            byz_flagged += 1
        return stacked.median(dim=0).values

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
    # Threat-model scope audit: a ring buffer over the outbound backward wire.
    # Empty and never written unless --grad-channel-bundle is passed.
    grad_channel = (collections.deque(maxlen=max(1, args.grad_channel_frames))
                    if args.grad_channel_bundle else None)
    started = time.perf_counter()
    # E5: optional public-data phase first (same protocol, gauges, chaff —
    # uniform exposure; the phase simply does not need privacy). The private
    # phase then fine-tunes from the warm-started cloud.
    phase_plan = []
    if args.public_steps > 0:
        if not args.public_corpus:
            raise ValueError("--public-steps requires --public-corpus")
        public_blocks = load_blocks(tokenizer, Path(args.public_corpus),
                                    args.seq_len, args.public_steps)
        phase_plan += [(public_blocks, s) for s in range(args.public_steps)]
    phase_plan += [(train_blocks, s) for s in range(args.steps)]

    # W1.3 acceptance: functional fingerprint of each fragment channel, taken
    # before and after training. The cloud modules are deterministic, so a
    # non-zero output delta on an identical probe means the parameters moved
    # (was exactly zero before the fragmentation backward fix).
    fragment_probe = None
    fragment_probe_baseline = []
    if remote and args.fragment_channels > 1:
        probe_gen = torch.Generator(device="cpu")
        probe_gen.manual_seed(args.seed + 777001)
        fragment_probe = torch.randn(
            1, 8, args.latent_dim // args.fragment_channels,
            generator=probe_gen).to(device=device, dtype=dtype)
        for chan in channels[:args.fragment_channels]:
            probe_out, _, _ = chan.forward(fragment_probe, training=False)
            fragment_probe_baseline.append(probe_out.detach().float().cpu())

    for global_step, (phase_blocks, step) in enumerate(phase_plan):
        input_ids, labels = tensors(
            phase_blocks[step % len(phase_blocks)])
        prefix, middle_target, positions = base_states(input_ids)
        properties = torch.tensor([step % 2], device=device)
        nonce = secrets.token_hex(16)

        latent, _ = tln.encode(prefix, nonce)
        chaff = chaff_sample()
        released, release_meta = release(latent, chaff)
        chaff_push(latent, labels, prefix)
        if chaff is not None:
            attack_labels_input = torch.cat([labels, chaff[1]], dim=1)
            attack_hidden_input = torch.cat(
                [prefix.detach(), chaff[2].to(prefix.dtype)], dim=1)
        else:
            attack_labels_input = labels
            attack_hidden_input = prefix.detach()
        attack_token_labels, attack_hidden = attack_targets(
            attack_labels_input, attack_hidden_input, release_meta)
        attack_labels = attack_classes(attack_token_labels)
        for _ in range(args.attacker_updates):
            attacker_opt.zero_grad(set_to_none=True)
            loss_a, _ = attacker_loss(
                attackers, released.detach(), attack_labels,
                properties, attack_hidden)
            loss_a.backward(); attacker_opt.step()
            if mine_stats is not None:
                mine_opt.zero_grad(set_to_none=True)
                mine_loss_for_training(
                    mine_stats, released.detach(),
                    attack_token_labels.clamp_min(0)).backward()
                mine_opt.step()

        defender_opt.zero_grad(set_to_none=True)
        for parameter in attackers.parameters():
            parameter.requires_grad_(False)
        if mine_stats is not None:
            for parameter in mine_stats.parameters():
                parameter.requires_grad_(False)
        if remote and args.fragment_channels > 1:
            # A1: fragment the gauged frame across nodes; no node sees the
            # full D-width row. Fragments are post-rotation slices, so each
            # carries a mixed channel subset.
            frag_width = args.latent_dim // args.fragment_channels
            cloud_paths = []
            for frag_index, chan in enumerate(
                    channels[:args.fragment_channels]):
                frag = released[..., frag_index * frag_width:
                                (frag_index + 1) * frag_width]
                cloud_k, mb_k, meta_wire = chan.forward(
                    frag.detach(), training=True)
                remote_tampered_frames += int(bool(meta_wire.get("tampered")))
                cloud_paths.append((cloud_k, mb_k, chan, frag))
            trusted_return = restore(
                torch.cat([path[0] for path in cloud_paths], dim=-1),
                release_meta)
        elif remote and args.byzantine_verify:
            # One release, one gauge set: broadcast the SAME frame to all
            # identically-seeded replicas and compare restored returns.
            cloud_paths = []
            for chan in channels:
                cloud_k, mb_k, meta_wire = chan.forward(
                    released.detach(), training=True)
                remote_tampered_frames += int(bool(meta_wire.get("tampered")))
                cloud_paths.append((cloud_k, mb_k, chan))
            stacked = torch.stack(
                [restore(path[0], release_meta) for path in cloud_paths])
            trusted_return = byzantine_combine(stacked)
        elif remote:
            # E4: every channel gets its own fresh gauges and chaff; TLN
            # averages the restored returns (ensemble correction).
            channel_states = [(released, release_meta, channels[0])]
            for chan in channels[1:]:
                rel_k, meta_k = release(latent, chaff_sample())
                channel_states.append((rel_k, meta_k, chan))
            cloud_paths = []
            for rel_k, meta_k, chan in channel_states:
                cloud_k, mb_k, meta_wire = chan.forward(
                    rel_k.detach(), training=True)
                remote_tampered_frames += int(bool(meta_wire.get("tampered")))
                cloud_paths.append((cloud_k, mb_k, chan, rel_k, meta_k))
            trusted_return = torch.stack(
                [restore(path[0], path[4])
                 for path in cloud_paths]).mean(dim=0)
        else:
            cloud_latent = ucn(released)
            trusted_return = restore(cloud_latent, release_meta)
        restored, _ = tln.decode(trusted_return, nonce, residual=prefix)
        if trusted_recorder is not None:
            # Before defender_opt.step(): the recorded state is the one that
            # produced this step's frames (encode output and decode I/O).
            trusted_recorder.record_step(
                global_step, tln, encode_output=latent,
                decode_input=trusted_return, decode_output=restored)
        restored_norm = F.layer_norm(restored, (restored.shape[-1],))
        target_norm = F.layer_norm(middle_target,
                                   (middle_target.shape[-1],))
        distill = (F.mse_loss(restored_norm, target_norm)
                   + 0.001 * F.mse_loss(restored, middle_target))
        language = tail_loss(restored, positions, labels)
        if global_step >= args.warmup_steps:
            privacy = defender_privacy_loss(
                attackers, released, attack_labels, properties, attack_hidden,
                cfg.adversary_strength)
            if mine_stats is not None:
                privacy = privacy - args.mine_penalty * mine_loss_for_training(
                    mine_stats, released, attack_token_labels.clamp_min(0))
        else:
            privacy = latent.sum() * 0.0
        if args.gram_flatten > 0 and global_step >= args.warmup_steps:
            gram_flat = args.gram_flatten * gram_flatten_loss(released)
        else:
            gram_flat = latent.sum() * 0.0
        total = distill + language + privacy + gram_flat
        total.backward(retain_graph=bool(remote))
        if remote and args.fragment_channels > 1:
            # A1 backward: return each fragment's output gradient to its own
            # channel, pull it back through that fragment's slice of the
            # released frame, and step the remote optimizer. The fragments
            # share the encoder subgraph, so it is retained until the last one.
            for path_index, (cloud_k, mb_k, chan, frag) in enumerate(cloud_paths):
                if not torch.isfinite(cloud_k.grad).all():
                    raise RuntimeError("non-finite gradient at cloud output")
                send_grad = outbound_gradient(cloud_k.grad, nonce)
                remote_input_grad = chan.backward(mb_k, send_grad)
                frag.backward(
                    clip_remote_gradient(remote_input_grad),
                    retain_graph=path_index < len(cloud_paths) - 1)
                chan.step()
        elif remote and args.byzantine_verify:
            # Lockstep replicas: every node receives the same (mean) output
            # gradient; identical inputs + identical gradients keep the
            # replicas identical, so any later deviation is a tamper signal.
            mean_grad = outbound_gradient(
                torch.stack([path[0].grad for path in cloud_paths]).mean(
                    dim=0), nonce)
            input_grads = []
            for cloud_k, mb_k, chan in cloud_paths:
                input_grads.append(clip_remote_gradient(
                    chan.backward(mb_k, mean_grad)))
                chan.step()
            released.backward(
                torch.stack(input_grads).mean(dim=0))
        elif remote:
            for path_index, (cloud_k, mb_k, chan, rel_k,
                             meta_k) in enumerate(cloud_paths):
                if not torch.isfinite(cloud_k.grad).all():
                    raise RuntimeError("non-finite gradient at cloud output")
                send_grad = outbound_gradient(cloud_k.grad, nonce)
                if grad_channel is not None and path_index == 0:
                    record_grad_channel(rel_k, send_grad, attack_token_labels,
                                        meta_k, global_step)
                remote_input_grad = chan.backward(mb_k, send_grad)
                # All channels share the encoder subgraph: retain it until
                # the last channel's backward has run.
                rel_k.backward(
                    clip_remote_gradient(remote_input_grad),
                    retain_graph=path_index < len(cloud_paths) - 1)
                chan.step()
        defender_opt.step()
        for parameter in attackers.parameters():
            parameter.requires_grad_(True)
        if mine_stats is not None:
            for parameter in mine_stats.parameters():
                parameter.requires_grad_(True)
        train_metrics.append({"step": global_step, "phase_step": step,
                              "attacker_loss": float(loss_a.item()),
                              "distill_loss": float(distill.item()),
                              "language_loss": float(language.item()),
                              "gram_flatten_loss": float(gram_flat.item())})

    if device == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started

    fragment_verification = None
    if fragment_probe is not None:
        deltas = []
        for chan, baseline in zip(channels[:args.fragment_channels],
                                  fragment_probe_baseline):
            probe_out, _, _ = chan.forward(fragment_probe, training=False)
            after = probe_out.detach().float().cpu()
            deltas.append(float((after - baseline).norm()))
        fragment_verification = {
            "fragment_channels": args.fragment_channels,
            "steps": args.steps,
            "probe": "fixed probe, identical before and after training",
            "probe_l2_delta_per_channel": deltas,
            "cloud_trained": all(delta > 0 for delta in deltas),
        }

    if grad_channel is not None:
        if not grad_channel:
            raise RuntimeError("--grad-channel-bundle needs a remote cloud "
                               "training path; nothing was recorded")
        grad_path = Path(args.grad_channel_bundle)
        grad_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "dtraining.outbound_grad_channel.v1",
            "wire": torch.cat([f["wire"] for f in grad_channel], dim=0),
            "grad": torch.cat([f["grad"] for f in grad_channel], dim=0),
            "tokens": torch.cat([f["tokens"] for f in grad_channel], dim=0),
            "is_real": torch.stack([f["is_real"] for f in grad_channel]),
            "frame_steps": torch.tensor([f["step"] for f in grad_channel]),
            "train_blocks": len(train_blocks), "seq_len": args.seq_len,
            "chaff_tokens": args.chaff_tokens, "wire_quant": args.wire_quant,
            "remote_grad_clip": args.remote_grad_clip,
            "noise_multiplier": args.noise_multiplier,
            "outbound_grad_dp": args.outbound_grad_dp,
            "grad_clip_norm": args.grad_clip_norm if grad_dp_on else None,
            "grad_noise_multiplier": (grad_noise_multiplier if grad_dp_on
                                      else None),
            "split_after": args.split_after, "steps": args.steps,
        }, grad_path)

    # Evaluate protected utility and collect detached, held-out latent rows.
    candidate_losses, zero_cloud_losses = [], []
    held_latents, held_labels, held_prefix = [], [], []
    if device == "cuda":
        torch.cuda.synchronize()
    candidate_started = time.perf_counter()
    prepared_eval = []
    for index, block in enumerate(eval_blocks):
        input_ids, labels = tensors(block)
        prefix, _, positions = base_states(input_ids, include_teacher=False)
        with torch.no_grad():
            zero_cloud_losses.append(float(tail_loss(prefix, positions,
                                                     labels).item()))
        latent, _ = tln.encode(prefix, secrets.token_hex(16))
        if (args.bundle_canonical_fraction > 0
                and float(torch.rand((), generator=canonical_gen))
                < args.bundle_canonical_fraction):
            canonical_eval["wire"].append(latent.detach().float().cpu())
            canonical_eval["tokens"].append(labels.detach().cpu())
        channel_entries = []
        for chan_index, chan in enumerate(channels or [None]):
            if args.byzantine_verify and chan_index > 0:
                # Same gauged frame to every replica; exposure is unchanged
                # (the K copies carry identical, already-gauged content).
                channel_entries.append(channel_entries[0])
                continue
            chaff_k = chaff_sample()
            released_k, meta_k = release(latent, chaff_k)
            if chaff_k is not None:
                eval_labels_k = torch.cat([labels, chaff_k[1]], dim=1)
                eval_hidden_k = torch.cat(
                    [prefix.detach(), chaff_k[2].to(prefix.dtype)], dim=1)
            else:
                eval_labels_k = labels
                eval_hidden_k = prefix.detach()
            labels_k, prefix_k = attack_targets(
                eval_labels_k, eval_hidden_k, meta_k)
            channel_entries.append((released_k, meta_k, labels_k, prefix_k))
        prepared_eval.append((channel_entries, labels, prefix, positions))
    with torch.no_grad():
        if remote and args.fragment_channels > 1:
            frag_width = args.latent_dim // args.fragment_channels
            per_channel_results = [
                chan.forward_many([block[0][0][0][..., k * frag_width:
                                                  (k + 1) * frag_width]
                                   for block in prepared_eval])
                for k, chan in enumerate(
                        channels[:args.fragment_channels])]
        elif remote:
            # Per channel: coalesced eval over all blocks; TLN averages the
            # restored returns across channels before decode.
            per_channel_results = [
                chan.forward_many([block[0][k][0] for block in prepared_eval])
                for k, chan in enumerate(channels)]
        else:
            per_channel_results = [[(ucn(block[0][k][0]), None,
                                     {"tampered": False})
                                    for block in prepared_eval]
                                   for k in range(len(prepared_eval[0][0]))]
        for block_index, (channel_entries, labels, prefix,
                          positions) in enumerate(prepared_eval):
            if args.fragment_channels > 1:
                # Fragments are half-width raw returns: concatenate first,
                # then un-gauge once (per-fragment restore would be invalid).
                frag_returns = []
                for frag_index in range(args.fragment_channels):
                    cloud_eval, _mb, eval_meta = per_channel_results[
                        frag_index][block_index]
                    remote_tampered_frames += int(
                        bool(eval_meta.get("tampered")))
                    frag_returns.append(cloud_eval)
                combined = restore(torch.cat(frag_returns, dim=-1),
                                   channel_entries[0][1])
                released_k, meta_k, labels_k, prefix_k = channel_entries[0]
                held_latents.append(released_k.detach())
                held_labels.append(labels_k.detach())
                held_prefix.append(prefix_k.detach())
            else:
                returns = []
                for chan_index, (released_k, meta_k, labels_k,
                                 prefix_k) in enumerate(channel_entries):
                    cloud_eval, _mb, eval_meta = per_channel_results[
                        chan_index][block_index]
                    remote_tampered_frames += int(
                        bool(eval_meta.get("tampered")))
                    returns.append(restore(cloud_eval, meta_k))
                    if chan_index == 0 or not args.byzantine_verify:
                        held_latents.append(released_k.detach())
                        held_labels.append(labels_k.detach())
                        held_prefix.append(prefix_k.detach())
                stacked_returns = torch.stack(returns)
                combined = (byzantine_combine(stacked_returns)
                            if args.byzantine_verify
                            else stacked_returns.mean(dim=0))
            predicted, _ = tln.decode(combined, secrets.token_hex(16),
                                        residual=prefix)
            candidate_losses.append(float(tail_loss(predicted, positions,
                                                    labels).item()))
    if device == "cuda":
        torch.cuda.synchronize()
    candidate_eval_seconds = time.perf_counter() - candidate_started

    # Fresh post-training token attacker: train on train blocks, evaluate only
    # on the disjoint held-out blocks. No canonical eval labels enter training.
    probe_recoveries = []
    untransmitted = {"calls": 0, "rows": 0}
    bundle_train_views, bundle_train_labels = [], []
    total_rows = sum(labels.numel() for labels in held_labels)
    for probe_restart in range(args.probe_restarts):
        torch.manual_seed(args.seed + 10000 + probe_restart)
        _, _, probe = build_latent_native_split(
            cfg, len(attack_class_tokens))
        probe = probe.to(device)
        probe_opt = torch.optim.AdamW(probe.parameters(), lr=args.lr)
        for attack_step in range(args.attack_steps):
            input_ids, labels = tensors(
                train_blocks[attack_step % len(train_blocks)])
            prefix, _, _ = base_states(input_ids, include_teacher=False)
            # E4: the probe trains on every channel's released view (that is
            # the real exposure), summing per-channel losses into one step.
            probe_views = []
            n_probe_channels = (len(channels)
                                if remote and args.fragment_channels == 1
                                else 1)
            with torch.no_grad():
                # Issue #105: this phase re-derives a released view from the
                # frozen encoder and never calls chan.forward, so nothing
                # crosses the boundary and nothing may be charged for it.
                latent, probe_meta = tln.encode(
                    prefix, secrets.token_hex(16),
                    transmitted=args.dp_account_untransmitted)
                untransmitted["calls"] += int(not args.dp_account_untransmitted)
                untransmitted["rows"] += int(
                    probe_meta["token_releases"]
                    * (not args.dp_account_untransmitted))
                if (args.bundle_canonical_fraction > 0
                        and float(torch.rand((), generator=canonical_gen))
                        < args.bundle_canonical_fraction):
                    canonical_train["wire"].append(
                        latent.detach().float().cpu())
                    canonical_train["tokens"].append(labels.detach().cpu())
                for _chan in range(n_probe_channels):
                    chaff_k = chaff_sample()
                    probe_release_k, probe_meta_k = release(latent, chaff_k)
                    if chaff_k is not None:
                        probe_labels_k = torch.cat([labels, chaff_k[1]], dim=1)
                        probe_hidden_k = torch.cat(
                            [prefix, chaff_k[2].to(prefix.dtype)], dim=1)
                    else:
                        probe_labels_k = labels
                        probe_hidden_k = prefix
                    probe_labels_k, probe_hidden_k = attack_targets(
                        probe_labels_k, probe_hidden_k, probe_meta_k)
                    probe_views.append(
                        (probe_release_k, probe_labels_k, probe_hidden_k))
                chaff_push(latent, labels, prefix)
                if probe_restart == 0 and attack_step < len(train_blocks):
                    for view in probe_views:
                        bundle_train_views.append(view[0].detach().cpu())
                        bundle_train_labels.append(view[1].detach().cpu())
            props = torch.tensor([attack_step % 2], device=device)
            probe_opt.zero_grad(set_to_none=True)
            probe_loss = latent.sum() * 0.0
            for probe_release_k, probe_labels_k, probe_hidden_k in probe_views:
                probe_attack_labels = attack_classes(probe_labels_k)
                loss_k, _ = attacker_loss(
                    probe, probe_release_k, probe_attack_labels, props,
                    probe_hidden_k)
                probe_loss = probe_loss + loss_k
            # The adaptive known-plaintext probe predicts only token classes
            # present in its train partition, matching /attacker latent-probe.
            probe_loss.backward(); probe_opt.step()
        correct = 0
        with torch.no_grad():
            for latent_view, labels, _prefix in zip(
                    held_latents, held_labels, held_prefix):
                outputs = probe(latent_view)
                eval_attack_labels = attack_classes(labels)
                correct += int(((outputs["token"].argmax(-1)
                                 == eval_attack_labels)
                                & (eval_attack_labels >= 0)).sum().item())
        probe_recoveries.append(100.0 * correct / total_rows)
    # Majority control must match the frozen attacker's definition exactly:
    # mode of the released train labels scored over ALL released eval rows.
    # With chaff, chaff labels are included on both sides — otherwise chaff
    # rows dilute only the denominator and falsely trip
    # privacy_above_band_le_1pct (12B seed-42 postmortem).
    if args.chaff_tokens > 0 and bundle_train_labels:
        train_token_pool = torch.cat(bundle_train_labels).reshape(-1).tolist()
        held_token_pool = [int(token) for value in held_labels
                           for token in value.reshape(-1).cpu().tolist()]
    else:
        train_token_pool = [token for block in train_blocks
                            for token in block[1:]]
        held_token_pool = [token for block in eval_blocks
                           for token in block[1:]]
    majority = max(set(train_token_pool), key=train_token_pool.count)
    majority_correct = sum(token == majority for token in held_token_pool)
    mine_mi_nats = None
    if mine_stats is not None and held_latents:
        with torch.no_grad():
            estimates = [mine_estimate(mine_stats, view, lab.clamp_min(0))
                         for view, lab in zip(held_latents, held_labels)]
            mine_mi_nats = mean(estimates)
    recovery_pct = max(probe_recoveries)
    label_free_pct = 100.0 * majority_correct / total_rows
    if args.attacker_bundle:
        bundle_path = Path(args.attacker_bundle)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "schema": "dtraining.latent_release_bundle.v1",
            "train_wire": torch.cat(bundle_train_views, dim=0),
            "train_tokens": torch.cat(bundle_train_labels, dim=0),
            "eval_wire": torch.cat([value.cpu() for value in held_latents], dim=0),
            "eval_tokens": torch.cat([value.cpu() for value in held_labels], dim=0),
        }
        if canonical_eval["wire"]:
            bundle["canonical_train_wire"] = torch.cat(
                canonical_train["wire"], dim=0)
            bundle["canonical_train_tokens"] = torch.cat(
                canonical_train["tokens"], dim=0)
            bundle["canonical_eval_wire"] = torch.cat(
                canonical_eval["wire"], dim=0)
            bundle["canonical_eval_tokens"] = torch.cat(
                canonical_eval["tokens"], dim=0)
        torch.save(bundle, bundle_path)
    baseline = mean(baseline_losses)
    candidate = mean(candidate_losses)
    zero_cloud = mean(zero_cloud_losses)
    result = {
        "schema": "dtraining.latent_native_v5_06b.v1",
        "status": "completed",
        "model": str(args.model), "device": device, "seed": args.seed,
        "tln_peak_cuda_memory_mb": (
            round(torch.cuda.max_memory_allocated() / 1e6, 1)
            if device == "cuda" else None),
        "split_after": args.split_after, "resume_after": args.resume_after,
        "sequence_length": args.seq_len, "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "train_blocks": len(train_blocks), "eval_blocks": len(eval_blocks),
        "hidden_dim": hidden_dim, "latent_dim": args.latent_dim,
        "noise_multiplier": args.noise_multiplier,
        "adversary_strength": args.adversary_strength,
        "cloud_kind": args.cloud_kind,
        "cloud_experts": args.cloud_experts,
        "cloud_layers": args.cloud_layers,
        "cloud_channels": args.cloud_channels,
        "fragment_channels": args.fragment_channels,
        "public_steps": args.public_steps,
        "byzantine_verify": bool(args.byzantine_verify),
        "byzantine_threshold": (args.byzantine_threshold
                                if args.byzantine_verify else None),
        "byzantine_verified_frames": byz_verified,
        "byzantine_flagged_frames": byz_flagged,
        "byzantine_max_relative_deviation": byz_max_dev,
        "secret_wire_rotation": args.secret_wire_rotation,
        "secret_token_permutation": args.secret_token_permutation,
        "secret_token_gauge": args.secret_token_gauge,
        "token_scale_sigma": args.token_scale_sigma,
        "chaff_tokens": args.chaff_tokens,
        "mine_penalty": args.mine_penalty,
        "mine_mi_nats_held_out": mine_mi_nats,
        "bundle_canonical_fraction": args.bundle_canonical_fraction,
        "wire_quant": args.wire_quant,
        "gram_flatten": args.gram_flatten,
        "remote_grad_clip": args.remote_grad_clip,
        "probe_restarts": args.probe_restarts,
        "attacker_updates": args.attacker_updates,
        "probe_recovery_pct": probe_recoveries,
        "baseline_eval_loss": baseline, "candidate_eval_loss": candidate,
        "zero_cloud_eval_loss": zero_cloud,
        "cloud_correction_loss_improvement": zero_cloud - candidate,
        "eval_loss_delta": candidate - baseline,
        "token_recovery_pct": recovery_pct,
        "label_free_majority_pct": label_free_pct,
        "recovery_above_label_free_pct": recovery_pct - label_free_pct,
        "train_seconds": train_seconds,
        "mean_step_seconds": train_seconds / args.steps,
        "baseline_eval_seconds": baseline_eval_seconds,
        "candidate_eval_seconds": candidate_eval_seconds,
        "eval_time_ratio": candidate_eval_seconds / baseline_eval_seconds,
        "dp": tln.dp.report(),
        "cloud_state_contains_hidden_width": (False if remote else any(
            hidden_dim in value.shape for value in ucn.state_dict().values())),
        "remote_protocol": (remote.audit if remote else None),
        "transport_tls": bool(args.cloud_url and args.cloud_url.startswith("wss://")),
        "remote_tampered_frames": remote_tampered_frames,
        "active_cloud_delta": args.active_cloud_delta,
        "gates": {
            "utility_delta_le_0_35": candidate - baseline <= 0.35,
            "privacy_above_band_le_1pct": recovery_pct - label_free_pct <= 1.0,
            "eval_time_ratio_le_3": candidate_eval_seconds / baseline_eval_seconds <= 3.0,
            "cloud_latent_only": (bool(remote.audit["latent_only_audit"])
                                  if remote else not any(
                hidden_dim in value.shape for value in ucn.state_dict().values())),
            "cloud_correction_improves_loss": zero_cloud - candidate > 0.01,
            "secret_coordinates_hidden_from_ucn": (
                args.secret_wire_rotation
                and args.cloud_kind in ("equivariant", "monomial",
                                        "monomial_moe",
                                        "monomial_moe_radial",
                                        "invariant_mlp")),
            "token_order_hidden_from_ucn": (
                args.secret_token_permutation
                and args.cloud_kind in ("equivariant", "monomial",
                                        "monomial_moe",
                                        "monomial_moe_radial",
                                        "invariant_mlp")),
            "token_norm_and_sign_gauged": (
                args.secret_token_gauge
                and args.cloud_kind in ("monomial", "monomial_moe")),
        },
        "training_tail": train_metrics[-3:],
        "limitations": [
            ("two-node latent protocol exercised"
             if remote else "single-process pilot; two-node protocol not exercised"),
            "short training run; not convergence or paper evidence",
            "fresh probe is a diagnostic subset, not the complete /attacker suite"
        ]
    }
    if fragment_verification is not None:
        # Emitted only in fragmentation mode, so every other configuration
        # keeps its exact committed key set.
        result["fragmentation_training_verification"] = fragment_verification
    if grad_dp_on:
        # Emitted only when engaged, so --outbound-grad-dp off keeps the exact
        # key set of every artifact written before issue #105.
        result["outbound_grad_dp"] = {
            "mode": args.outbound_grad_dp,
            "clip_norm": args.grad_clip_norm,
            "noise_multiplier": grad_noise_multiplier,
            "noise_std": grad_noise_multiplier * args.grad_clip_norm,
            "protected_frames": grad_dp["frames"],
            "protected_rows": grad_dp["rows"],
            "mean_clip_scale": (grad_dp["clip_scale_sum"] / grad_dp["frames"]
                                if grad_dp["frames"] else None),
            "max_preclip_row_norm": grad_dp["max_preclip_norm"],
        }
    if trusted_recorder is not None:
        # Emitted only when engaged, like outbound_grad_dp above, so a default
        # run keeps the exact key set of every committed artifact.
        result["trusted_checkpoint"] = {
            "snapshot": str(trusted_recorder.snapshot_path),
            "usage_trace": str(trusted_recorder.usage_path),
            "steps_recorded": trusted_recorder.entries,
        }
    if not args.dp_account_untransmitted:
        result["dp_untransmitted_releases_excluded"] = {
            # The site is named by description only; a line-number literal
            # would rot on every edit.
            "site": "post-training probe phase: tln.encode with no "
                    "chan.forward",
            "encode_calls": untransmitted["calls"],
            "rows": untransmitted["rows"],
        }
    if args.delegate_sublayer != "full":
        # Emitted only when engaged, on purpose: a default run keeps the exact
        # key set of every committed artifact, and bin/deleg6040_config_diff.py
        # gates absent-vs-present as a CONFIG violation.
        result["delegate_sublayer"] = args.delegate_sublayer
        result["limitations"].append(
            "sublayer delegation: one channel crossing per block, so the "
            "trusted sublayers run on a trajectory that omits the delegated "
            "sublayers; not the interleaved per-sublayer protocol")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if remote:
        remote.close()
    return 0
if __name__ == "__main__":
    raise SystemExit(main())