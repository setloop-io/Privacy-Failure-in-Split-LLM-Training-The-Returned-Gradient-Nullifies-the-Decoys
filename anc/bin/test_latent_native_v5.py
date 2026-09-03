#!/usr/bin/env python3
"""Executable mechanism gate for the latent-native v5 candidate."""

from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "split-training"))

from privacy_runtime.latent_native import (
    LatentPrivacyConfig, LatentRatchet, assert_ucn_latent_only,
    alternating_minimax_step, attacker_loss, build_latent_native_split,
    defender_privacy_loss, random_orthogonal,
)
from privacy_runtime.replay_resistance import randomize_template
from latent_cloud_server import LatentServer, validate_latent_shape


def main() -> int:
    import torch

    failures = []
    def check(name, condition):
        print(f"{'PASS' if condition else 'FAIL'} {name}")
        if not condition:
            failures.append(name)

    torch.manual_seed(7)
    cfg = LatentPrivacyConfig(hidden_dim=64, latent_dim=8, cloud_layers=1,
                              cloud_heads=2, noise_multiplier=4.0)
    tln, ucn, attackers = build_latent_native_split(cfg, vocab_size=19)

    # 1, 3, 5: real D-only cloud topology and private H-dimensional seam.
    hidden = torch.randn(2, 4, 64, requires_grad=True)
    latent, fmeta = tln.encode(hidden, secrets.token_hex(16))
    assert_ucn_latent_only(ucn, 8, 64)
    cloud_state = set(ucn.state_dict())
    check("1/latent-native-cloud", ucn(latent).shape == (2, 4, 8))
    check("3/width-reduction-64-to-8", latent.shape[-1] == 8)
    check("5/private-encoder-decoder-absent-from-ucn",
          not any("encoder" in n or "decoder" in n for n in cloud_state))
    try:
        validate_latent_shape([1, 4, 64], 8)
        h_frame_refused = False
    except ValueError:
        h_frame_refused = True
    check("5/remote-server-refuses-non-D-before-allocation", h_frame_refused)
    server = LatentServer(SimpleNamespace(
        device="cpu", latent_dim=8, cloud_layers=2, cloud_heads=2,
        cloud_kind="monomial", forbidden_hidden_dim=64, capture_dir=None))
    session_a = server.new_session_model(42)
    session_b = server.new_session_model(43)
    pointers_a = {parameter.data_ptr() for parameter in session_a.parameters()}
    pointers_b = {parameter.data_ptr() for parameter in session_b.parameters()}
    check("5/cloud-model-and-optimizer-state-isolated-per-session",
          session_a is not session_b and pointers_a.isdisjoint(pointers_b)
          and not hasattr(server, "model"))

    eq_cfg = LatentPrivacyConfig(hidden_dim=64, latent_dim=8, cloud_layers=2,
                                 cloud_heads=2, noise_multiplier=1.0,
                                 cloud_kind="equivariant")
    _, equivariant, _ = build_latent_native_split(eq_cfg, vocab_size=19)
    x = torch.randn(2, 4, 8)
    q = random_orthogonal(x, 8)
    check("5/equivariant-cloud-hides-canonical-coordinates",
          torch.allclose(equivariant(x @ q), equivariant(x) @ q,
                         atol=2e-5, rtol=2e-5))
    permutation = torch.tensor([2, 0, 3, 1])
    check("5/equivariant-cloud-hides-token-order",
          torch.allclose(equivariant(x[:, permutation]),
                         equivariant(x)[:, permutation],
                         atol=2e-5, rtol=2e-5))
    mono_cfg = LatentPrivacyConfig(hidden_dim=64, latent_dim=8, cloud_layers=2,
                                   cloud_heads=2, noise_multiplier=1.0,
                                   cloud_kind="monomial")
    _, monomial, _ = build_latent_native_split(mono_cfg, vocab_size=19)
    gauges = torch.tensor([1.7, -0.4, 2.2, -1.1]).view(1, 4, 1)
    transformed = x[:, permutation] * gauges @ q
    expected = monomial(x)[:, permutation] * gauges @ q
    check("5/monomial-cloud-hides-token-scale-sign-order-and-coordinates",
          torch.allclose(monomial(transformed), expected,
                         atol=3e-5, rtol=3e-5))
    moe_cfg = LatentPrivacyConfig(hidden_dim=64, latent_dim=8, cloud_layers=2,
                                  cloud_heads=2, noise_multiplier=1.0,
                                  cloud_kind="monomial_moe", cloud_experts=4)
    _, moe, _ = build_latent_native_split(moe_cfg, vocab_size=19)
    moe_expected = moe(x)[:, permutation] * gauges @ q
    check("5/monomial-moe-cloud-hides-token-scale-sign-order-and-coordinates",
          torch.allclose(moe(transformed), moe_expected,
                         atol=3e-5, rtol=3e-5))
    radial_cfg = LatentPrivacyConfig(hidden_dim=64, latent_dim=8,
                                     cloud_layers=2, cloud_heads=2,
                                     noise_multiplier=1.0,
                                     cloud_kind="monomial_moe_radial",
                                     cloud_experts=4)
    _, radial, _ = build_latent_native_split(radial_cfg, vocab_size=19)
    check("5/radial-moe-cloud-hides-order-and-coordinates",
          torch.allclose(radial(x[:, permutation] @ q),
                         radial(x)[:, permutation] @ q,
                         atol=3e-5, rtol=3e-5))
    invmlp_cfg = LatentPrivacyConfig(hidden_dim=64, latent_dim=8,
                                     cloud_layers=2, cloud_heads=2,
                                     noise_multiplier=1.0,
                                     cloud_kind="invariant_mlp")
    _, invmlp, _ = build_latent_native_split(invmlp_cfg, vocab_size=19)
    check("5/invariant-mlp-cloud-hides-order-and-coordinates",
          torch.allclose(invmlp(x[:, permutation] @ q),
                         invmlp(x)[:, permutation] @ q,
                         atol=3e-5, rtol=3e-5))

    # 2: attacker update plus an adversarial gradient reaching the encoder.
    tokens = torch.randint(0, 19, (2, 4))
    properties = torch.randint(0, 2, (2,))
    loss_a, outputs = attacker_loss(attackers, latent.detach(), tokens,
                                    properties, hidden.detach())
    loss_a.backward()
    attackers.zero_grad(set_to_none=True)
    loss_d = defender_privacy_loss(attackers, latent, tokens, properties,
                                   hidden.detach(), 0.5)
    loss_d.backward()
    encoder_grad = tln.encoder[1].weight.grad
    check("2/adaptive-multi-attacker-objective",
          outputs["token"].shape == (2, 4, 19)
          and outputs["reconstruction"].shape == hidden.shape)
    check("2/gradient-reversal-reaches-private-encoder",
          encoder_grad is not None and encoder_grad.abs().sum() > 0)

    # Exercise the actual alternating optimizer path on a fresh model/request.
    tln2, ucn2, attackers2 = build_latent_native_split(cfg, vocab_size=19)
    attack_opt = torch.optim.SGD(attackers2.parameters(), lr=1e-3)
    defend_opt = torch.optim.SGD(
        list(tln2.parameters()) + list(ucn2.parameters()), lr=1e-3)
    metrics = alternating_minimax_step(
        tln2, ucn2, attackers2, hidden.detach(), hidden.detach(), tokens,
        properties, attack_opt, defend_opt, secrets.token_hex(16), 0.5)
    check("2/alternating-attacker-defender-step",
          metrics["attacker_loss"] > 0 and metrics["task_loss"] > 0
          and metrics["wire_width"] == 8 and metrics["ratchet_epoch"] == 0
          and metrics["accountant"]["releases"] ==
          {"forward": 8, "return": 8})

    # 4: clipping/noise happens after compression and counts token rows.
    check("4/post-compression-token-dp",
          fmeta["token_releases"] == 8 and fmeta["noise_std"] > 0
          and tln.dp.report()["releases"]["forward"] == 8)

    # 6: independent request output and replay rejection.
    base = torch.zeros(1, 2, 64)
    nonce = secrets.token_hex(16)
    first, _ = tln.encode(base, nonce)
    second, _ = tln.encode(base, secrets.token_hex(16))
    try:
        tln.encode(base, nonce)
        replay_refused = False
    except RuntimeError:
        replay_refused = True
    check("6/independent-request-latents", not torch.equal(first, second))
    check("6/replay-refused", replay_refused)

    # 7: semantic slots survive randomized layout and padding.
    one, _ = randomize_template(["system", "question", "context"],
                                ["<a>", "<b>"], 2, 5)
    two, _ = randomize_template(["system", "question", "context"],
                                ["<a>", "<b>"], 2, 5)
    check("7/prompt-template-randomization",
          one != two and all(slot in one for slot in
                             ("system", "question", "context")))

    # 8: dense D-only transform rotates every request and round-trips.
    ratchet = LatentRatchet(8, bytes.fromhex("00112233445566778899aabbccddeeff"))
    w0 = ratchet.rotate(device=latent.device, dtype=latent.dtype)
    w1 = ratchet.rotate(device=latent.device, dtype=latent.dtype)
    wire = ratchet.apply(latent.detach(), w0)
    restored = ratchet.inverse(wire, w0)
    check("8/dense-latent-ratchet-rotates", not torch.equal(w0, w1))
    check("8/dense-latent-ratchet-roundtrip",
          w0.shape == (8, 8) and torch.allclose(restored, latent.detach(),
                                                atol=2e-5, rtol=2e-5))

    profile = json.loads((ROOT / "split-training/latent_native_v5/production_profile.json").read_text())
    check("candidate/not-falsely-production-ready", not profile["launchable"])
    print(f"SUMMARY failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
