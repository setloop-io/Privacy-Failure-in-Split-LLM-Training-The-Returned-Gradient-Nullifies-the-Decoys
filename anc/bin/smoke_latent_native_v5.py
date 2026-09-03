#!/usr/bin/env python3
"""Production-shape smoke for the v5 latent-native training mechanics."""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from privacy_runtime.latent_native import (
    LatentPrivacyConfig, alternating_minimax_step, assert_ucn_latent_only,
    build_latent_native_split,
)


def main() -> int:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--output")
    args = parser.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.manual_seed(19)
    cfg = LatentPrivacyConfig()
    tln, ucn, attackers = build_latent_native_split(cfg, vocab_size=256)
    assert_ucn_latent_only(ucn, cfg.latent_dim, cfg.hidden_dim)
    tln.to(device); ucn.to(device); attackers.to(device)
    hidden = torch.randn(1, 8, cfg.hidden_dim, device=device)
    tokens = torch.randint(0, 256, (1, 8), device=device)
    properties = torch.randint(0, 2, (1,), device=device)
    attack_opt = torch.optim.AdamW(attackers.parameters(), lr=1e-4)
    defend_opt = torch.optim.AdamW(
        list(tln.parameters()) + list(ucn.parameters()), lr=1e-4)
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    metrics = alternating_minimax_step(
        tln, ucn, attackers, hidden, hidden, tokens, properties,
        attack_opt, defend_opt, secrets.token_hex(16), cfg.adversary_strength)
    if device == "cuda":
        torch.cuda.synchronize()
    result = {
        "status": "mechanics_passed", "device": device,
        "hidden_dim": cfg.hidden_dim, "latent_dim": cfg.latent_dim,
        "elapsed_seconds": time.perf_counter() - started,
        "cloud_state_contains_hidden_width": any(
            cfg.hidden_dim in value.shape for value in ucn.state_dict().values()),
        "metrics": metrics,
        "claim": "mechanics only; adaptive privacy and utility not established",
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
