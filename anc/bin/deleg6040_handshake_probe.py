#!/usr/bin/env python3
"""D0 handshake probe for the 60/40 delegation cell.

Uses the production client (privacy_runtime.latent_protocol.RemoteLatentCloud),
not a re-implementation, so a passing probe proves the same code path the runner
takes. Two assertions:

  AC0.3           the radial-8/D=64 server accepts the v13 hello and echoes
                  cloud_kind=monomial_moe_radial, cloud_experts=8.
  AC0.4-negative  the monomial/1-expert server REFUSES the same hello. The
                  fail-closed path (split-training/latent_cloud_server.py:100-113)
                  is what licenses trusting a passing run.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from privacy_runtime.latent_protocol import RemoteLatentCloud  # noqa: E402

V13_HELLO = dict(latent_dim=64, lr=3e-4, cloud_kind="monomial_moe_radial",
                 cloud_seed=42, cloud_experts=8, cloud_layers=2)


def probe(url: str, ca: str) -> tuple[bool, object]:
    try:
        remote = RemoteLatentCloud(url=url, tls_ca=ca, **V13_HELLO)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    audit = dict(remote.audit)
    remote.ws.close()
    return True, audit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--good-url", required=True, help="radial-8 D=64 server")
    ap.add_argument("--bad-url", required=True, help="monomial/1-expert server")
    ap.add_argument("--ca", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    good_ok, good = probe(args.good_url, args.ca)
    bad_ok, bad = probe(args.bad_url, args.ca)

    report = {
        "schema": "dtraining.deleg6040.handshake_probe.v1",
        "hello": V13_HELLO,
        "ac0_3_positive": {"url": args.good_url, "accepted": good_ok,
                           "ack" if good_ok else "error": good},
        "ac0_4_negative": {"url": args.bad_url, "refused": not bad_ok,
                           "error" if not bad_ok else "ack": bad},
        "ac0_3_pass": bool(good_ok
                           and isinstance(good, dict)
                           and good.get("cloud_kind") == "monomial_moe_radial"
                           and int(good.get("cloud_experts", -1)) == 8
                           and int(good.get("latent_dim", -1)) == 64
                           and good.get("latent_only_audit") is True),
        "ac0_4_pass": not bad_ok,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if (report["ac0_3_pass"] and report["ac0_4_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
