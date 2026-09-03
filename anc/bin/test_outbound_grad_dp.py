#!/usr/bin/env python3
"""Acceptance test for the ported outbound-gradient DP leg (experiment W1.2).

The E1 result was produced by code that is not in this repository. This test is
the evidence that the port is faithful: it exercises the flag surface, the
accountant's third direction, the trusted-side protection, and the mechanism E1
measured -- a chaff row's gradient is identically zero, and that zero pattern
discloses the real/decoy partition.

Two sections. The pure-Python section runs anywhere. The torch section runs only
where torch is importable (the container), and says so rather than passing
silently.

    python3 bin/test_outbound_grad_dp.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from privacy_runtime.activation_dp import BidirectionalBoundaryDP  # noqa: E402
from privacy_runtime.latent_native import LatentPrivacyConfig  # noqa: E402

RUNNER = ROOT / "bin" / "run_latent_native_v5_06b.py"

# The exact key set every artifact written before the fix carries. Guarded so a
# default run cannot start reporting a third leg by accident.
PREFIX_DIRECTIONS = {"forward", "return"}


def expect_raises(call, message: str) -> None:
    """Assert that ``call`` raises ValueError containing ``message``."""
    try:
        call()
    except ValueError as error:
        assert message in str(error), f"wrong message: {error}"
        return
    raise AssertionError(f"expected ValueError containing {message!r}")


def test_config_validation() -> None:
    """A half-configured or non-positive gradient leg is rejected at config."""
    base = dict(hidden_dim=64, latent_dim=16, cloud_heads=4)
    LatentPrivacyConfig(**base).validate()
    LatentPrivacyConfig(**base, gradient_clip_norm=0.01,
                        gradient_noise_multiplier=0.35).validate()
    expect_raises(
        LatentPrivacyConfig(**base, gradient_clip_norm=0.01).validate,
        "must be set together")
    expect_raises(
        LatentPrivacyConfig(**base, gradient_noise_multiplier=0.35).validate,
        "must be set together")
    expect_raises(
        LatentPrivacyConfig(**base, gradient_clip_norm=0.0,
                            gradient_noise_multiplier=0.35).validate,
        "must be positive")


def test_accountant_directions() -> None:
    """Undeclared, the accountant is exactly the pre-fix two-direction one."""
    plain = BidirectionalBoundaryDP(1.0, 8.0, 1.0, 8.0, 1e-6)
    assert set(plain.accountant.releases) == PREFIX_DIRECTIONS
    report = plain.report()
    assert set(report["releases"]) == PREFIX_DIRECTIONS
    assert "gradient_clip" not in report["parameters"]
    try:
        plain.protect_gradient(None)
    except RuntimeError as error:
        assert "not declared" in str(error)
    else:
        raise AssertionError("undeclared gradient direction must be rejected")

    leg = BidirectionalBoundaryDP(1.0, 8.0, 1.0, 8.0, 1e-6,
                                  gradient_clip=0.01, gradient_noise=0.35)
    assert set(leg.accountant.releases) == PREFIX_DIRECTIONS | {"gradient"}
    assert leg.report()["parameters"]["gradient_clip"] == 0.01


def runner_help() -> str:
    """The runner's own --help text, read from a real subprocess."""
    done = subprocess.run([sys.executable, str(RUNNER), "--help"],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout


def test_runner_flag_surface() -> None:
    """The four ported flags exist, with the source-of-record defaults."""
    text = " ".join(runner_help().split())
    for flag in ("--outbound-grad-dp {clip_noise,off}", "--grad-clip-norm",
                 "--grad-noise-multiplier", "--dp-account-untransmitted"):
        assert flag in text, f"missing flag surface: {flag}"
    assert "'off' restores the unprotected, unaccounted backward wire" in text, \
        "the 'off' arm's semantics must be documented in --help"


def test_runner_rejects_bad_gradient_dp() -> None:
    """Validation fires on argv, before any model or corpus is touched."""
    done = subprocess.run(
        [sys.executable, str(RUNNER), "--model", "/nonexistent",
         "--corpus", "/nonexistent", "--output", "/dev/null",
         "--grad-clip-norm", "0"],
        capture_output=True, text=True, timeout=300)
    assert done.returncode != 0
    assert "outbound gradient clip and noise must be positive" in done.stderr, \
        done.stderr[-2000:]


def torch_config() -> LatentPrivacyConfig:
    return LatentPrivacyConfig(
        hidden_dim=64, latent_dim=16, cloud_layers=1, cloud_heads=4,
        clip_norm=1.0, noise_multiplier=0.35,
        gradient_clip_norm=0.01, gradient_noise_multiplier=0.35)


def test_zero_row_support_destroyed() -> None:
    """The E1 mechanism: protection removes the chaff partition's zero support.

    A chaff row's raw outbound gradient is identically zero. E1 measured that
    zero pattern disclosing the real/decoy split in 4,096 of 4,096 frames. After
    clip-and-noise every wire row is non-zero, so the pattern is gone.
    """
    import torch
    from privacy_runtime.latent_native import build_latent_native_split

    torch.manual_seed(0)
    tln, _, _ = build_latent_native_split(torch_config(), vocab_size=8)
    raw = torch.randn(1, 80, 16)
    raw[:, 32:, :] = 0.0                      # 48 chaff rows, 32 real
    assert int((raw.abs().sum(dim=-1) == 0).sum()) == 48

    protected, meta = tln.protect_gradient(raw, "a" * 32)
    zero_rows = int((protected.abs().sum(dim=-1) == 0).sum())
    assert zero_rows == 0, f"{zero_rows} rows still disclose the partition"
    assert meta["token_releases"] == 80
    assert tln.dp.accountant.releases["gradient"] == 80


def test_gradient_leg_absent_by_default() -> None:
    """With no leg configured the tensor is returned untouched, unaccounted."""
    import torch
    from privacy_runtime.latent_native import build_latent_native_split

    torch.manual_seed(0)
    plain = LatentPrivacyConfig(hidden_dim=64, latent_dim=16, cloud_layers=1,
                                cloud_heads=4, noise_multiplier=0.35)
    tln, _, _ = build_latent_native_split(plain, vocab_size=8)
    raw = torch.randn(1, 8, 16)
    out, meta = tln.protect_gradient(raw, "b" * 32)
    assert torch.equal(out, raw), "default path must not alter the gradient"
    assert meta == {"protected": False}
    assert set(tln.dp.accountant.releases) == PREFIX_DIRECTIONS


def test_untransmitted_encode_is_not_charged() -> None:
    """transmitted=False protects the latent but charges no release."""
    import torch
    from privacy_runtime.latent_native import build_latent_native_split

    torch.manual_seed(0)
    tln, _, _ = build_latent_native_split(torch_config(), vocab_size=8)
    hidden = torch.randn(1, 8, 64)

    _, charged = tln.encode(hidden, "c" * 32, transmitted=True)
    after_charged = tln.dp.accountant.releases["forward"]
    assert after_charged == charged["token_releases"] == 8

    latent, free = tln.encode(hidden, "d" * 32, transmitted=False)
    assert tln.dp.accountant.releases["forward"] == after_charged, \
        "an untransmitted re-derivation must not be charged"
    assert free["token_releases"] == 8, "metadata still reports the rows"
    assert not torch.equal(latent, hidden[..., :16]), "still protected"


def test_bundle_report_config_is_backward_compatible() -> None:
    """The bundle report gains the gradient keys without breaking old captures.

    Captures recorded before the fix carry none of the three optional keys.
    Reading one must still work, and must not invent a value that would let a
    pre-fix capture read as protected.
    """
    sys.path.insert(0, str(ROOT / "bin"))
    from deleg6040_grad_bundle import (CONFIG_KEYS, OPTIONAL_CONFIG_KEYS,
                                       capture_config)

    legacy = {key: 1 for key in CONFIG_KEYS}
    config = capture_config(legacy)
    assert set(config) == set(CONFIG_KEYS), "a legacy capture gained keys"
    for key in OPTIONAL_CONFIG_KEYS:
        assert key not in config

    modern = dict(legacy, outbound_grad_dp="off", grad_clip_norm=None,
                  grad_noise_multiplier=None)
    config = capture_config(modern)
    assert config["outbound_grad_dp"] == "off"
    assert config["grad_clip_norm"] is None

    del legacy["steps"]
    try:
        capture_config(legacy)
    except KeyError:
        pass
    else:
        raise AssertionError("a missing required key must fail hard")


# The runner imports torch inside main(), before it parses argv, so even its
# --help needs the container. That is why the flag-surface tests are here.
PURE = [test_config_validation, test_accountant_directions]
TORCH = [test_runner_flag_surface, test_runner_rejects_bad_gradient_dp,
         test_bundle_report_config_is_backward_compatible,
         test_zero_row_support_destroyed, test_gradient_leg_absent_by_default,
         test_untransmitted_encode_is_not_charged]


def main() -> int:
    failures = 0
    for test in PURE:
        try:
            test()
            print(f"ok   {test.__name__}")
        except Exception as error:                        # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {error}")

    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP torch section: torch is not importable here. "
              "Run this inside split-inference:spark; the port is NOT "
              "verified without it.")
        return 1 if failures else 0

    for test in TORCH:
        try:
            test()
            print(f"ok   {test.__name__}")
        except Exception as error:                        # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
