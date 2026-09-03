"""Formal Gaussian privacy mechanisms for split-compute boundaries.

Replace-one adjacency: per-sample vectors clipped to L2 norm C have
sensitivity 2C, and the Gaussian noise has standard deviation
``noise_multiplier * C``.  Composition is reported through zCDP and
converted to (epsilon, delta)-DP; forward and return directions are
accounted separately and jointly.
"""

from __future__ import annotations

import math


def rho_for_epsilon(epsilon: float, delta: float) -> float:
    """Invert epsilon = rho + 2*sqrt(rho*log(1/delta))."""
    if epsilon <= 0 or not 0 < delta < 1:
        raise ValueError("epsilon must be positive and delta must be in (0,1)")
    root = math.sqrt(math.log(1.0 / delta) + epsilon) - math.sqrt(
        math.log(1.0 / delta))
    return root * root


def noise_for_rho(releases: int, rho: float,
                  adjacency: str = "replace_one") -> float:
    if releases <= 0 or rho <= 0:
        raise ValueError("releases and rho must be positive")
    sensitivity = 2.0 if adjacency == "replace_one" else 1.0
    return sensitivity * math.sqrt(releases / (2.0 * rho))


class BoundaryDPAccountant:
    def __init__(self, delta: float, adjacency: str = "replace_one",
                 directions: tuple[str, ...] = ("forward", "return")):
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if adjacency not in ("replace_one", "add_remove"):
            raise ValueError("unsupported adjacency")
        if not directions:
            raise ValueError("at least one accounted direction is required")
        self.delta = float(delta)
        self.adjacency = adjacency
        self.releases = {name: 0 for name in directions}
        self.rho = {name: 0.0 for name in directions}

    @property
    def sensitivity_multiplier(self) -> float:
        return 2.0 if self.adjacency == "replace_one" else 1.0

    def record(self, direction: str, noise_multiplier: float,
               releases: int = 1) -> None:
        if direction not in self.releases:
            raise ValueError(
                f"direction must be one of {sorted(self.releases)}")
        if noise_multiplier <= 0 or releases <= 0:
            raise ValueError("noise_multiplier and releases must be positive")
        sensitivity = self.sensitivity_multiplier
        self.releases[direction] += int(releases)
        self.rho[direction] += releases * sensitivity ** 2 / (
            2.0 * noise_multiplier ** 2)

    def epsilon(self, direction: str | None = None) -> float:
        rho = sum(self.rho.values()) if direction is None else self.rho[direction]
        if rho == 0:
            return 0.0
        return rho + 2.0 * math.sqrt(rho * math.log(1.0 / self.delta))

    def report(self) -> dict:
        return {
            "method": "zcdp_gaussian_replace_one" if self.adjacency == "replace_one"
                      else "zcdp_gaussian_add_remove",
            "adjacency": self.adjacency,
            "delta": self.delta,
            "releases": dict(self.releases),
            "rho": dict(self.rho),
            "epsilon": {**{name: self.epsilon(name) for name in self.rho},
                        "composed": self.epsilon()},
        }


def clip_and_noise(tensor, max_norm: float, noise_multiplier: float,
                   accountant: BoundaryDPAccountant, direction: str,
                   generator=None, account: bool = True):
    """Clip each batch item and add calibrated Gaussian noise.

    Each final-dimension row is one token-level privacy vector, matching the
    labelled token-pair attacker: a sequence is never counted as one release
    while exposing many independently labelled rows.

    ``account=False`` protects the tensor without charging the budget, for
    trusted-side re-derivations that never cross the boundary: a release is
    a crossing, so counting one that never happened misstates the budget.
    """
    import torch

    if tensor.ndim < 2:
        raise ValueError("boundary tensor must include batch and feature dimensions")
    if max_norm <= 0 or noise_multiplier <= 0:
        raise ValueError("max_norm and noise_multiplier must be positive")
    flat = tensor.float().reshape(-1, tensor.shape[-1])
    norms = flat.norm(2, dim=1, keepdim=True)
    scales = (max_norm / norms.clamp_min(1e-12)).clamp(max=1.0)
    clipped = (flat * scales).reshape_as(tensor.float())
    noise = torch.randn(clipped.shape, dtype=clipped.dtype,
                        device=clipped.device, generator=generator)
    protected = clipped + noise * (noise_multiplier * max_norm)
    releases = int(flat.shape[0])
    if account:
        accountant.record(direction, noise_multiplier, releases=releases)
    return protected.to(dtype=tensor.dtype), {
        "max_preclip_norm": float(norms.max().item()),
        "mean_clip_scale": float(scales.mean().item()),
        "noise_std": float(noise_multiplier * max_norm),
        "token_releases": releases,
        "adjacency_unit": "one_boundary_token_row",
    }


class BidirectionalBoundaryDP:
    """Forward and return boundary DP, plus an optional outbound-gradient leg.

    ``gradient_clip``/``gradient_noise`` declare the third direction: the
    output gradient the trusted node sends to the untrusted node during
    training.  Left unset the object is exactly the two-direction accountant
    every committed artifact was produced with, key for key.
    """

    def __init__(self, forward_clip: float, forward_noise: float,
                 return_clip: float, return_noise: float, delta: float,
                 adjacency: str = "replace_one",
                 gradient_clip: float | None = None,
                 gradient_noise: float | None = None):
        if (gradient_clip is None) != (gradient_noise is None):
            raise ValueError("gradient clip and noise must be set together")
        self.forward_clip = float(forward_clip)
        self.forward_noise = float(forward_noise)
        self.return_clip = float(return_clip)
        self.return_noise = float(return_noise)
        self.gradient_clip = (None if gradient_clip is None
                              else float(gradient_clip))
        self.gradient_noise = (None if gradient_noise is None
                               else float(gradient_noise))
        directions = ("forward", "return") if self.gradient_clip is None \
            else ("forward", "return", "gradient")
        self.accountant = BoundaryDPAccountant(delta, adjacency, directions)

    def protect_forward(self, tensor, generator=None, account: bool = True):
        return clip_and_noise(tensor, self.forward_clip, self.forward_noise,
                              self.accountant, "forward", generator, account)

    def protect_return(self, tensor, generator=None):
        return clip_and_noise(tensor, self.return_clip, self.return_noise,
                              self.accountant, "return", generator)

    def protect_gradient(self, tensor, generator=None):
        """Clip and noise the output gradient that leaves the trusted node."""
        if self.gradient_clip is None:
            raise RuntimeError("gradient direction was not declared")
        return clip_and_noise(tensor, self.gradient_clip, self.gradient_noise,
                              self.accountant, "gradient", generator)

    def report(self) -> dict:
        result = self.accountant.report()
        result["parameters"] = {
            "forward_clip": self.forward_clip,
            "forward_noise_multiplier": self.forward_noise,
            "return_clip": self.return_clip,
            "return_noise_multiplier": self.return_noise,
        }
        if self.gradient_clip is not None:
            result["parameters"]["gradient_clip"] = self.gradient_clip
            result["parameters"]["gradient_noise_multiplier"] = \
                self.gradient_noise
        return result
