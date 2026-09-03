"""Latent-native split-compute prototype.

TLN owns the H->D encoder and D->H decoder.  The object exported to UCN
contains only D-width operations.  This is an empirical privacy architecture,
not encryption: a compromised UCN still observes the released D-dimensional
latents and may train arbitrary adaptive attackers against them.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

from privacy_runtime.activation_dp import BidirectionalBoundaryDP
from privacy_runtime.bottleneck import gradient_reverse
from privacy_runtime.ratchet_v2 import derive_orthogonal
from privacy_runtime.replay_resistance import ReplayResistantSampler


@dataclass(frozen=True)
class LatentPrivacyConfig:
    hidden_dim: int = 1024
    latent_dim: int = 128
    cloud_layers: int = 2
    cloud_heads: int = 4
    clip_norm: float = 1.0
    noise_multiplier: float = 8.0
    delta: float = 1e-6
    adversary_strength: float = 0.25
    cloud_kind: str = "transformer"
    cloud_experts: int = 1
    cloud_hidden: int = 0
    # Outbound output-gradient DP.  Unset = the unprotected, unaccounted
    # backward wire every pre-gradient-DP artifact was produced with.
    gradient_clip_norm: float | None = None
    gradient_noise_multiplier: float | None = None

    def validate(self) -> None:
        # latent_dim == hidden_dim is the degenerate no-bottleneck point,
        # allowed as an experimental control (full-width defended channel).
        if not 0 < self.latent_dim <= self.hidden_dim:
            raise ValueError("latent_dim must not exceed hidden_dim")
        if (self.gradient_clip_norm is None) != (
                self.gradient_noise_multiplier is None):
            raise ValueError("gradient clip and noise must be set together")
        if self.gradient_clip_norm is not None and (
                self.gradient_clip_norm <= 0
                or self.gradient_noise_multiplier <= 0):
            raise ValueError("gradient clip and noise must be positive")
        if self.latent_dim % self.cloud_heads:
            raise ValueError("latent_dim must be divisible by cloud_heads")
        if self.cloud_layers <= 0:
            raise ValueError("cloud_layers must be positive")
        if self.cloud_kind not in ("transformer", "equivariant", "monomial",
                                   "monomial_moe", "monomial_moe_radial",
                                   "invariant_mlp", "invariant_mlp_deep"):
            raise ValueError(
                "cloud_kind must be transformer, equivariant, monomial, "
                "monomial_moe, monomial_moe_radial, invariant_mlp or "
                "invariant_mlp_deep")
        if self.cloud_experts < 1:
            raise ValueError("cloud_experts must be positive")
        if self.cloud_kind in ("monomial_moe", "monomial_moe_radial") \
                and self.cloud_experts < 2:
            raise ValueError("monomial_moe requires at least two experts")
        if self.cloud_kind == "invariant_mlp_deep" and self.cloud_hidden < 64:
            raise ValueError("invariant_mlp_deep requires cloud_hidden >= 64")


def random_orthogonal(tensor, latent_dim: int):
    """Generate a fresh Haar-like orthogonal matrix on TLN.

    Fresh 128-bit CSPRNG master per call (drawn from the OS CSPRNG, never
    leaves the trusted process), expanded through the v2 portable stream.
    QR is intentionally paid at D width, not the canonical H width.
    """
    matrix = derive_orthogonal(secrets.token_bytes(16), 0, latent_dim)
    return matrix.to(device=tensor.device, dtype=tensor.dtype)


def latent_invariants(latent):
    """Rotation-invariant per-token features available to compromised UCN."""
    import torch

    x = latent.float()
    norms = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    unit = x / norms
    gram = unit @ unit.transpose(-1, -2)
    seq = latent.shape[-2]
    eye = torch.eye(seq, device=latent.device, dtype=torch.bool)
    other = gram.masked_fill(eye.unsqueeze(0), 0.0)
    denom = max(1, seq - 1)
    mean = other.sum(-1, keepdim=True) / denom
    variance = ((other - mean) ** 2).masked_fill(
        eye.unsqueeze(0), 0.0).sum(-1, keepdim=True) / denom
    maximum = other.masked_fill(eye.unsqueeze(0), -1.0).max(
        -1, keepdim=True).values
    minimum = other.masked_fill(eye.unsqueeze(0), 1.0).min(
        -1, keepdim=True).values
    similarities = other.sort(dim=-1, descending=True).values
    take = min(8, seq)
    top = similarities[..., :take]
    if take < 8:
        top = torch.nn.functional.pad(top, (0, 8 - take))
    position = torch.linspace(-1.0, 1.0, seq, device=latent.device,
                              dtype=x.dtype).view(1, seq, 1)
    position = position.expand(latent.shape[0], -1, -1)
    absolute_top = other.abs().sort(dim=-1, descending=True).values[..., :take]
    if take < 8:
        absolute_top = torch.nn.functional.pad(absolute_top, (0, 8 - take))
    return torch.cat((norms, mean, variance.sqrt(), maximum, minimum,
                      top, absolute_top, position), -1)


def _request_generator(tensor, nonce: str):
    """Create an independent local noise stream without publishing its seed."""
    import torch

    entropy = secrets.token_bytes(32)
    digest = hashlib.sha256(entropy + nonce.encode()).digest()
    seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    device = tensor.device.type if tensor.device.type == "cuda" else "cpu"
    return torch.Generator(device=device).manual_seed(seed)


class LatentRatchet:
    """Dense, request-scoped D-dimensional wire transform.

    This is defense in depth for separated wire captures. It is not relied on
    for irreversibility because the compromised cloud necessarily receives the
    current transform (or an already unwrapped latent) to execute its module.
    """

    def __init__(self, latent_dim: int, master: bytes | None = None):
        self.latent_dim = int(latent_dim)
        self._master = master or secrets.token_bytes(16)
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def rotate(self, *, device, dtype):
        matrix = derive_orthogonal(self._master, self._epoch, self.latent_dim)
        self._epoch += 1
        return matrix.to(device=device, dtype=dtype)

    @staticmethod
    def apply(value, matrix):
        return value @ matrix

    @staticmethod
    def inverse(value, matrix):
        return value @ matrix.transpose(-1, -2)


def build_ucn_latent_middle(latent_dim: int, cloud_layers: int = 2,
                              cloud_heads: int = 4,
                              cloud_kind: str = "transformer",
                              cloud_experts: int = 1,
                              cloud_hidden: int = 0):
    """Construct only the D-width module allowed on compromised UCN.

    This entry point deliberately takes no model hidden width (H) and never
    constructs a private encoder, decoder, attacker reconstruction head, or
    H-sized tensor.  ``cloud_hidden`` is an *internal* latent-space width for
    the big-compute kinds; the module's interface stays [batch, tokens, D].
    """
    import torch
    from torch import nn

    if latent_dim <= 0 or cloud_layers <= 0:
        raise ValueError("latent_dim and cloud_layers must be positive")
    if latent_dim % cloud_heads:
        raise ValueError("latent_dim must be divisible by cloud_heads")

    if cloud_kind not in ("transformer", "equivariant", "monomial",
                          "monomial_moe", "monomial_moe_radial",
                          "invariant_mlp", "invariant_mlp_deep"):
        raise ValueError(
            "cloud_kind must be transformer, equivariant, monomial, "
            "monomial_moe, monomial_moe_radial, invariant_mlp or "
            "invariant_mlp_deep")

    class UCNLatentMiddle(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=latent_dim, nhead=cloud_heads,
                dim_feedforward=4 * latent_dim, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True)
            self.layers = nn.TransformerEncoder(layer, cloud_layers)
            self.output_norm = nn.LayerNorm(latent_dim)
            self.latent_width = latent_dim

        def forward(self, latent):
            if latent.ndim != 3 or latent.shape[-1] != self.latent_width:
                raise ValueError("UCN accepts only [batch, tokens, D] latents")
            result = self.output_norm(self.layers(latent))
            if result.shape[-1] != self.latent_width:
                raise RuntimeError("UCN emitted a non-latent-width tensor")
            return result

    class EquivariantLatentMiddle(nn.Module):
        """D-only O(D)-equivariant cloud; no canonical coordinate system.

        Attention depends only on Gram matrices and each nonlinear gate is a
        scalar function of vector norms. Therefore f(XQ)=f(X)Q for every
        orthogonal Q, allowing TLN to rotate each request without sharing Q.
        """
        def __init__(self):
            super().__init__()
            self.latent_width = latent_dim
            self.attn_log_temperature = nn.Parameter(
                torch.zeros(cloud_layers))
            self.attn_gain = nn.Parameter(torch.full((cloud_layers,), 0.1))
            self.radial_scale = nn.Parameter(torch.ones(cloud_layers))
            self.radial_bias = nn.Parameter(torch.zeros(cloud_layers))
            self.radial_gain = nn.Parameter(torch.full((cloud_layers,), 0.1))
            self.output_gain = nn.Parameter(torch.ones(()))

        def forward(self, latent):
            if latent.ndim != 3 or latent.shape[-1] != self.latent_width:
                raise ValueError("UCN accepts only [batch, tokens, D] latents")
            x = latent
            for index in range(cloud_layers):
                xf = x.float()
                scores = (xf @ xf.transpose(-1, -2)) / latent_dim ** 0.5
                temperature = self.attn_log_temperature[index].exp().clamp(
                    0.05, 20.0)
                mixed = torch.softmax(scores * temperature, dim=-1) @ xf
                x = x + self.attn_gain[index] * mixed.to(x.dtype)
                norms = x.float().norm(dim=-1, keepdim=True) / latent_dim ** 0.5
                gate = torch.sigmoid(
                    self.radial_scale[index] * norms
                    + self.radial_bias[index])
                x = x + self.radial_gain[index] * gate.to(x.dtype) * x
            return self.output_gain * x

    class MonomialEquivariantLatentMiddle(nn.Module):
        """Equivariant to token permutation, signed scale, and O(D) gauge."""
        def __init__(self):
            super().__init__()
            self.latent_width = latent_dim
            self.log_temperature = nn.Parameter(torch.zeros(cloud_layers))
            self.message_gain = nn.Parameter(torch.full((cloud_layers,), 0.1))
            self.output_gain = nn.Parameter(torch.ones(()))

        def forward(self, latent):
            if latent.ndim != 3 or latent.shape[-1] != self.latent_width:
                raise ValueError("UCN accepts only [batch, tokens, D] latents")
            norms = latent.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            unit = latent.float() / norms
            for index in range(cloud_layers):
                gram = unit @ unit.transpose(-1, -2)
                temperature = self.log_temperature[index].exp().clamp(0.05, 20.0)
                weights = torch.softmax(temperature * gram.square(), dim=-1)
                message = (weights * gram) @ unit
                unit = torch.nn.functional.normalize(
                    unit + self.message_gain[index] * message,
                    dim=-1, eps=1e-6)
            return (self.output_gain * norms * unit).to(latent.dtype)

    class MonomialMoELatentMiddle(nn.Module):
        """Equivariant mixture-of-experts cloud.

        Same equivariance group as the monomial module (token permutation,
        signed scale, O(D) coordinate gauge) but with E experts per layer,
        each owning a distinct Gram kernel shape (quadratic temperature +
        signed linear weight + gain).  Routing uses per-row statistics of the
        squared Gram matrix only, which are invariant to rotation, sign and
        scale and covariant with row order — so the router itself respects
        every gauge.  Capacity and cloud FLOPs scale with the expert count
        while the wire width stays at D.
        """
        def __init__(self):
            super().__init__()
            self.latent_width = latent_dim
            self.log_temperature = nn.Parameter(
                torch.zeros(cloud_layers, cloud_experts))
            self.linear_weight = nn.Parameter(
                torch.zeros(cloud_layers, cloud_experts))
            self.message_gain = nn.Parameter(
                torch.full((cloud_layers, cloud_experts), 0.1))
            self.routers = nn.ModuleList(
                nn.Linear(3, cloud_experts) for _ in range(cloud_layers))
            self.output_gain = nn.Parameter(torch.ones(()))

        def forward(self, latent):
            if latent.ndim != 3 or latent.shape[-1] != self.latent_width:
                raise ValueError("UCN accepts only [batch, tokens, D] latents")
            norms = latent.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            unit = latent.float() / norms
            for index in range(cloud_layers):
                gram = unit @ unit.transpose(-1, -2)
                gram2 = gram.square()
                seq = gram2.shape[-1]
                eye = torch.eye(seq, device=gram2.device,
                                dtype=torch.bool).unsqueeze(0)
                others = gram2.masked_fill(eye, 0.0)
                denom = max(1, seq - 1)
                mean = others.sum(-1) / denom
                variance = ((others - mean.unsqueeze(-1))
                            .masked_fill(eye, 0.0).square()
                            .sum(-1) / denom)
                maximum = gram2.masked_fill(eye, -1.0).amax(-1)
                features = torch.stack(
                    (mean, variance.sqrt(), maximum), dim=-1)
                route = torch.softmax(self.routers[index](features.float()),
                                      dim=-1)
                temperature = self.log_temperature[index].exp().clamp(
                    0.05, 20.0)
                messages = []
                for expert in range(cloud_experts):
                    scores = (temperature[expert] * gram2
                              + self.linear_weight[index][expert] * gram)
                    weights = torch.softmax(scores, dim=-1)
                    messages.append(self.message_gain[index][expert]
                                    * ((weights * gram) @ unit))
                stacked = torch.stack(messages, dim=2)
                combined = (route.unsqueeze(-1) * stacked).sum(dim=2)
                unit = torch.nn.functional.normalize(
                    unit + combined, dim=-1, eps=1e-6)
            return (self.output_gain * norms * unit).to(latent.dtype)

    class MonomialMoERadialLatentMiddle(nn.Module):
        """Gram-kernel experts plus a per-row radial (norm) channel.

        Equivariant to token permutation and O(D) rotation ONLY.  The radial
        channel reads per-row norms, which the token-scale gauge randomizes —
        this cloud kind is therefore intentionally incompatible with the scale
        gauge, and exists to measure the privacy-vs-capacity tradeoff of
        dropping that gauge (rotation, permutation, noise and chaff stay).
        """
        def __init__(self):
            super().__init__()
            self.latent_width = latent_dim
            self.log_temperature = nn.Parameter(
                torch.zeros(cloud_layers, cloud_experts))
            self.linear_weight = nn.Parameter(
                torch.zeros(cloud_layers, cloud_experts))
            self.message_gain = nn.Parameter(
                torch.full((cloud_layers, cloud_experts), 0.1))
            self.radial_scale = nn.Parameter(
                torch.ones(cloud_layers, cloud_experts))
            self.radial_bias = nn.Parameter(
                torch.zeros(cloud_layers, cloud_experts))
            self.radial_gain = nn.Parameter(
                torch.full((cloud_layers, cloud_experts), 0.1))
            self.routers = nn.ModuleList(
                nn.Linear(3, cloud_experts) for _ in range(cloud_layers))
            self.output_gain = nn.Parameter(torch.ones(()))

        def forward(self, latent):
            if latent.ndim != 3 or latent.shape[-1] != self.latent_width:
                raise ValueError("UCN accepts only [batch, tokens, D] latents")
            norms = latent.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            unit = latent.float() / norms
            for index in range(cloud_layers):
                gram = unit @ unit.transpose(-1, -2)
                gram2 = gram.square()
                seq = gram2.shape[-1]
                eye = torch.eye(seq, device=gram2.device,
                                dtype=torch.bool).unsqueeze(0)
                others = gram2.masked_fill(eye, 0.0)
                denom = max(1, seq - 1)
                mean = others.sum(-1) / denom
                variance = ((others - mean.unsqueeze(-1))
                            .masked_fill(eye, 0.0).square()
                            .sum(-1) / denom)
                maximum = gram2.masked_fill(eye, -1.0).amax(-1)
                features = torch.stack(
                    (mean, variance.sqrt(), maximum), dim=-1)
                route = torch.softmax(self.routers[index](features.float()),
                                      dim=-1)
                temperature = self.log_temperature[index].exp().clamp(
                    0.05, 20.0)
                messages = []
                radials = []
                for expert in range(cloud_experts):
                    scores = (temperature[expert] * gram2
                              + self.linear_weight[index][expert] * gram)
                    weights = torch.softmax(scores, dim=-1)
                    messages.append(self.message_gain[index][expert]
                                    * ((weights * gram) @ unit))
                    radials.append(self.radial_gain[index][expert]
                                   * torch.sigmoid(
                                       self.radial_scale[index][expert] * norms
                                       + self.radial_bias[index][expert]))
                stacked = torch.stack(messages, dim=2)
                combined = (route.unsqueeze(-1) * stacked).sum(dim=2)
                unit = torch.nn.functional.normalize(
                    unit + combined, dim=-1, eps=1e-6)
                radial_total = (route.unsqueeze(-1)
                                * torch.stack(radials, dim=2)).sum(dim=2)
                norms = norms * (1.0 + radial_total)
            return (self.output_gain * norms * unit).to(latent.dtype)

    class InvariantMLPLatentMiddle(nn.Module):
        """Learned nonlinear cloud on gauge-invariant per-row features.

        Keeps the monomial message-passing skeleton but adds a per-row MLP
        over gauge-invariant features (norms + Gram-row statistics, position
        channel excluded) that gates both the norm scale and the message
        direction.  Norms are read, so like the radial kind it is
        intentionally incompatible with the token-scale gauge.
        """
        def __init__(self):
            super().__init__()
            self.latent_width = latent_dim
            feat_dim = latent_invariants(
                torch.zeros(1, 8, latent_dim)).shape[-1] - 1
            self.feat_dim = feat_dim
            hidden = max(64, 2 * feat_dim)
            # No LayerNorm: invariant features are O(1)-bounded by
            # construction (clipped norms, unit Gram), and the D-only audit
            # requires every norm to be exactly latent-width.
            self.gate_mlp = nn.ModuleList([nn.Sequential(
                nn.Linear(feat_dim, hidden),
                nn.GELU(), nn.Linear(hidden, 2))
                for _ in range(cloud_layers)])
            self.log_temperature = nn.Parameter(torch.zeros(cloud_layers))
            self.output_gain = nn.Parameter(torch.ones(()))

        def forward(self, latent):
            if latent.ndim != 3 or latent.shape[-1] != self.latent_width:
                raise ValueError("UCN accepts only [batch, tokens, D] latents")
            norms = latent.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            unit = latent.float() / norms
            for index in range(cloud_layers):
                gram = unit @ unit.transpose(-1, -2)
                gram2 = gram.square()
                temperature = self.log_temperature[index].exp().clamp(
                    0.05, 20.0)
                weights = torch.softmax(temperature * gram2, dim=-1)
                message = (weights * gram) @ unit
                # gauge-invariant per-row features, position channel dropped
                feats = latent_invariants(unit * norms)[..., :-1]
                gates = torch.tanh(self.gate_mlp[index](feats))
                scale_gate = gates[..., 0:1]
                message_gate = gates[..., 1:2]
                unit = torch.nn.functional.normalize(
                    unit + message_gate * message, dim=-1, eps=1e-6)
                norms = norms * (1.0 + scale_gate)
            return (self.output_gain * norms * unit).to(latent.dtype)

    class InvariantMLPDeepLatentMiddle(nn.Module):
        """The gauge-invariant family scaled to real compute.

        Same message-passing skeleton and gauge-invariant features as
        invariant_mlp, but each layer's gate is a deep wide MLP
        (feat -> H -> H -> 2) so the cloud's per-token FLOPs match the real
        layers it replaces.  Exists to test whether the D=64 utility ceiling
        is capacity-limited — a much bigger latent cloud would lift it — or
        information-limited — it would not.  Norms are read, so like the
        radial/invariant kinds it is intentionally incompatible with the
        token-scale gauge.
        """
        def __init__(self):
            super().__init__()
            if cloud_hidden < 64:
                raise ValueError("invariant_mlp_deep needs cloud_hidden >= 64")
            self.latent_width = latent_dim
            feat_dim = latent_invariants(
                torch.zeros(1, 8, latent_dim)).shape[-1] - 1
            self.feat_dim = feat_dim
            # No LayerNorm: invariant features are O(1)-bounded by
            # construction (clipped norms, unit Gram), and the D-only audit
            # requires every norm to be exactly latent-width.
            self.gate_mlp = nn.ModuleList([nn.Sequential(
                nn.Linear(feat_dim, cloud_hidden), nn.GELU(),
                nn.Linear(cloud_hidden, cloud_hidden), nn.GELU(),
                nn.Linear(cloud_hidden, 2))
                for _ in range(cloud_layers)])
            # Zero-init the gate heads: every gate starts at exactly 0, so
            # the deep stack is the identity map.  Without this, random
            # gates in (-1,1) compound multiplicatively over 28 layers and
            # the first backward goes non-finite.
            for gate in self.gate_mlp:
                nn.init.zeros_(gate[-1].weight)
                nn.init.zeros_(gate[-1].bias)
            self.log_temperature = nn.Parameter(torch.zeros(cloud_layers))
            self.output_gain = nn.Parameter(torch.ones(()))

        def forward(self, latent):
            if latent.ndim != 3 or latent.shape[-1] != self.latent_width:
                raise ValueError("UCN accepts only [batch, tokens, D] latents")
            norms = latent.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            unit = latent.float() / norms
            for index in range(cloud_layers):
                gram = unit @ unit.transpose(-1, -2)
                gram2 = gram.square()
                temperature = self.log_temperature[index].exp().clamp(
                    0.05, 20.0)
                weights = torch.softmax(temperature * gram2, dim=-1)
                message = (weights * gram) @ unit
                # gauge-invariant per-row features, position channel dropped
                feats = latent_invariants(unit * norms)[..., :-1]
                gates = torch.tanh(self.gate_mlp[index](feats))
                scale_gate = gates[..., 0:1]
                message_gate = gates[..., 1:2]
                unit = torch.nn.functional.normalize(
                    unit + message_gate * message, dim=-1, eps=1e-6)
                # Bounded log-multiplier: the per-layer factor stays in
                # [e^-0.1, e^0.1], so 28 layers compound to at most ~16x in
                # either direction.  The unbounded (1 + gate) product went
                # non-finite stochastically during training.
                norms = norms * torch.exp(0.1 * scale_gate)
            return (self.output_gain * norms * unit).to(latent.dtype)

    if cloud_kind == "transformer":
        return UCNLatentMiddle()
    if cloud_kind == "equivariant":
        return EquivariantLatentMiddle()
    if cloud_kind == "monomial_moe":
        return MonomialMoELatentMiddle()
    if cloud_kind == "monomial_moe_radial":
        return MonomialMoERadialLatentMiddle()
    if cloud_kind == "invariant_mlp":
        return InvariantMLPLatentMiddle()
    if cloud_kind == "invariant_mlp_deep":
        return InvariantMLPDeepLatentMiddle()
    return MonomialEquivariantLatentMiddle()


def build_latent_native_split(config: LatentPrivacyConfig, vocab_size: int,
                              property_classes: int = 2):
    """Build trusted TLN modules, a D-only UCN module, and attackers."""
    import torch
    from torch import nn

    config.validate()
    h, d = config.hidden_dim, config.latent_dim

    class TLNPrivateBoundary(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.LayerNorm(h), nn.Linear(h, 2 * d), nn.GELU(),
                nn.Linear(2 * d, d), nn.Tanh(),
            )
            self.decoder = nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(),
                nn.Linear(2 * d, h),
            )
            self.dp = BidirectionalBoundaryDP(
                config.clip_norm, config.noise_multiplier,
                config.clip_norm, config.noise_multiplier, config.delta,
                gradient_clip=config.gradient_clip_norm,
                gradient_noise=config.gradient_noise_multiplier,
            )
            self.nonces = ReplayResistantSampler(1)
            self.ratchet = LatentRatchet(d)

        def encode(self, hidden, nonce: str, transmitted: bool = True):
            """Protect a latent for release.

            ``transmitted=False`` marks a trusted-side re-derivation that never
            reaches the cloud, so it is protected but not charged to the budget:
            a release is a boundary crossing, and counting one that never
            happened misstates epsilon.
            """
            self.nonces.accept_nonce(nonce)
            latent = self.encoder(hidden)
            return self.dp.protect_forward(
                latent, _request_generator(latent, nonce + ":forward"),
                account=transmitted)

        def protect_gradient(self, gradient, nonce: str):
            """Clip and noise the output gradient sent back to the cloud.

            Runs on the trusted side before the tensor leaves the process, so the
            untrusted node never sees the raw pullback.  Every wire row is protected,
            chaff included -- a chaff row's raw gradient is identically zero, and that
            zero pattern disclosed the real/decoy partition in 4,096 of 4,096 frames.

            Returns the tensor unchanged when no gradient leg is configured, so the
            default path is exactly what every committed artifact was produced with.
            """
            if getattr(self.dp, "gradient_clip", None) is None:
                return gradient, {"protected": False}
            return self.dp.protect_gradient(
                gradient, _request_generator(gradient, nonce + ":gradient"))

        def decode(self, latent, nonce: str, residual=None):
            protected, metadata = self.dp.protect_return(
                latent, _request_generator(latent, nonce + ":return"))
            correction = self.decoder(protected)
            if residual is not None:
                if residual.shape != correction.shape:
                    raise ValueError("trusted residual shape does not match decoder")
                correction = residual + correction
            return correction, metadata

    class AdaptiveAttackers(nn.Module):
        def __init__(self):
            super().__init__()
            trunk = lambda: nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(),
            )
            self.token_trunk = trunk()
            self.token_head = nn.Linear(2 * d + 22, vocab_size)
            self.property_trunk = trunk()
            self.property_head = nn.Linear(2 * d, property_classes)
            self.reconstruction = nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(),
                nn.Linear(2 * d, h),
            )

        def forward(self, latent):
            token_features = torch.cat(
                (self.token_trunk(latent), latent_invariants(latent)), -1)
            token = self.token_head(token_features)
            pooled = latent.mean(dim=1)
            prop = self.property_head(self.property_trunk(pooled))
            reconstruction = self.reconstruction(latent)
            return {"token": token, "property": prop,
                    "reconstruction": reconstruction}

    return (TLNPrivateBoundary(),
            build_ucn_latent_middle(d, config.cloud_layers,
                                      config.cloud_heads, config.cloud_kind,
                                      config.cloud_experts,
                                      config.cloud_hidden),
            AdaptiveAttackers())


def attacker_loss(attackers, latent, token_labels, property_labels,
                  hidden_targets):
    """Attacker objective, used on detached latents for attacker updates."""
    import torch.nn.functional as F

    outputs = attackers(latent)
    # Compare direction/content rather than unbounded residual scale. Without
    # normalization this term can be three orders of magnitude larger than the
    # token objective and the minimax game optimizes scale instead of privacy.
    reconstruction = F.layer_norm(outputs["reconstruction"],
                                  (outputs["reconstruction"].shape[-1],))
    target = F.layer_norm(hidden_targets.float(),
                          (hidden_targets.shape[-1],))
    return (F.cross_entropy(outputs["token"].flatten(0, 1),
                            token_labels.flatten(), ignore_index=-1)
            + F.cross_entropy(outputs["property"], property_labels)
            + F.mse_loss(reconstruction, target)), outputs


def defender_privacy_loss(attackers, latent, token_labels, property_labels,
                          hidden_targets, strength: float):
    """Same attacks with gradient reversal for a defender/encoder update."""
    return attacker_loss(attackers, gradient_reverse(latent, strength),
                         token_labels, property_labels, hidden_targets)[0]


def alternating_minimax_step(tln, ucn, attackers, hidden, target_hidden,
                             token_labels, property_labels,
                             attacker_optimizer, defender_optimizer,
                             nonce: str, strength: float = 0.25):
    """Run one attacker update followed by one task/defender update.

    The attacker learns from a detached release first. During the defender
    phase its weights are frozen while gradient reversal drives the private
    encoder away from token, property, and hidden-reconstruction leakage.
    This helper is intentionally optimizer-agnostic so production training can
    include the private model, latent cloud, and decoder in its optimizer.
    """
    import torch.nn.functional as F

    attacker_optimizer.zero_grad(set_to_none=True)
    latent, forward_meta = tln.encode(hidden, nonce)
    attack_objective, _ = attacker_loss(
        attackers, latent.detach(), token_labels, property_labels,
        hidden.detach())
    attack_objective.backward()
    attacker_optimizer.step()

    defender_optimizer.zero_grad(set_to_none=True)
    requires_grad = [parameter.requires_grad for parameter in attackers.parameters()]
    for parameter in attackers.parameters():
        parameter.requires_grad_(False)
    # The wire rotates in D-space only. UCN receives the current matrix to
    # unwrap/rewrap this request, but never the ratchet master. Therefore this
    # limits cross-request capture pooling; it does not hide the live latent
    # from a compromised cloud process.
    transform = tln.ratchet.rotate(device=latent.device, dtype=latent.dtype)
    wire_forward = tln.ratchet.apply(latent, transform)
    cloud_input = tln.ratchet.inverse(wire_forward, transform)
    cloud_latent = ucn(cloud_input)
    wire_return = tln.ratchet.apply(cloud_latent, transform)
    trusted_return = tln.ratchet.inverse(wire_return, transform)
    restored, return_meta = tln.decode(trusted_return, nonce,
                                         residual=hidden)
    task_objective = F.mse_loss(restored, target_hidden)
    privacy_objective = defender_privacy_loss(
        attackers, latent, token_labels, property_labels, hidden.detach(),
        strength)
    defender_objective = task_objective + privacy_objective
    defender_objective.backward()
    defender_optimizer.step()
    for parameter, enabled in zip(attackers.parameters(), requires_grad):
        parameter.requires_grad_(enabled)

    return {
        "attacker_loss": float(attack_objective.detach().item()),
        "task_loss": float(task_objective.detach().item()),
        "defender_privacy_loss": float(privacy_objective.detach().item()),
        "forward_dp": forward_meta,
        "return_dp": return_meta,
        "accountant": tln.dp.report(),
        "ratchet_epoch": tln.ratchet.epoch - 1,
        "wire_width": int(wire_forward.shape[-1]),
    }


def assert_ucn_latent_only(module, latent_dim: int, forbidden_hidden: int,
                             allowed_internal_width: int | None = None) -> None:
    """Fail closed if a serialized cloud parameter exposes an H-width seam.

    The hard privacy invariant is the first check: no parameter may carry the
    model hidden width H.  Linear layers are additionally capped at
    4*latent_dim — or at ``allowed_internal_width`` when the deployment
    declares a bigger internal latent-space width: an internal D-space width
    cannot ingest H tensors (the interface check below fixes the module I/O
    at D), but must still never equal H.
    """
    import torch
    from torch import nn

    width_cap = 4 * latent_dim
    if allowed_internal_width:
        width_cap = max(width_cap, allowed_internal_width)
    for name, value in module.state_dict().items():
        if forbidden_hidden in value.shape:
            raise RuntimeError(f"UCN parameter {name} exposes hidden width")
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            if child.in_features > width_cap or child.out_features > width_cap:
                raise RuntimeError(f"UCN layer {name} exceeds latent-native width")
            if forbidden_hidden in (child.in_features, child.out_features):
                raise RuntimeError(f"UCN layer {name} touches hidden width")
        if isinstance(child, nn.LayerNorm):
            if tuple(child.normalized_shape) != (latent_dim,):
                raise RuntimeError(f"UCN norm {name} is not latent-width")
    first_parameter = next(module.parameters())
    sample = torch.randn(2, 3, latent_dim, device=first_parameter.device,
                         dtype=first_parameter.dtype)
    output = module(sample)
    if output.shape != sample.shape:
        raise RuntimeError("UCN module did not preserve latent shape")
