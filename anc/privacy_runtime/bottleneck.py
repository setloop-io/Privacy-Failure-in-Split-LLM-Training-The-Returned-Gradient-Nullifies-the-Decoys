"""Learned irreversible boundary sanitizer and adversarial training tools."""

from __future__ import annotations


def build_privacy_bottleneck(hidden_dim: int, bottleneck_dim: int,
                             adversary_classes: int = 0):
    """Construct trusted encoder, cloud adapter, and optional adversary.

    The dimensionality reduction is intentionally non-invertible.  The cloud
    adapter is not secret and must be assumed known to the attacker.
    """
    import torch
    from torch import nn

    if not 0 < bottleneck_dim < hidden_dim:
        raise ValueError("bottleneck_dim must be positive and smaller than hidden_dim")

    class TrustedEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(hidden_dim)
            self.project = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
            self.gate = nn.Linear(hidden_dim, bottleneck_dim)

        def forward(self, hidden):
            x = self.norm(hidden)
            return torch.tanh(self.project(x)) * torch.sigmoid(self.gate(x))

    class CloudAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.expand = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, protected):
            return self.norm(self.expand(protected))

    class TokenAdversary(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(bottleneck_dim),
                nn.Linear(bottleneck_dim, bottleneck_dim), nn.GELU(),
                nn.Linear(bottleneck_dim, adversary_classes),
            )

        def forward(self, protected):
            return self.net(protected)

    adversary = TokenAdversary() if adversary_classes > 0 else None
    return TrustedEncoder(), CloudAdapter(), adversary


def gradient_reverse(value, strength: float = 1.0):
    import torch

    class Reverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.strength = float(strength)
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad):
            return -ctx.strength * grad

    return Reverse.apply(value)


def adversarial_privacy_loss(adversary, protected, private_labels,
                             strength: float = 1.0):
    """Train the adversary to predict labels while reversing encoder grads."""
    import torch.nn.functional as F

    if adversary is None:
        raise ValueError("an adversary module is required")
    logits = adversary(gradient_reverse(protected, strength))
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           private_labels.reshape(-1)), logits
