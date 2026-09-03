#!/usr/bin/env python3
"""MINE-style mutual information estimator for the latent boundary.

Estimates I(released latent row ; token identity) with a Donsker-Varadhan
statistics network:  I >= E_joint[T] - log E_marginal[exp(T)].  The token
identity enters through a learned embedding inside the statistics network.

Used two ways by the runner:
  - estimator training (maximize the bound — the attacker's best MI view)
  - defender penalty (minimize the bound through the encoder, via
    gradient-reversal alternation)

CE-based MI bounds are vacuous at LLM vocabularies; MINE at D=64 is the
tighter estimator.
"""

from __future__ import annotations


def build_mine_stats(latent_dim: int, vocab_size: int, embed_dim: int = 64,
                     hidden: int = 256):
    """Statistics network T(z, x): latent row + token embedding -> scalar."""
    import torch
    from torch import nn

    class MineStatsNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embed = nn.Embedding(vocab_size, embed_dim)
            self.net = nn.Sequential(
                nn.Linear(latent_dim + embed_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, 1))

        def forward(self, z, token_ids):
            emb = self.token_embed(token_ids)
            return self.net(torch.cat([z.float(), emb], dim=-1))

    return MineStatsNet()


def mine_estimate(stats_net, z, tokens, marginal_shuffle=True):
    """Donsker-Varadhan lower bound estimate in nats.

    z: [..., D] latent rows; tokens: [...] token ids.  The marginal term
    uses a row-permuted pairing within the batch.
    """
    import torch

    joint = stats_net(z, tokens)
    if marginal_shuffle:
        perm = torch.randperm(tokens.numel(), device=tokens.device)
        shuffled = tokens.reshape(-1)[perm].reshape_as(tokens)
    else:
        shuffled = tokens.roll(1, dims=0)
    marginal = stats_net(z, shuffled)
    return float((joint.mean()
                  - torch.logsumexp(marginal.reshape(-1), dim=0)
                  + torch.log(torch.tensor(float(marginal.numel()),
                                           device=z.device))).item())


def mine_loss_for_training(stats_net, z, tokens):
    """Negative DV bound as a differentiable loss (maximize => minimize)."""
    import torch

    joint = stats_net(z, tokens)
    perm = torch.randperm(tokens.numel(), device=tokens.device)
    shuffled = tokens.reshape(-1)[perm].reshape_as(tokens)
    marginal = stats_net(z, shuffled)
    return -(joint.mean()
             - (torch.logsumexp(marginal.reshape(-1), dim=0)
                - torch.log(torch.tensor(float(marginal.numel()),
                                         device=z.device))))
