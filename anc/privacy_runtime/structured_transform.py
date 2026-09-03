"""Fast, deterministic orthogonal boundary transform.

A signed-permuted Walsh-Hadamard transform costing O(d log d) instead of a
dense d x d matrix multiply.  It is intentionally a *wire-capture*
mechanism, not encryption: a peer holding the seed can invert it, and its
structure must be re-evaluated by the attacker.
"""

from __future__ import annotations

import hashlib
import math

from .ratchet_v2 import epoch_key


DOMAIN = b"dtraining/structured-hadamard/v1\x00"


def _stream(master, epoch: int):
    key = epoch_key(master, epoch)
    counter = 0
    while True:
        yield from hashlib.sha256(DOMAIN + key + counter.to_bytes(8, "big")).digest()
        counter += 1


def _parameters(master, epoch: int, hidden_dim: int):
    if hidden_dim <= 0 or hidden_dim & (hidden_dim - 1):
        raise ValueError("structured_hadamard requires a power-of-two hidden dimension")
    source = _stream(master, epoch)
    signs = [1.0 if next(source) & 1 else -1.0 for _ in range(hidden_dim)]
    permutation = list(range(hidden_dim))
    # Deterministic Fisher-Yates. Rejection removes modulo bias.
    for i in range(hidden_dim - 1, 0, -1):
        modulus = 1 << 32
        limit = modulus - (modulus % (i + 1))
        value = int.from_bytes(bytes(next(source) for _ in range(4)), "big")
        while value >= limit:
            value = int.from_bytes(bytes(next(source) for _ in range(4)), "big")
        j = value % (i + 1)
        permutation[i], permutation[j] = permutation[j], permutation[i]
    return signs, permutation


def _fwht(tensor):
    import torch

    out = tensor.contiguous()
    width = out.shape[-1]
    step = 1
    while step < width:
        view = out.reshape(*out.shape[:-1], -1, step * 2)
        left = view[..., :step]
        right = view[..., step:]
        out = torch.cat((left + right, left - right), dim=-1).reshape_as(out)
        step *= 2
    return out / math.sqrt(width)


class StructuredHadamard:
    """Signed-permuted orthogonal transform with exact inverse wiring."""

    mode = "structured_hadamard_v1"

    def __init__(self, master, epoch: int, hidden_dim: int, device=None,
                 dtype=None):
        import torch

        signs, permutation = _parameters(master, epoch, hidden_dim)
        self.signs = torch.tensor(signs, device=device, dtype=dtype or torch.float32)
        self.permutation = torch.tensor(permutation, device=device, dtype=torch.long)
        self.inverse_permutation = torch.argsort(self.permutation)

    def apply(self, tensor):
        work = tensor.index_select(-1, self.permutation) * self.signs
        return _fwht(work)

    def inverse(self, tensor):
        work = _fwht(tensor) * self.signs
        return work.index_select(-1, self.inverse_permutation)
