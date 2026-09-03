"""Finite-field additive sharing for two non-colluding workers.

Represents and linearly transforms quantized tensors.  Nonlinear operations
are deliberately refused: transformer nonlinearities need an audited MPC
protocol and must never be implemented by reconstructing on an untrusted
worker.
"""

from __future__ import annotations

import secrets


PRIME = 2 ** 61 - 1


class TwoServerShareCodec:
    def __init__(self, scale: int = 1 << 20, prime: int = PRIME):
        if scale <= 0 or prime <= 2:
            raise ValueError("invalid fixed-point field")
        self.scale = int(scale)
        self.prime = int(prime)

    def _quantize(self, tensor):
        import torch
        q = torch.round(tensor.double() * self.scale).to(torch.int64)
        if bool((q.abs() >= self.prime // 4).any()):
            raise OverflowError("fixed-point value exceeds safe field range")
        return q.remainder(self.prime)

    def share(self, tensor):
        import torch
        q = self._quantize(tensor)
        # Generate field masks independently of torch's reproducible PRNG.
        raw = [secrets.randbelow(self.prime) for _ in range(q.numel())]
        a = torch.tensor(raw, dtype=torch.int64).reshape_as(q)
        b = (q - a).remainder(self.prime)
        return a, b

    def reconstruct(self, share_a, share_b):
        q = (share_a + share_b).remainder(self.prime)
        signed = q.clone()
        signed[q > self.prime // 2] -= self.prime
        return signed.double() / self.scale

    def linear(self, share, weight_integer):
        """Apply a public integer linear map independently to one share.

        Python integers are used deliberately. Native int64 matmul can overflow
        before the field reduction and is therefore not a correct finite-field
        implementation. This reference path is slow but exact; an optimized
        kernel must use widened products or chunked modular accumulation.
        """
        import torch

        if share.ndim < 2 or weight_integer.ndim != 2:
            raise ValueError("share must be [..., in] and weight must be [in, out]")
        if share.shape[-1] != weight_integer.shape[0]:
            raise ValueError("linear dimensions do not match")
        flat = share.reshape(-1, share.shape[-1]).cpu()
        weight = weight_integer.to(dtype=torch.int64, device="cpu")
        rows = []
        for i in range(flat.shape[0]):
            row = []
            for j in range(weight.shape[1]):
                acc = 0
                for k in range(flat.shape[1]):
                    acc = (acc + int(flat[i, k]) * int(weight[k, j])) % self.prime
                row.append(acc)
            rows.append(row)
        out = torch.tensor(rows, dtype=torch.int64)
        return out.reshape(*share.shape[:-1], weight.shape[1]).to(share.device)

    @staticmethod
    def nonlinear(*_args, **_kwargs):
        raise RuntimeError(
            "nonlinear operations require audited MPC; reconstruction on an "
            "untrusted node is forbidden")
