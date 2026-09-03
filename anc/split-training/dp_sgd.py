"""Auditable DP-SGD helpers for the private local layers in E-R9."""

import math
import secrets


class LocalDPSGD:
    """Clip one-example local gradients, then noise their average."""

    def __init__(self, params, max_grad_norm, noise_multiplier, accumulation,
                 device):
        if max_grad_norm <= 0 or noise_multiplier <= 0 or accumulation <= 0:
            raise ValueError("DP-SGD requires positive C, sigma, and accumulation")
        import torch
        self.params = list(params)
        self.max_grad_norm = float(max_grad_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.accumulation = int(accumulation)
        self.buffers = [None for _ in self.params]
        self.examples = 0
        self.generator = torch.Generator(device=device)
        # Do not derive privacy noise from the reproducible experiment seed.
        self.generator.manual_seed(secrets.randbits(63))

    def clip_and_accumulate(self):
        import torch
        device = self.params[0].device if self.params else "cpu"
        squared = torch.zeros((), dtype=torch.float64, device=device)
        for p in self.params:
            if p.grad is not None:
                squared += p.grad.detach().double().pow(2).sum()
        norm = squared.sqrt()
        scale = min(1.0, self.max_grad_norm / (float(norm) + 1e-12))
        for i, p in enumerate(self.params):
            if p.grad is not None:
                grad = p.grad.detach() * scale
                if self.buffers[i] is None:
                    self.buffers[i] = torch.zeros_like(grad)
                self.buffers[i].add_(grad)
            p.grad = None
        self.examples += 1
        return float(norm), scale

    def materialize_noisy_average(self):
        import torch
        if self.examples != self.accumulation:
            raise RuntimeError(f"expected {self.accumulation} examples, got {self.examples}")
        std = self.noise_multiplier * self.max_grad_norm / self.accumulation
        for p, buf in zip(self.params, self.buffers):
            if buf is not None:
                noise = torch.randn(buf.shape, dtype=buf.dtype,
                                    device=buf.device,
                                    generator=self.generator) * std
                p.grad = buf.div(self.accumulation).add_(noise)
        self.buffers = [None for _ in self.params]
        self.examples = 0


def conservative_zcdp_epsilon(steps, noise_multiplier, delta):
    """Replace-one zCDP bound, conservatively ignoring amplification."""
    if steps < 0 or noise_multiplier <= 0 or not 0 < delta < 1:
        raise ValueError("invalid accountant inputs")
    # replace-one sensitivity of the average is 2C/G; noise std is sigma*C/G, hence rho_step=2/sigma^2
    rho = 2.0 * steps / noise_multiplier ** 2
    epsilon = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
    return {"method": "zcdp_no_subsampling_replace_one_upper_bound", "rho": rho,
            "epsilon": epsilon, "delta": delta}
