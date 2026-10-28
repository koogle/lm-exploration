"""Local value features plus a zero-initialized, signed mixture of learned EMAs."""

import torch
from torch import nn
from torch.nn import functional as F


class ValueTimescales(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor([8.0, 32.0, 128.0, 512.0]).log())
        self.mix = nn.Parameter(torch.zeros(4, width))
        self.register_buffer("theta", torch.zeros(4))
        self.use_kernel = True

    def reference(self, u, positions):
        rho = torch.exp(-torch.exp(-self.log_tau))
        state = u.new_zeros(u.size(0), 4, u.size(-1), dtype=torch.float32)
        outputs = []
        for t in range(u.size(1)):
            state = state * (positions[:, t] > 0)[:, None, None]
            state = rho[None, :, None] * state + (1 - rho)[None, :, None] * u[:, t, None].float()
            outputs.append((state * self.mix).sum(1))
        return torch.stack(outputs, 1)

    def forward(self, u, weight, positions):
        # Identical accumulation order to short_convolution, before its SiLU.
        local = u * weight[:, 0, -1]
        for shift in range(1, weight.size(-1)):
            previous = F.pad(u, (0, 0, shift, 0))[:, : u.size(1)]
            previous = previous.masked_fill((positions < shift)[..., None], 0)
            local = local + previous * weight[:, 0, -1 - shift]
        if u.is_cuda and self.use_kernel:
            from .kernels.rotating import rotating_memory

            tail = rotating_memory(u.float(), positions, self.log_tau, self.theta, self.mix, False)
        else:
            tail = self.reference(u, positions)
        return F.silu(local + tail).to(u.dtype)
