import os

import torch
import torch.nn as nn

# Resume-time safety valve (off by default). Export RMSNORM_GAMMA_CLAMP=2.0 to
# hard-cap the normalization gain |γ| → defuses any single layer trying to
# dominate via a runaway γ (the failure mode is a late decoder layer inflating
# its read-out until gradients spike).
# Read once at import; no-op when unset so it can't change default behavior.
_GAMMA_CLAMP = os.environ.get("RMSNORM_GAMMA_CLAMP")
_GAMMA_CLAMP = float(_GAMMA_CLAMP) if _GAMMA_CLAMP else None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the norm in fp32 — bf16 underflows on small activations and
        # overflows on large ones, both of which destabilise training.
        dtype = x.dtype
        x_fp32 = x.float()
        norm = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        w = self.weight if _GAMMA_CLAMP is None else self.weight.clamp(-_GAMMA_CLAMP, _GAMMA_CLAMP)
        return (norm.to(dtype)) * w
