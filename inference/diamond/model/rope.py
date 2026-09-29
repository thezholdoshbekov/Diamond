"""Rotary Positional Embedding (RoPE), with optional learnable per-layer α.

Standard RoPE (Su et al. 2021): θ_i = base^(-2i/d), angles = pos · θ_i.
Single fixed frequency ladder shared by all layers.

Learnable variant (Kani TTS 2 style; "Optimizing the Learnable RoPE Theta
Parameter in Transformers", IEEE Xplore doc. 11084811): θ_i^(l) = α^(l) · base^(-2i/d) where
α is a sigmoid-bounded scalar parameter per layer. Each layer learns its
own "position zoom" — early layers may pick fast rotation (fine local
detail), later layers may pick slow rotation (longer context). Init at
α ≈ 1.05 (≈ standard RoPE) so the worst case is identity to baseline.

Optimizer note: α params should go in their own group with higher LR
(×1.5 in Kani's setup) — they're scalars with tiny gradients otherwise.
"""

import torch
import torch.nn as nn


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to query or key tensor.

    x:   [B, H, T, D] where D is head_dim (must be even)
    cos: [T, D/2]
    sin: [T, D/2]
    """
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated


class RotaryEmbedding(nn.Module):
    """RoPE module owned per-layer. With learnable=True, exposes one scalar
    α as a trainable parameter; the position angle schedule becomes
    angles[t, i] = α · t · base^(-2i/d).

    Cost: recomputes cos/sin per forward (one element-wise scale + two trig
    ops over [T, head_dim/2]). Negligible vs attention compute.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096,
                 base: float = 10000.0, learnable: bool = True):
        super().__init__()
        self.learnable = learnable
        half = head_dim // 2
        freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        # angles_base[t, i] = t · base^(-2i/d) — the standard-RoPE schedule
        # before any α scaling. Held as buffer so it travels with .to(device).
        angles_base = torch.outer(positions, freqs)
        self.register_buffer("angles_base", angles_base, persistent=False)
        if learnable:
            # sigmoid(0) = 0.5 → α = 0.1 + 1.9 * 0.5 = 1.05 (≈ standard RoPE)
            # so the init is effectively a no-op vs baseline.
            self.alpha_raw = nn.Parameter(torch.zeros(1))

    def alpha_value(self) -> torch.Tensor:
        """Current α in [0.1, 2.0]. Returns scalar tensor (learnable) or 1.0 const."""
        if self.learnable:
            return 0.1 + 1.9 * torch.sigmoid(self.alpha_raw)
        return torch.tensor(1.0, device=self.angles_base.device)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        base_angles = self.angles_base[:seq_len]
        if self.learnable:
            angles = self.alpha_value() * base_angles
        else:
            angles = base_angles
        return angles.cos(), angles.sin()

    def at(self, pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin for a SINGLE position (KV-cache incremental decoding).
        Returns [1, head_dim/2] tensors. Same schedule as forward(), just one row."""
        a = self.angles_base[pos:pos + 1]
        if self.learnable:
            a = self.alpha_value() * a
        return a.cos(), a.sin()
