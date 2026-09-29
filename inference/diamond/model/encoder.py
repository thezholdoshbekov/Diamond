import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms_norm import RMSNorm
from .rope import RotaryEmbedding, apply_rope


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 qk_norm: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        # QK-Norm (Henry 2020; ViT-22B Dehghani 2023; Wortsman 2023): RMSNorm on
        # per-head Q and K before attention → bounds attention-logit magnitude,
        # the dominant source of grad spikes in deep transformers.
        self.q_norm = RMSNorm(self.d_head) if qk_norm else None
        self.k_norm = RMSNorm(self.d_head) if qk_norm else None

    def forward(self, x: torch.Tensor, rope_cache: tuple[torch.Tensor, torch.Tensor],
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = rope_cache
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :].expand(B, self.n_heads, T, T)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1,
                 layer_scale_init: float = 0.1, max_seq_len: int = 4096,
                 rope_learnable: bool = True, qk_norm: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, qk_norm=qk_norm)
        self.norm2 = RMSNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)
        self.drop = nn.Dropout(dropout)
        # LayerScale (CaiT, Touvron 2021): trainable per-channel γ on each
        # residual branch. Each layer learns its own effective contribution,
        # so no layer is forced to inflate weight norms to "compete" for
        # representational burden. Without it the encoder's first layer runs
        # away: its weights grew ~7× while every other layer grew only ~1.4×.
        self.ls1 = nn.Parameter(torch.full((d_model,), layer_scale_init))
        self.ls2 = nn.Parameter(torch.full((d_model,), layer_scale_init))
        # Per-layer RoPE (optionally with learnable α scaling — Kani TTS 2).
        # When learnable, each layer picks its own position-zoom; init at
        # α≈1.05 means standard RoPE behavior as the starting point.
        self.rope = RotaryEmbedding(
            head_dim=d_model // n_heads,
            max_seq_len=max_seq_len,
            learnable=rope_learnable,
        )

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        T = x.size(1)
        rope_cache = self.rope(T)
        x = x + self.ls1 * self.drop(self.attn(self.norm1(x), rope_cache, key_padding_mask))
        x = x + self.ls2 * self.drop(self.ff(self.norm2(x)))
        return x


class MelEncoder(nn.Module):
    def __init__(self, n_mels: int, d_model: int, n_heads: int, n_layers: int,
                 ff_dim: int, dropout: float = 0.1, max_seq_len: int = 4096,
                 layer_scale_init: float = 0.1, rope_learnable: bool = True,
                 qk_norm: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1)
        # Bound the input distribution that hits L0. GELU(conv2) is unbounded;
        # without norm_in, L0's pre-norm RMSNorm has to absorb varying input
        # amplitude through its own γ, which couples to weight growth.
        self.norm_in = RMSNorm(d_model)

        # Each layer owns its own RoPE module (so the learnable α — when
        # enabled — can specialize per layer).
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, ff_dim, dropout,
                         layer_scale_init=layer_scale_init,
                         max_seq_len=max_seq_len,
                         rope_learnable=rope_learnable,
                         qk_norm=qk_norm)
            for _ in range(n_layers)
        ])
        self.norm_out = RMSNorm(d_model)

    def forward(self, mel: torch.Tensor,
                pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        mel:      [B, T_mel, n_mels]   values in [-1, 1] from mel-extractor
        pad_mask: [B, T_mel] — True at valid positions, False at padding
        returns:  [B, T_mel, d_model]
        """
        x = mel.transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)
        x = self.norm_in(x)

        for layer in self.layers:
            x = layer(x, key_padding_mask=pad_mask)

        return self.norm_out(x)
