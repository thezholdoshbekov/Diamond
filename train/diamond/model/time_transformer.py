"""TimeTransformer — autoregressive token decoder over the time axis.

It supersedes an earlier delayed-pattern token decoder:
  • Input is a single frame embedding per step (sum of 9 cb embeddings),
    NOT a delayed/right-shifted 9-codebook pattern of length T+8. Sequence
    stays T_tok long — no expansion by delay. cb causality is handled
    inside the depth-transformer, not by the delayed pattern.
  • Output is the time-T hidden state per frame [B, T_tok, d_model] — ONE
    vector per frame. The 9 output heads are removed; they live in the
    depth-transformer, where each head reads off a slot whose hidden state
    was specifically computed for its codebook (causal-in-depth).
  • Causal self-attn + cross-attn(encoder) + FFN per layer. RoPE, RMSNorm,
    GELU, LayerScale and QK-Norm are all kept.

Autoregression in time:
  decoder_input[:, 0, :] = BOS (one embedding, shared sentinel)
  decoder_input[:, t, :] = embed(target_tokens[:, t-1, :])   for t>=1
  hidden_t[:, t, :] is the time-T's frame-t conditioning that the depth-T
                    will turn into 9 codebook predictions for frame t.

Vocabulary:
  Per-cb embedding table size = vocab_size + 1 ; the extra index is BOS,
  shared across all 9 cb embeddings. PAD is not used here — sequences are
  fixed-length (T_tok) and the dataset's mel_pad_mask handles batch padding.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms_norm import RMSNorm
from .rope import RotaryEmbedding, apply_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 qk_norm: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.q_norm = RMSNorm(self.d_head) if qk_norm else None
        self.k_norm = RMSNorm(self.d_head) if qk_norm else None

    def forward(self, x: torch.Tensor, rope_cache) -> torch.Tensor:
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

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)

    def forward_step(self, x_new: torch.Tensor, cos_p: torch.Tensor,
                     sin_p: torch.Tensor, cache: dict) -> torch.Tensor:
        """Incremental KV-cache step. x_new: [B, 1, D] at position `pos`
        (rope row passed in cos_p/sin_p [1, d_head/2]). cache holds growing
        k/v [B, H, T_prev, d_head]. Single query attends all cached keys (all
        ≤ pos) → causal by construction, no mask needed."""
        B, _, D = x_new.shape
        qkv = self.qkv(x_new).reshape(B, 1, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))   # [B,H,1,dh]
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rope(q, cos_p, sin_p)
        k = apply_rope(k, cos_p, sin_p)
        if cache.get("k") is not None:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        cache["k"], cache["v"] = k, v
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(B, 1, D)
        return self.out(out)


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 qk_norm: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.q_norm = RMSNorm(self.d_head) if qk_norm else None
        self.k_norm = RMSNorm(self.d_head) if qk_norm else None

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor,
                encoder_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T_q, D = x.shape
        T_kv = encoder_out.size(1)

        q = self.q_proj(x).reshape(B, T_q, self.n_heads, self.d_head).transpose(1, 2)
        kv = self.kv_proj(encoder_out).reshape(B, T_kv, 2, self.n_heads, self.d_head)
        k, v = kv.unbind(dim=2)
        k, v = (t.transpose(1, 2) for t in (k, v))

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn_mask = None
        if encoder_pad_mask is not None:
            attn_mask = encoder_pad_mask[:, None, None, :].expand(B, self.n_heads, T_q, T_kv)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(B, T_q, D)
        return self.out(out)

    def forward_step(self, x_new: torch.Tensor, cache: dict,
                     encoder_out: torch.Tensor,
                     encoder_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Incremental cross-attn. Encoder K/V are identical every step → cache
        them once (on first call). x_new: [B, 1, D]."""
        B, _, D = x_new.shape
        if cache.get("k") is None:
            T_kv = encoder_out.size(1)
            kv = self.kv_proj(encoder_out).reshape(B, T_kv, 2, self.n_heads, self.d_head)
            k, v = kv.unbind(dim=2)
            k, v = (t.transpose(1, 2) for t in (k, v))
            if self.k_norm is not None:
                k = self.k_norm(k)
            cache["k"], cache["v"] = k, v
            cache["mask"] = (encoder_pad_mask[:, None, None, :].expand(B, self.n_heads, 1, T_kv)
                             if encoder_pad_mask is not None else None)
        q = self.q_proj(x_new).reshape(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
        out = F.scaled_dot_product_attention(q, cache["k"], cache["v"],
                                             attn_mask=cache["mask"], is_causal=False)
        out = out.transpose(1, 2).reshape(B, 1, D)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class TimeLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1,
                 max_seq_len: int = 4096, rope_learnable: bool = False,
                 qk_norm: bool = False, layer_scale_init: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = CausalSelfAttention(d_model, n_heads, dropout, qk_norm=qk_norm)
        self.norm2 = RMSNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout, qk_norm=qk_norm)
        self.norm3 = RMSNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)
        self.drop = nn.Dropout(dropout)
        self.ls1 = nn.Parameter(torch.full((d_model,), layer_scale_init))
        self.ls2 = nn.Parameter(torch.full((d_model,), layer_scale_init))
        self.ls3 = nn.Parameter(torch.full((d_model,), layer_scale_init))
        self.rope = RotaryEmbedding(
            head_dim=d_model // n_heads,
            max_seq_len=max_seq_len,
            learnable=rope_learnable,
        )

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor,
                encoder_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        T = x.size(1)
        rope_cache = self.rope(T)
        x = x + self.ls1 * self.drop(self.self_attn(self.norm1(x), rope_cache))
        x = x + self.ls2 * self.drop(self.cross_attn(self.norm2(x), encoder_out, encoder_pad_mask))
        x = x + self.ls3 * self.drop(self.ff(self.norm3(x)))
        return x

    def forward_step(self, x_new: torch.Tensor, pos: int, self_cache: dict,
                     cross_cache: dict, encoder_out: torch.Tensor,
                     encoder_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Incremental (KV-cache) variant of forward for one new frame at
        position `pos`. Eval-only: dropout is identity. Mirrors forward()
        exactly (same norms, LayerScale, residuals)."""
        cos_p, sin_p = self.rope.at(pos)
        x = x_new + self.ls1 * self.self_attn.forward_step(self.norm1(x_new), cos_p, sin_p, self_cache)
        x = x + self.ls2 * self.cross_attn.forward_step(self.norm2(x), cross_cache, encoder_out, encoder_pad_mask)
        x = x + self.ls3 * self.ff(self.norm3(x))
        return x


class TimeTransformer(nn.Module):
    """Causal AR over time frames; outputs a hidden state per frame for the
    depth-transformer to expand into 9 codebook predictions.

    Args:
        d_model:         hidden dim
        n_heads:         attention heads
        n_layers:        stack depth
        ff_dim:          FFN hidden
        n_codebooks:     K = 9 for DAC 44.1 kHz
        vocab_size:      RVQ codebook size = 1024
        bos_token_id:    sentinel index in the embedding table (= vocab_size,
                         giving table size vocab_size + 1)
        ...
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int, ff_dim: int,
                 n_codebooks: int = 9, vocab_size: int = 1024,
                 bos_token_id: int = 1024, dropout: float = 0.1,
                 max_seq_len: int = 4096, rope_learnable: bool = False,
                 qk_norm: bool = False, layer_scale_init: float = 0.1):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.embed_size = vocab_size + 1  # +1 for BOS
        self.d_model = d_model

        # One embedding per cb; sum across K gives the per-frame input. This is
        # the same scheme the earlier delayed-pattern decoder used, so the time-T
        # input distribution is unchanged — minus the delayed pattern itself.
        self.embeddings = nn.ModuleList([
            nn.Embedding(self.embed_size, d_model)
            for _ in range(n_codebooks)
        ])

        self.layers = nn.ModuleList([
            TimeLayer(d_model, n_heads, ff_dim, dropout,
                      max_seq_len=max_seq_len,
                      rope_learnable=rope_learnable,
                      qk_norm=qk_norm,
                      layer_scale_init=layer_scale_init)
            for _ in range(n_layers)
        ])
        self.norm_out = RMSNorm(d_model)

    def _build_input(self, target_tokens: torch.Tensor) -> torch.Tensor:
        """Right-shift by one frame; prepend BOS frame.

        target_tokens: [B, T_tok, K] — ground-truth RVQ codes
        Returns:       [B, T_tok, d_model] decoder input (frame t sees frames 0..t-1)
        """
        B, T_tok, K = target_tokens.shape
        # BOS frame: all-BOS tokens, sum embeddings.
        bos_frame = torch.full((B, 1, K), self.bos_token_id,
                               dtype=target_tokens.dtype, device=target_tokens.device)
        # Shifted: BOS + target_tokens[:, :-1, :] → length T_tok
        shifted = torch.cat([bos_frame, target_tokens[:, :-1, :]], dim=1)
        # Sum embeddings across K
        x = sum(self.embeddings[k](shifted[..., k]) for k in range(K))
        return x

    def forward(self, target_tokens: torch.Tensor, encoder_out: torch.Tensor,
                encoder_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Teacher-forced training pass.

        target_tokens: [B, T_tok, K]
        encoder_out:   [B, T_mel, d_model]
        encoder_pad_mask: [B, T_mel]

        Returns hidden: [B, T_tok, d_model]
        """
        x = self._build_input(target_tokens)
        for layer in self.layers:
            x = layer(x, encoder_out, encoder_pad_mask)
        return self.norm_out(x)

    def step(self, prev_tokens: torch.Tensor, encoder_out: torch.Tensor,
             encoder_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """One-step inference helper. Caller passes the running history of
        already-generated tokens (BOS-prefixed) and gets back the hidden state
        for ONLY the latest frame.

        prev_tokens: [B, T_so_far, K] — INCLUDING the BOS frame at index 0;
                                        index 1..T_so_far-1 are predicted frames
        Returns hidden_last: [B, d_model] — to be fed into depth-T for the next frame
        """
        B, T_so_far, K = prev_tokens.shape
        x = sum(self.embeddings[k](prev_tokens[..., k]) for k in range(K))
        for layer in self.layers:
            x = layer(x, encoder_out, encoder_pad_mask)
        x = self.norm_out(x)
        return x[:, -1, :]

    def init_caches(self) -> list:
        """Per-layer KV caches for incremental decoding: self-attn k/v grow
        per step, cross-attn k/v are filled once from encoder_out."""
        return [{"self": {"k": None, "v": None}, "cross": {}} for _ in self.layers]

    def step_cached(self, frame_tokens: torch.Tensor, pos: int, caches: list,
                    encoder_out: torch.Tensor,
                    encoder_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """O(1)-per-step incremental decode (vs step()'s O(T) full recompute).

        frame_tokens: [B, 1, K] — the NEW input frame (BOS frame at pos 0, then
                      the codes generated at the previous step).
        Returns hidden_last: [B, d_model] for this frame → feed to depth-T.

        Equivalence to step(): at step i over history [BOS, fc0..fc_{i-1}],
        step() returns the transformer output at the last position; here the
        cached k/v for past frames equal what the full pass would compute at
        each layer (standard causal KV-cache identity), so output matches.
        """
        K = self.n_codebooks
        x = sum(self.embeddings[k](frame_tokens[..., k]) for k in range(K))  # [B,1,D]
        for L, layer in enumerate(self.layers):
            x = layer.forward_step(x, pos, caches[L]["self"], caches[L]["cross"],
                                   encoder_out, encoder_pad_mask)
        x = self.norm_out(x)
        return x[:, -1, :]
