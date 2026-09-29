"""DepthTransformer — RQ-Transformer (Lee 2022) / Moshi (Kyutai 2024) depth axis.

Replaces the earlier read-out (nine parallel heads reading one shared 512-d
hidden vector) with a tiny causal transformer that walks the 9 RVQ codebooks
sequentially inside each time frame. Each codebook gets a dedicated readout
pathway — this fixes the decoder weight-norm runaway that LayerScale and
z-loss could only contain, never cure.

Per frame t the depth-T processes a 9-token sequence:

    position 0 : proj(hidden_t)             → cb_0 head → cb_0 logits
    position 1 : proj(hidden_t), cb_0_emb   → cb_1 head → cb_1 logits
    ...
    position 8 : proj(hidden_t), cb_0..7    → cb_8 head → cb_8 logits

Vectorized over (B, T_tok): reshape to [B*T_tok, 9, depth_d] before the
layers — attention is over seq_len=9 only, so memory/compute is cheap.

Design choices:
  • Shared layer weights across the 9 depth steps (a per-step parameterization,
    as in Moshi, is a possible ablation — we start simpler).
  • Codebook-id positional embedding signals which cb each slot predicts.
    No RoPE on depth — seq=9 is too short to need rotation; the cb-id embed
    plus causal mask is enough positional signal.
  • No cross-attention here — the encoder context is already baked into
    hidden_t by the time-transformer.
  • Separate input embedding per cb_0..cb_7 (8 embeddings, only those are
    used as inputs — cb_8 is never an input to anything). Separate output
    head per cb_0..cb_8 (9 heads). Each cb has its own RVQ vocab space.
  • LayerScale + QK-Norm.
  • SDPA backend (flash where available).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms_norm import RMSNorm


class _CausalSelfAttnNoRoPE(nn.Module):
    """Plain causal self-attention. No RoPE — depth seq is 9 tokens, cb-id
    embedding at input is the positional signal."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 qk_norm: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.q_norm = RMSNorm(self.d_head) if qk_norm else None
        self.k_norm = RMSNorm(self.d_head) if qk_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class _FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class DepthLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int,
                 dropout: float = 0.1, qk_norm: bool = True,
                 layer_scale_init: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = _CausalSelfAttnNoRoPE(d_model, n_heads, dropout, qk_norm=qk_norm)
        self.norm2 = RMSNorm(d_model)
        self.ff = _FeedForward(d_model, ff_dim, dropout)
        self.drop = nn.Dropout(dropout)
        self.ls1 = nn.Parameter(torch.full((d_model,), layer_scale_init))
        self.ls2 = nn.Parameter(torch.full((d_model,), layer_scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1 * self.drop(self.self_attn(self.norm1(x)))
        x = x + self.ls2 * self.drop(self.ff(self.norm2(x)))
        return x


class DepthTransformer(nn.Module):
    """Walks the codebook axis inside one frame.

    Args:
        time_d_model:    d of the time-transformer hidden state (input to depth-T)
        d_model:         depth-T hidden dim (smaller than time-T)
        n_heads:         attention heads for depth-T
        n_layers:        number of DepthLayer blocks
        ff_dim:          FFN hidden dim
        n_codebooks:     K (= 9 for DAC 44.1 kHz)
        vocab_size:      RVQ codebook size (= 1024 for DAC)
        dropout:         dropout for attn + FFN
        qk_norm:         QK-Norm on attention
        layer_scale_init: CaiT LayerScale γ init
    """

    def __init__(self, time_d_model: int, d_model: int, n_heads: int,
                 n_layers: int, ff_dim: int, n_codebooks: int = 9,
                 vocab_size: int = 1024, dropout: float = 0.1,
                 qk_norm: bool = True, layer_scale_init: float = 0.1):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Project time-T hidden into depth-T space (slot 0 of every depth seq).
        self.input_proj = nn.Linear(time_d_model, d_model)

        # Input embeddings for cb_0..cb_{K-2}: those tokens become depth inputs
        # at slots 1..K-1. cb_{K-1} is only ever a target, never an input.
        # Table size = vocab_size + 1: the extra (top) row mirrors the time-T
        # embedding convention and reserves a sentinel slot. It is never indexed
        # at train or inference (codes/argmax are 0..vocab_size-1), but keeping
        # it lets the released [vocab_size+1, d] tables load directly.
        self.cb_input_embed = nn.ModuleList([
            nn.Embedding(vocab_size + 1, d_model) for _ in range(n_codebooks - 1)
        ])

        # Codebook-id positional embedding — tells each slot which cb it must
        # predict. Slot k's job is to emit cb_k logits. Without this slots are
        # interchangeable, the model can't tell which one to specialize.
        self.cb_pos_embed = nn.Parameter(torch.zeros(n_codebooks, d_model))
        nn.init.normal_(self.cb_pos_embed, std=0.02)

        self.layers = nn.ModuleList([
            DepthLayer(d_model, n_heads, ff_dim, dropout,
                       qk_norm=qk_norm, layer_scale_init=layer_scale_init)
            for _ in range(n_layers)
        ])
        self.norm_out = RMSNorm(d_model)

        # Output heads — each cb has its own RVQ vocab space, so a dedicated head
        # is the right inductive bias. This is NOT the read-out inflation the
        # shared-hidden design suffered from: each slot's hidden state was
        # computed specifically for cb_k by the causal-in-depth attention, so
        # per-cb specialization sits at the right altitude here.
        self.cb_output_head = nn.ModuleList([
            nn.Linear(d_model, vocab_size, bias=False)
            for _ in range(n_codebooks)
        ])

    def _build_depth_seq(self, hidden_t: torch.Tensor,
                         gt_codes: torch.Tensor | None) -> torch.Tensor:
        """Build the [B, T_tok, K, d_model] depth sequence.

        Slot 0 always = projected time-T hidden.
        Slot k (k>=1) = embedding of cb_{k-1} from gt_codes.
        If gt_codes is None (generate mode), caller fills slots 1..K-1 iteratively.
        """
        B, T_tok, _ = hidden_t.shape
        K = self.n_codebooks

        h0 = self.input_proj(hidden_t)  # [B, T_tok, d_model]

        if gt_codes is None:
            seq = torch.zeros(B, T_tok, K, self.d_model, device=hidden_t.device,
                              dtype=hidden_t.dtype)
            seq[:, :, 0, :] = h0
            return seq

        # Stack cb_0..cb_{K-2} embeddings — these become slots 1..K-1
        cb_in = torch.stack([
            self.cb_input_embed[k](gt_codes[:, :, k])
            for k in range(K - 1)
        ], dim=2)  # [B, T_tok, K-1, d_model]

        seq = torch.cat([h0.unsqueeze(2), cb_in], dim=2)  # [B, T_tok, K, d_model]
        return seq

    def _run_layers(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: [B, T_tok, K, d_model] → [B, T_tok, K, d_model]"""
        B, T_tok, K, D = seq.shape

        # Add codebook-id position embedding (broadcast over B, T_tok).
        seq = seq + self.cb_pos_embed.view(1, 1, K, D)

        # Flatten the (B, T_tok) axes — depth-T attention is over K=9 only.
        seq = seq.reshape(B * T_tok, K, D)
        for layer in self.layers:
            seq = layer(seq)
        seq = self.norm_out(seq)
        return seq.reshape(B, T_tok, K, D)

    def _heads(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: [B, T_tok, K, d_model] → logits [B, T_tok, K, vocab_size]"""
        return torch.stack([
            self.cb_output_head[k](seq[:, :, k, :])
            for k in range(self.n_codebooks)
        ], dim=2)

    def forward(self, hidden_t: torch.Tensor, gt_codes: torch.Tensor) -> torch.Tensor:
        """Teacher-forced training pass.

        hidden_t: [B, T_tok, time_d_model] — output of time-transformer
        gt_codes: [B, T_tok, K]            — ground-truth RVQ codes

        Returns logits: [B, T_tok, K, vocab_size]
        """
        seq = self._build_depth_seq(hidden_t, gt_codes)
        seq = self._run_layers(seq)
        return self._heads(seq)

    def forward_scheduled(self, hidden_t: torch.Tensor, gt_codes: torch.Tensor,
                          ss_prob: float) -> torch.Tensor:
        """Depth-axis scheduled sampling — training pass that addresses
        exposure bias in the DEPTH dimension. BOUNDED to ≤K steps within a
        frame, so a bad prediction can poison at most cb_{k+1..K-1} of the SAME
        frame (≤8 downstream) — unlike time-axis scheduled sampling, which fed
        errors into an unbounded AR chain and produced a gradient-spike storm
        the loss never recovered from. Here the next frame starts fresh from a
        clean time-T hidden.

        At depth step k: slot k+1's input is the model's OWN argmax cb_k
        (detached — no grad through the fed-back token) with prob ss_prob, else
        the GT cb_k. Targets/loss are unchanged (still CE vs gt_codes). Costs
        ~K× depth-T compute (depth-T is small) and is only paid when ss_prob>0.

        Returns logits: [B, T_tok, K, vocab_size]
        """
        B, T_tok, _ = hidden_t.shape
        K = self.n_codebooks
        D = self.d_model
        h0 = self.input_proj(hidden_t)                         # [B, T, D] (grad)
        # Future (not-yet-predicted) slots are zeros; depth attention is causal
        # (is_causal=True) so slot k never sees slots >k → zeros are harmless.
        zero = torch.zeros(B, T_tok, D, device=hidden_t.device, dtype=h0.dtype)
        slots = [h0] + [zero] * (K - 1)
        logits_all = []
        for k in range(K):
            seq = torch.stack(slots, dim=2)                    # [B, T, K, D] fresh
            seq = self._run_layers(seq)
            logits_k = self.cb_output_head[k](seq[:, :, k, :])  # [B, T, vocab]
            logits_all.append(logits_k)
            if k < K - 1:
                gt_tok = gt_codes[:, :, k]
                if ss_prob > 0.0:
                    own = logits_k.detach().argmax(dim=-1)      # [B, T] detached
                    mix = torch.rand(B, T_tok, device=hidden_t.device) < ss_prob
                    fed = torch.where(mix, own, gt_tok)
                else:
                    fed = gt_tok
                slots[k + 1] = self.cb_input_embed[k](fed)      # [B, T, D]
        return torch.stack(logits_all, dim=2)                  # [B, T, K, vocab]

    @torch.no_grad()
    def generate_frames(self, hidden_t: torch.Tensor,
                        sample_fn=None) -> torch.Tensor:
        """Greedy (or sampled) inference — iterates depth axis frame-by-frame.

        hidden_t: [B, T_tok, time_d_model]
        sample_fn: callable(logits_step: [B, vocab], cb_idx: int) → tokens [B]
                   Default None = greedy argmax.

        Returns codes: [B, T_tok, K] (long).

        Implementation: we vectorize across time but iterate across depth.
        At depth step k we know cb_0..cb_{k-1} (filled into slots 1..k of the
        depth seq); run depth-T, take slot k's output, head_k → logits_k,
        sample → cb_k, write into slot k+1 (if k<K-1) for the next step.
        """
        B, T_tok, _ = hidden_t.shape
        K = self.n_codebooks

        seq = self._build_depth_seq(hidden_t, gt_codes=None)
        codes = torch.zeros(B, T_tok, K, dtype=torch.long, device=hidden_t.device)

        for k in range(K):
            seq_step = self._run_layers(seq)            # [B, T_tok, K, D]
            slot_k = seq_step[:, :, k, :]                # [B, T_tok, D]
            logits_k = self.cb_output_head[k](slot_k)    # [B, T_tok, vocab]

            if sample_fn is None:
                tok_k = logits_k.argmax(dim=-1)          # [B, T_tok]
            else:
                # sample_fn flattened then reshaped
                flat = sample_fn(logits_k.reshape(-1, self.vocab_size), k)
                tok_k = flat.reshape(B, T_tok)

            codes[:, :, k] = tok_k

            # Write cb_k into slot k+1 (next depth step's input).
            if k < K - 1:
                seq[:, :, k + 1, :] = self.cb_input_embed[k](tok_k)

        return codes
