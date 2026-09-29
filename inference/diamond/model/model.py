"""SpeechRestorationModel — RQ-Transformer wiring.

    mel  →  encoder  →  time-transformer  →  depth-transformer  →  RVQ codes
                                                                       ↓
                                                                 (frozen DAC decoder, outside model)

The earlier read-out — nine parallel heads on one shared hidden vector, which
drove a decoder weight-norm runaway — is gone. Instead the time-T outputs ONE
hidden per frame and the depth-T walks the 9 codebooks sequentially inside each
frame, with a dedicated readout per codebook.

API:
  • forward(mel, target_tokens, mel_pad_mask) → (logits, target, target_mask).
    There is no delayed-pattern PAD masking — target_mask is just the
    mel_pad_mask broadcast to the token grid (1:1 alignment since
    hop=280 ≈ DAC 86 Hz).
  • generate(mel, n_tok_frames, mel_pad_mask) → codes [B, T, K] ready for DAC.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .depth_transformer import DepthTransformer
from .encoder import MelEncoder
from .time_transformer import TimeTransformer


class SpeechRestorationModel(nn.Module):
    def __init__(self, encoder_cfg: dict, time_cfg: dict | None = None,
                 depth_cfg: dict | None = None, decoder_cfg: dict | None = None,
                 max_seq_len: int = 4096):
        super().__init__()
        # Backward-compat alias: train.py and old configs use `decoder_cfg`,
        # which conceptually IS the time-transformer config. Accept both names.
        if time_cfg is None:
            assert decoder_cfg is not None, "must pass time_cfg= or decoder_cfg="
            time_cfg = decoder_cfg
        if depth_cfg is None:
            raise ValueError("depth_cfg is required for v10 RQ-Transformer model")
        self.encoder = MelEncoder(
            n_mels=encoder_cfg["n_mels"],
            d_model=encoder_cfg["d_model"],
            n_heads=encoder_cfg["n_heads"],
            n_layers=encoder_cfg["n_layers"],
            ff_dim=encoder_cfg["ff_dim"],
            dropout=encoder_cfg.get("dropout", 0.1),
            max_seq_len=max_seq_len,
            layer_scale_init=encoder_cfg.get("layer_scale_init", 0.1),
            rope_learnable=encoder_cfg.get("rope_learnable", False),
            qk_norm=encoder_cfg.get("qk_norm", False),
        )
        self.time_t = TimeTransformer(
            d_model=time_cfg["d_model"],
            n_heads=time_cfg["n_heads"],
            n_layers=time_cfg["n_layers"],
            ff_dim=time_cfg["ff_dim"],
            n_codebooks=time_cfg["n_codebooks"],
            vocab_size=time_cfg["vocab_size"],
            bos_token_id=time_cfg["bos_token_id"],
            dropout=time_cfg.get("dropout", 0.1),
            max_seq_len=max_seq_len,
            rope_learnable=time_cfg.get("rope_learnable", False),
            qk_norm=time_cfg.get("qk_norm", False),
            layer_scale_init=time_cfg.get("layer_scale_init", 0.1),
        )
        self.depth_t = DepthTransformer(
            time_d_model=time_cfg["d_model"],
            d_model=depth_cfg["d_model"],
            n_heads=depth_cfg["n_heads"],
            n_layers=depth_cfg["n_layers"],
            ff_dim=depth_cfg["ff_dim"],
            n_codebooks=time_cfg["n_codebooks"],
            vocab_size=time_cfg["vocab_size"],
            dropout=depth_cfg.get("dropout", 0.1),
            qk_norm=depth_cfg.get("qk_norm", True),
            layer_scale_init=depth_cfg.get("layer_scale_init", 0.1),
        )
        self.n_codebooks = time_cfg["n_codebooks"]
        self.vocab_size = time_cfg["vocab_size"]
        self.bos_token_id = time_cfg["bos_token_id"]

    @property
    def decoder(self):
        """Backward-compat alias — pre-v10 code calls model.decoder.* for
        diagnostics and parameter access. The time-transformer fills that role.
        """
        return self.time_t

    def forward(self, mel: torch.Tensor, target_tokens: torch.Tensor,
                mel_pad_mask: torch.Tensor | None = None,
                target_mask: torch.Tensor | None = None,
                ss_prob: float = 0.0
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        mel:           [B, T_mel, n_mels]
        target_tokens: [B, T_tok, K] — RVQ codes from frozen DAC
        mel_pad_mask:  [B, T_mel] — True where valid (so True passes through cross-attn)
        target_mask:   [B, T_tok, K] — True where target is real. If None,
                       falls back to mel_pad_mask projected to T_tok, or all-True.

        Returns:
            logits:      [B, T_tok, K, vocab_size]
            target:      [B, T_tok, K]              — same as target_tokens (no delay)
            target_mask: [B, T_tok, K]
        """
        encoder_out = self.encoder(mel, pad_mask=mel_pad_mask)
        hidden_t = self.time_t(target_tokens, encoder_out, encoder_pad_mask=mel_pad_mask)
        # ss_prob>0 → depth-axis scheduled sampling (bounded, safe; see
        # DepthTransformer.forward_scheduled). 0 → pure teacher forcing.
        if self.training and ss_prob > 0.0:
            logits = self.depth_t.forward_scheduled(hidden_t, target_tokens, ss_prob)
        else:
            logits = self.depth_t(hidden_t, gt_codes=target_tokens)

        if target_mask is None:
            if mel_pad_mask is not None and mel_pad_mask.size(1) >= target_tokens.size(1):
                tok_mask = mel_pad_mask[:, :target_tokens.size(1)]
                target_mask = tok_mask.unsqueeze(-1).expand_as(target_tokens)
            else:
                target_mask = torch.ones_like(target_tokens, dtype=torch.bool)

        return logits, target_tokens, target_mask

    @torch.no_grad()
    def generate(self, mel: torch.Tensor, n_tok_frames: int,
                 mel_pad_mask: torch.Tensor | None = None,
                 sample_fn=None, use_cache: bool = True) -> torch.Tensor:
        """Autoregressive inference.

        Outer loop walks time frames. For each frame:
          1. run TimeTransformer.step on history → hidden_t for the new frame
          2. run DepthTransformer iteratively over 9 codebooks → cb_0..cb_8
          3. append that 9-tuple to the running history; loop.

        mel:           [B, T_mel, n_mels]
        n_tok_frames:  number of DAC frames to produce
        mel_pad_mask:  [B, T_mel]
        sample_fn:     optional callable(logits_step: [B, vocab], cb_idx: int)
                       → tokens [B]. None = greedy argmax on every codebook.
                       Lets callers do per-codebook sampling (e.g. greedy for
                       coarse cb, top-p for fine cb) without touching the model.

        Returns codes: [B, n_tok_frames, K] ready to feed into DAC decoder.

        Note on cost: this naive loop recomputes the time-T full pass at every
        new frame. KV-cache support would cut this to O(T) total instead of
        O(T²). It is acceptable here because chunked inference keeps the
        effective T below ~220 frames per chunk.
        """
        B = mel.size(0)
        K = self.n_codebooks
        device = mel.device

        encoder_out = self.encoder(mel, pad_mask=mel_pad_mask)

        out_codes = []

        if use_cache:
            # O(T) incremental decode: per-layer KV caches; each step processes
            # only the new frame. Mathematically identical to the naive path
            # (standard causal KV-cache identity); ~5-10x faster on long chunks.
            caches = self.time_t.init_caches()
            frame = torch.full((B, 1, K), self.bos_token_id, dtype=torch.long, device=device)
            for i in range(n_tok_frames):
                hidden_last = self.time_t.step_cached(
                    frame, i, caches, encoder_out, encoder_pad_mask=mel_pad_mask)
                frame_codes = self.depth_t.generate_frames(
                    hidden_last.unsqueeze(1), sample_fn=sample_fn)  # [B, 1, K]
                out_codes.append(frame_codes)
                frame = frame_codes  # next step's input is what we just produced
            return torch.cat(out_codes, dim=1)

        # Naive O(T^2) path — recomputes the full time-T pass every frame.
        # Kept for validating the cached path produces matching output.
        history = torch.full((B, 1, K), self.bos_token_id, dtype=torch.long, device=device)
        for _ in range(n_tok_frames):
            hidden_last = self.time_t.step(history, encoder_out, encoder_pad_mask=mel_pad_mask)
            frame_codes = self.depth_t.generate_frames(
                hidden_last.unsqueeze(1), sample_fn=sample_fn)  # [B, 1, K]
            out_codes.append(frame_codes)
            history = torch.cat([history, frame_codes], dim=1)

        return torch.cat(out_codes, dim=1)  # [B, n_tok_frames, K]
