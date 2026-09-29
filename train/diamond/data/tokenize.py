"""Mel extractor (input side) + DAC encode/decode wrappers (target side).

DAC is loaded as frozen — gradients never flow through it.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchaudio


class MelExtractor(nn.Module):
    """Log-mel spectrogram with min-max normalization to [-1, 1]."""

    def __init__(self, sample_rate: int = 24000, n_fft: int = 1024,
                 hop_length: int = 280, win_length: int = 1024,
                 n_mels: int = 80, fmin: float = 0, fmax: float = 8000,
                 log_clamp_db: float = -80.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.log_clamp_db = log_clamp_db

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav:     [B, num_samples] or [num_samples], mono, expected sample_rate.
        returns: [B, T_mel, n_mels] in [-1, 1]
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mel_power = self.mel(wav)
        mel_db = 10.0 * torch.log10(mel_power.clamp(min=1e-10))
        mel_db = mel_db.clamp(min=self.log_clamp_db)
        mel_norm = (mel_db - self.log_clamp_db) / (-self.log_clamp_db) * 2.0 - 1.0
        return mel_norm.transpose(1, 2)


def resample_to(wav: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    if sr_in == sr_out:
        return wav
    return torchaudio.functional.resample(wav, sr_in, sr_out)


class FrozenDAC:
    """Wrapper around descript-audio-codec for tokenization + waveform synthesis.

    Both encode and decode run under torch.no_grad(). Targets are .detach()'d
    before being used downstream.
    """

    def __init__(self, checkpoint_path: str | Path | None = None,
                 model_type: str = "44khz", device: str = "cuda"):
        try:
            import dac
        except ImportError as e:
            raise ImportError(
                "descript-audio-codec is required. "
                "Install via: uv pip install descript-audio-codec"
            ) from e

        if checkpoint_path is None:
            checkpoint_path = dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(str(checkpoint_path))
        self.model.eval()
        self.model.to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        self.sample_rate = self.model.sample_rate
        self.n_codebooks = self.model.n_codebooks
        self.codebook_size = self.model.codebook_size

    @torch.no_grad()
    def encode(self, wav: torch.Tensor, sr_in: int) -> torch.Tensor:
        """
        wav:    [B, num_samples] @ sr_in
        returns codes: [B, T_tok, n_codebooks] (long)
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = resample_to(wav, sr_in, self.sample_rate).to(self.device)
        wav = wav.unsqueeze(1) if wav.dim() == 2 else wav
        z, codes, _, _, _ = self.model.encode(wav)
        return codes.transpose(1, 2).detach().contiguous()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes:   [B, T_tok, n_codebooks] (long)
        returns: [B, num_samples] @ self.sample_rate
        """
        codes = codes.transpose(1, 2).to(self.device)
        z = self.model.quantizer.from_codes(codes)[0]
        wav = self.model.decode(z)
        return wav.squeeze(1).detach()
