"""Paired (degraded_mel, clean_dac_tokens) dataset.

Two modes:
  • Pre-tokenized: clean DAC tokens cached on disk as .pt; loader returns
    (clean_wav, clean_tokens) and applies augmentation online.
  • Online tokenize: encode clean wav with DAC inside __getitem__ (slow,
    only for small experiments).

The augmentor runs on CPU inside DataLoader workers — keep num_workers high.

The wav crop and the corresponding token slice are taken in lockstep: if we
randomly crop the studio wav at sample `start`, we also slice the pre-computed
DAC tokens at frame `start * 44100/sr // 512`, so the model never sees a
(degraded_5s_wav, full_clip_tokens) pair.

After augmentation we resample every sample to `output_sr` (the mel SR) so
the collator can stack them without mixed-rate assertions — the augmentor's
codec stage can flip the rate to anything from 16k to 32k per sample.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .augmentation import augment_waveform, get_augmentor

DAC_SR = 44100
DAC_HOP = 512
PAD_TOKEN_ID = 1024


@dataclass
class PairedSample:
    degraded_wav: np.ndarray
    degraded_sr: int
    clean_tokens: torch.Tensor
    valid_wav_samples: int   # samples of `degraded_wav` that are real (rest is zero-pad)
    valid_token_count: int   # leading frames of `clean_tokens` that are real (rest is PAD)
    sample_id: str = ""      # manifest id — surfaces in spike logs for forensics
    aug_info: dict = None    # per-sample augmentor roll record — for spike forensics


class StudioPairedDataset(Dataset):
    """Iterates over studio audio files; returns (degraded_wav, clean_tokens)."""

    def __init__(
        self,
        manifest: list[dict],
        augmentor_path: str | Path,
        audio_seconds: float = 5.0,
        random_crop: bool = True,
        online_augmentation: bool = True,
        n_samples: int | None = None,
        output_sr: int = 24000,
        disable_noise: bool = False,
        augmentor_kwargs: dict | None = None,
        start_at_zero_prob: float = 0.0,
        crop_seconds_range: tuple[float, float] | list | None = None,
        noise_aug_cfg=None,
    ):
        self.manifest = manifest if n_samples is None else manifest[:n_samples]
        self.audio_seconds = audio_seconds
        self.random_crop = random_crop
        self.online_augmentation = online_augmentation
        self.augmentor_path = augmentor_path
        self.output_sr = output_sr
        self.disable_noise = disable_noise
        self.augmentor_kwargs = dict(augmentor_kwargs) if augmentor_kwargs else {}
        self.start_at_zero_prob = float(start_at_zero_prob)
        # Variable-length crop: per-sample crop length drawn uniformly from
        # [crop_min, crop_max]. None = fixed `audio_seconds` (legacy behavior).
        # Why: training at one fixed length leaves encoder OOD when inference
        # chunks are shorter (chunked decoding uses 2.5s; training was 10s).
        # Variable crop teaches encoder length-robustness across the inference
        # range without needing chunk_sec to match training length exactly.
        if crop_seconds_range is not None:
            lo, hi = float(crop_seconds_range[0]), float(crop_seconds_range[1])
            if not (0.1 < lo <= hi):
                raise ValueError(
                    f"crop_seconds_range must be (lo, hi) with 0.1 < lo <= hi, got ({lo}, {hi})"
                )
            self.crop_seconds_range: tuple[float, float] | None = (lo, hi)
        else:
            self.crop_seconds_range = None
        self._aug_cache = None
        # Robustness track (RIR pre-codec + broadband noise post-codec).
        # Built lazily per worker; None when data.noise_aug.enabled is false.
        self.noise_aug_cfg = noise_aug_cfg
        self._noise_aug_cache = None
        self._noise_aug_built = False

    def _augmentor(self):
        if self._aug_cache is None:
            self._aug_cache = get_augmentor(
                self.augmentor_path,
                disable_noise=self.disable_noise,
                **self.augmentor_kwargs,
            )
        return self._aug_cache

    def _noise_augmentor(self):
        if not self._noise_aug_built:
            from .noise_aug import get_noise_augmentor
            self._noise_aug_cache = get_noise_augmentor(self.noise_aug_cfg)
            self._noise_aug_built = True
        return self._noise_aug_cache

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> PairedSample:
        entry = self.manifest[idx]
        wav_path = Path(entry["wav_path"])
        tokens_path = Path(entry["tokens_path"])
        sample_id = str(entry.get("id", wav_path.stem))

        wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        full_tokens = torch.load(tokens_path, map_location="cpu")

        wav, clean_tokens, n_valid_wav, n_valid_tok = self._crop_paired(wav, sr, full_tokens)

        aug_info: dict = {}
        if self.online_augmentation:
            noise_aug = self._noise_augmentor()
            # RIR convolution happens BEFORE the codec (reverb survives codec).
            rir_info = {}
            if noise_aug is not None:
                aug_in, rir_info = noise_aug.maybe_apply_rir(wav, sr)
            else:
                aug_in = wav
            degraded, deg_sr, aug_info = augment_waveform(aug_in, sr, self._augmentor())
            n_valid_wav = int(round(n_valid_wav * deg_sr / sr))
            n_valid_wav = min(n_valid_wav, len(degraded))

            # SAFETY: detect silent/garbage augmentor output (e.g. from ffmpeg
            # subprocess failure). If output is silent while input wasn't,
            # log + replace with clean wav (fallback) to avoid training on garbage.
            out_peak = float(np.max(np.abs(degraded))) if len(degraded) > 0 else 0.0
            in_peak = float(np.max(np.abs(wav)))
            if in_peak > 0.01 and out_peak < 0.001:
                # Augmentor returned silence on non-silent input — broken roundtrip
                aug_info = dict(aug_info) if aug_info else {}
                aug_info["aug_silent_output"] = True
                aug_info["in_peak"] = in_peak
                aug_info["out_peak"] = out_peak
                # Fall back to clean input — training continues on safe data
                degraded, deg_sr = wav, sr
                n_valid_wav = int(round(n_valid_wav * deg_sr / sr)) if deg_sr != sr else n_valid_wav

            # Broadband noise mixed AFTER the codec (post-codec survives; placed
            # after the silent-output check so it can't mask a broken roundtrip).
            noise_info = {}
            if noise_aug is not None:
                degraded, noise_info = noise_aug.maybe_mix_noise(degraded, deg_sr)
            # Fold RIR + noise provenance into aug_info so spike_events.jsonl can
            # blame a specific RIR / noise clip / SNR when a batch spikes grad.
            if rir_info or noise_info:
                aug_info = dict(aug_info) if aug_info else {}
                aug_info["noise_aug"] = {**rir_info, **noise_info}
        else:
            degraded, deg_sr = wav, sr

        # Normalize SR — augmentor's codec stage outputs variable rates.
        if deg_sr != self.output_sr:
            degraded = librosa.resample(
                degraded.astype(np.float32),
                orig_sr=deg_sr, target_sr=self.output_sr,
                res_type="soxr_hq",
            )
            n_valid_wav = int(round(n_valid_wav * self.output_sr / deg_sr))
            n_valid_wav = min(n_valid_wav, len(degraded))
            deg_sr = self.output_sr

        return PairedSample(
            degraded_wav=degraded.astype(np.float32),
            degraded_sr=deg_sr,
            clean_tokens=clean_tokens,
            valid_wav_samples=n_valid_wav,
            valid_token_count=n_valid_tok,
            sample_id=sample_id,
            aug_info=aug_info,
        )

    def _crop_paired(
        self, wav: np.ndarray, sr: int, tokens: torch.Tensor,
    ) -> tuple[np.ndarray, torch.Tensor, int, int]:
        """Take a paired (wav, tokens) crop of `audio_seconds` length.

        Returns:
            wav_out:        [target_wav_len] np.float32 (zero-padded if input shorter)
            tokens_out:     [target_n_tok, K] long-dtype (PAD-padded if input shorter)
            n_valid_wav:    samples of wav_out that are real (not zero-pad)
            n_valid_tok:    leading frames of tokens_out that are real (not PAD)
        """
        # Pick crop length for THIS sample. Variable mode draws a fresh value
        # per __getitem__ call → samples within one batch may have different
        # lengths; collate pads to batch-max and `valid_wav_samples` keeps
        # loss masking honest.
        if self.crop_seconds_range is not None:
            lo, hi = self.crop_seconds_range
            crop_sec = random.uniform(lo, hi)
        else:
            crop_sec = self.audio_seconds
        target_wav_len = int(crop_sec * sr)
        target_n_tok = int(round(crop_sec * DAC_SR)) // DAC_HOP

        if len(wav) >= target_wav_len:
            n_tok_avail = tokens.shape[0]
            # Constrain start so the corresponding token window stays in range.
            max_start_wav = len(wav) - target_wav_len
            max_tok_start = max(0, n_tok_avail - target_n_tok)
            max_start_for_tok = int(max_tok_start * DAC_HOP * sr / DAC_SR)
            max_start = max(0, min(max_start_wav, max_start_for_tok))

            # With probability start_at_zero_prob force start=0 so the model
            # sees utterance-initial frames during training (fixes inference
            # cold-start: at infer, chunk 0 begins at an absolute onset that
            # random_crop almost never samples for files >= target length).
            force_zero = (
                self.random_crop
                and self.start_at_zero_prob > 0.0
                and np.random.random() < self.start_at_zero_prob
            )
            if force_zero or not self.random_crop or max_start == 0:
                start = 0
            else:
                start = int(np.random.randint(0, max_start + 1))

            wav_out = wav[start : start + target_wav_len].astype(np.float32, copy=False)
            tok_start = int(start * DAC_SR / sr) // DAC_HOP
            tok_out = tokens[tok_start : tok_start + target_n_tok]
            # Pad to target length if rounding edge-cases under-shoot.
            if tok_out.shape[0] < target_n_tok:
                pad = torch.full(
                    (target_n_tok - tok_out.shape[0], tok_out.shape[1]),
                    PAD_TOKEN_ID, dtype=tok_out.dtype,
                )
                tok_out = torch.cat([tok_out, pad], dim=0)
            return wav_out, tok_out, target_wav_len, target_n_tok

        # Wav shorter than target window — pad both wav and tokens with PAD.
        wav_padded = np.pad(wav, (0, target_wav_len - len(wav)), mode="constant").astype(np.float32, copy=False)
        n_real_tok = min(tokens.shape[0], target_n_tok)
        if tokens.shape[0] >= target_n_tok:
            tok_out = tokens[:target_n_tok]
        else:
            pad = torch.full(
                (target_n_tok - tokens.shape[0], tokens.shape[1]),
                PAD_TOKEN_ID, dtype=tokens.dtype,
            )
            tok_out = torch.cat([tokens, pad], dim=0)
        return wav_padded, tok_out, len(wav), n_real_tok


def collate_paired(batch: list[PairedSample], pad_token_id: int = 1024) -> dict:
    """Pad-collate degraded waveforms and stack token tensors.

    `wav_lengths` reports VALID (non-zero-padded) samples; `token_lengths`
    reports VALID (non-PAD) tokens. Both come from `_crop_paired` and may be
    smaller than the tensor's first dim when the source clip was shorter than
    `audio_seconds`. Augmentor SR can differ from input SR (codec resample),
    so wav lengths may not match across the batch — we pad to max.
    """
    max_wav_len = max(b.degraded_wav.shape[0] for b in batch)
    sr = batch[0].degraded_sr
    assert all(b.degraded_sr == sr for b in batch), "mixed SR in batch — resample upstream"

    wav_lengths = torch.tensor([b.valid_wav_samples for b in batch], dtype=torch.long)
    wavs = np.stack([
        np.pad(b.degraded_wav, (0, max_wav_len - len(b.degraded_wav)))
        for b in batch
    ])

    max_tok_len = max(b.clean_tokens.shape[0] for b in batch)
    K = batch[0].clean_tokens.shape[1]
    tokens = torch.full((len(batch), max_tok_len, K), pad_token_id, dtype=torch.long)
    token_lengths = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        T = b.clean_tokens.shape[0]
        tokens[i, :T] = b.clean_tokens
        token_lengths[i] = b.valid_token_count

    return {
        "wav": torch.from_numpy(wavs),
        "wav_sr": sr,
        "wav_lengths": wav_lengths,
        "clean_tokens": tokens,
        "token_lengths": token_lengths,
        "sample_ids": [b.sample_id for b in batch],
        "aug_infos": [b.aug_info if b.aug_info is not None else {} for b in batch],
    }
