"""Online augmentation wrapper around PodcastAugmentorV4.

Lives here so the DataLoader can apply DSP per sample. The augmentor itself is
vendored under `augmentation/podcast_augmentor_v4.py` (see `data.augmentor_path`)
— do NOT duplicate its code here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_AUGMENTOR_SINGLETON = None


def _ensure_sibling_on_path(sibling_repo: str | Path) -> Path:
    sibling = Path(sibling_repo).resolve()
    if not sibling.exists():
        raise FileNotFoundError(
            f"Augmentor directory not found at {sibling}. "
            "Set data.augmentor_path in config to its location."
        )
    p = str(sibling)
    if p not in sys.path:
        sys.path.insert(0, p)
    return sibling


def get_augmentor(sibling_repo: str | Path, disable_noise: bool = False,
                  jitter: float | None = None,
                  strategy_mix: tuple | list | None = None,
                  clean_pass_prob: float | None = None,
                  packet_loss_prob: float | None = None,
                  reverb_prob: float | None = None,
                  hard_clip_prob: float | None = None):
    """Lazy-load PodcastAugmentorV4. Returns a singleton (one per process).

    Chdirs to the augmentor directory before instantiation because PodcastAugmentorV4
    reads `calibration_results.csv`, `podcast_target_pool.csv`,
    and `noise_bank/` via *relative* paths.

    Optional params override PodcastAugmentorV4 defaults:
      jitter           (default 0.40) — DSP param jitter range; lower = milder
      strategy_mix     (default (0.6, 0.25, 0.15)) — (retrieve, random_palette,
                       random_synth) probabilities
      clean_pass_prob  (default 0.05) — probability of pass-through (no aug)
      packet_loss_prob (default 0.15) — P(apply packet-loss gaps, post-chain)
      reverb_prob      (default 0.10) — P(convolve synth RIR, pre-chain)
      hard_clip_prob   (default 0.10) — P(hard quantile clip, pre-chain)
    The last three are the URGENT-recipe out-of-distribution augs. Pass 0.0 to
    disable them — useful when fine-tuning from a checkpoint that never saw
    them, so the training distribution stays fixed.
    """
    global _AUGMENTOR_SINGLETON
    if _AUGMENTOR_SINGLETON is not None:
        return _AUGMENTOR_SINGLETON

    sibling = _ensure_sibling_on_path(sibling_repo)
    # Augmentor opens CSVs / noise_bank with relative paths — make cwd sibling
    # for this worker process (DataLoader workers each have their own cwd).
    os.chdir(sibling)
    from podcast_augmentor_v4 import PodcastAugmentorV4

    init_kwargs = {}
    if jitter is not None:
        init_kwargs["jitter"] = float(jitter)
    if strategy_mix is not None:
        init_kwargs["strategy_mix"] = tuple(strategy_mix)
    if clean_pass_prob is not None:
        init_kwargs["clean_pass_prob"] = float(clean_pass_prob)
    if packet_loss_prob is not None:
        init_kwargs["packet_loss_prob"] = float(packet_loss_prob)
    if reverb_prob is not None:
        init_kwargs["reverb_prob"] = float(reverb_prob)
    if hard_clip_prob is not None:
        init_kwargs["hard_clip_prob"] = float(hard_clip_prob)
    aug = PodcastAugmentorV4(**init_kwargs)
    if disable_noise:
        # PodcastAugmentorV4.__init__ has no disable_noise flag; instead force noise
        # params to 0 after each jitter call so apply_chain skips both
        # apply_speech_noise (gated `if snr > 0`) and noise floor injection
        # (gated `if noise_floor`). Affects all 3 strategies (retrieve,
        # random_palette, random_synth).
        _orig_jitter = aug._jitter_params

        def _jitter_no_noise(params, jitter=None):
            p = _orig_jitter(params, jitter=jitter)
            p["noise_target_db"] = 0
            p["speech_noise_snr_db"] = 0
            return p

        aug._jitter_params = _jitter_no_noise
    _AUGMENTOR_SINGLETON = aug
    return _AUGMENTOR_SINGLETON


def augment_waveform(wav: np.ndarray, sr: int, augmentor) -> tuple[np.ndarray, int, dict]:
    """Single-sample augmentation. Returns (degraded_wav, output_sr, aug_info).

    aug_info is the augmentor's per-sample roll record (strategy, combo_idx,
    jitter values, codec target, etc.) — useful for post-hoc spike forensics
    (see train.py spike logger). Dataset stores it on PairedSample.aug_info.
    """
    degraded, out_sr, info = augmentor.augment(wav, sr)
    return degraded, out_sr, info if isinstance(info, dict) else {}
