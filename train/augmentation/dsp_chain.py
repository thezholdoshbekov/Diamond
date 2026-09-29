"""DSP degradation chain — the runtime augmentation primitive.

`apply_chain` takes a clean waveform plus a parameter dict (one calibrated
combo) and applies the codec + EQ degradation chain. It is the only piece of
the augmentor that runs during training; it has no dataset, CSV or plotting
dependencies, so importing it is side-effect free.

The calibrated parameter palette (`calibration_results.csv`) is built offline
and shipped with the repo — see `augmentation/README.md`.
"""

import numpy as np

from augment_pipeline import apply_codec, apply_mic_coloration
from podcast_augmentor import (
    apply_bass_boost, apply_downsampling, apply_dynamic_compression,
    apply_hf_instability, apply_lowpass_podcast, apply_real_noise_floor,
    apply_speech_noise,
)

# ── Parameter space ─────────────────────────────────────────────────────────
# Value ranges the augmentor's random_synth strategy samples from. Mirrors the
# grid used to build the calibrated palette (calibration_results.csv).
PARAM_SPACE = {
    # Downsampling: always happens (podcast is 24k or 32k)
    "target_sr": [24000, 32000],
    "lowpass_hz": [0, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000],
    "lowpass_order": [4, 6, 8],
    "compression_threshold": [0, -35, -30, -25, -20, -15, -12],
    "compression_ratio": [3, 5, 8, 12],
    "bass_boost_db": [0, 2, 3],
    "bass_cutoff_hz": [800, 1200],
    "noise_target_db": [0, -55, -50, -45, -40],
    "codec_type": ["none", "mp3", "aac", "ogg"],
    "codec_bitrate": [16, 24, 32, 48, 64, 96, 128],
    "codec_passes": [1, 2, 3],
    "codec_engine": ["default", "strict", "fast", "vbr"],
    "mic_bands": [0, 3, 4, 5],  # 0 = no coloration
    "hf_instability_strength": [0, 0.5, 0.7, 0.9],
    "hf_instability_cutoff_hz": [4000, 5000, 6000],
    "hf_instability_mod_hz": [1.0, 2.0, 3.0],
    "speech_noise_snr_db": [0, 20, 25, 30, 35],
}

# Weighted distributions — match real podcast audio's parameter incidence so the
# generated palette isn't dominated by extreme-tail combos.
PARAM_WEIGHTS = {
    "codec_bitrate": {16: 0.02, 24: 0.05, 32: 0.13, 48: 0.20,
                      64: 0.30, 96: 0.20, 128: 0.10},
    "codec_passes": {1: 0.60, 2: 0.30, 3: 0.10},
    "lowpass_hz": {5000: 0.05, 6000: 0.10, 7000: 0.20, 8000: 0.25,
                   9000: 0.20, 10000: 0.15, 12000: 0.05},
    "compression_threshold": {-35: 0.05, -30: 0.10, -25: 0.20, -20: 0.30,
                              -15: 0.25, -12: 0.10},
    "compression_ratio": {3: 0.30, 5: 0.40, 8: 0.20, 12: 0.10},
    "mic_bands": {3: 0.45, 4: 0.35, 5: 0.20},
    "lowpass_order": {4: 0.45, 6: 0.35, 8: 0.20},
    "codec_engine": {"default": 0.50, "vbr": 0.25, "strict": 0.15, "fast": 0.10},
}


def _wchoice(param: str):
    """Weighted random pick from PARAM_WEIGHTS[param]."""
    items = PARAM_WEIGHTS[param]
    keys = list(items.keys())
    probs = list(items.values())
    return np.random.choice(keys, p=probs)


def generate_param_combos(n_combos=500):
    """Generate random parameter combinations using weighted sampling.

    PARAM_WEIGHTS-based picks keep the generated palette close to the real
    podcast parameter distribution (reduces extreme-tail combo prevalence).
    """
    combos = []
    for _ in range(n_combos):
        combo = {}
        combo["target_sr"] = np.random.choice(PARAM_SPACE["target_sr"])

        # Lowpass: 95% chance. Weighted toward 7-10 kHz (typical mp3 cutoff).
        if np.random.random() < 0.95:
            combo["lowpass_hz"] = int(_wchoice("lowpass_hz"))
            combo["lowpass_order"] = int(_wchoice("lowpass_order"))
        else:
            combo["lowpass_hz"] = 0
            combo["lowpass_order"] = 0

        # Compression: 90% chance, weighted toward -20 dB threshold + ratio 5.
        if np.random.random() < 0.90:
            combo["compression_threshold"] = int(_wchoice("compression_threshold"))
            combo["compression_ratio"] = int(_wchoice("compression_ratio"))
        else:
            combo["compression_threshold"] = 0
            combo["compression_ratio"] = 0

        # Bass boost: 30% chance, capped at 3 dB.
        if np.random.random() < 0.30:
            combo["bass_boost_db"] = int(np.random.choice([2, 3], p=[0.7, 0.3]))
            combo["bass_cutoff_hz"] = int(np.random.choice(PARAM_SPACE["bass_cutoff_hz"]))
        else:
            combo["bass_boost_db"] = 0
            combo["bass_cutoff_hz"] = 0

        # Noise floor raise: 50% chance, weighted toward -55 (mid).
        if np.random.random() < 0.50:
            combo["noise_target_db"] = int(np.random.choice([-60, -55, -50],
                                                              p=[0.30, 0.45, 0.25]))
        else:
            combo["noise_target_db"] = 0

        # Codec: 95% chance — mp3 dominant, with weighted bitrate/passes.
        if np.random.random() < 0.95:
            combo["codec_type"] = str(np.random.choice(
                ["mp3", "aac", "ogg"], p=[0.60, 0.25, 0.15]
            ))
            combo["codec_bitrate"] = int(_wchoice("codec_bitrate"))
            combo["codec_passes"] = int(_wchoice("codec_passes"))
            # Engine variant — only meaningful for mp3 (libmp3lame flags).
            if combo["codec_type"] == "mp3":
                combo["codec_engine"] = str(_wchoice("codec_engine"))
            else:
                combo["codec_engine"] = "default"
        else:
            combo["codec_type"] = "none"
            combo["codec_bitrate"] = 0
            combo["codec_passes"] = 0
            combo["codec_engine"] = "default"

        # Mic coloration: 75% chance, weighted toward 3-4 bands.
        if np.random.random() < 0.75:
            combo["mic_bands"] = int(_wchoice("mic_bands"))
        else:
            combo["mic_bands"] = 0

        # HF instability: 25% chance, strength capped at 0.5.
        if np.random.random() < 0.25:
            combo["hf_instability_strength"] = 0.5
            combo["hf_instability_cutoff_hz"] = int(np.random.choice(PARAM_SPACE["hf_instability_cutoff_hz"]))
            combo["hf_instability_mod_hz"] = float(np.random.choice(PARAM_SPACE["hf_instability_mod_hz"]))
        else:
            combo["hf_instability_strength"] = 0.0
            combo["hf_instability_cutoff_hz"] = 0
            combo["hf_instability_mod_hz"] = 0.0

        # Broadband speech noise: DISABLED (Variant B).
        combo["speech_noise_snr_db"] = 0

        combos.append(combo)

    return combos


def apply_chain(y, sr, params):
    """Apply augmentation chain with given parameters. Returns (audio, new_sr)."""
    out = y.copy()
    out_sr = sr

    # 1. Downsampling FIRST — then the rest of the augmentation chain.
    target_sr = params.get("target_sr", 24000)
    if target_sr and target_sr != out_sr:
        out, out_sr = apply_downsampling(out, out_sr, target_sr=target_sr)

    # 2. Bass boost (if any)
    bass_db = params.get("bass_boost_db", 0)
    if bass_db > 0:
        out = apply_bass_boost(out, out_sr, gain_db=bass_db,
                               cutoff_hz=params.get("bass_cutoff_hz", 1000))

    # 3. Mic coloration (if any)
    mic_bands = params.get("mic_bands", 0)
    if mic_bands > 0:
        out = apply_mic_coloration(out, out_sr, n_bands=mic_bands)

    # 4. Lowpass (if any)
    lp_hz = params.get("lowpass_hz", 0)
    if lp_hz > 0:
        out = apply_lowpass_podcast(out, out_sr, cutoff_hz=lp_hz,
                                   order=params.get("lowpass_order", 4))

    # 5. Dynamic compression (if any)
    comp_thresh = params.get("compression_threshold", 0)
    if comp_thresh < 0:
        out = apply_dynamic_compression(out, out_sr,
                                         threshold_db=comp_thresh,
                                         ratio=params.get("compression_ratio", 4))

    # 6. Codec (if any)
    codec_type = params.get("codec_type", "none")
    if codec_type != "none" and codec_type:
        bitrate = params.get("codec_bitrate", 64)
        n_passes = params.get("codec_passes", 1)
        # Engine variant — picks the libmp3lame flag combo for mp3; ignored by
        # apply_codec for aac/ogg. The default keeps backward compatibility with
        # any palette row predating this column.
        engine = params.get("codec_engine", "default") or "default"
        out = apply_codec(out, out_sr, codec=codec_type, bitrate=bitrate,
                          n_passes=n_passes, engine=engine)

    # 6.5. Broadband speech noise — applied AFTER codec so the MP3 psycho-
    # acoustic model can't strip it. Injecting it before the codec was tried
    # first, and the codec removed the noise entirely. Post-codec injection
    # lands on an already-quantized signal → the broadband component survives
    # into the final output.
    sn_snr = params.get("speech_noise_snr_db", 0)
    if sn_snr and sn_snr > 0:
        out = apply_speech_noise(out, out_sr, snr_db=float(sn_snr), noise_type="pink")

    # 7. HF instability — after codec, before noise (adds temporal HF flicker).
    hf_str = params.get("hf_instability_strength", 0)
    if hf_str and hf_str > 0:
        out = apply_hf_instability(
            out, out_sr,
            strength=float(hf_str),
            hf_cutoff_hz=float(params.get("hf_instability_cutoff_hz", 5000) or 5000),
            mod_rate_hz=float(params.get("hf_instability_mod_hz", 2.0) or 2.0),
        )

    # NB: noise_floor_raise was previously applied here (before loudness norm)
    # but that caused audible TV-static-like hiss in heavily-compressed combos.
    # Reason: loudness norm boosts everything by +15..+22 dB, including the
    # injected noise. Pre-loudness noise at target=-43 dB became ≈-21 dB
    # post-loudness — way above real podcast audio's -43..-50 dB floor. It is now
    # applied AFTER loudness norm, so the requested target_floor_db is the level
    # relative to the final peak=0.99 output, matching real podcast behaviour.

    # Loudness normalization to match real podcast audio profile.
    # Real podcast audio: peak=1.0 (brick-wall), RMS ≈ -16 dB (range -12..-20).
    # Without this, heavy compression + low-bitrate codec combos produced
    # peak=0.99 but RMS down to -34 dB — audibly 4× quieter than real podcast,
    # and it was the dominant complaint in listening checks.
    #
    # Why gain-then-peak-normalize alone doesn't work:
    #   linear gain + peak normalize is mathematically a no-op for RMS:
    #   final_rms = pre_rms - 20*log10(pre_peak/0.99). The gain cancels.
    #   To raise RMS we MUST compress crest factor — i.e. soft-saturate.
    #
    # Soft tanh saturation: nearly linear for |x|<0.5, smoothly rolls off
    # transients beyond. Adds even harmonics like analog tape — sonically
    # gentler than hard clipping. After saturation peak is bounded to 0.99
    # by construction (tanh asymptote), so no separate peak normalize.
    TARGET_RMS_DB = -16.0
    # +22 dB cap chosen empirically by listening: combos with thr=-38 + a heavy
    # ratio drop pre-loudness RMS to ~-44 dB. A +15 dB cap left such samples at
    # -29 dB (audibly too quiet); +22 dB recovers them to ~-22 dB while still
    # bounding noise amplification on edge cases.
    MAX_GAIN_DB = 22.0

    current_rms = float(np.sqrt(np.mean(out ** 2)) + 1e-12)
    current_rms_db = 20.0 * np.log10(current_rms)
    gain_db = min(TARGET_RMS_DB - current_rms_db, MAX_GAIN_DB)
    if gain_db > 0:
        gain_lin = 10 ** (gain_db / 20.0)
        # Soft saturate: 0.99*tanh(x*gain/0.99). Quiet content (|x*gain|<0.4)
        # is essentially gain*x; transients above 0.99 fold smoothly to 0.99.
        out = 0.99 * np.tanh(out * gain_lin / 0.99)
    else:
        # Signal already loud enough — peak-normalize down to 0.99 if needed.
        peak = float(np.max(np.abs(out)))
        if peak > 0.99:
            out = out * 0.99 / peak

    # Noise floor raise — applied AFTER loudness so target_floor_db is the
    # level relative to the final peak=0.99 output (matches real podcast audio,
    # which has peak=1.0 + noise floor at -43..-50 dB by codec quantization).
    #
    # apply_real_noise_floor uses real silence segments mined from podcast
    # audio (run extract_noise_bank.py first). If the bank is missing, it
    # transparently falls back to synthetic apply_noise_floor_raise so this
    # chain works either way.
    noise_floor = params.get("noise_target_db", 0)
    if noise_floor < 0:
        out = apply_real_noise_floor(out, out_sr, target_floor_db=noise_floor)
        peak_post_noise = float(np.max(np.abs(out)))
        if peak_post_noise > 0.99:
            out = out * 0.99 / peak_post_noise

    return out, out_sr
