"""DSP degradation primitives — the building blocks of the augmentation chain.

Two families live here:

  • podcast-channel degradations (bass boost, dynamic compression, lowpass,
    downsampling, noise floor, HF instability) — assembled by
    `dsp_chain.apply_chain` from a calibrated parameter combo;
  • OOD phone artifacts (packet loss, reverb, hard clipping) — applied as
    low-probability add-ons by `PodcastAugmentorV4.augment`.

Not an entry point: use `podcast_augmentor_v4.PodcastAugmentorV4`.
"""

import os
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, resample_poly
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PODCAST-SPECIFIC DEGRADATIONS
# =============================================================================

def apply_bass_boost(y: np.ndarray, sr: int,
                     gain_db: float = 6.0,
                     cutoff_hz: float = 1000) -> np.ndarray:
    """
    Apply bass boost typical of podcast recordings.
    Analysis showed podcast is +6 dB louder in 0-3 kHz vs studio.
    """
    nyq = sr / 2
    sos = butter(2, min(cutoff_hz / nyq, 0.99), btype='low', output='sos')
    bass = sosfiltfilt(sos, y)
    gain_linear = 10 ** (gain_db / 20) - 1.0
    return y + gain_linear * bass


def apply_dynamic_compression(y: np.ndarray, sr: int,
                              threshold_db: float = -20,
                              ratio: float = 4.0,
                              attack_ms: float = 5.0,
                              release_ms: float = 50.0) -> np.ndarray:
    """
    Apply dynamic range compression.
    Studio DR ~66 dB → Podcast DR ~46 dB.
    """

    attack_coeff = np.exp(-1.0 / (attack_ms * sr / 1000))
    release_coeff = np.exp(-1.0 / (release_ms * sr / 1000))

    threshold_linear = 10 ** (threshold_db / 20)
    out = y.copy()
    envelope = 0.0

    for i in range(len(y)):
        abs_sample = abs(y[i])
        if abs_sample > envelope:
            envelope = attack_coeff * envelope + (1 - attack_coeff) * abs_sample
        else:
            envelope = release_coeff * envelope + (1 - release_coeff) * abs_sample

        if envelope > threshold_linear:
            gain_reduction = threshold_linear * (envelope / threshold_linear) ** (1 / ratio - 1)
            out[i] = y[i] * (gain_reduction / (envelope + 1e-10))

    # Makeup gain to match original peak
    peak_in = np.max(np.abs(y))
    peak_out = np.max(np.abs(out))
    if peak_out > 0:
        out = out * (peak_in / peak_out) * 0.9

    return out


def apply_lowpass_podcast(y: np.ndarray, sr: int,
                         cutoff_hz: float = 8000,
                         order: int = 6) -> np.ndarray:
    """
    Apply lowpass filter mimicking podcast's spectral cutoff.
    Analysis showed podcast drops sharply at 8 kHz, dead by 10 kHz.
    """
    nyq = sr / 2
    cutoff_norm = min(cutoff_hz / nyq, 0.99)
    sos = butter(order, cutoff_norm, btype='low', output='sos')
    return sosfiltfilt(sos, y)


def apply_downsampling(y: np.ndarray, sr: int,
                       target_sr: int = 24000) -> Tuple[np.ndarray, int]:
    """
    Downsample audio to target sample rate.
    Podcast is 24000 or 32000 Hz, studio is 44100 Hz.
    """
    if sr == target_sr:
        return y, sr

    # Anti-aliasing filter before downsampling
    nyq = target_sr / 2
    cutoff_norm = min(nyq / (sr / 2), 0.99)
    sos = butter(8, cutoff_norm, btype='low', output='sos')
    y_filtered = sosfiltfilt(sos, y)

    # Resample using rational approximation
    from fractions import Fraction
    frac = Fraction(target_sr, sr).limit_denominator(1000)
    y_resampled = resample_poly(y_filtered, frac.numerator, frac.denominator)

    return y_resampled, target_sr


def _pink_noise(n: int) -> np.ndarray:
    """Generate pink (1/f) noise via FFT shaping. Broadband, natural slope."""
    white = np.random.randn(n).astype(np.float32)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0
    spec = spec / np.sqrt(freqs)
    spec[0] = 0  # kill DC
    pink = np.fft.irfft(spec, n=n).astype(np.float32)
    return pink


def apply_noise_floor_raise(y: np.ndarray, sr: int,
                            target_floor_db: float = -48) -> np.ndarray:
    """
    Raise noise floor to podcast level with LOW-PASS-SHAPED noise.
    Why low-pass, not pink: MP3 psychoacoustic model aggressively removes
    broadband noise masked by speech, but low-frequency noise (0-3 kHz)
    SURVIVES the codec. An earlier experiment with pink noise confirmed this —
    the pink noise was killed by the codec → the augmented audio came out
    cleaner than the target.

    NOTE: this is the SYNTHETIC fallback. apply_real_noise_floor
    is preferred when a noise_bank is available — it carries real codec
    quantization micro-structure that the synthetic version can't reproduce.
    """
    noise_level = 10 ** (target_floor_db / 20)
    noise = np.random.randn(len(y)) * noise_level
    nyq = sr / 2
    sos = butter(2, min(3000 / nyq, 0.99), btype='low', output='sos')
    noise = sosfiltfilt(sos, noise) * 2  # compensate for filter attenuation
    noise = noise * (noise_level / (np.std(noise) + 1e-10))
    return y + noise


# ── Real noise transfer ─────────────────────────────────────────────────────
# Process-local cache of the noise bank. Loaded once on first use; subsequent
# calls reuse the cached arrays. Each worker in a multiproc calibration run
# gets its own copy (no shared memory needed — total bank is small, ≤2 GB).
_NOISE_BANK_CACHE = None


def _load_noise_bank(bank_dir: str = "noise_bank"):
    """Load noise bank index + audio segments lazily into a list of (sr, np.ndarray).
    Returns None if bank dir or index missing — caller falls back to synthetic.
    """
    global _NOISE_BANK_CACHE
    if _NOISE_BANK_CACHE is not None:
        return _NOISE_BANK_CACHE

    idx_path = os.path.join(bank_dir, "noise_bank_index.csv")
    if not os.path.isfile(idx_path):
        _NOISE_BANK_CACHE = []  # cache the negative result
        return None

    df = pd.read_csv(idx_path)
    segments = []
    for _, row in df.iterrows():
        path = os.path.join(bank_dir, row["path"])
        try:
            y, sr = sf.read(path, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            segments.append((int(sr), y))
        except Exception:
            continue
    print(f"[noise_bank] loaded {len(segments)} segments from {bank_dir}/")
    _NOISE_BANK_CACHE = segments
    return segments if segments else None


def apply_real_noise_floor(y: np.ndarray, sr: int,
                           target_floor_db: float = -48,
                           bank_dir: str = "noise_bank",
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Add real noise from podcast silence segments scaled to target floor dB.

    Preferred over the synthetic apply_noise_floor_raise. Real silence carries
    the actual codec quantization noise + mic hiss + room tone of real podcast
    audio, with full spectral micro-structure. Synthetic noise is uniform
    low-pass white — it adds an audible TV-static-like hiss that is absent from
    real recordings.

    Behavior:
      - Picks a random silence segment from bank
      - Resamples to target sr if needed
      - Loops/crops to match input length
      - Scales to target_floor_db RMS
      - Adds to signal

    Falls back to synthetic apply_noise_floor_raise if bank missing/empty.
    """
    bank = _load_noise_bank(bank_dir)
    if bank is None or not bank:
        return apply_noise_floor_raise(y, sr, target_floor_db=target_floor_db)

    if rng is None:
        rng = np.random.default_rng()

    # Pick random segment
    seg_sr, seg = bank[int(rng.integers(0, len(bank)))]

    # Resample to target sr if needed
    if seg_sr != sr:
        from fractions import Fraction
        frac = Fraction(sr, seg_sr).limit_denominator(1000)
        seg = resample_poly(seg, frac.numerator, frac.denominator).astype(np.float32)

    # Loop/crop to match input length
    if len(seg) < len(y):
        repeats = (len(y) // len(seg)) + 1
        seg = np.tile(seg, repeats)
    seg = seg[:len(y)]

    # Scale to target floor (RMS-based)
    seg_rms = float(np.sqrt(np.mean(seg ** 2)) + 1e-12)
    target_rms = 10 ** (target_floor_db / 20)
    seg = seg * (target_rms / seg_rms)

    return y + seg.astype(y.dtype)


def apply_speech_noise(y: np.ndarray, sr: int,
                       snr_db: float = 30,
                       noise_type: str = "pink") -> np.ndarray:
    """
    Inject broadband noise at signal level (SNR-based scaling).
    Present DURING speech - destroys HNR and harmonic periodicity.
    Without this op the augmented audio showed harmonic_degradation = 0.04 vs
    2.18 for real podcast (too clean harmonically; the codec alone doesn't break
    periodicity).

    Args:
        snr_db: target signal-to-noise ratio (20-35 realistic for podcast)
        noise_type: "pink" (default, 1/f) or "white"
    """
    n = len(y)
    if noise_type == "white":
        noise = np.random.randn(n).astype(np.float32)
    else:
        noise = _pink_noise(n)
    signal_power = float(np.mean(y.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2)) + 1e-12
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return y + noise.astype(y.dtype)


def apply_hf_instability(y: np.ndarray, sr: int,
                         strength: float = 0.5,
                         hf_cutoff_hz: float = 5000,
                         mod_rate_hz: float = 2.0) -> np.ndarray:
    """
    Simulate HF temporal instability (codec bitrate allocation jitter).
    Real podcast audio's C4 cluster has hf_temporal_std 5+ SD above studio baseline —
    no existing DSP op produces this. We apply a slow random gain modulation
    to the HF band, mimicking the flicker that appears when a codec starves
    HF bits frame-to-frame.

    Parameters:
        strength:     modulation depth 0..1 (1.0 = full mute possible)
        hf_cutoff_hz: split signal at this frequency
        mod_rate_hz:  how fast HF level fluctuates (1..5 Hz is realistic)
    """
    nyq = sr / 2
    cutoff_norm = min(hf_cutoff_hz / nyq, 0.99)
    sos_hp = butter(4, cutoff_norm, btype='high', output='sos')
    hf = sosfiltfilt(sos_hp, y)
    lf = y - hf

    n = len(y)
    mod_samples = max(2, int(sr / mod_rate_hz))
    n_points = n // mod_samples + 2
    gain_points = np.random.uniform(1.0 - strength, 1.0, n_points)

    xp = np.arange(n_points) * mod_samples
    gain = np.interp(np.arange(n), xp, gain_points)

    smooth_hz = max(mod_rate_hz * 3, 1.0)
    sos_smooth = butter(2, min(smooth_hz / nyq, 0.99), btype='low', output='sos')
    gain = sosfiltfilt(sos_smooth, gain)

    return lf + hf * gain


# =============================================================================
# URGENT/Sidon-family degradations that the retrieval palette does not produce.
# They cover OOD phone artifacts: network packet
# loss, room reverb, ADC/mic hard-clipping. Applied as low-probability add-ons in
# PodcastAugmentorV4.augment() — NOT in the retrieval palette — so they widen
# coverage without disturbing the podcast-matched base distribution.
# Acoustic ordering: reverb + hard_clip are capture-side (BEFORE the chain),
# packet_loss is network-side (AFTER the chain).
# =============================================================================

def apply_packet_loss(y: np.ndarray, sr: int,
                      coverage: float = 0.06,
                      min_ms: float = 40.0, max_ms: float = 250.0,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Zero out short bursts to mimic VoIP packet loss (real podcast audio phone artifact).

    coverage: approximate fraction of total duration to drop (podcast-modest 4-8%;
              Sidon uses ~10%). Bursts are min_ms..max_ms long.
    """
    r = rng if rng is not None else np.random.default_rng()
    out = y.copy()
    n = len(y)
    dur = n / sr
    if dur <= max_ms / 1000:
        return out
    n_bursts = max(1, int(dur * coverage / ((min_ms + max_ms) / 2 / 1000)))
    for _ in range(n_bursts):
        burst_ms = r.uniform(min_ms, max_ms)
        burst_n = int(burst_ms / 1000 * sr)
        if burst_n >= n:
            continue
        st = int(r.uniform(0, n - burst_n))
        out[st:st + burst_n] = 0.0
    return out


def _synth_rir(sr: int, rt60: float, rng: np.random.Generator) -> np.ndarray:
    """Dependency-free synthetic room impulse response (no pyroomacoustics).
    Direct path + exponentially-decaying diffuse tail (Schroeder-style).
    rt60 in seconds (-60 dB decay time). Small for phone/room (0.1-0.6s).
    """
    L = max(8, int(rt60 * sr))
    t = np.arange(L) / sr
    decay = np.exp(-6.908 * t / rt60)  # reaches -60 dB at t=rt60
    rir = rng.standard_normal(L).astype(np.float32) * decay.astype(np.float32)
    rir[0] += 1.0  # direct path (dominant) → bounded wet/dry mix
    m = np.abs(rir).max()
    if m > 0:
        rir = rir / m
    return rir


def apply_reverb(y: np.ndarray, sr: int,
                 rt60: float | None = None,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Convolve with a synthetic RIR (room/phone reverberation)."""
    r = rng if rng is not None else np.random.default_rng()
    if rt60 is None:
        rt60 = float(r.uniform(0.1, 0.6))
    rir = _synth_rir(sr, rt60, r)
    wet = np.convolve(y.astype(np.float32), rir)[: len(y)]
    m = np.abs(wet).max()
    if m > 0:
        wet = wet / m * (np.abs(y).max() + 1e-9)  # preserve input level
    return wet.astype(np.float32)


def apply_hard_clip(y: np.ndarray,
                    min_quantile: float | None = None,
                    max_quantile: float | None = None,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Hard quantile clipping (ADC/mic overload). Distinct from the soft tanh
    saturation in apply_chain's loudness norm — this introduces real flat-top
    distortion + odd harmonics. min_q in [0,0.1], max_q in [0.9,1.0] (Sidon range).
    """
    r = rng if rng is not None else np.random.default_rng()
    if min_quantile is None:
        min_quantile = float(r.uniform(0.0, 0.1))
    if max_quantile is None:
        max_quantile = float(r.uniform(0.9, 1.0))
    lo, hi = np.quantile(y, [min_quantile, max_quantile])
    return np.clip(y, lo, hi).astype(np.float32)
