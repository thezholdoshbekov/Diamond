"""DSP primitives modelling the real podcast recording chain.

    Voice → Microphone → Room → Cable/Interface → DAW → Export (codec)

Measured on 3000 real podcast samples: degradations are not random. Effectively
all samples carry mic coloration and codec artifacts, ~99% are band-limited, and
~22% have poor SNR — while clipping and white noise are essentially absent.

The primitives here are the ones that survived into the calibrated palette:
mic coloration, real codec encode/decode (ffmpeg), and SNR degradation. They are
consumed by `podcast_augmentor.py` and, through it, by `dsp_chain.apply_chain` —
this module is not an entry point.
"""

import numpy as np
import subprocess
import tempfile
import os
import soundfile as sf
from scipy.signal import lfilter
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# PART 1: INDIVIDUAL DEGRADATIONS
#
# Each one is a function that takes audio + parameters and returns degraded audio.
# No magic, no hidden state. Pure functions.
# =============================================================================


def apply_mic_coloration(y: np.ndarray, sr: int,
                         n_bands: int = 4,
                         freqs: list = None,
                         gains_db: list = None,
                         qs: list = None) -> np.ndarray:
    """
    Simulate cheap microphone frequency response.

    WHAT HAPPENS IN REALITY:
        Every microphone has an imperfect frequency response. Cheap mics have:
        - A "presence peak" at 2-5 kHz (makes voice sound harsh)
        - Bass roll-off or proximity boost below 300 Hz
        - Resonances at random frequencies (body vibration, capsule modes)

        A $50 USB mic and a $3000 Neumann U87 sound completely different
        on the SAME voice — this is why.

    HOW WE SIMULATE IT:
        Apply 3-6 parametric EQ bands at random frequencies with random gains.
        This creates the same kind of uneven frequency response.

    Parameters:
        n_bands:  number of EQ bands (3-6 is realistic)
        freqs:    center frequencies in Hz (if None, random)
        gains_db: boost/cut in dB per band (if None, random from [-6, +8])
        qs:       Q factors per band (if None, random from [0.5, 3.0])
    """
    out = y.copy()

    if freqs is None:
        freqs = np.random.uniform(200, 8000, n_bands)
    if gains_db is None:
        # Cheap mics boost more than they cut — hence asymmetric range
        gains_db = np.random.uniform(-6, 8, n_bands)
    if qs is None:
        qs = np.random.uniform(0.5, 3.0, n_bands)

    for freq, gain, q in zip(freqs, gains_db, qs):
        # Skip if frequency is above Nyquist
        if freq >= sr / 2:
            continue

        # Peaking EQ filter (same as parametric EQ in any DAW)
        w0 = 2 * np.pi * freq / sr
        A = 10 ** (gain / 40)  # amplitude
        alpha = np.sin(w0) / (2 * q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        b = np.array([b0 / a0, b1 / a0, b2 / a0])
        a = np.array([1.0, a1 / a0, a2 / a0])

        out = lfilter(b, a, out)

    return out


# Maps a CBR target bitrate -> LAME VBR quality scale (0=best..9=worst) for the
# "vbr" engine variant of the multi-encoder pool. LAME's VBR
# allocates bits psychoacoustically rather than at a constant rate, so the
# fingerprint differs from CBR even at the same nominal bitrate.
_LAME_VBR_Q_MAP = {16: 9, 24: 9, 32: 9, 48: 8, 64: 7, 96: 5, 128: 4}


def _build_encode_cmd(codec: str, bitrate: int, sr: int, engine: str,
                      in_path: str, out_path: str) -> list:
    """Construct ffmpeg encode command for (codec, engine) combination.

    For mp3 there are 4 engine variants — default / strict / fast / vbr — each
    leaving a distinct bit-allocation fingerprint while using the same encoder
    binary (libmp3lame). For aac/ogg the engine arg is currently a label only
    (single backend); kept in the signature so future variants can plug in.
    """
    cmd = ['ffmpeg', '-y', '-i', in_path]

    if codec == 'mp3':
        if engine == 'vbr':
            q = _LAME_VBR_Q_MAP.get(int(bitrate), 4)
            cmd += ['-c:a', 'libmp3lame', '-q:a', str(q)]
        else:
            cmd += ['-c:a', 'libmp3lame', '-b:a', f'{bitrate}k']
            if engine == 'strict':
                cmd += ['-compression_level', '0']  # best/slowest
            elif engine == 'fast':
                cmd += ['-compression_level', '9']  # fastest
            # 'default' -> no extra flags (ffmpeg's compression_level=4)
    elif codec == 'aac':
        cmd += ['-c:a', 'aac', '-b:a', f'{bitrate}k']
    elif codec == 'ogg':
        cmd += ['-c:a', 'libvorbis', '-b:a', f'{bitrate}k']
    else:
        raise ValueError(f"Unknown codec: {codec!r}")

    cmd += ['-ar', str(sr), out_path]
    return cmd


def apply_codec(y: np.ndarray, sr: int,
                codec: str = None,
                bitrate: int = None,
                n_passes: int = None,
                engine: str = "default") -> np.ndarray:
    """
    Apply real codec artifacts by encoding/decoding through ffmpeg.

    WHAT HAPPENS IN REALITY:
        Most podcast audio goes through multiple encoding stages:
        1. DAW exports to WAV or high-bitrate MP3
        2. Upload platform re-encodes (YouTube = AAC 128, Spotify = OGG 96)
        3. Sometimes downloaded and re-uploaded = triple encoding

        Each pass adds more artifacts: pre-echo, spectral holes,
        bandwidth reduction, quantization noise.

    DATA SAYS:
        100% of samples show codec artifacts.
        mel_cutoff_sharpness is very negative (sharp spectral cutoff).
        spectral_holes_count = 17-18 (many empty sub-bands).
        pre_echo_ratio elevated.

    REQUIRES: ffmpeg installed and on PATH.

    Parameters:
        codec:    'mp3', 'aac', or 'ogg'
        bitrate:  in kbps (32-192)
        n_passes: how many times to encode/decode (1-3)
        engine:   for mp3, one of 'default'/'strict'/'fast'/'vbr' — picks
                  different LAME compression flags so the same encoder binary
                  produces 4 distinct fingerprints. Ignored for aac/ogg
                  (a single backend each).
    """
    if codec is None:
        codec = np.random.choice(['mp3', 'aac', 'ogg'], p=[0.5, 0.3, 0.2])
    if bitrate is None:
        bitrate = np.random.choice([32, 48, 64, 96, 128], p=[0.1, 0.2, 0.3, 0.25, 0.15])
    if n_passes is None:
        n_passes = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])

    codec_ext = {'mp3': '.mp3', 'aac': '.m4a', 'ogg': '.ogg'}
    if codec not in codec_ext:
        raise ValueError(f"Unknown codec: {codec!r}")

    # Per-codec bitrate floor. Real podcast audio at 16-24 kbps mp3 is realistic
    # (LAME survives extreme low-bitrate by aggressive bandlimiting). AAC
    # and OGG below ~32 kbps produce audible hiss/swirl artifacts that don't
    # appear in real podcast audio — e.g. aac@25 kbps with a 10 kHz lowpass
    # yields broadband HF artifacts.
    if codec in ('aac', 'ogg') and int(bitrate) < 32:
        bitrate = 32

    current = y.copy()

    for pass_i in range(n_passes):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_in  = os.path.join(tmpdir, 'in.wav')
            encoded = os.path.join(tmpdir, f'out{codec_ext[codec]}')
            wav_out = os.path.join(tmpdir, 'out.wav')

            sf.write(wav_in, current, sr)

            cmd_encode = _build_encode_cmd(codec, int(bitrate), sr, engine,
                                            wav_in, encoded)
            cmd_decode = [
                'ffmpeg', '-y', '-i', encoded,
                '-c:a', 'pcm_f32le',
                '-ar', str(sr),
                wav_out
            ]

            try:
                subprocess.run(cmd_encode, capture_output=True, check=True)
                subprocess.run(cmd_decode, capture_output=True, check=True)
                current, _ = sf.read(wav_out, dtype='float32')
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                warnings.warn(f"ffmpeg failed on pass {pass_i+1}: {e}. "
                              f"Returning audio without this codec pass.")
                break

        # Vary bitrate slightly between passes (realistic)
        if pass_i < n_passes - 1:
            bitrate = int(bitrate * np.random.uniform(0.8, 1.5))
            bitrate = max(32, min(192, bitrate))

    # Match length
    if len(current) > len(y):
        current = current[:len(y)]
    elif len(current) < len(y):
        current = np.pad(current, (0, len(y) - len(current)))

    return current
