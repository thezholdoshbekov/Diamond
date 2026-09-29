"""Robustness augmentation track: RIR reverb (pre-codec) + broadband noise
(post-codec), applied AROUND the podcast augmentor without modifying it.

Rationale + placement (see also configs/default.yaml `data.noise_aug`):
  • RIR is convolved on the clean/degraded input BEFORE the codec stage —
    reverb reshapes the envelope and survives codec compression.
  • Broadband noise (MUSAN + DNS) is mixed AFTER the codec — pre-codec broadband
    is stripped by MP3 psychoacoustic masking, so post-codec is the only place
    the model can actually see (and learn to remove) it.
The supervised target stays the dry clean studio tokens, so the model learns
dereverberation + denoising, not noise reproduction.

Corpus layout expected under noise_dir / rir_dir (built by scripts/download_noise.sh):
  noise_dir/musan_noise/noise/**/*.wav
  noise_dir/DNS-Challenge/datasets/noise/*.wav
  rir_dir/simulated_rirs/**/*.wav, rir_dir/real_rirs_isotropic_noises/*.wav
Missing dirs/files degrade gracefully to a no-op (logged once).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Module-level file-list cache so each DataLoader worker scans a corpus dir only
# once (rglob over ~65k DNS wavs is not free). Keyed by resolved dir string.
_FILE_LIST_CACHE: dict[str, list[str]] = {}


def _scan_wavs(root: str | None, exclude_root: str | None = None,
               exclude_substrings: tuple[str, ...] = ()) -> list[str]:
    """Recursively list *.wav under `root`, excluding anything under
    `exclude_root` or whose path contains any of `exclude_substrings`.

    `exclude_root` keeps the RIR corpus out of the noise pool (it lives inside
    the same parent). `exclude_substrings` drops non-noise / never-fetched dirs
    that ship with the DNS-Challenge clone (datasets/clean, test_set,
    blind_test_set) — those are clean speech and git-lfs pointers we never pull,
    so they'd otherwise pollute the noise pool and fail to open at runtime."""
    if not root:
        return []
    root = str(Path(root).resolve())
    excl = str(Path(exclude_root).resolve()) if exclude_root else None
    key = f"{root}|{excl}|{','.join(exclude_substrings)}"
    if key in _FILE_LIST_CACHE:
        return _FILE_LIST_CACHE[key]
    p = Path(root)
    files = []
    if p.exists():
        for f in p.rglob("*.wav"):
            fp = str(f)
            if excl and fp.startswith(excl):
                continue
            if any(sub in fp for sub in exclude_substrings):
                continue
            files.append(fp)
    if not files:
        print(f"[noise_aug] WARN: no .wav under {root} — that track disabled")
    _FILE_LIST_CACHE[key] = files
    return files


# DNS-Challenge clone ships clean speech + held-out test sets alongside the
# noise set; only datasets/noise/ is broadband noise. Exclude the rest.
_NOISE_EXCLUDE_SUBSTRINGS = (
    "/datasets/clean/",
    "/datasets/test_set/",
    "/datasets/blind_test_set/",
)


def _load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return np.ascontiguousarray(wav, dtype=np.float32), int(sr)


def _resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or wav.size == 0:
        return wav
    import librosa
    return librosa.resample(wav, orig_sr=sr_in, target_sr=sr_out).astype(np.float32)


class NoiseRIRAugmentor:
    """Lazy, process-local pools of RIRs and noise wavs. One per worker.

    Methods are no-ops (return input unchanged) when their corpus is empty, so
    enabling the track before the download finishes can't crash training.
    """

    def __init__(self, noise_dir: str | None, rir_dir: str | None,
                 noise_prob: float, rir_prob: float,
                 snr_db_range: tuple[float, float], seed: int | None = None):
        # Exclude the RIR corpus (impulses, not broadband) and the DNS clean/
        # test dirs from the noise pool.
        self.noise_files = _scan_wavs(noise_dir, exclude_root=rir_dir,
                                      exclude_substrings=_NOISE_EXCLUDE_SUBSTRINGS)
        self.rir_files = _scan_wavs(rir_dir)
        self.noise_prob = float(noise_prob) if self.noise_files else 0.0
        self.rir_prob = float(rir_prob) if self.rir_files else 0.0
        self.snr_lo, self.snr_hi = float(snr_db_range[0]), float(snr_db_range[1])
        self.rng = np.random.default_rng(seed)

    # ── RIR (pre-codec) ──────────────────────────────────────────────────────
    def maybe_apply_rir(self, wav: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Returns (wav, info). info records what was applied so spike forensics
        (spike_events.jsonl) can blame a specific RIR if it triggers a grad spike."""
        if self.rir_prob <= 0.0 or self.rng.random() >= self.rir_prob:
            return wav, {"rir": False}
        try:
            rir_path = self.rir_files[self.rng.integers(len(self.rir_files))]
            rir, rir_sr = _load_wav_mono(rir_path)
            rir = _resample(rir, rir_sr, sr)
            if rir.size == 0 or not np.isfinite(rir).all():
                return wav, {"rir": False}
            peak = np.abs(rir).max()
            if peak < 1e-8:
                return wav, {"rir": False}
            rir = rir / peak
            from scipy.signal import fftconvolve
            in_peak = float(np.abs(wav).max())
            wet = fftconvolve(wav, rir, mode="full")[: len(wav)].astype(np.float32)
            # preserve original loudness (convolution otherwise rescales)
            wet_peak = float(np.abs(wet).max())
            if wet_peak > 1e-8 and in_peak > 0:
                wet = wet * (in_peak / wet_peak)
            return wet, {"rir": True, "rir_file": Path(rir_path).name}
        except Exception as e:  # noqa: BLE001 — any failure → skip reverb for this sample
            print(f"[noise_aug] RIR skipped: {e}")
            return wav, {"rir": False, "rir_error": str(e)[:60]}

    # ── Broadband noise (post-codec) ─────────────────────────────────────────
    def maybe_mix_noise(self, wav: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """Returns (wav, info). info records the noise file + SNR so a spike can
        be traced to a specific noise clip / level."""
        if self.noise_prob <= 0.0 or self.rng.random() >= self.noise_prob:
            return wav, {"noise": False}
        try:
            noise_path = self.noise_files[self.rng.integers(len(self.noise_files))]
            noise, n_sr = _load_wav_mono(noise_path)
            noise = _resample(noise, n_sr, sr)
            if noise.size == 0:
                return wav, {"noise": False}
            # tile / random-crop noise to match signal length
            if len(noise) < len(wav):
                reps = int(np.ceil(len(wav) / len(noise)))
                noise = np.tile(noise, reps)
            start = int(self.rng.integers(0, len(noise) - len(wav) + 1))
            noise = noise[start : start + len(wav)]

            sig_p = float(np.mean(wav.astype(np.float64) ** 2))
            noise_p = float(np.mean(noise.astype(np.float64) ** 2))
            if sig_p < 1e-10 or noise_p < 1e-10:
                return wav, {"noise": False}
            snr_db = float(self.rng.uniform(self.snr_lo, self.snr_hi))
            target_noise_p = sig_p / (10.0 ** (snr_db / 10.0))
            noise = noise * np.sqrt(target_noise_p / noise_p)
            out = (wav + noise).astype(np.float32)
            # guard against clipping; keep peak ≤ 0.99 like the augmentor
            peak = float(np.abs(out).max())
            if peak > 0.99:
                out = out * (0.99 / peak)
            return out, {"noise": True, "noise_file": Path(noise_path).name,
                         "noise_snr_db": round(snr_db, 2)}
        except Exception as e:  # noqa: BLE001
            print(f"[noise_aug] noise mix skipped: {e}")
            return wav, {"noise": False, "noise_error": str(e)[:60]}


_SINGLETON: NoiseRIRAugmentor | None = None


def get_noise_augmentor(cfg_noise) -> NoiseRIRAugmentor | None:
    """Build (once per process) the noise/RIR augmentor from a `data.noise_aug`
    config node, or return None if disabled."""
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON
    if cfg_noise is None or not bool(getattr(cfg_noise, "enabled", False)):
        return None
    _SINGLETON = NoiseRIRAugmentor(
        noise_dir=getattr(cfg_noise, "noise_dir", None),
        rir_dir=getattr(cfg_noise, "rir_dir", None),
        noise_prob=getattr(cfg_noise, "noise_prob", 0.15),
        rir_prob=getattr(cfg_noise, "rir_prob", 0.15),
        snr_db_range=tuple(getattr(cfg_noise, "snr_db_range", (0.0, 20.0))),
    )
    n_noise, n_rir = len(_SINGLETON.noise_files), len(_SINGLETON.rir_files)
    print(f"[noise_aug] enabled: {n_noise} noise wavs (p={_SINGLETON.noise_prob}), "
          f"{n_rir} RIRs (p={_SINGLETON.rir_prob}), SNR [{_SINGLETON.snr_lo},{_SINGLETON.snr_hi}] dB")
    return _SINGLETON
