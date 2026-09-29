"""
Per-sample retrieval augmentor.

For each augment call:
  1. Pick random real podcast audio profile from target pool (bad+borderline only)
  2. k-NN retrieve nearest combo from calibration palette in 17-D feature space
  3. Apply that combo's DSP chain to the studio sample

Validated empirically: mean retrieval distance 3.84 in scaled 17-D space, below
the intra-podcast spread (4.01) — i.e. at the physical floor of the problem.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dsp_chain import apply_chain
from podcast_augmentor import apply_packet_loss, apply_reverb, apply_hard_clip

PALETTE_CSV = "calibration_results.csv"
# Default target = per-sample bad+borderline podcast (matches the actual
# distribution model will see at inference, including phonetic+session noise).
# Listening tests confirmed per-sample targets sound CLOSER to real podcast audio
# than a speaker-mean target (which oversmoothed the degradation).
# To use speaker-mean (softer, fewer outliers), pass
# `targets_csv="speaker_mean_target_50k.csv"` and `target_filter=()`.
TARGETS_CSV = "podcast_target_pool.csv"  # 96,899 bad+borderline real profiles

# Full 17-feature set — carries speaker timbre, see MATCH_FEATURES_CHANNEL below.
MATCH_FEATURES_FULL = [
    "snr_estimate_db", "dynamic_range_db", "noise_floor_db",
    "spectral_centroid_mean", "spectral_bandwidth_mean", "spectral_rolloff_mean",
    "hnr_db", "harmonic_ratio",
    "band_300_3400_ratio", "band_3400_8000_ratio", "band_8k_plus_ratio",
    "spectral_flatness_mean",
    "hf_energy_ratio",
    "proximity_bass_boost_db",
    "mic_resonance_peak_db",
    "crest_factor",
    "hf_temporal_std",
]

# Channel-only subset — kills speaker-timbre contamination from retrieval metric.
# Picked from MATCH_FEATURES_FULL by intra-speaker correlation: 5 strict-channel
# features + 3 channel-dominated ones; the 9 speaker/mixed features are dropped.
# A few further strict-channel features exist in the corpus but are absent from
# calibration_results.csv (the palette predates them); rebuilding the palette to
# include them is a separate task.
MATCH_FEATURES_CHANNEL = [
    "snr_estimate_db", "dynamic_range_db", "noise_floor_db",
    "band_8k_plus_ratio", "hf_temporal_std",
    "spectral_rolloff_mean", "band_3400_8000_ratio", "hf_energy_ratio",
]

PARAM_COLS = [
    "target_sr",
    "lowpass_hz", "lowpass_order",
    "compression_threshold", "compression_ratio",
    "bass_boost_db", "bass_cutoff_hz",
    "noise_target_db",
    "codec_type", "codec_bitrate", "codec_passes", "codec_engine",
    "mic_bands",
    "hf_instability_strength", "hf_instability_cutoff_hz", "hf_instability_mod_hz",
    "speech_noise_snr_db",
]


class PodcastAugmentorV4:
    # Continuous params subject to jitter and their safe (low, high) bounds.
    # Discrete params (target_sr, lowpass_order, codec_type, codec_passes,
    # mic_bands) are NOT jittered — they would break the DSP chain.
    JITTER_BOUNDS = {
        # Tightened from (2000, 14000) to (4000, 13500): listening found clips
        # whose lowpass jittered down to ~3.2-3.5 kHz (phone-band) and came out
        # thin and muffled. Real podcast audio is 5-10 kHz typical; a 4000 Hz
        # floor keeps voice intelligibility.
        "lowpass_hz":               (4000.0, 13500.0),
        # Tightened from (-45, -6) to (-38, -10): a threshold jittered to -39 dB
        # produced catastrophic compression, RMS -34.7 dB (far below real podcast
        # audio at ~-16 dB). The palette has thresholds in [-35, -12]; this bound
        # keeps tail expansion from falling into "all-silenced" territory.
        "compression_threshold":    (-38.0, -10.0),
        "compression_ratio":        (2.0, 15.0),
        "bass_boost_db":            (0.5, 8.0),
        "bass_cutoff_hz":           (400.0, 2000.0),
        # Upper bound tightened from -30 to -42: a noise target jittered into the
        # -30..-39 range produced audible background hiss, whereas the real
        # podcast noise floor sits around -43 to -50 dB. -42 keeps the jitter
        # slightly wider than the corpus P95 but stays out of the hiss zone.
        "noise_target_db":          (-65.0, -42.0),
        # Tightened from (8, 256) to (16, 256). Real podcast audio rarely below
        # 16 kbps; 8 kbps mp3 is unlistenable.
        "codec_bitrate":            (16.0, 256.0),
        "hf_instability_strength":  (0.2, 1.0),
        "hf_instability_cutoff_hz": (3000.0, 8000.0),
        "hf_instability_mod_hz":    (0.5, 5.0),
        "speech_noise_snr_db":      (15.0, 50.0),
    }
    JITTER_INT_PARAMS = {"codec_bitrate", "bass_cutoff_hz", "lowpass_hz",
                         "noise_target_db", "hf_instability_cutoff_hz",
                         "speech_noise_snr_db"}

    # Retrieval distance ceiling. Above this, target is in palette gap; we
    # fall back to random_palette rather than picking a forced-bad combo.
    # Tuned by listening: typical distance is 0.4-1.5, and a target retrieved at
    # 10.12 produced audio that did not sound like podcast at all. 5.0 is a safe
    # upper bound.
    MAX_RETRIEVAL_DIST = 5.0

    def __init__(
        self,
        palette_csv: str = PALETTE_CSV,
        targets_csv: str = TARGETS_CSV,
        target_filter: tuple = ("bad", "borderline"),
        clean_pass_prob: float = 0.05,
        seed: int | None = None,
        feature_set: str = "channel",
        jitter: float = 0.40,
        strategy_mix: tuple = (0.60, 0.25, 0.15),
        packet_loss_prob: float = 0.15,
        reverb_prob: float = 0.10,
        hard_clip_prob: float = 0.10,
    ):
        """
        strategy_mix: (retrieve, random_palette, random_synth) probabilities.
            retrieve       — k-NN combo for target's channel features (default 60%)
            random_palette — uniform random combo from palette (no retrieval bias, 25%)
            random_synth   — fresh combo from generate_param_combos + wider jitter (15%)

        Hybrid sampling stops the model from memorizing specific retrieval
        signatures. Pure retrieval yields only ~400-1000
        unique DSP signatures regardless of target pool size. Hybrid adds
        random_palette (~5000 distinct combos) and random_synth (effectively
        infinite via on-the-fly generation + wider jitter).
        """
        self.rng = np.random.default_rng(seed)
        self.clean_pass_prob = clean_pass_prob
        self.jitter = jitter
        # OOD phone-artifact add-ons. Podcast-modest probabilities —
        # widen coverage (network/room/ADC distortions) without dominating the
        # retrieval-matched base. Applied around apply_chain in augment().
        self.packet_loss_prob = packet_loss_prob
        self.reverb_prob = reverb_prob
        self.hard_clip_prob = hard_clip_prob

        if abs(sum(strategy_mix) - 1.0) > 1e-6:
            raise ValueError(f"strategy_mix must sum to 1.0, got {strategy_mix} -> {sum(strategy_mix)}")
        self.strategy_mix = tuple(strategy_mix)
        self.jitter_synth = jitter * 1.5  # wider for random_synth (domain randomization)

        if feature_set == "channel":
            requested = MATCH_FEATURES_CHANNEL
        elif feature_set == "full":
            requested = MATCH_FEATURES_FULL
        elif isinstance(feature_set, (list, tuple)):
            requested = list(feature_set)
        else:
            raise ValueError(f"feature_set must be 'channel', 'full', or a list; got {feature_set!r}")

        palette = pd.read_csv(palette_csv)
        avail = [f for f in requested if f in palette.columns]
        if len(avail) < len(requested):
            missing = set(requested) - set(avail)
            raise ValueError(f"Palette missing features: {missing}")
        self.match_features = avail
        self.feature_set = feature_set

        avail_params = [p for p in PARAM_COLS if p in palette.columns]
        # codec_engine (the multi-encoder pool) was added after the first
        # palettes were built. Tolerate older palettes — apply_codec falls back to
        # engine="default" when the param dict lacks the key, which matches
        # legacy single-engine behavior.
        OPTIONAL_PARAMS = {"codec_engine"}
        required_missing = (set(PARAM_COLS) - set(avail_params)) - OPTIONAL_PARAMS
        if required_missing:
            raise ValueError(f"Palette missing required params: {required_missing}")
        self.param_cols = avail_params

        targets_full = pd.read_csv(targets_csv)
        if "__verdict" in targets_full.columns:
            targets = targets_full[targets_full["__verdict"].isin(target_filter)]
        else:
            targets = targets_full
        targets = targets.dropna(subset=self.match_features)
        if len(targets) == 0:
            raise ValueError(f"No targets after filtering by {target_filter}")

        self.palette = palette.reset_index(drop=True)
        self.targets = targets.reset_index(drop=True)

        P = self.palette[self.match_features].fillna(0).values
        T = self.targets[self.match_features].fillna(0).values
        self.scaler = StandardScaler().fit(np.vstack([P, T]))
        self._palette_scaled = self.scaler.transform(P)

        print(f"[augmentor] palette: {len(self.palette)} combos | "
              f"targets: {len(self.targets)} (filter={target_filter}) | "
              f"feature_set={feature_set!r} ({len(self.match_features)} features)")

    def _retrieve_combo(self, target_features: np.ndarray) -> dict:
        """Return param dict of nearest combo in scaled feature space."""
        target_sc = self.scaler.transform(target_features.reshape(1, -1))[0]
        dists = np.sqrt(((self._palette_scaled - target_sc) ** 2).sum(axis=1))
        best_idx = int(np.argmin(dists))
        row = self.palette.iloc[best_idx]
        params = {}
        for col in self.param_cols:
            v = row[col]
            if pd.isna(v):
                params[col] = 0
            elif col == "codec_type":
                params[col] = str(v) if v else "none"
            elif col in ("hf_instability_strength", "hf_instability_mod_hz"):
                params[col] = float(v)
            else:
                try:
                    params[col] = int(v) if v == int(v) else float(v)
                except (ValueError, TypeError):
                    params[col] = v
        return params, best_idx, float(dists[best_idx])

    def _jitter_params(self, params: dict, jitter: float | None = None) -> dict:
        """Apply ±jitter% noise to continuous params. Disabled (=0) params stay 0.
        Each call returns a new dict so retrieval combo is reused without mutation.
        Anti-overfit measure: turns each combo from a point into a cloud, so the
        model can't memorize specific DSP signatures.

        jitter override: lets hybrid sampling apply wider jitter to
        random_synth combos (domain randomization).
        """
        j = self.jitter if jitter is None else jitter
        if j <= 0:
            return dict(params)
        out = dict(params)
        for key, (lo, hi) in self.JITTER_BOUNDS.items():
            v = out.get(key, 0)
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v == 0:
                continue
            sign = 1.0 if v > 0 else -1.0
            mag = abs(v)
            delta = mag * j
            new_v = sign * (mag + self.rng.uniform(-delta, delta))
            new_v = max(lo, min(hi, new_v))
            if key in self.JITTER_INT_PARAMS:
                new_v = int(round(new_v))
            out[key] = new_v
        return out

    def _row_to_params(self, row) -> dict:
        """Convert a palette row (pd.Series) to params dict, same as _retrieve_combo."""
        params = {}
        for col in self.param_cols:
            v = row[col]
            if pd.isna(v):
                params[col] = 0
            elif col == "codec_type":
                params[col] = str(v) if v else "none"
            elif col in ("hf_instability_strength", "hf_instability_mod_hz"):
                params[col] = float(v)
            else:
                try:
                    params[col] = int(v) if v == int(v) else float(v)
                except (ValueError, TypeError):
                    params[col] = v
        return params

    def _random_palette_combo(self) -> tuple:
        """Pick a random combo from palette (uniform, no retrieval bias)."""
        idx = int(self.rng.integers(0, len(self.palette)))
        return self._row_to_params(self.palette.iloc[idx]), idx

    def _random_synth_combo(self) -> dict:
        """Generate a fresh combo via dsp_chain.generate_param_combos.
        Uses the weighted parameter distributions — same as palette sampling
        but bypasses the palette's finiteness."""
        # Lazy import to avoid circular dependency at module load
        from dsp_chain import generate_param_combos
        # generate_param_combos uses np.random; reseed from our rng for reproducibility
        seed = int(self.rng.integers(0, 2**31 - 1))
        np.random.seed(seed)
        return generate_param_combos(1)[0]

    def augment(self, y: np.ndarray, sr: int):
        """Augment one studio sample. Returns (audio, new_sr, info).

        Hybrid sampling:
          - Roll dice on strategy_mix
          - retrieve: target-driven k-NN combo (matches podcast distribution)
          - random_palette: uniform combo from palette (no retrieval bias)
          - random_synth: on-the-fly combo + wider jitter (domain randomization)
        Each path then goes through jitter and apply_chain.
        """
        if self.rng.random() < self.clean_pass_prob:
            target_sr = int(self.rng.choice([24000, 32000]))
            from podcast_augmentor import apply_downsampling
            out, out_sr = apply_downsampling(y, sr, target_sr=target_sr)
            return out, out_sr, {"clean_pass": True, "strategy": "clean_pass"}

        target_idx = int(self.rng.integers(0, len(self.targets)))
        target_row = self.targets.iloc[target_idx]
        target_features = target_row[self.match_features].fillna(0).values.astype(float)

        strategy = str(self.rng.choice(
            ["retrieve", "random_palette", "random_synth"],
            p=self.strategy_mix,
        ))

        retrieval_dist = None
        retrieval_fallback = False
        if strategy == "retrieve":
            params, combo_idx, retrieval_dist = self._retrieve_combo(target_features)
            jitter_used = self.jitter
            # Outlier-target safeguard: if the closest combo is still far in
            # scaled feature space, retrieval has effectively failed (target
            # is in a region the palette doesn't cover). Listening found a target
            # retrieved at dist=10.12 (vs a typical 0.4-1.5) whose output didn't
            # sound like podcast at all. Falling back to random_palette here
            # gives a believable combo instead of a forced bad match.
            if retrieval_dist > self.MAX_RETRIEVAL_DIST:
                params, combo_idx = self._random_palette_combo()
                retrieval_fallback = True
        elif strategy == "random_palette":
            params, combo_idx = self._random_palette_combo()
            jitter_used = self.jitter
        else:  # random_synth
            params = self._random_synth_combo()
            combo_idx = -1  # synthetic, not from palette
            jitter_used = self.jitter_synth  # wider jitter for domain randomization

        params = self._jitter_params(params, jitter=jitter_used)

        # OOD add-ons — capture-side BEFORE the chain (acoustic ordering):
        # reverb (room) then hard-clip (mic/ADC overload), so the codec in
        # apply_chain processes the already-distorted signal realistically.
        ood = []
        y_in = y
        if self.rng.random() < self.reverb_prob:
            y_in = apply_reverb(y_in, sr, rng=self.rng); ood.append("reverb")
        if self.rng.random() < self.hard_clip_prob:
            y_in = apply_hard_clip(y_in, rng=self.rng); ood.append("hard_clip")

        out, out_sr = apply_chain(y_in, sr, params)

        # network-side AFTER the chain: packet loss (post-codec transmission).
        if self.rng.random() < self.packet_loss_prob:
            out = apply_packet_loss(out, out_sr, rng=self.rng); ood.append("packet_loss")

        info = {
            "clean_pass": False,
            "strategy": strategy + ("+fallback" if retrieval_fallback else ""),
            "target_idx": target_idx,
            "combo_idx": combo_idx,
            "retrieval_dist": retrieval_dist,
            "retrieval_fallback": retrieval_fallback,
            "jitter": jitter_used,
            "params": params,
            "ood_augs": ood,
        }
        if "__verdict" in self.targets.columns:
            info["target_verdict"] = target_row["__verdict"]
        return out, out_sr, info


if __name__ == "__main__":
    aug = PodcastAugmentorV4(seed=42)
    print("\nSample retrieval check:")
    rng = np.random.default_rng(0)
    for i in range(3):
        idx = int(rng.integers(0, len(aug.targets)))
        target_row = aug.targets.iloc[idx]
        target_feats = target_row[aug.match_features].fillna(0).values.astype(float)
        params, combo_idx, dist = aug._retrieve_combo(target_feats)
        verdict = target_row.get("__verdict", "?")
        print(f"  target #{idx} ({verdict}) -> combo #{combo_idx}, dist={dist:.2f}")
        print(f"    codec={params['codec_type']} br={params['codec_bitrate']} "
              f"lp={params['lowpass_hz']} comp={params['compression_threshold']}")
