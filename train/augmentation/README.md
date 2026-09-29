# Augmentation — PodcastAugmentorV4

Vendored DSP augmentor used by the training DataLoader. It degrades clean
studio audio so the model learns to restore realistic low-quality speech.

For each sample the augmentor:

1. picks a random real degraded-speech profile from the target pool;
2. retrieves the nearest calibrated DSP combo (k-NN in a channel-feature
   space);
3. applies that combo's DSP chain — codec compression, band-limiting,
   dynamic-range and EQ changes.

## Noise transfer

`configs/pilot.yaml` sets `data.disable_noise: false`, so degradation is
**codec + EQ + real-noise transfer**: real silence/background segments from
`noise_bank/` (non-speech noise drawn from the Emolia corpus, Silero-VAD
extracted) are mixed into each sample. `noise_bank/` is committed, so this
works on a fresh clone with no extra download.

Setting `data.disable_noise: true` zeroes `noise_target_db` and
`speech_noise_snr_db` for every strategy, leaving codec + EQ only.

The broadband-noise / RIR track (`data.noise_aug`) is a **separate** mechanism
that lives outside this directory — see `diamond/data/noise_aug.py`.

## Files

| Path | Role |
|---|---|
| `podcast_augmentor_v4.py` | Entry point — `PodcastAugmentorV4` |
| `dsp_chain.py` | Runtime DSP degradation chain (`apply_chain`) + parameter sampling |
| `podcast_augmentor.py`, `augment_pipeline.py` | DSP chain primitives |
| `calibration_results.csv` | Calibrated DSP-combo palette (pre-built, committed) |
| `noise_bank/` | Real noise segments for noise transfer (committed) |
| `podcast_target_pool.csv` | Retrieval target pool — **not committed**, ~284 MB, fetched by `make augmentor` |

## Usage

The training pipeline loads the augmentor via `data.augmentor_path` in the
config (default `augmentation`). It is not invoked directly — see
`diamond/data/augmentation.py` for the wrapper.

## Calibrated palette

`calibration_results.csv` is the calibrated DSP-combo palette — committed and
ready to use. Training never rebuilds it. It was produced offline against an
internal studio dataset; the regeneration tooling is not part of this public
training repo, and is not needed to train.
