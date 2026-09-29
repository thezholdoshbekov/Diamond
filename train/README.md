<div align="center">

# Diamond — Train

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Tech Report](https://img.shields.io/badge/📄-Tech%20Report-red.svg)](docs/TECH_REPORT.pdf) [![Model](https://img.shields.io/badge/🤗-Checkpoint-yellow.svg)](https://huggingface.co/nineninesix/diamond-1.0)

The reference **training** stack: the data pipeline, the DSP augmentation engine, and the training loop that produce **Diamond**, a speech restoration model.

</div>

## About

**Diamond** turns degraded speech - podcast, phone, streamed, over-compressed —
into near-studio 44.1 kHz audio. It is an **RQ-Transformer** (~166.6M trainable
params): degraded speech → log-mel spectrogram → Transformer encoder → decoder
(time-transformer, causal self- and cross-attention) → codebook read-out
(depth-transformer, a small local autoregression over the 9 RVQ codes of each
frame) → frozen DAC decoder → restored 44.1 kHz waveform.

This repository is the **training** side. It takes clean studio speech and runs
the full lifecycle:

- **Dataset build** — stream public studio corpora, encode them to frozen-DAC
  tokens, and assemble one manifest.
- **Augmentation** — degrade every clean sample online, per DataLoader worker,
  with a DSP chain retrieved per-sample from a palette calibrated against real
  degraded speech. The clean audio is the target; the degradation is the input.
- **Train** — cross-entropy over the 9 RVQ codebooks, with in-training DNSMOS
  and CER evaluation for checkpoint selection.
- **Evaluate** — DNSMOS-P.835 on held-out degraded audio, baseline vs restored.

The method, ablations, and stability analysis (encoder-L0 runaway, read-out
inflation, exposure bias) are in the technical report:
**[docs/TECH_REPORT.pdf](docs/TECH_REPORT.pdf)**. Every knob in the YAML is in
**[docs/CONFIG_GUIDE.md](docs/CONFIG_GUIDE.md)**; the code layout in
**[docs/STRUCTURE.md](docs/STRUCTURE.md)**.

This repository **trains** the model. To *restore audio* with a trained checkpoint,
use **[diamond-inference](https://github.com/nineninesix-ai/diamond-inference)**.

> [!NOTE]
> This project is under active development. Both the codebase and the speech restoration methodology will be continuously refined and improved over time.


## Installation

Requires Python ≥ 3.10 and a CUDA GPU (for training and `make smoke`). `make setup`
installs everything else — [`uv`](https://docs.astral.sh/uv/), the venv and deps,
[`ffmpeg`](https://ffmpeg.org/) (the augmentor runs audio through real mp3/aac/ogg
codecs), the frozen DAC codec, the DNSMOS scorer, and the augmentation target pool.

```bash
make setup
```

No Hugging Face token is needed to set up or train — the pre-tokenized corpus and
the augmentation target pool are public. Without `uv`, install directly:
`python -m venv .venv && pip install -e ".[dev,eval]"`, then call the modules
by hand (e.g. `python -m diamond.cli.train --config configs/pilot.yaml`).

### What gets downloaded, and from where

Nothing is fetched implicitly at import time — every download belongs to a `make`
target.

| Artifact | Source | Size | Fetched by | Lands in |
|---|---|---|---|---|
| Studio corpus, tokenized on your machine (681 h over the sources in the config) | streamed from HF, encoded with the DAC | ~40 GB of tokens + source audio | `make data` (auto-run by `make train`) | `data/studio_tokens/` |
| DAC 44.1 kHz codec (frozen encoder + vocoder) | Descript GitHub release, via `dac.utils.download()` — *not* HF | 307 MB | `make dac` | `data/pretrained/dac_44khz.pth` |
| Augmentation target pool (96,899 degradation fingerprints) | HF dataset `nineninesix/emolia_filtered_features` | 298 MB | `make augmentor` | `augmentation/podcast_target_pool.csv` |
| DNSMOS-P.835 scorer | GitHub `microsoft/DNS-Challenge` | 1.1 MB | `make dnsmos` | `data/pretrained/dnsmos_p835.onnx` |

> **Building the token cache is a one-off cost of several GPU-hours** — `make data`
> streams each corpus, encodes it with the frozen DAC, and writes the cache. Every
> run after that reuses it. `make check-env` prints your free disk space and checks
> every download URL before you commit to it.

A token is only needed for optional paths: `checkpoint.hf_repo` (mirror checkpoints
to the Hub as they save; `null` by default) and `make heldout HELDOUT_REPO=...` (if
your eval audio lives in a gated dataset).

## Quick Start

```bash
make setup                       # everything: uv, venv, deps, ffmpeg, DAC, DNSMOS, augmentor
make train CONFIG=pilot          # pulls the token cache on first run, then trains
```

That is it — no manual data step, no PATH editing, no config edits. To resume:

```bash
make resume CHECKPOINT=checkpoints/pilot/step_00015000.pt
```

Before committing to the 83 GB download, verify the pipeline on a mini-set:
`make all` (= `make setup` + a 100-step run on ~10 files per source).

**Fine-tune from the published checkpoint.** `nineninesix/diamond-1.0`
ships `.safetensors` — weights only, no optimizer state. `make resume` detects the
extension and warm-starts from it (fresh Adam moments and LR schedule, so expect a
brief loss bump over the first 1–2k steps):

```bash
uv run python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('nineninesix/diamond-1.0', 'diamond.safetensors', local_dir='ckpt/')"
make resume CHECKPOINT=ckpt/diamond.safetensors
```

**Train on your own audio.** You do not need the published corpus. Point the
tokenizer at a folder of your own **clean, studio-quality** audio — those clips are
the *targets*; the augmentor degrades them on the fly, so feeding it already-degraded
audio would teach the model to reproduce the damage:

```bash
make data-local AUDIO_DIR=path/to/clean_audio/   # → data/studio_tokens_local/
make train-local
```

The folder is searched recursively for `.wav`/`.mp3`/`.flac`/`.ogg`/`.m4a`/`.opus`;
any sample rate works (DAC resamples internally). Clips shorter than 3 s or longer
than 30 s are skipped. To mix your audio *with* the published corpus, tokenize into
the same directory under its own manifest and merge:

```bash
uv run python -m diamond.cli.prepare --config configs/pilot.yaml \
    --local-dir path/to/clean_audio/ --out data/studio_tokens --manifest_name manifest_mine.csv
uv run python -m diamond.data.preprocessing.merge_manifests \
    --dir data/studio_tokens --out manifest_all.csv
# then point data.manifest_file at data/studio_tokens/manifest_all.csv
```

**Evaluate.** Optional — skip it if you only want a model. `make eval` scores a
checkpoint with DNSMOS-P.835 on degraded audio **you provide** (the podcast set the
model was gauged on is gated, so there is no default source):

```bash
make heldout HELDOUT_DIR=path/to/degraded_audio/   # → indexes into data/heldout/
make eval CHECKPOINT=checkpoints/pilot/best_dnsmos.pt
```

It reports DNSMOS SIG/BAK/OVRL baseline vs restored, plus a per-sample CSV and the
restored `.wav` files for a listen test. The same held-out folder drives the
in-training CER eval (`eval.cer.enabled`, Whisper `tiny.en` on CPU), so build it
before a long run if you want CER-based checkpoint selection.

## Commands

```bash
make help          # show all commands
make setup         # install everything (uv + venv + deps + ffmpeg + DAC + DNSMOS + augmentor)
make all           # setup + smoke (one-shot install + verify)
make login         # authenticate Hugging Face (only for gated datasets / pushing ckpts)
make data          # get the token cache (pulls the pre-tokenized corpus if the config names one)
make data-local    # tokenize YOUR OWN clean audio        │ AUDIO_DIR=path
make heldout       # index your own degraded audio        │ HELDOUT_DIR=path
make train         # build the cache if missing, then train
make resume        # resume from .pt (or warm-start .safetensors) │ CHECKPOINT=path
make train-local   # train on the make data-local cache
make smoke         # quick verify: mini-data + 100-step run
make infer         # restore audio                        │ CHECKPOINT=path IN=file/dir OUT=file/dir
make eval          # DNSMOS-P.835 on the held-out set     │ CHECKPOINT=path
make test          # import smoke: model build + forward/backward (CPU)
make check-env     # environment / GPU / disk / download reachability
```

### Without `make` (Windows)

`make` is not installed on Windows by default. It is only a wrapper — every
command has a direct equivalent:

```bash
python -m diamond.cli.prepare --config configs/pilot.yaml    # make data
python -m diamond.cli.train   --config configs/pilot.yaml    # make train
python -m diamond.cli.train   --config configs/pilot.yaml     --override checkpoint.resume_from=checkpoints/pilot/step_50000.pt   # make resume
python -m diamond.cli.eval --checkpoint ckpt.pt --heldout data/heldout/ # make eval
python scripts/smoke.py                                      # make test
python scripts/download_pretrained.py                        # DAC weights
python scripts/download_dnsmos.py                            # DNSMOS scorer
```

## Augmentation

The DSP augmentor `PodcastAugmentorV4` is **vendored** in [`augmentation/`](augmentation/)
(see its [README](augmentation/README.md)). Per sample it picks a real degradation
fingerprint from the target pool, retrieves the nearest calibrated DSP combo (k-NN in
a channel-feature space), and applies that chain.

- Degradation is **codec + EQ + real-noise transfer** (`data.disable_noise: false`):
  real silence/background segments from `noise_bank/` are mixed in. `noise_bank/` and
  the calibrated palette are committed, so this works on a fresh clone.
- The large retrieval target pool (`podcast_target_pool.csv`, ~284 MB) is **not**
  committed — `make augmentor` downloads it.
- The pilot config also enables a **separate** broadband-noise + RIR track
  (`data.noise_aug`) expecting MUSAN/DNS noise and OpenSLR-28 RIRs under
  `data/noise_corpora/`. Those corpora are **not downloaded by any target** — supply
  them yourself. Without them the track disables itself with a warning and training
  proceeds on codec + EQ + noise transfer alone.
- `data.augmentor_path` points at `augmentation/`; the DataLoader applies the
  augmentor online, one sample per worker.

## License

Apache-2.0 — see [LICENSE](LICENSE).

Diamond loads the **Descript Audio Codec** (frozen, 44.1 kHz) at training and
inference time, and the **DNSMOS-P.835** scorer for evaluation. Neither is covered by
this repository's Apache-2.0 license, and the training corpora carry their own terms
— see [NOTICE](NOTICE) for third-party attribution.

## Citation

If you use Diamond, please cite:

```bibtex
@software{diamond2026,
  author       = {NineNineSix},
  title        = {{Diamond}: Speech Restoration with an RQ-Transformer},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/nineninesix/diamond-1.0}},
  note         = {Training stack: \url{https://github.com/nineninesix-ai/diamond-train}}
}
```

## References

The architecture reuses ideas (not code) from the following work; each is cited
inline in the source. BibTeX collected here for convenience.

```bibtex
@inproceedings{kumar2023dac,
  title     = {High-Fidelity Audio Compression with Improved {RVQGAN}},
  author    = {Kumar, Rithesh and Seetharaman, Prem and Luebs, Alejandro and
               Kumar, Ishaan and Kumar, Kundan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}

@inproceedings{lee2022rqvae,
  title     = {Autoregressive Image Generation using Residual Quantization},
  author    = {Lee, Doyup and Kim, Chiheon and Kim, Saehoon and Cho, Minsu and
               Han, Wook-Shin},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}

@article{defossez2024moshi,
  title   = {Moshi: a speech-text foundation model for real-time dialogue},
  author  = {D{\'e}fossez, Alexandre and Mazar{\'e}, Laurent and Orsini, Manu and
             R{\'o}zi{\`e}re, Am{\'e}lie and Kyutai},
  journal = {arXiv preprint arXiv:2410.00037},
  year    = {2024}
}

@article{su2021roformer,
  title   = {{RoFormer}: Enhanced Transformer with Rotary Position Embedding},
  author  = {Su, Jianlin and Lu, Yu and Pan, Shengfeng and Murtadha, Ahmed and
             Wen, Bo and Liu, Yunfeng},
  journal = {arXiv preprint arXiv:2104.09864},
  year    = {2021}
}

@article{learnablerope2024,
  title   = {Optimizing the Learnable {RoPE} Theta Parameter in Transformers},
  journal = {IEEE Xplore},
  note    = {Document 11084811. Source of the learnable per-layer RoPE-$\theta$
             scheme in diamond/model/rope.py},
  url     = {https://ieeexplore.ieee.org/abstract/document/11084811},
  year    = {2024}
}

@software{kani_tts_2,
  author       = {NineNineSix},
  title        = {{KaniTTS-2}: Text-to-Speech Model with Frame-level Position Encoding},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/nineninesix/kani-tts-2-en}},
  note         = {Reference implementation of the learnable RoPE-$\theta$ scheme}
}

@inproceedings{touvron2021cait,
  title     = {Going deeper with Image Transformers},
  author    = {Touvron, Hugo and Cord, Matthieu and Sablayrolles, Alexandre and
               Synnaeve, Gabriel and J{\'e}gou, Herv{\'e}},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2021}
}

@article{dehghani2023vit22b,
  title   = {Scaling Vision Transformers to 22 Billion Parameters},
  author  = {Dehghani, Mostafa and others},
  journal = {International Conference on Machine Learning (ICML)},
  year    = {2023}
}

@article{chowdhery2022palm,
  title   = {{PaLM}: Scaling Language Modeling with Pathways},
  author  = {Chowdhery, Aakanksha and others},
  journal = {arXiv preprint arXiv:2204.02311},
  year    = {2022}
}

@inproceedings{reddy2022dnsmos,
  title     = {{DNSMOS P.835}: A Non-Intrusive Perceptual Objective Speech
               Quality Metric to Evaluate Noise Suppressors},
  author    = {Reddy, Chandan K. A. and Gopal, Vishak and Cutler, Ross},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal
               Processing (ICASSP)},
  year      = {2022}
}

@article{radford2022whisper,
  title   = {Robust Speech Recognition via Large-Scale Weak Supervision},
  author  = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and
             McLeavey, Christine and Sutskever, Ilya},
  journal = {arXiv preprint arXiv:2212.04356},
  year    = {2022}
}
```
