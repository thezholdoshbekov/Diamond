<div align="center">

# Diamond — Inference

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Model](https://img.shields.io/badge/🤗-Checkpoint-yellow.svg)](https://huggingface.co/nineninesix/diamond-1.0) [![Tech Report](https://img.shields.io/badge/📄-Tech%20Report-red.svg)](https://github.com/nineninesix-ai/diamond-train/blob/main/docs/TECH_REPORT.pdf) [![Train](https://img.shields.io/badge/⚙️-Training%20repo-lightgrey.svg)](https://github.com/nineninesix-ai/diamond-train)

Turn degraded audio — low-bitrate codecs, background noise, clipping, narrow band — into **44.1 kHz near-studio quality**.

</div>

## About

**Diamond** is a sequence-to-sequence speech restoration model. A degraded
mel-spectrogram is read by a bidirectional encoder; an autoregressive
**RQ-Transformer** decoder predicts the tokens of a frozen neural audio codec
(DAC, 9-book RVQ), which renders the restored waveform. It is trained **from
scratch** (~166.6M params, no pretrained backbone) and reaches the level of the
strongest open speech restorers.

```
degraded audio
  → resample 24 kHz → log-mel (80 bins, 85.7 Hz)
  → encoder (bidirectional, 20L)
  → time-transformer (causal self- + cross-attention, 20L)  → one hidden per frame
  → depth-transformer (local AR over the 9 RVQ codes)       → DAC tokens
  → frozen DAC decoder
  → restored 44.1 kHz waveform
```

Long audio is decoded in overlapping chunks that stay inside the trained horizon;
a repetition penalty, per-chunk warmup, tail padding and selective
silence-collapse recovery keep content intact across the seams.

This repository is **inference only**. Training, the DSP augmentation engine, and
the [technical report](https://github.com/nineninesix-ai/diamond-train/blob/main/docs/TECH_REPORT.pdf)
live in [diamond-train](https://github.com/nineninesix-ai/diamond-train).
The code layout is described in [docs/STRUCTURE.md](docs/STRUCTURE.md).

## Installation

Python ≥ 3.10, PyTorch ≥ 2.1. A CUDA GPU is strongly recommended — CPU works but is
much slower (`--device` is auto-detected; you never pass it).

```bash
git clone git@github.com:nineninesix-ai/diamond-inference.git
cd diamond-inference
make setup            # install the package + pull the weights into data/pretrained/
```

The ~330 MB checkpoint lives on the Hugging Face Hub
([`nineninesix/diamond-1.0`](https://huggingface.co/nineninesix/diamond-1.0)),
not in git. The DAC codec weights download automatically on the first run.

Without `make` (Windows / any OS):

```bash
pip install -e .
python scripts/download_pretrained.py
```

## Quick Start

```bash
make infer INPUT=degraded.mp3 OUTPUT=restored.wav
```

Or call the installed command directly:

```bash
diamond-restore degraded.mp3 restored.wav
```

From Python:

```python
from diamond import Diamond

model = Diamond.from_pretrained("data/pretrained/diamond.safetensors")
model.restore_path("degraded.mp3", "restored.wav")

# or work with numpy directly:
import soundfile as sf
wav, sr = sf.read("degraded.wav")
restored, out_sr = model.restore(wav, sr)
sf.write("restored.wav", restored, out_sr)
```

## CLI options

```
diamond-restore INPUT OUTPUT [options]

  --checkpoint PATH      weights (.safetensors or .pt)  [data/pretrained/diamond.safetensors]
  --device cuda|cpu      auto-detected if unset
  --chunk_sec 2.5        decoding chunk length (seconds)
  --overlap_sec 0.4      crossfaded overlap between chunks
  --warmup_sec 1.0       real preceding audio prepended per chunk (warm start)
  --tail_pad_sec 1.0     silence appended before decode (absorbs AR tail drift)
  --rep_penalty 1.3      repetition penalty on all codebooks (breaks AR loops)
  --rep_window 96        frames the penalty looks back over
  --max_batch 4          equal-length chunks per GPU batch (lower if OOM)
  --seed N               reproducible decoding
  --no_normalize         skip peak-normalization to 0.99
  --no_trim_leadin       keep the non-speech lead-in (default: trim + pad back)
  --no_recover_collapse  disable the silence/loop re-decode fallback
```

The defaults are the validated shipping configuration. Each one fixes a specific
failure, and **[docs/INFERENCE.md](docs/INFERENCE.md)** explains which — read it
before you change any of them. On hard, long phone recordings `--chunk_sec 3.0`
sometimes helps.

## Commands

```bash
make help          # show all commands
make setup         # install the package + download the weights
make install       # install the package only
make download      # pull the checkpoint into data/pretrained/
make infer         # restore a file  │ INPUT=in.mp3 [OUTPUT=out.wav] [CKPT=path]
make check-env     # Python / GPU / checkpoint status
make clean         # remove build artifacts
```

## Related Repositories

- **[diamond-train](https://github.com/nineninesix-ai/diamond-train)** —
  training pipeline, DSP augmentation engine, and the technical report.

## Citation

If you use Diamond, please cite:

```bibtex
@software{diamond2026,
  author       = {NineNineSix},
  title        = {{Diamond}: Speech Restoration with an RQ-Transformer},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/nineninesix/diamond-1.0}},
  note         = {Inference: \url{https://github.com/nineninesix-ai/diamond-inference}}
}
```

Method details and academic references (RoPE, RQ-Transformer, DAC, etc.) are in
the [diamond-train](https://github.com/nineninesix-ai/diamond-train) README and
the technical report.

## License

Apache-2.0 — see [LICENSE](LICENSE).

Diamond loads the **Descript Audio Codec** (frozen, 44.1 kHz) to synthesize the
waveform. It is not covered by this repository's Apache-2.0 license — see
[NOTICE](NOTICE) for third-party attribution.
