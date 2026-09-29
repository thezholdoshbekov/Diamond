# Repository structure

```
diamond/                    The training package (installed as `diamond`)
├── cli/                    Entry points — one module per phase
│   ├── prepare.py            python -m diamond.cli.prepare   (make data)
│   ├── train.py              python -m diamond.cli.train     (make train / resume / smoke)
│   └── eval.py               python -m diamond.cli.eval      (make eval)
├── data/
│   ├── preprocessing/      Offline data pipeline
│   │   ├── prepare.py        studio audio → DAC tokens + manifest
│   │   ├── pull.py           fetch the published pre-tokenized corpus
│   │   ├── heldout.py        build the held-out eval set from degraded audio
│   │   └── merge_manifests.py
│   ├── dataset.py          Torch Dataset: manifest → (degraded mel, clean DAC codes)
│   ├── augmentation.py     Online-augmentation hook (wraps the vendored augmentor)
│   ├── noise_aug.py        Separate broadband-noise + RIR track (opt-in)
│   └── tokenize.py         Frozen DAC encoder + mel extractor
├── model/                  RQ-Transformer: encoder + time- and depth-transformer
├── training/               Train loop, loss, optimizer, in-training CER eval
├── eval/                   DNSMOS evaluation of a checkpoint
└── checkpoint_io.py        Checkpoint save/load + safetensors export

augmentation/               Vendored DSP augmentor (PodcastAugmentorV4) + its data
├── podcast_augmentor_v4.py   Entry point: retrieval-driven degradation
├── podcast_augmentor.py      DSP primitives
├── dsp_chain.py              Runtime degradation chain
├── augment_pipeline.py       Codec / bitrate primitives
├── calibration_results.csv   Calibrated DSP-combo palette (committed)
├── noise_bank/               Real noise segments for noise transfer (committed)
└── podcast_target_pool.csv   Retrieval target pool (fetched by `make augmentor`)

configs/                    OmegaConf configs — default.yaml + pilot.yaml (the run recipe)
scripts/                    Ops helpers: DAC/DNSMOS download, sanity check, import smoke
docs/                       Model card, tech report (RU/EN), figures, this file
data/                       (gitignored) audio, token caches, pretrained weights, heldout set
checkpoints/                (gitignored) training checkpoints
```

## Where things run

| Phase | Command | Code |
|---|---|---|
| Build the token cache | `make data` | `diamond/data/preprocessing/{prepare,pull}.py` |
| Train | `make train` / `make resume` | `diamond/training/train.py` |
| Degrade audio online | (inside training) | `diamond/data/augmentation.py` → `augmentation/` |
| Evaluate | `make eval` | `diamond/eval/eval.py` |

`configs/pilot.yaml` is the recipe the released model was trained with; it inherits
from `configs/default.yaml` and is the only file most runs need to touch.
