# Repository structure

```
diamond/                    The inference package (installed as `diamond`)
├── cli/
│   └── infer.py            Entry point — `diamond-restore`, `python -m diamond.cli.infer`
├── infer.py                Restoration pipeline + the `Diamond` Python API
├── model/
│   ├── model.py            Encoder → time-transformer → depth-transformer (RQ read-out)
│   ├── encoder.py          Bidirectional mel encoder (20L)
│   ├── time_transformer.py Causal self- + cross-attention decoder (20L)
│   ├── depth_transformer.py Local AR over the 9 RVQ codes of each frame
│   ├── rope.py             Rotary positional embedding
│   └── rms_norm.py
├── tokenize.py             Mel extractor + frozen DAC wrapper
└── checkpoint_io.py        Load .safetensors / legacy .pt checkpoints

scripts/download_pretrained.py   Fetch the Diamond checkpoint from the Hub
docs/                            Model card, this file
data/pretrained/                 (gitignored) checkpoint + DAC weights
```

## Where things run

| Task | Command | Code |
|---|---|---|
| Restore a file | `make infer INPUT=… OUTPUT=…` | `diamond/infer.py` |
| Restore from Python | `Diamond.from_pretrained(...).restore(...)` | `diamond/infer.py` (`class Diamond`) |
| Fetch the weights | `make download` | `scripts/download_pretrained.py` |

Long inputs are decoded in overlapping chunks that stay inside the trained
horizon; repetition penalty, per-chunk warmup, tail padding and silence-collapse
recovery keep content intact across chunk seams.

Training, augmentation and the technical report live in the separate
[diamond-train](https://github.com/nineninesix-ai/diamond-train) repository.
