# Config guide

Training is driven by one YAML file. There are two of them:

- **`configs/default.yaml`** — every key, with its default. Nothing here is
  specific to a run; think of it as the schema.
- **`configs/pilot.yaml`** — the configuration the **released model** was trained
  with. It inherits `default.yaml` via `defaults: default.yaml` and overrides
  only what changes.

Pick one with `CONFIG=`, and override any single key from the command line:

```bash
make train CONFIG=pilot
python -m diamond.cli.train --config configs/pilot.yaml --override training.lr=1e-4
```

A checkpoint **embeds the config it was trained with**, so inference and eval do
not need the YAML — a checkpoint is self-describing.

The reasoning behind these values, and the failures that produced them, is in
[TECH_REPORT.pdf](TECH_REPORT.pdf). What follows is a map, not a repeat.

---

## The parts you will actually touch

### `data` — what the model trains on

| Key | Default | Meaning |
|---|---|---|
| `studio_sources` | LibriTTS-R | List of HuggingFace corpora of **clean** speech. Each entry is `{repo, config, split, id_prefix}`; `id_prefix` keeps filenames from colliding across sources. |
| `manifest_file` | — | The token cache index. `make data` builds it. |
| `tokenized_hf_repo` | — | If set, `make data` **pulls a pre-tokenized corpus** instead of encoding one, which saves many GPU-hours. |
| `crop_seconds_range` | `[2.5, 10.0]` (pilot) | Per-sample crop length, drawn uniformly. **Do not set this to a single fixed length.** Training only at 10 s leaves the encoder out-of-distribution on the 2.5 s chunks inference uses, and every chunk boundary cold-starts. |
| `start_at_zero_prob` | `0.2` | Fraction of crops forced to start at sample 0. Random crops almost never land on the true start of an utterance, so without this the model never sees an utterance beginning — and then collapses on the first chunk at inference. |
| `online_augmentation` | `true` | The degraded input is synthesized on the fly by the augmentor (see [augmentation/README.md](../augmentation/README.md)). The target is always the clean studio audio. |
| `noise_aug` | on in pilot | A separate robustness track: broadband noise mixed **after** the codec, room impulse responses convolved **before** it. The order matters — pre-codec broadband noise gets stripped by psychoacoustic masking, so the model would never see it. |
| `n_samples` | `null` | Subsample the corpus. Set a small integer for a smoke run. |

**Clean targets are the whole game.** The model learns to reproduce whatever the
target contains, breath and hiss included. Degraded inputs are synthesized; the
targets must be genuinely clean.

### `training` — the run

| Key | Pilot | Notes |
|---|---|---|
| `batch_size` × `grad_accumulation_steps` | 8 × 4 | Effective batch 32. Lower `batch_size` first if you hit OOM; keep the product constant. |
| `total_steps` | 250000 | The cosine schedule decays over this whole span, so changing it changes the learning-rate curve. |
| `lr` | 3e-4 | With `schedule: cosine` and `warmup_steps: 2000`. `min_lr_ratio: 0.1` sets the floor. |
| `precision` | `bf16-mixed` | **Not fp16** — attention scores overflow in fp16. |
| `grad_clip_norm` | 1.0 | The last line of defense against gradient spikes. Every deep run we did needed it. Do not remove it. |
| `scheduled_sampling` | off | Feeds the model its own predictions during training. See the warning below. |

### `logging` and `eval`

`logging.framework` is `wandb` by default; set it to `none` to train offline.
`eval.cer` runs a small ASR check during training and saves a `best_cer`
checkpoint — worth enabling, because **training loss does not predict inference
quality** (we measured a lower-loss checkpoint decoding *worse*). Select
checkpoints on CER, not on loss.

---

## The parts you should leave alone

These are not tuning knobs. Each one is load-bearing, and each was added to fix a
specific failure that stopped a run dead:

| Key | Why it exists |
|---|---|
| `encoder.layer_scale_init`, and the RMSNorm after the conv frontend | Without them the first encoder layer runs away — its weights grow while the rest stagnate — and the loss regresses mid-run. This was the single hardest bug in the project. |
| `encoder.qk_norm`, `decoder.qk_norm`, `depth.qk_norm` | Bounds attention-logit magnitude, the dominant source of gradient spikes in deep stacks. |
| `loss.z_loss_weight` | Penalizes logit inflation in the read-out. |
| `depth` (the whole section) | The depth-transformer **is** the architecture's point: it walks the nine codebooks inside each frame instead of reading them out with nine parallel heads. Its dimensions must match the checkpoint. |
| `loss.codebook_weights` | `[1.5, 0.95, ...]` — the leading 1.5 boosts the codebook that carries most of the phonetic content. Flattening these starves the fine codebooks and fricatives collapse. |
| `mel.hop_length: 280` | Gives 85.7 Hz, which lines up with the DAC's 86 Hz token rate. Change it and encoder and decoder stop being 1:1. |
| `dac.*` | Fixed by the frozen codec. |

### A specific warning: `scheduled_sampling`

Turning this on in the **time** axis — feeding the model its own predictions
across *frames* — destroyed a run: the errors fed an unbounded autoregressive
chain and the gradient exploded into a spike storm the loss never recovered from.

The config's scheduled sampling is **depth-axis** (within a frame, across the
nine codebooks), where the blast radius is bounded to one frame. It is still
`enabled: false`, and it should stay off until the coarse codebooks are learned
(accuracy ≥ 0.4). Feeding a model its own garbage early just teaches it garbage.

---

## Making your own config

Inherit and override. Do not copy `default.yaml` and edit it — you will silently
drift from the defaults when they change.

```yaml
# configs/mine.yaml
defaults: default.yaml

training:
  batch_size: 4              # smaller GPU
  grad_accumulation_steps: 8 # keep effective batch at 32
  total_steps: 50000

data:
  studio_sources:
    - repo: my-org/my-clean-speech
      config: null
      split: train
      id_prefix: mine
  n_samples: 2000            # start small; confirm the loss falls
```

Then:

```bash
make train CONFIG=mine
```

**Before a long run, do a short one.** A few thousand steps costs almost nothing
and catches the things that otherwise cost you a day: check that the loss falls,
that per-codebook accuracy rises, and that the gradient norm stays bounded and
does not cluster into spikes.
