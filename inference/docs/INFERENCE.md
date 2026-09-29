# Inference guide

The [README](../README.md) shows how to run Diamond. This explains **why the
defaults are what they are** — every one of them fixes a failure we measured, so
changing one without knowing which failure it guards against will usually make
the output worse.

The method and the numbers behind all of it are in the technical report:
**[TECH_REPORT.pdf](https://github.com/nineninesix-ai/diamond-train/blob/main/docs/TECH_REPORT.pdf)**.

---

## The one thing to understand

Diamond's decoder is **autoregressive**: it emits one frame at a time, and each
frame is conditioned on the frames it already emitted. During training it always
saw *correct* history (the ground truth). At inference it sees *its own* history,
mistakes included — and the mistakes compound. This is exposure bias, and every
default below exists to keep it in check.

Two things go wrong when it is not kept in check:

- **Silence collapse.** The decoder drifts into the silence-token basin and stays
  there — the audio simply stops. It gets worse the longer the decode runs.
- **Repetition loops.** The decoder repeats the same tokens. Loudness looks
  perfectly normal, which is why a level check misses it, but the words are
  mangled — these are the outliers where the transcription error explodes.

## Chunking: `--chunk_sec 2.5`, `--overlap_sec 0.4`

The model was trained on crops of 2.5-10 seconds. Ask it to decode a 30-second
file in one pass and it is far outside anything it ever saw, and it collapses.
So the file is decoded in chunks **shorter than the training crop**, and the
chunks are crossfaded back together over `overlap_sec`.

Chunks are decoded **independently** — that is the point. Each one starts from a
fresh encoder context, so a chunk that derails cannot poison the next.

- **Smaller is not always better.** 2.5 s is the aggregate optimum. Cut it much
  further and you break the audio into pieces too short to carry sentence context.
- **`--chunk_sec 3.0` is worth trying on hard, long recordings** (heavily degraded
  phone audio), where the slightly longer context wins.
- **`--chunk_sec 0` disables chunking.** Expect collapse beyond ~10 seconds.

## `--rep_penalty 1.3` — the single largest win

Recently emitted tokens get their logits penalized before the argmax. This is
what breaks the repetition loops, and it is **the biggest inference-side
improvement we measured**: on real degraded speech, WER dropped 21% and CER 23%.

It does not make healthy clips better — it rescues broken ones. About a third of
the clips in our test improved and the rest were untouched, which is exactly the
shape you want: no trade-off, only a fixed tail.

`1.0` disables it. Values above ~1.6 start introducing errors of their own.

## `--warmup_sec 1.0` — no cold starts

Each chunk is decoded with a second of the **real preceding audio** prepended;
that lead is trimmed off afterwards. Without it, every chunk after the first
cold-starts in the middle of a sentence, and the first word of each chunk tends
to get dropped.

## `--tail_pad_sec 1.0` — protect the last words

Silence is appended to the file before decoding, so the autoregressive tail decay
lands on samples that get thrown away instead of on the last real words of the
recording.

## Collapse recovery (on by default)

After the greedy pass, each chunk is checked for the two failure modes above —
silence (level far below the input's) and looping (low n-gram diversity in the
content codebook). A chunk that failed either check is **re-decoded with a
sampling fallback**, and the re-decode is kept only if it is measurably better.

Healthy chunks keep their greedy output. This matters: we measured that sampling
*everything* is worse than greedy on aggregate. Sampling only where greedy
visibly failed is what makes it a win. Disable with `--no_recover_collapse`.

## Speed

Decoding is serial by nature: 86 frames per second of audio, each walking nine
codebooks. Expect roughly **10-15× slower than realtime on a mid-range GPU**.
This is offline restoration and dataset cleansing — not a live filter.

`--max_batch` (default 4) decodes several chunks in parallel and is your main
lever; raise it if the GPU has headroom, lower it on out-of-memory.

## Limits worth knowing

- **A thin error tail.** Most clips come back transcribing as accurately as the
  input; a few derail. This is the structural price of autoregression, and the
  report documents it openly rather than hiding it. If content fidelity is
  critical, check the transcript.
- **English.** Training data is English; other languages are untested.
- **It generates, it does not filter.** Content above the codec's cutoff is
  *invented* from context — plausible, not recovered. Never treat restored audio
  as evidence of what was actually said.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Output is silent or trails into silence | Chunking disabled or too long — keep `--chunk_sec` at 2.5-3.0. |
| Words repeat in a loop | `--rep_penalty` was lowered to 1.0. Put it back to 1.3. |
| First word of a phrase missing | `--warmup_sec` set to 0. |
| Output quieter than expected | `--no_normalize` was passed; raw output is quiet by design. |
| Out of memory | Lower `--max_batch`. |
| Checkpoint fails to load | Checkpoints predating the RQ-Transformer read-out (no `depth` section in the embedded config) are not supported. |
