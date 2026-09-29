"""Build data/heldout/ — a held-out evaluation set the model never trains on.

Two sources:

  * default — stream real podcast audio samples from a HuggingFace dataset and keep
    the worst-quality ones (quality_verdict == 'bad'). Needs access to that
    dataset (it is gated/private).

  * --local-dir — index a folder of your own degraded audio. Use this if you
    do NOT have access to the podcast dataset: any .wav/.mp3/.flac/.ogg/.m4a
    files work. `make eval` then scores the model on them.

Heldout DNSMOS on real degraded audio is the only honest gauge of restoration
quality. These samples must NEVER appear in calibration, augmentation, or
training.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200, help="number of samples to keep")
    p.add_argument("--out", default="data/heldout/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local-dir", default=None,
                   help="build the heldout set from this folder of degraded audio "
                        "instead of the podcast HF dataset (no HF access needed)")
    # HF-source options (ignored when --local-dir is given):
    p.add_argument("--repo", default=None,
                   help="HF dataset to stream degraded audio from (alternative "
                        "to --local-dir)")
    p.add_argument("--quality", default="bad",
                   help="quality_verdict value to filter on")
    p.add_argument("--candidate_pool", type=int, default=20000,
                   help="pull this many matching samples then random-sample N")
    return p.parse_args()


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load any audio file as a float32 mono array. librosa fallback for mp3."""
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        import librosa
        wav, sr = librosa.load(str(path), sr=None, mono=False)
        wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return wav, int(sr)


def from_local_dir(src_dir: str, audio_dir: Path, n: int) -> list[dict]:
    """Index a user-supplied folder of degraded audio into the heldout layout."""
    files = sorted(f for f in Path(src_dir).rglob("*")
                   if f.suffix.lower() in AUDIO_EXTS)
    if not files:
        raise SystemExit(f"no audio files {sorted(AUDIO_EXTS)} found under {src_dir}")
    files = files[:n]
    rows = []
    for i, f in enumerate(tqdm(files, desc="indexing")):
        wav, sr = load_audio_mono(f)
        sid = f"heldout_{i:04d}"
        out_path = audio_dir / f"{sid}.wav"
        sf.write(str(out_path), wav, sr, subtype="PCM_16")
        rows.append({
            "id": sid,
            "audio_path": str(out_path),
            "sample_rate": sr,
            "duration_sec": round(len(wav) / sr, 4),
            "source": f.name,
        })
    return rows


def from_hf(repo: str, quality: str, candidate_pool: int, n: int,
            seed: int, audio_dir: Path) -> list[dict]:
    """Stream the podcast HF dataset, keep `quality` samples, random-pick N."""
    print(f"Streaming {repo}, filtering quality_verdict=='{quality}'")
    from datasets import load_dataset

    ds = load_dataset(repo, split="train", streaming=True)
    candidates: list[dict] = []
    for sample in tqdm(ds, desc="streaming"):
        if sample.get("quality_verdict") != quality:
            continue
        candidates.append(sample)
        if len(candidates) >= candidate_pool:
            break

    if len(candidates) < n:
        raise RuntimeError(
            f"Found only {len(candidates)} '{quality}' samples; need {n}")

    print(f"Sampling {n} from {len(candidates)} candidates with seed={seed}")
    chosen = random.sample(candidates, n)

    rows = []
    for i, sample in enumerate(tqdm(chosen, desc="writing")):
        sid = f"heldout_{i:04d}"
        audio = sample["audio"]
        wav = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        out_path = audio_dir / f"{sid}.wav"
        sf.write(str(out_path), wav, sr, subtype="PCM_16")
        rows.append({
            "id": sid,
            "audio_path": str(out_path),
            "sample_rate": sr,
            "duration_sec": round(len(wav) / sr, 4),
            "speaker": sample.get("speaker"),
            "dnsmos_ovrl": sample.get("dnsmos_ovrl"),
            "quality_verdict": sample.get("quality_verdict"),
        })
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.local_dir:
        print(f"Building heldout set from local folder: {args.local_dir}")
        rows = from_local_dir(args.local_dir, audio_dir, args.n)
    elif args.repo:
        rows = from_hf(args.repo, args.quality, args.candidate_pool,
                       args.n, args.seed, audio_dir)
    else:
        raise SystemExit(
            "provide --local-dir <folder> of your own degraded audio, "
            "or --repo <hf-dataset> to stream from HuggingFace")

    manifest_path = out_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"Wrote {len(rows)} heldout samples -> {audio_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
