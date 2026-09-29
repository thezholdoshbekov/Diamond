"""Pre-tokenize studio audio: studio.wav -> DAC tokens (.pt) on disk.

Done once before training. Saves time during DataLoader iteration since
DAC encoding is expensive (~50ms per sample on GPU).

Multi-source: cfg.data.studio_sources is a list of {repo, config, split,
id_prefix} entries; output of all sources merges into a single manifest.csv
(per-source id_prefix prevents filename collisions).

Output:
  <out>/
    manifest.csv           - id, wav_path, tokens_path, duration_sec, source
    wavs/<id>.flac         - original studio audio (FLAC, lossless ~50% of PCM_16)
    tokens/<id>.pt         - torch.Tensor [T_tok, 9] int16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from diamond.data.tokenize import FrozenDAC


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None,
                   help="Output dir. Defaults to cfg.data.studio_tokens_dir.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None,
                   help="Per-source: take at most this many samples. Useful "
                        "for local debugging on a small subset.")
    p.add_argument("--min_seconds", type=float, default=3.0,
                   help="Skip files shorter than this. Default 3.0s: heldout "
                        "min is 3.14s, training audio_seconds=5.0, so files "
                        "<3s would train mostly on zero-padded chunks.")
    p.add_argument("--max_seconds", type=float, default=30.0,
                   help="Skip overly long samples - they spike GPU memory.")
    p.add_argument("--source", action="append", default=None,
                   help="Override cfg.data.studio_sources with this single "
                        "entry. Format: repo[:config]:split:id_prefix. May "
                        "be passed multiple times. Example: "
                        "mythicinfinity/libritts_r:clean:train.clean.100:libritts_c100")
    p.add_argument("--local-dir", action="append", default=None,
                   help="Tokenize your OWN studio audio instead of an HF dataset: "
                        "path to a folder of clean .wav/.mp3/.flac/.ogg/.m4a "
                        "(searched recursively). May be passed multiple times. "
                        "These are the CLEAN targets — the augmentor degrades "
                        "them online during training.")
    p.add_argument("--id-prefix", default=None,
                   help="id_prefix for --local-dir sources (default: folder name). "
                        "Keeps filenames from colliding with other sources.")
    p.add_argument("--max_hours", type=float, default=None,
                   help="Stop after total cumulative kept duration reaches "
                        "this many hours (across all sources). Useful when "
                        "disk-bound; sources processed in declared order.")
    p.add_argument("--manifest_name", default="manifest.csv",
                   help="Output manifest filename (e.g. manifest_200h.csv).")
    p.add_argument("--no_streaming", action="store_true",
                   help="Disable HF streaming and bulk-download dataset to "
                        "local cache. Default is streaming=True to avoid "
                        "filling disk on big datasets like libritts_r.")
    return p.parse_args()


def normalize_peak(wav: np.ndarray, peak: float = 0.99) -> np.ndarray:
    m = np.abs(wav).max()
    if m < 1e-8:
        return wav
    return wav * (peak / m)


def parse_source_arg(s: str) -> dict:
    """`repo[:config]:split:id_prefix` -> dict."""
    parts = s.split(":")
    if len(parts) == 3:
        repo, split, id_prefix = parts
        config = None
    elif len(parts) == 4:
        repo, config, split, id_prefix = parts
    else:
        raise ValueError(
            f"Bad --source value '{s}'. Expected repo[:config]:split:id_prefix"
        )
    return {"repo": repo, "config": config, "split": split, "id_prefix": id_prefix}


def resolve_sources(cfg, cli_sources: list[str] | None,
                    local_dirs: list[str] | None = None,
                    id_prefix: str | None = None) -> list[dict]:
    """Local folders win, then --source, then cfg.data.studio_sources, then the
    legacy cfg.data.studio_repo single-source fallback."""
    if local_dirs:
        out = []
        for i, d in enumerate(local_dirs):
            name = id_prefix or Path(d).resolve().name or "local"
            # Several folders + one --id-prefix: number them so ids stay unique.
            out.append({
                "local_dir": d,
                "id_prefix": name if len(local_dirs) == 1 else f"{name}{i}",
            })
        return out
    if cli_sources:
        return [parse_source_arg(s) for s in cli_sources]
    if "studio_sources" in cfg.data:
        out = []
        for s in cfg.data.studio_sources:
            entry = OmegaConf.to_container(s, resolve=True)
            entry.setdefault("config", None)
            entry.setdefault("split", "train")
            out.append(entry)
        return out
    # Legacy single-repo fallback for old configs.
    return [{
        "repo": cfg.data.studio_repo,
        "config": None,
        "split": "train",
        "id_prefix": "studio",
    }]


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read any audio file as float32 mono. librosa handles what soundfile can't."""
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        import librosa
        wav, sr = librosa.load(str(path), sr=None, mono=True)
        wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return wav, int(sr)


def encode_and_write(sample_id: str, wav: np.ndarray, sr: int, label: str,
                     dac: FrozenDAC, wavs_dir: Path, tokens_dir: Path,
                     device: str) -> dict:
    """Peak-normalize, save the studio FLAC, DAC-encode, save tokens, and return
    the manifest row. Paths are relative to the manifest dir so the cache stays
    portable across machines (pushed to HF here, pulled there)."""
    wav = normalize_peak(wav, peak=0.99)

    wav_path = wavs_dir / f"{sample_id}.flac"
    tokens_path = tokens_dir / f"{sample_id}.pt"
    sf.write(str(wav_path), wav, sr, format="FLAC", subtype="PCM_16")

    wav_t = torch.from_numpy(wav).to(device)
    codes = dac.encode(wav_t, sr_in=sr)                # [1, T_tok, K]
    codes = codes.squeeze(0).to(torch.int16).cpu().contiguous()
    torch.save(codes, tokens_path)

    return {
        "id": sample_id,
        "wav_path": f"wavs/{sample_id}.flac",
        "tokens_path": f"tokens/{sample_id}.pt",
        "wav_sr": sr,
        "duration_sec": round(len(wav) / sr, 4),
        "n_tok_frames": int(codes.shape[0]),
        "source": label,
    }


def process_local_dir(
    source: dict, dac: FrozenDAC, wavs_dir: Path, tokens_dir: Path,
    args, device: str, hours_remaining: float | None,
) -> tuple[list[dict], float]:
    """Same pipeline as process_source, but over a folder of your own audio.

    Any .wav/.mp3/.flac/.ogg/.m4a/.opus under the folder (recursive) is treated
    as one studio clip: it is the CLEAN target, and the augmentor degrades it
    online during training. Feed it studio-quality audio, not degraded audio.
    """
    src_dir = Path(source["local_dir"])
    id_prefix = source["id_prefix"]
    label = f"local:{src_dir.name}"

    files = sorted(f for f in src_dir.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
    if not files:
        raise SystemExit(
            f"No audio {sorted(AUDIO_EXTS)} found under {src_dir}"
        )
    if args.limit:
        files = files[: args.limit]
    print(f"\nLoading local dir: {src_dir}  ({len(files)} files, id_prefix={id_prefix})")

    rows: list[dict] = []
    skipped = 0
    src_seconds = 0.0
    budget_seconds = hours_remaining * 3600 if hours_remaining is not None else None

    for i, f in enumerate(tqdm(files, desc=id_prefix)):
        if budget_seconds is not None and src_seconds >= budget_seconds:
            print(f"  ↳ budget reached after {i} files ({src_seconds/3600:.2f}h)")
            break
        try:
            wav, sr = load_audio_mono(f)
        except Exception as e:
            print(f"  ⚠️  unreadable, skipping {f.name}: {e}")
            skipped += 1
            continue

        duration = len(wav) / sr
        if duration < args.min_seconds or duration > args.max_seconds:
            skipped += 1
            continue

        rows.append(encode_and_write(f"{id_prefix}_{i:07d}", wav, sr, label,
                                     dac, wavs_dir, tokens_dir, device))
        src_seconds += duration

    src_hours = src_seconds / 3600
    print(f"  {label}: {len(rows)} kept ({src_hours:.2f}h), {skipped} skipped "
          f"(outside {args.min_seconds}-{args.max_seconds}s or unreadable)")
    return rows, src_hours


def process_source(
    source: dict, dac: FrozenDAC, wavs_dir: Path, tokens_dir: Path,
    args, device: str, hours_remaining: float | None,
) -> tuple[list[dict], float]:
    """Stream one HF source through DAC and dump (wav, tokens, manifest row).

    Returns (rows, hours_kept). If `hours_remaining` is set, stops once the
    source contributes that many hours (across multiple sources the caller
    decrements its own budget).
    """
    if source.get("local_dir"):
        return process_local_dir(source, dac, wavs_dir, tokens_dir, args,
                                 device, hours_remaining)

    from datasets import load_dataset

    repo = source["repo"]
    config = source.get("config")
    split = source.get("split", "train")
    id_prefix = source["id_prefix"]

    label = f"{repo}" + (f":{config}" if config else "") + f":{split}"
    streaming = not args.no_streaming
    print(f"\nLoading HF: {label}  (streaming={streaming})")

    kwargs = {"split": split, "streaming": streaming}
    if config:
        ds = load_dataset(repo, config, **kwargs)
    else:
        ds = load_dataset(repo, **kwargs)

    total = None
    if not streaming:
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        total = len(ds)
        print(f"  {total} samples to process (id_prefix={id_prefix})")
    else:
        if args.limit:
            ds = ds.take(args.limit)
        print("  streaming (no len); will stop on --max_hours or end-of-stream")
    if hours_remaining is not None:
        print(f"  hours budget remaining: {hours_remaining:.2f}h")

    rows = []
    skipped = 0
    src_seconds = 0.0
    budget_seconds = hours_remaining * 3600 if hours_remaining is not None else None

    for i, sample in enumerate(tqdm(ds, total=total, desc=id_prefix)):
        if budget_seconds is not None and src_seconds >= budget_seconds:
            print(f"  ↳ budget reached after {i} samples ({src_seconds/3600:.2f}h)")
            break

        audio = sample["audio"]
        wav = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])

        if wav.ndim > 1:
            wav = wav.mean(axis=-1)

        duration = len(wav) / sr
        if duration < args.min_seconds or duration > args.max_seconds:
            skipped += 1
            continue

        rows.append(encode_and_write(f"{id_prefix}_{i:07d}", wav, sr, label,
                                     dac, wavs_dir, tokens_dir, device))
        src_seconds += duration

    src_hours = src_seconds / 3600
    print(f"  {label}: {len(rows)} kept ({src_hours:.2f}h), {skipped} skipped")
    return rows, src_hours


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if "defaults" in cfg:
        base = OmegaConf.load(Path(args.config).parent / cfg.defaults)
        cfg = OmegaConf.merge(base, cfg)
        OmegaConf.set_struct(cfg, False)
        cfg.pop("defaults", None)

    out_dir = (Path(args.out) if args.out else Path(cfg.data.studio_tokens_dir)).resolve()
    wavs_dir = out_dir / "wavs"
    tokens_dir = out_dir / "tokens"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    tokens_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir.resolve()}")

    sources = resolve_sources(cfg, args.source, args.local_dir, args.id_prefix)
    print(f"Sources to process ({len(sources)}):")
    for s in sources:
        if s.get("local_dir"):
            desc = f"{s['local_dir']} (local folder)"
        else:
            desc = (f"{s['repo']}" + (f":{s['config']}" if s.get("config") else "")
                    + f":{s.get('split', 'train')}")
        print(f"  - {desc}  (id_prefix={s['id_prefix']})")

    print(f"\nLoading FrozenDAC ({cfg.dac.model})...")
    dac = FrozenDAC(model_type=cfg.dac.model, device=args.device)
    print(f"DAC sr={dac.sample_rate}, n_codebooks={dac.n_codebooks}, "
          f"codebook_size={dac.codebook_size}")

    all_rows = []
    unreachable = []
    hours_remaining = args.max_hours
    for source in sources:
        name = source.get("repo") or source.get("local_dir")
        if hours_remaining is not None and hours_remaining <= 0:
            print(f"\n↳ skipping {name}: --max_hours budget exhausted")
            continue
        # One source being gated or withdrawn must not sink the whole corpus:
        # skip it, keep the rest, and say so at the end.
        try:
            rows, src_hours = process_source(
                source, dac, wavs_dir, tokens_dir, args, args.device, hours_remaining,
            )
        except Exception as e:
            unreachable.append(name)
            print(f"\n⚠️  SKIPPED {name}: {type(e).__name__}: {e}")
            print("   (gated dataset? run `make login` if you have access)")
            continue
        all_rows.extend(rows)
        if hours_remaining is not None:
            hours_remaining -= src_hours

    if not all_rows:
        raise SystemExit("No source produced any samples — nothing to write.")

    manifest_path = out_dir / args.manifest_name
    pd.DataFrame(all_rows).to_csv(manifest_path, index=False)
    total_hours = sum(r["duration_sec"] for r in all_rows) / 3600
    n_ok = len(sources) - len(unreachable)
    print("\n=== DONE ===")
    print(f"Total: {len(all_rows)} samples ({total_hours:.2f}h) across {n_ok}/{len(sources)} sources")
    if unreachable:
        print(f"Skipped (unreachable): {', '.join(unreachable)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
