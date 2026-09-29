"""Merge per-source manifests (from parallel tokenization) into one, shuffled.

scripts/tokenize_parallel.sh writes manifest_<prefix>.csv per source. This
concatenates them into a single manifest and shuffles on disk (default on) so
the DataLoader doesn't see source-clumped order — a clumped order lets one
source dominate a batch, which hurts gradient diversity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="data/studio_tokens",
                   help="Dir holding manifest_<prefix>.csv files.")
    p.add_argument("--out", default="manifest.csv",
                   help="Output manifest filename (written under --dir). This is the "
                        "name training reads by default.")
    p.add_argument("--glob", default="manifest_*.csv",
                   help="Pattern for per-source manifests to merge.")
    p.add_argument("--no_shuffle", action="store_true",
                   help="Disable on-disk shuffle (NOT recommended for training).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_seconds", type=float, default=None,
                   help="Optional extra duration floor applied at merge time.")
    args = p.parse_args()

    d = Path(args.dir).resolve()
    out_path = d / args.out
    parts = sorted(f for f in d.glob(args.glob)
                   if f.name != args.out and f.name != "manifest.csv")
    if not parts:
        raise SystemExit(f"No per-source manifests matching {args.glob} in {d}")

    frames = []
    print(f"Merging {len(parts)} per-source manifests from {d}:")
    for f in parts:
        df = pd.read_csv(f)
        hrs = df["duration_sec"].sum() / 3600 if "duration_sec" in df else float("nan")
        print(f"  {f.name:34s} {len(df):>7d} rows  {hrs:6.1f}h")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    if args.min_seconds is not None and "duration_sec" in merged:
        merged = merged[merged["duration_sec"] >= args.min_seconds]
        print(f"  min_seconds={args.min_seconds}: {before} → {len(merged)} rows")

    if not args.no_shuffle:
        merged = merged.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        print(f"  shuffled (seed={args.seed})")

    merged.to_csv(out_path, index=False)
    total_h = merged["duration_sec"].sum() / 3600 if "duration_sec" in merged else float("nan")
    print(f"\n=== MERGED ===\n{len(merged)} rows  {total_h:.1f}h  →  {out_path}")


if __name__ == "__main__":
    main()
