"""Pull the pre-tokenized studio corpus from an HF dataset repo.

This is the fast path onto a fresh machine: it skips the multi-hour DAC
encoding pass that `prepare.py` would otherwise run over ~900 h of source
audio.

The repo holds `wavs/` and `tokens/` as tar archives (a few large files upload
and download far faster than ~350k small ones). They are unpacked into
cfg.data.studio_tokens_dir and the tars are deleted afterwards, so peak disk is
archive + extracted contents of one tar at a time — pass --keep-tar to retain them.

Manifest paths are stored relative to the manifest dir (see prepare.py),
so no path rewriting is needed after the pull — load_manifest resolves them.

A token is only needed for private repos:
  export HF_TOKEN=hf_...          # optional
  python -m diamond.data.preprocessing.pull --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--repo", default=None,
                   help="Override cfg.data.tokenized_hf_repo")
    p.add_argument("--out", default=None,
                   help="Override cfg.data.studio_tokens_dir")
    p.add_argument("--keep-tar", action="store_true",
                   help="Keep the downloaded .tar archives after extraction "
                        "(default: delete them to free disk)")
    return p.parse_args()


def extract_tars(out_dir: Path, keep_tar: bool) -> None:
    """Unpack every .tar in out_dir. Archives carry their own `tokens/` and
    `wavs/` prefixes, so they extract straight into out_dir."""
    tars = sorted(out_dir.glob("*.tar"))
    if not tars:
        return
    for tar_path in tars:
        print(f"[extract] {tar_path.name} ({tar_path.stat().st_size / 1e9:.1f} GB)...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(out_dir)
        if not keep_tar:
            tar_path.unlink()
            print(f"[extract] done, removed {tar_path.name}")
        else:
            print(f"[extract] done, kept {tar_path.name}")


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if "defaults" in cfg:
        base = OmegaConf.load(Path(args.config).parent / cfg.defaults)
        cfg = OmegaConf.merge(base, cfg)
        OmegaConf.set_struct(cfg, False)
        cfg.pop("defaults", None)

    repo_id = args.repo or cfg.data.get("tokenized_hf_repo")
    if not repo_id:
        raise SystemExit(
            "No HF dataset repo configured. Set --repo or cfg.data.tokenized_hf_repo"
        )

    # Public repos need no token; hf passes None through anonymously.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    out_dir = Path(args.out) if args.out else Path(cfg.data.studio_tokens_dir)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Repo: {repo_id}  →  {out_dir}")
    if not token:
        print("(no HF_TOKEN set — pulling anonymously; fine for a public repo)")

    from huggingface_hub import snapshot_download
    # snapshot_download: parallel, resumable, content-hash-deduped via cache.
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(out_dir),
        token=token,
    )

    extract_tars(out_dir, args.keep_tar)

    n_wav = sum(1 for _ in (out_dir / "wavs").glob("*.flac"))
    n_tok = sum(1 for _ in (out_dir / "tokens").glob("*.pt"))
    manifests = [p.name for p in out_dir.glob("manifest*.csv")]
    bytes_total = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    print(f"\nDone: {n_wav} wavs / {n_tok} tokens / {bytes_total / 1e9:.2f} GB")
    print(f"Manifests: {manifests or '(none — check the repo contents)'}")
    print(f"Local path: {path}")


if __name__ == "__main__":
    main()
