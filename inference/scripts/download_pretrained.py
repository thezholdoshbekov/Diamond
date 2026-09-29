"""Download the pretrained Diamond checkpoint from the Hugging Face Hub.

Pulls the weights (.safetensors) + config (.json) into data/pretrained/.
The DAC codec weights are downloaded automatically on first inference by
`descript-audio-codec` — nothing to fetch here for that.

    python scripts/download_pretrained.py
    python scripts/download_pretrained.py --repo <org/name> --out data/pretrained
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Public HF repo holding the released weights. Override with --repo or the
# DIAMOND_HF_REPO env var.
DEFAULT_REPO = os.environ.get("DIAMOND_HF_REPO", "nineninesix/diamond-1.0")
WEIGHTS = "diamond.safetensors"
CONFIG = "diamond.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=DEFAULT_REPO, help="HF repo id (org/name)")
    p.add_argument("--out", default="data/pretrained", help="download dir")
    p.add_argument("--weights", default=WEIGHTS)
    p.add_argument("--config", default=CONFIG)
    args = p.parse_args()

    from huggingface_hub import hf_hub_download

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for fname in (args.weights, args.config):
        print(f"downloading {args.repo}/{fname} ...", flush=True)
        local = hf_hub_download(repo_id=args.repo, filename=fname,
                                local_dir=str(out))
        print(f"  -> {local}")
    print(f"done. run:  diamond-restore degraded.wav restored.wav "
          f"--checkpoint {out / args.weights}")


if __name__ == "__main__":
    main()
