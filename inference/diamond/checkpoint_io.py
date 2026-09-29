"""Checkpoint I/O: format-agnostic load.

Training writes two artifacts per checkpoint:
  • step_XXXXXXXX.pt           — full resumable state (model + optimizer +
                                 scheduler + step + config). torch.save / pickle.
  • step_XXXXXXXX.safetensors  — model weights ONLY, portable + safe (no pickle).
    step_XXXXXXXX.json         — the resolved config, sidecar for the safetensors.

Inference accepts either. The .safetensors+.json pair is the shareable artifact —
loadable without trusting a pickle — and is what the published checkpoint ships as.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    """Load a checkpoint regardless of format.

    Accepts either a .pt full-state checkpoint, or a .safetensors weights file
    (its config is read from the sibling .json). Always returns a dict with at
    least {"model": state_dict, "config": dict|None, "step": int|None} so
    callers can treat both formats uniformly.
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        device = str(map_location) if not isinstance(map_location, str) else map_location
        state = load_file(str(path), device=device)

        step = None
        try:
            with safe_open(str(path), framework="pt") as f:  # type: ignore[arg-type]
                meta = f.metadata() or {}
            if "step" in meta:
                step = int(meta["step"])
        except Exception:
            pass

        json_path = path.with_suffix(".json")
        config = json.loads(json_path.read_text()) if json_path.exists() else None
        return {"model": state, "config": config, "step": step}

    return torch.load(path, map_location=map_location)
