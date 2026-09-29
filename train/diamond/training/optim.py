"""AdamW + cosine LR schedule with warmup, biases/norm/embeddings excluded from decay."""

import math

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, lr: float, betas: tuple[float, float],
                    weight_decay: float, exclude_keywords: list[str],
                    rope_alpha_lr_ratio: float = 1.5) -> torch.optim.AdamW:
    decay_params, no_decay_params, layer_scale_params, rope_alpha_params = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Learnable-RoPE α (alpha_raw): scalar params with tiny gradients
        # vs the rest of the network. Kani uses ×1.5 lr to keep them
        # learning at comparable pace. No weight decay (sigmoid handles bounds).
        if name.endswith(".alpha_raw"):
            rope_alpha_params.append(p)
        # LayerScale γ params: must not be weight-decayed — WD would pull γ
        # toward 0, killing the residual contribution and defeating the
        # per-layer gating mechanism that fixes L0 runaway.
        # ls3 exists only in TimeLayer (self-attn / cross-attn / FF = 3 branches);
        # encoder and depth layers have 2. Omitting it decayed the FF-branch γ of
        # every decoder layer, which the FF weights then compensated by growing.
        elif name.endswith(".ls1") or name.endswith(".ls2") or name.endswith(".ls3"):
            layer_scale_params.append(p)
        elif any(kw in name.lower() for kw in exclude_keywords):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": layer_scale_params, "weight_decay": 0.0},
    ]
    if rope_alpha_params:
        # Higher base_lr for α group — LambdaLR scales all groups by the
        # same lambda, so this 1.5× scaling persists through the schedule.
        groups.append({
            "params": rope_alpha_params,
            "weight_decay": 0.0,
            "lr": lr * rope_alpha_lr_ratio,
        })

    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def cosine_warmup_lambda(warmup_steps: int, total_steps: int,
                         min_lr_ratio: float = 0.1):
    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos
    return lr_at


def flat_lambda(warmup_steps: int):
    """Linear warmup, then constant peak — for resume + stay-in-narrow-valley experiments."""
    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    return lr_at


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int,
                    total_steps: int, min_lr_ratio: float = 0.1,
                    schedule: str = "cosine"):
    if schedule == "cosine":
        lr_lambda = cosine_warmup_lambda(warmup_steps, total_steps, min_lr_ratio)
    elif schedule == "flat":
        lr_lambda = flat_lambda(warmup_steps)
    else:
        raise ValueError(f"Unknown schedule: {schedule!r} (expected 'cosine' or 'flat')")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
