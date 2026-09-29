"""Import + wiring smoke test for the training stack.

Verifies the package imports resolve and a tiny model builds + does one
forward/backward — no data, no DAC, no GPU needed. Run before a real train:

    python scripts/smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from diamond.model.model import SpeechRestorationModel
from diamond.training.loss import compute_loss
from diamond.training.optim import build_optimizer, build_scheduler

# tiny config (real dims live in configs/*.yaml)
enc = dict(n_mels=80, d_model=64, n_heads=4, n_layers=2, ff_dim=128)
tim = dict(d_model=64, n_heads=4, n_layers=2, ff_dim=128, n_codebooks=9,
           vocab_size=1024, bos_token_id=1024)
dep = dict(d_model=48, n_heads=4, n_layers=2, ff_dim=96)

model = SpeechRestorationModel(encoder_cfg=enc, time_cfg=tim, depth_cfg=dep)
n = sum(p.numel() for p in model.parameters())
print(f"model builds: {n/1e6:.2f}M params")

B, T = 2, 20
mel = torch.randn(B, T, 80)
targets = torch.randint(0, 1024, (B, T, 9))
logits, tgt, mask = model(mel, targets)
print(f"forward OK: logits {tuple(logits.shape)}")

cb_weights = torch.tensor([1.5, 0.95, 0.85, 0.75, 0.65, 0.55, 0.5, 0.5, 0.5])
loss, metrics = compute_loss(logits, tgt, mask, cb_weights,
                             label_smoothing=0.1, z_loss_weight=1e-4)
loss.backward()
print(f"loss+backward OK: loss={loss.item():.3f}")

opt = build_optimizer(model, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01,
                      exclude_keywords=["bias", "norm", "emb"])
sched = build_scheduler(opt, warmup_steps=100, total_steps=1000,
                        min_lr_ratio=0.1, schedule="cosine")
opt.step(); sched.step()
print("optimizer + scheduler OK")
print("\nSMOKE PASSED — training stack wired correctly")
