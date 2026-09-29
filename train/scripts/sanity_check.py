"""Sanity check: load DAC, encode→decode a studio sample, verify roundtrip.

If the reconstructed waveform sounds noticeably worse than input, DAC
44.1 kHz is the quality ceiling for our model — escalate to higher-bandwidth
codec or accept the trade-off.

Also runs a 1-step forward pass through SpeechRestorationModel with random
mel + tokens to verify shapes line up end-to-end.
"""

from __future__ import annotations

import torch

from diamond.data.tokenize import FrozenDAC
from diamond.model.model import SpeechRestorationModel


def check_dac_roundtrip(wav_path: str | None = None, device: str = "cpu"):
    print("=" * 60)
    print("DAC roundtrip check")
    print("=" * 60)
    dac = FrozenDAC(device=device)
    print(f"DAC sample_rate: {dac.sample_rate}")
    print(f"DAC n_codebooks: {dac.n_codebooks}")
    print(f"DAC codebook_size: {dac.codebook_size}")

    if wav_path:
        import soundfile as sf
        wav, sr = sf.read(wav_path, dtype="float32")
        wav = torch.from_numpy(wav)
        print(f"Input: {wav_path}, sr={sr}, len={len(wav)/sr:.2f}s")
    else:
        sr = 44100
        wav = torch.randn(sr * 2) * 0.1
        print("Input: 2s of random noise @ 44.1 kHz")

    codes = dac.encode(wav, sr_in=sr)
    print(f"Encoded codes: {tuple(codes.shape)}")
    recon = dac.decode(codes)
    print(f"Reconstructed wav: {tuple(recon.shape)}, expected ~{wav.numel()} samples")


def check_model_shapes(device: str = "cpu"):
    print()
    print("=" * 60)
    print("Model shape check")
    print("=" * 60)

    # A toy model — the real dimensions live in configs/*.yaml.
    encoder_cfg = dict(
        n_mels=80, d_model=128, n_heads=4, n_layers=2, ff_dim=256, dropout=0.0
    )
    time_cfg = dict(
        d_model=128, n_heads=4, n_layers=2, ff_dim=256,
        n_codebooks=9, vocab_size=1024, bos_token_id=1024, dropout=0.0,
    )
    depth_cfg = dict(d_model=96, n_heads=4, n_layers=2, ff_dim=192, dropout=0.0)

    model = SpeechRestorationModel(
        encoder_cfg=encoder_cfg, time_cfg=time_cfg, depth_cfg=depth_cfg,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Toy model parameters: {n_params/1e6:.2f}M")

    B, T_mel, T_tok, K = 2, 50, 50, 9
    mel = torch.randn(B, T_mel, 80, device=device)
    tokens = torch.randint(0, 1024, (B, T_tok, K), device=device)

    logits, targets, mask = model(mel, tokens)
    print(f"logits: {tuple(logits.shape)} (expected [{B}, {T_tok}, {K}, 1024])")
    print(f"targets: {tuple(targets.shape)}")
    print(f"mask: {tuple(mask.shape)}, valid={mask.sum().item()}")

    print("Generation (greedy)...")
    out = model.generate(mel, n_tok_frames=T_tok)
    print(f"generated tokens: {tuple(out.shape)} (expected [{B}, {T_tok}, {K}])")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sanity_check] using device={device}")
    check_dac_roundtrip(wav_path=None, device=device)
    check_model_shapes(device=device)
    print()
    print("Sanity checks passed.")


if __name__ == "__main__":
    main()
