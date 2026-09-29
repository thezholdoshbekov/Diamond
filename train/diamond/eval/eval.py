"""Evaluate a checkpoint on heldout real podcast samples using DNSMOS.

DNSMOS is a reference-free perceptual metric (~0.85 corr with human MOS).
Scored on real degraded audio it is the only honest gauge of restoration
quality — metrics on augmented validation data flatter the model, since it
trained on that exact distribution.

Outputs:
  • restored .wav files in <out>/ for listen test (always written)
  • mean DNSMOS OVRL/SIG/BAK with delta vs baseline (if DNSMOS model present)
  • per-sample CSV report

Usage:
  python -m diamond.eval.eval \
    --checkpoint checkpoints/pilot/step_50000.pt \
    --heldout data/heldout/ \
    --dnsmos_onnx data/pretrained/dnsmos_p835.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF
from omegaconf import OmegaConf
from tqdm import tqdm

from ..data.tokenize import FrozenDAC, MelExtractor, resample_to
from ..model.model import SpeechRestorationModel

DEFAULT_DNSMOS_ONNX = "data/pretrained/dnsmos_p835.onnx"

# The P.835 ONNX graph takes exactly 9.01 s @ 16 kHz (144160 samples) and emits
# the three *raw* network outputs; the released scores are those raw values put
# through the polynomial fits published with DNS-Challenge.
DNSMOS_SR = 16000
DNSMOS_WIN = 144160
DNSMOS_POLY = {
    "sig": np.poly1d([-0.08397278, 1.22083953, 0.0052439]),
    "bak": np.poly1d([-0.13166888, 1.60915514, -0.39604546]),
    "ovrl": np.poly1d([-0.06766283, 1.11546468, 0.04602535]),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--heldout", required=True,
                   help="Directory with manifest.csv produced by heldout.py")
    p.add_argument("--out", default=None,
                   help="Where to write restored .wav files. Defaults to checkpoint_dir/eval/")
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dnsmos_onnx", default=DEFAULT_DNSMOS_ONNX,
                   help="Path to DNSMOS-P.835 ONNX model (`make dnsmos` fetches "
                        "it here). If absent, only restored audio is written.")
    return p.parse_args()


def load_model(checkpoint: str, device: torch.device) -> tuple[SpeechRestorationModel, dict]:
    from diamond.checkpoint_io import load_checkpoint
    ckpt = load_checkpoint(checkpoint, map_location=device)
    cfg = OmegaConf.create(ckpt["config"])
    # `decoder` is the time-transformer; `depth` is the codebook read-out. Both
    # are required — a checkpoint without `depth` predates the RQ-Transformer.
    depth = cfg.get("depth")
    if depth is None:
        raise ValueError(f"{checkpoint}: config has no `depth` section (unsupported checkpoint)")
    model = SpeechRestorationModel(
        encoder_cfg=OmegaConf.to_container(cfg.encoder, resolve=True),
        time_cfg=OmegaConf.to_container(cfg.decoder, resolve=True),
        depth_cfg=OmegaConf.to_container(depth, resolve=True),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_dnsmos(onnx_path: str | None):
    if onnx_path is None:
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping DNSMOS")
        return None
    if not Path(onnx_path).exists():
        print(f"DNSMOS model not found at {onnx_path}; skipping metric")
        return None
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def dnsmos_score(session, wav: np.ndarray, sr: int) -> dict | None:
    """DNSMOS-P.835 inference: 9.01 s windows @ 16 kHz, hop 1 s, averaged.

    Mel pre-processing is inside the ONNX graph, so raw float32 audio goes in.

    Returns BOTH scales. `*_raw` is the windowed mean of the network's own
    outputs — the scale TECH_REPORT and the model card (in the model repo)
    report, comparable only
    within those tables. `sig`/`bak`/`ovrl` put those through the polynomial
    fits shipped with DNS-Challenge, giving the 1-5 MOS scale other papers use.
    """
    if session is None:
        return None
    if sr != DNSMOS_SR:
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav = AF.resample(wav_t, sr, DNSMOS_SR).squeeze(0).numpy()

    hop = DNSMOS_SR
    if len(wav) < DNSMOS_WIN:
        wav = np.pad(wav, (0, DNSMOS_WIN - len(wav)))

    in_name = session.get_inputs()[0].name

    raw = []
    for start in range(0, len(wav) - DNSMOS_WIN + 1, hop):
        chunk = wav[start : start + DNSMOS_WIN].astype(np.float32)[None, :]
        out = session.run(None, {in_name: chunk})
        raw.append(np.asarray(out[0]).reshape(3))   # [sig, bak, ovrl], raw

    sig, bak, ovrl = (float(x) for x in np.mean(raw, axis=0))
    return {
        "sig_raw": sig,
        "bak_raw": bak,
        "ovrl_raw": ovrl,
        "sig": float(DNSMOS_POLY["sig"](sig)),
        "bak": float(DNSMOS_POLY["bak"](bak)),
        "ovrl": float(DNSMOS_POLY["ovrl"](ovrl)),
    }


def restore_one(model, mel_extractor, dac, wav: np.ndarray, sr_in: int,
                cfg, device) -> np.ndarray:
    wav_t = torch.from_numpy(wav).to(device)
    wav_24k = resample_to(wav_t, sr_in, cfg.mel.sample_rate).unsqueeze(0)
    mel = mel_extractor(wav_24k)

    # Match what dac.encode would produce: floor(samples_at_44k / 512).
    duration_sec = wav_24k.shape[-1] / cfg.mel.sample_rate
    samples_at_dac_sr = int(round(duration_sec * cfg.dac.sample_rate))
    n_tok_frames = max(1, samples_at_dac_sr // 512)

    with torch.no_grad():
        tokens = model.generate(mel, n_tok_frames=n_tok_frames)
        restored_44k = dac.decode(tokens)

    return restored_44k.squeeze(0).cpu().numpy()


def main():
    args = parse_args()
    device = torch.device(args.device)

    model, cfg = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {n_params/1e6:.2f}M params")

    mel_extractor = MelExtractor(
        sample_rate=cfg.mel.sample_rate,
        n_fft=cfg.mel.n_fft,
        hop_length=cfg.mel.hop_length,
        win_length=cfg.mel.win_length,
        n_mels=cfg.mel.n_mels,
        fmin=cfg.mel.fmin,
        fmax=cfg.mel.fmax,
        log_clamp_db=cfg.mel.log_clamp_db,
    ).to(device).eval()

    dac = FrozenDAC(
        checkpoint_path=cfg.dac.checkpoint_path
        if Path(cfg.dac.checkpoint_path).exists() else None,
        model_type=cfg.dac.model,
        device=str(device),
    )

    heldout_dir = Path(args.heldout)
    manifest = pd.read_csv(heldout_dir / "manifest.csv")
    if args.n_samples:
        manifest = manifest.head(args.n_samples)

    out_dir = Path(args.out) if args.out else Path(args.checkpoint).parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing restored audio to {out_dir}")

    dnsmos = load_dnsmos(args.dnsmos_onnx)

    rows = []
    for _, entry in tqdm(manifest.iterrows(), total=len(manifest)):
        wav, sr = sf.read(entry["audio_path"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)

        restored = restore_one(model, mel_extractor, dac, wav, sr, cfg, device)
        out_path = out_dir / f"{entry['id']}_restored.wav"
        sf.write(str(out_path), restored, dac.sample_rate, subtype="PCM_16")

        row = {"id": entry["id"], "restored_path": str(out_path)}
        if dnsmos is not None:
            base_scores = dnsmos_score(dnsmos, wav, sr)
            rest_scores = dnsmos_score(dnsmos, restored, dac.sample_rate)
            for k, v in base_scores.items():
                row[f"baseline_{k}"] = v
            for k, v in rest_scores.items():
                row[f"restored_{k}"] = v
                row[f"delta_{k}"] = v - base_scores[k]
        rows.append(row)

    df = pd.DataFrame(rows)
    report_path = out_dir / "report.csv"
    df.to_csv(report_path, index=False)
    print(f"Per-sample report: {report_path}")

    if dnsmos is not None:
        # Raw first: that is the scale the model card (in the model repo) /
        # TECH_REPORT quote, so these are the numbers to compare a run against.
        for label, keys in (
            ("raw sig_bak_ovr.onnx scale — comparable with the model card",
             ("sig_raw", "bak_raw", "ovrl_raw")),
            ("P.835 polynomial-mapped MOS (1-5) — comparable with other papers",
             ("sig", "bak", "ovrl")),
        ):
            print(f"\nDNSMOS-P.835, {label}:")
            for k in keys:
                base = df[f"baseline_{k}"].mean()
                rest = df[f"restored_{k}"].mean()
                name = k.replace("_raw", "").upper()
                print(f"  {name:4} baseline={base:.3f}  restored={rest:.3f}  Δ={rest - base:+.3f}")
        improved = (df["delta_ovrl_raw"] > 0).mean()
        print(f"\nClips improved (OVRL): {improved:.1%}")
    else:
        print("\nDNSMOS skipped (model not provided). Listen test on the .wav files in",
              out_dir)


if __name__ == "__main__":
    main()
