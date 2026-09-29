"""Periodic CER eval on heldout samples — drives best-ckpt selection.

Why: training loss does NOT predict inference CER. We have repeatedly seen the
lowest-loss checkpoint give WORSE greedy inference than a neighbouring one —
lower CE means a sharper softmax, which means stronger greedy mode-collapse.
We need a downstream signal to pick ckpts that actually transcribe correctly.

Design:
  • Loads Whisper-tiny.en once on CPU at start (well under 1 GB RAM)
  • Every `eval_cer_every_steps` steps: greedy-decode N heldout samples,
    DAC decode → resample 16k → Whisper transcribe → CER vs ground truth
  • No chunked inference (training-time fast eval) — caps each sample to
    `eval_seconds` (default 5s) so a 20-sample eval finishes in ~30 sec on GPU
  • Best-CER ckpt saved separately as `best_cer.pt` (does not interfere with
    periodic step_*.pt saves)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF


class CEREvaluator:
    """Holds heldout samples + ASR model; computes mean CER on demand."""

    def __init__(
        self,
        heldout_dir: str,
        n_samples: int = 20,
        eval_seconds: float = 5.0,
        whisper_model: str = "tiny.en",
        whisper_device: str = "cpu",
        whisper_compute_type: str = "int8",
    ):
        self.heldout_dir = Path(heldout_dir)
        self.n_samples = n_samples
        self.eval_seconds = float(eval_seconds)

        manifest = pd.read_csv(self.heldout_dir / "manifest.csv")
        if "text" not in manifest.columns:
            raise RuntimeError(
                f"manifest at {self.heldout_dir} lacks `text` column — re-run "
                "diamond.data.preprocessing.heldout to include it"
            )
        manifest = manifest.dropna(subset=["text"])
        manifest = manifest[manifest["text"].str.strip().str.len() > 0]
        self.manifest = manifest.head(n_samples).reset_index(drop=True)
        if len(self.manifest) == 0:
            raise RuntimeError(f"no usable heldout samples with text in {heldout_dir}")

        # Pre-load audio so we don't hit disk during training.
        self.samples: list[dict] = []
        for _, row in self.manifest.iterrows():
            audio_path = self.heldout_dir / "audio" / row.get(
                "audio_relpath", f"{row['id']}.wav"
            )
            if not audio_path.exists():
                # heldout.py writes to data/heldout/audio/<id>.wav by default
                audio_path = self.heldout_dir / "audio" / f"{row['id']}.wav"
            wav, sr = sf.read(str(audio_path), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)
            # Cap duration upfront — keeps eval fast and matches model's
            # trained context length (no AR drift from generating past T_dec).
            max_n = int(self.eval_seconds * sr)
            if len(wav) > max_n:
                wav = wav[:max_n]
            self.samples.append({
                "id": row["id"],
                "wav": wav,
                "sr": sr,
                "text": str(row["text"]).strip(),
            })

        # Lazy-load ASR — only when first eval is called (avoid cost if
        # eval never fires, e.g. crashed run).
        self._whisper = None
        self._jiwer = None
        self.whisper_model = whisper_model
        self.whisper_device = whisper_device
        self.whisper_compute_type = whisper_compute_type

    def _load_asr(self):
        if self._whisper is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise RuntimeError(
                    "faster-whisper not installed. Add to deps or "
                    "`pip install faster-whisper jiwer`"
                ) from e
            self._whisper = WhisperModel(
                self.whisper_model,
                device=self.whisper_device,
                compute_type=self.whisper_compute_type,
            )
        if self._jiwer is None:
            try:
                import jiwer
            except ImportError as e:
                raise RuntimeError(
                    "jiwer not installed. `pip install jiwer`"
                ) from e
            self._jiwer = jiwer

    def _transcribe(self, wav: np.ndarray, sr: int) -> str:
        if sr != 16000:
            wav_t = torch.from_numpy(wav).unsqueeze(0)
            wav = AF.resample(wav_t, sr, 16000).squeeze(0).numpy()
        segments, _ = self._whisper.transcribe(wav, language="en", beam_size=1)
        return " ".join(seg.text for seg in segments).strip()

    @torch.no_grad()
    def evaluate(self, model, mel_extractor, dac, cfg, device) -> dict:
        """Run greedy restoration → ASR → CER on cached heldout samples.

        Returns dict with mean_cer, median_cer, per-sample list (for logging).
        Model is left in train() mode externally — caller controls.
        """
        self._load_asr()
        from ..data.tokenize import resample_to

        was_training = model.training
        model.eval()
        try:
            per_sample = []
            for sample in self.samples:
                wav_t = torch.from_numpy(sample["wav"]).to(device)
                wav_24k = resample_to(wav_t, sample["sr"], cfg.mel.sample_rate).unsqueeze(0)
                mel = mel_extractor(wav_24k)

                dur_sec = wav_24k.shape[-1] / cfg.mel.sample_rate
                samples_at_dac = int(round(dur_sec * cfg.dac.sample_rate))
                n_tok_frames = max(1, samples_at_dac // 512)

                tokens = model.generate(mel, n_tok_frames=n_tok_frames)
                restored = dac.decode(tokens).squeeze(0).cpu().numpy()

                hyp = self._transcribe(restored, dac.sample_rate)
                ref = sample["text"]
                try:
                    cer = float(self._jiwer.cer(ref, hyp)) if ref else 1.0
                except Exception:
                    cer = 1.0
                per_sample.append({"id": sample["id"], "cer": cer, "ref": ref, "hyp": hyp})
        finally:
            if was_training:
                model.train()

        cers = [s["cer"] for s in per_sample]
        return {
            "mean_cer": float(np.mean(cers)),
            "median_cer": float(np.median(cers)),
            "max_cer": float(np.max(cers)),
            "n": len(cers),
            "per_sample": per_sample,
        }
