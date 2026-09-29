"""Diamond speech restoration — inference.

degraded audio (any sr/format) -> restored 44.1 kHz studio-quality wav.

High-level use:

    from diamond import Diamond
    model = Diamond.from_pretrained("data/pretrained/diamond.safetensors")
    model.restore_path("degraded.mp3", "restored.wav")

CLI:

    diamond-restore degraded.mp3 restored.wav

The pipeline is chunked autoregressive decoding with the defenses that make a
token-AR restorer robust on real degraded audio (see the technical report):
chunking + crossfade, per-chunk warmup, tail-padding, a repetition penalty on
every codebook, and selective silence-collapse recovery. Ship defaults below
are the validated configuration (chunk 2.5s, rep-penalty 1.3, recovery on).
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from .checkpoint_io import load_checkpoint
from .model.model import SpeechRestorationModel
from .tokenize import FrozenDAC, MelExtractor, resample_to

# Depth-transformer dims for checkpoints whose config stored `depth: null`
# (early v10 runs passed depth dims in code). Matches the released model.
_V10T_DEPTH_CFG = {
    "d_model": 384, "n_heads": 6, "n_layers": 4, "ff_dim": 1536,
    "dropout": 0.1, "qk_norm": True, "layer_scale_init": 0.05,
}


# ─────────────────────────── sampling ────────────────────────────────────────
@dataclass
class SampleConfig:
    """Per-codebook decoding policy. greedy_cb=9 + rep_penalty=1.0 → pure greedy."""
    greedy_cb: int = 9          # codebooks [0, greedy_cb) greedy; rest top-p sampled
    temperature: float = 1.0
    top_p: float = 0.95
    rep_penalty: float = 1.3    # applied to ALL codebooks before argmax/sample
    rep_window: int = 96        # how many recent frames the penalty looks back over


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """logits [.., V] -> -inf outside the top-p nucleus."""
    if top_p >= 1.0:
        return logits
    sl, si = torch.sort(logits, descending=True, dim=-1)
    cp = F.softmax(sl, dim=-1).cumsum(dim=-1)
    rm = cp > top_p
    rm[..., 0] = False
    scatter = torch.zeros_like(rm).scatter_(-1, si, rm)
    return logits.masked_fill(scatter, float("-inf"))


def make_sample_fn(cfg: SampleConfig):
    """Build a per-(frame, codebook) sample_fn for model.generate.

    Returns None for pure greedy with no repetition penalty (fastest path).
    Otherwise a stateful fn: repetition penalty on every codebook (breaks AR
    repetition loops), greedy for coarse cb < greedy_cb, temp+top-p for the rest.

    History (recent tokens) is per-codebook and per-batch-row. ONE instance
    serves a SINGLE generate() call (one chunk) so history never leaks between
    chunks — restore_file makes a fresh fn per chunk.
    """
    use_sampling = cfg.greedy_cb < 9
    use_rep = cfg.rep_penalty > 1.0
    if not use_sampling and not use_rep:
        return None

    state: dict[int, list] = {}  # cb_idx -> [deque per batch row]

    def sample_fn(logits: torch.Tensor, cb_idx: int) -> torch.Tensor:
        B = logits.shape[0]
        sl = logits.float().clone()
        bufs = state.get(cb_idx)
        if bufs is None or len(bufs) != B:
            bufs = [deque(maxlen=cfg.rep_window) for _ in range(B)]
            state[cb_idx] = bufs
        if use_rep:
            for b in range(B):
                if bufs[b]:
                    idx = torch.tensor(sorted(set(bufs[b])), device=sl.device)
                    row = sl[b].index_select(0, idx)
                    row = torch.where(row > 0, row / cfg.rep_penalty, row * cfg.rep_penalty)
                    sl[b].index_copy_(0, idx, row)
        if cb_idx < cfg.greedy_cb:
            tok = sl.argmax(dim=-1)
        else:
            if cfg.temperature != 1.0:
                sl = sl / cfg.temperature
            sl = top_p_filter(sl, cfg.top_p)
            probs = F.softmax(sl, dim=-1).clamp(min=0)
            tok = torch.multinomial(probs, 1).squeeze(-1)
        for b in range(B):
            bufs[b].append(int(tok[b].item()))
        return tok

    return sample_fn


# ─────────────────────────── core decode ─────────────────────────────────────
def restore_chunk(wav_24k, model, mel_extractor, dac, cfg, sample_fn) -> np.ndarray:
    """One chunk [1, samples@24k] -> restored np array @ dac.sample_rate."""
    mel = mel_extractor(wav_24k)
    duration = wav_24k.shape[-1] / cfg.mel.sample_rate
    samples_dac = int(round(duration * cfg.dac.sample_rate))
    n_tok = max(1, math.ceil(samples_dac / 512))
    tokens = model.generate(mel, n_tok_frames=n_tok, sample_fn=sample_fn)
    restored = dac.decode(tokens).squeeze(0).cpu().numpy()
    if len(restored) > samples_dac:
        restored = restored[:samples_dac]
    elif len(restored) < samples_dac:
        restored = np.pad(restored, (0, samples_dac - len(restored)))
    return restored


def _ngram_diversity(codes_chunk, n: int = 4) -> float:
    """Unique-n-gram ratio of the content codebook (cb0). ~1.0 = healthy varied
    speech; low (<0.5) = repetition loop. 1.0 if too short. codes: [T, K] int."""
    if codes_chunk is None:
        return 1.0
    cb0 = np.asarray(codes_chunk)[:, 0]
    T = len(cb0)
    if T < n + 4:
        return 1.0
    grams = [tuple(cb0[i:i + n]) for i in range(T - n)]
    return len(set(grams)) / max(1, len(grams))


def restore_file(wav, sr, model, mel, dac, cfg, *,
                 chunk_sec: float = 2.5, overlap_sec: float = 0.4,
                 warmup_sec: float = 1.0, tail_pad_sec: float = 1.0,
                 normalize: bool = True, trim_leadin: bool = True,
                 leadin_hi_thresh: float = 0.05, max_batch: int = 4,
                 sample: SampleConfig | None = None,
                 recover_collapse: bool = True, collapse_rms: float = 0.02,
                 collapse_in_rms: float = 0.02, loop_div_thresh: float = 0.0):
    """Full restoration of one clip (numpy wav @ sr) -> (restored np, out_sr).

    Chunked AR decoding with crossfade stitching + per-chunk warmup + tail-pad +
    repetition penalty + selective silence-collapse recovery. See module docstring.
    """
    device = next(model.parameters()).device
    sample = sample or SampleConfig()

    def mk_sample_fn():
        return make_sample_fn(sample)

    wav_t = torch.from_numpy(wav.astype(np.float32)).to(device)
    wav_24k = resample_to(wav_t, sr, cfg.mel.sample_rate).unsqueeze(0)
    total = wav_24k.shape[-1]
    dur = total / cfg.mel.sample_rate
    out_sr = dac.sample_rate

    # ── Phone-recording lead-in trim: non-speech intro (handling noise / room
    # tone) is OOD; detect speech onset via 4-8kHz mel energy, trim, pad back.
    lead_in_out = 0
    if trim_leadin:
        full_mel = mel(wav_24k)
        hi_band = full_mel[0, :, 40:].mean(dim=-1)
        sf_idx = (hi_band > leadin_hi_thresh).nonzero(as_tuple=True)[0]
        if len(sf_idx) > 0 and sf_idx[0].item() > 0:
            lead_24k = int(sf_idx[0].item()) * cfg.mel.hop_length
            lead_sec = lead_24k / cfg.mel.sample_rate
            if lead_sec > 0.1:
                wav_24k = wav_24k[:, lead_24k:]
                total = wav_24k.shape[-1]
                dur = total / cfg.mel.sample_rate
                lead_in_out = int(round(lead_sec * out_sr))

    if chunk_sec <= 0 or dur <= chunk_sec:
        restored = restore_chunk(wav_24k, model, mel, dac, cfg, mk_sample_fn())
    else:
        sr24 = cfg.mel.sample_rate
        # Tail padding: append silence so AR tail-degradation lands on throwaway
        # samples, not the real final words. Trimmed back below.
        real_total = total
        tail_pad_n = int(round(max(0.0, tail_pad_sec) * sr24))
        if tail_pad_n > 0:
            wav_24k = torch.cat(
                [wav_24k, torch.zeros(1, tail_pad_n, device=wav_24k.device)], dim=-1)
            total = wav_24k.shape[-1]
            dur = total / sr24
        chunk_n = int(round(chunk_sec * sr24))
        ov_n = int(round(overlap_sec * sr24))
        step_n = chunk_n - ov_n
        starts = list(range(0, total, step_n))
        ov_out = int(round(overlap_sec * out_sr))
        total_out = int(round(dur * out_sr))
        real_total_out = int(round(real_total / sr24 * out_sr))
        acc = np.zeros(total_out + out_sr, dtype=np.float64)
        wsum = np.zeros_like(acc)
        min_n = int(0.3 * sr24)

        # Per-chunk warmup: prepend up to warmup_n samples of REAL preceding
        # audio so the decoder reaches this chunk's content already warm (no
        # mid-stream cold-start). The warm-up lead is trimmed from the output.
        warmup_n = int(round(max(0.0, warmup_sec) * sr24))
        items = []  # (ci, st, en, chunk[1,L], trim_out)
        for ci, st in enumerate(starts):
            en = min(st + chunk_n, total)
            if (en - st) < min_n:
                continue
            ws = max(0, st - warmup_n)
            chunk = wav_24k[:, ws:en]
            trim_out = int(round((st - ws) / sr24 * out_sr))
            items.append((ci, st, en, chunk, trim_out))

        # Batch-generate, grouped by equal sample-length (no pad masks needed).
        # Chunks are independent -> identical greedy output, just parallelized.
        results = {}
        results_codes = {}
        bylen = defaultdict(list)
        for it in items:
            bylen[it[3].shape[-1]].append(it)
        for L, group in bylen.items():
            samples_dac = int(round(L / sr24 * cfg.dac.sample_rate))
            n_tok = max(1, math.ceil(samples_dac / 512))
            for i in range(0, len(group), max_batch):
                sub = group[i:i + max_batch]
                wav_b = torch.cat([s[3] for s in sub], dim=0)
                mel_b = mel(wav_b)
                codes = model.generate(mel_b, n_tok_frames=n_tok, sample_fn=mk_sample_fn())
                for j, s in enumerate(sub):
                    r = dac.decode(codes[j:j + 1]).squeeze(0).cpu().numpy()
                    if len(r) > samples_dac:
                        r = r[:samples_dac]
                    elif len(r) < samples_dac:
                        r = np.pad(r, (0, samples_dac - len(r)))
                    if s[4] > 0:
                        r = r[s[4]:]
                    results[s[0]] = r
                    results_codes[s[0]] = codes[j].detach().cpu().numpy()

        # Selective collapse recovery: a chunk can (1) silence-collapse (out_rms
        # ~0 while input had speech) or (2) repetition-loop (normal RMS, low cb0
        # n-gram diversity). Both blow up CER. Re-decode ONLY those with a
        # sampling fallback (cb6-8 top-p + stronger rep-penalty); healthy greedy
        # chunks are untouched.
        if recover_collapse:
            fb = SampleConfig(greedy_cb=6, top_p=0.9, temperature=1.0,
                              rep_penalty=max(1.6, sample.rep_penalty),
                              rep_window=sample.rep_window)
            for ci, st, en, chunk, trim_out in items:
                r = results[ci]
                in_rms = float(np.sqrt(np.mean(
                    chunk.squeeze(0).detach().cpu().numpy() ** 2)) + 1e-12)
                out_rms = float(np.sqrt(np.mean(r ** 2)) + 1e-12)
                div = _ngram_diversity(results_codes.get(ci))
                silent = in_rms >= collapse_in_rms and out_rms < collapse_rms
                looped = in_rms >= collapse_in_rms and div < loop_div_thresh
                if not (silent or looped):
                    continue
                samples_dac = int(round(chunk.shape[-1] / sr24 * cfg.dac.sample_rate))
                n_tok = max(1, math.ceil(samples_dac / 512))
                codes2 = model.generate(mel(chunk), n_tok_frames=n_tok,
                                        sample_fn=make_sample_fn(fb))
                r2 = dac.decode(codes2[0:1]).squeeze(0).cpu().numpy()
                if len(r2) > samples_dac:
                    r2 = r2[:samples_dac]
                elif len(r2) < samples_dac:
                    r2 = np.pad(r2, (0, samples_dac - len(r2)))
                if trim_out > 0:
                    r2 = r2[trim_out:]
                r2_rms = float(np.sqrt(np.mean(r2 ** 2)) + 1e-12)
                div2 = _ngram_diversity(codes2[0].detach().cpu().numpy())
                accept = ((silent and r2_rms > out_rms) or
                          (looped and div2 > div and r2_rms >= collapse_rms))
                if accept:
                    results[ci] = r2

        # Crossfade-stitch in time order.
        for ci, st, en, chunk, _trim in items:
            r = results[ci]
            out_st = int(round(st / sr24 * out_sr))
            L = len(r)
            win = np.ones(L)
            if ci > 0 and L > ov_out:
                win[:ov_out] = np.linspace(0, 1, ov_out)
            if st + chunk_n < total and L > ov_out:
                win[-ov_out:] = np.linspace(1, 0, ov_out)
            seg = slice(out_st, out_st + L)
            acc[seg] += r * win
            wsum[seg] += win
        wsum[wsum < 1e-6] = 1.0
        restored = (acc / wsum)[:real_total_out].astype(np.float32)

    if normalize:
        peak = np.abs(restored).max()
        if peak > 1e-6:
            restored = restored * (0.99 / peak)
    if lead_in_out > 0:
        restored = np.concatenate([np.zeros(lead_in_out, dtype=np.float32), restored])
    return restored, out_sr


# ─────────────────────────── high-level API ──────────────────────────────────
def _load_audio(path: str):
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        import librosa
        wav, sr = librosa.load(path, sr=None, mono=False)
        wav = wav.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return wav, sr


class Diamond:
    """Loaded Diamond model ready to restore audio."""

    def __init__(self, model, mel, dac, cfg):
        self.model = model
        self.mel = mel
        self.dac = dac
        self.cfg = cfg

    @classmethod
    def from_pretrained(cls, checkpoint: str, device: str | None = None) -> "Diamond":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)
        ckpt = load_checkpoint(checkpoint, map_location=dev)
        if ckpt.get("config") is None:
            raise ValueError(
                f"{checkpoint}: no config found (need a sibling .json for a "
                ".safetensors, or a .pt with an embedded config).")
        cfg = OmegaConf.create(ckpt["config"])
        depth_cfg = (OmegaConf.to_container(cfg.depth, resolve=True)
                     if cfg.get("depth") else dict(_V10T_DEPTH_CFG))
        model = SpeechRestorationModel(
            encoder_cfg=OmegaConf.to_container(cfg.encoder, resolve=True),
            time_cfg=OmegaConf.to_container(cfg.decoder, resolve=True),
            depth_cfg=depth_cfg,
        ).to(dev).eval()
        model.load_state_dict(ckpt["model"])
        mel = MelExtractor(
            sample_rate=cfg.mel.sample_rate, n_fft=cfg.mel.n_fft,
            hop_length=cfg.mel.hop_length, win_length=cfg.mel.win_length,
            n_mels=cfg.mel.n_mels, fmin=cfg.mel.fmin, fmax=cfg.mel.fmax,
            log_clamp_db=cfg.mel.log_clamp_db,
        ).to(dev).eval()
        dac_path = cfg.dac.checkpoint_path
        dac = FrozenDAC(
            checkpoint_path=dac_path if dac_path and Path(dac_path).exists() else None,
            model_type=cfg.dac.model, device=str(dev),
        )
        return cls(model, mel, dac, cfg)

    @torch.no_grad()
    def restore(self, wav: np.ndarray, sr: int, **kw):
        """Restore a numpy waveform. Returns (restored np.ndarray, out_sr)."""
        return restore_file(wav, sr, self.model, self.mel, self.dac, self.cfg, **kw)

    def restore_path(self, in_path: str, out_path: str, **kw):
        """Restore an audio file and write a 44.1 kHz wav. Returns out_sr."""
        wav, sr = _load_audio(in_path)
        restored, out_sr = self.restore(wav, sr, **kw)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, restored, out_sr, subtype="PCM_16")
        return out_sr


# ─────────────────────────── CLI ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        prog="diamond-restore",
        description="Restore degraded speech to 44.1 kHz studio quality.")
    p.add_argument("input", help="degraded audio (wav/mp3/flac/...)")
    p.add_argument("output", help="output wav path (44.1 kHz)")
    p.add_argument("--checkpoint", default="data/pretrained/diamond.safetensors",
                   help="model weights (.safetensors or .pt)")
    p.add_argument("--device", default=None, help="cuda | cpu (auto if unset)")
    p.add_argument("--chunk_sec", type=float, default=2.5)
    p.add_argument("--overlap_sec", type=float, default=0.4)
    p.add_argument("--warmup_sec", type=float, default=1.0)
    p.add_argument("--tail_pad_sec", type=float, default=1.0)
    p.add_argument("--rep_penalty", type=float, default=1.3)
    p.add_argument("--rep_window", type=int, default=96)
    p.add_argument("--max_batch", type=int, default=4)
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--no_trim_leadin", action="store_true")
    p.add_argument("--no_recover_collapse", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dia = Diamond.from_pretrained(args.checkpoint, device=args.device)
    sample = SampleConfig(rep_penalty=args.rep_penalty, rep_window=args.rep_window)
    dev = next(dia.model.parameters()).device
    print(f"loaded Diamond on {dev}  |  input: {args.input}")
    out_sr = dia.restore_path(
        args.input, args.output,
        chunk_sec=args.chunk_sec, overlap_sec=args.overlap_sec,
        warmup_sec=args.warmup_sec, tail_pad_sec=args.tail_pad_sec,
        normalize=not args.no_normalize, trim_leadin=not args.no_trim_leadin,
        recover_collapse=not args.no_recover_collapse,
        max_batch=args.max_batch, sample=sample)
    print(f"wrote {args.output}  @ {out_sr} Hz")


if __name__ == "__main__":
    main()
