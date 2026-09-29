"""Training loop for SpeechRestorationModel.

Usage:
  python -m diamond.training.train --config configs/pilot.yaml
  python -m diamond.training.train --config configs/pilot.yaml \
         --override training.total_steps=100 training.batch_size=2 data.n_samples=10
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ..data.dataset import StudioPairedDataset, collate_paired
from ..data.tokenize import FrozenDAC, MelExtractor, resample_to
from ..model.model import SpeechRestorationModel
from .eval_cer import CEREvaluator
from .loss import compute_loss
from .optim import build_optimizer, build_scheduler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()


def _module_param_l2(module: torch.nn.Module) -> float:
    """Total L2 norm of a module's trainable parameters. Cheap, ~ms even for 100M."""
    total_sq = 0.0
    for p in module.parameters():
        if p.requires_grad:
            total_sq += float(p.detach().norm().item()) ** 2
    return total_sq ** 0.5


def _json_safe(obj):
    """Recursive coerce of numpy/Path/Tensor types into JSON-serializable Python.

    Used for spike_events.jsonl which records per-sample aug_info dicts that
    may contain numpy scalars (combo_idx as np.int32, jitter as np.float32, etc.)
    from the augmentor's roll record.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)  # fallback


def _dump_spike_event(
    path: Path,
    step: int,
    grad_norm: float,
    loss: float,
    lr: float,
    sample_ids: list[str],
    aug_infos: list[dict],
) -> None:
    """Append one spike-event record (per-sample IDs + aug rolls) to JSONL."""
    record = {
        "step": int(step),
        "grad_norm": float(grad_norm),
        "loss": float(loss),
        "lr": float(lr),
        "per_sample": [
            {"id": _json_safe(sid), "aug": _json_safe(aug)}
            for sid, aug in zip(sample_ids, aug_infos)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _load_with_defaults(path: str) -> DictConfig:
    """Load a config, resolving its `defaults:` chain RECURSIVELY. Single-level
    resolution silently drops the grandparent (a fine-tune config inheriting
    from pilot, which inherits from default, would lose default.yaml's mel and
    encoder sections). Walk the whole chain so config inheritance of any depth
    composes correctly."""
    cfg = OmegaConf.load(path)
    if "defaults" in cfg:
        base = _load_with_defaults(str(Path(path).parent / cfg.defaults))
        cfg = OmegaConf.merge(base, cfg)
        OmegaConf.set_struct(cfg, False)
        cfg.pop("defaults", None)
    return cfg


def load_config(path: str, overrides: list[str]) -> DictConfig:
    cfg = _load_with_defaults(path)
    OmegaConf.set_struct(cfg, False)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(overrides))
    return cfg


def load_manifest(tokens_dir: str | Path, manifest_file: str | None = None) -> list[dict]:
    """Read manifest CSV produced by diamond.data.preprocessing.prepare.

    Path resolution rule: relative `wav_path`/`tokens_path` entries (the new
    portable format) resolve against the manifest's parent dir. Absolute
    entries (legacy manifests) are passed through. Either way, downstream
    code receives absolute paths, which matters because the augmentor
    singleton `os.chdir`s into the augmentor directory on first use.
    """
    manifest_path = (Path(manifest_file) if manifest_file
                     else Path(tokens_dir) / "manifest.csv").resolve()
    try:
        df = pd.read_csv(manifest_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{manifest_path} not found. Run diamond.data.preprocessing.prepare first."
        )
    base = manifest_path.parent
    records = df.to_dict(orient="records")
    for r in records:
        for key in ("wav_path", "tokens_path"):
            p = Path(r[key])
            r[key] = str(p.resolve() if p.is_absolute() else (base / p).resolve())
    return records


def build_dataloader(cfg: DictConfig) -> DataLoader:
    manifest = load_manifest(cfg.data.studio_tokens_dir, cfg.data.get("manifest_file"))
    aug_kwargs_cfg = cfg.data.get("augmentor_kwargs")
    aug_kwargs = OmegaConf.to_container(aug_kwargs_cfg, resolve=True) if aug_kwargs_cfg else None
    dataset = StudioPairedDataset(
        manifest=manifest,
        augmentor_path=cfg.data.augmentor_path,
        audio_seconds=cfg.data.audio_seconds,
        random_crop=cfg.data.random_crop,
        online_augmentation=cfg.data.online_augmentation,
        n_samples=cfg.data.n_samples,
        output_sr=cfg.mel.sample_rate,
        disable_noise=cfg.data.get("disable_noise", False),
        augmentor_kwargs=aug_kwargs,
        start_at_zero_prob=cfg.data.get("start_at_zero_prob", 0.0),
        crop_seconds_range=(
            OmegaConf.to_container(cfg.data.crop_seconds_range, resolve=True)
            if cfg.data.get("crop_seconds_range") is not None
            else None
        ),
        noise_aug_cfg=cfg.data.get("noise_aug", None),
    )
    pad_id = cfg.decoder.pad_token_id

    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor if cfg.dataloader.num_workers > 0 else None,
        persistent_workers=cfg.dataloader.persistent_workers and cfg.dataloader.num_workers > 0,
        pin_memory=cfg.dataloader.pin_memory,
        collate_fn=partial(collate_paired, pad_token_id=pad_id),
        drop_last=True,
    )


def make_pad_masks(
    wav_lengths: torch.Tensor,
    wav_sr: int,
    token_lengths: torch.Tensor,
    mel_sr: int,
    mel_hop: int,
    n_codebooks: int,
    T_mel: int,
    T_tok: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build pad masks for encoder (mel) and the token grid.

    mel_pad_mask[b, t]      = True at valid mel frames (covered by un-padded wav).
    target_mask[b, t, k]    = True where token frame t is real (t < token_length).

    The RQ-Transformer has NO delayed pattern: the time-transformer is 1:1 with
    the DAC frames and the depth-transformer walks the K codebooks inside each
    frame. So the target grid is [B, T_tok, K] with the per-frame validity
    broadcast across all K codebooks — no per-codebook right-shift (the old
    delayed `[B, T_dec=T_tok+K-1, K]` staircase is gone).
    """
    B = wav_lengths.size(0)
    # Mel frames covered by valid wav (+1 because mel uses center=True).
    wav_in_mel_sr = (wav_lengths.to(device).float() * (mel_sr / wav_sr)).long()
    valid_mel_frames = (wav_in_mel_sr // mel_hop).clamp(max=T_mel) + 1
    valid_mel_frames = valid_mel_frames.clamp(max=T_mel)
    arange_T = torch.arange(T_mel, device=device).unsqueeze(0)
    mel_pad_mask = arange_T < valid_mel_frames.unsqueeze(1)

    K = n_codebooks
    arange_tok = torch.arange(T_tok, device=device).unsqueeze(0)         # [1, T_tok]
    tok_valid = arange_tok < token_lengths.to(device).view(B, 1)         # [B, T_tok]
    target_mask = tok_valid.unsqueeze(-1).expand(B, T_tok, K)            # [B, T_tok, K]

    return mel_pad_mask, target_mask


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer, scheduler,
                    step: int, cfg: DictConfig, wandb_run_id: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "wandb_run_id": wandb_run_id,
        },
        path,
    )
    # Also write a portable, pickle-free weights artifact (+ config sidecar)
    # for inference / eval / HF sharing. The .pt above remains the resume source.
    try:
        from diamond.checkpoint_io import export_safetensors
        export_safetensors(model, path, cfg, step)
    except Exception as e:  # noqa: BLE001 — never let export failure abort training
        print(f"[ckpt] safetensors export skipped for {path.name}: {e}")


def _push_to_hf(local_path: Path, repo_id: str, token: str, max_retries: int = 3) -> bool:
    """Sync upload with exponential backoff retry. Returns True on success.

    Sync (not daemon thread) because the prior daemon implementation lost
    checkpoints when the process crashed mid-upload — they then got pruned
    locally by keep_last_n. ~10-20s upload at 1k-step cadence is < 1% overhead
    and worth the durability.
    """
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)
    except Exception as e:
        print(f"[hf] create_repo warn (continuing): {e}")
    delay = 4
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=local_path.name,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"[hf] uploaded {local_path.name} → {repo_id}")
            return True
        except Exception as e:
            if attempt == max_retries:
                print(f"[hf] upload FAILED after {max_retries} tries: {local_path.name}: {e}")
                return False
            print(f"[hf] upload retry {attempt}/{max_retries} for {local_path.name} in {delay}s: {e}")
            time.sleep(delay)
            delay *= 2
    return False


def _catchup_uploads(save_dir: Path, repo_id: str, token: str) -> None:
    """At training start, upload any local checkpoint not yet on HF.

    Closes the race window where a previous run's checkpoint was saved
    locally but the upload daemon thread was killed before completion.
    """
    if not save_dir.exists():
        return
    local_ckpts = sorted(save_dir.glob("step_*.pt"))
    if not local_ckpts:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        remote = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    except Exception as e:
        print(f"[hf] catchup: cannot list repo (skipping): {e}")
        return
    missing = [p for p in local_ckpts if p.name not in remote]
    if not missing:
        print(f"[hf] catchup: all {len(local_ckpts)} local ckpts already on HF")
        return
    print(f"[hf] catchup: uploading {len(missing)} pending ckpt(s) before training")
    for p in missing:
        _push_to_hf(p, repo_id, token)


def _override_lr_after_resume(optimizer, scheduler, new_lr: float) -> list[float]:
    """Override scheduler.base_lrs and current optimizer LRs to honor a
    config-level LR change after resume.

    Without this, scheduler.load_state_dict restores the *old* base_lrs
    from the checkpoint and our config edit is silently ignored. The
    LambdaLR formula is `base_lr * lambda(last_epoch)`, so we set
    base_lrs to the new peak and recompute the current point on the
    schedule.
    """
    scheduler.base_lrs = [new_lr] * len(scheduler.base_lrs)
    for pg in optimizer.param_groups:
        pg["initial_lr"] = new_lr
    new_lrs = [new_lr * lmbda(scheduler.last_epoch)
               for lmbda in scheduler.lr_lambdas]
    for pg, lr in zip(optimizer.param_groups, new_lrs):
        pg["lr"] = lr
    return new_lrs


def _auto_download_resume_ckpt(local_path: Path, hf_repo: str, token: str) -> bool:
    """Download resume_from from HF if it's missing locally. Returns True
    if the file exists locally after this call.

    Closes the loop where [resume] auto-cleanup deletes the file on each
    successful resume — if the train then crashes before the next save,
    the resume_from file is gone and the next launch can't start.
    """
    if local_path.exists():
        return True
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id=hf_repo, repo_type="model",
            filename=local_path.name,
            local_dir=str(local_path.parent),
            token=token,
        )
        print(f"[resume] auto-downloaded missing {local_path.name} from HF {hf_repo}")
        return Path(path).exists()
    except Exception as e:
        print(f"[resume] auto-download failed for {local_path.name}: {e}")
        return False


def maybe_resume(cfg: DictConfig, model, optimizer, scheduler, device) -> tuple[int, str | None]:
    resume = cfg.checkpoint.resume_from
    if not resume:
        return 0, None
    resume_path = Path(resume)
    if not resume_path.exists():
        hf_repo = cfg.checkpoint.get("hf_repo")
        hf_token = os.environ.get("HF_TOKEN")
        if hf_repo and hf_token:
            _auto_download_resume_ckpt(resume_path, hf_repo, hf_token)
    ckpt = torch.load(resume, map_location=device)
    model.load_state_dict(ckpt["model"])

    reset_opt = bool(cfg.checkpoint.get("reset_optimizer_on_resume", False))
    if reset_opt:
        # Skip optimizer/scheduler restore: keep model weights but start with
        # fresh m, v moments. Use when the loaded ckpt's optimizer state is
        # poisoned by past gradient spikes (Adam moments retain "memory" of
        # spikes well past their occurrence due to beta1=0.9, beta2=0.95).
        # Scheduler still gets fresh state (base_lrs from new optimizer + step
        # counter from ckpt so cosine lr resumes at the right point).
        scheduler.last_epoch = ckpt["scheduler"]["last_epoch"]
        scheduler._step_count = ckpt["scheduler"]["_step_count"]
        # Recompute optimizer LR for the resume step from current schedule.
        new_lrs = [base * lmbda(scheduler.last_epoch)
                   for base, lmbda in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
        for pg, lr in zip(optimizer.param_groups, new_lrs):
            pg["lr"] = lr
            pg["initial_lr"] = scheduler.base_lrs[0]
        print(f"[reset] optimizer state ZEROED (Adam m, v fresh); scheduler "
              f"restored to step {scheduler.last_epoch}, lr={new_lrs[0]:.2e}")
    else:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

    wandb_run_id = ckpt.get("wandb_run_id")
    step = ckpt["step"]
    print(f"Resumed from {resume} @ step {step} (wandb_run_id={wandb_run_id})")

    # Honor config LR even when scheduler state restored an older value.
    config_lr = float(cfg.training.lr)
    if abs(scheduler.base_lrs[0] - config_lr) > 1e-9:
        old_peak = scheduler.base_lrs[0]
        new_lrs = _override_lr_after_resume(optimizer, scheduler, config_lr)
        print(f"[lr] overriding scheduler base_lr {old_peak:.2e} → "
              f"{config_lr:.2e} (current lr now {new_lrs[0]:.2e})")
    del ckpt
    # Free the resume file only when there's an HF backup. Without HF (storage
    # disabled), deleting it can lose the only copy of a critical reference
    # checkpoint. Skip deletion when hf_repo is None.
    hf_repo = cfg.checkpoint.get("hf_repo")
    if hf_repo:
        try:
            Path(resume).unlink()
            print(f"[resume] deleted local {resume} (state in memory, copy on HF)")
        except FileNotFoundError:
            pass
    else:
        print(f"[resume] keeping {resume} (no HF backup configured)")
    return step, wandb_run_id


def init_wandb(cfg: DictConfig, config_path: str, resume_run_id: str | None = None):
    if cfg.logging.framework != "wandb":
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed; skipping logging framework")
        return None
    if not os.environ.get("WANDB_API_KEY") and not Path.home().joinpath(".netrc").exists():
        print(
            "[wandb] WARN: no WANDB_API_KEY env var and no ~/.netrc found. "
            "Either run `wandb login` or `export WANDB_API_KEY=...`. Skipping wandb."
        )
        return None

    run_name = cfg.logging.get("run_name") or _auto_run_name(cfg, config_path)
    tags = list(cfg.logging.get("tags") or [])
    notes = cfg.logging.get("notes")

    # `force_new_wandb_run: true` overrides both config wandb_run_id and the
    # one stored in the checkpoint — useful when the previous run is in
    # another wandb account or when wandb's monotonic-step constraint would
    # silently drop our metrics (resume from older ckpt than wandb's last step).
    force_new = bool(cfg.logging.get("force_new_wandb_run", False))
    if force_new:
        effective_run_id = None
        print("[wandb] force_new_wandb_run=true — starting fresh run")
    else:
        # Priority: config > checkpoint
        effective_run_id = cfg.logging.get("wandb_run_id") or resume_run_id

    init_kwargs = dict(
        project=cfg.logging.project,
        name=run_name,
        tags=tags,
        notes=notes,
        config=OmegaConf.to_container(cfg, resolve=True),
        save_code=False,
    )
    if effective_run_id:
        init_kwargs["id"] = effective_run_id
        init_kwargs["resume"] = "allow"
        print(f"[wandb] resuming run id={effective_run_id}")

    run = wandb.init(**init_kwargs)
    print(f"[wandb] run: {run.name}")
    print(f"[wandb] url: {run.url}")
    return run


def _auto_run_name(cfg: DictConfig, config_path: str) -> str:
    config_stem = Path(config_path).stem
    n_steps = cfg.training.get("total_steps", 0)
    today = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    return f"{config_stem}_{n_steps}st_{today}"


def resolve_cfg_paths(cfg: DictConfig) -> None:
    """Resolve all on-disk paths to absolute. Mutates cfg in place.

    The augmentor singleton chdirs into the augmentor directory on first use;
    any relative path in cfg becomes ambiguous after that point.
    """
    cfg.data.studio_tokens_dir = str(Path(cfg.data.studio_tokens_dir).resolve())
    cfg.data.augmentor_path = str(Path(cfg.data.augmentor_path).resolve())
    if cfg.data.get("manifest_file"):
        cfg.data.manifest_file = str(Path(cfg.data.manifest_file).resolve())
    cfg.checkpoint.save_dir = str(Path(cfg.checkpoint.save_dir).resolve())
    if cfg.checkpoint.resume_from:
        cfg.checkpoint.resume_from = str(Path(cfg.checkpoint.resume_from).resolve())


def main():
    args = parse_args()
    cfg = load_config(args.config, args.override)
    resolve_cfg_paths(cfg)
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.training.device)
    use_amp = cfg.training.precision == "bf16-mixed" and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_amp else torch.float32

    # --- Model ---------------------------------------------------------------
    model = SpeechRestorationModel(
        encoder_cfg=OmegaConf.to_container(cfg.encoder, resolve=True),
        decoder_cfg=OmegaConf.to_container(cfg.decoder, resolve=True),  # = time-transformer cfg
        depth_cfg=OmegaConf.to_container(cfg.depth, resolve=True),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params/1e6:.2f}M")

    # --- Activation probes (diagnostics) ------------------------------------
    # Register forward hooks on each encoder/decoder layer to capture output
    # activation magnitudes. Helps detect saturation BEFORE gradient explosion.
    # Stored per layer in a shared dict; updated on every forward pass.
    activation_stats: dict = {}
    if cfg.logging.get("log_activation_norms", True):
        def make_hook(name):
            def hook(_m, _i, output):
                t = output if torch.is_tensor(output) else (output[0] if output else None)
                if t is None:
                    return
                with torch.no_grad():
                    activation_stats[f"{name}_norm"] = t.float().norm().item() / max(t.numel() ** 0.5, 1.0)
                    activation_stats[f"{name}_max"] = t.float().abs().max().item()
            return hook
        for li, layer in enumerate(getattr(model.encoder, "layers", [])):
            layer.register_forward_hook(make_hook(f"enc_L{li}"))
        for li, layer in enumerate(getattr(model.decoder, "layers", [])):
            layer.register_forward_hook(make_hook(f"dec_L{li}"))
        print(f"[diag] activation hooks registered on {len(getattr(model.encoder, 'layers', []))} enc + {len(getattr(model.decoder, 'layers', []))} dec layers")

    # --- Mel extractor (lives on GPU; DAC is offline-tokenized) --------------
    mel_extractor = MelExtractor(
        sample_rate=cfg.mel.sample_rate,
        n_fft=cfg.mel.n_fft,
        hop_length=cfg.mel.hop_length,
        win_length=cfg.mel.win_length,
        n_mels=cfg.mel.n_mels,
        fmin=cfg.mel.fmin,
        fmax=cfg.mel.fmax,
        log_clamp_db=cfg.mel.log_clamp_db,
    ).to(device)
    mel_extractor.eval()

    # --- CER eval (optional) — downstream signal for ckpt selection ---------
    # Training loss does NOT predict inference CER: low CE = sharper softmax =
    # stronger greedy mode-collapse, so the lowest-loss checkpoint is often not
    # the best one. We periodically restore N heldout samples → ASR → CER,
    # save best_cer.pt separately from step_*.pt cadence.
    cer_eval_cfg = cfg.get("eval", {}).get("cer", None) if hasattr(cfg, "get") else None
    cer_evaluator = None
    cer_dac = None
    if cer_eval_cfg and bool(cer_eval_cfg.get("enabled", False)):
        try:
            cer_evaluator = CEREvaluator(
                heldout_dir=cer_eval_cfg["heldout_dir"],
                n_samples=int(cer_eval_cfg.get("n_samples", 20)),
                eval_seconds=float(cer_eval_cfg.get("eval_seconds", 5.0)),
                whisper_model=cer_eval_cfg.get("whisper_model", "tiny.en"),
                whisper_device=cer_eval_cfg.get("whisper_device", "cpu"),
                whisper_compute_type=cer_eval_cfg.get("whisper_compute_type", "int8"),
            )
            cer_dac = FrozenDAC(
                checkpoint_path=cfg.dac.checkpoint_path
                if Path(cfg.dac.checkpoint_path).exists() else None,
                model_type=cfg.dac.model,
                device=cer_eval_cfg.get("dac_device", str(device)),
            )
            print(f"[eval-cer] enabled: n={cer_evaluator.n_samples} samples, "
                  f"whisper={cer_evaluator.whisper_model}, "
                  f"every {cer_eval_cfg.get('every_steps', 5000)} steps")
        except Exception as e:
            print(f"[eval-cer] DISABLED — init failed: {e}")
            cer_evaluator = None
            cer_dac = None

    # --- Optim ---------------------------------------------------------------
    optimizer = build_optimizer(
        model,
        lr=cfg.training.lr,
        betas=tuple(cfg.training.betas),
        weight_decay=cfg.training.weight_decay,
        exclude_keywords=list(cfg.training.weight_decay_exclude),
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=cfg.training.warmup_steps,
        total_steps=cfg.training.total_steps,
        min_lr_ratio=cfg.training.min_lr_ratio,
        schedule=cfg.training.get("schedule", "cosine"),
    )

    start_step, resume_wandb_id = maybe_resume(cfg, model, optimizer, scheduler, device)

    codebook_weights = torch.tensor(cfg.loss.codebook_weights, device=device)

    # --- Data ----------------------------------------------------------------
    loader = build_dataloader(cfg)
    data_iter = iter(loader)

    # --- Logging -------------------------------------------------------------
    wandb_run = init_wandb(cfg, args.config, resume_run_id=resume_wandb_id)
    save_dir = Path(cfg.checkpoint.save_dir)

    # Catch up any local ckpts not yet on HF (closes crash-mid-upload window).
    hf_repo = cfg.checkpoint.get("hf_repo")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_repo and hf_token:
        _catchup_uploads(save_dir, hf_repo, hf_token)

    grad_accum = cfg.training.grad_accumulation_steps
    total_steps = cfg.training.total_steps
    log_every = cfg.logging.log_every_steps
    save_every = cfg.checkpoint.save_every_steps
    spike_gn_threshold = float(cfg.logging.get("spike_gn_threshold", 2.0))
    spike_log_path = save_dir / "spike_events.jsonl"
    spike_count = 0

    # DEPTH-axis scheduled sampling — addresses exposure bias in the codebook
    # (depth) dimension only. BOUNDED to ≤K steps/frame (safe), unlike time-axis
    # SS, which feeds errors into an unbounded AR chain and produced a
    # gradient-spike storm the loss never recovered from.
    # Linear ramp 0 → max_prob over warmup_steps
    # from start_step, then held. Costs ~K× depth-T compute when active.
    # Gated by enabled (default OFF): turn on only after cb0 acc is reasonable
    # (≥0.4) — feeding garbage cb_{k-1} early would just add noise. Smoke first.
    ss_cfg = cfg.training.get("scheduled_sampling", None)
    ss_enabled = bool(ss_cfg.get("enabled", False)) if ss_cfg else False
    ss_max_prob = float(ss_cfg.get("max_prob", 0.25)) if (ss_cfg and ss_enabled) else 0.0
    ss_warmup = int(ss_cfg.get("warmup_steps", 20000)) if ss_cfg else 0
    ss_start_step = int(ss_cfg.get("start_step", 0)) if ss_cfg else 0
    ss_prob = 0.0  # updated each step inside loop

    best_cer = float("inf")  # tracked across CER evals; used to save best_cer.pt
    aug_silent_count = 0    # running count of augmentor → silent output failures
                            # (per-sample, replaced with clean input by dataset.py)

    # --- Train loop ----------------------------------------------------------
    model.train()
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step, total_steps):
        accum_loss = 0.0
        accum_metrics = None

        if ss_enabled:
            if step < ss_start_step:
                ss_prob = 0.0
            elif ss_warmup > 0:
                ss_prob = min(ss_max_prob, ss_max_prob * (step - ss_start_step) / ss_warmup)
            else:
                ss_prob = ss_max_prob
        # Collect sample_ids + aug_infos across all microbatches of this step,
        # so spike dump captures every sample that contributed to the gradient.
        step_sample_ids: list[str] = []
        step_aug_infos: list[dict] = []

        for micro in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            step_sample_ids.extend(batch.get("sample_ids", []))
            step_aug_infos.extend(batch.get("aug_infos", []))
            # Count silent-output augmentor failures in this microbatch (set
            # by dataset.py SAFETY check). Helps catch ffmpeg/codec regressions
            # without polluting the training data — silent samples are replaced
            # with clean input upstream.
            for info in batch.get("aug_infos", []):
                if isinstance(info, dict) and info.get("aug_silent_output"):
                    aug_silent_count += 1

            wav = batch["wav"].to(device, non_blocking=True)
            wav_sr = batch["wav_sr"]
            wav_lengths = batch["wav_lengths"]
            target_tokens = batch["clean_tokens"].to(device, non_blocking=True)
            token_lengths = batch["token_lengths"]

            wav_24k = resample_to(wav, wav_sr, cfg.mel.sample_rate)
            with torch.no_grad():
                mel = mel_extractor(wav_24k)

            # Mel distribution snapshot from the LAST microbatch only — used
            # for logging (`diag/mel_*`). Cheap (4 reductions) and tells us
            # whether the input distribution shifts over training (data drift,
            # broken augmentor, mel extractor change).
            with torch.no_grad():
                last_mel_stats = {
                    "mean": float(mel.float().mean().item()),
                    "std": float(mel.float().std().item()),
                    "min": float(mel.float().min().item()),
                    "max": float(mel.float().max().item()),
                }

            T_mel = mel.size(1)
            K = cfg.decoder.n_codebooks
            T_tok = target_tokens.size(1)

            mel_pad_mask, target_mask = make_pad_masks(
                wav_lengths=wav_lengths,
                wav_sr=wav_sr,
                token_lengths=token_lengths,
                mel_sr=cfg.mel.sample_rate,
                mel_hop=cfg.mel.hop_length,
                n_codebooks=K,
                T_mel=T_mel,
                T_tok=T_tok,
                device=device,
            )

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                # No delayed pattern. Pure teacher forcing UNLESS depth-axis
                # SS is enabled (ss_prob>0) → depth-T mixes its own cb_{k-1} preds
                # (bounded ≤K/frame, safe). ss_prob=0 → teacher forcing, the
                # MusicGen/VALL-E default.
                logits, target, tmask = model(
                    mel, target_tokens,
                    mel_pad_mask=mel_pad_mask,
                    target_mask=target_mask,
                    ss_prob=ss_prob,
                )
                loss, metrics = compute_loss(
                    logits=logits,
                    target_tokens=target,
                    target_mask=tmask,
                    codebook_weights=codebook_weights,
                    label_smoothing=cfg.loss.label_smoothing,
                    z_loss_weight=float(cfg.loss.get("z_loss_weight", 0.0)),
                )

            (loss / grad_accum).backward()
            accum_loss += loss.detach().item() / grad_accum
            if accum_metrics is None:
                accum_metrics = {k: v.clone() if torch.is_tensor(v) else v for k, v in metrics.items()}
            else:
                for k, v in metrics.items():
                    if torch.is_tensor(v):
                        accum_metrics[k] += v
            del logits, loss

        for k in accum_metrics:
            if torch.is_tensor(accum_metrics[k]):
                accum_metrics[k] = accum_metrics[k] / grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)

        # Per-block grad norms — computed NOW (before optimizer.zero_grad clears them).
        # Only when we'll log this step, to avoid overhead every step.
        will_log = (step + 1) % log_every == 0 or step == start_step
        block_grad_norms: dict = {}
        if will_log:
            with torch.no_grad():
                block_sq = {}
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    # Real param namespaces: encoder.* / time_t.* / depth_t.*
                    if name.startswith("encoder.layers."):
                        li = name.split(".")[2]
                        block = f"encoder.L{li}"
                    elif name.startswith("encoder."):
                        block = "encoder.other"
                    elif name.startswith("time_t.layers."):
                        li = name.split(".")[2]
                        block = f"time_t.L{li}"
                    elif name.startswith("time_t.embeddings"):
                        block = "time_t.embeddings"
                    elif name.startswith("time_t."):
                        block = "time_t.other"
                    elif name.startswith("depth_t.cb_output_head"):
                        block = "depth_t.heads"
                    elif name.startswith("depth_t.layers."):
                        li = name.split(".")[2]
                        block = f"depth_t.L{li}"
                    elif name.startswith("depth_t."):
                        block = "depth_t.other"
                    else:
                        block = "other"
                    g2 = (p.grad.float() ** 2).sum().item()
                    block_sq[block] = block_sq.get(block, 0.0) + g2
                block_grad_norms = {b: sq ** 0.5 for b, sq in block_sq.items()}

        if not math.isfinite(accum_loss) or not math.isfinite(float(grad_norm)):
            msg = f"non-finite at step {step+1}: loss={accum_loss}, grad_norm={float(grad_norm)}"
            print(f"[ALERT] {msg}")
            if wandb_run is not None and cfg.logging.get("alert_on_nan", True):
                try:
                    wandb_run.alert(title="Training NaN/Inf", text=msg, level="ERROR")
                except Exception as e:
                    print(f"[wandb] alert failed: {e}")
            raise RuntimeError(msg)

        # Spike forensics: when gradient norm crosses the threshold, dump every
        # sample_id + aug_info that contributed to this step's gradient. Lets us
        # later answer "which clip + which augmentor roll caused the 19-21k spike".
        if float(grad_norm) > spike_gn_threshold:
            spike_count += 1
            _dump_spike_event(
                spike_log_path,
                step=step + 1,
                grad_norm=float(grad_norm),
                loss=accum_loss,
                lr=scheduler.get_last_lr()[0],
                sample_ids=step_sample_ids,
                aug_infos=step_aug_infos,
            )
            print(f"[SPIKE] step {step+1}: gn={float(grad_norm):.2f}, "
                  f"loss={accum_loss:.4f}, "
                  f"ids[:3]={step_sample_ids[:3]} (+{len(step_sample_ids)-3} more) "
                  f"-> appended to {spike_log_path.name}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "spike/grad_norm": float(grad_norm),
                        "spike/cumulative_count": spike_count,
                        "spike/ids_head": ",".join(map(str, step_sample_ids[:5])),
                    },
                    step=step + 1,
                )

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if (step + 1) % log_every == 0 or step == start_step:
            lr = scheduler.get_last_lr()[0]
            dt = time.time() - t0
            steps_done = (step + 1) - start_step
            sps = steps_done / max(dt, 1e-6)
            per_cb_acc = accum_metrics["per_codebook_acc"].cpu().tolist()
            per_cb_str = "/".join(f"{a:.2f}" for a in per_cb_acc)
            print(
                f"step {step+1:>7}/{total_steps} | loss {accum_loss:.4f} "
                f"| acc {accum_metrics['acc_mean']:.3f} [{per_cb_str}] "
                f"| lr {lr:.2e} | gn {grad_norm:.2f} | {sps:.2f} step/s"
            )
            if wandb_run is not None:
                log_payload = {
                    "train/loss": accum_loss,
                    "train/loss_unweighted": float(accum_metrics["loss_unweighted"]),
                    "train/acc_mean": float(accum_metrics["acc_mean"]),
                    "train/grad_norm": float(grad_norm),
                    "train/lr": lr,
                    "train/steps_per_sec": sps,
                    # diagnostic norms — catch parameter / embedding drift before regression
                    "diag/encoder_param_norm": _module_param_l2(model.encoder),
                    "diag/decoder_param_norm": _module_param_l2(model.decoder),
                }
                # Per-encoder-layer + per-decoder-layer parameter norms.
                # Reveals which specific layer is growing (e.g. encoder runaway at step 12.5K).
                with torch.no_grad():
                    if hasattr(model.encoder, "layers"):
                        for li, layer in enumerate(model.encoder.layers):
                            log_payload[f"diag/enc_L{li}_param_norm"] = _module_param_l2(layer)
                    if hasattr(model.decoder, "layers"):
                        for li, layer in enumerate(model.decoder.layers):
                            log_payload[f"diag/dec_L{li}_param_norm"] = _module_param_l2(layer)
                    # Per-layer attention QKV + output projection norms.
                    # QKV grow → attention scores grow → softmax saturation risk.
                    if hasattr(model.encoder, "layers"):
                        for li, layer in enumerate(model.encoder.layers):
                            if hasattr(layer, "attn"):
                                attn = layer.attn
                                if hasattr(attn, "qkv"):
                                    log_payload[f"diag/enc_L{li}_qkv_norm"] = float(attn.qkv.weight.float().norm().item())
                                if hasattr(attn, "out"):
                                    log_payload[f"diag/enc_L{li}_attn_out_norm"] = float(attn.out.weight.float().norm().item())
                    # Per-decoder-layer cross-attention probes — feedback loop to encoder
                    if hasattr(model.decoder, "layers"):
                        for li, layer in enumerate(model.decoder.layers):
                            if hasattr(layer, "cross_attn"):
                                ca = layer.cross_attn
                                # Cross-attn separates Q (from decoder) and KV (from encoder).
                                # Names match time_transformer.py CrossAttention: q_proj / kv_proj / out.
                                # The legacy ("q", "kv", "k", "v", "qkv") names are kept for
                                # forward-compat with any future refactor.
                                for proj_name in ("q_proj", "kv_proj", "q", "kv", "k", "v", "out", "qkv"):
                                    p = getattr(ca, proj_name, None)
                                    if p is not None and hasattr(p, "weight"):
                                        log_payload[f"diag/dec_L{li}_cross_{proj_name}_norm"] = float(p.weight.float().norm().item())
                # Activation norms captured by forward hooks (from last microbatch).
                # Normalized per-element so values are scale-comparable across layers.
                # Spike in activation_max → upcoming gradient explosion.
                for k, v in activation_stats.items():
                    log_payload[f"diag/act_{k}"] = v
                for k_idx, a in enumerate(per_cb_acc):
                    log_payload[f"train/acc_cb{k_idx}"] = a
                per_cb_loss = accum_metrics["per_codebook_loss"].cpu().tolist()
                for k_idx, l in enumerate(per_cb_loss):
                    log_payload[f"train/loss_cb{k_idx}"] = l
                # Per-codebook Token Error Rate (1 - accuracy)
                per_cb_ter = accum_metrics["per_codebook_ter"].cpu().tolist()
                for k_idx, t in enumerate(per_cb_ter):
                    log_payload[f"train/ter_cb{k_idx}"] = t
                log_payload["train/ter_mean"] = sum(per_cb_ter) / len(per_cb_ter)
                # Logits / softmax saturation stats — detect degeneracy early
                # (uniform predictions = stuck, saturated softmax = numerical risk)
                per_cb_max_logit = accum_metrics["per_codebook_max_logit"].cpu().tolist()
                per_cb_logit_range = accum_metrics["per_codebook_logit_range"].cpu().tolist()
                per_cb_max_prob = accum_metrics["per_codebook_max_prob"].cpu().tolist()
                per_cb_entropy = accum_metrics["per_codebook_entropy"].cpu().tolist()
                for k_idx in range(cfg.decoder.n_codebooks):
                    log_payload[f"diag/cb{k_idx}_max_logit"] = per_cb_max_logit[k_idx]
                    log_payload[f"diag/cb{k_idx}_logit_range"] = per_cb_logit_range[k_idx]
                    log_payload[f"diag/cb{k_idx}_max_prob"] = per_cb_max_prob[k_idx]
                    log_payload[f"diag/cb{k_idx}_entropy"] = per_cb_entropy[k_idx]
                # Per-codebook embedding + output-head norms + per-row stats
                with torch.no_grad():
                    # Codebook embeddings live on the time-transformer
                    # (model.decoder alias); output heads moved to the
                    # depth-transformer (cb_output_head). Probe both.
                    depth_heads = getattr(model.depth_t, "cb_output_head", None)
                    for k_idx in range(cfg.decoder.n_codebooks):
                        emb_w = model.decoder.embeddings[k_idx].weight
                        log_payload[f"diag/cb{k_idx}_emb_norm"] = float(emb_w.norm().item())
                        if depth_heads is not None:
                            head_w = depth_heads[k_idx].weight
                            log_payload[f"diag/cb{k_idx}_head_norm"] = float(head_w.norm().item())
                        # Per-row embedding norm distribution (max/min/std) —
                        # spots dominant-row (one token's embedding way bigger than rest)
                        row_norms = emb_w.float().norm(dim=1)
                        log_payload[f"diag/cb{k_idx}_emb_row_max"] = float(row_norms.max().item())
                        log_payload[f"diag/cb{k_idx}_emb_row_min"] = float(row_norms.min().item())
                        log_payload[f"diag/cb{k_idx}_emb_row_std"] = float(row_norms.std().item())
                    # Effective rank of decoder embeddings (entropy of singular values).
                    # Falls when embeddings collapse to few dominant directions.
                    for k_idx in range(cfg.decoder.n_codebooks):
                        emb_w = model.decoder.embeddings[k_idx].weight.float()
                        s = torch.linalg.svdvals(emb_w)
                        p = s / s.sum().clamp(min=1e-12)
                        eff_rank = float(torch.exp(-(p * p.clamp(min=1e-12).log()).sum()).item())
                        log_payload[f"diag/cb{k_idx}_emb_eff_rank"] = eff_rank
                # Per-block grad norms (computed pre-zero_grad above)
                for block, gn_val in block_grad_norms.items():
                    log_payload[f"grad/{block}"] = gn_val
                log_payload["diag/spike_cumulative"] = spike_count
                log_payload["diag/aug_silent_cumulative"] = aug_silent_count
                if ss_enabled:
                    log_payload["train/scheduled_sampling_prob"] = ss_prob

                # Learnable-RoPE α per layer — only emitted when the modules
                # carry alpha_raw (rope_learnable=true). Lets us see whether
                # layers actually diverge from baseline α≈1.05 or stay near it.
                with torch.no_grad():
                    for li, layer in enumerate(getattr(model.encoder, "layers", [])):
                        rope_mod = getattr(layer, "rope", None)
                        if rope_mod is not None and getattr(rope_mod, "learnable", False):
                            log_payload[f"rope/enc_L{li}_alpha"] = float(rope_mod.alpha_value().item())
                    for li, layer in enumerate(getattr(model.decoder, "layers", [])):
                        rope_mod = getattr(layer, "rope", None)
                        if rope_mod is not None and getattr(rope_mod, "learnable", False):
                            log_payload[f"rope/dec_L{li}_alpha"] = float(rope_mod.alpha_value().item())

                # ── Extended diagnostics ─────────────────────────────────────
                # Captures the failure modes we've seen historically + the new
                # LayerScale gating mechanism, so we can tell what's actually
                # happening if a regression appears in a new shape.
                with torch.no_grad():
                    # Mel input distribution — shifts signal data drift / aug bug
                    for stat, val in last_mel_stats.items():
                        log_payload[f"diag/mel_{stat}"] = val

                    # LayerScale γ per encoder layer — the new gating mechanism.
                    # Each layer's effective contribution = γ * residual_branch.
                    # Tracks both mean (overall gate level) and max (worst-case
                    # channel) so we see if any layer collapses γ → 0 (dead
                    # branch) or balloons γ ≫ 1 (over-amplification).
                    if hasattr(model.encoder, "layers"):
                        for li, layer in enumerate(model.encoder.layers):
                            for branch in ("ls1", "ls2"):
                                ls = getattr(layer, branch, None)
                                if ls is None:
                                    continue
                                log_payload[f"diag/enc_L{li}_{branch}_mean"] = float(ls.mean().item())
                                log_payload[f"diag/enc_L{li}_{branch}_max"] = float(ls.abs().max().item())

                    # Encoder FF weight norms per layer — closes the gap left
                    # by the existing QKV/out probe. L0 runaway could have been
                    # driven by FF inflation just as well as attention.
                    if hasattr(model.encoder, "layers"):
                        for li, layer in enumerate(model.encoder.layers):
                            ff = getattr(layer, "ff", None)
                            if ff is not None:
                                if hasattr(ff, "fc1"):
                                    log_payload[f"diag/enc_L{li}_ff_fc1_norm"] = float(ff.fc1.weight.float().norm().item())
                                if hasattr(ff, "fc2"):
                                    log_payload[f"diag/enc_L{li}_ff_fc2_norm"] = float(ff.fc2.weight.float().norm().item())

                    # Decoder self-attn + FF — symmetric to encoder probes.
                    # Until now only cross-attn was instrumented; a decoder-side
                    # runaway would have been invisible.
                    if hasattr(model.decoder, "layers"):
                        for li, layer in enumerate(model.decoder.layers):
                            sa = getattr(layer, "self_attn", None)
                            if sa is not None:
                                if hasattr(sa, "qkv"):
                                    log_payload[f"diag/dec_L{li}_self_qkv_norm"] = float(sa.qkv.weight.float().norm().item())
                                if hasattr(sa, "out"):
                                    log_payload[f"diag/dec_L{li}_self_out_norm"] = float(sa.out.weight.float().norm().item())
                            ff = getattr(layer, "ff", None)
                            if ff is not None:
                                if hasattr(ff, "fc1"):
                                    log_payload[f"diag/dec_L{li}_ff_fc1_norm"] = float(ff.fc1.weight.float().norm().item())
                                if hasattr(ff, "fc2"):
                                    log_payload[f"diag/dec_L{li}_ff_fc2_norm"] = float(ff.fc2.weight.float().norm().item())

                    # RMSNorm γ weights — RMSNorm scales activations by a
                    # learnable per-channel γ. If γ blows up, it usually means
                    # the layer is using normalization to absorb scale drift
                    # from upstream (which is what we tried to prevent with
                    # norm_in). Watch encoder.norm_in.weight specifically.
                    enc = model.encoder
                    for nm in ("norm_in", "norm_out"):
                        n_mod = getattr(enc, nm, None)
                        if n_mod is not None and hasattr(n_mod, "weight"):
                            log_payload[f"diag/enc_{nm}_gamma_norm"] = float(n_mod.weight.float().norm().item())
                    if hasattr(enc, "layers"):
                        for li, layer in enumerate(enc.layers):
                            for nm in ("norm1", "norm2"):
                                n_mod = getattr(layer, nm, None)
                                if n_mod is not None and hasattr(n_mod, "weight"):
                                    log_payload[f"diag/enc_L{li}_{nm}_gamma_norm"] = float(n_mod.weight.float().norm().item())
                    dec = model.decoder
                    n_out = getattr(dec, "norm_out", None)
                    if n_out is not None and hasattr(n_out, "weight"):
                        log_payload["diag/dec_norm_out_gamma_norm"] = float(n_out.weight.float().norm().item())
                    if hasattr(dec, "layers"):
                        for li, layer in enumerate(dec.layers):
                            for nm in ("norm1", "norm2", "norm3"):
                                n_mod = getattr(layer, nm, None)
                                if n_mod is not None and hasattr(n_mod, "weight"):
                                    log_payload[f"diag/dec_L{li}_{nm}_gamma_norm"] = float(n_mod.weight.float().norm().item())

                    # Adam state per encoder/decoder layer — m, v moments tell
                    # us about the optimizer's "memory" of past gradients. A
                    # huge v on one layer means effective LR there is small;
                    # huge m means a sustained directional push. Both are
                    # signatures of the same momentum lock-in we tried to
                    # break with reset_optimizer_on_resume.
                    def _layer_adam_norms(layer_module):
                        m_sq, v_sq = 0.0, 0.0
                        for p in layer_module.parameters():
                            st = optimizer.state.get(p)
                            if not st:
                                continue
                            if "exp_avg" in st:
                                m_sq += float((st["exp_avg"].float() ** 2).sum().item())
                            if "exp_avg_sq" in st:
                                v_sq += float(st["exp_avg_sq"].float().sum().item())
                        return m_sq ** 0.5, v_sq ** 0.5
                    if hasattr(model.encoder, "layers"):
                        for li, layer in enumerate(model.encoder.layers):
                            m_n, v_n = _layer_adam_norms(layer)
                            log_payload[f"diag/enc_L{li}_adam_m_norm"] = m_n
                            log_payload[f"diag/enc_L{li}_adam_v_norm"] = v_n
                    if hasattr(model.decoder, "layers"):
                        for li, layer in enumerate(model.decoder.layers):
                            m_n, v_n = _layer_adam_norms(layer)
                            log_payload[f"diag/dec_L{li}_adam_m_norm"] = m_n
                            log_payload[f"diag/dec_L{li}_adam_v_norm"] = v_n

                wandb_run.log(log_payload, step=step + 1)

        if (step + 1) % save_every == 0 or (step + 1) == total_steps:
            ckpt_path = save_dir / f"step_{step+1:08d}.pt"
            wid = wandb_run.id if wandb_run is not None else None
            save_checkpoint(ckpt_path, model, optimizer, scheduler, step + 1, cfg, wandb_run_id=wid)
            print(f"saved {ckpt_path}")
            keep_n = cfg.checkpoint.keep_last_n
            if keep_n and keep_n > 0:
                ckpts = sorted(save_dir.glob("step_*.pt"))
                for old in ckpts[:-keep_n]:
                    old.unlink(missing_ok=True)
                    # also prune the sibling safetensors + json (dual-format save)
                    old.with_suffix(".safetensors").unlink(missing_ok=True)
                    old.with_suffix(".json").unlink(missing_ok=True)
            hf_repo = cfg.checkpoint.get("hf_repo")
            hf_token = os.environ.get("HF_TOKEN")
            if hf_repo and hf_token:
                # Sync upload (not daemon thread) — prior async impl lost
                # checkpoints when process crashed before upload completed.
                _push_to_hf(ckpt_path, hf_repo, hf_token)

        # --- CER eval + best-ckpt selection ---------------------------------
        if cer_evaluator is not None and cer_dac is not None:
            cer_every = int(cer_eval_cfg.get("every_steps", 5000))
            cer_start = int(cer_eval_cfg.get("start_step", cer_every))
            if (step + 1) >= cer_start and (step + 1) % cer_every == 0:
                t_eval0 = time.time()
                try:
                    res = cer_evaluator.evaluate(model, mel_extractor, cer_dac, cfg, device)
                    t_eval = time.time() - t_eval0
                    print(f"[eval-cer] step={step+1} mean_cer={res['mean_cer']:.4f} "
                          f"median={res['median_cer']:.4f} max={res['max_cer']:.4f} "
                          f"n={res['n']} ({t_eval:.1f}s)")
                    if wandb_run is not None:
                        wandb_run.log({
                            "eval/cer_mean": res["mean_cer"],
                            "eval/cer_median": res["median_cer"],
                            "eval/cer_max": res["max_cer"],
                            "eval/cer_eval_sec": t_eval,
                        }, step=step + 1)
                    if res["mean_cer"] < best_cer:
                        best_cer = res["mean_cer"]
                        best_cer_path = save_dir / "best_cer.pt"
                        wid = wandb_run.id if wandb_run is not None else None
                        save_checkpoint(best_cer_path, model, optimizer, scheduler,
                                        step + 1, cfg, wandb_run_id=wid)
                        print(f"[eval-cer] new best CER {best_cer:.4f} → saved {best_cer_path}")
                        if wandb_run is not None:
                            wandb_run.log({"eval/cer_best": best_cer,
                                           "eval/cer_best_step": step + 1},
                                          step=step + 1)
                except Exception as e:
                    print(f"[eval-cer] FAILED at step {step+1}: {e}")

    print("training complete")


if __name__ == "__main__":
    main()
