.PHONY: help all setup deps uv ffmpeg login dac dnsmos augmentor heldout data data-local \
        noise train-local smoke-data pull-data sanity smoke train resume eval test \
        clean check-env

# uv is auto-located: prefer one already on PATH, else the standard install
# location (~/.local/bin/uv). `make uv` installs it automatically if missing,
# so the user never has to touch PATH.
UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)

# Prefix for system-package installs: empty when already root, else `sudo`.
SUDO := $(shell if [ "$$(id -u)" != "0" ] && command -v sudo >/dev/null 2>&1; then echo sudo; fi)

# printf escapes in `make help` need bash, not dash.
SHELL := /bin/bash

PYTHON ?= $(UV) run python
CONFIG ?= pilot

# `make smoke` tokenizes a tiny subset into its own cache, separate from the
# full `make data` corpus. SMOKE_LIMIT = files kept per studio source.
SMOKE_LIMIT ?= 10
SMOKE_TOKENS := data/studio_tokens_smoke

# Full-corpus token cache for `make train`. `make data` builds it; `make train`
# depends on it and builds it on demand if missing.
TOKENS_DIR := data/studio_tokens

# Augmentation target pool — 298 MB, kept out of git, fetched from HF (public).
# The HF dataset keeps its original id; the file is stored locally under the
# name the augmentor reads (TARGET_POOL).
BB_FEATURES_REPO := nineninesix/emolia_filtered_features
BB_FEATURES_FILE := emolia_filtered_v1_bb_features.csv
TARGET_POOL := augmentation/podcast_target_pool.csv

# ─── Palette ──────────────────────────────────────────────────────────────────

CYAN  := \033[38;2;103;208;227m
BLUE  := \033[38;2;66;133;244m
GREEN := \033[0;32m
DIM   := \033[2m
BOLD  := \033[1m
RESET := \033[0m

define BANNER
	@printf "\n"
	@printf "  $(CYAN)$(BOLD)██████╗ ██╗ █████╗ ███╗   ███╗ ██████╗ ███╗   ██╗██████╗ $(RESET)\n"
	@printf "  $(CYAN)$(BOLD)██╔══██╗██║██╔══██╗████╗ ████║██╔═══██╗████╗  ██║██╔══██╗$(RESET)\n"
	@printf "  $(CYAN)$(BOLD)██║  ██║██║███████║██╔████╔██║██║   ██║██╔██╗ ██║██║  ██║$(RESET)\n"
	@printf "  $(CYAN)$(BOLD)██║  ██║██║██╔══██║██║╚██╔╝██║██║   ██║██║╚██╗██║██║  ██║$(RESET)\n"
	@printf "  $(CYAN)$(BOLD)██████╔╝██║██║  ██║██║ ╚═╝ ██║╚██████╔╝██║ ╚████║██████╔╝$(RESET)\n"
	@printf "  $(CYAN)$(BOLD)╚═════╝ ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═════╝ $(RESET)\n"
	@printf "  $(DIM)speech restoration — training & augmentation$(RESET)\n\n"
endef

# ─── Default: help ────────────────────────────────────────────────────────────

help:
	$(BANNER)
	@printf "$(BOLD)$(BLUE)Setup:$(RESET)\n"
	@printf "  $(GREEN)make setup$(RESET)                     - install everything (uv + venv + deps + ffmpeg + DAC + DNSMOS + augmentor)\n"
	@printf "  $(GREEN)make all$(RESET)                       - setup + smoke (one-shot install + verify)\n"
	@printf "  $(GREEN)make login$(RESET)                     - log in to Hugging Face (only for gated datasets / pushing ckpts)\n"
	@printf "  $(GREEN)make check-env$(RESET)                 - print environment / GPU / disk / download reachability\n"
	@printf "\n"
	@printf "$(BOLD)$(BLUE)Data:$(RESET)\n"
	@printf "  $(GREEN)make data$(RESET)                      - get the token cache (pulls the pre-tokenized corpus if the config names one)\n"
	@printf "  $(GREEN)make noise$(RESET)                     - fetch MUSAN + DNS + RIRS noise corpora if the config enables noise_aug (auto-run by train)\n"
	@printf "  $(GREEN)make data-local$(RESET) AUDIO_DIR=path - tokenize YOUR OWN clean audio into data/studio_tokens_local/\n"
	@printf "  $(GREEN)make heldout$(RESET) HELDOUT_DIR=path  - index your own degraded audio as the eval set\n"
	@printf "\n"
	@printf "$(BOLD)$(BLUE)Training (config: configs/$(CONFIG).yaml):$(RESET)\n"
	@printf "  $(GREEN)make train$(RESET)                     - build the token cache if missing, then train\n"
	@printf "  $(GREEN)make resume$(RESET) CHECKPOINT=path    - resume from a .pt checkpoint (or warm-start from .safetensors)\n"
	@printf "  $(GREEN)make train-local$(RESET)               - train on the cache built by 'make data-local'\n"
	@printf "  $(GREEN)make smoke$(RESET)                     - quick verify: mini-data + 100-step run\n"
	@printf "\n"
	@printf "$(BOLD)$(BLUE)Evaluation:$(RESET)\n"
	@printf "  $(GREEN)make eval$(RESET) CHECKPOINT=path      - DNSMOS-P.835 on the held-out set\n"
	@printf "  $(GREEN)make test$(RESET)                      - import smoke: model build + forward/backward (CPU)\n"
	@printf "\n"
	@printf "$(BOLD)$(BLUE)Examples:$(RESET)\n"
	@printf "  $(CYAN)make train CONFIG=pilot$(RESET)\n"
	@printf "  $(CYAN)make resume CHECKPOINT=checkpoints/pilot/step_00015000.pt$(RESET)\n"
	@printf "  $(CYAN)make eval CHECKPOINT=checkpoints/pilot/best_dnsmos.pt$(RESET)\n"
	@printf "\n"

# ─── One-shot ────────────────────────────────────────────────────────────────

# Single command that installs everything and verifies the pipeline end-to-end.
# Run this on a fresh clone — no further setup needed.
all: setup smoke
	@echo "🎉 All done. For full training: make data && make train CONFIG=$(CONFIG)"

# ─── Setup ────────────────────────────────────────────────────────────────────

setup: deps ffmpeg dac dnsmos augmentor
	@echo "✅ Setup complete. Run: make train CONFIG=$(CONFIG)  (or 'make smoke' to verify on a mini-set first)"

# Locate uv; install it automatically if neither PATH nor ~/.local/bin has it.
uv:
	@if [ -x "$(UV)" ] || command -v uv >/dev/null 2>&1; then \
		echo "✅ uv found: $(UV)"; \
	else \
		echo "⬇️  uv not found — installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "✅ uv installed: $(UV)"; \
	fi

# ffmpeg is required by the DSP augmentor. Install it via whatever package
# manager the host provides, so the user never has to fetch it by hand.
ffmpeg:
	@if command -v ffmpeg >/dev/null 2>&1; then \
		echo "✅ ffmpeg found: $$(command -v ffmpeg)"; \
	elif command -v apt-get >/dev/null 2>&1; then \
		echo "⬇️  Installing ffmpeg via apt..."; \
		$(SUDO) apt-get update && $(SUDO) apt-get install -y ffmpeg; \
	elif command -v dnf >/dev/null 2>&1; then \
		echo "⬇️  Installing ffmpeg via dnf..."; $(SUDO) dnf install -y ffmpeg; \
	elif command -v yum >/dev/null 2>&1; then \
		echo "⬇️  Installing ffmpeg via yum..."; $(SUDO) yum install -y ffmpeg; \
	elif command -v pacman >/dev/null 2>&1; then \
		echo "⬇️  Installing ffmpeg via pacman..."; $(SUDO) pacman -S --noconfirm ffmpeg; \
	elif command -v brew >/dev/null 2>&1; then \
		echo "⬇️  Installing ffmpeg via brew..."; brew install ffmpeg; \
	elif command -v choco >/dev/null 2>&1; then \
		echo "⬇️  Installing ffmpeg via choco..."; choco install -y ffmpeg; \
	else \
		echo "❌ No known package manager — install ffmpeg manually: https://ffmpeg.org/"; \
		exit 1; \
	fi

deps: uv
	@echo "🔧 Creating venv and installing dependencies..."
	@$(UV) venv --clear
	@$(UV) pip install -e ".[dev,eval]"

login:
	@echo "🔑 Logging in to Hugging Face..."
	@$(UV) run hf auth login
	@echo "✅ Logged in"

dac:
	@echo "⬇️  Downloading DAC checkpoint..."
	@$(PYTHON) scripts/download_pretrained.py

# DNSMOS-P.835 scorer used by `make eval` (~1.1 MB, public, no token).
dnsmos:
	@$(PYTHON) scripts/download_dnsmos.py

# The augmentor (PodcastAugmentorV4) is vendored under augmentation/, noise_bank
# included. Only the 298 MB retrieval target pool is fetched here, from a public
# HF dataset — no token needed. hf_hub_download still picks up a cached token if
# you have one, so a private mirror works too.
augmentor:
	@if [ ! -f "$(TARGET_POOL)" ]; then \
		echo "⬇️  Fetching augmentation target pool from HF (298 MB)..."; \
		$(PYTHON) -c "from huggingface_hub import hf_hub_download; hf_hub_download('$(BB_FEATURES_REPO)', '$(BB_FEATURES_FILE)', repo_type='dataset', local_dir='augmentation')"; \
		mv "augmentation/$(BB_FEATURES_FILE)" "$(TARGET_POOL)"; \
	fi
	@for f in podcast_augmentor_v4.py calibration_results.csv podcast_target_pool.csv; do \
		test -e "augmentation/$$f" || { echo "❌ MISSING augmentation/$$f"; exit 1; }; \
	done
	@echo "✅ Augmentor ready (codec + EQ + noise transfer)"

# ─── Data prep ────────────────────────────────────────────────────────────────

# Build the held-out eval set from degraded audio you supply:
#   make heldout HELDOUT_DIR=path/to/degraded_audio/
# Or stream it from an HF dataset you have access to:
#   make heldout HELDOUT_REPO=org/dataset
# There is no default source — the podcast set the model was gauged on is gated.
heldout:
	@if [ -z "$(HELDOUT_DIR)" ] && [ -z "$(HELDOUT_REPO)" ]; then \
		echo "❌ Provide the degraded audio to evaluate on:"; \
		echo "   make heldout HELDOUT_DIR=path/to/degraded_audio/   (your own .wav/.mp3/.flac)"; \
		echo "   make heldout HELDOUT_REPO=org/dataset              (stream from HF)"; \
		exit 1; \
	fi
	@echo "📊 Building held-out eval set..."
	@$(PYTHON) -m diamond.data.preprocessing.heldout --n 200 --out data/heldout/ \
		$(if $(HELDOUT_DIR),--local-dir $(HELDOUT_DIR),) \
		$(if $(HELDOUT_REPO),--repo $(HELDOUT_REPO),)

# Two ways to get the token cache, picked automatically:
#   • cfg.data.tokenized_hf_repo set → pull the pre-tokenized corpus (fast path)
#   • otherwise                      → stream the studio sources and DAC-encode
#                                      them locally (hours of GPU)
data:
	@if ls $(TOKENS_DIR)/manifest*.csv >/dev/null 2>&1; then \
		echo "✅ Studio token cache present ($(TOKENS_DIR))"; \
	else \
		repo=$$($(PYTHON) -c "from omegaconf import OmegaConf; c=OmegaConf.load('configs/$(CONFIG).yaml'); b=OmegaConf.load('configs/'+c.defaults) if 'defaults' in c else OmegaConf.create({}); print(OmegaConf.merge(b,c).data.get('tokenized_hf_repo') or '')"); \
		if [ -n "$$repo" ]; then \
			echo "📥 Pulling pre-tokenized corpus from HF ($$repo) — big download, then it unpacks..."; \
			$(PYTHON) -m diamond.data.preprocessing.pull --config configs/$(CONFIG).yaml; \
		else \
			echo "📊 Tokenizing studio audio -> DAC tokens (streams the sources; hours of GPU)..."; \
			$(PYTHON) -m diamond.cli.prepare --config configs/$(CONFIG).yaml; \
		fi; \
	fi

# Tokenize YOUR OWN studio audio instead of the published corpus:
#   make data-local AUDIO_DIR=path/to/clean_audio/
# The folder is searched recursively for .wav/.mp3/.flac/.ogg/.m4a. Those clips
# are the CLEAN targets — the augmentor degrades them online — so feed it
# studio-quality audio, not the degraded kind. Tokens land in a separate cache
# (LOCAL_TOKENS) so they never mix with the published corpus.
LOCAL_TOKENS ?= data/studio_tokens_local
data-local:
	@if [ -z "$(AUDIO_DIR)" ]; then \
		echo "❌ Error: AUDIO_DIR=path/to/clean_audio/ required"; \
		echo "Example: make data-local AUDIO_DIR=~/my_studio_wavs/"; \
		exit 1; \
	fi
	@echo "📊 Tokenizing your audio ($(AUDIO_DIR)) -> DAC tokens in $(LOCAL_TOKENS)..."
	@$(PYTHON) -m diamond.cli.prepare --config configs/$(CONFIG).yaml \
		--local-dir "$(AUDIO_DIR)" --out $(LOCAL_TOKENS)
	@echo "✅ Done. Train on it:  make train-local"

# Train on the cache built by `make data-local`.
train-local:
	@if [ ! -f "$(LOCAL_TOKENS)/manifest.csv" ]; then \
		echo "❌ No local token cache. Build it first:"; \
		echo "   make data-local AUDIO_DIR=path/to/clean_audio/"; \
		exit 1; \
	fi
	@echo "🚀 Training on your own corpus ($(LOCAL_TOKENS))..."
	@$(PYTHON) -m diamond.cli.train \
		--config configs/$(CONFIG).yaml \
		--override data.studio_tokens_dir=$(LOCAL_TOKENS) data.manifest_file=null

pull-data:
	@$(PYTHON) -m diamond.data.preprocessing.pull --config configs/$(CONFIG).yaml

sanity:
	@$(PYTHON) scripts/sanity_check.py

# ─── Train / eval ─────────────────────────────────────────────────────────────

# Tokenize a tiny subset into a dedicated cache (only if not already built).
# Keeps the full `make data` corpus untouched.
smoke-data:
	@if [ -f "$(SMOKE_TOKENS)/manifest.csv" ]; then \
		echo "✅ Smoke data cache present ($(SMOKE_TOKENS))"; \
	else \
		echo "📊 Tokenizing mini-set ($(SMOKE_LIMIT) files/source) for smoke test..."; \
		$(PYTHON) -m diamond.cli.prepare --config configs/$(CONFIG).yaml \
			--out $(SMOKE_TOKENS) --limit $(SMOKE_LIMIT); \
	fi

# data.manifest_file=null is required: pilot pins the full-corpus manifest, which
# a fresh clone does not have — the smoke set writes its own manifest.csv.
smoke: smoke-data
	@echo "🔍 Running 100-step smoke test..."
	@$(PYTHON) -m diamond.cli.train \
		--config configs/$(CONFIG).yaml \
		--override training.total_steps=100 training.batch_size=2 \
		data.n_samples=10 data.studio_tokens_dir=$(SMOKE_TOKENS) \
		data.manifest_file=null

# Fetch the external noise/RIR corpora (MUSAN + DNS + RIRS_NOISES) used by the
# noise_aug track — but ONLY when the active config enables it. A no-op when
# noise_aug.enabled=false, and idempotent when the corpora are already on disk
# (download_noise.sh skips each corpus it already has). Auto-run before `train`.
noise:
	@enabled=$$($(PYTHON) -c "from omegaconf import OmegaConf; c=OmegaConf.load('configs/$(CONFIG).yaml'); b=OmegaConf.load('configs/'+c.defaults) if 'defaults' in c else OmegaConf.create({}); m=OmegaConf.merge(b,c); na=m.data.get('noise_aug') or {}; print(bool(na.get('enabled', False)))"); \
	dir=$$($(PYTHON) -c "from omegaconf import OmegaConf; c=OmegaConf.load('configs/$(CONFIG).yaml'); b=OmegaConf.load('configs/'+c.defaults) if 'defaults' in c else OmegaConf.create({}); m=OmegaConf.merge(b,c); na=m.data.get('noise_aug') or {}; print(na.get('noise_dir') or 'data/noise_corpora')"); \
	if [ "$$enabled" != "True" ]; then \
		echo "⏭️  noise_aug disabled in configs/$(CONFIG).yaml — skipping external noise download"; \
	else \
		echo "🔊 noise_aug enabled — ensuring MUSAN + DNS + RIRS corpora in $$dir (idempotent; ~25GB on first run)..."; \
		NOISE_DEST=$$dir bash scripts/download_noise.sh; \
	fi

train: data noise
	@echo "🚀 Starting training (config: $(CONFIG))..."
	@$(PYTHON) -m diamond.cli.train --config configs/$(CONFIG).yaml

resume:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "❌ Error: please specify CHECKPOINT=path"; \
		echo "Example: make resume CHECKPOINT=checkpoints/pilot/step_15000.pt"; \
		exit 1; \
	fi
	@if [ ! -f "$(CHECKPOINT)" ]; then echo "❌ Checkpoint not found: $(CHECKPOINT)"; exit 1; fi
	@echo "🔄 Resuming training from $(CHECKPOINT)..."
	@$(PYTHON) -m diamond.cli.train \
		--config configs/$(CONFIG).yaml \
		--override checkpoint.resume_from=$(CHECKPOINT)

eval: dnsmos
	@if [ -z "$(CHECKPOINT)" ]; then echo "❌ Error: CHECKPOINT=path required"; exit 1; fi
	@if [ ! -f "data/heldout/manifest.csv" ]; then \
		echo "❌ No held-out set. Build one first:"; \
		echo "   make heldout HELDOUT_DIR=path/to/degraded_audio/"; \
		exit 1; \
	fi
	@echo "📈 Evaluating $(CHECKPOINT)..."
	@$(PYTHON) -m diamond.cli.eval \
		--checkpoint $(CHECKPOINT) \
		--heldout data/heldout/

# ─── Misc ─────────────────────────────────────────────────────────────────────

test:
	@echo "🧪 Import-smoke: model build + forward/backward (CPU, no downloads)..."
	@$(PYTHON) scripts/smoke.py

check-env:
	@echo "🔍 Environment"
	@echo ""
	@echo "Python:"
	@$(PYTHON) --version
	@echo ""
	@echo "Torch / CUDA:"
	@$(PYTHON) -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
	@echo ""
	@echo "GPU:"
	@nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || echo "(no nvidia-smi)"
	@echo ""
	@echo "ffmpeg (required by the DSP augmentor):"
	@command -v ffmpeg >/dev/null 2>&1 \
		&& ffmpeg -version 2>/dev/null | head -1 \
		|| echo "❌ ffmpeg NOT found on PATH — install it (https://ffmpeg.org/)"
	@echo ""
	@echo "Disk (the pre-tokenized corpus needs ~170 GB free while unpacking):"
	@df -h . 2>/dev/null | tail -1 || echo "(df unavailable)"
	@echo ""
	@echo "Downloads reachable:"
	@for u in "https://huggingface.co/api/datasets/$(BB_FEATURES_REPO)" \
	          "https://huggingface.co/api/datasets/mythicinfinity/libritts_r" \
	          "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"; do \
		code=$$(curl -sIL -o /dev/null -w "%{http_code}" "$$u"); \
		[ "$$code" = "200" ] && echo "  ✅ $$code  $$u" || echo "  ❌ $$code  $$u"; \
	done

clean:
	@echo "🧹 Cleaning..."
	@rm -rf .venv build dist *.egg-info release
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned"
