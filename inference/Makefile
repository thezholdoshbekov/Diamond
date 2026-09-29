# Diamond speech restoration — inference.
#
# CPU works but is slow; a CUDA GPU is strongly recommended.

# printf escapes in `make help` need bash, not dash.
SHELL := /bin/bash

PYTHON ?= python
CKPT   ?= data/pretrained/diamond.safetensors
INPUT  ?=
OUTPUT ?= restored.wav

.PHONY: help setup install download infer check-env clean

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
	@printf "  $(DIM)speech restoration — inference$(RESET)\n\n"
endef

# ─── Default: help ────────────────────────────────────────────────────────────

help:
	$(BANNER)
	@printf "$(BOLD)$(BLUE)Setup:$(RESET)\n"
	@printf "  $(GREEN)make setup$(RESET)                     - install the package + download the weights\n"
	@printf "  $(GREEN)make install$(RESET)                   - install the package only\n"
	@printf "  $(GREEN)make download$(RESET)                  - pull the Diamond checkpoint into data/pretrained/\n"
	@printf "  $(GREEN)make check-env$(RESET)                 - print Python / GPU / checkpoint status\n"
	@printf "\n"
	@printf "$(BOLD)$(BLUE)Restore:$(RESET)\n"
	@printf "  $(GREEN)make infer$(RESET) INPUT=in.mp3        - restore a file (OUTPUT defaults to restored.wav)\n"
	@printf "  $(GREEN)make clean$(RESET)                     - remove build artifacts\n"
	@printf "\n"
	@printf "$(BOLD)$(BLUE)Examples:$(RESET)\n"
	@printf "  $(CYAN)make infer INPUT=degraded.mp3 OUTPUT=restored.wav$(RESET)\n"
	@printf "  $(CYAN)make infer INPUT=degraded.mp3 CKPT=path/to/other.safetensors$(RESET)\n"
	@printf "\n"

# ─── Setup ────────────────────────────────────────────────────────────────────

setup: install download
	@printf "✅ Ready. Try:  make infer INPUT=degraded.wav OUTPUT=restored.wav\n"

install:
	$(PYTHON) -m pip install -e .

download:
	$(PYTHON) scripts/download_pretrained.py --out data/pretrained

# ─── Inference ────────────────────────────────────────────────────────────────

infer:
	@if [ -z "$(INPUT)" ]; then \
		printf "❌ Usage: make infer INPUT=degraded.wav [OUTPUT=restored.wav]\n"; exit 1; fi
	@if [ ! -f "$(CKPT)" ]; then \
		printf "❌ Checkpoint $(CKPT) not found — run 'make setup' first\n"; exit 1; fi
	@printf "🎧 Restoring $(INPUT) -> $(OUTPUT) with $(CKPT)...\n"
	$(PYTHON) -m diamond.cli.infer "$(INPUT)" "$(OUTPUT)" --checkpoint "$(CKPT)"

# ─── Misc ─────────────────────────────────────────────────────────────────────

check-env:
	@printf "🔍 Environment\n\n"
	@printf "Python:  "; $(PYTHON) --version
	@printf "Torch:   "; $(PYTHON) -c "import torch; print(torch.__version__, '| CUDA', torch.version.cuda, '| available', torch.cuda.is_available())"
	@printf "GPU:\n"; nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || printf "  (no nvidia-smi)\n"
	@printf "Checkpoint:  "
	@if [ -f "$(CKPT)" ]; then printf "✅ $(CKPT)\n"; else printf "❌ $(CKPT) — run 'make setup'\n"; fi

clean:
	rm -rf build *.egg-info
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@printf "✅ Cleaned\n"
