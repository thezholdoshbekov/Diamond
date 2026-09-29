#!/usr/bin/env bash
# Download the external noise + RIR corpora used by the noise_aug track.
#   • MUSAN noise subset  (OpenSLR 17)  — free-sound + sound-bible, ~6h loose wav
#   • DNS-Challenge noise (~180h)        — interspeech2020 set, git-lfs noise-only
#   • OpenSLR 28 RIRS_NOISES (~1.3GB)    — room impulse responses + isotropic/point noises
# Music is deliberately NOT downloaded — it is not a degradation we model.
# Lands loose .wav under $DEST; point data.noise_aug.noise_dir / rir_dir at it.
set -euo pipefail

DEST="${NOISE_DEST:-data/noise_corpora}"
mkdir -p "$DEST"
cd "$DEST"
echo "=== noise corpora → $DEST ==="

# ─── 1. MUSAN noise (extract only musan/noise from the full tar) ──────────────
if [ ! -d "$DEST/musan_noise/noise" ]; then
  echo "[musan] downloading musan.tar.gz (~11GB) …"
  wget -q --show-progress -O musan.tar.gz https://www.openslr.org/resources/17/musan.tar.gz
  echo "[musan] extracting musan/noise/ only …"
  mkdir -p musan_noise
  tar -xzf musan.tar.gz -C musan_noise --strip-components=1 musan/noise
  rm -f musan.tar.gz
  echo "[musan] done: $(find musan_noise/noise -name '*.wav' | wc -l) wavs"
else
  echo "[musan] already present — skip"
fi

# ─── 2. OpenSLR 28 RIRS_NOISES ────────────────────────────────────────────────
if [ ! -d "$DEST/RIRS_NOISES" ]; then
  echo "[rirs] downloading rirs_noises.zip (~1.3GB) …"
  wget -q --show-progress -O rirs_noises.zip https://www.openslr.org/resources/28/rirs_noises.zip
  unzip -q rirs_noises.zip
  rm -f rirs_noises.zip
  echo "[rirs] done: $(find RIRS_NOISES -name '*.wav' | wc -l) wavs"
else
  echo "[rirs] already present — skip"
fi

# ─── 3. DNS-Challenge ~180h noise (git-lfs, noise folder only) ────────────────
# interspeech2020/master branch stores ~65k noise wavs under datasets/noise/.
# GIT_LFS_SKIP_SMUDGE=1 on clone avoids pulling the ~500h clean speech set;
# `git lfs pull --include` then fetches ONLY the noise blobs.
if [ ! -d "$DEST/DNS-Challenge/datasets/noise" ] || \
   [ -z "$(find "$DEST/DNS-Challenge/datasets/noise" -name '*.wav' 2>/dev/null | head -1)" ]; then
  echo "[dns] cloning interspeech2020/master (metadata only) …"
  if [ ! -d "$DEST/DNS-Challenge/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone -b interspeech2020/master \
      https://github.com/microsoft/DNS-Challenge.git DNS-Challenge
  fi
  cd "$DEST/DNS-Challenge"
  echo "[dns] git lfs pull datasets/noise/* (~180h, ~20GB) …"
  git lfs pull --include="datasets/noise/*"
  echo "[dns] done: $(find datasets/noise -name '*.wav' | wc -l) wavs"
  cd "$DEST"
else
  echo "[dns] already present — skip"
fi

echo "=== ALL NOISE CORPORA READY ==="
du -sh "$DEST"/* 2>/dev/null || true
