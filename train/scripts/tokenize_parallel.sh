#!/usr/bin/env bash
# Tokenize every studio source in the config IN PARALLEL, one process per source.
#
# Why: a serial run leaves the GPU near 0%. The bottleneck is single-threaded HF
# streaming + audio decode, not compute — so fanning out across sources is what
# actually speeds this up. Each process holds its own DAC (~3.3 GB of VRAM), so
# check you have room for as many as you launch.
#
# Each process writes its own manifest_<prefix>.csv; merge_manifests then stitches
# them into the single manifest.csv that training reads, shuffled on disk.
#
# Usage:
#   bash scripts/tokenize_parallel.sh                 # sources from configs/pilot.yaml
#   CONFIG=configs/mine.yaml bash scripts/tokenize_parallel.sh
#   JOBS=4 bash scripts/tokenize_parallel.sh          # cap concurrent processes
set -u

CONFIG="${CONFIG:-configs/pilot.yaml}"
PYTHON="${PYTHON:-python}"
JOBS="${JOBS:-8}"
LOGDIR="${LOGDIR:-logs/tokenize}"
mkdir -p "$LOGDIR"

# Read the source list out of the config (repo:config:split:id_prefix per line).
mapfile -t SOURCES < <("$PYTHON" - "$CONFIG" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
if "defaults" in cfg:
    base = OmegaConf.load(Path(sys.argv[1]).parent / cfg.defaults)
    cfg = OmegaConf.merge(base, cfg)
for s in cfg.data.studio_sources:
    parts = [s["repo"]]
    if s.get("config"):
        parts.append(s["config"])
    parts += [s["split"], s["id_prefix"]]
    print(":".join(parts))
PY
)

if [ "${#SOURCES[@]}" -eq 0 ]; then
  echo "no studio_sources in $CONFIG"; exit 1
fi

echo "tokenizing ${#SOURCES[@]} sources from $CONFIG (up to $JOBS at a time)"

running=0
for src in "${SOURCES[@]}"; do
  prefix="${src##*:}"
  echo "  -> $prefix"
  "$PYTHON" -m diamond.cli.prepare \
      --config "$CONFIG" \
      --source "$src" \
      --manifest_name "manifest_${prefix}.csv" \
      > "$LOGDIR/${prefix}.log" 2>&1 &

  running=$((running + 1))
  if [ "$running" -ge "$JOBS" ]; then
    wait -n 2>/dev/null || wait   # -n needs bash>=4.3; plain wait is the fallback
    running=$((running - 1))
  fi
done
wait

echo "merging per-source manifests -> manifest.csv (shuffled)"
"$PYTHON" -m diamond.data.preprocessing.merge_manifests --out manifest.csv

echo "done. Logs in $LOGDIR/"
