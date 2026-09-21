#!/usr/bin/env bash
# Copy the trained checkpoints (and their summaries) from the WSL cache into
# backend/weights and backend/config so the Windows/CPU server and the tests
# use them.   wsl bash scripts/pull-models.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${VENUS_CACHE_DIR:-$HOME/venus-cache}/models"
mkdir -p "$ROOT/backend/weights" "$ROOT/models/cards"
for name in grader_v2 lesion_unet lesion_unet_1024 quality_cnn; do
  if [ -f "$CACHE/$name.weights.h5" ]; then
    cp -v "$CACHE/$name.weights.h5" "$ROOT/backend/weights/"
    [ -f "$CACHE/$name.summary.json" ] && cp -v "$CACHE/$name.summary.json" "$ROOT/models/cards/"
  else
    echo "  (no $name checkpoint yet)"
  fi
done
