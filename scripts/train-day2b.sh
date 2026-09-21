#!/usr/bin/env bash
# Day-2 afternoon chain: the 1024 px lesion network is promoted for
# microaneurysms (backend/weights/lesion_unet_1024.weights.h5 +
# config/lesion_thresholds_1024.json), so the validation-sample analyses that
# depend on the served lesion path are re-measured on raw dataset files:
#   flag rate / attention agreement -> review policy -> docs
# Timing is measured on Windows CPU afterwards (backend.eval.timing).
#
#   powershell -ExecutionPolicy Bypass -File scripts\launch-night.ps1 -Script train-day2b.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${VENUS_CACHE_DIR:-$HOME/venus-cache}"
LOG="$CACHE/train-day2b.log"
run() { echo "=== $(date '+%F %T')  $*" | tee -a "$LOG"; bash "$ROOT/scripts/wsl-gpu.sh" "$@" 2>&1 | grep --line-buffered -av "oneDNN\|^I0000\|^W0000\|^E0000\|absl::InitializeLog" | tee -a "$LOG"; }
run backend.eval.flag_rate --n 600
run backend.eval.review_policy
run backend.eval.export_models
run backend.eval.write_docs
echo "=== $(date '+%F %T')  DAY2B DONE" | tee -a "$LOG"
