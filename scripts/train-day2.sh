#!/usr/bin/env bash
# Day-2 GPU chain (21 Sep 2026). The night chain lost only its last steps to a
# shutdown (grader seed 7 kept its best-on-val checkpoint, epoch 8/15), so:
#   1. score grader_v2_s7 on calibration + val (+ TTA on calibration) and
#      compare against grader_v2 (ensemble decision aid; external tests untouched)
#   2. a longer 1024 px patch-trained U-Net (v2 at 80 epochs was still improving
#      on valid: MA 0.16 -> 0.20; HE/SE below the served 512 px model)
#
#   powershell -ExecutionPolicy Bypass -File scripts\launch-night.ps1 -Script train-day2.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${VENUS_CACHE_DIR:-$HOME/venus-cache}"
LOG="$CACHE/train-day2.log"
run() { echo "=== $(date '+%F %T')  $*" | tee -a "$LOG"; bash "$ROOT/scripts/wsl-gpu.sh" "$@" 2>&1 | grep --line-buffered -av "oneDNN\|^I0000\|^W0000\|^E0000\|absl::InitializeLog" | tee -a "$LOG"; }
run backend.eval.score_grader --weights "$CACHE/models/grader_v2_s7.weights.h5" --tag grader_v2_s7 --manifests calibration val
run backend.eval.score_grader --weights "$CACHE/models/grader_v2_s7.weights.h5" --tag grader_v2_s7 --tta --manifests calibration
run backend.eval.compare_graders --tags grader_v2 grader_v2_s7 --manifest calibration
run backend.eval.compare_graders --tags grader_v2 grader_v2_s7 --manifest calibration --tta
run backend.eval.compare_graders --tags grader_v2 grader_v2_s7 --manifest val
run backend.training.train_lesion_unet --size 1024 --patch 512 --repeats 4 --epochs 160 --batch 4 --tag lesion_unet_v3_1024
echo "=== $(date '+%F %T')  DAY2 DONE" | tee -a "$LOG"
