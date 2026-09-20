#!/usr/bin/env bash
# Overnight model track, day 2 (20 -> 21 Sep 2026), on the WSL GPU. Nothing
# here touches the external test sets or the served checkpoints: the outputs
# land in $VENUS_CACHE_DIR/models under new tags and are compared on the
# calibration / valid splits the next morning before anything is promoted.
#
#   1. lesion cache at 1024 px  (DDR lesion split, ~760 images, a few minutes)
#   2. lesion U-Net v2: 1024 px frames, trained on 512 px lesion-biased crops
#      (microaneurysms are 1-3 px at 512; the served U-Net's MA AUPR is 0.08)
#   3. grader seed 2 (same recipe, seed 7) -> calibration predictions only,
#      for an ensemble decision against grader_v2 on the calibration set
#
#   wsl bash scripts/train-night.sh            (foreground)
#   nohup wsl bash scripts/train-night.sh &    (from Windows: see scripts/launch-night.ps1)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${VENUS_CACHE_DIR:-$HOME/venus-cache}"
LOG="$CACHE/train-night.log"
run() { echo "=== $(date '+%F %T')  $*" | tee -a "$LOG"; bash "$ROOT/scripts/wsl-gpu.sh" "$@" 2>&1 | grep --line-buffered -av "oneDNN\|^I0000\|^W0000\|^E0000\|absl::InitializeLog" | tee -a "$LOG"; }
run backend.data.cache_stage0 --only lesions --size 1024
run backend.training.train_lesion_unet --size 1024 --patch 512 --repeats 4 --epochs 80 --batch 4 --tag lesion_unet_v2_1024
run backend.training.train_grader --epochs 15 --seed 7 --tag grader_v2_s7
run backend.eval.score_grader --weights "$CACHE/models/grader_v2_s7.weights.h5" --tag grader_v2_s7 --manifests calibration val
run backend.eval.score_grader --weights "$CACHE/models/grader_v2_s7.weights.h5" --tag grader_v2_s7 --tta --manifests calibration
echo "=== $(date '+%F %T')  NIGHT DONE" | tee -a "$LOG"
