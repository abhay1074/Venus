#!/usr/bin/env bash
# The whole model track, in order, on the WSL GPU. Each step logs to
# $VENUS_CACHE_DIR/train.log; re-run any step alone with scripts/wsl-gpu.sh.
#
#   wsl bash scripts/train-all.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${VENUS_CACHE_DIR:-$HOME/venus-cache}"
LOG="$CACHE/train.log"
run() { echo "=== $(date '+%F %T')  $*" | tee -a "$LOG"; bash "$ROOT/scripts/wsl-gpu.sh" "$@" 2>&1 | grep -av "oneDNN\|^I0000\|^W0000\|^E0000\|absl::InitializeLog" | tee -a "$LOG"; }
run backend.training.train_grader --epochs 15 --tag grader_v2
run backend.eval.score_grader --weights "$CACHE/models/grader_v2.weights.h5" --tag grader_v2
run backend.eval.score_grader --weights "$CACHE/models/grader_v2.weights.h5" --tag grader_v2 --tta --manifests calibration external_test_ddr heldout_eyepacs_frozen
run backend.training.train_lesion_unet --epochs 60
run backend.training.train_quality --epochs 12
# Flag rate / attention-agreement analysis on validation images needs the
# checkpoints in backend/weights (the served path), so pull them first.
bash "$ROOT/scripts/pull-models.sh" | tee -a "$LOG"
run backend.eval.flag_rate --n 600
echo "=== $(date '+%F %T')  ALL DONE" | tee -a "$LOG"
