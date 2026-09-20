#!/usr/bin/env bash
# After scripts/train-all.sh: pull the checkpoints into the repo, lock the v2
# operating point (calibration set only; external test scored once), export
# for MATLAB, regenerate the docs. Run from WSL:   wsl bash scripts/finalise-models.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
bash scripts/pull-models.sh
bash scripts/wsl-gpu.sh backend.eval.calibrate --tag grader_v2 --model-version venus-dr-2.0.0 "$@"
bash scripts/wsl-gpu.sh backend.eval.export_models
bash scripts/wsl-gpu.sh backend.eval.write_docs
echo
echo "Next, on Windows:  .venv\Scripts\python -m pytest backend\tests -q"
echo "                   .venv\Scripts\python -m backend.eval.timing --images 50"
echo "                   restart the API (the sweep re-runs itself when sweep_cache.json is deleted)"
