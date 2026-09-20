#!/usr/bin/env bash
# Serve the API from WSL on the GPU (TensorFlow 2.21 in ~/mediscan-env, or any
# env with the requirements installed). The Windows front end talks to it the
# same way; only the timing column changes.
#
#   wsl bash scripts/serve-gpu.sh 8000
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${VENUS_PY:-$HOME/mediscan-env/bin/python}"
NV="$(dirname "$PY")/../lib/python3.11/site-packages/nvidia"
LP=""
while IFS= read -r dir; do LP="$LP$dir:"; done < <(find "$NV" -name lib -type d 2>/dev/null)
export LD_LIBRARY_PATH="${LP}/usr/lib/wsl/lib"
export PYTHONPATH="$ROOT" TF_CPP_MIN_LOG_LEVEL=2 PYTHONIOENCODING=utf-8
cd "$ROOT"
exec "$PY" -m uvicorn backend.main:app --host 0.0.0.0 --port "${1:-8000}"
