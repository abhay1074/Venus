#!/usr/bin/env bash
# Run a Venus AI Python module on the GPU inside WSL.
#
#   wsl bash scripts/wsl-gpu.sh backend.training.train_grader --epochs 15
#
# 1. LD_LIBRARY_PATH: TensorFlow's pip CUDA wheels are not on the loader path;
#    without this TF silently falls back to CPU.
# 2. PTX JIT cache: the RTX 5060 is Blackwell (sm_120) and TF 2.21 ships no
#    prebuilt kernels for it, so kernels are JIT-compiled from PTX on first use.
#    A large persistent cache on the Linux filesystem pays that once.
# 3. Caches and checkpoints live on the Linux filesystem (VENUS_CACHE_DIR), not
#    /mnt/c: memory-mapping across the 9p mount collapses under load.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${VENUS_PY:-$HOME/mediscan-env/bin/python}"
NV="$(dirname "$PY")/../lib/python3.11/site-packages/nvidia"
LP=""
while IFS= read -r dir; do LP="$LP$dir:"; done < <(find "$NV" -name lib -type d 2>/dev/null)
export LD_LIBRARY_PATH="${LP}/usr/lib/wsl/lib"
export CUDA_CACHE_PATH="$HOME/.nv/ComputeCache"
export CUDA_CACHE_MAXSIZE=4294967296
mkdir -p "$CUDA_CACHE_PATH"
export PYTHONPATH="$ROOT"
export PYTHONIOENCODING=utf-8
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export VENUS_DATA_ROOT="${VENUS_DATA_ROOT:-/mnt/c/Users/anilm/Downloads/MediScan-main/MediScan-main/backend/data/raw}"
export VENUS_CACHE_DIR="${VENUS_CACHE_DIR:-$HOME/venus-cache}"
mkdir -p "$VENUS_CACHE_DIR"
cd "$ROOT"
exec "$PY" -u -m "$@"
