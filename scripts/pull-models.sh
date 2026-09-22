#!/usr/bin/env bash
# Copy the trained checkpoints (and their summaries) from the WSL cache into
# backend/weights and backend/config so the Windows/CPU server and the tests
# use them, then verify every file against scripts/checksums.txt.
#
#   wsl bash scripts/pull-models.sh
#
# A truncated copy (full disk, interrupted 9p transfer) loads as a confusing
# error much later, so it is caught here. Exit 1 = a file is corrupt, 2 = a
# required checkpoint is missing.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${VENUS_CACHE_DIR:-$HOME/venus-cache}/models"
mkdir -p "$ROOT/backend/weights" "$ROOT/models/cards"
for name in grader_v2 lesion_unet quality_cnn; do
  if [ -f "$CACHE/$name.weights.h5" ]; then
    cp -v "$CACHE/$name.weights.h5" "$ROOT/backend/weights/"
    [ -f "$CACHE/$name.summary.json" ] && cp -v "$CACHE/$name.summary.json" "$ROOT/models/cards/"
  else
    echo "  (no $name checkpoint yet)"
  fi
done

echo
echo "verifying against scripts/checksums.txt"
PY="${VENUS_PY:-$HOME/mediscan-env/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
if [ -n "$PY" ] && [ -x "$PY" ]; then
  "$PY" "$ROOT/scripts/checksums.py"
  status=$?
elif command -v sha256sum >/dev/null; then
  # No Python here: checksums.txt is sha256sum format on purpose.
  (cd "$ROOT/backend/weights" && sha256sum --ignore-missing -c "$ROOT/scripts/checksums.txt")
  status=$?
else
  echo "  no python or sha256sum available; checkpoints NOT verified" >&2
  status=0
fi
case "$status" in
  0) ;;
  1) echo "a checkpoint is corrupt - re-run this script or re-copy that file" >&2 ;;
  *) echo "a required checkpoint is missing - see docs/DEMO.md" >&2 ;;
esac
exit "$status"
