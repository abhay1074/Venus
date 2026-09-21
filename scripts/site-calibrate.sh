#!/usr/bin/env bash
# Calibrate Venus AI for a new site from its own labelled images (the deployment
# step measured in backend/eval/site_calibration.py: ~400 images, ~100 referable).
#
#   wsl bash scripts/site-calibrate.sh <site-name> <images-folder> <labels.csv> [--activate]
#
#   1. backend.data.site_manifest   Stage 0 over the images -> fingerprinted manifest site_<name>.csv
#   2. backend.eval.score_grader    the served grader's raw scores on it
#   3. backend.eval.calibrate       Platt + 90 % sensitivity threshold on the site, external test
#                                   scored once under version <model>+site_<name>
#                                   -> config/operating_point_site_<name>.json
#   --activate copies that file over config/operating_point.json (the server then
#   verifies the site manifest's SHA-256 on every start); without it nothing served changes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${VENUS_CACHE_DIR:-$HOME/venus-cache}"
NAME="${1:?site name}"; IMAGES="${2:?images folder}"; LABELS="${3:?labels.csv}"; ACTIVATE="${4:-}"
SITE="site_$(echo "$NAME" | tr 'A-Z ' 'a-z-')"
run() { bash "$ROOT/scripts/wsl-gpu.sh" "$@" 2>&1 | grep --line-buffered -av "oneDNN\|^I0000\|^W0000\|^E0000\|absl::InitializeLog"; }
run backend.data.site_manifest --name "$NAME" --images "$IMAGES" --labels "$LABELS"
run backend.eval.score_grader --weights "$CACHE/models/grader_v2.weights.h5" --tag grader_v2 --manifests "$SITE"
run backend.eval.calibrate --tag grader_v2 --site "$SITE"
if [ "$ACTIVATE" = "--activate" ]; then
  cp -v "$ROOT/backend/config/operating_point_$SITE.json" "$ROOT/backend/config/operating_point.json"
  echo "activated: restart the API; re-run backend.eval.flag_rate, review_policy, write_docs for the site numbers"
fi
