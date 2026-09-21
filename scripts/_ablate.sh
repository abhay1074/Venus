#!/usr/bin/env bash
cd "/mnt/c/Users/anilm/OneDrive/Venus AI SIH 26"
export VENUS_UNET_HIRES_SPEC=/nonexistent.json
bash scripts/wsl-gpu.sh backend.eval.flag_rate --n 600 --out "$HOME/venus-cache/validation_flags_no_hires.json" 2>&1 | grep --line-buffered -av "oneDNN\|^I0000\|^W0000\|^E0000\|absl\|retracing\|tf.function" > "$HOME/venus-cache/ablate.log"
