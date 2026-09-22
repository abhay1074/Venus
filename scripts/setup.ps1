# One-time setup on Windows: Python venv + dependencies, frontend packages,
# calibration (locks the operating point), and a smoke test.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (!(Test-Path ".venv")) { python -m venv .venv }
& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\python.exe -m pip install -r backend\requirements.txt

# The served checkpoints (not committed): grader_v2 + modality gate are required,
# quality CNN and lesion U-Net optional (the result says which path ran).
$weights = @("backend\weights\grader_v2.weights.h5", "backend\weights\eye_modality_gate.weights.h5",
             "backend\weights\quality_cnn.weights.h5", "backend\weights\lesion_unet.weights.h5")
foreach ($w in $weights) {
  if (!(Test-Path $w)) { Write-Warning "Missing $w - copy the trained checkpoints into backend\weights (wsl bash scripts/pull-models.sh, or from the offline bundle)." }
}

# config\operating_point.json is committed and locked to grader_v2; it is never
# regenerated here (backend.eval.calibrate is a deliberate, once-per-version step).

Push-Location frontend
npm install
Pop-Location

$env:TF_CPP_MIN_LOG_LEVEL = "3"
& .venv\Scripts\python.exe -m pytest backend\tests -q
Write-Host "`nSetup complete. Start the demo with scripts\serve.ps1"
