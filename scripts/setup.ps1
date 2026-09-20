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

$weights = @("backend\weights\eye_best.weights.h5", "backend\weights\eye_modality_gate.weights.h5")
foreach ($w in $weights) {
  if (!(Test-Path $w)) { Write-Warning "Missing $w - copy the trained checkpoints into backend\weights (see README)." }
}

if (!(Test-Path "backend\config\operating_point.json")) {
  & .venv\Scripts\python.exe -m backend.eval.calibrate
}

Push-Location frontend
npm install
Pop-Location

$env:TF_CPP_MIN_LOG_LEVEL = "3"
& .venv\Scripts\python.exe -m pytest backend\tests -q
Write-Host "`nSetup complete. Start the demo with scripts\serve.ps1"
