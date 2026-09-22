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

# The served checkpoints (not committed, ~1 GB): grader_v2 + modality gate are
# required, quality CNN and lesion U-Net optional (each result says which path
# ran). Every file is hashed against scripts\checksums.txt: a truncated or
# half-downloaded checkpoint loads as a confusing error hours later, so it is
# caught here instead.
& .venv\Scripts\python.exe scripts\checksums.py
$hashStatus = $LASTEXITCODE
if ($hashStatus -eq 1) {
  throw "A checkpoint does not match scripts\checksums.txt (see the expected-vs-actual hash above). Re-download or re-copy that file, then re-run setup."
}
if ($hashStatus -eq 2) {
  Write-Warning "A required checkpoint is missing. Get them with 'wsl bash scripts/pull-models.sh', from the release assets, or from USB - see the 'If the checkpoints will not download' section of docs\DEMO.md. Setup continues; the API will refuse to grade until they are in place."
}

# config\operating_point.json is committed and locked to grader_v2; it is never
# regenerated here (backend.eval.calibrate is a deliberate, once-per-version step).

Push-Location frontend
npm install
Pop-Location

$env:TF_CPP_MIN_LOG_LEVEL = "3"
& .venv\Scripts\python.exe -m pytest backend\tests -q
Write-Host "`nSetup complete. Start the demo with scripts\serve.ps1"
