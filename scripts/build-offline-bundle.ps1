# Build the offline PHC bundle: a folder with the built front end, the API, the
# checkpoints and a one-click launcher. Copy the folder to the PHC laptop, run
# setup once (creates a venv, no network needed if the wheel cache is included)
# and start with run.bat. Everything (models, SQLite records, PDF reports) stays
# on the laptop.
#
#   powershell -ExecutionPolicy Bypass -File scripts\build-offline-bundle.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "dist\venus-ai-offline"
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Force $out | Out-Null

Push-Location (Join-Path $root "frontend")
$env:VITE_API_URL = ""      # same-origin: the API serves the site
npm run build
Pop-Location

foreach ($d in @("backend", "samples", "frontend\dist", "scripts")) {
  $dest = Join-Path $out $d
  New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
  Copy-Item -Recurse -Force (Join-Path $root $d) $dest
}
Remove-Item -Recurse -Force (Join-Path $out "backend\reports") -ErrorAction SilentlyContinue
Remove-Item -Force (Join-Path $out "backend\data\venus.sqlite") -ErrorAction SilentlyContinue
Copy-Item (Join-Path $root "README.md") $out
Copy-Item (Join-Path $root "docs\VALIDATION.md") $out -ErrorAction SilentlyContinue

# Wheel cache so the PHC setup needs no internet.
$wheels = Join-Path $out "wheels"
& (Join-Path $root ".venv\Scripts\python.exe") -m pip download -r (Join-Path $root "backend\requirements.txt") -d $wheels --only-binary=:all: --platform win_amd64 --python-version 3.11 | Out-Null

@'
@echo off
cd /d %~dp0
if not exist .venv ( python -m venv .venv && .venv\Scripts\python -m pip install --no-index --find-links wheels -r backend\requirements.txt )
set TF_CPP_MIN_LOG_LEVEL=3
set PYTHONIOENCODING=utf-8
start "" http://127.0.0.1:8000/
.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
'@ | Out-File -Encoding ascii (Join-Path $out "run.bat")
Write-Host "Offline bundle written to $out"
