# Start the API (CPU, TensorFlow 2.21) and the web front end in two windows.
#
#   powershell -ExecutionPolicy Bypass -File scripts\serve.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (!(Test-Path $py)) { throw "Run scripts\setup.ps1 first." }

Start-Process powershell -ArgumentList "-NoExit", "-Command",
  "cd '$root'; `$env:TF_CPP_MIN_LOG_LEVEL='3'; `$env:PYTHONIOENCODING='utf-8'; & '$py' -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev"

Write-Host "API:      http://127.0.0.1:8000  (docs at /docs)"
Write-Host "Frontend: http://127.0.0.1:5173"
Write-Host "Deep link to a demo result: http://127.0.0.1:5173/?sample=dr_exudates.png&run=1"
