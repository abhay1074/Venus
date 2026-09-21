# Start a training chain in a hidden WSL session that outlives this terminal.
# A `nohup ... &` inside a wsl.exe call dies when that call returns (WSL tears
# the session down); a detached wsl.exe process running the script in the
# foreground keeps the distro up until the chain finishes.
#
#   powershell -ExecutionPolicy Bypass -File scripts\launch-night.ps1                      # train-night.sh
#   powershell -ExecutionPolicy Bypass -File scripts\launch-night.ps1 -Script train-day2.sh
#   wsl tail -f ~/venus-cache/train-night.log                                              # follow
param([string]$Script = "train-night.sh")
$root = Split-Path -Parent $PSScriptRoot
$wslRoot = "/mnt/" + $root.Substring(0, 1).ToLower() + $root.Substring(2).Replace("\", "/")
$path = "$wslRoot/scripts/$Script"
$p = Start-Process -FilePath "wsl.exe" -ArgumentList @("bash", "`"$path`"") -WindowStyle Hidden -PassThru
Write-Host "$Script started (wsl.exe pid $($p.Id)); log: ~/venus-cache/$($Script -replace '\.sh$','.log')"
