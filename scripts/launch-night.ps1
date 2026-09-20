# Start scripts/train-night.sh in a hidden WSL session that outlives this
# terminal. A `nohup ... &` inside a wsl.exe call dies when that call returns
# (WSL tears the session down); a detached wsl.exe process running the script
# in the foreground keeps the distro up until the chain finishes.
#
#   powershell -ExecutionPolicy Bypass -File scripts\launch-night.ps1
#   wsl tail -f ~/venus-cache/train-night.log          # follow
$root = Split-Path -Parent $PSScriptRoot
$wslRoot = "/mnt/" + $root.Substring(0, 1).ToLower() + $root.Substring(2).Replace("\", "/")
$script = "$wslRoot/scripts/train-night.sh"
$p = Start-Process -FilePath "wsl.exe" -ArgumentList @("bash", "`"$script`"") -WindowStyle Hidden -PassThru
Write-Host "train-night.sh started (wsl.exe pid $($p.Id)); log: ~/venus-cache/train-night.log"
