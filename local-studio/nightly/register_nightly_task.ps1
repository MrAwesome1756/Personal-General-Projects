# Registers a Windows Scheduled Task that runs the nightly loop every day.
# Default 1:30 AM. Pick a time when the 24/7 agent is quietest.
param([string]$Time = "01:30", [string]$TaskName = "LocalStudio Nightly")
$RepoRoot = Split-Path -Parent $PSScriptRoot
$script = Join-Path $RepoRoot "nightly\run_nightly.ps1"
$action = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$script`""
# Runs as the current user when logged in (the PC is on 24/7). To run while logged out, re-create
# the task in Task Scheduler with "Run whether user is logged on or not".
schtasks /Create /F /SC DAILY /ST $Time /TN $TaskName /TR $action
Write-Host "Registered '$TaskName' at $Time. Test now with: schtasks /Run /TN `"$TaskName`""
Write-Host "Remove with: schtasks /Delete /TN `"$TaskName`" /F"
