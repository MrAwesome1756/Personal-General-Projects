# Starts ComfyUI bound to localhost only (never expose it to the network/internet).
param([int]$Port = 8188, [string]$ExtraArgs = "")
$RepoRoot = Split-Path -Parent $PSScriptRoot
$cfgPath = Join-Path $RepoRoot "studio.config.local.json"
if (-not (Test-Path $cfgPath)) { throw "studio.config.local.json not found. Run setup\install_windows.ps1 or create it (see docs)." }
$cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
$py = if ($cfg.comfy.python) { $cfg.comfy.python } else { "python" }
Set-Location $cfg.comfy.dir
$argsList = @("main.py", "--listen", "127.0.0.1", "--port", "$Port")
if ($ExtraArgs) { $argsList += $ExtraArgs.Split(" ") }
& $py @argsList
