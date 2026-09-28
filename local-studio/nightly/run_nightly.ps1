# Nightly entry point (Windows). Ensures ComfyUI is up, then runs `python -m studio nightly`.
$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$logDir = Join-Path $RepoRoot "outputs\nightly"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "scheduler.log"
function Log($m) { "[$(Get-Date -Format s)] $m" | Tee-Object -FilePath $log -Append }

$cfg = Get-Content (Join-Path $RepoRoot "studio.config.local.json") -Raw | ConvertFrom-Json
$py = if ($cfg.comfy.python) { $cfg.comfy.python } else { "python" }
$url = "http://127.0.0.1:8188/system_stats"

function ComfyUp { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 $url | Out-Null; $true } catch { $false } }

$startedComfy = $false
if (-not (ComfyUp)) {
    Log "ComfyUI not running; starting it"
    Start-Process -WindowStyle Hidden -FilePath "powershell" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $RepoRoot "setup\start_comfy.ps1")
    $startedComfy = $true
    for ($i = 0; $i -lt 36 -and -not (ComfyUp); $i++) { Start-Sleep -Seconds 5 }
}
if (-not (ComfyUp)) { Log "ComfyUI failed to start"; exit 2 }

Log "Running nightly"
& $py -m studio nightly *>> $log
Log "Nightly finished with exit code $LASTEXITCODE"
# ComfyUI is left running; models were already unloaded by the nightly run.
