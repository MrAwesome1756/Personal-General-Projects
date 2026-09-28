<#
.SYNOPSIS
  Installs ComfyUI + ai-toolkit for Local Studio on a Windows PC with an NVIDIA RTX 50-series GPU.

.DESCRIPTION
  Idempotent: skips anything already present. Does NOT download model weights (do that from ComfyUI's
  template browser so the exact files match the workflows). Writes studio.config.local.json with the paths.

  If ComfyUI or ai-toolkit are already installed elsewhere, pass -ComfyDir / -AiToolkitDir to reuse them.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File setup\install_windows.ps1 -InstallRoot D:\AI
#>
param(
    [string]$InstallRoot = "C:\AI",
    [string]$ComfyDir = "",
    [string]$AiToolkitDir = "",
    # RTX 50-series (Blackwell) needs a PyTorch build for CUDA 12.8 or newer.
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128",
    [string]$PythonCmd = "py -3.12",
    [switch]$SkipAiToolkit,
    [switch]$WithManager
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Need($cmd, $hint) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { throw "Missing '$cmd'. $hint" }
}

Step "Preflight"
Need "nvidia-smi" "Install the latest NVIDIA Studio/Game Ready driver."
Need "git" "Install Git for Windows: winget install --id Git.Git -e"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$pyExe, $pyArgs = $PythonCmd.Split(" ", 2)
& $pyExe $pyArgs --version
if (-not (Get-Command "ffmpeg" -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg not found. Install it: winget install --id Gyan.FFmpeg -e   (then open a new terminal)"
}
New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null

function New-Venv($dir) {
    $venvPy = Join-Path $dir ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPy)) {
        & $pyExe $pyArgs -m venv (Join-Path $dir ".venv")
        & $venvPy -m pip install --upgrade pip
        & $venvPy -m pip install torch torchvision torchaudio --index-url $TorchIndex
    }
    return $venvPy
}

# ---- ComfyUI ---------------------------------------------------------------
if (-not $ComfyDir) { $ComfyDir = Join-Path $InstallRoot "ComfyUI" }
Step "ComfyUI at $ComfyDir"
if (-not (Test-Path (Join-Path $ComfyDir "main.py"))) {
    git clone https://github.com/comfyanonymous/ComfyUI.git $ComfyDir
}
$comfyPy = New-Venv $ComfyDir
& $comfyPy -m pip install -r (Join-Path $ComfyDir "requirements.txt")
if ($WithManager) {
    $mgr = Join-Path $ComfyDir "custom_nodes\comfyui-manager"
    if (-not (Test-Path $mgr)) { git clone https://github.com/ltdrdata/ComfyUI-Manager.git $mgr }
}
& $comfyPy -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# ---- ai-toolkit (training) --------------------------------------------------
$tkPy = ""
if (-not $SkipAiToolkit) {
    if (-not $AiToolkitDir) { $AiToolkitDir = Join-Path $InstallRoot "ai-toolkit" }
    Step "ai-toolkit at $AiToolkitDir"
    if (-not (Test-Path (Join-Path $AiToolkitDir "run.py"))) {
        git clone https://github.com/ostris/ai-toolkit.git $AiToolkitDir
        Push-Location $AiToolkitDir; git submodule update --init --recursive; Pop-Location
    }
    $tkPy = New-Venv $AiToolkitDir
    & $tkPy -m pip install -r (Join-Path $AiToolkitDir "requirements.txt")
}

# ---- local config -----------------------------------------------------------
Step "Writing studio.config.local.json"
$local = [ordered]@{
    comfy = [ordered]@{
        dir = $ComfyDir
        python = $comfyPy
        loras_dir = (Join-Path $ComfyDir "models\loras")
    }
    ai_toolkit = [ordered]@{ dir = $AiToolkitDir; python = $tkPy }
}
$localPath = Join-Path $RepoRoot "studio.config.local.json"
if (Test-Path $localPath) {
    Write-Warning "$localPath exists; writing studio.config.local.suggested.json instead. Merge by hand."
    $localPath = Join-Path $RepoRoot "studio.config.local.suggested.json"
}
$local | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $localPath

Step "Done"
Write-Host @"
Next:
  1. Start ComfyUI:            powershell -File setup\start_comfy.ps1
  2. Open http://127.0.0.1:8188 and add workflows (workflows\README.md). Download models from the template browser.
  3. Check everything:         python -m studio doctor
  4. Benchmark:                python -m studio benchmark
  5. Nightly task (optional):  powershell -ExecutionPolicy Bypass -File nightly\register_nightly_task.ps1
"@
