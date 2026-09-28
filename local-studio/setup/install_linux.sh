#!/usr/bin/env bash
# Installs ComfyUI + ai-toolkit for Local Studio on Linux with an NVIDIA RTX 50-series GPU.
# Idempotent. Does not download model weights (use ComfyUI's template browser).
# Usage: bash setup/install_linux.sh [INSTALL_ROOT]   (env: COMFY_DIR, AITK_DIR, TORCH_INDEX, PYTHON, SKIP_AITK=1)
set -euo pipefail

INSTALL_ROOT="${1:-$HOME/ai}"
COMFY_DIR="${COMFY_DIR:-$INSTALL_ROOT/ComfyUI}"
AITK_DIR="${AITK_DIR:-$INSTALL_ROOT/ai-toolkit}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"   # Blackwell needs CUDA 12.8+
PYTHON="${PYTHON:-python3.12}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

step() { printf '\n==> %s\n' "$1"; }
need() { command -v "$1" >/dev/null || { echo "Missing $1. $2" >&2; exit 1; }; }

step "Preflight"
need nvidia-smi "Install the NVIDIA driver (570+ for RTX 50-series)."
need git "sudo apt install git"
need "$PYTHON" "Install Python 3.12 (or set PYTHON=python3.11)."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg missing: sudo apt install ffmpeg"
mkdir -p "$INSTALL_ROOT"

venv() {
  local dir="$1"
  if [ ! -x "$dir/.venv/bin/python" ]; then
    "$PYTHON" -m venv "$dir/.venv"
    "$dir/.venv/bin/python" -m pip install --upgrade pip
    "$dir/.venv/bin/python" -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"
  fi
}

step "ComfyUI at $COMFY_DIR"
[ -f "$COMFY_DIR/main.py" ] || git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
venv "$COMFY_DIR"
"$COMFY_DIR/.venv/bin/python" -m pip install -r "$COMFY_DIR/requirements.txt"
"$COMFY_DIR/.venv/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

AITK_PY=""
if [ "${SKIP_AITK:-0}" != "1" ]; then
  step "ai-toolkit at $AITK_DIR"
  if [ ! -f "$AITK_DIR/run.py" ]; then
    git clone https://github.com/ostris/ai-toolkit.git "$AITK_DIR"
    (cd "$AITK_DIR" && git submodule update --init --recursive)
  fi
  venv "$AITK_DIR"
  "$AITK_DIR/.venv/bin/python" -m pip install -r "$AITK_DIR/requirements.txt"
  AITK_PY="$AITK_DIR/.venv/bin/python"
else
  AITK_DIR=""
fi

step "Writing studio.config.local.json"
LOCAL="$REPO_ROOT/studio.config.local.json"
[ -f "$LOCAL" ] && { echo "exists; writing studio.config.local.suggested.json"; LOCAL="$REPO_ROOT/studio.config.local.suggested.json"; }
cat > "$LOCAL" <<EOF
{
  "comfy": {"dir": "$COMFY_DIR", "python": "$COMFY_DIR/.venv/bin/python", "loras_dir": "$COMFY_DIR/models/loras"},
  "ai_toolkit": {"dir": "$AITK_DIR", "python": "$AITK_PY"}
}
EOF

step "Done"
cat <<EOF
Next:
  1. bash setup/start_comfy.sh
  2. open http://127.0.0.1:8188, add workflows (workflows/README.md)
  3. python3 -m studio doctor && python3 -m studio benchmark
  4. nightly: see nightly/systemd/README.md
EOF
