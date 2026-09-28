#!/usr/bin/env bash
# Starts ComfyUI bound to localhost only (never expose it to the network/internet).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="$REPO_ROOT/studio.config.local.json"
[ -f "$CFG" ] || { echo "studio.config.local.json not found; run setup/install_linux.sh" >&2; exit 1; }
read -r DIR PY < <(python3 -c "import json,sys; c=json.load(open(sys.argv[1]))['comfy']; print(c['dir'], c.get('python') or 'python3')" "$CFG")
cd "$DIR"
exec "$PY" main.py --listen 127.0.0.1 --port "${PORT:-8188}" "$@"
