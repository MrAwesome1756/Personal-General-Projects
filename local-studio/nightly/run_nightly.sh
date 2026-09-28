#!/usr/bin/env bash
# Nightly entry point (Linux). Ensures ComfyUI is up, then runs `python -m studio nightly`.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p outputs/nightly
LOG="outputs/nightly/scheduler.log"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
PY="$(python3 -c "import json; print(json.load(open('studio.config.local.json'))['comfy'].get('python') or 'python3')")"
up() { curl -sf --max-time 5 http://127.0.0.1:8188/system_stats >/dev/null; }

if ! up; then
  log "ComfyUI not running; starting it"
  nohup bash setup/start_comfy.sh >> outputs/nightly/comfy.log 2>&1 &
  for _ in $(seq 36); do up && break; sleep 5; done
fi
up || { log "ComfyUI failed to start"; exit 2; }

log "Running nightly"
"$PY" -m studio nightly >> "$LOG" 2>&1
log "Nightly finished with exit code $?"
