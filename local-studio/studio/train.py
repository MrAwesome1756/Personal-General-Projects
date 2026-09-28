"""LoRA training via ostris/ai-toolkit, plus dataset checks.

A training "plan" (training/active.json) says what to train next:
{
  "enabled": true,
  "name": "materials_v1",
  "config": "training/configs/materials_v1.yaml",   # ai-toolkit config
  "output_dir": "C:/AI/ai-toolkit/output/materials_v1",  # where ai-toolkit writes .safetensors
  "trigger": "prmtx style",
  "lora_strength": 1.0
}
"""

import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

from studio import config as C
from studio import media

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


class TrainError(RuntimeError):
    pass


def load_plan(cfg, plan_file=None):
    path = C.resolve(plan_file or cfg["nightly"]["plan_file"])
    if not path or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def check_dataset(folder, trigger=None):
    folder = Path(folder)
    if not folder.is_dir():
        raise TrainError(f"Dataset folder not found: {folder}")
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    report = {"folder": str(folder), "images": len(images), "missing_captions": [], "missing_trigger": [],
              "small_images": [], "other_files": []}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() not in IMAGE_EXTS | {".txt"}:
            report["other_files"].append(p.name)
    for img in images:
        cap = img.with_suffix(".txt")
        if not cap.exists():
            report["missing_captions"].append(img.name)
        elif trigger and trigger.lower() not in cap.read_text(encoding="utf-8").lower():
            report["missing_trigger"].append(img.name)
        try:
            info = media.probe(img)
            if info["width"] and min(info["width"], info["height"]) < 768:
                report["small_images"].append(f"{img.name} ({info['width']}x{info['height']})")
        except media.MediaError:
            pass
    report["ready"] = bool(images) and not report["missing_captions"] and not report["missing_trigger"]
    return report


def run_training(cfg, plan, log_dir):
    tk = cfg.get("ai_toolkit", {})
    tk_dir = C.resolve(tk.get("dir"))
    if not tk_dir or not (tk_dir / "run.py").exists():
        raise TrainError("ai_toolkit.dir is not set (or run.py missing). Set it in studio.config.local.json.")
    python = tk.get("python") or sys.executable
    config_path = C.resolve(plan["config"])
    if not config_path.exists():
        raise TrainError(f"Training config not found: {config_path}")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{plan['name']}.log"
    started = dt.datetime.now()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run([python, "run.py", str(config_path)], cwd=tk_dir, stdout=log, stderr=subprocess.STDOUT)
    minutes = round((dt.datetime.now() - started).total_seconds() / 60, 1)
    if proc.returncode != 0:
        raise TrainError(f"ai-toolkit exited with {proc.returncode}; see {log_path}")
    return {"log": str(log_path), "minutes": minutes}


def publish_lora(cfg, plan):
    """Copy the newest trained LoRA into ComfyUI's loras folder as <name>_latest.safetensors."""
    out_dir = C.resolve(plan.get("output_dir"))
    loras_dir = C.resolve(cfg.get("comfy", {}).get("loras_dir"))
    if not out_dir or not out_dir.exists():
        raise TrainError(f"Plan output_dir not found: {out_dir}")
    if not loras_dir or not loras_dir.exists():
        raise TrainError("comfy.loras_dir is not set or missing. Set it in studio.config.local.json.")
    candidates = sorted(out_dir.rglob("*.safetensors"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise TrainError(f"No .safetensors found under {out_dir}")
    newest = candidates[-1]
    name = f"{plan['name']}_latest.safetensors"
    shutil.copyfile(newest, loras_dir / name)
    return {"source": str(newest), "lora_name": name}
