"""GPU status via nvidia-smi, and a polite wait-for-free-GPU guard.

The 5090 also runs a 24/7 agent. Before heavy work (video, training) we check
free VRAM and utilization so we don't starve or crash whatever else is running.
"""

import shutil
import subprocess
import time


def _smi(args):
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    out = subprocess.run([exe, *args], capture_output=True, text=True, timeout=20)
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def status(index=0):
    raw = _smi(["--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"])
    if raw is None:
        return None
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if int(parts[0]) != int(index):
            continue

        def num(v):
            try:
                return float(v)
            except ValueError:
                return None

        return {
            "index": int(parts[0]),
            "name": parts[1],
            "vram_total_mb": num(parts[2]),
            "vram_used_mb": num(parts[3]),
            "vram_free_mb": num(parts[4]),
            "util_pct": num(parts[5]),
            "temp_c": num(parts[6]),
            "power_w": num(parts[7]),
        }
    return None


def processes():
    raw = _smi(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    if not raw:
        return []
    procs = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            procs.append({"pid": parts[0], "name": parts[1], "used_mb": parts[2]})
    return procs


def is_free(gpu_cfg, need_mb=None):
    st = status(gpu_cfg.get("index", 0))
    if st is None:
        return False, "nvidia-smi not available"
    need = need_mb if need_mb is not None else gpu_cfg.get("min_free_vram_mb", 0)
    if st["vram_free_mb"] is not None and st["vram_free_mb"] < need:
        return False, f"only {st['vram_free_mb']:.0f} MB VRAM free (need {need})"
    max_util = gpu_cfg.get("max_utilization_pct", 100)
    if st["util_pct"] is not None and st["util_pct"] > max_util:
        return False, f"GPU {st['util_pct']:.0f}% busy (limit {max_util}%)"
    return True, f"{st['vram_free_mb']:.0f} MB free, {st['util_pct']:.0f}% busy"


def wait_until_free(gpu_cfg, need_mb=None, log=print):
    minutes = gpu_cfg.get("wait_minutes", 0)
    deadline = time.monotonic() + minutes * 60
    while True:
        ok, why = is_free(gpu_cfg, need_mb)
        if ok:
            return True, why
        if time.monotonic() >= deadline:
            return False, why
        log(f"GPU not free yet ({why}); rechecking in 60s")
        time.sleep(60)
