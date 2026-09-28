"""`python -m studio <command>` — the interface Claude Code uses.

Generation commands print a JSON summary (file paths, seed, timing, which
workflow inputs were changed) so Claude can review outputs precisely.
"""

import argparse
import datetime as dt
import json
import platform
import random
import re
import shutil
import sys
from pathlib import Path

from studio import config as C
from studio import gpu, media, nightly, train
from studio import workflow as W
from studio.comfy import ComfyClient, ComfyError


def _slug(text, n=40):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "untitled"


def _print(obj):
    print(json.dumps(obj, indent=2, default=str))


def _client(cfg):
    return ComfyClient(cfg["comfy_url"])


def _parse_loras(values):
    loras = []
    for v in values or []:
        name, _, strength = v.partition(":")
        loras.append((name, float(strength) if strength else 1.0))
    return loras


def generate(cfg, *, workflow_key, prompt, negative=None, seed=None, width=None, height=None, frames=None,
             steps=None, image=None, loras=None, project="adhoc", shot_id=None, wait_gpu=False, timeout_s=None):
    """Run one generation and return a result dict (also written to the project's manifest folder)."""
    client = _client(cfg)
    if wait_gpu:
        ok, why = gpu.wait_until_free(cfg["gpu"], log=lambda m: print(m, file=sys.stderr))
        if not ok:
            raise ComfyError(f"GPU busy: {why}")
    wf_path = C.resolve(cfg["workflows"].get(workflow_key, workflow_key))
    wf, mapping = W.load(wf_path)
    for name, strength in loras or []:
        wf, _ = W.add_lora(wf, name, strength)
    uploaded = client.upload_image(image) if image else None
    seed = seed if seed is not None else random.randint(1, 2**31 - 1)
    shot_id = shot_id or f"{dt.datetime.now():%H%M%S}-{_slug(prompt, 24)}"
    prefix = f"studio/{_slug(project)}/{shot_id}"
    wf, report = W.apply_params(wf, mapping, prompt=prompt, negative=negative, seed=seed, width=width, height=height,
                                frames=frames, steps=steps, image=uploaded, filename_prefix=prefix)
    out_dir = C.resolve(cfg["output_dir"]) / _slug(project)
    timeout_s = timeout_s or cfg["timeouts"]["video_s" if workflow_key.startswith("video") else "image_s"]
    prompt_id, paths, secs = client.run(wf, out_dir, timeout_s=timeout_s)
    if cfg["gpu"].get("unload_models_after_job"):
        client.free()
    return {
        "shot": shot_id, "workflow": str(wf_path.relative_to(C.ROOT)) if wf_path.is_relative_to(C.ROOT) else str(wf_path),
        "prompt": prompt, "negative": negative, "seed": seed, "width": width, "height": height, "frames": frames,
        "input_image": str(image) if image else None, "loras": loras or [], "prompt_id": prompt_id,
        "seconds": secs, "files": [str(p) for p in paths], "changed_inputs": report,
    }


# ---- commands ---------------------------------------------------------------

def cmd_doctor(cfg, args):
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": ok, "detail": detail})

    add("python >= 3.10", sys.version_info >= (3, 10), platform.python_version())
    add("ffmpeg on PATH", bool(shutil.which("ffmpeg")), shutil.which("ffmpeg") or "install ffmpeg")
    add("ffprobe on PATH", bool(shutil.which("ffprobe")), shutil.which("ffprobe") or "comes with ffmpeg")
    st = gpu.status(cfg["gpu"].get("index", 0))
    add("nvidia-smi / GPU", st is not None, json.dumps(st) if st else "nvidia-smi not found")
    client = _client(cfg)
    up = client.is_up()
    add("ComfyUI reachable", up, cfg["comfy_url"])
    available = set(client.object_info()) if up else set()
    for key, rel in cfg["workflows"].items():
        path = C.resolve(rel)
        if not path.exists():
            add(f"workflow '{key}'", False, f"missing {rel} (export from ComfyUI, see workflows/README.md)")
            continue
        try:
            wf, _ = W.load(path)
        except W.WorkflowError as e:
            add(f"workflow '{key}'", False, str(e))
            continue
        missing = [c for c in W.required_classes(wf) if up and c not in available]
        add(f"workflow '{key}'", not missing, f"missing nodes: {missing}" if missing else rel)
    loras = C.resolve(cfg.get("comfy", {}).get("loras_dir"))
    add("comfy.loras_dir", bool(loras and loras.exists()), str(loras) if loras else "not set (needed for training)")
    tk = C.resolve(cfg.get("ai_toolkit", {}).get("dir"))
    add("ai-toolkit", bool(tk and (tk / "run.py").exists()), str(tk) if tk else "not set (needed for training)")
    add("claude CLI", bool(shutil.which(cfg["claude"]["command"])), shutil.which(cfg["claude"]["command"]) or "not on PATH")
    _print(checks)
    critical = {"python >= 3.10", "ffmpeg on PATH", "nvidia-smi / GPU", "ComfyUI reachable"}
    return 0 if all(c["ok"] for c in checks if c["check"] in critical) else 1


def cmd_gpu(cfg, args):
    ok, why = gpu.is_free(cfg["gpu"])
    _print({"status": gpu.status(cfg["gpu"].get("index", 0)), "processes": gpu.processes(),
            "free_for_studio": ok, "reason": why})
    return 0


def cmd_free(cfg, args):
    _client(cfg).free()
    print("Asked ComfyUI to unload models and free VRAM.")
    return 0


def _gen_common(cfg, args, workflow_key):
    results = []
    base_seed = args.seed if args.seed is not None else random.randint(1, 2**31 - 1)
    for i in range(args.count):
        results.append(generate(
            cfg, workflow_key=workflow_key, prompt=args.prompt, negative=args.negative, seed=base_seed + i,
            width=args.width, height=args.height, frames=getattr(args, "frames", None), steps=args.steps,
            image=getattr(args, "image", None), loras=_parse_loras(args.lora), project=args.project,
            wait_gpu=args.wait_gpu))
    _print(results if len(results) > 1 else results[0])
    return 0


def cmd_image(cfg, args):
    return _gen_common(cfg, args, args.workflow or "image")


def cmd_video(cfg, args):
    key = args.workflow or ("video_i2v" if args.image else "video_t2v")
    return _gen_common(cfg, args, key)


def cmd_shots(cfg, args):
    """Run a shot list. Video shots may use "image": "@<shot_id>" to animate an approved keyframe."""
    shotlist = json.loads(Path(args.file).read_text(encoding="utf-8"))
    project = shotlist.get("project", Path(args.file).stem)
    out_dir = C.resolve(cfg["output_dir"]) / _slug(project)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"project": project, "shots": {}}
    only = set(args.only.split(",")) if args.only else None
    defaults = shotlist.get("defaults", {})
    for shot in shotlist["shots"]:
        sid = shot["id"]
        if only and sid not in only:
            continue
        s = {**defaults, **shot}
        image = s.get("image")
        if isinstance(image, str) and image.startswith("@"):
            ref = manifest["shots"].get(image[1:])
            if not ref or not ref.get("files"):
                raise SystemExit(f"Shot {sid} needs {image} but it has no output yet")
            image = ref.get("approved") or ref["files"][0]
        kind = s.get("type", "image")
        key = s.get("workflow") or ("image" if kind == "image" else ("video_i2v" if image else "video_t2v"))
        print(f"-> {sid} ({kind}, {key})", file=sys.stderr)
        result = generate(cfg, workflow_key=key, prompt=s["prompt"], negative=s.get("negative"), seed=s.get("seed"),
                          width=s.get("width"), height=s.get("height"), frames=s.get("frames"), steps=s.get("steps"),
                          image=image, loras=[tuple(x) for x in s.get("loras", [])], project=project, shot_id=sid,
                          wait_gpu=args.wait_gpu)
        manifest["shots"][sid] = result
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    _print({"manifest": str(manifest_path), "shots": list(manifest["shots"])})
    return 0


def cmd_frames(cfg, args):
    out = args.out or str(Path(args.video).with_suffix("")) + "_frames"
    _print([str(p) for p in media.extract_frames(args.video, out, args.count)])
    return 0


def cmd_sheet(cfg, args):
    _print({"sheet": str(media.contact_sheet(args.images, args.out, args.cols))})
    return 0


def cmd_assemble(cfg, args):
    out = media.assemble(args.clips, args.out, aspect=args.aspect, fit=args.fit, fps=args.fps,
                         music=args.music, music_volume=args.music_volume, logo=args.logo)
    _print({"video": str(out), **media.probe(out)})
    return 0


def cmd_reframe(cfg, args):
    _print({"video": str(media.reframe(args.src, args.out, args.aspect, args.fit))})
    return 0


def cmd_dataset(cfg, args):
    report = train.check_dataset(args.folder, args.trigger)
    _print(report)
    return 0 if report["ready"] else 1


def cmd_train(cfg, args):
    plan = train.load_plan(cfg, args.plan)
    if not plan:
        raise SystemExit("No training plan found (training/active.json). See training/README.md.")
    out_dir = C.resolve(cfg["output_dir"]) / "training" / f"{dt.datetime.now():%Y%m%d-%H%M}_{plan['name']}"
    ok, why = gpu.wait_until_free(cfg["gpu"], log=lambda m: print(m, file=sys.stderr))
    if not ok:
        raise SystemExit(f"GPU busy: {why}")
    client = _client(cfg)
    if client.is_up():
        client.free()
    result = train.run_training(cfg, plan, out_dir)
    result["publish"] = train.publish_lora(cfg, plan)
    _print(result)
    return 0


def cmd_nightly(cfg, args):
    return nightly.run(cfg, skip_train=args.skip_train, skip_review=args.skip_review)


def cmd_benchmark(cfg, args):
    client = _client(cfg)
    results = {"gpu": gpu.status(cfg["gpu"].get("index", 0)), "runs": []}
    keys = [k for k in ("smoke", "image") if C.resolve(cfg["workflows"][k]).exists()]
    for key in keys:
        times = []
        for i in range(args.runs + 1):  # first run includes model load; reported separately
            r = generate(cfg, workflow_key=key, prompt="a cinematic photo of a lighthouse on a cliff at golden hour, 35mm",
                         seed=1000 + i, project="benchmark", shot_id=f"{key}_{i}")
            times.append(r["seconds"])
        results["runs"].append({"workflow": key, "first_run_s": times[0], "warm_runs_s": times[1:],
                                "warm_avg_s": round(sum(times[1:]) / max(1, len(times) - 1), 1)})
    if args.video and C.resolve(cfg["workflows"]["video_t2v"]).exists():
        r = generate(cfg, workflow_key="video_t2v", prompt="slow dolly-in on a steaming coffee cup by a rainy window, 35mm, shallow depth of field",
                     seed=7, project="benchmark", shot_id="video_t2v")
        results["runs"].append({"workflow": "video_t2v", "seconds": r["seconds"], "files": r["files"]})
    _print(results)
    return 0


# ---- parser -----------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(prog="python -m studio", description="Local image/video studio (ComfyUI on the local GPU)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check the whole setup")
    sub.add_parser("gpu", help="GPU status and whether it's free for studio work")
    sub.add_parser("free", help="unload ComfyUI models to free VRAM")

    def gen_args(sp, video=False):
        sp.add_argument("--prompt", required=True)
        sp.add_argument("--negative")
        sp.add_argument("--seed", type=int)
        sp.add_argument("--width", type=int)
        sp.add_argument("--height", type=int)
        sp.add_argument("--steps", type=int)
        sp.add_argument("--count", type=int, default=1)
        sp.add_argument("--lora", action="append", help="name.safetensors[:strength] (repeatable)")
        sp.add_argument("--workflow", help="workflow key from config or a path")
        sp.add_argument("--project", default="adhoc")
        sp.add_argument("--wait-gpu", action="store_true", help="wait until the GPU is free per config")
        if video:
            sp.add_argument("--image", help="keyframe for image-to-video")
            sp.add_argument("--frames", type=int)

    gen_args(sub.add_parser("image", help="generate image(s)"))
    gen_args(sub.add_parser("video", help="generate a video clip"), video=True)

    sp = sub.add_parser("shots", help="run a shot list JSON")
    sp.add_argument("file")
    sp.add_argument("--only", help="comma-separated shot ids to (re)run")
    sp.add_argument("--wait-gpu", action="store_true")

    sp = sub.add_parser("frames", help="extract QC frames from a video")
    sp.add_argument("video")
    sp.add_argument("--count", type=int, default=6)
    sp.add_argument("--out")

    sp = sub.add_parser("sheet", help="make a contact sheet")
    sp.add_argument("images", nargs="+")
    sp.add_argument("--out", required=True)
    sp.add_argument("--cols", type=int, default=4)

    sp = sub.add_parser("assemble", help="stitch clips/stills into a finished video")
    sp.add_argument("clips", nargs="+")
    sp.add_argument("--out", required=True)
    sp.add_argument("--aspect", default="16:9", choices=sorted(media.ASPECTS))
    sp.add_argument("--fit", default="crop", choices=["crop", "pad"])
    sp.add_argument("--fps", type=int, default=24)
    sp.add_argument("--music")
    sp.add_argument("--music-volume", type=float, default=0.35)
    sp.add_argument("--logo")

    sp = sub.add_parser("reframe", help="make a different aspect-ratio version")
    sp.add_argument("src")
    sp.add_argument("--out", required=True)
    sp.add_argument("--aspect", default="9:16", choices=sorted(media.ASPECTS))
    sp.add_argument("--fit", default="crop", choices=["crop", "pad"])

    sp = sub.add_parser("dataset", help="check a LoRA training dataset folder")
    sp.add_argument("folder")
    sp.add_argument("--trigger")

    sp = sub.add_parser("train", help="run the active training plan now")
    sp.add_argument("--plan")

    sp = sub.add_parser("nightly", help="nightly train + eval + review")
    sp.add_argument("--skip-train", action="store_true")
    sp.add_argument("--skip-review", action="store_true")

    sp = sub.add_parser("benchmark", help="time the local setup")
    sp.add_argument("--runs", type=int, default=3)
    sp.add_argument("--video", action="store_true")
    return p


COMMANDS = {
    "doctor": cmd_doctor, "gpu": cmd_gpu, "free": cmd_free, "image": cmd_image, "video": cmd_video,
    "shots": cmd_shots, "frames": cmd_frames, "sheet": cmd_sheet, "assemble": cmd_assemble,
    "reframe": cmd_reframe, "dataset": cmd_dataset, "train": cmd_train, "nightly": cmd_nightly,
    "benchmark": cmd_benchmark,
}


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = C.load()
    try:
        return COMMANDS[args.cmd](cfg, args)
    except (ComfyError, W.WorkflowError, media.MediaError, train.TrainError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
