"""Nightly loop: (optional) train -> fixed eval set -> contact sheets -> Claude review.

Training runs outside Claude (it takes hours); Claude is only invoked at the
end to look at the results and write the morning report.
"""

import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

from studio import config as C
from studio import gpu, media, train
from studio import workflow as W
from studio.comfy import ComfyClient


def _log(log_path, msg):
    line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_eval(cfg, client, out_dir, lora_name=None, lora_strength=1.0, trigger="", tag="base", log=print):
    wf_base, mapping = W.load(C.resolve(cfg["workflows"]["image"]))
    if lora_name:
        wf_base, _ = W.add_lora(wf_base, lora_name, lora_strength)
    prompts = json.loads(C.resolve(cfg["nightly"]["eval_prompts"]).read_text(encoding="utf-8"))
    n = cfg["nightly"]
    results = []
    for item in prompts["prompts"]:
        if item.get("needs_trigger") and not trigger:
            continue
        text = item["prompt"].replace("{trigger}", trigger).strip()
        wf, _ = W.apply_params(wf_base, mapping, prompt=text, negative=item.get("negative"), seed=n["eval_seed"],
                               width=item.get("width", n["eval_width"]), height=item.get("height", n["eval_height"]),
                               filename_prefix=f"nightly/{tag}_{item['id']}")
        _, paths, secs = client.run(wf, out_dir / tag, timeout_s=cfg["timeouts"]["image_s"])
        log(f"eval {tag}/{item['id']}: {secs}s")
        results.append({"id": item["id"], "category": item.get("category"), "prompt": text,
                        "files": [str(p) for p in paths], "seconds": secs})
    return results


def run(cfg, skip_train=False, skip_review=False):
    today = dt.date.today().isoformat()
    out_dir = C.resolve(cfg["output_dir"]) / "nightly" / today
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "nightly.log"
    log = lambda m: _log(log_path, m)  # noqa: E731
    summary = {"date": today, "stages": {}}

    client = ComfyClient(cfg["comfy_url"])
    if not client.is_up():
        log("ComfyUI is not reachable; aborting (the run script should start it first).")
        return 2

    ok, why = gpu.wait_until_free(cfg["gpu"], log=log)
    summary["gpu_start"] = gpu.status(cfg["gpu"].get("index", 0))
    if not ok:
        log(f"GPU never became free ({why}); skipping tonight.")
        summary["skipped"] = why
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 3
    log(f"GPU ready: {why}")

    plan = train.load_plan(cfg)
    lora_name, trigger, strength = None, "", 1.0
    if plan and plan.get("enabled") and not skip_train:
        try:
            client.free()  # give the trainer the whole card
            log(f"Training '{plan['name']}' ...")
            summary["stages"]["train"] = train.run_training(cfg, plan, out_dir)
            summary["stages"]["publish"] = train.publish_lora(cfg, plan)
            log(f"Training done in {summary['stages']['train']['minutes']} min")
        except train.TrainError as e:
            log(f"Training failed: {e}")
            summary["stages"]["train_error"] = str(e)
    if plan and plan.get("enabled"):
        trigger = plan.get("trigger", "")
        strength = plan.get("lora_strength", 1.0)
        candidate = f"{plan['name']}_latest.safetensors"
        loras_dir = C.resolve(cfg.get("comfy", {}).get("loras_dir"))
        if loras_dir and (loras_dir / candidate).exists():
            lora_name = candidate

    eval_results = {}
    try:
        eval_results["base"] = run_eval(cfg, client, out_dir, tag="base", trigger="", log=log)
        if lora_name:
            eval_results["lora"] = run_eval(cfg, client, out_dir, lora_name, strength, trigger, tag="lora", log=log)
    except Exception as e:  # keep going so the review can report the failure
        log(f"Eval failed: {e}")
        summary["stages"]["eval_error"] = str(e)
    summary["stages"]["eval"] = eval_results

    for tag, results in eval_results.items():
        files = [f for r in results for f in r["files"] if Path(f).suffix.lower() in media.IMAGE_EXTS]
        if files:
            try:
                sheet = media.contact_sheet(files, out_dir / f"sheet_{tag}.jpg")
                summary["stages"][f"sheet_{tag}"] = str(sheet)
            except media.MediaError as e:
                log(f"Contact sheet failed: {e}")

    try:
        client.free()  # always hand VRAM back to the 24/7 agent after the nightly run
    except Exception:
        pass
    summary["gpu_end"] = gpu.status(cfg["gpu"].get("index", 0))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Summary written to {out_dir / 'summary.json'}")

    if skip_review or not cfg["nightly"].get("review_with_claude", True):
        return 0
    return review(cfg, out_dir, log)


def review(cfg, out_dir, log=print):
    claude = shutil.which(cfg["claude"]["command"])
    if not claude:
        log("Claude Code CLI not found on PATH; skipping review.")
        return 0
    prompt = C.resolve(cfg["claude"]["review_prompt_file"]).read_text(encoding="utf-8")
    prompt += f"\n\nTonight's results folder: {out_dir.relative_to(C.ROOT).as_posix()}\n"
    log("Starting Claude review ...")
    proc = subprocess.run([claude, "-p", prompt, "--allowedTools", cfg["claude"]["allowed_tools"]],
                          cwd=C.ROOT, capture_output=True, text=True, timeout=45 * 60)
    (out_dir / "claude_review.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    log(f"Claude review finished (exit {proc.returncode})")
    return 0 if proc.returncode == 0 else 4
