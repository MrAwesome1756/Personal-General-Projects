# Local Studio — Claude Code project guide

This project turns Claude Code into a photo/video director that generates **only on this PC's GPU**
(RTX 5090, via ComfyUI). Owner: Camden. Machine admin: Kyler.

## Non-negotiables
- **No external image/video generation services or APIs.** All generation goes through `python -m studio`.
- The GPU is shared with a 24/7 agent. Check `python -m studio gpu` before heavy work, use `--wait-gpu` for
  video/batches, and run `python -m studio free` when done with a long session.
- Look at outputs before reporting them (Read images; `python -m studio frames` for video).
- Datasets and personal photos stay out of git (`.gitignore` covers `outputs/` and `datasets/`).

## Skills
- `studio`: making images and videos (treatment → keyframes → QC → image-to-video → edit).
- `studio-train`: LoRA training, captioning, and nightly reports.

## Commands (run from this folder)
```
python -m studio doctor                 # check setup
python -m studio gpu                    # GPU status / is it free
python -m studio image --prompt "..." [--count 3] [--width W --height H] [--lora name:0.9] [--project P]
python -m studio video --prompt "..." [--image keyframe.png] [--frames 97] --wait-gpu
python -m studio shots outputs/<project>/shots.json [--only s01,s03]
python -m studio frames <clip.mp4>      # QC stills from a clip
python -m studio sheet <images...> --out sheet.jpg
python -m studio assemble <clips...> --out final.mp4 --aspect 16:9 [--music m.m4a] [--logo logo.png]
python -m studio reframe final.mp4 --out final_9x16.mp4 --aspect 9:16
python -m studio dataset <folder> --trigger "cmdn person"
python -m studio train                  # run training/active.json now
python -m studio nightly                # what the scheduled task runs
python -m studio benchmark [--video]
python -m unittest discover -s tests -t .   # tests (no GPU needed)
```

## Layout
- `studio/`: the CLI (stdlib + ffmpeg only)
- `workflows/`: ComfyUI API-format workflows (see workflows/README.md)
- `eval/prompts.json`: fixed nightly test set
- `training/`: plans, config templates, learnings
- `nightly/`: scheduled-run scripts and the unattended review prompt
- `reports/nightly/`: morning reports written by Claude
- `docs/`: setup brief (for Kyler), model/license guide
- `studio.config.json`: shared defaults. `studio.config.local.json` holds machine paths (git-ignored).
