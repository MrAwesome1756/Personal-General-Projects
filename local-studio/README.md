# Local Studio

Photorealistic image and video generation that runs **entirely on a local RTX 5090**, directed by Claude Code.
No outside image/video AI services.

- **Setting it up?** Start with [docs/KYLER_SETUP_BRIEF.md](docs/KYLER_SETUP_BRIEF.md).
- **Models and licenses:** [docs/MODELS.md](docs/MODELS.md)
- **How Claude uses it:** [CLAUDE.md](CLAUDE.md), `.claude/skills/studio/` (directing) and `.claude/skills/studio-train/` (training)

## What it does
1. **Directs**: turns a brief into a treatment and shot list using real cinematography (lens, angle, light, movement).
2. **Keyframes**: renders stills locally (Qwen-Image / HiDream / FLUX.2) and QC-checks every image.
3. **Animates**: turns approved stills into short clips (Wan 2.2 / LTX-2).
4. **Edits**: stitches, adds logo and music, and exports 16:9 / 9:16 / 1:1 / 4:5 with ffmpeg.
5. **Learns**: nightly LoRA training (e.g. a consistent likeness), a fixed eval set, and a morning report by Claude.

## Pieces
| Path | What |
|---|---|
| `studio/` | `python -m studio` CLI: ComfyUI API client, workflow parameter injection, LoRA insertion, GPU-sharing guard, ffmpeg tools, training and nightly orchestration (stdlib + ffmpeg only) |
| `workflows/` | ComfyUI API-format workflows (smoke test included; others exported from ComfyUI templates) |
| `eval/prompts.json` | Fixed nightly test prompts (skin, hands, text, architecture, camera, trained subject) |
| `training/` | Training plan, ai-toolkit config template, learnings |
| `nightly/` | Scheduler scripts (Windows/Linux) and the unattended review prompt |
| `setup/` | Installers and ComfyUI start scripts |
| `tests/` | Tests using a fake ComfyUI server and real ffmpeg (no GPU needed) |

Run the tests: `python -m unittest discover -s tests -t .`
