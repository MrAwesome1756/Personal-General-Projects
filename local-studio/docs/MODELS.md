# Models: what to install, and what we're allowed to use it for

All models run locally in ComfyUI. Download them through ComfyUI's **template browser** (it fetches the exact
files each workflow expects). **Licenses change. Confirm on the model's page when downloading.** This table
reflects public information as of September 2026.

## Recommended starting set (RTX 5090, 32 GB)

| Role | Model | License (verify) | Business use for Serve Funding? | Fits 5090? | Fits 4070 (12 GB)? |
|---|---|---|---|---|---|
| Smoke test | SDXL base 1.0 | CreativeML OpenRAIL++-M | Yes, with use restrictions | Yes | Yes |
| **Main photoreal stills** | **Qwen-Image** (latest release) | Apache-2.0 for the original release; newer versions may use a different license | Yes if Apache-2.0; check the version you download | Yes (FP8/BF16) | Quantized only |
| Photoreal stills (alt) | HiDream (I1 / O1) | MIT | Yes | Yes | Quantized only |
| Top-quality stills, personal only | FLUX.2 [dev] | FLUX non-commercial license | **No**, unless a commercial license is bought | Yes (quantized) | Tight / quantized |
| **Image-to-video / text-to-video** | **Wan 2.2** (14B; 5B TI2V for speed) | Apache-2.0 | Yes | 14B in FP8, 5B easily | 5B only |
| Fast video with audio | LTX-2.x | LTX community license | Check the terms (free-use thresholds may apply) | Yes (FP8) | No |
| Higher-res video (alt) | HunyuanVideo 1.5 | Tencent community license | Check it; has regional restrictions | Yes | Tight |
| Upscaling | Real-ESRGAN / 4x upscalers | Varies (Real-ESRGAN is BSD-3) | Yes (Real-ESRGAN) | Yes | Yes |
| Training | ostris/ai-toolkit | MIT | Yes (tool) | Yes | Small LoRAs only |

**Default for anything public-facing (Serve Funding): Qwen-Image or HiDream for stills, Wan 2.2 for video.**

## Rough expectations on a 5090
Estimates from published benchmarks; `python -m studio benchmark` gives real numbers on this machine.
- Stills: roughly 5–40 s per ~1 MP image depending on model, steps and precision.
- Wan 2.2 14B image-to-video: several minutes per 5 s 720p clip. The 5B model is much faster.
- LTX-2.x: short 720p clips in well under a minute.
- LoRA training: about 1–3 h for a person or style LoRA (fits in one night).

## Resolutions (keep to what the model was trained on)
- Qwen-Image: around 1.3 MP. Examples: 1328×1328, 1664×928 (16:9), 928×1664 (9:16), 1472×1140 (4:3).
- FLUX.2 / HiDream: around 1 MP. Examples: 1024×1024, 1344×768, 768×1344.
- Wan 2.2: 1280×720 or 832×480 (and portrait equivalents). Frame counts are usually 4n+1 (81 ≈ 5 s at 16 fps).
- LTX-2.x: dimensions divisible by 32; frame counts are usually 8n+1.

## VRAM sharing with the 24/7 agent
The 5090 has 32 GB. Big video models can use most of it. If the other agent keeps a model loaded on the same
GPU, either run studio work when it's idle (nightly), move the agent's model to the 4070 PC, or use smaller or
quantized studio models. `python -m studio gpu` shows what's using VRAM right now.

## Safety
- Prefer `.safetensors` files. Avoid `.ckpt` / `.pt` / `.bin` from unknown sources (they can contain code).
- Only install ComfyUI custom nodes that are widely used and needed by a workflow. Custom nodes are
  arbitrary Python code running on this PC.
