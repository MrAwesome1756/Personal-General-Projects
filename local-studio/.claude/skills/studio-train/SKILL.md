---
name: studio-train
description: Train and improve local LoRA models on the RTX 5090 for premium photorealism — real materials and textures, product/object lighting and angles, in-image text, and later object motion — including dataset curation, captioning, ai-toolkit configs, the nightly train→eval→review loop, and reading/acting on nightly reports. Use when asked to train, fine-tune, teach the model a look or material, build a dataset, check training progress, or review the nightly report.
---

# Studio training (LoRA fine-tuning, fully local)

Training happens with **ostris/ai-toolkit** on this PC. A LoRA is a small add-on file that teaches a base
model one new thing — here, a **look**: premium, photorealistic objects with believable materials, textures,
lighting and camera angles. You (Claude) are the trainer: you curate and caption the data, choose settings,
judge results, and adjust. Base model weights never leave this machine.

**Current goal (set by Camden):** general premium photorealism for **objects and materials** (metal, glass,
liquid, wood, leather, fabric, ceramic, stone), realistic textures and angles, strong **in-image text**
(engraving, embossing, packaging, signage), then **moving video of objects**. No people/persona training.

## Training roadmap
| Stage | LoRA | Base model | Data | Typical run on a 5090 |
|---|---|---|---|---|
| 1 | `materials_v1`: premium product/material look | Qwen-Image (commercial-safe) | 60–150 images | ~1–3 h |
| 2 | Focused add-ons if stage 1 is weak somewhere (e.g. `glass_liquid_v1`, `text_surfaces_v1`) | Qwen-Image | 30–80 each | ~1–2 h |
| 3 | `object_motion_v1`: clean product camera moves (turntable, slow push-in, macro slide) | Wan 2.2 | 30–60 short clips | ~1.5–3 h |
Always test the base model first: modern bases are already good, and a LoRA should fix a measured weakness
(the nightly base-vs-LoRA sheets show whether it does). Check docs/MODELS.md for the base model's license.

## 1. Build the dataset
Folder per dataset, e.g. `D:/AI/datasets/materials_v1/` (outside git).
- **Rights:** use only images you own, commissioned, or that are explicitly free for training/commercial use
  (e.g. CC0 / public domain). Never scrape copyrighted product photography or stock without a license.
- **Quality bar:** sharp, well-lit, high-resolution (≥1024 px short side), no watermarks, no heavy filters,
  no text overlays unless the text IS the subject (text datasets should have correctly spelled, legible text).
- **Coverage over repetition:** mix materials, object types, backgrounds, light setups (rim, backlight, raking,
  softbox, window), and angles (3/4 hero, low angle, top-down flat lay, macro detail). Avoid 20 near-identical shots.
- For stage 3 (video): 2–6 s clips of single, smooth camera moves on objects; trim shakes and cuts.

## 2. Caption every image (you do this)
Read each image and write a `.txt` with the same name next to it.
- Start with the trigger, e.g. `prmtx style` (rare token + "style").
- Describe the **content** literally — object, materials, surface, background, light direction, angle, lens feel —
  so the LoRA learns the *look* and stays flexible about *what* is shown.
- For text images, quote the text exactly as it appears.
- Example: `prmtx style, a brushed-steel wristwatch on dark slate, three-quarter angle, soft top-left key light with
  a thin rim light, macro lens, black background.`
Then verify: `python -m studio dataset <folder> --trigger "prmtx style"` → must say `"ready": true`.

## 3. Create the ai-toolkit config and plan
- Copy the matching example from ai-toolkit's own `config/examples/` (it matches the installed version), then carry
  over our values from `training/configs/TEMPLATE_image_lora.yaml`. Save as `training/configs/<name>.yaml`.
- Write `training/active.json` (see `training/active.example.json`) with `"enabled": true`.
- Start conservatively for a style: rank 16, lr 1e-4, ~2000–3000 steps, save every 250 steps.

## 4. Run
Tonight automatically (the nightly task), or now: `python -m studio train` (waits for a free GPU, unloads ComfyUI
models, runs ai-toolkit, copies the result to ComfyUI's loras folder as `<name>_latest.safetensors`).

## 5. Evaluate & iterate (the nightly review does this; do the same when asked)
Compare `outputs/nightly/<date>/sheet_base.jpg` vs `sheet_lora.jpg` and the individual images, per category
(metal, glass & liquid, wood, leather & fabric, ceramic & stone, reflections, text, angles, motion freeze):
- **Material realism:** do surfaces read as the real material (grain, micro-scratches, subsurface glow in liquids,
  correct reflections/refraction), or as plastic/CGI?
- **Text:** exact spelling, crisp edges, correct depth for engraving/embossing.
- **Flexibility:** does it still follow each prompt's object, angle and lighting, or does everything drift toward
  the training images? (overfit)
- **Base-model regression:** anatomy of objects (straight lines, correct part counts), artifacts, color.
| Symptom | Change |
|---|---|
| Look too weak | More steps (+500), more varied high-quality images, raise LoRA strength to 1.0 |
| Overfit (same backgrounds/colors everywhere, "fried" contrast) | Earlier checkpoint, strength 0.7–0.85, more variety, fewer steps |
| Quality loss / artifacts | lr 5e-5, rank 8, remove weak images |
| One material still weak | Add 15–30 strong images of it, or train a focused add-on LoRA (stage 2) |
| Text worse than base | Remove text-heavy images with small/illegible text; keep text LoRAs separate |
Record what changed and why in `training/notes.md`.

## Guardrails
- Never delete datasets or past LoRAs; version names (`materials_v2`).
- Don't enable a plan that would run while the GPU is meant to be kept free for the 24/7 agent.
- Report honestly: "v3 fixed glass refraction but brushed metal still looks plastic" beats "looks great".
