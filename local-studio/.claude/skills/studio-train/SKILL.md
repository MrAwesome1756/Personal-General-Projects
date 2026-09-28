---
name: studio-train
description: Train and improve local LoRA models on the RTX 5090 (a consistent digital Camden, the Serve Funding look, property/real-estate style, or a cinematic style) — dataset curation, captioning, ai-toolkit configs, the nightly train→eval→review loop, and reading/acting on nightly reports. Use when Camden asks to train, fine-tune, teach the model a face/style, build a dataset, check training progress, or review the nightly report.
---

# Studio training (LoRA fine-tuning, fully local)

Training happens with **ostris/ai-toolkit** on this PC. A LoRA is a small add-on file that teaches a base
model one new thing (a person, a style, a product). You (Claude) are the trainer: you curate and caption the
data, choose settings, judge results, and adjust. The base model weights never leave this machine.

## What is worth training
| Target | Base model | Data | Typical run on a 5090 |
|---|---|---|---|
| Camden (consistent likeness) | Qwen-Image or FLUX.2 (image); Wan 2.2 (video identity) | 20–40 photos | ~1–3 h |
| Serve Funding brand look | Qwen-Image | 30–80 on-brand images | ~1–3 h |
| Property/real-estate style | Qwen-Image | 40–100 pro property photos (licensed/own) | ~2–3 h |
| Cinematic grade / look | Qwen-Image or Wan | 30–80 frames of the look | ~1–3 h |
Check docs/MODELS.md for the license of the base model before training anything for business use.

## 1. Build the dataset
Folder per dataset, e.g. `D:/AI/datasets/camden_v1/` (keep datasets out of git — they're personal).
For a person:
- 20–40 sharp, recent photos; varied angles (front, 3/4, profile), expressions, outfits, lighting, backgrounds.
- Mostly head-and-shoulders, some half-body, a few full-body. Only the subject in frame (crop others out).
- No heavy filters, sunglasses, or hats in most shots. At least 1024 px on the short side if possible.
- **Consent**: only train on a person who has agreed (Camden for himself). Never train on public figures.

## 2. Caption every image (you do this)
Read each image and write a `.txt` file with the same name next to it.
- Start with the trigger word, e.g. `cmdn person` (rare token + class).
- Describe everything that **should stay changeable**: clothing, pose, expression, background, lighting,
  camera angle. Do **not** describe fixed identity features (face shape, eye color) — the LoRA should absorb those.
- One or two plain sentences. Example: `cmdn person, smiling, wearing a navy quarter-zip, standing in a bright
  office, soft window light from the left, three-quarter view, medium close-up.`
Then verify: `python -m studio dataset <folder> --trigger "cmdn person"` → must say `"ready": true`.

## 3. Create the ai-toolkit config and plan
- Copy the matching example from ai-toolkit's own `config/examples/` folder (it matches the installed version),
  then carry over our choices from `training/configs/TEMPLATE_image_lora.yaml` (name, dataset path, trigger,
  steps, rank, sample prompts). Save as `training/configs/<name>.yaml`.
- Write `training/active.json` (see `training/active.example.json`) with `"enabled": true`.
- Start conservatively: rank 16, lr 1e-4, ~1500–2500 steps, save every 250 steps.

## 4. Run
- Tonight automatically (the nightly task), or now: `python -m studio train` (checks the GPU is free first,
  unloads ComfyUI models, runs ai-toolkit, copies the result to ComfyUI's loras folder as `<name>_latest.safetensors`).

## 5. Evaluate & iterate (the nightly review does this; do the same when asked)
Compare `outputs/nightly/<date>/sheet_base.jpg` vs `sheet_lora.jpg` and the individual images:
- **Likeness** (subject prompts): does it look like the person from several angles?
- **Flexibility**: can it change clothes/setting/lighting, or does every image copy the training photos? (overfit)
- **Quality**: skin texture, hands, artifacts — did the LoRA degrade the base model?
Adjustments:
| Symptom | Change |
|---|---|
| Weak likeness | More steps (+500), check captions aren't describing identity, add better close-ups |
| Overfit (same pose/background every time, fried colors) | Fewer steps or use an earlier checkpoint, lower strength (0.7–0.85), more varied data |
| Artifacts/quality loss | Lower lr (5e-5), lower rank (8), remove bad training images |
| Only works at one angle | Add missing angles to the dataset |
Record what you changed and why in `training/notes.md` so future sessions learn from it.

## Guardrails
- Never delete datasets or past LoRAs; version names (`camden_v2`).
- Don't enable a plan that would run on the GPU while Camden has asked for it to be kept free.
- Report honestly: "v3 improved likeness at 3/4 angles but profiles still drift" beats "looks great".
