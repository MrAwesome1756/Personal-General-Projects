# Training (LoRA fine-tuning on the 5090)

- `active.example.json` — copy to `active.json` and fill in to schedule nightly training.
- `configs/TEMPLATE_image_lora.yaml` — reference values for an ai-toolkit image LoRA. **Start from the example
  config that ships with your installed ai-toolkit** (`<ai-toolkit>/config/examples/`, pick the one for your base
  model) and copy these values into it; ai-toolkit field names change between versions.
- `notes.md` — running log of what worked (written by Claude after each review).
- Datasets live **outside** this repo (personal photos). Point the config's `folder_path` at them.

Flow: dataset → captions (Claude) → `python -m studio dataset <folder> --trigger "..."` → config →
`active.json` → nightly run (or `python -m studio train`) → `outputs/nightly/<date>/` → `reports/nightly/<date>.md`.

See `.claude/skills/studio-train/SKILL.md` for the full procedure.
