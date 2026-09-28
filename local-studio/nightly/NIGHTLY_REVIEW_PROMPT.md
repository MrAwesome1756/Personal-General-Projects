You are running unattended as the nightly reviewer for the Local Studio on this PC. Nobody is watching;
do the work and write files. Do not generate new images, do not start training, and do not change any
configuration except the files named below.

Steps:
1. Read `summary.json` and `nightly.log` in tonight's results folder (given at the end). If a stage failed,
   the report's first line must say what failed and the likely fix.
2. Look at `sheet_base.jpg` (and `sheet_lora.jpg` if present), then open the individual images for anything
   that looks off. The `.index.json` next to each sheet lists the image order.
3. Score each eval image 1–5 on: material realism, object integrity, text accuracy (text prompts), lighting,
   angle/prompt adherence, and (for trained-style prompts) style strength and flexibility. Use
   `.claude/skills/studio/references/qc-checklist.md`.
4. Compare with the most recent previous report in `reports/nightly/` if one exists: better, worse, same — per category.
5. Write `reports/nightly/<date>.md` with:
   - One-paragraph summary a busy non-technical person can read in 20 seconds (what improved, what didn't).
   - A score table.
   - Specific observations (cite image ids).
   - GPU/timing notes from summary.json (training minutes, seconds per image, VRAM).
   - Recommendation for tomorrow night (e.g. "+500 steps", "drop 4 blurry dataset photos", "try strength 0.85").
6. If there is an active training plan, write your recommended next plan to `training/next_plan_suggestion.json`
   (same shape as `training/active.json`) — do NOT edit `training/active.json` itself.
7. Append 1–3 durable learnings (settings that worked or failed) to `training/notes.md`.
