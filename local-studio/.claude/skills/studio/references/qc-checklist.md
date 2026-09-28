# Visual QC checklist

Read the actual image (or extracted frames) before judging. Score each item Pass / Minor / Fail.
Any Fail on a hero element = regenerate. Report Minor issues to Camden honestly.

## Stills
1. **Anatomy** — correct finger count, natural hands and joints, symmetric eyes, teeth look real, ears/hairline plausible.
2. **Faces** — not waxy/plastic; skin texture present; expression matches intent; no uncanny stare.
3. **Text & logos** — spelled exactly as requested, no garbled glyphs. If text keeps failing, generate the image
   without text and add it in the edit (assemble/logo overlay) instead.
4. **Geometry** — straight verticals on buildings, consistent perspective, no melted objects, furniture legs intact.
5. **Lighting** — one coherent light direction; shadows match; reflections plausible.
6. **Prompt adherence** — shot size, angle, lens feel, wardrobe, props, setting all match the shot list.
7. **Composition** — subject placement, headroom, negative space for overlays if needed.
8. **Brand fit** — palette/grade consistent with the project bible; nothing off-brand or tacky.
9. **Artifacts** — no watermarks, signatures, noise blotches, seams, duplicated limbs or objects.
10. **Material realism** — metal shows directional grain and clean reflections; glass refracts and has edge lines;
    liquids glow when backlit; wood/leather/fabric show real texture; nothing reads as plastic or CGI.
11. **Object integrity** — correct part counts (watch hands, buttons, laces), symmetric where the real object is,
    no fused or floating parts, plausible scale against the surface.
12. **Safety** — no real person's likeness without consent; no real brand logos presented as genuine.

## Video clips (check 6–8 frames + first/last)
1. **Shape drift** — the object keeps its exact shape, proportions, materials and any text across frames.
2. **Warping** — backgrounds and hands don't morph; straight lines stay straight.
3. **Motion** — the camera move is the one requested, smooth, and physically believable.
4. **Flicker / boiling textures** — especially skin, hair, foliage, text.
5. **Start frame fidelity** (image-to-video) — first frame matches the approved keyframe.
6. **Ending** — clip ends cleanly (no sudden morph in the last frames); trim in the edit if needed.

## Common fixes
| Problem | Try |
|---|---|
| Plastic skin | Add "visible pores, natural skin texture, subtle film grain"; lower CFG/guidance a bit; avoid "perfect", "flawless" |
| Bad hands | Show fewer hands, give them an action (holding a mug), change seed, crop tighter |
| Garbled text | Shorten the text, quote it exactly, or add text in post |
| Generic "stock" look | Specify lens, light direction, time of day, a specific location detail, candid action |
| Video morphing | Shorter clip, simpler single camera move, less subject motion, start from a cleaner keyframe |
| Identity drift | Use the trained LoRA, closer framing, fewer frames |
