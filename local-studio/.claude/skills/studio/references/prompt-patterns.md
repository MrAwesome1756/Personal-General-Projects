# Prompt patterns by model family

Modern local models (Qwen-Image, FLUX.2, HiDream, Wan, LTX) understand full natural-language sentences.
Write like a cinematographer briefing a crew, not a list of tags. Keep the most important thing first.

## Stills (Qwen-Image / FLUX.2 / HiDream)
Structure: **[shot type + subject] + [action/expression] + [setting] + [lighting] + [lens/camera] + [look/grade] + [realism details]**

Example:
> Medium close-up of a confident woman in her 40s in a charcoal blazer, mid-laugh while reading her phone,
> standing in a bright modern office lobby. Soft window light from camera left with a warm practical lamp in the
> background. Shot on an 85mm lens at f/2, shallow depth of field. Natural editorial color, subtle 35mm film grain,
> visible skin texture.

Tips:
- Qwen-Image is strongest for exact in-image text; put the text in quotes.
- FLUX.2 [dev] is non-commercial by license — personal projects only unless a commercial license is bought.
- Negative prompts: only some workflows use them (many FLUX/Qwen templates ignore them). The CLI reports when
  a negative prompt was ignored.
- SDXL (smoke test only) prefers shorter, comma-separated descriptions.

## Image-to-video (Wan 2.2 I2V)
The image already defines the look. The prompt should describe **motion only**:
> The camera slowly dollies in toward the woman as she looks up from her phone and smiles. Her hair moves
> slightly. Background stays still. Smooth, steady, cinematic.

- One camera move + one subject action. Name the speed ("slowly", "gently").
- Mention what should NOT move if the model tends to wobble it ("the building remains still").
- 81–121 frames at 16 fps ≈ 5–7.5 s.

## Text-to-video (Wan 2.2 T2V / LTX-2.x)
Describe chronologically, like a paragraph of a screenplay, with camera language:
> A wide establishing drone shot at golden hour slowly glides over a quiet main street of small shops. Warm light
> glows from the storefronts, a few pedestrians walk on the sidewalk. The camera tilts down gently toward a
> bakery entrance as the door opens.

- LTX works best with detailed, literal, chronological descriptions (lighting, camera, subject, environment).
- LTX-2 can generate ambient audio; describe sounds if wanted ("soft city ambience, distant traffic").

## Consistent character without a LoRA
Reuse the exact same description block ("a man in his 30s, short dark hair, trimmed beard, navy quarter-zip")
and the same seed family. For true consistency (Camden himself), train a LoRA — see the studio-train skill.
