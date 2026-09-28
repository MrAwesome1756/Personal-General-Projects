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

## Objects, materials and text (the main focus)
Structure: **[shot type + object] + [materials, named precisely] + [surface/set] + [light: type, direction, what it
does to the material] + [lens/aperture] + [angle] + [texture proof details]**

> Premium product photograph of a matte black ceramic pour-over coffee dripper with a raw walnut base, on wet
> dark slate. Long strip softbox from camera left creating a clean gradient along the glaze, thin rim light from
> behind. 100mm macro lens at f/5.6, three-quarter angle slightly above. Visible glaze speckle, wood grain, water
> droplets beading on the slate.

In-image text:
- Put the exact text in quotes, keep it short (1–4 words), and name the method: "embossed in gold foil",
  "laser-engraved", "screen-printed", "debossed into leather", "etched into glass".
- Name the typeface feel ("clean serif capitals", "condensed sans-serif") and where the text sits.
- Qwen-Image is the strongest local model for text. If text still fails, render without it and add it in the edit.

Object motion (image-to-video from an approved still):
> The camera slowly orbits 90 degrees around the watch from left to right while a soft highlight sweeps across
> the brushed steel case. The watch stays perfectly still and undistorted. Smooth, steady, premium commercial.

- Tell the model what must stay rigid ("the bottle keeps its exact shape", "the text stays sharp and legible").
- Liquids: describe the physics plainly ("amber liquid swirls once and settles, small bubbles rise").

## Consistent object across shots
Reuse the exact same description block for the object (materials, colors, proportions, any text) and the same
seed family. For a reusable look across many products, train a style LoRA (studio-train skill).
