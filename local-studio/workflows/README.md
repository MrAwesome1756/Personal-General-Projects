# Workflows

The studio CLI runs ComfyUI workflows saved in **API format**. Four are expected (names set in `studio.config.json`):

| File | Purpose | Suggested stock template in ComfyUI |
|---|---|---|
| `smoke_sdxl_t2i.json` | Smoke test (included). Needs `sd_xl_base_1.0.safetensors` in `models/checkpoints`. | — |
| `image_t2i.json` | Main photoreal text-to-image | "Qwen-Image text to image" (commercial-safe) or a FLUX.2 template (non-commercial) |
| `video_i2v.json` | Animate an approved keyframe | "Wan 2.2 14B image to video" |
| `video_t2v.json` | Text-to-video | "Wan 2.2 14B text to video" or an LTX-2 template |

## How to add one (about 5 minutes each)
1. In ComfyUI, open **Workflow → Browse Templates** and pick the template. ComfyUI lists any missing model files
   with download links. Download them and let them land in the folders it names.
2. Press **Run** once in the UI and confirm you get a good result. This proves the models and nodes are right.
3. *(Optional but recommended)* Give key nodes these exact titles (right-click → Title): `PROMPT`, `NEGATIVE`,
   `SEED`, `SIZE`, `FRAMES`, `INPUT_IMAGE`, `OUTPUT`. The CLI auto-detects most stock templates anyway. Titles
   just remove the guesswork.
4. **Workflow → Export (API)** and save it here with the filename above. If "Export (API)" isn't in the menu,
   turn on dev mode in ComfyUI settings.
5. Run `python -m studio doctor`. It checks every node the workflow needs is installed.
6. Test: `python -m studio image --prompt "test" --project smoke`. The JSON output lists `changed_inputs`,
   so check the prompt, seed and size landed on the right nodes.

## Optional explicit map
If auto-detection picks the wrong node, add `<workflow>.map.json` next to it:
```json
{"prompt": [["6", "text"]], "negative": [["7", "text"]], "seed": [["3", "seed"]],
 "width": [["58", "width"]], "height": [["58", "height"]], "frames": [["55", "length"]],
 "image": [["52", "image"]], "filename_prefix": [["60", "filename_prefix"]]}
```
Node ids come from the exported JSON.
