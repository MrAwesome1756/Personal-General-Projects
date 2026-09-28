"""Load ComfyUI API-format workflows and inject parameters.

A workflow exported from ComfyUI with "Export (API)" is a dict of
{node_id: {"class_type": ..., "inputs": {...}, "_meta": {"title": ...}}}.
Linked inputs look like ["<node_id>", <output_index>].

Parameters are located, in priority order:
  1. an explicit sidecar map file  <workflow>.map.json
  2. node titles set in the ComfyUI editor (PROMPT, NEGATIVE, SEED, SIZE,
     FRAMES, INPUT_IMAGE, OUTPUT)
  3. auto-detection by graph structure (works for most stock templates)

`apply_params` returns a report of every input it changed, so a caller (or
Claude) can confirm the right nodes were touched.
"""

import copy
import json
from collections import deque
from pathlib import Path

TEXT_KEYS = ("text", "prompt", "text_g", "text_l")
SEED_KEYS = ("seed", "noise_seed")
FRAME_KEYS = ("length", "num_frames", "frames", "video_length")
# Conditioning nodes that must not be traversed when tracing a negative prompt
# (e.g. Qwen/FLUX templates derive the negative by zeroing the positive).
STOP_CLASSES = {"ConditioningZeroOut"}
TITLE_KEYS = {"PROMPT", "NEGATIVE", "SEED", "SIZE", "FRAMES", "INPUT_IMAGE", "OUTPUT"}


class WorkflowError(ValueError):
    pass


def load(path):
    path = Path(path)
    if not path.exists():
        raise WorkflowError(
            f"Workflow file not found: {path}. Export it from ComfyUI with Workflow > Export (API) "
            f"and save it there (see workflows/README.md)."
        )
    wf = json.loads(path.read_text(encoding="utf-8"))
    if "nodes" in wf and "links" in wf:
        raise WorkflowError(
            f"{path} is a UI-format workflow. Re-export it with Workflow > Export (API) "
            f"(enable dev mode in ComfyUI settings if that menu item is hidden)."
        )
    mapping_path = path.with_suffix(".map.json")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else {}
    return wf, mapping


def _is_link(value):
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)


def _title(node):
    return (node.get("_meta", {}).get("title") or "").strip().upper()


def _titled(wf, title):
    return [nid for nid, node in wf.items() if _title(node) == title]


def _text_fields(node):
    return [k for k in TEXT_KEYS if isinstance(node.get("inputs", {}).get(k), str)]


def _trace_text_nodes(wf, start_id, role=None):
    """Walk upstream from a conditioning link to the text-encode node(s) feeding it.

    When passing through a node that itself has positive/negative inputs (e.g.
    WanImageToVideo), only the input matching `role` is followed, so the
    positive trace never leaks into the negative branch.
    """
    found, seen, queue = [], set(), deque([start_id])
    while queue:
        nid = queue.popleft()
        if nid in seen or nid not in wf:
            continue
        seen.add(nid)
        node = wf[nid]
        if node.get("class_type") in STOP_CLASSES:
            continue
        if _text_fields(node):
            found.append(nid)
            continue
        inputs = node.get("inputs", {})
        if role and _is_link(inputs.get(role)):
            queue.append(inputs[role][0])
            continue
        for key, value in inputs.items():
            if _is_link(value) and key not in ("positive", "negative"):
                queue.append(value[0])
    return found


def _auto_prompt_nodes(wf):
    positive, negative = [], []
    for node in wf.values():
        inputs = node.get("inputs", {})
        if _is_link(inputs.get("positive")):
            positive += _trace_text_nodes(wf, inputs["positive"][0], "positive")
        if _is_link(inputs.get("negative")):
            negative += _trace_text_nodes(wf, inputs["negative"][0], "negative")
    if not positive:
        for node in wf.values():
            cond = node.get("inputs", {}).get("conditioning")
            if _is_link(cond) and "Guider" in node.get("class_type", ""):
                positive += _trace_text_nodes(wf, cond[0])
    if not positive:
        text_nodes = [nid for nid, node in wf.items() if _text_fields(node)]
        if len(text_nodes) == 1:
            positive = text_nodes
    positive = list(dict.fromkeys(positive))
    negative = [n for n in dict.fromkeys(negative) if n not in positive]
    return positive, negative


def _targets(wf, mapping, key, title, finder):
    """Return [(node_id, input_name)] for a parameter."""
    if key in mapping:
        return [tuple(t) for t in mapping[key]]
    titled = _titled(wf, title) if title else []
    if titled:
        return finder(titled)
    return finder(None)


def apply_params(workflow, mapping=None, *, prompt=None, negative=None, seed=None, width=None,
                 height=None, frames=None, steps=None, image=None, filename_prefix=None):
    """Return (new_workflow, report). `image` is a ComfyUI input filename (already uploaded)."""
    wf = copy.deepcopy(workflow)
    mapping = mapping or {}
    report = []

    def set_input(nid, key, value):
        if nid not in wf:
            raise WorkflowError(f"Map refers to missing node {nid}")
        old = wf[nid]["inputs"].get(key)
        wf[nid]["inputs"][key] = value
        report.append({"node": nid, "class": wf[nid].get("class_type"), "input": key,
                       "old": old if not isinstance(old, str) else old[:60], "new": value if not isinstance(value, str) else value[:60]})

    auto_pos, auto_neg = _auto_prompt_nodes(wf)

    def text_finder(auto_ids):
        def finder(titled):
            ids = titled if titled is not None else auto_ids
            return [(nid, k) for nid in ids for k in _text_fields(wf[nid])]
        return finder

    def literal_finder(keys, require=None):
        def finder(titled):
            ids = titled if titled is not None else list(wf)
            out = []
            for nid in ids:
                inputs = wf[nid].get("inputs", {})
                if require and not require(wf[nid]):
                    continue
                out += [(nid, k) for k in keys if k in inputs and isinstance(inputs[k], (int, float)) and not isinstance(inputs[k], bool)]
            return out
        return finder

    def has_size(node):
        inputs = node.get("inputs", {})
        return isinstance(inputs.get("width"), int) and isinstance(inputs.get("height"), int)

    if prompt is not None:
        targets = _targets(wf, mapping, "prompt", "PROMPT", text_finder(auto_pos))
        if not targets:
            raise WorkflowError("Could not find the positive prompt node. Title it PROMPT in ComfyUI or add a .map.json.")
        for nid, key in targets:
            set_input(nid, key, prompt)

    if negative is not None:
        targets = _targets(wf, mapping, "negative", "NEGATIVE", text_finder(auto_neg))
        if not targets:
            report.append({"note": "workflow has no separate negative prompt; --negative ignored"})
        for nid, key in targets:
            set_input(nid, key, negative)

    if seed is not None:
        for nid, key in _targets(wf, mapping, "seed", "SEED", literal_finder(SEED_KEYS)):
            set_input(nid, key, int(seed))

    if width is not None or height is not None:
        if "width" in mapping or "height" in mapping:
            w_targets = [tuple(t) for t in mapping.get("width", [])]
            h_targets = [tuple(t) for t in mapping.get("height", [])]
        else:
            size_nodes = _targets(wf, mapping, "size", "SIZE", literal_finder(("width", "height"), require=has_size))
            w_targets = [t for t in size_nodes if t[1] == "width"]
            h_targets = [t for t in size_nodes if t[1] == "height"]
        for nid, key in w_targets if width is not None else []:
            set_input(nid, key, int(width))
        for nid, key in h_targets if height is not None else []:
            set_input(nid, key, int(height))

    if frames is not None:
        targets = _targets(wf, mapping, "frames", "FRAMES", literal_finder(FRAME_KEYS))
        if not targets:
            report.append({"note": "no frame-count input found; --frames ignored"})
        for nid, key in targets:
            set_input(nid, key, int(frames))

    if steps is not None:
        for nid, key in _targets(wf, mapping, "steps", None, literal_finder(("steps",))):
            set_input(nid, key, int(steps))

    if image is not None:
        def image_finder(titled):
            ids = titled if titled is not None else [n for n, node in wf.items() if node.get("class_type") == "LoadImage"]
            return [(nid, "image") for nid in ids[:1]]
        targets = _targets(wf, mapping, "image", "INPUT_IMAGE", image_finder)
        if not targets:
            raise WorkflowError("This workflow has no LoadImage node, so it cannot take an input image.")
        for nid, key in targets:
            set_input(nid, key, image)

    if filename_prefix is not None:
        def prefix_finder(titled):
            ids = titled if titled is not None else list(wf)
            return [(nid, "filename_prefix") for nid in ids if isinstance(wf[nid].get("inputs", {}).get("filename_prefix"), str)]
        for nid, key in _targets(wf, mapping, "filename_prefix", "OUTPUT", prefix_finder):
            set_input(nid, key, filename_prefix)

    return wf, report


MODEL_LOADERS = {"UNETLoader", "CheckpointLoaderSimple", "UnetLoaderGGUF", "DiffusionModelLoader"}


def add_lora(workflow, lora_name, strength=1.0):
    """Insert a LoraLoaderModelOnly after every model loader and rewire its consumers.

    Returns (new_workflow, [new_node_ids]). Only the MODEL output (index 0) is rewired.
    """
    wf = copy.deepcopy(workflow)
    loaders = [nid for nid, node in wf.items() if node.get("class_type") in MODEL_LOADERS]
    if not loaders:
        raise WorkflowError("No model loader node found to attach a LoRA to.")
    next_id = max(int(n) for n in wf if n.isdigit()) + 1 if any(n.isdigit() for n in wf) else 1000
    created = []
    for loader in loaders:
        new_id = str(next_id)
        next_id += 1
        for nid, node in wf.items():
            for key, value in node.get("inputs", {}).items():
                if _is_link(value) and value[0] == loader and value[1] == 0:
                    node["inputs"][key] = [new_id, 0]
        wf[new_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": [loader, 0], "lora_name": lora_name, "strength_model": float(strength)},
            "_meta": {"title": f"LoRA {lora_name}"},
        }
        created.append(new_id)
    return wf, created


def required_classes(workflow):
    return sorted({node.get("class_type") for node in workflow.values() if node.get("class_type")})
