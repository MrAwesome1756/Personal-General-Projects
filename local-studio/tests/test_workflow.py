import json
import tempfile
import unittest
from pathlib import Path

from studio import workflow as W

ROOT = Path(__file__).resolve().parent.parent


def smoke():
    return json.loads((ROOT / "workflows" / "smoke_sdxl_t2i.json").read_text())


def qwen_like():
    """Mimics the stock Qwen-Image/FLUX template: negative is a zeroed copy of the positive."""
    return {
        "37": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen.safetensors", "weight_dtype": "default"}},
        "38": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_vl.safetensors", "type": "qwen_image"}},
        "66": {"class_type": "ModelSamplingAuraFlow", "inputs": {"shift": 3.1, "model": ["37", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "old", "clip": ["38", 0]}},
        "7": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
        "58": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1328, "height": 1328, "batch_size": 1}},
        "3": {"class_type": "KSampler", "inputs": {"seed": 5, "steps": 20, "cfg": 2.5, "sampler_name": "euler",
                                                   "scheduler": "simple", "denoise": 1, "model": ["66", 0],
                                                   "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["58", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["39", 0]}},
        "39": {"class_type": "VAELoader", "inputs": {"vae_name": "vae.safetensors"}},
        "60": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]}},
    }


def wan_i2v_like():
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "pos", "clip": ["9", 0]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "neg", "clip": ["9", 0]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": "example.png"}},
        "5": {"class_type": "WanImageToVideo", "inputs": {"width": 832, "height": 480, "length": 81, "batch_size": 1,
                                                          "positive": ["2", 0], "negative": ["3", 0], "vae": ["10", 0],
                                                          "start_image": ["4", 0]}},
        "6": {"class_type": "KSamplerAdvanced", "inputs": {"noise_seed": 1, "steps": 20, "model": ["1", 0],
                                                           "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["5", 2]}},
        "7": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "video/ComfyUI", "video": ["8", 0]}},
        "8": {"class_type": "CreateVideo", "inputs": {"fps": 16, "images": ["11", 0]}},
        "9": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5.safetensors", "type": "wan"}},
        "10": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_vae.safetensors"}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["10", 0]}},
    }


class ApplyParamsTests(unittest.TestCase):
    def test_smoke_auto_detect(self):
        wf, report = W.apply_params(smoke(), prompt="P", negative="N", seed=42, width=768, height=512, filename_prefix="x/y")
        self.assertEqual(wf["6"]["inputs"]["text"], "P")
        self.assertEqual(wf["7"]["inputs"]["text"], "N")
        self.assertEqual(wf["3"]["inputs"]["seed"], 42)
        self.assertEqual((wf["5"]["inputs"]["width"], wf["5"]["inputs"]["height"]), (768, 512))
        self.assertEqual(wf["9"]["inputs"]["filename_prefix"], "x/y")
        self.assertTrue(report)

    def test_original_not_mutated(self):
        base = smoke()
        W.apply_params(base, prompt="changed")
        self.assertNotEqual(base["6"]["inputs"]["text"], "changed")

    def test_zeroed_negative_does_not_clobber_positive(self):
        wf, report = W.apply_params(qwen_like(), prompt="P", negative="N")
        self.assertEqual(wf["6"]["inputs"]["text"], "P")
        self.assertTrue(any("no separate negative" in r.get("note", "") for r in report))

    def test_video_graph(self):
        wf, _ = W.apply_params(wan_i2v_like(), prompt="P", negative="N", seed=9, width=1280, height=720,
                               frames=121, image="kf.png", filename_prefix="studio/p/s1")
        self.assertEqual(wf["2"]["inputs"]["text"], "P")
        self.assertEqual(wf["3"]["inputs"]["text"], "N")
        self.assertEqual(wf["6"]["inputs"]["noise_seed"], 9)
        self.assertEqual(wf["5"]["inputs"]["length"], 121)
        self.assertEqual(wf["5"]["inputs"]["width"], 1280)
        self.assertEqual(wf["4"]["inputs"]["image"], "kf.png")
        self.assertEqual(wf["7"]["inputs"]["filename_prefix"], "studio/p/s1")

    def test_titles_override_auto(self):
        wf = smoke()
        wf["7"]["_meta"]["title"] = "PROMPT"  # deliberately odd: user says node 7 is the prompt
        out, _ = W.apply_params(wf, prompt="P")
        self.assertEqual(out["7"]["inputs"]["text"], "P")
        self.assertNotEqual(out["6"]["inputs"]["text"], "P")

    def test_map_file_overrides_everything(self):
        mapping = {"prompt": [["7", "text"]], "width": [["5", "width"]]}
        out, _ = W.apply_params(smoke(), mapping, prompt="P", width=640, height=480)
        self.assertEqual(out["7"]["inputs"]["text"], "P")
        self.assertEqual(out["5"]["inputs"]["width"], 640)
        self.assertEqual(out["5"]["inputs"]["height"], 1024)  # height not mapped -> untouched

    def test_image_on_text_only_workflow_errors(self):
        with self.assertRaises(W.WorkflowError):
            W.apply_params(smoke(), image="x.png")


class LoraTests(unittest.TestCase):
    def test_add_lora_rewires_model_consumers(self):
        wf, created = W.add_lora(qwen_like(), "me.safetensors", 0.8)
        new = created[0]
        self.assertEqual(wf[new]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(wf[new]["inputs"]["model"], ["37", 0])
        self.assertEqual(wf["66"]["inputs"]["model"], [new, 0])

    def test_checkpoint_clip_output_untouched(self):
        wf, created = W.add_lora(smoke(), "me.safetensors")
        self.assertEqual(wf["3"]["inputs"]["model"], [created[0], 0])
        self.assertEqual(wf["6"]["inputs"]["clip"], ["4", 1])
        self.assertEqual(wf["8"]["inputs"]["vae"], ["4", 2])


class LoadTests(unittest.TestCase):
    def test_rejects_ui_format(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ui.json"
            p.write_text(json.dumps({"nodes": [], "links": []}))
            with self.assertRaises(W.WorkflowError):
                W.load(p)

    def test_missing_file_message(self):
        with self.assertRaises(W.WorkflowError) as ctx:
            W.load(ROOT / "workflows" / "nope.json")
        self.assertIn("Export (API)", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
