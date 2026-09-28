import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from studio import cli
from studio import config as C
from tests import fake_comfy

ROOT = Path(__file__).resolve().parent.parent


class CliTests(unittest.TestCase):
    def setUp(self):
        self.server, self.state, url = fake_comfy.start()
        self.tmp = Path(tempfile.mkdtemp())
        # an image-to-video style workflow built only from nodes the fake server knows
        i2v = json.loads((ROOT / "workflows" / "smoke_sdxl_t2i.json").read_text())
        i2v["20"] = {"class_type": "LoadImage", "inputs": {"image": "example.png"}}
        (self.tmp / "i2v.json").write_text(json.dumps(i2v))
        cfg = C.load()
        cfg["comfy_url"] = url
        cfg["output_dir"] = str(self.tmp / "out")
        cfg["workflows"] = {"smoke": "workflows/smoke_sdxl_t2i.json", "image": "workflows/smoke_sdxl_t2i.json",
                            "video_i2v": str(self.tmp / "i2v.json"), "video_t2v": "workflows/smoke_sdxl_t2i.json"}
        self.patch = mock.patch.object(C, "load", return_value=cfg)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.server.shutdown()
        shutil.rmtree(self.tmp)

    def test_image_command_downloads_outputs(self):
        with mock.patch("builtins.print") as p:
            rc = cli.main(["image", "--prompt", "a red bicycle", "--seed", "3", "--width", "832", "--height", "1216",
                           "--lora", "me.safetensors:0.7", "--project", "Test Proj"])
        self.assertEqual(rc, 0)
        result = json.loads(p.call_args[0][0])
        self.assertEqual(result["seed"], 3)
        self.assertTrue(Path(result["files"][0]).exists())
        sent = list(self.state.prompts.values())[0]
        self.assertTrue(any(n["class_type"] == "LoraLoaderModelOnly" for n in sent.values()))
        self.assertTrue(sent["9"]["inputs"]["filename_prefix"].startswith("studio/test-proj/"))

    def test_shotlist_uses_keyframe_reference(self):
        shots = {"project": "promo", "defaults": {"width": 1024, "height": 576},
                 "shots": [{"id": "s01", "type": "image", "prompt": "keyframe", "seed": 1},
                           {"id": "s01v", "type": "video", "prompt": "slow push-in", "image": "@s01", "frames": 97}]}
        f = self.tmp / "shots.json"
        f.write_text(json.dumps(shots))
        with mock.patch("builtins.print"):
            rc = cli.main(["shots", str(f)])
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.state.uploads), 1)  # keyframe uploaded for the video shot
        manifest = json.loads((self.tmp / "out" / "promo" / "manifest.json").read_text())
        self.assertEqual(set(manifest["shots"]), {"s01", "s01v"})
        self.assertEqual(manifest["shots"]["s01v"]["input_image"], manifest["shots"]["s01"]["files"][0])

    def test_rejected_workflow_reports_error(self):
        bad = json.loads((ROOT / "workflows" / "smoke_sdxl_t2i.json").read_text())
        bad["3"]["class_type"] = "SomeCustomNodeNotInstalled"
        (self.tmp / "bad.json").write_text(json.dumps(bad))
        with mock.patch("sys.stderr"):
            rc = cli.main(["image", "--prompt", "x", "--workflow", str(self.tmp / "bad.json")])
        self.assertEqual(rc, 1)

    def test_free(self):
        with mock.patch("builtins.print"):
            self.assertEqual(cli.main(["free"]), 0)
        self.assertEqual(self.state.freed, 1)


if __name__ == "__main__":
    unittest.main()
