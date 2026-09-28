import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from studio import config as C
from studio import nightly
from tests import fake_comfy


class NightlyTests(unittest.TestCase):
    def setUp(self):
        self.server, self.state, url = fake_comfy.start()
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = C.load()
        self.cfg["comfy_url"] = url
        self.cfg["output_dir"] = str(self.tmp / "out")
        self.cfg["workflows"]["image"] = "workflows/smoke_sdxl_t2i.json"
        self.cfg["nightly"]["plan_file"] = str(self.tmp / "no_plan.json")

    def tearDown(self):
        self.server.shutdown()
        shutil.rmtree(self.tmp)

    def test_base_eval_without_plan(self):
        with mock.patch("studio.gpu.wait_until_free", return_value=(True, "ok")), \
             mock.patch("studio.gpu.status", return_value=None), \
             mock.patch("studio.media.contact_sheet", side_effect=lambda files, out, **k: out), \
             mock.patch("builtins.print"):
            rc = nightly.run(self.cfg, skip_review=True)
        self.assertEqual(rc, 0)
        day = next((self.tmp / "out" / "nightly").iterdir())
        summary = json.loads((day / "summary.json").read_text())
        prompts = json.loads(C.resolve(self.cfg["nightly"]["eval_prompts"]).read_text())["prompts"]
        expected = sum(1 for p in prompts if not p.get("needs_trigger"))
        self.assertEqual(len(summary["stages"]["eval"]["base"]), expected)
        self.assertNotIn("lora", summary["stages"]["eval"])
        self.assertGreaterEqual(self.state.freed, 1)  # VRAM handed back afterwards

    def test_skips_when_gpu_busy(self):
        with mock.patch("studio.gpu.wait_until_free", return_value=(False, "agent busy")), \
             mock.patch("studio.gpu.status", return_value=None), mock.patch("builtins.print"):
            rc = nightly.run(self.cfg, skip_review=True)
        self.assertEqual(rc, 3)
        self.assertEqual(self.state.prompts, {})


if __name__ == "__main__":
    unittest.main()
