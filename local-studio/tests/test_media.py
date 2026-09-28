import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from studio import media

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not installed")
class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        t = cls.tmp
        ff("-f", "lavfi", "-i", "testsrc=size=832x480:rate=16:duration=2", "-pix_fmt", "yuv420p", str(t / "silent.mp4"))
        ff("-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=24:duration=2", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
           "-pix_fmt", "yuv420p", "-shortest", str(t / "voiced.mp4"))
        for i, color in enumerate(["red", "green", "blue", "yellow", "white"]):
            ff("-f", "lavfi", "-i", f"color=c={color}:size=1024x768", "-frames:v", "1", str(t / f"img{i}.png"))
        ff("-f", "lavfi", "-i", "color=c=orange:size=400x200", "-frames:v", "1", str(t / "logo.png"))
        ff("-f", "lavfi", "-i", "sine=frequency=220:duration=1", str(t / "music.m4a"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_frames(self):
        frames = media.extract_frames(self.tmp / "silent.mp4", self.tmp / "frames", count=4)
        self.assertEqual(len(frames), 4)
        self.assertTrue(all(p.exists() for p in frames))

    def test_contact_sheet_partial_row(self):
        imgs = [self.tmp / f"img{i}.png" for i in range(5)]
        sheet = media.contact_sheet(imgs, self.tmp / "sheet.jpg", cols=4, cell=128)
        info = media.probe(sheet)
        self.assertEqual(info["width"], 4 * 128 + 3 * 4)  # 4 cells + 3 gaps of padding
        self.assertEqual(info["height"], 2 * 128 + 1 * 4)  # 5 images -> 2 rows
        self.assertTrue(sheet.with_suffix(".index.json").exists())

    def test_assemble_mixed_inputs_with_logo_and_music(self):
        out = media.assemble([self.tmp / "silent.mp4", self.tmp / "img0.png", self.tmp / "voiced.mp4"],
                             self.tmp / "final.mp4", aspect="9:16", music=self.tmp / "music.m4a", logo=self.tmp / "logo.png")
        info = media.probe(out)
        self.assertEqual((info["width"], info["height"]), (1080, 1920))
        self.assertTrue(info["has_audio"])
        self.assertAlmostEqual(info["duration"], 7.0, delta=0.6)

    def test_reframe(self):
        out = media.reframe(self.tmp / "voiced.mp4", self.tmp / "square.mp4", aspect="1:1")
        info = media.probe(out)
        self.assertEqual((info["width"], info["height"]), (1080, 1080))


if __name__ == "__main__":
    unittest.main()
