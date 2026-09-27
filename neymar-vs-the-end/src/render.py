"""Render driver.

  python render.py sheet  [fps=1] [out.png]        contact sheet for review
  python render.py still  <seconds> [out.png]       single full-res frame
  python render.py video  [out.mp4] [--audio a.wav] [--start s --end e]
"""
import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from core import FPS, H, W, post
from shots import DURATION, render_raw

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")


def frame(i):
    c, fx = render_raw(i / FPS)
    img = post(c, i, fx)
    return (img * 255 + 0.5).astype(np.uint8)


def _chunk(args):
    start, end, path = args
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
           "-pix_fmt", "yuv420p", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(start, end):
        p.stdin.write(frame(i).tobytes())
    p.stdin.close()
    p.wait()
    return path


def video(out, audio=None, start=0.0, end=DURATION, workers=None):
    workers = workers or os.cpu_count() or 4
    f0, f1 = int(round(start * FPS)), int(round(end * FPS))
    n = f1 - f0
    tmp = tempfile.mkdtemp(prefix="nvte_", dir=os.environ.get("TMPDIR"))
    k = workers * 3
    bounds = [f0 + n * i // k for i in range(k + 1)]
    jobs = [(bounds[i], bounds[i + 1], os.path.join(tmp, f"c{i:03d}.mp4")) for i in range(k) if bounds[i + 1] > bounds[i]]
    with mp.Pool(workers) as pool:
        for j, p in enumerate(pool.imap(_chunk, jobs)):
            print(f"chunk {j + 1}/{len(jobs)} done", flush=True)
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as fh:
        for _, _, p in jobs:
            fh.write(f"file '{p}'\n")
    silent = os.path.join(tmp, "silent.mp4")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent],
                   check=True)
    if audio:
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", silent, "-ss", str(start), "-t", str(end - start),
                        "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                        "-shortest", "-movflags", "+faststart", out], check=True)
    else:
        os.replace(silent, out)
    print("wrote", out)


def _small(i):
    img = frame(i)
    return cv2.resize(img, (W // 5, H // 5), interpolation=cv2.INTER_AREA)


def sheet(fps=1.0, out="sheet.png", start=0.0, end=DURATION, cols=10):
    idx = [int(round(t * FPS)) for t in np.arange(start, end, 1.0 / fps)]
    with mp.Pool(os.cpu_count() or 4) as pool:
        ims = pool.map(_small, idx)
    h, w = ims[0].shape[:2]
    rows = (len(ims) + cols - 1) // cols
    S = np.zeros((rows * (h + 22), cols * w, 3), np.uint8)
    for n, (i, im) in enumerate(zip(idx, ims)):
        r, c = divmod(n, cols)
        S[r * (h + 22) + 22:r * (h + 22) + 22 + h, c * w:(c + 1) * w] = im
        cv2.putText(S, f"{i / FPS:.1f}s", (c * w + 4, r * (h + 22) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out, S[..., ::-1])
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sheet", "still", "video"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--audio")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=DURATION)
    ap.add_argument("--cols", type=int, default=10)
    a = ap.parse_args()
    if a.mode == "sheet":
        sheet(float(a.arg or 1), a.out or "sheet.png", a.start, a.end, a.cols)
    elif a.mode == "still":
        img = frame(int(round(float(a.arg) * FPS)))
        cv2.imwrite(a.out or "still.png", img[..., ::-1])
    else:
        video(a.arg or "out.mp4", a.audio, a.start, a.end)
    sys.exit(0)
