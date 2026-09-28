"""ffmpeg helpers: QC frames, contact sheets, reframing, stitching, logo, music."""

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".gif"}

ASPECTS = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350)}


class MediaError(RuntimeError):
    pass


def _bin(name):
    exe = shutil.which(name)
    if not exe:
        raise MediaError(f"{name} not found on PATH. Install ffmpeg (it includes ffprobe).")
    return exe


def _run(args):
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MediaError(f"{Path(args[0]).name} failed:\n{proc.stderr[-2000:]}")
    return proc.stdout


def probe(path):
    out = _run([_bin("ffprobe"), "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,r_frame_rate",
                "-of", "json", str(path)])
    data = json.loads(out)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    return {
        "duration": float(data.get("format", {}).get("duration") or 0),
        "width": video.get("width"),
        "height": video.get("height"),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
    }


def extract_frames(video, out_dir, count=6):
    """Evenly spaced stills for visual QC (Claude reads these)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = probe(video)["duration"] or 1.0
    paths = []
    for i in range(count):
        t = duration * (i + 0.5) / count
        dest = out_dir / f"{Path(video).stem}_f{i + 1:02d}.jpg"
        _run([_bin("ffmpeg"), "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", "-q:v", "3", str(dest)])
        paths.append(dest)
    return paths


def contact_sheet(images, out_path, cols=4, cell=512):
    """Tile images into one grid (order is left-to-right, top-to-bottom)."""
    images = [Path(p) for p in images]
    if not images:
        raise MediaError("No images for contact sheet")
    rows = math.ceil(len(images) / cols)
    with tempfile.TemporaryDirectory() as tmp:
        for i, img in enumerate(images):
            _run([_bin("ffmpeg"), "-y", "-v", "error", "-i", str(img), "-vf",
                  f"scale={cell}:{cell}:force_original_aspect_ratio=decrease,pad={cell}:{cell}:(ow-iw)/2:(oh-ih)/2:color=black",
                  "-frames:v", "1", str(Path(tmp) / f"{i:04d}.png")])
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _run([_bin("ffmpeg"), "-y", "-v", "error", "-framerate", "1", "-i", str(Path(tmp) / "%04d.png"),
              "-vf", f"tile={cols}x{rows}:padding=4:color=white", "-frames:v", "1", str(out_path)])
    index = out_path.with_suffix(".index.json")
    index.write_text(json.dumps([str(p) for p in images], indent=2), encoding="utf-8")
    return out_path


def _fit_filter(w, h, fit):
    if fit == "crop":
        return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    return f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black"


def normalize_clip(src, dest, aspect="16:9", fit="crop", fps=24):
    """Re-encode a clip to a common size/fps/codec with a stereo audio track (silent if none)."""
    w, h = ASPECTS[aspect]
    info = probe(src)
    args = [_bin("ffmpeg"), "-y", "-v", "error", "-i", str(src)]
    if not info["has_audio"]:
        args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-shortest"]
    args += ["-vf", f"{_fit_filter(w, h, fit)},fps={fps},format=yuv420p",
             "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-c:a", "aac", "-ar", "48000", "-ac", "2", str(dest)]
    _run(args)
    return Path(dest)


def assemble(clips, out_path, aspect="16:9", fit="crop", fps=24, music=None, music_volume=0.35, logo=None):
    """Stitch clips (images become 3s stills), optionally add music and a corner logo."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parts = []
        for i, clip in enumerate(clips):
            clip = Path(clip)
            src = clip
            if clip.suffix.lower() in IMAGE_EXTS:
                src = tmp / f"still_{i:03d}.mp4"
                _run([_bin("ffmpeg"), "-y", "-v", "error", "-loop", "1", "-t", "3", "-i", str(clip), "-r", str(fps), str(src)])
            part = tmp / f"part_{i:03d}.mp4"
            normalize_clip(src, part, aspect, fit, fps)
            parts.append(part)
        listing = tmp / "list.txt"
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        joined = tmp / "joined.mp4"
        _run([_bin("ffmpeg"), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)])

        current = joined
        if logo:
            w, _ = ASPECTS[aspect]
            logo_w = int(w * 0.12)
            branded = tmp / "branded.mp4"
            _run([_bin("ffmpeg"), "-y", "-v", "error", "-i", str(current), "-i", str(logo), "-filter_complex",
                  f"[1:v]scale={logo_w}:-1[lg];[0:v][lg]overlay=W-w-40:H-h-40", "-c:a", "copy",
                  "-c:v", "libx264", "-crf", "18", str(branded)])
            current = branded
        if music:
            scored = tmp / "scored.mp4"
            _run([_bin("ffmpeg"), "-y", "-v", "error", "-i", str(current), "-stream_loop", "-1", "-i", str(music),
                  "-filter_complex", f"[1:a]volume={music_volume}[m];[0:a][m]amix=inputs=2:duration=first[a]",
                  "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-shortest", str(scored)])
            current = scored
        shutil.copyfile(current, out_path)
    return out_path


def reframe(src, out_path, aspect="9:16", fit="crop"):
    """Make a platform-specific version (e.g. 9:16 for Reels/TikTok)."""
    w, h = ASPECTS[aspect]
    _run([_bin("ffmpeg"), "-y", "-v", "error", "-i", str(src), "-vf", f"{_fit_filter(w, h, fit)},format=yuv420p",
          "-c:v", "libx264", "-crf", "18", "-c:a", "copy", str(out_path)])
    return Path(out_path)
