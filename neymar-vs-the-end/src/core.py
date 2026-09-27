"""Core rendering engine: camera, textured-quad blitting, text, particles, post FX.

Everything is a pure function of time so frames can be rendered in parallel.
World units are Minecraft texels (16 per block). World y points DOWN; the
island's walking surface is y = 0.
"""
import math
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
BPM = 120
BEAT = 60.0 / BPM
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(ROOT, "assets", "fonts")

cv2.setNumThreads(1)


# ----------------------------------------------------------------- easing ---
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def inv(a, b, x):
    return clamp((x - a) / (b - a)) if b != a else 0.0


def smooth(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def ease_out(t, p=3):
    t = clamp(t)
    return 1 - (1 - t) ** p


def ease_in(t, p=3):
    return clamp(t) ** p


def ease_io(t):
    t = clamp(t)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def back_out(t, k=1.9):
    t = clamp(t) - 1
    return 1 + (k + 1) * t ** 3 + k * t ** 2


def elastic_out(t):
    t = clamp(t)
    if t in (0, 1):
        return t
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * (2 * math.pi / 3)) + 1


def keys(t, pts, ease=smooth):
    """Piecewise interpolation through [(time, value), ...]; values may be tuples."""
    if t <= pts[0][0]:
        return pts[0][1]
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t <= t1:
            k = ease(inv(t0, t1, t))
            if isinstance(v0, (tuple, list)):
                return tuple(lerp(a, b, k) for a, b in zip(v0, v1))
            return lerp(v0, v1, k)
    return pts[-1][1]


def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def hash01(*args):
    """Deterministic pseudo-random in [0,1) from integers/floats."""
    x = 0
    for a in args:
        x = (x * 1000003) ^ int(a * 7919 + 13)
        x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x45D9F3B) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x45D9F3B) & 0xFFFFFFFF
    x ^= x >> 16
    return (x & 0xFFFFFF) / float(0x1000000)


def noise1(t, seed=0):
    """Smooth 1D value noise in [-1,1]."""
    i = math.floor(t)
    f = t - i
    a = hash01(i, seed) * 2 - 1
    b = hash01(i + 1, seed) * 2 - 1
    return lerp(a, b, f * f * (3 - 2 * f))


# ----------------------------------------------------------------- camera ---
class Cam:
    def __init__(self, cx=0.0, cy=-40.0, s=6.0, roll=0.0, shake=0.0, t=0.0):
        self.cx, self.cy, self.s, self.roll = cx, cy, s, roll
        if shake:
            self.cx += noise1(t * 23.0, 1) * shake / s
            self.cy += noise1(t * 21.0, 2) * shake / s
            self.roll += noise1(t * 17.0, 3) * shake * 0.04

    def M(self):
        s = self.s
        return np.array([[s, 0, W / 2 - self.cx * s], [0, s, H / 2 - self.cy * s]], np.float64)

    def to_screen(self, x, y):
        return ((x - self.cx) * self.s + W / 2, (y - self.cy) * self.s + H / 2)

    def layer(self, depth):
        """Camera for a parallax layer. depth 1 = world plane, <1 = farther."""
        s = 6.0 * (self.s / 6.0) ** depth
        c = Cam(self.cx * depth, self.cy * depth, s, 0)
        return c


def mat3(m):
    return np.vstack([m, [0, 0, 1]])


def T(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], np.float64)


def R(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def S(sx, sy=None):
    return np.array([[sx, 0, 0], [0, sx if sy is None else sy, 0], [0, 0, 1]], np.float64)


# --------------------------------------------------------------- textures ---
def tex_from_rows(rows, pal, shade=1.0):
    """Build a premultiplied RGBA float texture from ASCII rows + palette."""
    h, w = len(rows), max(len(r) for r in rows)
    a = np.zeros((h, w, 4), np.float32)
    for y, r in enumerate(rows):
        for x, ch in enumerate(r):
            if ch in pal and pal[ch] is not None:
                c = pal[ch]
                a[y, x, :3] = c * shade
                a[y, x, 3] = 1.0
    return a


_UPCACHE = {}


def upscaled(tex, k):
    key = (id(tex), k)
    u = _UPCACHE.get(key)
    if u is None:
        u = cv2.resize(tex, (tex.shape[1] * k, tex.shape[0] * k), interpolation=cv2.INTER_NEAREST)
        if len(_UPCACHE) > 3000:
            _UPCACHE.clear()
        _UPCACHE[key] = u
    return u


def blit(canvas, tex, M, alpha=1.0, add=False, pixel=True, tint=None):
    """Composite premultiplied RGBA `tex` onto float canvas using affine M (tex px -> screen px)."""
    M = np.asarray(M, np.float64)
    if M.shape == (3, 3):
        M = M[:2]
    h, w = tex.shape[:2]
    if pixel:
        sc = math.sqrt(abs(np.linalg.det(M[:, :2])))
        k = int(clamp(round(sc), 1, 32))
        if k > 1:
            tex = upscaled(tex, k)
            M = M @ np.array([[1.0 / k, 0, 0], [0, 1.0 / k, 0], [0, 0, 1]])
            h, w = tex.shape[:2]
    corners = M @ np.array([[0, w, 0, w], [0, 0, h, h], [1, 1, 1, 1]], np.float64)
    CH, CW = canvas.shape[:2]
    x0 = max(int(math.floor(corners[0].min())) - 1, 0)
    y0 = max(int(math.floor(corners[1].min())) - 1, 0)
    x1 = min(int(math.ceil(corners[0].max())) + 1, CW)
    y1 = min(int(math.ceil(corners[1].max())) + 1, CH)
    if x1 <= x0 or y1 <= y0:
        return
    M2 = M.copy()
    M2[0, 2] -= x0
    M2[1, 2] -= y0
    out = cv2.warpAffine(tex, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = out[..., 3:4] * alpha
    rgb = out[..., :3] * alpha
    if tint is not None:
        rgb = rgb * tint
    reg = canvas[y0:y1, x0:x1]
    if canvas.shape[2] == 4:
        if add:
            reg[..., :3] += rgb
        else:
            reg[..., :3] = reg[..., :3] * (1 - a) + rgb
            reg[..., 3:4] = reg[..., 3:4] * (1 - a) + a
    else:
        if add:
            reg += rgb
        else:
            reg[:] = reg * (1 - a) + rgb


def comp(dst, src_rgba, alpha=1.0):
    """Composite a full-frame premultiplied RGBA layer onto dst RGB."""
    a = src_rgba[..., 3:4] * alpha
    dst *= (1 - a)
    dst += src_rgba[..., :3] * alpha


def solid(color, h=H, w=W):
    c = np.empty((h, w, 3), np.float32)
    c[:] = color
    return c


def new_layer():
    return np.zeros((H, W, 4), np.float32)


def rect(canvas, x0, y0, x1, y1, color, alpha=1.0, add=False):
    CH, CW = canvas.shape[:2]
    xa, xb = int(max(0, min(x0, x1))), int(min(CW, max(x0, x1)))
    ya, yb = int(max(0, min(y0, y1))), int(min(CH, max(y0, y1)))
    if xb <= xa or yb <= ya:
        return
    reg = canvas[ya:yb, xa:xb]
    col = np.asarray(color, np.float32)
    if canvas.shape[2] == 4:
        reg[..., :3] = reg[..., :3] * (1 - alpha) + col * alpha
        reg[..., 3] = reg[..., 3] * (1 - alpha) + alpha
    elif add:
        reg += col * alpha
    else:
        reg[:] = reg * (1 - alpha) + col * alpha


def poly(canvas, pts, color, alpha=1.0, add=False):
    """Anti-aliased filled polygon (screen px)."""
    pts = np.asarray(pts, np.float64)
    CH, CW = canvas.shape[:2]
    x0 = max(int(pts[:, 0].min()) - 2, 0)
    y0 = max(int(pts[:, 1].min()) - 2, 0)
    x1 = min(int(pts[:, 0].max()) + 3, CW)
    y1 = min(int(pts[:, 1].max()) + 3, CH)
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    p = np.round((pts - [x0, y0]) * 16).astype(np.int32)
    cv2.fillPoly(m, [p], 255, lineType=cv2.LINE_AA, shift=4)
    a = (m.astype(np.float32) / 255.0)[..., None] * alpha
    reg = canvas[y0:y1, x0:x1]
    col = np.asarray(color, np.float32)
    if canvas.shape[2] == 4:
        reg[..., :3] = reg[..., :3] * (1 - a) + col * a
        reg[..., 3:4] = reg[..., 3:4] * (1 - a) + a
    elif add:
        reg += col * a
    else:
        reg[:] = reg * (1 - a) + col * a


def line(canvas, p0, p1, color, width=2.0, alpha=1.0, add=False):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1e-6
    nx, ny = -dy / L * width / 2, dx / L * width / 2
    poly(canvas, [(x0 + nx, y0 + ny), (x1 + nx, y1 + ny), (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)],
         color, alpha, add)


def disc(canvas, cx, cy, r, color, alpha=1.0, add=False, soft=0.0):
    CH, CW = canvas.shape[:2]
    R_ = r + soft + 2
    x0, x1 = int(max(0, cx - R_)), int(min(CW, cx + R_ + 1))
    y0, y1 = int(max(0, cy - R_)), int(min(CH, cy + R_ + 1))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    if soft > 0:
        a = np.clip(1 - (d - r) / soft, 0, 1)
        a = a * a
    else:
        a = np.clip(r - d + 0.5, 0, 1)
    a = (a * alpha)[..., None]
    reg = canvas[y0:y1, x0:x1]
    col = np.asarray(color, np.float32)
    if add:
        reg[..., :3] += col * a
    else:
        reg[..., :3] = reg[..., :3] * (1 - a) + col * a
        if canvas.shape[2] == 4:
            reg[..., 3:4] = reg[..., 3:4] * (1 - a) + a


def glow_from(layer_rgba, color, radius=18, strength=1.0):
    """Colored glow computed from a layer's alpha; returns RGB additive image."""
    a = layer_rgba[..., 3]
    small = cv2.resize(a, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    r = max(1, radius // 4)
    small = cv2.GaussianBlur(small, (0, 0), r)
    big = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    return big[..., None] * (np.asarray(color, np.float32) * strength)


# ------------------------------------------------------------------- text ---
@lru_cache(maxsize=16)
def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


@lru_cache(maxsize=1024)
def text_tex(text, size=200, style="gold", fontname="Anton-Regular.ttf", spacing=0.0):
    """Pre-render styled title text to premultiplied RGBA float array."""
    f = font(fontname, size)
    pad = int(size * 0.45)
    bbox = f.getbbox(text)
    tw = bbox[2] - bbox[0] + int(spacing * size * max(0, len(text) - 1))
    th = bbox[3] - bbox[1]
    Wt, Ht = tw + pad * 2, th + pad * 2
    ox, oy = pad - bbox[0], pad - bbox[1]

    def draw_mask(stroke=0, dx=0, dy=0):
        im = Image.new("L", (Wt, Ht), 0)
        d = ImageDraw.Draw(im)
        if spacing:
            x = ox
            for ch in text:
                d.text((x + dx, oy + dy), ch, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
                x += f.getlength(ch) + spacing * size
        else:
            d.text((ox + dx, oy + dy), text, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
        return np.asarray(im, np.float32) / 255.0

    fill = draw_mask()
    yy = np.linspace(0, 1, Ht, dtype=np.float32)[:, None]
    yt = np.clip((yy * Ht - oy) / max(th, 1), 0, 1)[..., None]
    out = np.zeros((Ht, Wt, 4), np.float32)
    if style == "gold":
        top, mid, bot = hexc("#fff3b0"), hexc("#f4b41a"), hexc("#9a5a06")
        grad = np.where(yt < 0.5, top + (mid - top) * (yt / 0.5), mid + (bot - mid) * ((yt - 0.5) / 0.5))
        grad = np.broadcast_to(grad, (Ht, Wt, 3))
        band = np.exp(-((yt - 0.52) ** 2) / 0.002) * 0.35  # horizon glint
        col = np.clip(grad + band, 0, 1.2)
        stroke = draw_mask(max(3, size // 22))
        extr = np.zeros_like(fill)
        depth = max(4, size // 18)
        for i in range(1, depth + 1):
            extr = np.maximum(extr, draw_mask(max(3, size // 22), 0, i))
        shadow = cv2.GaussianBlur(draw_mask(max(3, size // 22), size // 30, size // 14), (0, 0), size / 18)
        base = np.maximum(stroke, extr)
        # layered: shadow -> dark extrusion -> outline -> gradient fill + bevel highlight
        out[..., :3] = 0
        out[..., 3] = np.clip(shadow * 0.7, 0, 1)
        dark = hexc("#2a1402")
        ex_col = hexc("#6b3a05")
        a = extr
        out[..., :3] = out[..., :3] * (1 - a[..., None]) + ex_col * a[..., None]
        out[..., 3] = out[..., 3] * (1 - a) + a
        a = stroke
        out[..., :3] = out[..., :3] * (1 - a[..., None]) + dark * a[..., None]
        out[..., 3] = out[..., 3] * (1 - a) + a
        inner = cv2.GaussianBlur(fill, (0, 0), max(1, size / 60))
        bevel = np.clip((fill - np.roll(inner, size // 40, axis=0)) * 2.5, 0, 1)[..., None]
        c = np.clip(col + bevel * 0.6, 0, 1)
        a = fill
        out[..., :3] = out[..., :3] * (1 - a[..., None]) + c * a[..., None]
        out[..., 3] = out[..., 3] * (1 - a) + a
        _ = base
    elif style in ("silver", "white", "purple", "red"):
        palettes = {
            "silver": ("#ffffff", "#d6d6e6", "#7d7d92"),
            "white": ("#ffffff", "#ffffff", "#e8e0ff"),
            "purple": ("#fbe3ff", "#d27bff", "#7a22c9"),
            "red": ("#ffe0d0", "#ff4a2a", "#8a0c0c"),
        }
        top, mid, bot = (hexc(p) for p in palettes[style])
        grad = np.where(yt < 0.5, top + (mid - top) * (yt / 0.5), mid + (bot - mid) * ((yt - 0.5) / 0.5))
        grad = np.broadcast_to(grad, (Ht, Wt, 3))
        stroke = draw_mask(max(3, size // 20))
        shadow = cv2.GaussianBlur(draw_mask(max(3, size // 20), size // 30, size // 14), (0, 0), size / 16)
        out[..., 3] = np.clip(shadow * 0.75, 0, 1)
        dark = hexc("#12071e") if style != "red" else hexc("#200404")
        a = stroke
        out[..., :3] = out[..., :3] * (1 - a[..., None]) + dark * a[..., None]
        out[..., 3] = out[..., 3] * (1 - a) + a
        a = fill
        out[..., :3] = out[..., :3] * (1 - a[..., None]) + grad * a[..., None]
        out[..., 3] = out[..., 3] * (1 - a) + a
    elif style == "plain":
        out[..., :3] = fill[..., None]
        out[..., 3] = fill
    # premultiply
    out[..., :3] *= 1.0  # already composited premultiplied (colors blended against alpha)
    return out


def draw_text(canvas, text, x, y, size=200, style="gold", scale=1.0, rot=0.0, alpha=1.0,
              fontname="Anton-Regular.ttf", spacing=0.0, anchor=(0.5, 0.5), add=False, tint=None):
    tex = text_tex(text, size, style, fontname, spacing)
    h, w = tex.shape[:2]
    M = T(x, y) @ R(rot) @ S(scale) @ T(-w * anchor[0], -h * anchor[1])
    blit(canvas, tex, M, alpha=alpha, pixel=False, add=add, tint=tint)
    return w * scale, h * scale


def title_slam(canvas, text, x, y, t, size=200, style="gold", dur=0.35, rot=0.0, hold=99, out_dur=0.25,
               spacing=0.0, fontname="Anton-Regular.ttf"):
    """Big title that slams in (scale 2.6->1 with overshoot) and optionally flies out."""
    if t < 0:
        return 0.0
    k = back_out(t / dur, 2.2) if t < dur else 1.0
    sc = lerp(2.6, 1.0, clamp(k, 0, 1.2)) if t < dur else 1.0
    a = clamp(t / (dur * 0.5))
    if t > hold:
        o = clamp((t - hold) / out_dur)
        sc *= 1 + o * 0.6
        a *= 1 - o
    shake = (1 - clamp(t / 0.5)) * 18
    dx, dy = noise1(t * 40, 5) * shake, noise1(t * 40, 6) * shake
    draw_text(canvas, text, x + dx, y + dy, size, style, sc, rot, a, fontname=fontname, spacing=spacing)
    return a


# -------------------------------------------------------------- particles ---
def burst(canvas, cam, t, t0, n, x, y, seed, speed=(40, 140), life=(0.5, 1.2), size=(1.5, 3.5),
          colors=None, gravity=0.0, spread=360, direction=0, drag=0.0, add=True, alpha=1.0, spin=True,
          upward=0.0):
    """Deterministic square-particle burst in world space (texel units)."""
    dt_all = t - t0
    if dt_all < 0:
        return
    colors = colors if colors is not None else [hexc("#e27bff"), hexc("#b44bff"), hexc("#ffd1ff")]
    for i in range(n):
        lf = lerp(life[0], life[1], hash01(seed, i, 1))
        dt = dt_all
        if dt > lf:
            continue
        ang = math.radians(direction + (hash01(seed, i, 2) - 0.5) * spread)
        sp = lerp(speed[0], speed[1], hash01(seed, i, 3))
        if drag > 0:
            dist = sp * (1 - math.exp(-drag * dt)) / drag
        else:
            dist = sp * dt
        px = x + math.cos(ang) * dist
        py = y + math.sin(ang) * dist + 0.5 * gravity * dt * dt - upward * dt
        sz = lerp(size[0], size[1], hash01(seed, i, 4)) * (1 - 0.5 * dt / lf)
        col = colors[int(hash01(seed, i, 5) * len(colors)) % len(colors)]
        a = alpha * (1 - (dt / lf) ** 2)
        sx, sy = cam.to_screen(px, py)
        r = sz * cam.s / 2
        if -r < sx < W + r and -r < sy < H + r:
            if spin:
                rot = hash01(seed, i, 6) * 360 + dt * 400 * (hash01(seed, i, 7) - 0.5)
                c, s_ = math.cos(math.radians(rot)) * r, math.sin(math.radians(rot)) * r
                pts = [(sx - c + s_, sy - s_ - c), (sx + c + s_, sy + s_ - c),
                       (sx + c - s_, sy + s_ + c), (sx - c - s_, sy - s_ + c)]
                poly(canvas, pts, col, a, add=add)
            else:
                rect(canvas, sx - r, sy - r, sx + r, sy + r, col, a, add=add)


def ambient_particles(canvas, cam, t, n=40, seed=7, depth=1.0, area=(-400, -300, 400, 60),
                      colors=None, size=(1.2, 2.6), rise=6.0, alpha=0.85):
    """Floating End particles drifting upward, deterministic and looping."""
    colors = colors or [hexc("#d98bff"), hexc("#f4b8ff"), hexc("#9c4dff")]
    lc = cam.layer(depth)
    x0, y0, x1, y1 = area
    for i in range(n):
        px = lerp(x0, x1, hash01(seed, i, 1)) + math.sin(t * 0.6 + i) * 4
        span = y1 - y0
        py = y1 - ((hash01(seed, i, 2) * span + t * rise * lerp(0.6, 1.4, hash01(seed, i, 3))) % span)
        sz = lerp(size[0], size[1], hash01(seed, i, 4))
        tw = 0.6 + 0.4 * math.sin(t * 3 + i * 1.7)
        sx, sy = lc.to_screen(px, py)
        r = sz * lc.s / 2
        if -r < sx < W + r and -r < sy < H + r:
            col = colors[i % len(colors)]
            rect(canvas, sx - r, sy - r, sx + r, sy + r, col, alpha * tw, add=False)


# ---------------------------------------------------------------- post FX ---
_VIG = None
_GRAIN = None


def vignette_mask():
    global _VIG
    if _VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
        _VIG = np.clip(1 - (d ** 2.4) * 0.75, 0.15, 1)[..., None]
    return _VIG


def grain_tile(i):
    global _GRAIN
    if _GRAIN is None:
        rng = np.random.default_rng(3)
        _GRAIN = [cv2.resize(rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H),
                             interpolation=cv2.INTER_LINEAR)[..., None] for _ in range(6)]
    return _GRAIN[i % 6]


def bloom(img, thresh=0.72, strength=0.55, radius=28):
    small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    bright = np.clip(small - thresh, 0, None)
    b1 = cv2.GaussianBlur(bright, (0, 0), radius / 4)
    b2 = cv2.GaussianBlur(bright, (0, 0), radius / 1.5)
    b = cv2.resize(b1 * 0.6 + b2 * 0.6, (W, H), interpolation=cv2.INTER_LINEAR)
    return img + b * strength


def chroma(img, amt):
    if amt <= 0.2:
        return img
    a = int(round(amt))
    out = img.copy()
    M1 = np.array([[1 + amt / W, 0, -amt / 2], [0, 1 + amt / H, -amt / 2]], np.float32)
    M2 = np.array([[1 - amt / W, 0, amt / 2], [0, 1 - amt / H, amt / 2]], np.float32)
    out[..., 0] = cv2.warpAffine(np.ascontiguousarray(img[..., 0]), M1, (W, H), borderMode=cv2.BORDER_REFLECT)
    out[..., 2] = cv2.warpAffine(np.ascontiguousarray(img[..., 2]), M2, (W, H), borderMode=cv2.BORDER_REFLECT)
    _ = a
    return out


def roll_zoom(img, roll=0.0, zoom=1.0, dx=0.0, dy=0.0):
    if abs(roll) < 0.01 and abs(zoom - 1) < 1e-4 and dx == 0 and dy == 0:
        return img
    ang = abs(math.radians(roll))
    cover = math.cos(ang) + (H / W) * math.sin(ang) if W < H else 1
    cover = max(1.0, math.cos(ang) + (H / W) * math.sin(ang))
    M = cv2.getRotationMatrix2D((W / 2, H / 2), roll, zoom * cover)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def keystone(img, amt):
    """Fake low-angle: amt>0 converges verticals toward the top (looking up); <0 looking down."""
    if abs(amt) < 1e-3:
        return img
    d = W * abs(amt)
    full = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    if amt > 0:
        src = np.float32([[-d, 0], [W + d, 0], [W, H], [0, H]])
    else:
        src = np.float32([[0, 0], [W, 0], [W + d, H], [-d, H]])
    Mp = cv2.getPerspectiveTransform(src, full)
    return cv2.warpPerspective(img, Mp, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def motion_blur(img, dx, dy):
    """Directional blur along (dx, dy) screen px, computed at half res."""
    L = math.hypot(dx, dy) / 2
    if L < 1.5:
        return img
    L = min(L, 90)
    n = int(L) | 1
    k = np.zeros((n, n), np.float32)
    c = n // 2
    ux, uy = dx / (2 * L) * c, dy / (2 * L) * c
    cv2.line(k, (int(round(c - ux * 2)), int(round(c - uy * 2))), (int(round(c + ux * 2)), int(round(c + uy * 2))), 1.0, 1)
    k /= max(k.sum(), 1e-6)
    small = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    b = cv2.filter2D(small, -1, k, borderType=cv2.BORDER_REFLECT)
    return cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR)


def zoom_blur(img, amt, cx=W / 2, cy=H / 2, n=6):
    if amt < 0.005:
        return img
    small = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    acc = small.copy()
    for i in range(1, n):
        z = 1 + amt * i / n
        M = cv2.getRotationMatrix2D((cx / 2, cy / 2), 0, z)
        acc += cv2.warpAffine(small, M, (W // 2, H // 2), borderMode=cv2.BORDER_REFLECT)
    acc /= n
    return cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR)


def speed_lines(canvas, t, strength=1.0, seed=3, cx=W / 2, cy=H / 2, color=(1, 1, 1)):
    """Anime-style radial speed lines."""
    if strength <= 0:
        return
    n = 70
    for i in range(n):
        if hash01(seed, i, int(t * 24)) > 0.55:
            continue
        ang = hash01(seed, i) * math.tau
        r0 = lerp(0.38, 0.62, hash01(seed, i, 2, int(t * 24))) * H
        r1 = H * 1.3
        w0 = lerp(2, 10, hash01(seed, i, 3))
        ca, sa = math.cos(ang), math.sin(ang)
        p0 = (cx + ca * r0, cy + sa * r0)
        p1 = (cx + ca * r1 - sa * w0 * 3, cy + sa * r1 + ca * w0 * 3)
        p2 = (cx + ca * r1 + sa * w0 * 3, cy + sa * r1 - ca * w0 * 3)
        poly(canvas, [p0, p1, p2], color, 0.35 * strength)


def glitch(img, t, amt, seed=11):
    """Horizontal slice displacement + RGB split."""
    if amt <= 0:
        return img
    out = img.copy()
    k = int(t * 30)
    nslices = int(6 + 18 * amt)
    for i in range(nslices):
        y0 = int(hash01(seed, k, i, 1) * H)
        hh = int(lerp(8, 120, hash01(seed, k, i, 2)) * amt) + 4
        sh = int((hash01(seed, k, i, 3) - 0.5) * 260 * amt)
        y1 = min(H, y0 + hh)
        out[y0:y1] = np.roll(img[y0:y1], sh, axis=1)
        if hash01(seed, k, i, 4) > 0.6:
            out[y0:y1, :, 0] = np.roll(img[y0:y1, :, 0], sh + int(30 * amt), axis=1)
            out[y0:y1, :, 2] = np.roll(img[y0:y1, :, 2], sh - int(30 * amt), axis=1)
    return chroma(out, 14 * amt)


def grade(img, lift=(0.035, 0.0, 0.06), gain=(1.03, 1.0, 1.06), contrast=1.08, sat=1.12):
    out = img * np.asarray(gain, np.float32) + np.asarray(lift, np.float32) * (1 - np.clip(img, 0, 1))
    out = (out - 0.5) * contrast + 0.5
    lum = out @ np.array([0.299, 0.587, 0.114], np.float32)
    out = lum[..., None] + (out - lum[..., None]) * sat
    return out


def post(img, fr, fx):
    """Global post chain. fx keys: bloom, chroma, glitch, roll, zoom, key, flash, fade, letterbox,
    grade(dict), desat, mblur(dx,dy), zblur, vhs."""
    t = fr / FPS
    if fx.get("key"):
        img = keystone(img, fx["key"])
    if fx.get("roll") or fx.get("zoom", 1.0) != 1.0 or fx.get("dx") or fx.get("dy"):
        img = roll_zoom(img, fx.get("roll", 0.0), fx.get("zoom", 1.0), fx.get("dx", 0.0), fx.get("dy", 0.0))
    if fx.get("mblur"):
        img = motion_blur(img, *fx["mblur"])
    if fx.get("zblur"):
        img = zoom_blur(img, fx["zblur"], *fx.get("zcenter", (W / 2, H / 2)))
    img = bloom(img, fx.get("bthresh", 0.72), fx.get("bloom", 0.55))
    img = grade(img, **fx.get("grade", {}))
    if fx.get("desat"):
        lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
        d = fx["desat"]
        tintc = np.asarray(fx.get("desat_tint", (1.0, 0.95, 1.08)), np.float32)
        img = img * (1 - d) + (lum[..., None] * tintc) * d
    if fx.get("chroma"):
        img = chroma(img, fx["chroma"])
    if fx.get("glitch"):
        img = glitch(img, t, fx["glitch"])
    if fx.get("vhs"):
        v = fx["vhs"]
        lines = (np.arange(H) % 4 < 2).astype(np.float32)[:, None, None]
        img = img * (1 - 0.18 * v * lines)
        img = chroma(img, 6 * v)
        roll_y = int((t * 300) % H)
        img[roll_y:roll_y + 26] += 0.08 * v
    img = img * vignette_mask()
    img = img + grain_tile(fr) * fx.get("grain", 0.011)
    if fx.get("flash"):
        img = img + (1 - img) * clamp(fx["flash"])
    if fx.get("fade"):
        img = img * (1 - clamp(fx["fade"]))
    lb = fx.get("letterbox", 0.0)
    if lb > 0:
        bh = int(H * 0.11 * lb)
        img[:bh] = 0
        img[H - bh:] = 0
    return np.clip(img, 0, 1)
