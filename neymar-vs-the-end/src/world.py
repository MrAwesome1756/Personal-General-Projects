"""Environment: sky, nebula, stars, obsidian pillars + End crystals, island, hoop, portal."""
import math

import cv2
import numpy as np

from core import (BEAT, Cam, H, R, S, T, W, blit, clamp, disc, hash01, hexc, lerp, line, mat3,
                  poly, rect, tex_from_rows)

RNG = np.random.default_rng(42)


def blit_axis(canvas, tex, cam, wx0, wy0, sc=1.0, alpha=1.0, tint=None):
    """Axis-aligned nearest-neighbour blit of a (possibly huge) texel texture whose
    top-left is at world (wx0, wy0); crops to the visible area first."""
    th, tw = tex.shape[:2]
    s = cam.s * sc
    sx0, sy0 = cam.to_screen(wx0, wy0)
    # visible tex range
    u0 = max(0.0, (0 - sx0) / s)
    v0 = max(0.0, (0 - sy0) / s)
    u1 = min(tw, (W - sx0) / s)
    v1 = min(th, (H - sy0) / s)
    if u1 <= u0 or v1 <= v0:
        return
    iu0, iv0 = int(math.floor(u0)), int(math.floor(v0))
    iu1, iv1 = int(math.ceil(u1)), int(math.ceil(v1))
    crop = tex[iv0:iv1, iu0:iu1]
    X0 = sx0 + iu0 * s
    Y0 = sy0 + iv0 * s
    ow, oh = int(round((iu1 - iu0) * s)), int(round((iv1 - iv0) * s))
    if ow <= 0 or oh <= 0:
        return
    big = cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_NEAREST)
    xa, ya = int(round(X0)), int(round(Y0))
    xb, yb = xa + ow, ya + oh
    cx0, cy0 = max(0, xa), max(0, ya)
    cx1, cy1 = min(W, xb), min(H, yb)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    src = big[cy0 - ya:cy1 - ya, cx0 - xa:cx1 - xa]
    a = src[..., 3:4] * alpha
    rgb = src[..., :3] * alpha
    if tint is not None:
        rgb = rgb * np.asarray(tint, np.float32)
    reg = canvas[cy0:cy1, cx0:cx1]
    if canvas.shape[2] == 4:
        reg[..., :3] = reg[..., :3] * (1 - a) + rgb
        reg[..., 3:4] = reg[..., 3:4] * (1 - a) + a
    else:
        reg[:] = reg[:] * (1 - a) + rgb


# -------------------------------------------------------------------- sky ---
def _make_sky():
    g = np.zeros((H, W, 3), np.float32)
    top, bot = hexc("#07030f"), hexc("#1c0b33")
    k = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    g[:] = top * (1 - k) + bot * k
    return g


def _make_nebula():
    w, h = 540, 960
    acc = np.zeros((h, w, 3), np.float32)
    for oc, (sc, amp) in enumerate(((0.012, 1.0), (0.03, 0.5), (0.07, 0.25))):
        n = RNG.random((int(h * sc) + 2, int(w * sc) + 2)).astype(np.float32)
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
        acc[..., 0] += n * amp
        acc[..., 1] += np.roll(n, 40 + oc * 13, axis=1) * amp
    m = acc[..., 0] / 1.75
    m2 = acc[..., 1] / 1.75
    m = np.clip((m - 0.45) * 2.2, 0, 1) ** 1.6
    m2 = np.clip((m2 - 0.5) * 2.4, 0, 1) ** 1.8
    neb = m[..., None] * hexc("#5a1a8a") * 0.55 + m2[..., None] * hexc("#b3288f") * 0.3
    neb = cv2.GaussianBlur(neb, (0, 0), 6)
    return cv2.resize(neb, (W * 2, H * 2), interpolation=cv2.INTER_CUBIC)


def _make_stars(n=1400, w=W * 2, h=H * 2):
    st = np.zeros((h, w, 3), np.float32)
    for i in range(n):
        x, y = int(RNG.random() * (w - 4)), int(RNG.random() * (h - 4))
        r = RNG.random()
        sz = 1 if r < 0.7 else (2 if r < 0.93 else 3)
        c = hexc("#ffffff") if RNG.random() < 0.7 else (hexc("#e6b8ff") if RNG.random() < 0.6 else hexc("#b8d8ff"))
        br = lerp(0.35, 1.0, RNG.random() ** 2)
        st[y:y + sz * 2, x:x + sz * 2] = c * br
    return st


SKY = _make_sky()
NEB = _make_nebula()
STARS = _make_stars()
STARS2 = _make_stars(500)


def draw_sky(canvas, cam, t, nebula=1.0, stars=1.0, brightness=1.0, extra_glow=0.0):
    canvas[:] = SKY * brightness
    ox = int((cam.cx * 0.25 * cam.s / 6) % (W))
    oy = int(clamp((cam.cy + 40) * 0.25 * cam.s / 6 + H / 2, 0, H))
    if nebula > 0:
        canvas += NEB[oy:oy + H, ox:ox + W] * nebula * brightness
    if stars > 0:
        ox2 = int((cam.cx * 0.4 * cam.s / 6) % W)
        oy2 = int(clamp((cam.cy + 40) * 0.4 * cam.s / 6 + H / 2, 0, H))
        tw = 0.85 + 0.15 * math.sin(t * 2.3)
        canvas += STARS[oy:oy + H, ox:ox + W] * (stars * tw * brightness)
        canvas += STARS2[oy2:oy2 + H, ox2:ox2 + W] * (stars * (1.7 - tw) * 0.8 * brightness)
    if extra_glow:
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        canvas += (yy ** 2) * hexc("#6a1fa8") * extra_glow


# ---------------------------------------------------------------- pillars ---
OBP = {"o": hexc("#0f0a16"), "O": hexc("#1a1026"), "q": hexc("#2a1a3d"), "Q": hexc("#43275e")}


def _obsidian(w, h, seed):
    rng = np.random.default_rng(seed)
    v = rng.random((h, w))
    rows = []
    for y in range(h):
        r = ""
        for x in range(w):
            vv = v[y, x]
            c = "Q" if vv < 0.03 else "q" if vv < 0.12 else "O" if vv < 0.45 else "o"
            if x == 0 or x == w - 1:
                c = "O" if c == "o" else c
            if y % 16 == 0 and vv < 0.5:
                c = "O"
            r += c
        rows.append(r)
    t = tex_from_rows(rows, OBP)
    # side shading: left face lighter (3/4 look)
    face = int(w * 0.35)
    t[:, :face, :3] *= 1.35
    t[:, face:face + 1, :3] *= 0.6
    return t


PILLARS = {  # layer depth -> list of (x, top_y, w, h, tex)
}


def _init_pillars():
    spec = {
        0.45: [(-610, -170, 30), (-470, -225, 34), (-330, -150, 28), (-205, -240, 36), (-60, -185, 30),
               (70, -230, 34), (205, -160, 28), (330, -215, 32), (470, -175, 30), (600, -235, 34)],
        0.7: [(-520, -120, 34), (-330, -95, 30), (-150, -135, 36), (175, -110, 32), (345, -140, 36), (540, -100, 30)],
    }
    for d, lst in spec.items():
        out = []
        for i, (x, top, w) in enumerate(lst):
            h = int(-top + 260)
            out.append((x, top, w, h, _obsidian(w, h, int(d * 100) + i)))
        PILLARS[d] = out


_init_pillars()

BEDROCK = tex_from_rows(["".join("kKk"[(x * 7 + y * 3) % 3] for x in range(16)) for y in range(10)],
                        {"k": hexc("#3a3a40"), "K": hexc("#55555e")})


def pillar_tops(depth):
    return [(x + w / 2, top) for (x, top, w, h, _) in PILLARS[depth]]


def draw_crystal(canvas, glow, cam, x, y, t, size=1.0, seed=0, flare=0.0):
    """End crystal: rotating magenta core inside rotating glass frame, on a bedrock base."""
    blit_axis(canvas, BEDROCK, cam, x - 8 * size, y - 10 * size, size)
    bobv = math.sin(t * 2.2 + seed) * 4
    cx, cy = cam.to_screen(x, y - 26 * size + bobv * size)
    r = 7 * size * cam.s
    for k, (rr, col, spd, al) in enumerate(((1.3, hexc("#f6e0ff"), 1.6, 0.55), (1.0, hexc("#ff4fd8"), -2.3, 1.0))):
        ang = t * spd + seed + k
        pts = []
        for j in range(4):
            a = ang + j * math.pi / 2
            pts.append((cx + math.cos(a) * r * rr, cy + math.sin(a) * r * rr * 0.9))
        if k == 0:
            for j in range(4):
                line(canvas, pts[j], pts[(j + 1) % 4], col, max(2, cam.s * size * 0.9), al)
        else:
            poly(canvas, pts, col, al)
            poly(canvas, [(cx, cy - r * 0.45), (cx + r * 0.4, cy), (cx, cy + r * 0.45), (cx - r * 0.4, cy)],
                 hexc("#ffd6f6"), 0.9)
    if glow is not None:
        disc(glow, cx, cy, r * 1.2, hexc("#ff3fd0"), 0.5 + 0.8 * flare, add=True, soft=r * (4 + 6 * flare))
    return (x, y - 26 * size + bobv * size)


def draw_pillars(canvas, glow, cam, t, depth, crystals=True, flare=0.0, tint=1.0):
    lc = cam.layer(depth)
    tops = []
    fog = hexc("#2a1245")
    for (x, top, w, h, tex) in PILLARS[depth]:
        blit_axis(canvas, tex, lc, x, top, 1.0, tint=(tint, tint, tint))
        if crystals:
            tops.append((lc, draw_crystal(canvas, glow, lc, x + w / 2, top, t, 1.0, seed=x, flare=flare)))
    # atmospheric haze over far layer
    if depth < 0.6:
        canvas[:] = canvas * 0.82 + fog * 0.18 * (canvas.mean() * 0 + 1)
    return tops


# ----------------------------------------------------------------- island ---
ISP = {"a": hexc("#e6e3ac"), "b": hexc("#d6d298"), "c": hexc("#c3be82"), "d": hexc("#aca76c"),
       "e": hexc("#948f58"), "f": hexc("#7a7548"), "g": hexc("#5e5a38"), "x": hexc("#f3f1c8"),
       ".": None}
ISL_X0, ISL_Y0 = -560, -2


def _make_island():
    w, h = 1120, 320
    rng = np.random.default_rng(7)
    tex = np.zeros((h, w, 4), np.float32)
    band_cols = ["x", "a", "a", "b", "b", "c"]
    for y in range(h):
        for_row = []
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    wx = xs + ISL_X0
    # silhouette: flat top at y=2 (world 0), cliffs taper to a point ~y=300
    half = 540.0
    edge = half * np.clip(1 - np.clip((ys - 30) / 280.0, 0, 1) ** 1.3, 0, 1)
    edge = edge + (rng.random((h, 1)) * 18 - 9) * (ys > 30)
    inside = (np.abs(wx) <= np.where(ys < 30, half, edge)) & (ys >= 2)
    # step the silhouette into block-ish chunks
    inside &= ~((ys > 30) & (np.abs(wx) > edge - (((ys // 8) * 37 % 11) - 5)))
    noise = rng.random((h, w))
    shade = np.zeros((h, w), np.int32)
    # top surface bands (0..30), then cliff darkening with depth
    band = np.where(ys < 30, np.clip((ys - 2) // 5, 0, 5), 0)
    depthk = np.clip((ys - 30) / 260.0, 0, 1)
    cliff_idx = 2 + (depthk * 4.5).astype(np.int32)
    idx = np.where(ys < 30, band.repeat(w, 1) if band.shape[1] == 1 else band, cliff_idx)
    idx = idx + (noise < 0.18).astype(np.int32) - (noise > 0.93).astype(np.int32)
    names = ["x", "a", "b", "c", "d", "e", "f", "g"]
    idx = np.clip(idx, 0, 7)
    cols = np.stack([ISP[n] for n in names])
    tex[..., :3] = cols[idx]
    tex[..., 3] = inside.astype(np.float32)
    tex[..., :3] *= tex[..., 3:4]
    # purple rim light on underside
    rim = np.clip(1 - np.abs(np.abs(wx) - edge) / 14.0, 0, 1) * (ys > 40)
    tex[..., :3] += (rim[..., None] * hexc("#7a3fbf") * 0.35) * tex[..., 3:4]
    _ = shade
    return tex


ISLAND = _make_island()


def draw_island(canvas, cam, tint=1.0):
    blit_axis(canvas, ISLAND, cam, ISL_X0, ISL_Y0, 1.0, tint=(tint, tint, tint))


def draw_shadow(canvas, cam, x, w=10, alpha=0.35, y=0.5):
    sx, sy = cam.to_screen(x, y)
    rx = w * cam.s
    ry = max(2, 1.6 * cam.s)
    yy0, yy1 = int(sy - ry * 2), int(sy + ry * 2)
    xx0, xx1 = int(sx - rx * 1.4), int(sx + rx * 1.4)
    xx0, yy0 = max(0, xx0), max(0, yy0)
    xx1, yy1 = min(W, xx1), min(H, yy1)
    if xx1 <= xx0 or yy1 <= yy0:
        return
    yy, xx = np.mgrid[yy0:yy1, xx0:xx1].astype(np.float32)
    d = ((xx - sx) / rx) ** 2 + ((yy - sy) / ry) ** 2
    a = np.clip(1 - d, 0, 1)[..., None] * alpha
    canvas[yy0:yy1, xx0:xx1] *= (1 - a)


# ------------------------------------------------------------------- hoop ---
QP = {"q": hexc("#efece6"), "Q": hexc("#cfc9c0"), "z": hexc("#a8a198"), "r": hexc("#d2261c"),
      "o": hexc("#ff6a14"), "O": hexc("#b8420a"), "n": hexc("#f5f5f5"), ".": None}


def _quartz(w, h):
    rows = []
    for y in range(h):
        r = ""
        for x in range(w):
            c = "Q" if x == 0 else ("z" if x == w - 1 else "q")
            if y % 10 == 0:
                c = "Q"
            r += c
        rows.append(r)
    return tex_from_rows(rows, QP)


HOOP_POST = _quartz(6, 96)
HOOP_ARM = _quartz(20, 4)
HOOP_BOARD = tex_from_rows(["zqqz"] + ["rqqr"] * 3 + ["qqqq"] * 22 + ["rqqr"] * 3 + ["zqqz"], QP)
HOOP_RIM = tex_from_rows(["oooooooooooooo", "OOOOOOOOOOOOOO"], QP)
NET_POINTS = None


def hoop_geom(hx):
    """Key world points for a hoop whose post stands at x=hx (facing left)."""
    return {"rim_l": (hx - 36, -62), "rim_r": (hx - 22, -62), "rim_c": (hx - 29, -62),
            "board_x": hx - 22, "board_y0": -90, "board_y1": -60}


def draw_hoop(canvas, cam, hx, t, bend=0.0, board=True, swish=0.0, alpha=1.0, net_t=None):
    C = mat3(cam.M())
    blit(canvas, HOOP_POST, C @ T(hx, -96), alpha)
    blit(canvas, HOOP_ARM, C @ T(hx - 19, -80), alpha)
    if board:
        blit(canvas, HOOP_BOARD, C @ T(hx - 22, -90), alpha)
    # rim pivots at the board
    m = C @ T(hx - 22, -62) @ R(-bend * 38) @ T(-14, -1)
    blit(canvas, HOOP_RIM, m, alpha)
    # net: strands from rim down, swaying
    rim_pts = [(m @ np.array([14 * i / 5, 2, 1.0]))[:2] for i in range(6)]
    sway = math.sin(t * 9) * swish * 3
    lw = max(1.5, cam.s * 0.55)
    bottom = []
    for i, p in enumerate(rim_pts):
        f = i / 5
        bx = lerp(rim_pts[0][0], rim_pts[-1][0], 0.2 + 0.6 * f) + sway * cam.s * (0.5 + f)
        by = p[1] + (13 - swish * 2) * cam.s
        bottom.append((bx, by))
        line(canvas, p, (bx, by), QP["n"], lw, 0.9 * alpha)
    for i in range(5):
        line(canvas, rim_pts[i], bottom[i + 1], QP["n"], lw * 0.8, 0.7 * alpha)
        line(canvas, rim_pts[i + 1], bottom[i], QP["n"], lw * 0.8, 0.7 * alpha)
    for j in (0.45, 0.8):
        pa = [(lerp(rim_pts[i][0], bottom[i][0], j), lerp(rim_pts[i][1], bottom[i][1], j)) for i in range(6)]
        for i in range(5):
            line(canvas, pa[i], pa[i + 1], QP["n"], lw * 0.8, 0.7 * alpha)


# ----------------------------------------------------------------- portal ---
PFP = {"a": hexc("#dfe4b3"), "b": hexc("#c7cd98"), "c": hexc("#aeb57f"), "g": hexc("#2f6b58"),
       "G": hexc("#1d4a3c"), "k": hexc("#0d1f1a")}


def _frame_block(seed):
    rng = np.random.default_rng(seed)
    rows = []
    for y in range(16):
        r = ""
        for x in range(16):
            v = rng.random()
            if y < 4:
                c = "g" if v > 0.25 else "G"
            else:
                c = "a" if v < 0.55 else ("b" if v < 0.9 else "c")
            if x in (0, 15) or y == 15:
                c = "c" if y >= 4 else "G"
            r += c
        rows.append(r)
    return tex_from_rows(rows, PFP)


FRAME = [_frame_block(i) for i in range(4)]
EYEP = {"g": hexc("#1d8f6e"), "G": hexc("#136048"), "l": hexc("#6fe0b8"), "p": hexc("#08241b"), ".": None}
EYE = tex_from_rows([
    "..GggG..",
    ".GglggG.",
    "GglpplgG",
    "GgppppgG",
    "GgppppgG",
    "GgllllgG",
    ".GggggG.",
    "..GGGG..",
], EYEP)

PORTAL_X = -230          # centre x of the portal
PORTAL_BLOCKS = 4, 5     # outer size in blocks (w, h)


def portal_rect():
    bw, bh = PORTAL_BLOCKS
    x0 = PORTAL_X - bw * 8
    y0 = -bh * 16
    return x0, y0, x0 + bw * 16, 0


def portal_inner(t, w=32, h=48, sub=4, energy=1.0):
    """Animated End-portal surface texture (premultiplied RGBA)."""
    W_, H_ = w * sub, h * sub
    img = np.zeros((H_, W_, 4), np.float32)
    img[..., :3] = hexc("#050a0e")
    img[..., 3] = 1.0
    cols = [hexc("#2fe0c0"), hexc("#b77bff"), hexc("#5a8bff"), hexc("#e0fff6"), hexc("#ff78e0")]
    for layer in range(4):
        spd = 6 + layer * 5
        for i in range(60):
            px = (hash01(i, layer, 1) * W_ + t * spd * (1 + layer * 0.3) * sub) % W_
            py = (hash01(i, layer, 2) * H_ + t * spd * 0.6 * sub * (1 if layer % 2 else -1)) % H_
            sz = int(sub * lerp(0.6, 1.6, hash01(i, layer, 3)))
            c = cols[int(hash01(i, layer, 4) * 5)]
            a = energy * (0.4 + 0.6 * hash01(i, layer, 5)) * (0.6 + 0.4 * math.sin(t * 4 + i))
            x0, y0 = int(px), int(py)
            img[y0:y0 + sz, x0:x0 + sz, :3] = img[y0:y0 + sz, x0:x0 + sz, :3] * (1 - a) + c * a
    # swirl haze
    yy, xx = np.mgrid[0:H_, 0:W_].astype(np.float32)
    sw = 0.5 + 0.5 * np.sin((xx / W_ * 6 + yy / H_ * 3) + t * 3)
    img[..., :3] += sw[..., None] * hexc("#0d3a3a") * 0.5 * energy
    return img


def draw_portal(canvas, glow, cam, t, eyes=12, energy=0.0, open_k=1.0):
    """Vertical End portal frame. eyes: number of eyes inserted (0..12, fractional = popping in)."""
    bw, bh = PORTAL_BLOCKS
    x0, y0, x1, y1 = portal_rect()
    C = mat3(cam.M())
    idx = 0
    ring = []
    for by in range(bh):
        for bx in range(bw):
            if 0 < bx < bw - 1 and 0 < by < bh - 1:
                continue
            ring.append((bx, by))
    for i, (bx, by) in enumerate(ring):
        m = C @ T(x0 + bx * 16, y0 + by * 16)
        blit(canvas, FRAME[i % 4], m)
    # eye slots on inner-facing ring blocks
    slots = [(bx, by) for (bx, by) in ring]
    for i, (bx, by) in enumerate(slots):
        k = clamp(eyes - i)
        if k <= 0:
            continue
        s = 1.0 + 0.6 * (1 - k) ** 2
        m = C @ T(x0 + bx * 16 + 8, y0 + by * 16 + 9) @ S(s) @ T(-4, -4)
        blit(canvas, EYE, m, alpha=clamp(k * 3))
        if glow is not None:
            sx, sy = cam.to_screen(x0 + bx * 16 + 8, y0 + by * 16 + 9)
            disc(glow, sx, sy, cam.s * 3, hexc("#3cffc0"), 0.35 * k + 1.2 * (1 - k), add=True, soft=cam.s * 8)
    if energy > 0:
        inner = portal_inner(t, energy=energy)
        sub = 4
        ih = int(48 * open_k)
        if ih > 0:
            m = C @ T(x0 + 16, y0 + 16 + (48 - ih) / 2) @ S(1 / sub, (ih / 48) / sub)
            blit(canvas, inner, m, alpha=clamp(energy), pixel=False)
        if glow is not None:
            sx, sy = cam.to_screen(PORTAL_X, y0 + 40)
            disc(glow, sx, sy, cam.s * 20, hexc("#2de0c0"), 0.6 * energy, add=True, soft=cam.s * 40)


# --------------------------------------------------------------- env pack ---
def draw_env(canvas, glow, cam, t, nebula=1.0, pillars=True, flare=0.0, island=True, sky_bright=1.0,
             crystals=True):
    draw_sky(canvas, cam, t, nebula=nebula, brightness=sky_bright)
    tops = []
    if pillars:
        tops += draw_pillars(canvas, glow, cam, t, 0.45, crystals, flare, 0.75)
        tops += draw_pillars(canvas, glow, cam, t, 0.7, crystals, flare, 0.95)
    if island:
        draw_island(canvas, cam)
    return tops
