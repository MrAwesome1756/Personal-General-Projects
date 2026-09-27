"""The 60-second cut: shot list, choreography, cameras and transitions.

Storyline - "NEYMAR JR vs THE END" (boss-fight arc)
  ACT 1  Arrival   0-14s   sky tilt-down reveal, portal ignites, Neymar arrives, title
  ACT 2  Guard    14-27s   eyes in the dark, dribble, Enderman teleports in, ANKLES BROKEN + replay
  ACT 3  Boss     27-47s   ground shakes, dragon flyover + landing, sprint up the dragon, slow-mo air,
                           POSTERIZED + replays
  ACT 4  Victory  47-60s   dragon defeated, Endermen crowd, dragon-egg trophy poster
"""
import math

import cv2
import numpy as np

import world
from core import (BEAT, FPS, H, W, Cam, R, S, T, back_out, blit, burst, clamp, comp, disc,
                  draw_text, ease_in, ease_io, ease_out, glow_from, hash01, hexc, inv, keys, lerp,
                  line, mat3, new_layer, noise1, poly, rect, smooth, solid, speed_lines, text_tex,
                  title_slam, ambient_particles)
from rigs import (DRA, END, END_FRONT, END_HIP, END_SIDE, NEY, NEY_FRONT, NEY_HIP, NEY_SIDE,
                  draw_ball, draw_dragon, end_idle, ney_idle, ney_run, point, tex_from_rows)
from world import (draw_env, draw_hoop, draw_island, draw_pillars, draw_portal, draw_shadow,
                   draw_sky, hoop_geom, pillar_tops)

# ------------------------------------------------------------------ layout ---
world.PORTAL_X = -150
PORTAL_X = -150
END_X = -55
HOOP_X = 150
RIM = hoop_geom(HOOP_X)["rim_c"]      # (121, -62)
DRAGON_X = 86
DRAGON_SC = 1.8
DRAGON_Y = -27.5 * DRAGON_SC          # body centre when standing

PURPLE = hexc("#b040ff")
GOLD = hexc("#ffc53a")
TEAL = hexc("#39f0c8")

EGG = tex_from_rows([
    "...kkkk...",
    "..kKkkKk..",
    ".kkkpkkkk.",
    ".kKkkkkpk.",
    "kkkkKkkkkk",
    "kpkkkkKkkk",
    "kkkKkkkkpk",
    "kkkkkpkkkk",
    "kKkkkkkkKk",
    ".kkpkkKkk.",
    ".kkkkkkkk.",
    "..kkkkkk..",
], {"k": hexc("#15101c"), "K": hexc("#2a1f38"), "p": hexc("#b24dff"), ".": None})


# ------------------------------------------------------------------ frame ----
class Fr:
    def __init__(self, cam, t):
        self.cam, self.t = cam, t
        self.c = solid((0, 0, 0))
        self.g = np.zeros((H, W, 3), np.float32)

    def env(self, **kw):
        return draw_env(self.c, self.g, self.cam, self.t, **kw)

    def glowcomp(self, L, color=PURPLE, radius=20, strength=1.0, alpha=1.0):
        self.c += glow_from(L, color, radius, strength) * alpha
        comp(self.c, L, alpha)

    def done(self):
        self.c += self.g
        return self.c


def ney(c, cam, x, y, pose, facing=1, head=None, scale=1.0, alpha=1.0, tint=None, front=False):
    rig = NEY_FRONT if front else NEY_SIDE
    over = {"head": NEY[head]} if head else None
    return rig.draw(c, cam, x, y, pose, facing, scale, alpha, over, tint)


def hand(mats, part="arm"):
    return point(mats[part], 2, 11.5)


def ender(L, cam, xf, pose, facing=-1, theta=0.0, scale=1.0, alpha=1.0, tint=None, front=False,
          hip_drop=0.0, yf=0.0):
    """Enderman whose feet are at (xf, yf); theta = whole-body world rotation (deg, clockwise)
    about the feet (used for falling)."""
    th = math.radians(theta)
    hh = (END_HIP - hip_drop) * scale
    hx, hy = xf + math.sin(th) * hh, yf - math.cos(th) * hh
    p = dict(pose)
    p["torso"] = p.get("torso", 0) + theta * facing
    rig = END_FRONT if front else END_SIDE
    return rig.draw(L, cam, hx, hy, p, facing, scale, alpha, None, tint)


def ender_stance(t):
    s = math.sin(t * 5)
    return {"leg": 22, "leg_far": -22, "arm": -70 + 5 * s, "arm_far": -45 - 5 * s, "torso": -8,
            "head": 4, "jaw": 0}


def ender_lying():
    return {"leg": 8, "leg_far": -12, "arm": -150, "arm_far": -110, "head": 10, "jaw": 0}


def dragon_mats(x, y, t, **kw):
    """World matrices of the dragon without drawing (tiny dummy canvas)."""
    dummy = np.zeros((2, 2, 3), np.float32)
    return draw_dragon(dummy, Cam(0, 0, 1), x, y, t, **kw)


def dragon(f, x, y, t, glow=1.0, **kw):
    L = new_layer()
    kw.setdefault("facing", -1)
    kw.setdefault("scale", DRAGON_SC)
    m = draw_dragon(L, f.cam, x, y, t, glow_layer=f.g, **kw)
    f.glowcomp(L, hexc("#9a3cff"), 26, 0.9 * glow)
    return m


def eye_rects(canvas, glow, M, openk=1.0, rows=(4, 5), spans=((0, 3), (5, 8)), bright=1.0, cam_s=6.0):
    """Glowing Enderman eyes drawn from a head's world->screen matrix."""
    y0, y1 = rows
    ym = (y0 + y1) / 2
    hh = (y1 - y0) / 2 * openk
    for (x0, x1) in spans:
        pts = [point(M, x0, ym - hh), point(M, x1, ym - hh), point(M, x1, ym + hh), point(M, x0, ym + hh)]
        poly(canvas, pts, hexc("#e27bff") * min(1.6, bright), 1.0)
        cx = sum(p[0] for p in pts) / 4
        cy = sum(p[1] for p in pts) / 4
        inner = [((p[0] - cx) * 0.4 + cx, (p[1] - cy) * 0.6 + cy) for p in pts]
        poly(canvas, inner, hexc("#ffe0ff"), 1.0)
        if glow is not None:
            r = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
            disc(glow, cx, cy, r * 0.5, hexc("#c040ff"), 0.9 * openk * bright, add=True, soft=r * 2.2)


def beams(canvas, glow, starts, target, t, k=1.0, color=hexc("#ff7df0")):
    for i, (sx, sy) in enumerate(starts):
        w = (6 + 3 * math.sin(t * 20 + i)) * k
        line(canvas, (sx, sy), target, color, w, 0.85 * k, add=True)
        line(canvas, (sx, sy), target, hexc("#ffffff"), w * 0.35, 0.9 * k, add=True)
        line(glow, (sx, sy), target, color, w * 4, 0.35 * k, add=True)


def shockwave(canvas, cx, cy, t, t0, maxr=900, dur=0.5, color=(1, 1, 1), width=40):
    k = (t - t0) / dur
    if not 0 <= k <= 1:
        return
    r = ease_out(k, 2) * maxr
    wv = width * (1 - k)
    yy0, yy1 = int(max(0, cy - r - wv)), int(min(H, cy + r + wv))
    xx0, xx1 = int(max(0, cx - r - wv)), int(min(W, cx + r + wv))
    if yy1 <= yy0 or xx1 <= xx0:
        return
    yy, xx = np.mgrid[yy0:yy1, xx0:xx1].astype(np.float32)
    d = np.abs(np.sqrt((xx - cx) ** 2 + ((yy - cy) * 1.0) ** 2) - r)
    a = np.clip(1 - d / max(wv, 1), 0, 1) ** 2 * (1 - k)
    canvas[yy0:yy1, xx0:xx1] += a[..., None] * np.asarray(color, np.float32) * 0.9


def light_rays(canvas, cx, cy, t, n=14, k=1.0, color=hexc("#ffe7a8"), length=2200, width=0.09, spin=0.25,
               seed=1, alpha=0.5):
    for i in range(n):
        a = i / n * math.tau + t * spin + hash01(seed, i) * 0.3
        wd = width * (0.6 + 0.8 * hash01(seed, i, 2))
        L = length * k * (0.7 + 0.3 * hash01(seed, i, 3))
        p1 = (cx + math.cos(a - wd) * L, cy + math.sin(a - wd) * L)
        p2 = (cx + math.cos(a + wd) * L, cy + math.sin(a + wd) * L)
        poly(canvas, [(cx, cy), p1, p2], color, alpha * k, add=True)


def dust(f, x, y, t, t0, n=26, seed=1, spread=200, direction=-90, speed=(20, 70), colors=None):
    burst(f.c, f.cam, f.t if t is None else t, t0, n, x, y, seed, speed=speed, life=(0.5, 1.1),
          size=(2, 5), colors=colors or [hexc("#e6e3ac"), hexc("#c3be82"), hexc("#f3f1c8")],
          gravity=60, spread=spread, direction=direction, drag=2.0, add=False)


def mirror(img):
    return np.ascontiguousarray(img[:, ::-1])


def label_replay(c, t, extra="REPLAY"):
    draw_text(c, extra, 70, 130, 46, "white", fontname="PressStart2P-Regular.ttf", anchor=(0, 0.5))
    if int(t * 3) % 2 == 0:
        disc(c, 45, 128, 14, hexc("#ff2a2a"), 1.0)
    draw_text(c, "x0.5", W - 70, 130, 36, "white", fontname="PressStart2P-Regular.ttf", anchor=(1, 0.5),
              alpha=0.8)


def typewriter(c, text, x, y, t, size=42, cps=16, style="white", alpha=1.0):
    n = int(clamp(t * cps, 0, len(text)))
    if n <= 0:
        return
    full = text_tex(text, size, style, "PressStart2P-Regular.ttf")
    x0 = x - full.shape[1] / 2
    draw_text(c, text[:n], x0, y, size, style, fontname="PressStart2P-Regular.ttf", anchor=(0, 0.5),
              alpha=alpha)
    if n < len(text) and int(t * 8) % 2 == 0:
        part = text_tex(text[:n], size, style, "PressStart2P-Regular.ttf")
        cx = x0 + part.shape[1] - size * 0.35
        rect(c, cx, y - size / 2, cx + size * 0.6, y + size / 2, (1, 1, 1), alpha)


# ============================================================ ACT 1 ==========
def s_open(t):
    """0-8s: text in the void, then tilt down through the pillars to the island."""
    cy = keys(t, [(0, -760), (2.6, -735), (7.7, -70)], ease_io)
    cam = Cam(-5, cy, keys(t, [(0, 3.1), (8, 3.4)]))
    f = Fr(cam, t)
    f.env(flare=0.15)
    if 4.3 < t < 7.8:  # dragon silhouette crosses the far sky (foreshadowing)
        k = inv(4.3, 7.8, t)
        lc = cam.layer(0.45)
        L = new_layer()
        draw_dragon(L, lc, lerp(-330, 330, k), lerp(-300, -250, k) + math.sin(k * 6) * 6, t * 1.3,
                    facing=1, scale=0.55)
        L[..., :3] *= 0.35
        f.glowcomp(L, hexc("#8c35ff"), 10, 0.7)
    draw_portal(f.c, f.g, cam, t, eyes=0, energy=0)
    draw_hoop(f.c, cam, HOOP_X, t)
    ambient_particles(f.c, cam, t, 60, area=(-300, -900, 300, 40), rise=10)
    c = f.done()
    a = 1 - inv(3.2, 3.8, t)
    typewriter(c, "THE END.", W / 2, H * 0.40, t - 0.4, 64, 10, alpha=a)
    typewriter(c, "NO ONE HAS EVER", W / 2, H * 0.47, t - 1.5, 40, 18, style="purple", alpha=a)
    typewriter(c, "SCORED HERE.", W / 2, H * 0.51, t - 2.2, 40, 18, style="purple", alpha=a)
    fx = {"fade": 1 - inv(0.0, 1.4, t), "letterbox": 1 - inv(6.5, 7.8, t), "grain": 0.03}
    return c, fx


def s_portal(t):
    """8-10s: eyes of ender click into the frame, portal ignites."""
    cam = Cam(PORTAL_X, -40, keys(t, [(0, 8.4), (2, 10.8)]), shake=16 * inv(1.45, 1.5, t) * (1 - inv(1.5, 2.0, t)), t=t)
    f = Fr(cam, t)
    f.env(nebula=0.7)
    eyes = clamp(t / 0.1, 0, 14)
    energy = ease_out(inv(1.5, 1.8, t))
    draw_portal(f.c, f.g, cam, t, eyes=eyes, energy=energy, open_k=ease_out(inv(1.5, 1.75, t)))
    burst(f.c, cam, t, 1.5, 60, PORTAL_X, -40, 21, speed=(30, 120), colors=[TEAL, hexc("#e0fff6"), hexc("#b77bff")])
    c = f.done()
    flash = max(0.5 * (1 - inv(0, 0.1, t)), 0.6 * (1 - inv(1.5, 1.75, t)) * (t >= 1.5))
    return c, {"flash": flash, "chroma": 8 * (1 - inv(1.5, 1.9, t)) * (t >= 1.5)}


def s_arrival(t):
    """10-14s: Neymar materialises from the portal, walks out, spins the ball; title slam."""
    x = keys(t, [(0, PORTAL_X), (0.9, PORTAL_X), (2.0, -112)])
    cam = Cam(keys(t, [(0, PORTAL_X), (0.9, PORTAL_X + 4), (2.0, -118), (4, -116)]),
              keys(t, [(0, -18), (1.2, -22), (4, -27)]), keys(t, [(0, 12.5), (1.2, 10), (4, 8.4)]))
    f = Fr(cam, t)
    f.env(nebula=0.8)
    draw_portal(f.c, f.g, cam, t + 2, eyes=14, energy=1.0)
    ambient_particles(f.c, cam, t, 40, area=(-260, -120, -40, 10), rise=8)
    # materialise: crouched landing -> stand -> walk -> spin ball
    a = ease_out(inv(0.25, 0.6, t))
    if t < 0.9:
        crouch = 1 - ease_out(inv(0.35, 0.9, t))
        pose = {"torso": 18 * crouch, "leg": -50 * crouch, "leg_far": -30 * crouch, "arm": -30 * crouch,
                "arm_far": 20, "head": -10 * crouch}
        y = -NEY_HIP + 5 * crouch
    elif t < 2.0:
        ph = (x - PORTAL_X) * 0.28
        pose = ney_run(ph, 0.55, 5)
        pose["arm"] = -15
        y = -NEY_HIP - abs(math.sin(ph)) * 0.8
    else:
        pose = ney_idle(t)
        y = -NEY_HIP
    spin = t > 2.0
    if spin:
        pose["arm"] = keys(t, [(2.0, -15), (2.25, -172)], ease_out)
        pose["head"] = -12
    draw_shadow(f.c, cam, x, 8, 0.35 * a)
    tint = np.array([1, 1, 1], np.float32) + TEAL * 2.5 * (1 - inv(0.25, 0.9, t))
    L = new_layer()
    mats = ney(L, cam, x, y, pose, 1, head="head_q" if spin else None, tint=tint)
    if spin:
        hx, hy = hand(mats)
        draw_ball(L, cam, hx + 0.5, hy - 3.2, t * 1400, 7)
    else:
        hx, hy = hand(mats)
        draw_ball(L, cam, hx + 1.5, hy - 3, 0, 7)
    f.glowcomp(L, TEAL, 18, 1.2 * (1 - inv(0.3, 1.4, t)), alpha=a)
    burst(f.c, cam, t, 0.25, 70, PORTAL_X, -16, 31, speed=(20, 90), colors=[TEAL, hexc("#e0fff6"), hexc("#b77bff")])
    c = f.done()
    title_slam(c, "NEYMAR JR", W / 2, H * 0.2, t - 2.0, 200, "gold", hold=1.55, out_dur=0.3)
    title_slam(c, "THE  No.10", W / 2, H * 0.2 + 175, t - 2.25, 70, "white", hold=1.3, out_dur=0.3, spacing=0.12)
    fx = {"flash": 1 - inv(0, 0.35, t), "key": 0.07, "chroma": 10 * (1 - inv(2.0, 2.4, t)) * (t > 2.0),
          "zblur": 0.18 * (1 - inv(2.0, 2.25, t)) * (t > 2.0)}
    return c, fx


# ============================================================ ACT 2 ==========
EYE_POS = [(-70, 0.8, 0.10), (45, 0.95, 0.25), (-20, 0.7, 0.40), (80, 0.75, 0.55), (-95, 1.2, 0.65),
           (5, 1.35, 0.80), (100, 1.15, 0.95), (-50, 1.7, 1.05), (60, 1.8, 1.15)]


def s_eyes(t):
    """14-16s: darkness; Enderman eyes open one pair at a time."""
    cam = Cam(0, -42, keys(t, [(0, 5.2), (2, 5.9)]))
    f = Fr(cam, t)
    draw_sky(f.c, cam, t, nebula=0.25, brightness=0.45)
    draw_pillars(f.c, None, cam, t, 0.45, crystals=False, tint=0.3)
    draw_island(f.c, cam, tint=0.18)
    L = new_layer()
    heads = []
    for i, (x, sc, t0) in sorted(enumerate(EYE_POS), key=lambda e: e[1][1]):
        mats = ender(L, cam, x, end_idle(t + i), 1, scale=sc * 0.8, tint=0.3, front=True, yf=18 * (sc - 1))
        heads.append((mats["head"], t0, sc))
    f.glowcomp(L, PURPLE, 14, 0.25)
    C = mat3(cam.M())
    flare = 1 + 0.8 * ease_out(inv(1.5, 1.6, t)) * (1 - inv(1.6, 2.0, t))
    for (hm, t0, sc) in heads:
        k = ease_out(inv(t0, t0 + 0.12, t))
        if k > 0:
            eye_rects(f.c, f.g, C @ hm, k, bright=flare)
    c = f.done()
    return c, {"glitch": 0.7 * (1 - inv(0, 0.18, t)), "chroma": 3, "letterbox": 0.6}


def s_dribble(t):
    """16-19s: side tracking shot, dribbling on the beat; Enderman teleports in."""
    T_ = 16 + t
    x = keys(t, [(0, -175), (2.45, -104), (2.9, -98)], lambda k: k if k < 0.99 else 1)
    cam = Cam(keys(t, [(0, -160), (2.4, -94), (3, -78)], ease_io), -24, keys(t, [(0, 8.6), (2.4, 8.2), (3, 7.4)]),
              shake=12 * inv(2.5, 2.55, t) * (1 - inv(2.55, 2.9, t)), t=t)
    f = Fr(cam, t)
    f.env()
    ambient_particles(f.c, cam, t, 50, area=(-300, -160, 60, 10))
    stopping = t > 2.5
    ph = (x + 175) * 0.34
    if not stopping:
        pose = ney_run(ph, 0.85, 14)
        y = -NEY_HIP - abs(math.sin(ph)) * 1.2
    else:
        pose = ney_idle(t)
        pose["torso"] = -6 * (1 - inv(2.5, 2.9, t)) + 2
        y = -NEY_HIP
    draw_shadow(f.c, cam, x, 8)
    # dribble arm follows ball
    bx = x + 5
    ground = -3.2
    if not stopping:
        b = (T_ % BEAT) / BEAT
        u = abs(2 * b - 1)
        hgt = 1 - u * u
        by = ground + (-13 - ground) * hgt
        pose["arm"] = lerp(-25, -62, hgt)
    else:
        by = -9
        pose["arm"] = -48
    mats = ney(f.c, cam, x, y, pose, 1)
    draw_ball(f.c, cam, bx, by, T_ * 500, 7)
    # Enderman teleports in
    if t > 2.5:
        fl = 1 if (t > 2.75 or int(t * 40) % 2 == 0) else 0.25
        L = new_layer()
        ender(L, cam, END_X, end_idle(t), -1, alpha=fl)
        f.glowcomp(L, PURPLE, 22, 1.4)
        burst(f.c, cam, t, 2.5, 80, END_X, -30, 41, speed=(20, 110))
    ambient_particles(f.c, cam, t, 12, seed=9, depth=1.25, area=(-400, -120, 200, 20), size=(2, 4))
    c = f.done()
    fx = {"zblur": 0.2 * (1 - inv(0, 0.2, t)), "chroma": 10 * inv(2.5, 2.52, t) * (1 - inv(2.52, 2.9, t))}
    return c, fx


def s_stare(t):
    """19-20s: extreme close-up on the Enderman, jaw drops, screech."""
    jaw = ease_out(inv(0.2, 0.35, t))
    cam = Cam(0, -15.5, keys(t, [(0, 58), (1, 66)]), shake=8 + 20 * jaw, t=t)
    f = Fr(cam, t)
    draw_sky(f.c, cam, t, nebula=1.0, brightness=0.8)
    L = new_layer()
    mats = END_FRONT.draw(L, cam, 0, 0, {"jaw_off": (0, 3.0 * jaw), "head": 3 * math.sin(t * 3)}, 1, 1.0)
    # mouth void behind jaw
    f.glowcomp(L, PURPLE, 40, 1.3)
    C = mat3(cam.M())
    if jaw > 0:
        hm = mats["head"]
        pts = [point(C @ hm, 0.2, 6), point(C @ hm, 7.8, 6), point(C @ hm, 7.8, 6 + 3 * jaw),
               point(C @ hm, 0.2, 6 + 3 * jaw)]
        poly(f.c, pts, hexc("#030205"), 1.0)
        mats = END_FRONT.draw(f.c, cam, 0, 0, {"jaw_off": (0, 3.0 * jaw), "head": 3 * math.sin(t * 3),
                                               "head_hide": True, "torso_hide": True, "arm_l_hide": True,
                                               "arm_r_hide": True, "leg_l_hide": True, "leg_r_hide": True}, 1, 1.0)
    eye_rects(f.c, f.g, C @ mats["head"], 1.0, bright=1.2 + 0.4 * jaw)
    ambient_particles(f.c, cam, t, 30, seed=4, area=(-12, -30, 12, 0), size=(0.3, 0.8), rise=4)
    c = f.done()
    return c, {"chroma": 4 + 14 * jaw, "zblur": 0.1 * jaw, "glitch": 0.35 * inv(0.2, 0.25, t) * (1 - inv(0.25, 0.45, t))}


def s_smirk(t):
    """20-21s: close-up on Neymar, unbothered; smirk at the camera."""
    cam = Cam(0, -22, keys(t, [(0, 30), (1, 37)]))
    f = Fr(cam, t)
    draw_sky(f.c, cam, t, nebula=1.2, brightness=0.9)
    head = "head_smile" if t > 0.42 else "head_front"
    pose = {"arm_l": 18, "arm_r": keys(t, [(0, -20), (0.4, -120)], ease_out), "head": 4 * math.sin(t * 2) - 3,
            "torso": 0}
    L = new_layer()
    mats = ney(L, cam, 0, 0, pose, 1, head=head, front=True)
    hx, hy = point(mats["arm_r"], 2, 11.5)
    draw_ball(L, cam, hx, hy - 3, t * 1200, 7)
    f.glowcomp(L, hexc("#ffcf5a"), 30, 0.35)
    # purple rim from the Enderman side
    c = f.done()
    c[:, :, :] += np.linspace(0, 1, W, dtype=np.float32)[None, :, None] ** 3 * hexc("#6a1fb0") * 0.35
    return c, {"mblur": (-160 * (1 - inv(0, 0.14, t)), 0), "chroma": 3}


# ---- crossover choreography (story time tau) ----
def crossover_scene(f, tau, T_):
    """Draw the crossover at story time tau (0..2.4). Returns Neymar x for camera."""
    cam = f.cam
    # Neymar
    nx = keys(tau, [(0, -112), (0.5, -95), (0.8, -92), (1.1, -95), (1.65, -18), (2.4, 5)],
              lambda k: smooth(k))
    ground = -3.2
    if tau < 0.8:  # approach + hesitation dribble
        ph = tau * 9
        pose = ney_run(ph, 0.6 * (1 - inv(0.5, 0.8, tau)) + 0.1, 10)
        b = (tau % 0.4) / 0.4
        hgt = 1 - abs(2 * b - 1) ** 2
        bx, by = nx + 5, ground + (-13 - ground) * hgt
        pose["arm"] = lerp(-25, -62, hgt)
        behind = False
        lean = 10 - 20 * inv(0.5, 0.8, tau)
        pose["torso"] = lean
    elif tau < 1.1:  # crossover through the legs
        k = inv(0.8, 1.1, tau)
        pose = {"torso": -14 + 10 * k, "leg": 30, "leg_far": -35, "arm": lerp(-40, 10, k), "arm_far": lerp(10, -40, k),
                "head": 8}
        bx = lerp(nx + 6, nx - 4, smooth(k))
        by = ground - math.sin(k * math.pi) * 2
        behind = 0.25 < k < 0.8
    else:  # explode past
        k = inv(1.1, 1.65, tau)
        ph = (nx + 95) * 0.33
        pose = ney_run(ph, 1.0, 24 * (1 - inv(1.9, 2.4, tau)))
        b = (tau % 0.3) / 0.3
        hgt = 1 - abs(2 * b - 1) ** 2
        bx, by = nx + 6, ground + (-12 - ground) * hgt
        pose["arm"] = lerp(-25, -62, hgt)
        behind = False
    ny = -NEY_HIP - (abs(math.sin(tau * 9)) * 1.2 if tau > 1.1 else 0)
    # Enderman: stance -> lunge wrong way -> legs cross -> falls backwards
    if tau < 0.85:
        ep = ender_stance(T_)
        theta, drop = 0.0, 3.0
    elif tau < 1.3:
        k = smooth(inv(0.85, 1.3, tau))
        ep = ender_stance(T_)
        ep.update({"torso": lerp(-8, 28, k), "leg": lerp(22, 40, k), "leg_far": lerp(-22, -30, k),
                   "arm": lerp(-70, -120, k), "arm_far": lerp(-45, -95, k), "head": 10})
        theta, drop = 0.0, lerp(3, 6, k)
    else:
        k = inv(1.3, 2.15, tau)
        kk = ease_in(k, 2.2)
        bounce = math.sin(inv(2.15, 2.4, tau) * math.pi) * 6 * (tau > 2.15)
        ep = {"torso": lerp(28, -20, smooth(k)), "leg": lerp(40, -25, smooth(k)), "leg_far": lerp(-30, 30, smooth(k)),
              "arm": lerp(-120, -175, k), "arm_far": lerp(-95, -150, k), "head": lerp(10, -20, k), "jaw": 0.0}
        theta, drop = -88 * kk + bounce, 6 * (1 - k)
    # ball behind body?
    if behind:
        draw_ball(f.c, cam, bx, by, T_ * 600, 7)
    draw_shadow(f.c, cam, nx, 8)
    draw_shadow(f.c, cam, END_X, 8)
    L = new_layer()
    ender(L, cam, END_X, ep, -1, theta=theta, hip_drop=drop)
    f.glowcomp(L, PURPLE, 22, 1.3)
    ney(f.c, cam, nx, ny, pose, 1)
    if not behind:
        draw_ball(f.c, cam, bx, by, T_ * 600, 7)
    if tau > 2.15:
        dust(f, END_X + 25, -2, tau, 2.15, 30, 5, spread=160)
        burst(f.c, cam, tau, 2.15, 30, END_X + 25, -6, 61, speed=(20, 80))
    return nx


def cross_tau(r):
    if r < 0.8:
        return r
    if r < 2.0:
        return 0.8 + (r - 0.8) * 0.3
    if r < 2.5:
        return 1.16 + (r - 2.0) * 2.4
    return 2.36


def s_crossover(t):
    """21-24.5s: speed-ramped crossover; freeze frame ANKLES BROKEN."""
    tau = cross_tau(t)
    slow = inv(0.8, 1.0, t) * (1 - inv(1.9, 2.0, t))
    cx = keys(t, [(0, -85), (0.8, -80), (2.0, -72), (2.5, -32), (3.5, -30)])
    s_ = keys(t, [(0, 7.2), (0.8, 7.6), (1.9, 10.5), (2.2, 7.8), (3.5, 8.6)])
    cam = Cam(cx, keys(t, [(0, -28), (1.9, -22), (2.2, -28)]), s_,
              shake=(30 * (1 - inv(2.45, 2.9, t)) * (t > 2.45)), t=t)
    f = Fr(cam, t)
    f.env()
    ambient_particles(f.c, cam, 21 + tau, 40, area=(-250, -150, 100, 10))
    crossover_scene(f, tau, 21 + tau)
    c = f.done()
    frz = t >= 2.5
    if frz:
        title_slam(c, "ANKLES", W / 2, H * 0.24, t - 2.55, 210, "silver", hold=0.75, out_dur=0.2)
        title_slam(c, "BROKEN", W / 2, H * 0.24 + 200, t - 2.68, 210, "silver", hold=0.62, out_dur=0.2)
    fx = {"roll": 5 * slow, "desat": 0.65 * frz + 0.25 * slow, "chroma": 3 + 10 * slow + 16 * frz * (1 - inv(2.5, 2.8, t)),
          "zoom": 1 + 0.04 * inv(2.5, 3.5, t), "flash": 0.8 * (1 - inv(2.5, 2.62, t)) * frz,
          "mblur": (0, 0) if t > 0.15 else (120 * (1 - t / 0.15), 0), "letterbox": 0.8 * slow}
    return c, fx


def s_replay(t):
    """24.5-27s: VHS rewind, then mirrored low-angle replay at half speed."""
    if t < 0.4:
        tau = lerp(2.36, 0.6, t / 0.4)
    else:
        tau = 0.6 + (t - 0.4) * 0.62
    tau = min(tau, 2.2)
    nx_est = keys(tau, [(0, -112), (0.5, -95), (0.8, -92), (1.1, -95), (1.65, -18), (2.4, 5)])
    cam = Cam(lerp(-70, nx_est, 0.35), -13, 12.5)
    f = Fr(cam, t)
    f.env()
    crossover_scene(f, tau, 21 + tau)
    c = f.done()
    c = mirror(c)
    label_replay(c, t)
    fx = {"vhs": 1.0, "key": 0.06, "glitch": 0.6 * (1 - inv(0.3, 0.45, t)), "desat": 0.2}
    return c, fx


# ============================================================ ACT 3 ==========
def s_rumble(t):
    """27-29s: the ground shakes, crystals flare and fire beams into the sky."""
    sh = 4 + 26 * ease_in(inv(0, 2, t))
    cam = Cam(-4, -66, keys(t, [(0, 4.6), (2, 4.2)]), shake=sh, t=t)
    f = Fr(cam, t)
    flare = ease_in(inv(0.3, 1.8, t))
    tops = f.env(flare=flare, sky_bright=1 + 0.3 * flare * (0.5 + 0.5 * math.sin(t * 30)))
    draw_hoop(f.c, cam, HOOP_X, t)
    draw_portal(f.c, f.g, cam, t, eyes=14, energy=0.8)
    L = new_layer()
    ender(L, cam, -60, ender_lying(), -1, theta=-88)
    f.glowcomp(L, PURPLE, 18, 1.0)
    pose = ney_idle(t)
    pose["head"] = keys(t, [(0.5, -1), (0.9, -28)], ease_out)
    ney(f.c, cam, -8, -NEY_HIP, pose, 1)
    draw_ball(f.c, cam, -3.5, -5, 0, 7)
    if flare > 0.05:
        starts = [lc.to_screen(*p) for (lc, p) in tops]
        beams(f.c, f.g, [s for s in starts if -100 < s[0] < W + 100], (W / 2 + 60, -80), t, flare)
    dust(f, -40, -1, t, 0.4, 20, 71, spread=120)
    dust(f, 60, -1, t, 1.0, 20, 72, spread=120)
    c = f.done()
    return c, {"fade": 1 - inv(0, 0.15, t), "chroma": 4 * flare, "letterbox": 1.0}


def dragon_below(L, cx, cy, k, t, flap_speed=5.5):
    """Dragon silhouette seen from below, head toward the viewer (down-screen)."""
    f = math.sin(t * flap_speed)
    col = hexc("#0e0b14")
    mem = hexc("#17121f")

    def P(x, y):
        return (cx + x * k, cy + y * k)

    for sgn in (-1, 1):
        wing = [P(sgn * 12, -8), P(sgn * 48, -30 + f * 14), P(sgn * 98, -18 + f * 34), P(sgn * 88, -2 + f * 26),
                P(sgn * 70, 2 + f * 18), P(sgn * 52, 8 + f * 12), P(sgn * 34, 6 + f * 6), P(sgn * 14, 12)]
        poly(L, wing, mem)
        for tip in ((98, -18 + f * 34), (88, -2 + f * 26), (70, 2 + f * 18)):
            line(L, P(sgn * 48, -30 + f * 14), P(sgn * tip[0], tip[1]), hexc("#2a2436"), max(2, k * 1.4))
        line(L, P(sgn * 12, -8), P(sgn * 48, -30 + f * 14), hexc("#2a2436"), max(3, k * 3))
    # tail
    px, py = 0, -28
    for i in range(12):
        nx = math.sin(t * 2.5 + i * 0.6) * (3 + i * 0.8)
        ny = -28 - (i + 1) * 7
        w = max(0.8, 6 - i * 0.45)
        line(L, P(px, py), P(nx, ny), col, w * k * 1.6)
        px, py = nx, ny
    poly(L, [P(-12, -28), P(12, -28), P(16, 0), P(12, 22), P(-12, 22), P(-16, 0)], col)
    poly(L, [P(-6, 20), P(6, 20), P(5, 36), P(-5, 36)], col)
    poly(L, [P(-10, 34), P(10, 34), P(9, 52), P(-9, 52)], col)
    return [P(-5.5, 47), P(5.5, 47)]


def s_flyover(t):
    """29-32s: looking up - the Ender Dragon swoops over the camera. FINAL BOSS."""
    cam = Cam(0, -1100, 3.2, shake=6 + 18 * inv(1.5, 2.7, t), t=t)
    f = Fr(cam, t)
    draw_sky(f.c, cam, t, nebula=1.3, brightness=1.1)
    kk = ease_in(inv(0, 2.9, t), 2.4)
    k = lerp(0.9, 26, kk)
    cy = lerp(H * 0.22, H * 1.05, ease_in(inv(0, 2.9, t), 1.6))
    L = new_layer()
    eyes = dragon_below(L, W / 2 + math.sin(t * 1.3) * 60, cy, k, t)
    f.glowcomp(L, hexc("#9a3cff"), 30, 1.1)
    for ex, ey in eyes:
        disc(f.c, ex, ey, max(3, k * 1.6), hexc("#ffd6ff"), 1.0)
        disc(f.g, ex, ey, max(3, k * 1.6), hexc("#c040ff"), 1.0, add=True, soft=max(10, k * 8))
    # crystal beams from the pillar tops at the frame edges
    starts = [(-40, H * 0.95), (W + 40, H * 0.9), (-40, H * 0.55), (W + 40, H * 0.6)]
    beams(f.c, f.g, starts, (W / 2 + math.sin(t * 1.3) * 60, cy), t, 0.8 * (1 - inv(2.4, 2.9, t)))
    c = f.done()
    if 0.9 < t < 2.7:
        title_slam(c, "FINAL BOSS", W / 2, H * 0.72, t - 0.9, 190, "red", hold=1.5, out_dur=0.25)
        typewriter(c, "THE ENDER DRAGON", W / 2, H * 0.72 + 150, t - 1.2, 38, 26, style="purple",
                   alpha=1 - inv(2.4, 2.65, t))
    wipe = inv(2.55, 2.95, t)
    fx = {"key": 0.16, "mblur": (0, -220 * (1 - inv(0, 0.18, t))), "glitch": 0.5 * inv(0.9, 0.95, t) * (1 - inv(0.95, 1.2, t)),
          "chroma": 3 + 6 * kk, "fade": wipe}
    return c, fx


def s_landing(t):
    """32-34s: the dragon slams down in front of the hoop and roars."""
    land = 0.75
    dy = keys(t, [(0, -250), (land, DRAGON_Y)], lambda k: ease_in(k, 2))
    imp = t - land
    shake = 34 * (1 - inv(0, 0.7, imp)) * (imp > 0)
    cam = Cam(38, -86, keys(t, [(0, 3.3), (2, 3.6)]), shake=shake, t=t)
    f = Fr(cam, t)
    f.env(flare=0.4)
    draw_hoop(f.c, cam, HOOP_X, t)
    L = new_layer()
    ender(L, cam, -60, ender_lying(), -1, theta=-88)
    f.glowcomp(L, PURPLE, 18, 1.0)
    pose = ney_idle(t)
    pose["head"] = -18 + 14 * inv(0.6, 1.0, t)
    ney(f.c, cam, -8, -NEY_HIP, pose, 1)
    draw_ball(f.c, cam, -3.5, -5, 0, 7)
    roar = ease_out(inv(1.05, 1.3, t)) * (1 - inv(1.85, 2.0, t))
    flap = t * 2 * math.pi * 1.6 if t < land + 0.15 else math.pi / 2
    draw_shadow(f.c, cam, DRAGON_X, 50 * clamp(1 - (dy - DRAGON_Y) / -200), 0.5)
    dm = dragon(f, DRAGON_X, dy, t, flap=flap, neck_raise=roar, jaw=roar, legs_down=1.0)
    if imp > 0:
        dust(f, DRAGON_X - 30, -1, t, land, 40, 81, spread=80, direction=-150, speed=(40, 140))
        dust(f, DRAGON_X + 30, -1, t, land, 40, 82, spread=80, direction=-30, speed=(40, 140))
    if roar > 0:
        mx, my = dm["mouth"]
        burst(f.c, cam, t, 1.1, 90, mx, my, 91, speed=(60, 160), life=(0.4, 0.9), spread=40, direction=170,
              size=(2, 5), colors=[hexc("#c35cff"), hexc("#7c2bd6"), hexc("#f0b8ff")])
    c = f.done()
    if imp > 0:
        sx, sy = cam.to_screen(DRAGON_X, 0)
        shockwave(c, sx, sy, t, land, 1100, 0.6, (0.9, 0.8, 1.0), 50)
    return c, {"fade": 1 - inv(0, 0.12, t), "chroma": 4 + 14 * roar + 12 * (1 - inv(0, 0.4, imp)) * (imp > 0),
               "zblur": 0.12 * roar}


def s_dragon_eye(t):
    """34-35s: extreme close-up of the dragon's eye; the pupil narrows."""
    dm = dragon_mats(DRAGON_X, DRAGON_Y, 34 + t, facing=-1, scale=DRAGON_SC, flap=math.pi / 2, neck_raise=0.3)
    ex, ey = point(dm["head"], 12.5, -2)
    cam = Cam(ex - 4, ey + 2, keys(t, [(0, 24), (1, 30)]), shake=5, t=t)
    f = Fr(cam, t)
    draw_sky(f.c, cam, t, nebula=1.0, brightness=0.8)
    dm = dragon(f, DRAGON_X, DRAGON_Y, 34 + t, flap=math.pi / 2, neck_raise=0.3, glow=0.5)
    C = mat3(cam.M())
    hm = dm["head"]
    wpx = lerp(1.2, 0.25, ease_out(inv(0.15, 0.5, t)))
    pts = [point(C @ hm, 13 - wpx / 2, 2.6), point(C @ hm, 13 + wpx / 2, 2.6), point(C @ hm, 13 + wpx / 2, 5.4),
           point(C @ hm, 13 - wpx / 2, 5.4)]
    poly(f.c, pts, hexc("#050208"), 1.0)
    burst(f.c, cam, t, 0.0, 40, ex - 4, ey + 8, 101, speed=(4, 14), life=(0.6, 1.0), direction=180, spread=60,
          size=(0.4, 1.0))
    c = f.done()
    return c, {"zblur": 0.18 * (1 - inv(0, 0.2, t)), "chroma": 6, "letterbox": 1.0}


def s_hero(t):
    """35-36s: low-angle close-up; Neymar spins the ball, catches it, crouches."""
    cam = Cam(-6, -25, keys(t, [(0, 15), (1, 18)]))
    f = Fr(cam, t)
    f.env(flare=0.3)
    pose = ney_idle(t)
    crouch = ease_out(inv(0.6, 0.85, t))
    pose.update({"torso": 4 + 16 * crouch, "leg": -25 * crouch, "leg_far": 18 * crouch, "head": -8 - 8 * crouch})
    y = -NEY_HIP + 3 * crouch
    if t < 0.6:
        pose["arm"] = -172
    else:
        pose["arm"] = lerp(-172, -60, ease_out(inv(0.6, 0.75, t)))
    mats = ney(f.c, cam, -8, y, pose, 1, head="head_q")
    hx, hy = hand(mats)
    draw_ball(f.c, cam, hx + (0.5 if t < 0.6 else 1.5), hy - (3.2 if t < 0.6 else 1), t * (1500 if t < 0.6 else 0), 7)
    c = f.done()
    c += np.linspace(0, 1, W, dtype=np.float32)[None, :, None] ** 2.5 * hexc("#7a22c9") * 0.45
    return c, {"key": 0.1, "zblur": 0.15 * (1 - inv(0, 0.15, t)), "letterbox": 1.0}


def sprint_path(t):
    """World path for 36-40s (sprint, spin, run up the dragon, launch)."""
    dm = dragon_mats(DRAGON_X, DRAGON_Y, 36 + t, facing=-1, scale=DRAGON_SC, flap=math.pi / 2,
                     head_dip=bite_k(t), jaw=bite_jaw(t))
    head_top = point(dm["head"], 6, -7)
    back_f = point(dm["base"], 12, -11)
    back_b = point(dm["base"], -8, -12)
    pts = [(0.0, (-8, -NEY_HIP)), (1.0, (26, -NEY_HIP)), (1.4, (34, -NEY_HIP)), (1.75, (40, -NEY_HIP)),
           (2.1, (head_top[0], head_top[1] - NEY_HIP - 2)), (2.6, (back_f[0], back_f[1] - NEY_HIP)),
           (3.0, (back_b[0], back_b[1] - NEY_HIP)), (4.0, (back_b[0] + 16, back_b[1] - NEY_HIP - 70))]
    x, y = keys(t, pts, lambda k: k)
    if 3.0 < t:  # ballistic launch
        k = inv(3.0, 4.0, t)
        y = back_b[1] - NEY_HIP - 70 * math.sin(k * math.pi / 2)
    return x, y


def bite_k(t):
    return ease_out(inv(0.85, 1.05, t)) * (1 - 0.4 * inv(1.3, 1.8, t))


def bite_jaw(t):
    return keys(t, [(0.7, 0), (0.95, 1), (1.08, 0), (1.3, 0.3)])


def s_sprint(t):
    """36-40s: sprint on the drop, spin move past the bite, run up the dragon, launch."""
    T_ = 36 + t
    x, y = sprint_path(t)
    xp, yp = sprint_path(max(0, t - 1 / FPS))
    ccx = keys(t, [(0, 18), (1.5, 50), (3.0, 88), (4, 110)], ease_io)
    ccy = keys(t, [(0, -34), (1.8, -40), (3.0, -80), (4, -120)], ease_io)
    s_ = keys(t, [(0, 5.6), (1.8, 5.2), (3.0, 4.6), (4, 5.4)])
    cam = Cam(ccx, ccy, s_, shake=10 * inv(0.9, 1.0, t) * (1 - inv(1.0, 1.4, t)) + 3, t=t)
    camp = Cam(keys(t - 1 / FPS, [(0, 18), (1.5, 50), (3.0, 88), (4, 110)], ease_io),
               keys(t - 1 / FPS, [(0, -34), (1.8, -40), (3.0, -80), (4, -120)], ease_io), s_)
    f = Fr(cam, t)
    f.env(flare=0.5)
    draw_hoop(f.c, cam, HOOP_X, t)
    dm = dragon(f, DRAGON_X, DRAGON_Y, T_, flap=math.pi / 2 + math.sin(t * 3) * 0.2, head_dip=bite_k(t),
                jaw=bite_jaw(t))
    spinning = 1.0 < t < 1.4
    facing = 1
    if spinning:
        facing = 1 if int((t - 1.0) / 0.07) % 2 == 0 else -1
    airborne = t > 1.75
    ph = t * 13
    if t > 3.0:
        pose = {"torso": -10, "leg": -70, "leg_far": 30, "arm": -200, "arm_far": -190, "head": -15}
    elif airborne and (2.05 < t < 2.15 or 2.95 < t < 3.05):
        pose = {"torso": 20, "leg": -60, "leg_far": 10, "arm": -120, "arm_far": 40, "head": -10}
    else:
        pose = ney_run(ph, 1.0, 20)
    ground = -3.2 if not airborne else y + NEY_HIP - 3
    b = (T_ % 0.25) / 0.25
    hgt = 1 - abs(2 * b - 1) ** 2
    if not airborne and not spinning:
        bx, by = x + 6, ground + (-12 - ground) * hgt
        pose["arm"] = lerp(-25, -62, hgt)
    elif t > 3.0:
        bx, by = None, None
    else:
        pose["arm"] = -40
        bx, by = None, None
    if not airborne:
        draw_shadow(f.c, cam, x, 8)
    mats = ney(f.c, cam, x, y, pose, facing)
    if bx is None:
        if t > 3.0:
            hx, hy = point(mats["arm"], 2, 11.5)
            draw_ball(f.c, cam, hx, hy + 2, 0, 7)
        else:
            hx, hy = hand(mats)
            draw_ball(f.c, cam, hx + 1.5 * facing, hy, 0, 7)
    else:
        draw_ball(f.c, cam, bx, by, T_ * 800, 7)
    if 0.95 < t < 1.3:
        mx, my = dm["mouth"]
        burst(f.c, cam, t, 0.95, 40, mx, my, 111, speed=(30, 90), life=(0.3, 0.6), direction=180, spread=90)
    if spinning:
        burst(f.c, cam, t, 1.0, 30, x, -16, 112, speed=(20, 60), life=(0.3, 0.5), colors=[hexc("#ffffff"), GOLD])
    if 1.75 < t < 2.2:
        dust(f, 42, -1, t, 1.75, 16, 113, spread=100)
    c = f.done()
    vx = (cam.cx - camp.cx) * cam.s
    vy = (cam.cy - camp.cy) * cam.s
    speed_lines(c, t, 0.9 * (1 - inv(2.6, 3.0, t)) + 0.6 * inv(3.0, 3.3, t), cx=W * 0.3, cy=H * 0.5)
    fx = {"flash": 0.9 * (1 - inv(0, 0.2, t)), "mblur": (-vx * 2.2, -vy * 2.2), "chroma": 4 + 8 * spinning,
          "zblur": 0.12 * (1 - inv(0, 0.3, t)) + 0.15 * inv(3.0, 3.1, t) * (1 - inv(3.1, 3.4, t))}
    if spinning:
        fx["zoom"] = 1.06
    return c, fx


def s_air(t):
    """40-43s: slow-motion hero shot mid-air, camera rolls, time nearly stops."""
    px = lerp(100, 108, t / 3)
    py = lerp(-132, -136, t / 3)
    cam = Cam(keys(t, [(0, px), (2.6, px + 4), (3, px + 16)], ease_in), keys(t, [(0, py + 2), (2.6, py + 4), (3, py + 22)], ease_in),
              keys(t, [(0, 12), (2.6, 10.5), (3, 8)]))
    f = Fr(cam, t)
    draw_sky(f.c, cam, t, nebula=1.3, brightness=1.0)
    draw_pillars(f.c, f.g, cam, t * 0.2, 0.45, flare=0.5, tint=0.7)
    draw_pillars(f.c, f.g, cam, t * 0.2, 0.7, flare=0.5, tint=0.9)
    # blur the background for depth of field
    f.c = cv2.GaussianBlur(f.c, (0, 0), 5)
    draw_island(f.c, cam)
    draw_hoop(f.c, cam, HOOP_X, t)
    tt = t * 0.12
    pose = {"torso": -12 + 3 * math.sin(tt * 3), "leg": -85, "leg_far": 25, "arm": -205, "arm_far": -195,
            "head": -12}
    L = new_layer()
    mats = ney(L, cam, px, py, pose, 1)
    hx, hy = point(mats["arm"], 2, 11.5)
    draw_ball(L, cam, hx - 0.5, hy + 1.5, 20 + tt * 100, 7)
    f.glowcomp(L, hexc("#ffe7a8"), 26, 0.35)
    ambient_particles(f.c, cam, tt, 70, seed=12, area=(40, -200, 180, -60), size=(0.4, 1.2), rise=2)
    c = f.done()
    fx = {"roll": keys(t, [(0, -14), (2.6, 12), (3, 18)], ease_io), "letterbox": 1.0, "desat": 0.2,
          "chroma": 3 + 6 * inv(2.6, 3.0, t), "zblur": 0.1 * (1 - inv(0, 0.3, t)) + 0.25 * inv(2.7, 3.0, t),
          "bloom": 0.75}
    return c, fx


# ---- the dunk (story time u: <0 approach, 0 = contact) ----
def dunk_scene(f, u, T_, hang_head=True, dragon_on=True):
    cam = f.cam
    rx, ry = RIM
    bend = ease_out(inv(0.0, 0.08, u)) * (1 + 0.15 * math.sin(max(0, u) * 25) * math.exp(-max(0, u) * 4))
    board = u < 0.02
    draw_hoop(f.c, cam, HOOP_X, T_, bend=min(bend, 1.15), board=board, swish=clamp(u * 3) * (1 - inv(0.5, 1.5, u)))
    if dragon_on:
        slump = ease_out(inv(0.02, 0.5, u))
        dragon(f, DRAGON_X, DRAGON_Y + 10 * slump, T_, flap=math.pi / 2 + slump * 1.2, head_dip=0.5 + 0.8 * slump,
               jaw=0.3 * slump, eye_glow=1 - 0.7 * slump, glow=1 - 0.4 * slump)
    # Neymar
    if u < 0:
        k = inv(-0.6, 0.0, u)
        nx = lerp(rx - 22, rx - 6, k)
        nyy = lerp(ry - 58, ry - 26, ease_in(k, 1.5))
        swing = ease_in(inv(-0.18, 0.0, u), 2)
        pose = {"torso": lerp(-12, 20, swing), "leg": -80 + 50 * swing, "leg_far": 25, "arm": lerp(-205, -130, swing),
                "arm_far": lerp(-195, -125, swing), "head": -10}
        mats = ney(f.c, cam, nx, nyy, pose, 1)
        hx, hy = point(mats["arm"], 2, 11.5)
        draw_ball(f.c, cam, hx, hy + 1, 0, 7)
    else:
        sw = math.sin(u * 7) * 18 * math.exp(-u * 2.2)
        hx_r, hy_r = point(T(HOOP_X - 22, -62) @ R(-min(bend, 1.15) * 38) @ T(-14, -1), 5, 1)
        pose = {"torso": sw, "leg": 18 + sw * 0.8, "leg_far": -8 + sw * 0.5, "arm": -178 - sw * 0.2,
                "arm_far": -170 - sw * 0.2, "head": 6}
        # hands on rim -> hip below
        a = math.radians(sw)
        hipx = hx_r - math.sin(a) * 21
        hipy = hy_r + math.cos(a) * 21
        ney(f.c, cam, hipx, hipy, pose, 1, head="head_q")
        # ball drops through the net and bounces
        tb = u
        if tb < 0.35:
            by = ry + 4 + 60 * tb * tb * 4
        else:
            tt = tb - 0.35
            bh = abs(math.sin(tt * 5)) * 22 * math.exp(-tt * 1.8)
            by = -3.5 - bh
        by = min(by, -3.5)
        bx = rx - 2 - 12 * clamp(u - 0.3)
        draw_ball(f.c, cam, bx, by, u * 400, 7)
    if u >= 0:
        bx0 = HOOP_X - 20
        burst(f.c, cam, u, 0.0, 70, bx0, -75, 121, speed=(40, 190), life=(0.6, 1.4), size=(2, 5), gravity=160,
              colors=[hexc("#ffffff"), hexc("#dcdcee"), hexc("#d2261c"), hexc("#a9d8ff")], add=False)
        burst(f.c, cam, u, 0.0, 50, rx, ry, 122, speed=(60, 220), life=(0.2, 0.5), size=(0.8, 2),
              colors=[hexc("#fff6d0"), GOLD], add=True)


def s_dunk(t):
    """43-45s: POSTERIZED."""
    u = t - 0.12
    sh = 45 * (1 - inv(0, 0.8, u)) * (u > 0)
    cam = Cam(RIM[0] - 14, RIM[1] + 6, keys(t, [(0, 7.2), (0.2, 7.8), (2, 8.6)]), shake=sh, t=t)
    f = Fr(cam, t)
    f.env(flare=0.4)
    dunk_scene(f, u, 43 + t)
    c = f.done()
    if u > 0:
        sx, sy = cam.to_screen(*RIM)
        shockwave(c, sx, sy, t, 0.12, 1300, 0.55, (1, 0.95, 0.85), 60)
        shockwave(c, sx, sy, t, 0.22, 1300, 0.7, (0.8, 0.5, 1.0), 40)
    title_slam(c, "POSTERIZED", W / 2, H * 0.2, t - 0.3, 190, "silver", hold=1.4, out_dur=0.2)
    fx = {"flash": clamp(1 - inv(0.12, 0.45, t)) * (t > 0.1), "zblur": 0.3 * (1 - inv(0.12, 0.5, t)) * (t > 0.12),
          "chroma": 3 + 16 * (1 - inv(0.12, 0.6, t)) * (t > 0.12), "zcenter": cam.to_screen(*RIM)}
    return c, fx


def s_dunk_replays(t):
    """45-47s: two instant replays of the dunk from new angles."""
    if t < 1.0:
        u = -0.35 + t * 0.6
        cam = Cam(RIM[0] - 6, RIM[1] + 4, 14.5, shake=30 * (1 - inv(0.58, 0.9, t)) * (u > 0), t=t)
        f = Fr(cam, t)
        f.env()
        dunk_scene(f, u, 45 + t)
        c = f.done()
        label_replay(c, t, "REPLAY")
        return c, {"key": 0.14, "flash": 0.8 * (1 - inv(0, 0.12, t)), "vhs": 0.5, "chroma": 6,
                   "zblur": 0.2 * (1 - inv(0.58, 0.8, t)) * (u > 0)}
    tt = t - 1.0
    u = -0.25 + tt * 0.7
    cam = Cam(70, -78, 4.0, shake=26 * (1 - inv(0.36, 0.8, tt)) * (u > 0), t=tt)
    f = Fr(cam, tt)
    f.env(flare=0.4)
    L = new_layer()
    ender(L, cam, -60, ender_lying(), -1, theta=-88)
    f.glowcomp(L, PURPLE, 18, 1.0)
    dunk_scene(f, u, 46 + tt)
    c = f.done()
    if u > 0:
        sx, sy = cam.to_screen(*RIM)
        shockwave(c, sx, sy, tt, 0.36, 1100, 0.5, (1, 1, 1), 40)
    c = mirror(c)
    label_replay(c, tt, "REPLAY")
    return c, {"flash": 0.8 * (1 - inv(0, 0.12, tt)), "vhs": 0.5, "chroma": 5, "letterbox": 1.0}


# ============================================================ ACT 4 ==========
def s_defeat(t):
    """47-51s: the dragon rises, light bursts from within, it explodes into particles + XP."""
    cam = Cam(62, -95, keys(t, [(0, 3.3), (4, 3.0)]), shake=4 + 30 * inv(2.6, 2.8, t) * (1 - inv(2.8, 3.5, t)), t=t)
    f = Fr(cam, t)
    f.env(flare=0.2)
    draw_hoop(f.c, cam, HOOP_X, t, bend=1.0, board=False)
    L = new_layer()
    ender(L, cam, -60, ender_lying(), -1, theta=-88)
    f.glowcomp(L, PURPLE, 18, 1.0)
    rise = ease_io(inv(0.0, 2.6, t))
    dy = DRAGON_Y + 10 - 110 * rise
    alive = t < 2.75
    if alive:
        fl = 1 + 2.5 * inv(1.8, 2.7, t) * (0.5 + 0.5 * math.sin(t * 60))
        dm = dragon(f, DRAGON_X, dy, 47 + t, flap=math.pi / 2 + 1.2 - rise * 0.9, head_dip=1.3 - rise * 1.6,
                    jaw=0.6 * rise, glow=fl)
        sx, sy = cam.to_screen(DRAGON_X, dy)
        light_rays(f.c, sx, sy, t, 16, ease_out(inv(0.6, 2.6, t)), hexc("#fff0ff"), 1600, 0.04, 0.6, alpha=0.55)
        light_rays(f.g, sx, sy, t, 16, ease_out(inv(0.6, 2.6, t)), hexc("#d070ff"), 1600, 0.06, 0.6, alpha=0.4)
    burst(f.c, cam, t, 2.75, 160, DRAGON_X, dy, 131, speed=(40, 240), life=(0.8, 1.6), size=(2, 6), drag=1.2,
          colors=[hexc("#e27bff"), hexc("#b44bff"), hexc("#ffd1ff"), hexc("#ffffff")])
    burst(f.c, cam, t, 2.8, 60, DRAGON_X, dy, 132, speed=(30, 120), life=(1.0, 1.25), size=(2.5, 4), gravity=80,
          colors=[hexc("#b8ff3a"), hexc("#f2ff6a"), hexc("#7dff4a")], add=True, spin=False)
    # Neymar hanging, then drops down
    if t < 3.2:
        hx_r, hy_r = point(T(HOOP_X - 22, -62) @ R(-38) @ T(-14, -1), 5, 1)
        pose = {"torso": 2 * math.sin(t * 2), "leg": 16, "leg_far": -6, "arm": -178, "arm_far": -172, "head": -14}
        ney(f.c, cam, hx_r, hy_r + 21, pose, 1, head="head_q")
    else:
        k = inv(3.2, 3.55, t)
        yy = lerp(-38, -NEY_HIP, ease_in(k, 2))
        pose = ney_idle(t) if k >= 1 else {"torso": 0, "leg": -20, "leg_far": 10, "arm": -150, "arm_far": -140}
        ney(f.c, cam, RIM[0] + 2, yy, pose, 1)
        if k >= 1:
            dust(f, RIM[0] + 2, -1, t, 3.55, 16, 133, spread=140)
    c = f.done()
    fx = {"flash": 0.9 * (1 - inv(2.75, 3.2, t)) * (t > 2.75) + 0.5 * (1 - inv(0, 0.3, t)), "letterbox": 1.0,
          "chroma": 4 + 12 * (1 - inv(2.75, 3.3, t)) * (t > 2.75), "bloom": 0.7}
    return c, fx


def confetti(c, t, n=120, seed=5, k=1.0):
    cols = [GOLD, hexc("#ffffff"), hexc("#c35cff"), hexc("#39f0c8"), hexc("#ff5ab8")]
    for i in range(n):
        x = (hash01(seed, i) * W * 1.2 - W * 0.1 + math.sin(t * 2 + i) * 40)
        y = ((hash01(seed, i, 2) * H * 1.3) + t * lerp(200, 420, hash01(seed, i, 3))) % (H * 1.3) - H * 0.15
        sz = lerp(8, 18, hash01(seed, i, 4))
        ang = t * lerp(2, 8, hash01(seed, i, 5)) + i
        w2 = abs(math.cos(ang)) * sz
        pts = [(x - w2, y - sz * 0.5), (x + w2, y - sz * 0.3), (x + w2, y + sz * 0.5), (x - w2, y + sz * 0.3)]
        poly(c, pts, cols[i % len(cols)], 0.95 * k)


CROWD = [(-150, 0.7, 0), (-110, 0.7, 1), (-70, 0.7, 2), (-30, 0.7, 3), (10, 0.7, 4), (50, 0.7, 5), (90, 0.7, 6),
         (130, 0.7, 7), (170, 0.7, 8)]


def s_crowd(t):
    """51-54s: the Endermen crowd celebrates; Neymar faces the camera, arms up."""
    cam = Cam(keys(t, [(0, -16), (3, 12)], ease_io), -34, keys(t, [(0, 7.0), (3, 7.8)]))
    f = Fr(cam, t)
    f.env(flare=0.2)
    lc = cam.layer(0.9)
    L = new_layer()
    for (x, d, i) in CROWD:
        ph = t * 7 + i * 1.3
        up = 0.5 + 0.5 * math.sin(ph)
        pose = {"arm_l": 40 + 120 * up, "arm_r": -40 - 120 * up, "leg_l": 4, "leg_r": -4, "head": 5 * math.sin(ph),
                "jaw": 0}
        mats = ender(L, lc, x * 0.8, pose, 1, scale=1.0, front=True, yf=-6, hip_drop=0)
    f.glowcomp(L, PURPLE, 16, 0.9)
    C = mat3(lc.M())
    # jump & celebrate
    jump = abs(math.sin(t * math.pi * 2)) * 6
    pose = {"arm_l": 150 + 10 * math.sin(t * 12), "arm_r": -150 - 10 * math.sin(t * 12), "leg_l": 6, "leg_r": -6,
            "head": 3 * math.sin(t * 6)}
    draw_shadow(f.c, cam, 0, 9, 0.4 * (1 - jump / 12))
    L2 = new_layer()
    ney(L2, cam, 0, -NEY_HIP - jump, pose, 1, head="head_smile", front=True)
    f.glowcomp(L2, GOLD, 24, 0.5)
    c = f.done()
    confetti(c, t)
    # foreground silhouettes (depth blur)
    fg = new_layer()
    fc = Cam(cam.cx * 1.4, cam.cy, cam.s * 1.8)
    for i, x in enumerate((-90, 70)):
        ender(fg, fc, x + math.sin(t * 3 + i) * 2, {"arm_l": 150, "arm_r": -150, "leg_l": 0, "leg_r": 0}, 1,
              front=True, tint=0.15, yf=30)
    fg = cv2.GaussianBlur(fg, (0, 0), 10)
    comp(c, fg)
    return c, {"bloom": 0.6}


def s_poster(t):
    """54-60s: the victory poster - Dragon Egg trophy, gold rays, title - then the tag line."""
    cam = Cam(0, -30, keys(t, [(0, 23), (4, 25)]))
    f = Fr(cam, t)
    draw_sky(f.c, cam, t, nebula=1.4, brightness=1.0)
    cx, cy = W / 2, H * 0.55
    light_rays(f.c, cx, cy, t, 22, 1.0, hexc("#ffcf5a"), 2400, 0.05, 0.12, seed=3, alpha=0.22)
    light_rays(f.c, cx, cy, t, 22, 1.0, hexc("#b44bff"), 2400, 0.04, -0.1, seed=4, alpha=0.18)
    # purple floor
    floor_y = cam.to_screen(0, 0)[1]
    k = np.clip((np.arange(H, dtype=np.float32) - floor_y) / 300, 0, 1)[:, None, None]
    f.c = f.c * (1 - k * 0.9) + k * 0.9 * hexc("#4a2a7a") * (1 - 0.4 * k)
    disc(f.g, cx, cy, 380, hexc("#ffb830"), 0.35, add=True, soft=420)
    # hero with the egg trophy
    raise_k = ease_out(inv(0.25, 0.9, t))
    pose = {"arm_l": lerp(20, 168, raise_k), "arm_r": lerp(-20, -168, raise_k), "leg_l": 4, "leg_r": -4,
            "head": 2 * math.sin(t * 2)}
    L = new_layer()
    mats = ney(L, cam, 0, -NEY_HIP, pose, 1, head="head_smile", front=True)
    ex = 0
    ey = lerp(-14, -43, raise_k)
    blit(L, EGG, mat3(cam.M()) @ T(ex, ey) @ S(0.95) @ T(-5, -6))
    f.glowcomp(L, GOLD, 30, 0.7)
    esx, esy = cam.to_screen(ex, ey - 1)
    disc(f.g, esx, esy, 24, hexc("#c040ff"), 0.3 * raise_k, add=True, soft=90)
    burst(f.c, cam, t % 1.2, 0, 14, ex, ey - 6, 141 + int(t / 1.2), speed=(8, 24), life=(0.7, 1.2), size=(0.3, 0.7), alpha=0.6)
    # sparkles
    for i in range(40):
        a = hash01(9, i) * math.tau + t * 0.3
        r = lerp(250, 520, hash01(9, i, 2))
        tw = max(0, math.sin(t * 5 + i * 2.1)) ** 6
        sx, sy = cx + math.cos(a) * r, cy + math.sin(a) * r * 1.2
        line(f.g, (sx - 18 * tw, sy), (sx + 18 * tw, sy), (1, 0.95, 0.7), 3, tw, add=True)
        line(f.g, (sx, sy - 18 * tw), (sx, sy + 18 * tw), (1, 0.95, 0.7), 3, tw, add=True)
    c = f.done()
    # anamorphic flare on the egg
    rect(c, 0, esy - 3, W, esy + 3, hexc("#d88bff"), 0.25 * raise_k, add=True)
    title_slam(c, "NEYMAR JR", W / 2, H * 0.12, t - 0.15, 210, "gold", hold=99)
    title_slam(c, "VS  THE  END", W / 2, H * 0.12 + 185, t - 0.5, 110, "purple", hold=99, spacing=0.06)
    confetti(c, t, 60, 8, 0.8)
    # outro
    fade = inv(3.9, 4.4, t)
    c *= (1 - fade)
    if t > 4.4:
        a = ease_out(inv(4.45, 4.8, t))
        typewriter(c, "THE END...", W / 2, H * 0.46, t - 4.45, 56, 14, alpha=a)
        typewriter(c, "WAS JUST THE BEGINNING.", W / 2, H * 0.52, t - 5.0, 34, 30, style="purple", alpha=a)
    return c, {"flash": 1 - inv(0, 0.3, t), "bloom": 0.7, "fade": inv(5.75, 6.0, t)}


# ================================================================ timeline ===
SHOTS = [
    (0.0, 8.0, s_open),
    (8.0, 10.0, s_portal),
    (10.0, 14.0, s_arrival),
    (14.0, 16.0, s_eyes),
    (16.0, 19.0, s_dribble),
    (19.0, 20.0, s_stare),
    (20.0, 21.0, s_smirk),
    (21.0, 24.5, s_crossover),
    (24.5, 27.0, s_replay),
    (27.0, 29.0, s_rumble),
    (29.0, 32.0, s_flyover),
    (32.0, 34.0, s_landing),
    (34.0, 35.0, s_dragon_eye),
    (35.0, 36.0, s_hero),
    (36.0, 40.0, s_sprint),
    (40.0, 43.0, s_air),
    (43.0, 45.0, s_dunk),
    (45.0, 47.0, s_dunk_replays),
    (47.0, 51.0, s_defeat),
    (51.0, 54.0, s_crowd),
    (54.0, 60.0, s_poster),
]
DURATION = 60.0

# block-dissolve transitions (need both shots): boundary time -> half-width (s)
DISSOLVES = {51.0: 0.3}


def shot_at(T_):
    for (a, b, fn) in SHOTS:
        if a <= T_ < b:
            return a, b, fn
    a, b, fn = SHOTS[-1]
    return a, b, fn


def block_mask(k, seed=77, bs=60):
    gh, gw = H // bs + 1, W // bs + 1
    rng = np.random.default_rng(seed)
    r = rng.random((gh, gw)).astype(np.float32)
    m = (r < k).astype(np.float32)
    return cv2.resize(m, (gw * bs, gh * bs), interpolation=cv2.INTER_NEAREST)[:H, :W, None]


def render_raw(T_):
    """Scene render + per-shot fx for global time T_ (seconds)."""
    a, b, fn = shot_at(T_)
    for bt, hw in DISSOLVES.items():
        if bt - hw <= T_ < bt + hw:
            aa, _, fa = shot_at(bt - 1e-3)
            ab, _, fb = shot_at(bt + 1e-3)
            ca, fxa = fa(T_ - aa)
            cb, fxb = fb(T_ - ab)
            k = (T_ - (bt - hw)) / (2 * hw)
            m = block_mask(k)
            return ca * (1 - m) + cb * m, (fxb if k > 0.5 else fxa)
    return fn(T_ - a)
