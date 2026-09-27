"""Pixel-art textures and articulated rigs (Neymar, Enderman, Ender Dragon, ball)."""
import math

import numpy as np

from core import (Cam, R, S, T, blit, clamp, disc, hash01, hexc, lerp, line, mat3, poly,
                  tex_from_rows)

# ---------------------------------------------------------------- palette ---
P = {
    "s": hexc("#9a6240"), "S": hexc("#74462b"), "t": hexc("#4d2a17"),
    "h": hexc("#f0cf5e"), "H": hexc("#c89a2e"), "d": hexc("#3b2413"),
    "e": hexc("#ffffff"), "p": hexc("#24160d"), "m": hexc("#5c2a19"),
    "y": hexc("#f7d417"), "Y": hexc("#d3a90b"), "g": hexc("#0d8f3c"), "G": hexc("#08622a"),
    "b": hexc("#2257e0"), "B": hexc("#173ea6"), "w": hexc("#f4f4f4"), "W": hexc("#c9c9d4"),
    "k": hexc("#18181c"), "c": hexc("#1fd6ea"), ".": None,
}

NEY = {}


def _ney_textures():
    head_side = [
        "hhhhhhhh",
        "hhhhhhhh",
        "Hhhhhhhh",
        "ddssssss",
        "dSsssseS",
        "dSssssps",
        "SSssssSs",
        "SSsssmms",
    ]
    head_side_face = [  # used when head faces camera more (3/4)
        "hhhhhhhh",
        "hhhhhhhh",
        "Hhhhhhhh",
        "dsssssss",
        "Ssepssep",
        "Sssssss.",
        "SsssSSss",
        "Sssmmmss",
    ]
    head_front = [
        "hhhhhhhh",
        "hhhhhhhh",
        "Hhhhhhhh",
        "dssssssd",
        "sepsseps",
        "sssSSsss",
        "ssmmmmss",
        "SSssssSS",
    ]
    head_front_smile = [
        "hhhhhhhh",
        "hhhhhhhh",
        "Hhhhhhhh",
        "dssssssd",
        "sppsspps",
        "sssSSsss",
        "smeeeems",
        "SSmmmmSS",
    ]
    torso_side = [
        "Yggggy",
        "Yygyyy",
        "Yyyyyy",
        "Yyyyyy",
        "Yyyggy",
        "Yyyggy",
        "Yyyggy",
        "Yyyyyy",
        "Yyyyyy",
        "Yyyyyy",
        "YYYYYY",
        "gggggg",
    ]
    torso_front = [
        "yyggggyy",
        "Yyyggyyy",
        "Yyyyyyyy",
        "YygyggGy",
        "YggygyGy",
        "YygygyGy",
        "YygygyGy",
        "YygygyGy",
        "YygyggGy",
        "Yyyyyyyy",
        "YYYYYYYY",
        "gggggggg",
    ]
    arm = [
        "yyyy",
        "yyyy",
        "Yyyy",
        "gggg",
        "ssss",
        "stts",
        "ssts",
        "ssss",
        "tsst",
        "stts",
        "ssss",
        "SsSs",
    ]
    leg = [
        "bbbb",
        "bbbb",
        "bbbb",
        "bbbb",
        "Bbbb",
        "BBBB",
        "ssss",
        "Ssss",
        "wwww",
        "wWww",
        "kkkk",
        "kcck",
    ]
    NEY["head_side"] = tex_from_rows(head_side, P)
    NEY["head_q"] = tex_from_rows(head_side_face, P)
    NEY["head_front"] = tex_from_rows(head_front, P)
    NEY["head_smile"] = tex_from_rows(head_front_smile, P)
    NEY["torso_side"] = tex_from_rows(torso_side, P)
    NEY["torso_front"] = tex_from_rows(torso_front, P)
    NEY["arm"] = tex_from_rows(arm, P)
    NEY["arm_far"] = tex_from_rows(arm, P, 0.62)
    NEY["leg"] = tex_from_rows(leg, P)
    NEY["leg_far"] = tex_from_rows(leg, P, 0.62)


_ney_textures()

# ----------------------------------------------------------------- ender ---
EP = {
    "K": hexc("#0c0910"), "k": hexc("#18121f"), "j": hexc("#221a2b"),
    "E": hexc("#d65cff"), "F": hexc("#f7b8ff"), "M": hexc("#050307"), ".": None,
}
END = {}


def _ender_textures():
    rng = np.random.default_rng(5)

    def speck(w, h, rows=None):
        rs = []
        for y in range(h):
            r = ""
            for x in range(w):
                v = rng.random()
                r += "j" if v < 0.1 else ("k" if v < 0.45 else "K")
            rs.append(r)
        if rows:
            for y, rr in rows.items():
                rs[y] = rr
        return rs

    END["head_side"] = tex_from_rows(speck(8, 6, {4: "kKKKKEFE"}), EP)
    END["head_front"] = tex_from_rows(speck(8, 6, {4: "EFEkKEFE"}), EP)
    END["jaw_side"] = tex_from_rows(speck(8, 2), EP)
    END["jaw_front"] = tex_from_rows(speck(8, 2), EP)
    END["torso"] = tex_from_rows(speck(4, 12), EP)
    END["torso_front"] = tex_from_rows(speck(8, 12), EP)
    END["limb"] = tex_from_rows(speck(2, 30), EP)
    END["limb_far"] = tex_from_rows(speck(2, 30), EP, 0.6)
    END["mouth"] = tex_from_rows(["MMMMMMMM"] * 2, EP)


_ender_textures()

# ---------------------------------------------------------------- dragon ---
DP = {
    "K": hexc("#141319"), "k": hexc("#1f1d27"), "j": hexc("#2c2a36"), "r": hexc("#3d3a4a"),
    "E": hexc("#d35cff"), "F": hexc("#ffd6ff"), "M": hexc("#07060a"), "w": hexc("#9b98a8"), ".": None,
}
DRA = {}


def _dragon_textures():
    rng = np.random.default_rng(9)

    def scales(w, h, ridge=False):
        rs = []
        for y in range(h):
            r = ""
            for x in range(w):
                v = rng.random()
                c = "j" if v < 0.12 else ("k" if v < 0.5 else "K")
                if (x + y * 3) % 7 == 0 and v > 0.3:
                    c = "j"
                r += c
            rs.append(r)
        if ridge:
            rs[0] = "".join("r" if (x % 4) < 2 else "j" for x in range(w))
        return rs

    DRA["body"] = tex_from_rows(scales(44, 18, True), DP)
    DRA["neck"] = tex_from_rows(scales(10, 9, True), DP)
    DRA["tail"] = tex_from_rows(scales(10, 7, True), DP)
    head = scales(18, 10, True)
    head[3] = head[3][:11] + "EEFE" + head[3][15:]
    head[4] = head[4][:11] + "EEEE" + head[4][15:]
    DRA["head"] = tex_from_rows(head, DP)
    snout = scales(12, 5, True)
    snout[1] = snout[1][:9] + "jj" + snout[1][11]
    DRA["snout"] = tex_from_rows(snout, DP)
    jaw = scales(14, 4)
    jaw[0] = "w.w.w.w.w.w.ww"
    DRA["jaw"] = tex_from_rows(jaw, DP)
    DRA["upper_teeth"] = tex_from_rows(["w.w.w.w.w.w."], DP)
    DRA["leg"] = tex_from_rows(scales(7, 20), DP)
    DRA["leg_far"] = tex_from_rows(scales(7, 20), DP, 0.6)
    DRA["claw"] = tex_from_rows(["wKwKw", "w.w.w"], DP)
    DRA["bone"] = tex_from_rows(scales(56, 4, True), DP)
    DRA["bone2"] = tex_from_rows(scales(66, 3, True), DP)
    DRA["spike"] = tex_from_rows(["..r..", ".rjr.", "rjjjr"], DP)


_dragon_textures()

# ------------------------------------------------------------------- ball ---
BALLP = {"o": hexc("#f07a1a"), "O": hexc("#c4550c"), "l": hexc("#2a1206"), ".": None}
BALL = tex_from_rows([
    "..oloo..",
    ".ooloOo.",
    "llllllll",
    "oOlooloO",
    "oolOoloo",
    "llllllll",
    ".OoloOo.",
    "..olOo..",
], BALLP)


def draw_ball(canvas, cam, x, y, rot=0.0, size=6.0, alpha=1.0):
    """Ball centred at world (x, y), diameter `size` texels."""
    M = cam.M()
    k = size / 8.0
    Mt = mat3(M) @ T(x, y) @ R(rot) @ S(k) @ T(-4, -4)
    blit(canvas, BALL, Mt, alpha)


# ------------------------------------------------------------------- rigs ---
class Rig:
    """Hierarchical 2D rig. parts: name -> dict(tex, pivot, parent, at, z)."""

    def __init__(self, parts):
        self.parts = parts
        self.order = sorted(parts, key=lambda n: parts[n]["z"])

    def draw(self, canvas, cam, x, y, pose, facing=1, scale=1.0, alpha=1.0, tex_over=None,
             tint=None, extra=None):
        """Draw at world (x, y) = root pivot. pose: name->angle (deg), plus 'lean'.
        Returns dict of world matrices (3x3) per part for attachment (e.g. ball in hand)."""
        mats = {}
        base = T(x, y) @ S(facing * scale, scale)

        def world(n):
            if n in mats:
                return mats[n]
            p = self.parts[n]
            ang = pose.get(n, 0.0)
            if p["parent"] is None:
                m = base @ R(ang) @ T(-p["pivot"][0], -p["pivot"][1])
            else:
                pm = world(p["parent"])
                ox, oy = p["at"]
                off = pose.get(n + "_off", (0.0, 0.0))
                m = pm @ T(ox + off[0], oy + off[1]) @ R(ang) @ T(-p["pivot"][0], -p["pivot"][1])
            mats[n] = m
            return m

        C = mat3(cam.M())
        tex_over = tex_over or {}
        for n in self.order:
            if pose.get(n + "_hide"):
                continue
            m = world(n)
            tex = tex_over.get(n, self.parts[n]["tex"])
            blit(canvas, tex, C @ m, alpha, tint=tint)
            if extra and n in extra:
                extra[n](canvas, C @ m, m)
        return mats


def point(m, px, py):
    v = m @ np.array([px, py, 1.0])
    return v[0], v[1]


NEY_SIDE = Rig({
    "arm_far": dict(tex=NEY["arm_far"], pivot=(2, 2), parent="torso", at=(3.2, 2), z=0),
    "leg_far": dict(tex=NEY["leg_far"], pivot=(2, 0), parent="torso", at=(2.4, 12), z=1),
    "torso": dict(tex=NEY["torso_side"], pivot=(3, 12), parent=None, at=(0, 0), z=2),
    "leg": dict(tex=NEY["leg"], pivot=(2, 0), parent="torso", at=(3.6, 12), z=3),
    "head": dict(tex=NEY["head_side"], pivot=(4, 8), parent="torso", at=(3, 0.5), z=4),
    "arm": dict(tex=NEY["arm"], pivot=(2, 2), parent="torso", at=(2.8, 2), z=5),
})

NEY_FRONT = Rig({
    "torso": dict(tex=NEY["torso_front"], pivot=(4, 12), parent=None, at=(0, 0), z=2),
    "leg_l": dict(tex=NEY["leg"], pivot=(2, 0), parent="torso", at=(2, 12), z=1),
    "leg_r": dict(tex=NEY["leg"], pivot=(2, 0), parent="torso", at=(6, 12), z=1),
    "head": dict(tex=NEY["head_front"], pivot=(4, 8), parent="torso", at=(4, 0.3), z=4),
    "arm_l": dict(tex=NEY["arm"], pivot=(2, 2), parent="torso", at=(-2, 2), z=3),
    "arm_r": dict(tex=NEY["arm"], pivot=(2, 2), parent="torso", at=(10, 2), z=3),
})

END_SIDE = Rig({
    "arm_far": dict(tex=END["limb_far"], pivot=(1, 1), parent="torso", at=(2, 1), z=0),
    "leg_far": dict(tex=END["limb_far"], pivot=(1, 0), parent="torso", at=(1.5, 12), z=1),
    "torso": dict(tex=END["torso"], pivot=(2, 12), parent=None, at=(0, 0), z=2),
    "leg": dict(tex=END["limb"], pivot=(1, 0), parent="torso", at=(2.5, 12), z=3),
    "jaw": dict(tex=END["jaw_side"], pivot=(1, 0), parent="head", at=(1, 6), z=4),
    "head": dict(tex=END["head_side"], pivot=(4, 6), parent="torso", at=(2, 0.3), z=5),
    "arm": dict(tex=END["limb"], pivot=(1, 1), parent="torso", at=(2, 1), z=6),
})

END_FRONT = Rig({
    "leg_l": dict(tex=END["limb"], pivot=(1, 0), parent="torso", at=(2, 12), z=1),
    "leg_r": dict(tex=END["limb"], pivot=(1, 0), parent="torso", at=(6, 12), z=1),
    "torso": dict(tex=END["torso_front"], pivot=(4, 12), parent=None, at=(0, 0), z=2),
    "jaw": dict(tex=END["jaw_front"], pivot=(0, 0), parent="head", at=(0, 6), z=3),
    "head": dict(tex=END["head_front"], pivot=(4, 6), parent="torso", at=(4, 0.3), z=4),
    "arm_l": dict(tex=END["limb"], pivot=(1, 1), parent="torso", at=(-1, 1), z=3),
    "arm_r": dict(tex=END["limb"], pivot=(1, 1), parent="torso", at=(9, 1), z=3),
})

NEY_HIP = 12.0   # hip height above ground (texels)
END_HIP = 30.0


# --------------------------------------------------------------- poses -------
def ney_run(ph, amp=1.0, lean=12):
    s = math.sin(ph)
    return {"torso": lean * amp, "leg": 48 * s * amp, "leg_far": -48 * s * amp,
            "arm": -50 * s * amp, "arm_far": 50 * s * amp, "head": -lean * 0.6 * amp}


def ney_idle(t):
    b = math.sin(t * 2.4)
    return {"torso": 2 + b, "leg": 4, "leg_far": -4, "arm": 6 + 2 * b, "arm_far": -6, "head": -1}


def ney_dribble(t, ph, amp=1.0):
    """Running dribble: near arm pumps down each bounce. ph = run phase."""
    p = ney_run(ph, 0.8 * amp, 14)
    p["arm"] = -38 + 22 * math.cos(ph * 2)  # dribbling arm forward
    return p


def bob(ph, amp=1.0):
    return -abs(math.sin(ph)) * 1.4 * amp


def end_idle(t):
    s = math.sin(t * 1.3)
    return {"torso": 1 * s, "leg": 3, "leg_far": -3, "arm": 4 + 2 * s, "arm_far": -4 - 2 * s,
            "head": 0, "jaw": 0}


# --------------------------------------------------------------- dragon -----
def draw_dragon(canvas, cam, x, y, t, facing=-1, scale=1.0, flap=None, head_dip=0.0, jaw=0.0,
                neck_raise=0.0, wing_spread=1.0, alpha=1.0, eye_glow=1.0, legs_down=1.0, glow_layer=None):
    """Side-view Ender Dragon. (x, y) = body centre in world texels."""
    C = mat3(cam.M())
    base = T(x, y) @ S(facing * scale, scale)
    ph = t * 2 * math.pi * 0.9 if flap is None else flap
    wa = math.sin(ph)  # wing angle driver

    def dblit(tex, m, a=alpha, tint=None):
        blit(canvas, tex, C @ m, a, tint=tint)

    # wings: u=0 raised (up/forward), u=1 swept back/down
    u = (1 - wa) / 2 if wing_spread >= 0 else 0.0

    def wing(far):
        shade = 0.55 if far else 1.0
        a1 = lerp(-72, -178, u) + (6 if far else 0)
        m1 = base @ T(6, -8) @ R(a1)
        ex, ey = point(m1, 56, 0)
        a2 = a1 - lerp(38, 8, u)
        m2 = T(ex, ey) @ S(facing * scale, scale) @ R(a2)
        pts_w = [point(base, 22, -8), (ex, ey), point(m2, 66, 0), point(m2, 46, -26 - 8 * u),
                 point(m2, 18, -40), point(m1, 30, -40), point(base, -20, -8)]
        pts = [cam.to_screen(*p) for p in pts_w]
        poly(canvas, pts, hexc("#221b30") * shade, alpha * 0.97)
        for f in (0.35, 0.7):
            p0 = point(m2, 66 * f, 0)
            p1 = point(m1, 48 * f + 6, -38)
            line(canvas, cam.to_screen(*p0), cam.to_screen(*p1), hexc("#2e2a3b") * shade,
                 max(1.5, cam.s * 1.2 * scale), alpha)
        dblit(DRA["bone"], m1 @ T(0, -2), tint=shade)
        dblit(DRA["bone2"], m2 @ T(0, -1.5), tint=shade)

    wing(True)

    # tail (chain with sine wave)
    tm = base @ T(-22, 0)
    for i in range(10):
        ang = 8 * math.sin(t * 2.2 - i * 0.5) + (4 if i > 5 else 0)
        tm = tm @ R(ang)
        sc = 1 - i * 0.06
        dblit(DRA["tail"], tm @ S(-1, sc) @ T(0, -3.5))
        if i % 2 == 0:
            dblit(DRA["spike"], tm @ S(-1, sc) @ T(3, -6.5))
        tm = tm @ T(-9 * sc, 0)

    # far legs
    for dx, tex in ((-14, DRA["leg_far"]), (14, DRA["leg_far"])):
        lm = base @ T(dx, 6) @ R(10 - 20 * (1 - legs_down))
        dblit(tex, lm @ T(-3.5, 0))

    # body
    dblit(DRA["body"], base @ T(-22, -9))
    for i in range(6):
        dblit(DRA["spike"], base @ T(-18 + i * 7, -12))

    # near legs
    for dx in (-12, 16):
        lm = base @ T(dx, 7) @ R(-8 + 25 * (1 - legs_down))
        dblit(DRA["leg"], lm @ T(-3.5, 0))
        dblit(DRA["claw"], lm @ T(-2.5, 19.5))

    # neck chain forward/up
    nm = base @ T(20, -4)
    neck_ang = [-28 - 10 * neck_raise, -14 - 6 * neck_raise, 4, 12 + 8 * head_dip, 16 + 12 * head_dip]
    for i, a in enumerate(neck_ang):
        a += 3 * math.sin(t * 1.8 - i * 0.6)
        nm = nm @ R(a / 2)
        dblit(DRA["neck"], nm @ T(-1, -4.5))
        dblit(DRA["spike"], nm @ T(2, -7.5))
        nm = nm @ R(a / 2) @ T(8, 0)
    # head
    hm = nm @ R(10 + 20 * head_dip)
    dblit(DRA["head"], hm @ T(-2, -6))
    dblit(DRA["snout"], hm @ T(14, -3))
    jm = hm @ T(10, 3.5) @ R(jaw * 32)
    dblit(DRA["jaw"], jm @ T(0, 0))
    dblit(DRA["spike"], hm @ T(0, -9))
    dblit(DRA["spike"], hm @ T(4, -9))
    eye = point(hm, 11, -2.5)
    if glow_layer is not None and eye_glow > 0:
        sx, sy = cam.to_screen(*eye)
        disc(glow_layer, sx, sy, cam.s * scale * 1.3, hexc("#c040ff"), 0.8 * eye_glow * alpha, add=True, soft=cam.s * scale * 4)

    wing(False)
    return {"head": hm, "jaw": jm, "base": base, "mouth": point(hm, 22, 3)}

