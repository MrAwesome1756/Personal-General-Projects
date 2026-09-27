"""Fully synthesized score + sound design for the 60s cut (no samples, no licensed music).

120 BPM, D minor (i-VI-iv-V) for the fight, D major / bVI-bVII-I lift for the victory.
Every SFX is placed on the same timestamps as the picture in shots.py.

  python audio.py out.wav
"""
import math
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, sosfilt, sosfilt_zi

SR = 44100
DUR = 60.0
N = int(SR * DUR)
BEAT = 0.5
BAR = 2.0
RNG = np.random.default_rng(1234)


def hz(note):
    """'D3', 'A#2', 'Bb2' -> Hz."""
    names = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    n = names[note[0]]
    i = 1
    if note[1] in "#b":
        n += 1 if note[1] == "#" else -1
        i = 2
    octv = int(note[i:])
    return 440.0 * 2 ** ((n + 12 * (octv + 1) - 69) / 12)


def tt(d):
    return np.arange(int(d * SR)) / SR


def env(n, a=0.005, d=0.1, s=0.0, r=0.05, hold=None):
    """ADSR over n samples (hold = sustain time before release)."""
    t = np.arange(n) / SR
    A = max(a, 1e-4)
    e = np.where(t < A, t / A, 1.0)
    if hold is None:
        dec = s + (1 - s) * np.exp(-(t - A) / max(d, 1e-4))
        e = np.where(t < A, e, dec)
    else:
        dec = s + (1 - s) * np.exp(-(t - A) / max(d, 1e-4))
        e = np.where(t < A, e, dec)
        rel_start = A + hold
        e = np.where(t > rel_start, e * np.exp(-(t - rel_start) / max(r, 1e-4)), e)
    return e


def lp(x, fc, order=2):
    fc = min(fc, SR / 2 * 0.95)
    return sosfilt(butter(order, fc, "low", fs=SR, output="sos"), x)


def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, "high", fs=SR, output="sos"), x)


def bp(x, f0, f1, order=2):
    return sosfilt(butter(order, [f0, min(f1, SR / 2 * 0.95)], "band", fs=SR, output="sos"), x)


def sweep_filter(x, fcs, kind="low", block=512, q_order=2):
    """Time-varying filter; fcs = callable(t_seconds_array_start) -> cutoff (or (lo, hi) for band)."""
    out = np.zeros_like(x)
    zi = None
    for i in range(0, len(x), block):
        fc = fcs(i / SR)
        if kind == "band":
            lo, hi = fc
            sos = butter(q_order, [max(20, lo), min(hi, SR / 2 * 0.95)], "band", fs=SR, output="sos")
        else:
            sos = butter(q_order, min(max(fc, 20), SR / 2 * 0.95), kind, fs=SR, output="sos")
        if zi is None or zi.shape[0] != sos.shape[0]:
            zi = sosfilt_zi(sos) * 0
        out[i:i + block], zi = sosfilt(sos, x[i:i + block], zi=zi)
    return out


def saw(f, t, phase=0.0):
    ph = np.cumsum(np.broadcast_to(f, t.shape) / SR) + phase if np.ndim(f) else f * t + phase
    return 2 * (ph % 1.0) - 1


def sq(f, t, duty=0.5):
    ph = f * t
    return np.where((ph % 1.0) < duty, 1.0, -1.0)


def sine_sweep(f0, f1, d, curve=2.0):
    t = tt(d)
    k = (t / d) ** (1 / curve) if f1 < f0 else (t / d) ** curve
    f = f0 + (f1 - f0) * k
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def noise(d):
    return RNG.standard_normal(int(d * SR))


def softclip(x, drive=1.0):
    return np.tanh(x * drive) / np.tanh(drive)


# ================================================================ SFX =====
def kick(d=0.45, punch=1.0):
    t = tt(d)
    f = 45 + 120 * np.exp(-t / 0.03)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.18)
    click = hp(noise(d), 3000) * np.exp(-t / 0.004) * 0.3
    return softclip((s + click) * 1.3 * punch, 1.5)


def clap(d=0.35):
    t = tt(d)
    n = bp(noise(d), 900, 5000)
    e = np.zeros_like(t)
    for o in (0.0, 0.011, 0.023):
        e += np.where(t >= o, np.exp(-(t - o) / 0.009), 0)
    e += np.where(t >= 0.03, np.exp(-(t - 0.03) / 0.09), 0) * 0.7
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05) * 0.3
    return (n * e * 0.5 + body)


def hat(d=0.06, open_=False):
    dd = 0.28 if open_ else d
    t = tt(dd)
    return hp(noise(dd), 7000) * np.exp(-t / (0.09 if open_ else 0.018)) * 0.35


def eight08(f, d=0.9, glide_from=None):
    t = tt(d)
    fr = np.full_like(t, f)
    if glide_from:
        fr = f + (glide_from - f) * np.exp(-t / 0.06)
    s = np.sin(2 * np.pi * np.cumsum(fr) / SR)
    s = softclip(s * 2.2, 1.8) * np.exp(-t / (d * 0.55)) * np.minimum(1, t / 0.004)
    return s * 0.8


def pad(freqs, d, cutoff=1400, attack=0.6, release=0.8, detune=0.012):
    t = tt(d + release)
    x = np.zeros_like(t)
    for f in freqs:
        for k in (-1, 0, 1):
            x += saw(f * (1 + detune * k), t, RNG.random())
    x = lp(x / (len(freqs) * 3), cutoff, 2)
    e = np.minimum(1, t / attack) * np.where(t > d, np.exp(-(t - d) / (release / 3)), 1)
    return x * e


def bell(f, d=1.0):
    t = tt(d)
    x = (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 2.0 * t) * np.exp(-t / 0.2)
         + 0.25 * np.sin(2 * np.pi * f * 3.01 * t) * np.exp(-t / 0.08))
    return x * np.exp(-t / 0.35) * np.minimum(1, t / 0.002) * 0.4


def lead(f, d, vib=5.5):
    t = tt(d + 0.15)
    fm = f * (1 + 0.006 * np.sin(2 * np.pi * vib * t) * np.minimum(1, t / 0.25))
    x = 0.6 * saw(fm, t) + 0.4 * saw(fm * 1.005, t, 0.3) + 0.3 * np.sin(2 * np.pi * np.cumsum(fm / 2) / SR)
    x = lp(x, 2600)
    e = np.minimum(1, t / 0.01) * np.where(t > d, np.exp(-(t - d) / 0.05), 1) * (0.8 + 0.2 * np.exp(-t / 0.2))
    return x * e * 0.35


def impact(d=2.2, big=1.0):
    t = tt(d)
    f = 30 + 70 * np.exp(-t / 0.12)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (0.5 * big))
    crack = lp(noise(d), 5000) * np.exp(-t / 0.06)
    body = lp(noise(d), 400) * np.exp(-t / 0.35) * 0.8
    return softclip((boom * 1.2 + crack * 0.6 + body) * big, 2.0)


def whoosh(d=0.7, f0=300, f1=3000, peak=0.6):
    n = noise(d)
    x = sweep_filter(n, lambda s: (f0 + (f1 - f0) * (s / d), (f0 + (f1 - f0) * (s / d)) * 2.2), "band")
    t = tt(d)
    e = np.where(t < d * peak, (t / (d * peak)) ** 2, np.exp(-(t - d * peak) / (d * 0.15)))
    return x * e * 1.6


def riser(d=2.0, f0=200, f1=6000):
    n = noise(d)
    x = sweep_filter(n, lambda s: (f0 * (f1 / f0) ** (s / d), f0 * (f1 / f0) ** (s / d) * 1.8), "band")
    t = tt(d)
    tone = saw(80 * 2 ** (3 * t / d), t) * 0.15
    return (x * 1.4 + lp(tone, 3000)) * (t / d) ** 2


def vworp(d=0.9):
    """Ender teleport."""
    t = tt(d)
    f = 1400 * np.exp(-t / 0.18) + 160
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * (0.6 + 0.4 * np.sin(2 * np.pi * 28 * t))
    x += bp(noise(d), 600, 3000) * 0.3 * np.exp(-t / 0.1)
    return x * np.exp(-t / 0.35) * np.minimum(1, t / 0.005)


def screech(d=0.8):
    t = tt(d)
    f = 700 + 500 * np.sin(2 * np.pi * 9 * t) * 0.3 + 300 * np.exp(-t / 0.2)
    x = saw(f, t) * np.sin(2 * np.pi * 37 * t)
    x = bp(x, 400, 4000) + bp(noise(d), 1500, 6000) * 0.4
    return softclip(x * 1.5, 2) * np.minimum(1, t / 0.02) * np.exp(-t / 0.45)


def roar(d=1.4, pitch=1.0):
    t = tt(d)
    f = (70 + 25 * np.sin(2 * np.pi * 1.3 * t)) * pitch
    src = saw(f, t) * (0.6 + 0.4 * np.sin(2 * np.pi * 31 * t)) + noise(d) * 0.8
    x = bp(src, 150 * pitch, 600) * 1.2 + bp(src, 700, 1400) * 0.7 + bp(src, 2000, 3500) * 0.25
    e = np.minimum(1, t / 0.12) * np.exp(-np.maximum(0, t - d * 0.6) / 0.25)
    return softclip(x * e * 2.5, 2.5)


def click(d=0.25):
    t = tt(d)
    return (np.sin(2 * np.pi * 3200 * t) * np.exp(-t / 0.006) * 0.5 +
            np.sin(2 * np.pi * 140 * t) * np.exp(-t / 0.04) + hp(noise(d), 4000) * np.exp(-t / 0.003) * 0.4)


def shimmer(d=2.0, n=24):
    t = tt(d)
    x = np.zeros_like(t)
    for i in range(n):
        f = 1200 + RNG.random() * 5000
        o = RNG.random() * 0.4
        x += np.sin(2 * np.pi * f * t) * np.where(t > o, np.exp(-(t - o) / (0.2 + RNG.random() * 0.6)), 0)
    return x / n * 2.5


def bounce(d=0.25):
    t = tt(d)
    f = 60 + 90 * np.exp(-t / 0.02)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.07) + \
        np.sin(2 * np.pi * 420 * t) * np.exp(-t / 0.02) * 0.15


def shatter(d=1.4, n=90):
    t = tt(d)
    x = hp(noise(d), 2500) * np.exp(-t / 0.12) * 0.5
    for i in range(n):
        o = RNG.random() ** 2 * d * 0.7
        f = 2000 + RNG.random() * 7000
        seg = t - o
        x += np.where(seg > 0, np.sin(2 * np.pi * f * seg) * np.exp(-seg / 0.02), 0) * 0.25
    return x


def heartbeat():
    t = tt(0.6)
    one = lambda o, a: np.where(t > o, np.sin(2 * np.pi * (48 + 30 * np.exp(-(t - o) / 0.02)) * (t - o)) * np.exp(-(t - o) / 0.09), 0) * a
    return softclip((one(0, 1.0) + one(0.18, 0.75)) * 1.6, 1.5)


def blip(f=880, d=0.035):
    t = tt(d)
    return sq(f, t, 0.5) * np.exp(-t / 0.02) * 0.25


def rewind(d=0.45):
    t = tt(d)
    f = 200 + 3000 * (t / d) ** 2 + 300 * np.sin(2 * np.pi * 18 * t)
    return bp(saw(f, t) * 0.5 + noise(d) * 0.4, 300, 6000) * 0.7


def scratch(d=0.35):
    t = tt(d)
    fc = lambda s: (500 + 2500 * abs(math.sin(s * 30)), (500 + 2500 * abs(math.sin(s * 30))) * 1.8)
    return sweep_filter(noise(d), fc, "band", 128) * np.exp(-t / 0.2) * 2.0


def shutter():
    t = tt(0.15)
    x = hp(noise(0.15), 2000)
    return x * (np.exp(-t / 0.005) + np.where(t > 0.06, np.exp(-(t - 0.06) / 0.006), 0)) * 0.6


def rumble(d):
    t = tt(d)
    x = lp(noise(d), 90, 4) * 3 + lp(noise(d), 300) * 0.3
    return x * (t / d) ** 1.5


def hum(d, f=110):
    t = tt(d)
    x = sum(np.sin(2 * np.pi * f * m * (1 + 0.003 * np.sin(2 * np.pi * 3 * t)) * t) / m for m in (1, 2, 3, 5))
    return x * (t / d) * 0.3


def crowd(d):
    t = tt(d)
    x = np.zeros_like(t)
    for i in range(10):
        lo = 300 + RNG.random() * 800
        layer = bp(noise(d), lo, lo * 2.5)
        am = 0.6 + 0.4 * np.sin(2 * np.pi * (0.3 + RNG.random() * 0.7) * t + RNG.random() * 6)
        x += layer * am
    x += bp(noise(d), 2000, 5000) * 0.3
    return x / 5 * np.minimum(1, t / 0.4) * np.exp(-np.maximum(0, t - d + 0.8) / 0.3)


def xp_ping(f):
    t = tt(0.3)
    return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2 * t)) * np.exp(-t / 0.06) * 0.3


def orch_hit(freqs, d=1.2):
    x = pad(freqs, 0.05, cutoff=5000, attack=0.003, release=d, detune=0.02)
    n = len(x)
    tn = np.arange(n) / SR
    x = x + lp(RNG.standard_normal(n), 2000) * np.exp(-tn / 0.05) * 0.3
    return softclip(x * 3, 1.5)


def choir(freqs, d, cutoff=2200):
    t = tt(d + 1.0)
    x = np.zeros_like(t)
    for f in freqs:
        for k in range(5):
            fm = f * (1 + 0.01 * (k - 2)) * (1 + 0.004 * np.sin(2 * np.pi * (4.5 + k * 0.3) * t))
            x += saw(fm, t, RNG.random())
    x = bp(x, 250, 900) * 1.2 + bp(x, 1000, cutoff) * 0.6
    e = np.minimum(1, t / 0.8) * np.where(t > d, np.exp(-(t - d) / 0.35), 1)
    return x * e / (len(freqs) * 5) * 3


# ================================================================ bus ======
class Bus:
    def __init__(self):
        self.b = np.zeros((N, 2))

    def add(self, x, at, gain=1.0, pan=0.0):
        i = int(at * SR)
        if i >= N or len(x) == 0:
            return
        x = np.asarray(x)
        if i < 0:
            x, i = x[-i:], 0
        x = x[:N - i]
        l = math.cos((pan + 1) * math.pi / 4) * math.sqrt(2)
        r = math.sin((pan + 1) * math.pi / 4) * math.sqrt(2)
        self.b[i:i + len(x), 0] += x * gain * l
        self.b[i:i + len(x), 1] += x * gain * r


def reverb(buf, decay=2.2, mix=0.25, pre=0.02):
    n = int(decay * SR)
    t = np.arange(n) / SR
    out = np.zeros_like(buf)
    for ch in range(2):
        ir = RNG.standard_normal(n) * np.exp(-t / (decay / 5))
        ir = lp(ir, 6000)
        ir[: int(pre * SR)] = 0
        ir /= np.sqrt((ir ** 2).sum())
        out[:, ch] = fftconvolve(buf[:, ch], ir)[: len(buf)]
    return buf * (1 - mix) + out * mix * 2.2


# ================================================================ score ====
MINOR = [("D", ["D3", "F3", "A3"], "D2"), ("Bb", ["Bb2", "D3", "F3"], "Bb1"),
         ("Gm", ["G2", "Bb2", "D3"], "G1"), ("A", ["A2", "C#3", "E3"], "A1")]
MAJOR = [("D", ["D3", "F#3", "A3"], "D2"), ("Bb", ["Bb2", "D3", "F3"], "Bb1"),
         ("C", ["C3", "E3", "G3"], "C2"), ("D", ["D3", "F#3", "A3"], "D2")]


def chord_at(t, prog=MINOR):
    return prog[int(t // BAR) % 4]


def drums(m, t0, t1, style="trap", gain=1.0):
    step = BEAT / 4
    t = t0
    while t < t1 - 1e-6:
        s = int(round((t % BAR) / step)) % 16
        bar = int(t // BAR)
        if style == "trap":
            if s in (0, 10):
                m.add(kick(), t, 0.95 * gain)
            if s == 8:
                m.add(clap(), t, 0.55 * gain)
            if s % 2 == 0:
                m.add(hat(), t, 0.3 * gain, 0.25)
            if bar % 2 == 1 and s >= 12:
                m.add(hat(0.03), t + step / 2, 0.22 * gain, 0.25)
            if s == 14:
                m.add(hat(open_=True), t, 0.18 * gain, -0.2)
        elif style == "drop":
            if s in (0, 6, 10):
                m.add(kick(punch=1.2), t, 1.0 * gain)
            if s in (8,):
                m.add(clap(), t, 0.65 * gain)
                m.add(clap(), t + 0.012, 0.3 * gain, 0.4)
            m.add(hat(), t, (0.28 if s % 2 == 0 else 0.16) * gain, 0.3)
            if s in (3, 11):
                m.add(hat(open_=True), t, 0.15 * gain, -0.3)
        elif style == "lofi":
            if s in (0, 7, 10):
                m.add(kick(), t, 0.7 * gain)
            if s in (4, 12):
                m.add(clap(), t, 0.4 * gain)
            if s % 2 == 0:
                m.add(hat(), t, 0.18 * gain)
        t += step


def bassline(m, t0, t1, prog=MINOR, gain=1.0, pattern=(0, 1.25)):
    t = t0
    while t < t1 - 1e-6:
        _, _, root = chord_at(t, prog)
        for off in pattern:
            if t + off < t1:
                m.add(eight08(hz(root), 0.7), t + off, 0.75 * gain)
        t += BAR


def pads(m, t0, t1, prog=MINOR, gain=1.0, cutoff=1400):
    t = t0 - (t0 % BAR) if t0 % BAR else t0
    while t < t1 - 1e-6:
        _, notes, _ = chord_at(t, prog)
        start = max(t, t0)
        d = min(t + BAR, t1) - start
        m.add(pad([hz(n) for n in notes], d, cutoff, attack=0.3, release=0.6), start, 0.55 * gain, 0)
        t += BAR


def arps(m, t0, t1, prog=MINOR, gain=1.0, rate=4):
    step = BEAT / rate * 2
    t = t0
    i = 0
    while t < t1 - 1e-6:
        _, notes, _ = chord_at(t, prog)
        seq = [notes[0], notes[1], notes[2], notes[1]]
        n = seq[i % 4]
        f = hz(n) * (4 if (i // 4) % 2 == 0 else 8)
        m.add(bell(f, 0.6), t, 0.22 * gain, 0.35 * math.sin(i * 1.3))
        i += 1
        t += step


LEAD_MOTIF = [("D5", 0, 0.5), ("F5", 0.5, 0.5), ("A5", 1.0, 0.25), ("G5", 1.25, 0.25), ("F5", 1.5, 0.5),
              ("E5", 2.0, 0.5), ("C#5", 2.5, 0.5), ("D5", 3.0, 0.75), ("A4", 3.75, 0.25)]


def leads(m, t0, t1, gain=1.0, motif=LEAD_MOTIF, span=4.0):
    t = t0
    while t < t1 - 1e-6:
        for n, o, d in motif:
            if t + o < t1:
                m.add(lead(hz(n), d * 0.95), t + o, 0.5 * gain, -0.1)
        t += span


def build_music():
    m = Bus()
    # 0-8 intro: drone + pad swell + sub pulses
    m.add(pad([hz("D2"), hz("A2"), hz("D3")], 8.0, 600, attack=3.0, release=1.0), 0.0, 0.9)
    m.add(pad([hz("F3"), hz("A3")], 4.0, 900, attack=2.0, release=1.5), 4.0, 0.5)
    for b in (2.0, 4.0, 6.0):
        m.add(eight08(hz("D1"), 1.6), b, 0.6)
    # 10-14 arrival: arps + pad, drums from the title slam at 12.0
    pads(m, 10.0, 14.0, gain=0.8, cutoff=1800)
    arps(m, 10.0, 14.0, gain=0.9)
    drums(m, 12.0, 14.0, "trap")
    bassline(m, 12.0, 14.0)
    # 14-16 eyes: sparse drone
    m.add(pad([hz("D2"), hz("G#2")], 2.0, 500, attack=0.2, release=0.5), 14.0, 0.8)
    # 16-19 dribble groove
    drums(m, 16.0, 19.0, "trap")
    bassline(m, 16.0, 19.0)
    pads(m, 16.0, 19.0, gain=0.7)
    arps(m, 16.0, 19.0, gain=0.7)
    # 19-21 stare/smirk: tension drone
    m.add(pad([hz("D2"), hz("Eb3"), hz("A3")], 2.0, 1200, attack=0.05, release=0.4), 19.0, 0.8)
    # 21-23.5 crossover (filtered during slow-mo below)
    drums(m, 21.0, 23.5, "trap")
    bassline(m, 21.0, 23.5)
    pads(m, 21.0, 23.5, gain=0.7)
    arps(m, 21.0, 23.5, gain=0.6)
    # 24.9-27 replay: lofi version
    drums(m, 24.9, 27.0, "lofi")
    pads(m, 24.9, 27.0, gain=0.6, cutoff=900)
    # 27-36: boss section - no beat; drones and hits
    m.add(pad([hz("D1"), hz("D2"), hz("Ab2")], 5.0, 400, attack=1.5, release=1.0), 27.0, 1.0)
    m.add(orch_hit([hz("D2"), hz("A2"), hz("D3"), hz("F3")], 2.0), 29.9, 0.9)
    m.add(choir([hz("D3"), hz("F3"), hz("Bb3")], 2.0), 30.0, 0.55)
    m.add(orch_hit([hz("Bb1"), hz("F2"), hz("Bb2"), hz("D3")], 2.0), 32.75, 0.9)
    m.add(choir([hz("D3"), hz("G3"), hz("Bb3")], 2.2), 32.8, 0.5)
    m.add(pad([hz("D2"), hz("A2"), hz("Eb3")], 1.0, 900, attack=0.05, release=0.3), 34.0, 0.7)
    # 35-36: build (snare roll)
    for i in range(16):
        tt_ = 35.0 + i * (0.85 / 16)
        m.add(clap(0.2), tt_, 0.15 + 0.35 * i / 16)
    # 36-40 DROP
    drums(m, 36.0, 40.0, "drop")
    bassline(m, 36.0, 40.0, pattern=(0, 0.75, 1.25, 1.75), gain=1.1)
    pads(m, 36.0, 40.0, gain=0.6, cutoff=2400)
    leads(m, 36.0, 40.0)
    # 40-43 slow-mo: suspended chord
    m.add(pad([hz("D3"), hz("A3"), hz("E4")], 3.0, 700, attack=0.1, release=0.6), 40.0, 0.8)
    m.add(choir([hz("D4"), hz("A4")], 3.0, 1800), 40.0, 0.35)
    # 43.4-45 drop again after the dunk
    drums(m, 43.5, 45.0, "drop")
    bassline(m, 43.5, 45.0, pattern=(0, 0.75, 1.25), gain=1.1)
    leads(m, 43.5, 45.0, gain=0.9)
    drums(m, 45.0, 47.0, "lofi", 0.8)
    bassline(m, 45.0, 47.0, gain=0.7)
    # 47-51 defeat: rising choir into the major key
    m.add(choir([hz("D3"), hz("F3"), hz("A3")], 1.4), 47.0, 0.6)
    m.add(choir([hz("Bb2"), hz("D3"), hz("F3")], 1.4), 48.4, 0.7)
    m.add(riser(1.35, 150, 8000), 48.4, 0.35)
    m.add(orch_hit([hz("D2"), hz("F#2"), hz("A2"), hz("D3"), hz("F#3")], 2.5), 49.75, 1.0)
    m.add(choir([hz("D3"), hz("F#3"), hz("A3")], 1.3), 49.75, 0.7)
    # 51-54 victory groove (major)
    drums(m, 51.0, 54.0, "drop", 0.95)
    bassline(m, 51.0, 54.0, prog=MAJOR, pattern=(0, 0.75, 1.25))
    pads(m, 51.0, 54.0, prog=MAJOR, gain=0.7, cutoff=2600)
    leads(m, 51.0, 54.0, 0.85, motif=[("A5", 0, 0.5), ("F#5", 0.5, 0.5), ("D5", 1.0, 0.5), ("E5", 1.5, 0.5),
                                      ("F5", 2.0, 0.75), ("G5", 2.75, 0.25), ("A5", 3.0, 1.0)])
    # 54-60 poster: final chord
    m.add(orch_hit([hz("D2"), hz("A2"), hz("D3"), hz("F#3"), hz("A3")], 2.5), 54.0, 1.0)
    m.add(choir([hz("D3"), hz("F#3"), hz("A3"), hz("D4")], 3.6), 54.0, 0.7)
    m.add(pad([hz("D2"), hz("A2"), hz("D3"), hz("F#3")], 3.9, 2200, attack=0.05, release=1.2), 54.0, 0.7)
    arps(m, 54.0, 58.0, MAJOR, 0.7)
    drums(m, 54.0, 58.0, "trap", 0.7)
    m.add(eight08(hz("D1"), 2.5), 59.0, 0.8)
    # filter automation: slow-mo + replay + eye/air moments get muffled
    def cutoff(s):
        if 21.8 <= s < 23.0 or 40.0 <= s < 43.0:
            return 420
        if 24.9 <= s < 27.0 or 45.0 <= s < 47.0:
            return 1600
        return 16000
    for ch in range(2):
        m.b[:, ch] = sweep_filter(m.b[:, ch], cutoff, "low", 1024)
    return m


def build_sfx():
    s = Bus()
    # ACT 1 ----------------------------------------------------------------
    s.add(rumble(8.0) * 0.4, 0.0, 0.5)
    s.add(heartbeat(), 0.2, 0.7)
    s.add(heartbeat(), 1.4, 0.6)
    for i in range(len("THE END.")):
        s.add(blip(660), 0.4 + i / 10, 0.6)
    for i in range(len("NO ONE HAS EVER")):
        s.add(blip(990), 1.5 + i / 18, 0.4, 0.2)
    for i in range(len("SCORED HERE.")):
        s.add(blip(990), 2.2 + i / 18, 0.4, 0.2)
    s.add(roar(2.0, 0.7) * 0.5, 5.3, 0.35, 0.4)             # distant dragon
    s.add(riser(2.0), 6.0, 0.35)
    s.add(impact(1.2, 0.5), 8.0, 0.6)
    for i in range(14):
        s.add(click(), 8.0 + i * 0.1, 0.5, (i % 2) * 0.4 - 0.2)
    s.add(impact(2.5, 1.0), 9.5, 0.9)
    s.add(shimmer(2.5), 9.5, 0.7)
    s.add(shimmer(1.5), 10.25, 0.6)
    s.add(vworp(0.7) * 0.5, 10.25, 0.4)
    for i in range(4):
        s.add(bounce() * 0.35, 11.0 + i * 0.28, 0.25)       # footsteps
    s.add(whoosh(0.3, 800, 5000), 11.7, 0.4)
    s.add(impact(2.0, 1.0), 12.0, 0.9)
    s.add(impact(1.0, 0.5), 12.25, 0.5)
    # ACT 2 ----------------------------------------------------------------
    s.add(shatter(0.5, 30) * 0.5, 14.0, 0.5)                 # glitch cut
    from shots import EYE_POS
    for i, (x, sc, t0) in enumerate(EYE_POS):
        s.add(vworp(0.5) * 0.4 * sc, 14.0 + t0, 0.4, max(-0.8, min(0.8, x / 150)))
    s.add(whoosh(0.35, 400, 4000), 15.8, 0.5)
    b = 16.0
    while b < 18.6:
        s.add(bounce(), b, 0.8, -0.1)
        b += BEAT
    s.add(vworp(1.0), 18.5, 0.9, 0.3)
    s.add(impact(1.2, 0.5), 18.5, 0.5)
    s.add(screech(0.9), 19.2, 0.8)
    s.add(impact(1.0, 0.4), 19.0, 0.4)
    s.add(whoosh(0.3, 600, 6000), 19.85, 0.6, -0.5)
    s.add(whoosh(0.3, 600, 6000), 20.9, 0.6, 0.5)
    for i, tb in enumerate((21.0, 21.4)):
        s.add(bounce(), tb, 0.8)
    # slow-mo crossover
    s.add(whoosh(1.2, 200, 1200), 21.8, 0.8)
    s.add(heartbeat(), 22.1, 0.9)
    s.add(heartbeat(), 22.7, 0.9)
    s.add(whoosh(0.4, 500, 4000), 23.0, 0.8, 0.3)
    s.add(impact(2.0, 1.0), 23.36, 1.0)                      # body hits the floor
    s.add(scratch(), 23.45, 0.5)
    s.add(shutter(), 23.5, 0.7)
    s.add(impact(1.2, 0.8), 23.55, 0.8)
    s.add(impact(1.2, 0.8), 23.68, 0.8)
    s.add(shatter(0.9, 40), 23.55, 0.35)
    s.add(rewind(0.45), 24.5, 0.8)
    s.add(impact(1.5, 0.6), 25.9, 0.5)                       # replay fall
    # ACT 3 ----------------------------------------------------------------
    s.add(rumble(2.2), 27.0, 1.3)
    s.add(hum(2.2, 110), 27.4, 0.8)
    s.add(hum(2.2, 164.8), 27.8, 0.5)
    s.add(whoosh(0.4, 300, 3000), 28.8, 0.6)
    for k in range(4):
        s.add(bounce() * 0.9, 29.2 + k * 1.14, 0.7 * (0.5 + k * 0.2))     # wing flaps
    s.add(impact(1.4, 0.7), 29.9, 0.7)
    s.add(roar(1.8), 30.4, 1.0)
    s.add(whoosh(1.2, 100, 1500, 0.7), 30.8, 1.2)
    s.add(impact(3.0, 1.5), 32.75, 1.1)
    s.add(shatter(0.8, 30) * 0.6, 32.8, 0.4)
    s.add(roar(1.1, 0.85), 33.05, 1.0)
    s.add(roar(1.0, 0.6) * 0.5, 34.0, 0.6)                   # growl on the eye
    s.add(whoosh(0.25, 700, 6000), 34.95, 0.6)
    s.add(riser(1.0, 300, 9000), 35.0, 0.6)
    s.add(impact(2.0, 1.2), 36.0, 1.0)
    for k in range(8):
        s.add(bounce(), 36.0 + k * 0.25, 0.55)
    s.add(clap(0.2) * 0.8, 36.95, 0.9)                       # jaw snap
    s.add(roar(0.6, 1.2), 36.9, 0.6)
    s.add(whoosh(0.4, 500, 5000), 37.0, 0.8, -0.3)
    s.add(whoosh(0.35, 700, 6000), 37.75, 0.7, 0.3)
    s.add(bounce(), 38.1, 0.8)
    s.add(bounce(), 38.35, 0.6)
    s.add(bounce(), 38.6, 0.7)
    s.add(bounce(), 38.85, 0.6)
    s.add(whoosh(0.9, 200, 6000, 0.4), 39.0, 1.0)
    for hb in (40.3, 41.1, 41.9, 42.6):
        s.add(heartbeat(), hb, 1.0)
    s.add(whoosh(2.0, 100, 700, 0.8), 40.0, 0.5)
    rev = riser(1.1, 300, 12000)
    s.add(rev, 42.0, 0.8)
    s.add(impact(3.0, 1.8), 43.12, 1.3)                      # THE DUNK
    s.add(shatter(1.6, 120), 43.12, 0.9)
    s.add(clap(0.3), 43.12, 0.8)
    s.add(impact(1.2, 0.8), 43.3, 0.7)
    s.add(bounce(), 43.47, 0.5)
    s.add(bounce(), 43.9, 0.4)
    s.add(whoosh(0.2, 1000, 6000), 44.9, 0.6)
    s.add(impact(1.5, 1.0), 45.58, 0.9)
    s.add(shatter(0.8, 40), 45.58, 0.5)
    s.add(whoosh(0.2, 1000, 6000), 45.9, 0.6)
    s.add(impact(1.5, 1.0), 46.36, 0.9)
    # ACT 4 ----------------------------------------------------------------
    s.add(hum(2.7, 220), 47.0, 0.6)
    s.add(roar(2.4, 1.3) * 0.6, 47.4, 0.55)
    s.add(impact(3.5, 2.0), 49.75, 1.3)
    s.add(shimmer(3.0, 40), 49.75, 0.8)
    for k in range(26):
        s.add(xp_ping(1400 + RNG.random() * 1400), 49.9 + RNG.random() * 1.3, 0.45, RNG.random() * 1.2 - 0.6)
    s.add(bounce(), 50.55, 0.7)
    s.add(crowd(3.2), 50.9, 1.0)
    s.add(impact(2.0, 1.2), 54.0, 1.0)
    s.add(impact(1.2, 0.8), 54.15, 0.7)
    s.add(impact(1.0, 0.6), 54.5, 0.6)
    s.add(shimmer(3.5, 40), 54.3, 0.5)
    s.add(crowd(3.0) * 0.5, 54.0, 0.6)
    for i in range(len("THE END...")):
        s.add(blip(660), 58.45 + i / 14, 0.5)
    for i in range(len("WAS JUST THE BEGINNING.")):
        s.add(blip(990), 59.0 + i / 30, 0.35)
    s.add(impact(1.0, 1.3) , 59.0, 0.6)
    return s


def duck_env(times, depth=0.55, rel=0.35):
    g = np.ones(N)
    t = np.arange(N) / SR
    for tm in times:
        i = int(tm * SR)
        j = min(N, i + int(rel * 4 * SR))
        seg = t[i:j] - tm
        g[i:j] = np.minimum(g[i:j], 1 - depth * np.exp(-seg / rel))
    return g


def build(out):
    music = build_music()
    sfx = build_sfx()
    mb = reverb(music.b, 1.8, 0.22)
    sb = reverb(sfx.b, 2.4, 0.2)
    duck = duck_env([12.0, 23.36, 29.9, 32.75, 36.0, 43.12, 49.75, 54.0])
    mix = mb * 0.5 * duck[:, None] + sb * 0.55
    # fade in/out
    t = np.arange(N) / SR
    mix *= np.minimum(1, t / 0.8)[:, None]
    mix *= np.clip((DUR - t) / 0.6, 0, 1)[:, None]
    mix = hp(mix.T, 25).T
    peak = np.max(np.abs(mix)) + 1e-9
    mix = softclip(mix / peak * 1.6, 1.4) * 0.93
    wavfile.write(out, SR, (mix * 32767).astype(np.int16))
    print("wrote", out)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "score.wav")
