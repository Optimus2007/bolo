"""Myna launch cartoon (Myna = the assistant, Bolo = the models under it): a short 2D story, drawn and animated in code (Pillow + ffmpeg).

  .venv/bin/python launch/cartoon.py --preview 3 12 30     # still frames -> launch/out/cartoon_<t>.png
  .venv/bin/python launch/cartoon.py                       # full video -> launch/out/bolo_cartoon.mp4

Story and voice lines: launch/story.json. Voice audio: launch/audio/story/<id>.wav (from Kaggle; missing lines are
timed from their length and stay silent). Numbers on the report card come from results/ (never typed by hand).
Myna has no voice of its own yet, so it only speaks in text bubbles.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
L = ROOT / "launch"
OUT = L / "out"
AUD = L / "audio" / "story"
W, H, FPS, SR = 1920, 1080, 30, 48000

# palette
NIGHT_TOP, NIGHT_BOT = (34, 42, 78), (24, 30, 58)
DAY_TOP, DAY_BOT = (255, 206, 158), (255, 232, 205)
DESK, DESK_D = (150, 98, 60), (112, 72, 44)
SKIN, SKIN_D = (198, 134, 66), (170, 110, 52)
HAIR = (40, 28, 22)
HOODIE, HOODIE_D = (231, 111, 81), (196, 88, 62)
TEAL, TEAL_D = (46, 196, 182), (28, 150, 140)
INK, PAPER = (30, 34, 48), (252, 250, 245)
AMBER, CORAL, GREY = (255, 183, 3), (239, 71, 111), (150, 156, 170)
BOARD, CHALK = (44, 82, 66), (240, 244, 236)

FONT = "/System/Library/Fonts/SFNSRounded.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"
_fc: dict = {}


def font(size, mono=False, weight=600):
    k = (size, mono, weight)
    if k not in _fc:
        f = ImageFont.truetype(MONO if mono else FONT, size)
        if not mono:
            try:
                axes = f.get_variation_axes()
                f.set_variation_by_axes([weight if a["name"] == b"Weight" else a["default"] for a in axes])
            except Exception:
                pass
        _fc[k] = f
    return _fc[k]


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def back_out(x):  # overshoot, for pops
    x = min(1.0, max(0.0, x))
    c = 1.7
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


def mix(a, b, k):
    return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))


# ------------------------------------------------------------------ numbers and timeline
story = json.loads((L / "story.json").read_text(encoding="utf-8"))
res = ROOT / "results/speech"
NUM = {}
if (res / "decoding_test.json").exists():
    d = json.loads((res / "decoding_test.json").read_text())
    k = next(k for k in d if k.startswith("chosen ("))
    NUM = {"e2e": d[k]["e2e"] * 100, "n": 240}
if (res / "launch_numbers.json").exists():  # written by the v1 evaluation, preferred when present
    NUM.update(json.loads((res / "launch_numbers.json").read_text()))


def wav_len(p: Path) -> float:
    with wave.open(str(p)) as w:
        return w.getnframes() / w.getframerate()


def read_audio(p: Path) -> np.ndarray:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from bolo.audio.augment import read_wav

    a, sr = read_wav(str(p))
    a = np.asarray(a, dtype="float32")
    if sr != SR:
        a = np.interp(np.arange(0, len(a), sr / SR), np.arange(len(a)), a).astype("float32")
    return a


beats, t = [], 0.4
for b in story["beats"]:
    a = AUD / f"{b['id']}.wav"
    if b["voice"] == "none":
        d = b.get("hold", 2.0)
    else:
        d = wav_len(a) if a.exists() else max(1.6, len(b["text"]) / 14.0)
    pad = b.get("hold", 0.5) if b["voice"] != "none" else 0.0
    beats.append({**b, "start": t, "dur": d, "end": t + d + pad, "audio": a if a.exists() else None})
    t += d + pad
TOTAL = t + 1.2
BY = {b["id"]: b for b in beats}

# lip sync envelopes for Aarav's lines
ENV: dict = {}
for b in beats:
    if b["voice"] == "aarav" and b["audio"]:
        a = read_audio(b["audio"])
        hop = SR // FPS
        rms = np.array([np.sqrt(np.mean(a[i:i + hop] ** 2)) for i in range(0, len(a), hop)])
        ENV[b["id"]] = np.clip(rms / max(1e-6, np.percentile(rms, 95)), 0, 1)


def beat_at(t):
    for b in beats:
        if t < b["end"]:
            return b
    return beats[-1]


def talk(t):
    """Aarav's mouth opening 0..1 at time t (from his audio, or a wobble if the audio is missing)."""
    b = beat_at(t)
    if b["voice"] != "aarav" or not (b["start"] <= t <= b["start"] + b["dur"]):
        return 0.0
    if b["id"] in ENV:
        e = ENV[b["id"]]
        i = int((t - b["start"]) * FPS)
        return float(e[i]) if i < len(e) else 0.0
    return 0.5 + 0.5 * math.sin(t * 22) * math.sin(t * 7.3)


def local(b, t):
    return (t - b["start"]) / max(0.01, b["dur"])


# ------------------------------------------------------------------ drawing helpers
def rr(d, box, r, fill, outline=None, width=0):
    box = [int(v) for v in box]
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:  # pop-in animations start from (almost) nothing
        return
    r = max(0, min(int(r), (box[2] - box[0]) // 2, (box[3] - box[1]) // 2))
    d.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


def ell(d, cx, cy, rx, ry, fill, outline=None, width=0):
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fill, outline=outline, width=width)


def text_c(d, xy, s, size, fill=INK, mono=False, weight=600, anchor="mm"):
    d.text(xy, s, font=font(size, mono, weight), fill=fill, anchor=anchor)


def wrap(s, size, maxw, mono=False):
    f, out, cur = font(size, mono), [], ""
    for w in s.split():
        nxt = (cur + " " + w).strip()
        if ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(nxt, font=f) > maxw and cur:
            out.append(cur)
            cur = w
        else:
            cur = nxt
    return out + [cur]


def bubble(img, box, tail, txt, size=40, k=1.0, border=INK, fill=PAPER, color=INK, typed=1.0, mono=False):
    """Speech bubble with a tail to `tail` (x, y); text typed in by fraction `typed`; pops in by k."""
    if k <= 0:
        return
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = back_out(k)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    bx = [cx + (x0 - cx) * s, cy + (y0 - cy) * s, cx + (x1 - cx) * s, cy + (y1 - cy) * s]
    tx, ty = tail
    base = (min(max(tx, bx[0] + 40), bx[2] - 40), bx[3] if ty > bx[3] else bx[1])
    d.polygon([(base[0] - 26, base[1]), (base[0] + 26, base[1]), (cx + (tx - cx) * s, cy + (ty - cy) * s)], fill=border)
    rr(d, bx, 34 * s, fill=border)
    rr(d, [bx[0] + 5, bx[1] + 5, bx[2] - 5, bx[3] - 5], 30 * s, fill=fill)
    d.polygon([(base[0] - 19, base[1] + (-6 if ty > bx[3] else 6)), (base[0] + 19, base[1] + (-6 if ty > bx[3] else 6)),
               (cx + (tx - cx) * s * 0.96, cy + (ty - cy) * s * 0.96)], fill=fill)
    if s > 0.7:
        shown = txt[: int(len(txt) * typed)]
        lines = wrap(shown, size, (x1 - x0) - 70, mono) if shown else []
        lh = size * 1.25
        y = cy - lh * (len(lines) - 1) / 2
        for ln in lines:
            d.text((cx, y), ln, font=font(size, mono), fill=color, anchor="mm")
            y += lh
    img.alpha_composite(layer)


# ------------------------------------------------------------------ the room
def room(img, t, day=0.0):
    d = ImageDraw.Draw(img)
    top, bot = mix(NIGHT_TOP, DAY_TOP, day), mix(NIGHT_BOT, DAY_BOT, day)
    for y in range(0, 760, 8):
        d.rectangle([0, y, W, y + 8], fill=mix(top, bot, y / 760))
    # window
    wx0, wy0, wx1, wy1 = 150, 130, 560, 500
    rr(d, [wx0 - 16, wy0 - 16, wx1 + 16, wy1 + 16], 18, fill=mix((92, 70, 52), (170, 130, 95), day))
    sky_t, sky_b = mix((14, 20, 52), (255, 160, 90), day), mix((30, 38, 86), (255, 220, 150), day)
    for y in range(wy0, wy1, 6):
        d.rectangle([wx0, y, wx1, y + 6], fill=mix(sky_t, sky_b, (y - wy0) / (wy1 - wy0)))
    if day < 0.5:
        a = 1 - day * 2
        for i, (sx, sy) in enumerate([(200, 180), (300, 240), (420, 170), (480, 300), (250, 380), (380, 330), (520, 220)]):
            tw = 0.6 + 0.4 * math.sin(t * 2 + i)
            ell(d, sx, sy, 3, 3, fill=mix(sky_t, (255, 255, 230), a * tw))
        ell(d, 450, 230, 46, 46, fill=mix(sky_t, (246, 241, 213), a))
        ell(d, 470, 218, 40, 40, fill=mix(sky_t, sky_t, 1))
    else:
        sy = 520 - 260 * (day - 0.5) * 2
        ell(d, 360, sy, 60, 60, fill=(255, 214, 102))
    d.rectangle([wx0 + (wx1 - wx0) // 2 - 6, wy0, wx0 + (wx1 - wx0) // 2 + 6, wy1], fill=mix((92, 70, 52), (170, 130, 95), day))
    d.rectangle([wx0, wy0 + (wy1 - wy0) // 2 - 6, wx1, wy0 + (wy1 - wy0) // 2 + 6], fill=mix((92, 70, 52), (170, 130, 95), day))
    # shelf with books
    rr(d, [1500, 250, 1800, 270], 6, fill=mix((92, 70, 52), (150, 110, 80), day))
    rr(d, [1500, 430, 1800, 450], 6, fill=mix((92, 70, 52), (150, 110, 80), day))
    cols = [(239, 71, 111), (255, 183, 3), (6, 214, 160), (17, 138, 178), (155, 135, 245), (240, 140, 60)]
    x = 1515
    for i in range(9):
        w = 22 + (i * 7) % 14
        h = 120 + (i * 13) % 40
        d.rectangle([x, 250 - h, x + w, 250], fill=mix(mix(cols[i % 6], NIGHT_BOT, 0.35), cols[i % 6], day))
        x += w + 4
    ell(d, 1700, 395, 30, 30, fill=mix((40, 120, 80), (60, 170, 110), day))  # plant
    rr(d, [1680, 395, 1720, 430], 6, fill=mix((150, 80, 60), (200, 110, 80), day))
    # clock
    cx, cy = 1250, 220
    ell(d, cx, cy, 70, 70, fill=mix((220, 220, 230), (255, 255, 255), day), outline=INK, width=6)
    hh, mm = (2, 13) if day < 0.5 else (6, 30)
    for ang, ln, wd in (((hh % 12 + mm / 60) / 12 * 2 * math.pi, 34, 8), (mm / 60 * 2 * math.pi, 52, 5)):
        d.line([cx, cy, cx + ln * math.sin(ang), cy - ln * math.cos(ang)], fill=INK, width=wd)
    # desk
    d.rectangle([0, 760, W, 800], fill=DESK)
    d.rectangle([0, 800, W, H], fill=DESK_D)
    # mug with steam
    rr(d, [330, 680, 400, 762], 10, fill=(240, 240, 245))
    d.arc([385, 695, 425, 740], -90, 90, fill=(240, 240, 245), width=9)
    for i in range(3):
        pts = [(345 + i * 18 + 6 * math.sin(t * 3 + y / 12 + i), 670 - y) for y in range(0, 70, 6)]
        d.line(pts, fill=mix(top, (255, 255, 255), 0.45), width=4)
    # lamp
    d.rectangle([1400, 745, 1480, 762], fill=(60, 64, 80))
    d.line([1440, 750, 1400, 600, 1470, 540], fill=(60, 64, 80), width=10)
    d.polygon([(1440, 520), (1520, 560), (1470, 600)], fill=(255, 196, 61))
    if day < 0.9:
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(glow).ellipse([1290, 700, 1640, 790], fill=(255, 214, 120, int(70 * (1 - day))))
        img.alpha_composite(glow)


# ------------------------------------------------------------------ characters
AX, AY = 860, 470  # Aarav's head centre


def aarav(img, t, mood="neutral", look=(0, 0), mouth=0.0, facepalm=0.0, stretch=0.0, dy=0.0):
    d = ImageDraw.Draw(img)
    bob = 4 * math.sin(t * 1.6) + dy
    hx, hy = AX, AY + bob - 40 * stretch
    # body (hoodie) behind the laptop
    rr(d, [hx - 190, hy + 120, hx + 190, 900], 70, fill=HOODIE)
    d.polygon([(hx - 70, hy + 118), (hx, hy + 190), (hx + 70, hy + 118)], fill=HOODIE_D)
    d.line([hx - 30, hy + 150, hx - 36, hy + 230], fill=PAPER, width=5)
    d.line([hx + 30, hy + 150, hx + 36, hy + 230], fill=PAPER, width=5)
    rr(d, [hx - 34, hy + 80, hx + 34, hy + 130], 10, fill=SKIN_D)
    if stretch > 0:
        for side in (-1, 1):
            d.line([hx + side * 150, hy + 170, hx + side * 210, hy - 60 * stretch], fill=HOODIE, width=60)
            ell(d, hx + side * 212, hy - 60 * stretch - 20, 32, 32, fill=SKIN)
    # head
    ell(d, hx - 112, hy + 6, 22, 30, fill=SKIN_D)
    ell(d, hx + 112, hy + 6, 22, 30, fill=SKIN_D)
    ell(d, hx, hy, 115, 125, fill=SKIN)
    # hair
    d.chord([hx - 122, hy - 140, hx + 122, hy + 40], 180, 360, fill=HAIR)
    for i in range(7):
        x = hx - 110 + i * 36
        d.polygon([(x, hy - 60), (x + 30, hy - 150 + (i % 2) * 18), (x + 52, hy - 55)], fill=HAIR)
    # eyes
    blink = (t * 0.7) % 4.0 < 0.12
    for side in (-1, 1):
        ex, ey = hx + side * 44, hy + 10
        if blink:
            d.line([ex - 18, ey, ex + 18, ey], fill=INK, width=5)
        else:
            ell(d, ex, ey, 19, 23, fill=(255, 255, 255))
            ell(d, ex + 6 * look[0], ey + 5 * look[1], 10, 12, fill=INK)
            ell(d, ex + 6 * look[0] + 4, ey + 5 * look[1] - 5, 3, 3, fill=(255, 255, 255))
        # brows
        tilt = {"angry": 10, "sad": -10, "happy": -3, "surprised": -6}.get(mood, 0)
        lift = {"surprised": -14, "happy": -4}.get(mood, 0)
        d.line([ex - 22, ey - 38 + lift + side * (-tilt) * 0.5 + tilt * 0.5 * (side == -1) * 0,
                ex + 22, ey - 38 + lift + side * tilt * 0.5], fill=HAIR, width=8)
    # nose
    d.arc([hx - 12, hy + 26, hx + 12, hy + 56], 20, 160, fill=SKIN_D, width=5)
    # mouth
    my = hy + 78
    if mouth > 0.08:
        ell(d, hx, my, 22 + 6 * mouth, 5 + 22 * mouth, fill=(110, 40, 40))
        ell(d, hx, my + 8 * mouth, 12 * mouth + 4, 5 * mouth, fill=(220, 100, 100))
    elif mood == "happy":
        d.chord([hx - 40, my - 30, hx + 40, my + 22], 10, 170, fill=(110, 40, 40))
    elif mood in ("sad", "angry"):
        d.arc([hx - 30, my, hx + 30, my + 34], 200, 340, fill=(110, 40, 40), width=6)
    elif mood == "surprised":
        ell(d, hx, my + 4, 14, 18, fill=(110, 40, 40))
    else:
        d.arc([hx - 28, my - 22, hx + 28, my + 10], 30, 150, fill=(110, 40, 40), width=6)
    if facepalm > 0:
        k = ease(facepalm)
        px, py = hx + 60, 820 - (820 - (hy - 10)) * k
        d.line([hx + 160, 880, px, py + 60], fill=HOODIE, width=64)
        ell(d, px, py, 52, 46, fill=SKIN)
        for i in range(3):
            d.line([px - 40 + i * 26, py - 30, px - 44 + i * 26, py - 62], fill=SKIN, width=18)


LID = (700, 610, 1020, 790)  # laptop lid, back side facing us


def laptop(img, t, glow=0.0):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = LID
    if glow > 0:
        g = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(g).ellipse([AX - 120, AY - 130, AX + 120, AY + 125], fill=(150, 240, 230, int(30 * glow)))
        img.alpha_composite(g)
    rr(d, [x0, y0, x1, y1], 18, fill=(200, 206, 216), outline=(150, 156, 170), width=4)
    ell(d, (x0 + x1) / 2, (y0 + y1) / 2, 22, 22, fill=TEAL)
    d.rectangle([x0 - 40, y1 - 4, x1 + 40, y1 + 8], fill=(150, 156, 170))


def bolo(img, t, x, y, scale=1.0, mood="happy", listen=0.0, squash=0.0, wave=0.0):
    """The mascot: a round teal speech blob with big eyes and sound-wave antennae."""
    d = ImageDraw.Draw(img)
    s = scale
    sq = squash + 0.04 * math.sin(t * 5)
    rx, ry = 78 * s * (1 + sq), 70 * s * (1 - sq)
    # sound waves on top
    for i in range(3):
        a = 0.35 + 0.65 * (listen * (0.5 + 0.5 * math.sin(t * 9 - i)))
        r = (26 + i * 18) * s
        d.arc([x - r, y - ry - r - 6 * s, x + r, y - ry + r - 6 * s], 220, 320, fill=mix(TEAL_D, (255, 255, 255), a * 0.6), width=int(6 * s))
    # tail of the speech bubble body
    d.polygon([(x - 30 * s, y + ry * 0.7), (x - 60 * s, y + ry + 26 * s), (x + 4 * s, y + ry * 0.85)], fill=TEAL)
    ell(d, x, y, rx, ry, fill=TEAL)
    ell(d, x - 24 * s, y - 26 * s, 22 * s, 14 * s, fill=mix(TEAL, (255, 255, 255), 0.35))
    for i, (ox, h) in enumerate(((-16, 30), (0, 40), (16, 28))):  # crest tuft
        d.polygon([(x + ox * s - 9 * s, y - ry + 8 * s), (x + (ox + 4) * s, y - ry - h * s), (x + ox * s + 9 * s, y - ry + 8 * s)], fill=TEAL_D)
    blink = (t * 0.9 + 1.3) % 3.3 < 0.12
    for side in (-1, 1):
        ex, ey = x + side * 26 * s, y - 6 * s
        if blink:
            d.line([ex - 14 * s, ey, ex + 14 * s, ey], fill=INK, width=int(5 * s))
            continue
        ell(d, ex, ey, 17 * s, 21 * s, fill=(255, 255, 255))
        ell(d, ex + 3 * s, ey + 3 * s, 9 * s, 11 * s, fill=INK)
        ell(d, ex + 6 * s, ey - 2 * s, 3 * s, 3 * s, fill=(255, 255, 255))
    d.polygon([(x - 13 * s, y + 10 * s), (x + 13 * s, y + 10 * s), (x, y + 26 * s)], fill=AMBER)  # myna's yellow beak
    my = y + 36 * s
    if mood == "happy":
        d.chord([x - 20 * s, my - 14 * s, x + 20 * s, my + 12 * s], 10, 170, fill=INK)
    elif mood == "unsure":
        d.line([x - 14 * s, my, x + 14 * s, my - 5 * s], fill=INK, width=int(5 * s))
    else:
        d.arc([x - 16 * s, my - 12 * s, x + 16 * s, my + 6 * s], 30, 150, fill=INK, width=int(5 * s))
    if wave > 0:
        ang = math.sin(t * 10) * 0.5
        hx, hy = x + rx + 10 * s, y - 10 * s
        d.line([x + rx * 0.8, y, hx + 30 * s * math.cos(ang - 1), hy - 40 * s], fill=TEAL, width=int(16 * s))
        ell(d, hx + 30 * s * math.cos(ang - 1), hy - 44 * s, 13 * s, 13 * s, fill=TEAL)


def screen(img, k, content, t, extra=0.0):
    """A floating 'what the laptop shows' panel, pointing at the laptop."""
    if k <= 0:
        return
    s = back_out(k)
    x0, y0, x1, y1 = 1100, 110, 1780, 520
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = [cx + (x0 - cx) * s, cy + (y0 - cy) * s, cx + (x1 - cx) * s, cy + (y1 - cy) * s]
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.polygon([(box[0] + 60, box[3] - 10), (box[0] + 160, box[3] - 10), cam(LID[2] - 40, LID[1] + 10)], fill=(20, 24, 36, 230))
    rr(d, box, 26, fill=(20, 24, 36, 245))
    for i, c in enumerate((CORAL, AMBER, TEAL)):
        ell(d, box[0] + 34 + i * 30, box[1] + 30, 9, 9, fill=c)
    if s > 0.8:
        if content == "generic_fail":
            rr(d, [x0 + 70, y0 + 140, x1 - 70, y0 + 280], 28, fill=(70, 76, 96))
            d.text(((x0 + x1) / 2, y0 + 210), "Sorry, I didn't get that.", font=font(46), fill=(235, 238, 245), anchor="mm")
        elif content == "generic_search":
            d.text((x0 + 60, y0 + 120), "Searching the web for", font=font(40), fill=(200, 205, 220))
            d.text((x0 + 60, y0 + 190), "“volume thoda", font=font(52, weight=700), fill=(255, 255, 255))
            d.text((x0 + 60, y0 + 260), "come car oh”  …", font=font(52, weight=700), fill=(255, 255, 255))
            for i in range(3):
                ell(d, x0 + 90 + i * 40, y0 + 350, 9, 9, fill=mix((70, 76, 96), (255, 255, 255), 0.5 + 0.5 * math.sin(t * 6 - i)))
        elif content == "terminal":
            d.text((x0 + 50, y0 + 100), "Terminal  ·  ~/client", font=font(34), fill=(160, 166, 186))
            d.text((x0 + 50, y0 + 170), "~/client $", font=font(48, mono=True), fill=TEAL)
            if (t * 2) % 2 < 1.2:
                d.rectangle([x0 + 340, y0 + 150, x0 + 366, y0 + 196], fill=(240, 240, 240))
            d.text((x0 + 50, y0 + 270), "opened by Myna, offline", font=font(32), fill=(160, 166, 186))
    img.alpha_composite(layer)


def card(img, box, title, sub, icon, k, accent=TEAL):
    if k <= 0:
        return
    s = back_out(k)
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    b = [cx + (x0 - cx) * s, cy + (y0 - cy) * s, cx + (x1 - cx) * s, cy + (y1 - cy) * s]
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    rr(d, [b[0] + 8, b[1] + 12, b[2] + 8, b[3] + 12], 30, fill=(0, 0, 0, 60))
    rr(d, b, 30, fill=PAPER)
    if s > 0.75:
        icon_draw(d, icon, cx, b[1] + 105, accent)
        d.text((cx, b[3] - 110), title, font=font(34, weight=700), fill=INK, anchor="mm")
        for i, ln in enumerate(wrap(sub, 28, (x1 - x0) - 50)):
            d.text((cx, b[3] - 66 + i * 34), ln, font=font(28), fill=(90, 96, 112), anchor="mm")
    img.alpha_composite(layer)


def icon_draw(d, name, x, y, c):
    if name == "music":
        ell(d, x, y, 52, 52, fill=c)
        d.rectangle([x - 18, y - 24, x - 6, y + 24], fill=PAPER)
        d.rectangle([x + 6, y - 24, x + 18, y + 24], fill=PAPER)
    elif name == "timer":
        ell(d, x, y + 6, 50, 50, fill=c)
        ell(d, x, y + 6, 38, 38, fill=PAPER)
        d.rectangle([x - 10, y - 60, x + 10, y - 44], fill=c)
        d.line([x, y + 6, x, y - 22], fill=INK, width=7)
        d.line([x, y + 6, x + 20, y + 16], fill=INK, width=7)
    elif name == "bell":
        d.chord([x - 44, y - 50, x + 44, y + 40], 180, 360, fill=c)
        d.rectangle([x - 44, y - 6, x + 44, y + 28], fill=c)
        d.rectangle([x - 56, y + 24, x + 56, y + 36], fill=c)
        ell(d, x, y + 48, 12, 12, fill=c)
    elif name == "chat":
        rr(d, [x - 56, y - 42, x + 56, y + 32], 22, fill=c)
        d.polygon([(x - 30, y + 30), (x - 46, y + 56), (x - 8, y + 30)], fill=c)
        for i in (-1, 0, 1):
            ell(d, x + i * 26, y - 5, 8, 8, fill=PAPER)
    elif name == "wifi":
        for i, r in enumerate((56, 38, 20)):
            d.arc([x - r, y - r + 20, x + r, y + r + 20], 225, 315, fill=c, width=11)
        ell(d, x, y + 22, 9, 9, fill=c)
        d.line([x - 50, y - 40, x + 50, y + 50], fill=CORAL, width=10)
    elif name == "camera":
        rr(d, [x - 58, y - 34, x + 58, y + 42], 16, fill=c)
        rr(d, [x - 22, y - 50, x + 22, y - 30], 6, fill=c)
        ell(d, x, y + 4, 26, 26, fill=PAPER)
        ell(d, x, y + 4, 14, 14, fill=c)
    elif name == "cloud":
        for ox, oy, r in ((-28, 8, 30), (6, -10, 38), (36, 10, 28)):
            ell(d, x + ox, y + oy, r, r, fill=c)
        d.rectangle([x - 56, y + 8, x + 62, y + 38], fill=c)
        d.line([x - 60, y - 50, x + 60, y + 60], fill=CORAL, width=10)
    elif name == "lock":
        d.arc([x - 30, y - 62, x + 30, y - 2], 180, 360, fill=c, width=12)
        rr(d, [x - 46, y - 30, x + 46, y + 44], 14, fill=c)
        ell(d, x, y + 6, 9, 9, fill=PAPER)


def dim(img, a):
    if a > 0:
        img.alpha_composite(Image.new("RGBA", (W, H), (12, 14, 30, int(160 * a))))


def caption(img, t):
    b = beat_at(t)
    if b["voice"] != "narrator" or not (b["start"] <= t <= b["start"] + b["dur"] + 0.3):
        return
    txt = b["text"].replace("{N_WORDS}", str(NUM.get("n", ""))).replace("{E2E_WORDS}", f"{NUM.get('e2e', 0):.0f}")
    lines = wrap(txt, 34, 1300)
    k = min(len(lines) - 1, int(local(b, t) * len(lines)))
    s = lines[k]
    d = ImageDraw.Draw(img)
    f = font(34, weight=500)
    tw = d.textlength(s, font=f)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rr(ImageDraw.Draw(layer), [W / 2 - tw / 2 - 24, H - 112, W / 2 + tw / 2 + 24, H - 56], 16, fill=(10, 12, 24, 200))
    img.alpha_composite(layer)
    d.text((W / 2, H - 84), s, font=f, fill=(255, 255, 255), anchor="mm")


# ------------------------------------------------------------------ scenes
def in_beat(id_, t):
    b = BY[id_]
    return b["start"] <= t < b["end"]


def since(id_, t):
    return t - BY[id_]["start"]


BOLO_HOME = (1150, 660)


def scene_room(img, t, b):
    sid = b["scene"]
    day = ease((t - BY["n09"]["start"]) / 2.5) if t >= BY["n09"]["start"] else 0.0
    room(img, t, day)
    mood, look, face, stretch = "neutral", (0, 0.4), 0.0, 0.0
    if sid in ("fail1b", "fail2b"):
        mood = "sad" if sid == "fail1b" else "angry"
    if sid == "facepalm":
        mood, face = "sad", min(1.0, since("n02", t) * 1.5) * (1 - ease((since("n02", t) - b["dur"] + 0.4) * 2))
    if sid in ("popup",):
        mood, look = "surprised", (0.8, 0.2)
    if sid in ("listen", "pipeline", "unsure", "unsure2", "offline"):
        mood, look = ("happy" if sid in ("pipeline", "offline") else "neutral"), (0.8, 0.3)
    if sid == "sunrise":
        mood, stretch = "happy", ease(since("n09", t) / 1.5) * (1 - ease((since("n09", t) - 3) / 1.0))
    aarav(img, t, mood, look, talk(t), facepalm=face, stretch=stretch)
    glow = 1.0 if sid in ("fail1", "fail1b", "fail2", "fail2b", "listen", "pipeline", "unsure", "unsure2") else 0.4
    bolo_on = t >= BY["n03"]["start"]
    if bolo_on:
        k = ease(since("n03", t) / 0.9)
        x, y = BOLO_HOME
        y = 860 - (860 - y) * back_out(k)
        sq = -0.25 * (1 - k) if k < 1 else 0.0
        if k < 1:  # emerging from behind the lid
            layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            bolo(layer, t, x, y, 1.0, "happy", squash=sq)
            mask = Image.new("L", (W, H), 255)
            ImageDraw.Draw(mask).rectangle([0, 760, W, H], fill=0)
            img.paste(layer, (0, 0), Image.composite(layer.getchannel("A"), Image.new("L", (W, H), 0), mask))
            laptop(img, t, glow)
        else:
            laptop(img, t, glow)
            listen = 1.0 if sid in ("listen", "unsure") else 0.0
            bm = "unsure" if sid in ("unsure", "unsure2") else "happy"
            wv = 1.0 if sid == "sunrise" or (sid == "popup" and since("n03", t) > 1.2) else 0.0
            hop = 18 * abs(math.sin(t * 3.2)) if sid in ("montage", "offline") else 0.0
            bolo(img, t, x, y - hop, 1.0, bm, listen=listen, wave=wv)
    else:
        laptop(img, t, glow)
    img.paste(zoom(img, ZOOM, ZC))  # camera on the world; overlays below are in screen space
    # speech and screens
    if sid in ("fail1", "fail1b", "fail2", "fail2b", "listen", "unsure", "unsure2") and b["voice"] in ("aarav", "none", "narrator"):
        ab = {"fail1": "a01", "fail1b": "a01", "fail2": "a02", "fail2b": "a02", "listen": "a03", "pipeline": "a03",
              "unsure": "a04", "unsure2": "a04"}[sid]
        A = BY[ab]
        k = ease((t - A["start"]) / 0.35) * (1 - ease((t - A["end"] - (2.2 if sid.endswith("b") or sid == "unsure2" else 0.3)) / 0.3))
        typed = min(1.0, (t - A["start"]) / max(0.3, A["dur"] * 0.9))
        bubble(img, (190, 140, 690, 300), (cam(AX - 90, AY - 40)), A.get("caption", A["text"]), 40, k, typed=typed)
    if sid in ("fail1b",):
        screen(img, ease(since("x01", t) / 0.3), "generic_fail", t)
    if sid in ("fail2b",):
        screen(img, ease(since("x02", t) / 0.3), "generic_search", t)
    if sid == "pipeline":
        p = since("n04", t)
        dim(img, 0.35 * ease(p / 0.4))
        steps = [("heard", "client folder mein terminal kholo"), ("wants", "open terminal · folder: client"),
                 ("checks", "~/client exists ✓")]
        for i, (lab, val) in enumerate(steps):
            k = ease((p - 0.3 - i * 1.6) / 0.35)
            if k > 0:
                y = 150 + i * 120
                layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                d = ImageDraw.Draw(layer)
                rr(d, [120, y, 1180, y + 96], 24, fill=(255, 255, 255, int(245 * k)))
                d.text((160, y + 48), lab, font=font(32, weight=700), fill=(*TEAL_D, int(255 * k)), anchor="lm")
                d.text((330, y + 48), val, font=font(36, mono=True), fill=(*INK, int(255 * k)), anchor="lm")
                img.alpha_composite(layer)
        screen(img, ease((p - 0.3 - 3 * 1.6) / 0.3), "terminal", t)
    if sid in ("unsure", "unsure2") and t >= BY["a04"]["start"] + BY["a04"]["dur"] * 0.7:
        k = ease((t - BY["a04"]["start"] - BY["a04"]["dur"] * 0.7) / 0.3)
        bubble(img, (1140, 200, 1720, 380), cam(BOLO_HOME[0] + 30, BOLO_HOME[1] - 70), "Kaun sa folder? Naam bata do.",
               40, k, border=TEAL_D, fill=(230, 252, 249))
    if sid == "offline":
        p = since("n07", t)
        dim(img, 0.25 * ease(p / 0.4))
        card(img, (1040, 90, 1320, 440), "No internet", "works with wifi off", "wifi", ease((p - 0.2) / 0.4), AMBER)
        card(img, (1340, 90, 1620, 440), "No cloud", "nothing is uploaded", "cloud", ease((p - 1.0) / 0.4), AMBER)
        card(img, (1640, 90, 1900, 440), "Private", "your voice stays here", "lock", ease((p - 1.8) / 0.4), TEAL)


MONTAGE = [("music", "Music", "gaana pause karo"), ("timer", "Timer", "10 minute ka timer lagao"),
           ("bell", "Reminder", "20 minute baad yaad dilana ki chai peeni hai"),
           ("chat", "Message", "Rahul ko message bhejo ki main late hoon"), ("wifi", "Settings", "wifi band karo"),
           ("camera", "Screenshot", "screenshot lo")]


def scene_montage(img, t, b):
    room(img, t, 0)
    aarav(img, t, "happy", (0.6, 0.3), 0)
    laptop(img, t, 0.6)
    bolo(img, t, BOLO_HOME[0], BOLO_HOME[1] - 18 * abs(math.sin(t * 3.2)), 1.0, "happy")
    p = t - b["start"]
    dim(img, 0.55 * ease(p / 0.4))
    for i, (ic, title, sub) in enumerate(MONTAGE):
        cx = 360 + (i % 3) * 600
        cy = 250 + (i // 3) * 420
        card(img, (cx - 250, cy - 170, cx + 250, cy + 190), title, "“" + sub + "”", ic, ease((p - 0.2 - i * 0.55) / 0.35))
    k = ease((p - 0.2 - 6 * 0.55) / 0.4)
    if k > 0:
        d = ImageDraw.Draw(img)
        d.text((W / 2, 925), "+ 14 more kinds of commands", font=font(40, weight=700), fill=(255, 255, 255), anchor="mm")


def scene_board(img, t, b):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, H], fill=(120, 84, 56))
    rr(d, [110, 70, W - 110, H - 150], 26, fill=BOARD, outline=(90, 62, 40), width=22)
    p = t - b["start"]
    e2e, n = NUM.get("e2e", 0), NUM.get("n", 0)
    text_c(d, (W / 2, 190), "Report card", 70, CHALK, weight=800)
    text_c(d, (W / 2, 280), f"{n} test commands  ·  computer voices it never trained on", 38, (210, 222, 210))
    x0, x1, y = 360, 1400, 470
    d.rectangle([x0, y - 50, x1, y + 50], outline=CHALK, width=6)
    k = ease((p - 0.6) / 1.4)
    d.rectangle([x0 + 10, y - 40, x0 + 10 + (x1 - x0 - 20) * e2e / 100 * k, y + 40], fill=(120, 220, 200))
    text_c(d, (x1 + 150, y), f"{e2e * k:.0f}%", 90, CHALK, weight=800)
    text_c(d, (W / 2, 600), "picked the right action", 40, (210, 222, 210))
    k2 = ease((p - 3.5) / 0.6)
    if k2 > 0:
        text_c(d, (W / 2, 730), "real human voices: being tested now", 44, mix(BOARD, AMBER, k2), weight=700)
    bolo(img, t, W - 300, H - 230, 0.9, "happy")
    d.line([W - 380, H - 250, W - 560, H - 420], fill=(150, 110, 70), width=10)


def scene_logo(img, t):
    room(img, t, 1.0)
    p = t - BY["n09"]["start"] - 3.2
    k = ease(p / 0.6)
    if k <= 0:
        return False
    layer = Image.new("RGBA", (W, H), (255, 250, 242, int(235 * k)))
    img.alpha_composite(layer)
    d = ImageDraw.Draw(img)
    bolo(img, t, W / 2 - 260, 430, 1.5 * back_out(k), "happy", wave=1.0)
    text_c(d, (W / 2 + 130, 430), "Myna", 180, mix((255, 250, 242), INK, k), weight=800)
    text_c(d, (W / 2, 620), "Hinglish voice commands, on your own laptop", 46, mix((255, 250, 242), (90, 96, 112), k))
    text_c(d, (W / 2, 710), "built on Bolo  ·  open source  ·  free  ·  offline", 42, mix((255, 250, 242), TEAL_D, k), weight=700)
    return True


ZOOM, ZC = 1.22, (960, 560)


def _box(z, c):
    w, h = W / z, H / z
    return min(max(0, c[0] - w / 2), W - w), min(max(0, c[1] - h / 2), H - h)


def cam(x, y):
    """World point -> screen point under the room camera."""
    x0, y0 = _box(ZOOM, ZC)
    return ((x - x0) * ZOOM, (y - y0) * ZOOM)


def zoom(img, z, c):
    """Simple camera: crop around c and scale back up, so the characters fill the frame."""
    w, h = W / z, H / z
    x0 = min(max(0, c[0] - w / 2), W - w)
    y0 = min(max(0, c[1] - h / 2), H - h)
    return img.resize((W, H), Image.LANCZOS, box=(x0, y0, x0 + w, y0 + h))


def frame(t):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    b = beat_at(t)
    if b["scene"] == "montage":
        scene_montage(img, t, b)
    elif b["scene"] == "board":
        scene_board(img, t, b)
    elif b["scene"] == "sunrise" and t >= b["start"] + 3.2:
        if not scene_logo(img, t):
            scene_room(img, t, b)
    else:
        scene_room(img, t, b)
    # quick fades at scene-kind changes and at the ends
    kinds = {"montage": 1, "board": 2}
    prev = beats[beats.index(b) - 1] if beats.index(b) else None
    if prev and kinds.get(prev["scene"], 0) != kinds.get(b["scene"], 0):
        a = 1 - ease((t - b["start"]) / 0.35)
        if a > 0:
            img.alpha_composite(Image.new("RGBA", (W, H), (10, 12, 24, int(255 * a))))
    caption(img, t)
    edge = min(1.0, t / 0.5, (TOTAL - t) / 0.8)
    if edge < 1:
        img.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, int(255 * (1 - max(0.0, edge))))))
    return img.convert("RGB")


# ------------------------------------------------------------------ sound
def sfx_pop(n=0.18):
    tt = np.arange(int(SR * n)) / SR
    f = 300 + 900 * tt / n
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 18) * 0.5).astype("float32")


def sfx_ding():
    tt = np.arange(int(SR * 0.6)) / SR
    return ((np.sin(2 * np.pi * 1320 * tt) + 0.5 * np.sin(2 * np.pi * 1980 * tt)) * np.exp(-tt * 6) * 0.22).astype("float32")


def sfx_buzz():
    tt = np.arange(int(SR * 0.35)) / SR
    return (np.sign(np.sin(2 * np.pi * 140 * tt)) * np.exp(-tt * 8) * 0.12).astype("float32")


def soundtrack():
    n = int(TOTAL * SR)
    out = np.zeros(n, "float32")
    tt = np.arange(n) / SR
    # light plucky background: a soft arpeggio, synthesised (no third party music)
    notes = [261.6, 329.6, 392.0, 523.3, 392.0, 329.6]
    step = 0.32
    for i in range(int(TOTAL / step)):
        f = notes[i % len(notes)] * (0.5 if (i // 12) % 2 else 1)
        s, e = int(i * step * SR), int((i * step + 0.3) * SR)
        if e >= n:
            break
        seg = np.arange(e - s) / SR
        out[s:e] += (np.sin(2 * np.pi * f * seg) * np.exp(-seg * 9) * 0.035).astype("float32")
    out *= np.minimum(1, np.minimum(tt / 1.5, (TOTAL - tt) / 1.5)).astype("float32")

    def put(a, at):
        s = int(at * SR)
        if s < n:
            out[s:s + len(a)] += a[: n - s]

    for b in beats:
        if b["audio"]:
            a = read_audio(b["audio"])
            put(a / max(1e-6, np.abs(a).max()) * (0.9 if b["voice"] == "aarav" else 0.85), b["start"])
    put(sfx_buzz(), BY["x01"]["start"] + 0.1)
    put(sfx_buzz(), BY["x02"]["start"] + 0.1)
    put(sfx_pop(), BY["n03"]["start"] + 0.2)
    put(sfx_ding(), BY["n04"]["start"] + 0.3 + 3 * 1.6)
    for i in range(6):
        put(sfx_pop(0.12), BY["n05"]["start"] + 0.2 + i * 0.55)
    put(sfx_pop(), BY["n09"]["start"] + 3.2)
    return np.clip(out, -1, 1)


def render():
    import imageio_ffmpeg

    OUT.mkdir(parents=True, exist_ok=True)
    wav = OUT / "cartoon_sound.wav"
    a = (soundtrack() * 32767).astype("<i2")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(a.tobytes())
    mp4 = OUT / "bolo_cartoon.mp4"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(wav), "-c:v", "libx264", "-preset", "medium",
           "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
           str(mp4)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(TOTAL * FPS)
    for i in range(n):
        pr.stdin.write(frame(i / FPS).tobytes())
        if i % 300 == 0:
            print(f"{i}/{n}", flush=True)
    pr.stdin.close()
    pr.wait()
    print(f"{mp4}  {TOTAL:.1f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=float, nargs="*")
    a = ap.parse_args()
    if a.preview is not None:
        OUT.mkdir(parents=True, exist_ok=True)
        for tt in a.preview:
            frame(tt).save(OUT / f"cartoon_{tt:05.1f}.png")
        print([(b["id"], round(b["start"], 1)) for b in beats], round(TOTAL, 1))
    else:
        render()
