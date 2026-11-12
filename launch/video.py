"""Render the Bolo launch video (1920x1080, 30 fps) from code: 2D motion graphics plus real Bolo outputs.

  .venv/bin/python launch/video.py --preview 12.5     # write one frame at t=12.5 s to launch/out/frame.png
  .venv/bin/python launch/video.py                    # full video -> launch/out/bolo_launch.mp4

Inputs: launch/narration.json (script), launch/audio/<id>.wav (narration takes from Kaggle, optional: missing lines
are timed from their length and stay silent), launch/demo.json (real Bolo outputs, made by launch/capture_demo.py),
results/speech/*.json (every number on screen is read from there, never typed in by hand).
Output goes to launch/out/ (gitignored).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
L = ROOT / "launch"
OUT = L / "out"
W, H, FPS, SR = 1920, 1080, 30, 48000

BG, BG2 = (11, 15, 28), (20, 27, 48)
INK, MUTED, DIM = (241, 243, 248), (138, 147, 166), (70, 78, 100)
TEAL, AMBER, CORAL, VIOLET = (46, 196, 182), (255, 183, 3), (255, 107, 107), (155, 135, 245)

FONT = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"
DEVA = "/System/Library/Fonts/Kohinoor.ttc"
_fc: dict = {}


def font(size, kind="sans", weight=None):
    key = (size, kind, weight)
    if key not in _fc:
        path = {"sans": FONT, "mono": MONO, "deva": DEVA}[kind]
        f = ImageFont.truetype(path, size)
        if kind == "sans":  # axes: width, optical size, grade, weight
            f.set_variation_by_axes([100, min(96, max(17, size * 0.6)), 400, weight or 420])
        _fc[key] = f
    return _fc[key]


def has_deva(s):
    return any("ऀ" <= c <= "ॿ" for c in s)


def ease(x):  # smooth 0..1
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def mix(a, b, k):
    return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))


def C(col, a=1.0):
    """colour with alpha, for drawing on the transparent scene layer"""
    return (*col[:3], int(255 * min(1.0, max(0.0, a))))


# ------------------------------------------------------------------ data
spec = json.loads((L / "narration.json").read_text(encoding="utf-8"))
res = ROOT / "results/speech"
dec = json.loads((res / "decoding_test.json").read_text())
NUM = {
    "bolo": dec["chosen (beam5_boost0_rerank)"]["e2e"] * 100,
    "greedy": dec["greedy (current default)"]["e2e"] * 100,
    "ceiling": 93.75,
    "ms": dec["chosen (beam5_boost0_rerank)"]["ms_median"],
}
sw = res / "swift_heldout.json"
if sw.exists():
    _sw = json.loads(sw.read_text())
    NUM["swift"] = _sw["greedy"]["e2e"] * 100
    NUM["swiftr"] = _sw["beam5_rerank"]["e2e"] * 100
demo = json.loads((L / "demo.json").read_text(encoding="utf-8")) if (L / "demo.json").exists() else []


def fill(text: str) -> str:
    return text.replace("{SWIFTR}", f"{NUM.get('swiftr', 0):.0f}").replace("{SWIFT}", f"{NUM.get('swift', 0):.0f}")


def wav_dur(p: Path) -> float:
    with wave.open(str(p)) as w:
        return w.getnframes() / w.getframerate()


# timeline: each line gets its audio length (or an estimate) plus a pause
lines = []
t = 0.6
for ln in spec["lines"]:
    text = fill(ln["text"])
    if not text.strip() or "{" in text:
        continue
    a = L / "audio" / f"{ln['id']}.wav"
    d = wav_dur(a) if a.exists() else max(2.0, len(text) / 14.5)
    hold = {"07_demo": 9.0, "08_numbers": 1.5, "11_close": 2.5}.get(ln["id"], 0.6)
    lines.append({"id": ln["id"], "text": text, "start": t, "dur": d, "end": t + d + hold, "audio": a if a.exists() else None})
    t += d + hold
TOTAL = t + 0.8
SC = {x["id"]: x for x in lines}


def scene_at(t):
    for x in lines:
        if x["start"] - 0.3 <= t < x["end"]:
            return x
    return lines[-1]


# ------------------------------------------------------------------ drawing helpers
def background(t):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # slow drifting glow
    for i, (c, r, sx, sy) in enumerate([(TEAL, 520, 0.05, 0.03), (VIOLET, 460, -0.04, 0.05)]):
        cx = W * (0.25 + 0.5 * i) + 120 * math.sin(t * sx * 6 + i)
        cy = H * 0.45 + 80 * math.cos(t * sy * 6 + i)
        glow = Image.new("RGB", (W, H), (0, 0, 0))
        ImageDraw.Draw(glow).ellipse([cx - r, cy - r, cx + r, cy + r], fill=mix(BG, c, 0.14))
        img = Image.blend(img, Image.composite(glow, img, glow.convert("L").point(lambda v: 255 if v else 0)), 0.5)
    return img


_bg_cache: dict = {}


def bg(t):
    k = int(t * 10)  # the glow moves slowly; 10 updates per second is enough
    if k not in _bg_cache:
        if len(_bg_cache) > 4:
            _bg_cache.clear()
        _bg_cache[k] = background(k / 10).filter(ImageFilter.GaussianBlur(90))
    return _bg_cache[k].copy()


def text(d, xy, s, size, color=INK, kind=None, anchor="la", weight=None, alpha=1.0, base=BG):
    kind = kind or ("deva" if has_deva(s) else "sans")
    d.text(xy, s, font=font(size, kind, weight), fill=C(color, alpha), anchor=anchor)


def rounded(d, box, r, fill=None, outline=None, width=2):
    d.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


def caption(img, t):
    x = scene_at(t)
    if not (x["start"] <= t <= x["start"] + x["dur"] + 0.4):
        return
    d = ImageDraw.Draw(img)
    words = x["text"].split()
    # wrap into lines of at most ~70 chars, show the line being spoken
    rows, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 72:
            rows.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    rows.append(cur)
    k = min(len(rows) - 1, int((t - x["start"]) / max(0.01, x["dur"]) * len(rows)))
    s = rows[k]
    f = font(34, "deva" if has_deva(s) else "sans")
    tw = d.textlength(s, font=f)
    rounded(d, (W / 2 - tw / 2 - 22, H - 118, W / 2 + tw / 2 + 22, H - 62), 14, fill=(6, 9, 18))
    d.text((W / 2, H - 90), s, font=f, fill=INK, anchor="mm")


def waveform(d, cx, cy, wdt, hgt, t, color=TEAL, energy=1.0, bars=48):
    gap = wdt / bars
    for i in range(bars):
        ph = i / bars
        a = energy * (0.25 + 0.75 * abs(math.sin(t * 7 + i * 0.55) * math.sin(t * 3.1 + i * 0.21)))
        a *= math.sin(math.pi * ph) ** 0.6
        h = max(4, hgt * a)
        x = cx - wdt / 2 + i * gap
        d.rounded_rectangle((x, cy - h / 2, x + gap * 0.55, cy + h / 2), 3, fill=color)


# ------------------------------------------------------------------ scenes
def s_hook(img, t, x):
    d = ImageDraw.Draw(img)
    p = (t - x["start"]) / x["dur"]
    waveform(d, W / 2, 330, 900, 160, t, TEAL, energy=0.9)
    for i, s in enumerate(["“docker chalu karo”", "“volume chaalis pe kar do”"]):
        k = ease((p - i * 0.45) * 3)
        text(d, (W / 2, 560 + i * 120 + 30 * (1 - k)), s, 72, INK, anchor="mm", alpha=k, weight=600)


def s_problem(img, t, x):
    d = ImageDraw.Draw(img)
    p = (t - x["start"]) / x["dur"]
    k = ease(p * 2.2)
    # two chips sliding together into one
    lx = W / 2 - 420 + 260 * k
    rx = W / 2 + 420 - 260 * k
    for cx, lab, col in ((lx, "Hindi", AMBER), (rx, "English", VIOLET)):
        rounded(d, (cx - 150, 340, cx + 150, 440), 50, fill=C(col, 0.25), outline=col, width=3)
        text(d, (cx, 390), lab, 52, INK, anchor="mm", weight=600)
    k2 = ease((p - 0.45) * 3)
    if k2 > 0:
        text(d, (W / 2, 560), "one sentence, two languages", 46, MUTED, anchor="mm", alpha=k2)
    k3 = ease((p - 0.7) * 4)
    if k3 > 0:
        cx, cy = W / 2, 720
        rounded(d, (cx - 330, cy - 60, cx + 330, cy + 60), 30, fill=C(CORAL, 0.18 * k3), outline=C(CORAL, k3), width=3)
        text(d, (cx, cy), "most voice tools:  “sorry, I didn’t get that”", 38, C(INK, k3), anchor="mm")


def badge(d, cx, cy, label, col, k):
    w = d.textlength(label, font=font(40)) + 80
    rounded(d, (cx - w / 2, cy - 42, cx + w / 2, cy + 42), 42, fill=C(col, 0.2 * k), outline=C(col, k), width=3)
    text(d, (cx, cy), label, 40, INK, anchor="mm", alpha=k)


def logo(d, cx, cy, size, k=1.0, t=0.0):
    text(d, (cx, cy), "Bolo", size, INK, anchor="mm", weight=800, alpha=k)
    # three little sound bars after the word
    fw = d.textlength("Bolo", font=font(size, weight=800))
    for i in range(3):
        h = size * (0.25 + 0.2 * abs(math.sin(t * 4 + i)))
        x0 = cx + fw / 2 + 24 + i * size * 0.16
        d.rounded_rectangle((x0, cy - h / 2, x0 + size * 0.09, cy + h / 2), 6, fill=C(TEAL, k))


def s_intro(img, t, x):
    d = ImageDraw.Draw(img)
    p = (t - x["start"]) / x["dur"]
    k = ease_out(p * 3)
    logo(d, W / 2 - 40, 360 - 20 * (1 - k), 190, k, t)
    text(d, (W / 2, 520), "understands Hinglish voice commands, on your own device", 44, MUTED, anchor="mm", alpha=ease((p - 0.15) * 3))
    for i, (lab, col) in enumerate((("runs on your laptop", TEAL), ("no internet needed", VIOLET), ("free and open source", AMBER))):
        badge(d, W / 2 + (i - 1) * 520, 700, lab, col, ease((p - 0.45 - i * 0.12) * 4))


STAGES = [("You speak", "", TEAL), ("Bolo Sun", "speech to text  ·  142 MB", TEAL),
          ("Bolo Tiny", "text to intent  ·  0.8 MB", VIOLET), ("Grounding", "does it exist here?", AMBER)]


def pipeline(img, t, active, sub_k=1.0):
    d = ImageDraw.Draw(img)
    y = 330
    xs = [W / 2 + (i - 1.5) * 440 for i in range(4)]
    for i in range(3):
        on = active > i
        d.line((xs[i] + 170, y, xs[i + 1] - 170, y), fill=TEAL if on else DIM, width=5)
        d.polygon([(xs[i + 1] - 170, y), (xs[i + 1] - 188, y - 11), (xs[i + 1] - 188, y + 11)], fill=TEAL if on else DIM)
    for i, (name, sub, col) in enumerate(STAGES):
        on = i <= active
        hot = i == active
        box = (xs[i] - 170, y - 95, xs[i] + 170, y + 95)
        rounded(d, box, 26, fill=C(col, 0.22 if hot else (0.1 if on else 0.04)), outline=col if on else DIM, width=4 if hot else 2)
        if i == 0:
            waveform(d, xs[i], y - 10, 220, 90, t, TEAL if on else DIM, energy=1.0 if hot else 0.4, bars=22)
            text(d, (xs[i], y + 62), name, 34, INK if on else MUTED, anchor="mm", weight=600)
        else:
            text(d, (xs[i], y - 22), name, 44, INK if on else MUTED, anchor="mm", weight=700)
            text(d, (xs[i], y + 34), sub, 26, MUTED if on else DIM, anchor="mm")
    return xs, y


def s_pipeline(img, t, x):
    stage = {"04_sun": 1, "05_tiny": 2, "06_ground": 3}[x["id"]]
    p = (t - x["start"]) / x["dur"]
    xs, y = pipeline(img, t, stage)
    d = ImageDraw.Draw(img)
    ex = demo[0] if demo else {"transcript": "rust learning folder mein terminal kholo", "intent": "open_terminal",
                               "slots": {"folder": "rust learning"}}
    cy = 600
    if stage >= 1:
        k = ease(p * 3) if stage == 1 else 1
        s = "“" + ex["transcript"] + "”"
        n = int(len(s) * k)
        text(d, (W / 2, cy), s[:n], 50, INK, anchor="mm", kind="mono")
    if stage >= 2:
        k = ease(p * 3) if stage == 2 else 1
        slots = ", ".join(f"{a}: “{b}”" for a, b in ex["slots"].items())
        rounded(d, (W / 2 - 520, cy + 70, W / 2 + 520, cy + 170), 22, fill=C(VIOLET, 0.15 * k), outline=C(VIOLET, k))
        text(d, (W / 2, cy + 120), f"intent: {ex['intent']}    {slots}", 38, INK, anchor="mm", kind="mono", alpha=k)
    if stage == 3:
        k1, k2 = ease((p - 0.05) * 4), ease((p - 0.5) * 4)
        text(d, (W / 2 - 40, cy + 225), "✓  it exists  →  Bolo hands back the action", 36, TEAL, anchor="rm", alpha=k1)
        text(d, (W / 2 + 40, cy + 225), "?  not sure  →  it asks you", 36, AMBER, anchor="lm", alpha=k2)


def terminal(img, t, x):
    d = ImageDraw.Draw(img)
    p = t - x["start"]
    box = (180, 150, W - 180, H - 170)
    rounded(d, box, 22, fill=(8, 11, 20), outline=(48, 56, 80), width=2)
    for i, c in enumerate((CORAL, AMBER, TEAL)):
        d.ellipse((box[0] + 28 + i * 34, box[1] + 22, box[0] + 48 + i * 34, box[1] + 42), fill=c)
    text(d, (W / 2, box[1] + 32), "Terminal  ·  MacBook Air M2  ·  offline", 24, MUTED, anchor="mm")
    yy = box[1] + 90
    per = 3.0
    for i, ex in enumerate(demo[:3]):
        t0 = i * per
        if p < t0:
            break
        k = ease((p - t0) * 2.5)
        cmd = f"$ bolo understand --audio {ex['clip_name']}"
        text(d, (box[0] + 50, yy), cmd[: int(len(cmd) * k)], 30, INK, kind="mono")
        yy += 48
        if p - t0 > 0.6:
            k2 = ease((p - t0 - 0.6) * 3)
            text(d, (box[0] + 50, yy), f"heard   “{ex['transcript']}”", 28, MUTED, kind="mono", alpha=k2)
            yy += 42
            if ex["abstain"]:
                res_s = f"→ asks  ({ex['reason']})"
                col = AMBER
            else:
                res_s = f"→ {ex['intent']}  " + "  ".join(f"{a}={b}" for a, b in ex["slots"].items())
                col = TEAL
            text(d, (box[0] + 50, yy), res_s + f"     {ex['ms']:.0f} ms", 28, col, kind="mono", alpha=k2)
            yy += 70
    text(d, (W / 2, H - 140), "real outputs  ·  input clips are macOS synthetic voices", 24, DIM, anchor="mm")


def s_numbers(img, t, x):
    d = ImageDraw.Draw(img)
    p = (t - x["start"]) / x["dur"]
    text(d, (W / 2, 190), "Correct action on 240 held out commands", 46, INK, anchor="mm", weight=700)
    text(d, (W / 2, 248), "synthetic macOS voices never used in training  ·  speech to action, end to end", 28, MUTED, anchor="mm")
    x0, x1 = 620, W - 400
    rows = [("Bolo", NUM["bolo"], TEAL)]
    if "swift" in NUM:
        rows.append(("its base model + Bolo decoding", NUM["swiftr"], (96, 106, 140)))
        rows.append(("its base model, plain", NUM["swift"], DIM))
    for i, (lab, v, col) in enumerate(rows):
        y = 390 + i * 130
        k = ease_out((p - 0.05 - i * 0.12) * 3)
        text(d, (x0 - 30, y), lab, 34, INK, anchor="rm", weight=600)
        d.rounded_rectangle((x0, y - 42, x1, y + 42), 16, fill=(24, 30, 50))
        d.rounded_rectangle((x0, y - 42, x0 + (x1 - x0) * v / 100 * k, y + 42), 16, fill=col)
        text(d, (x1 + 36, y), f"{v * k:.0f}%", 54, INK, anchor="lm", weight=700)
    cx = x0 + (x1 - x0) * NUM["ceiling"] / 100
    for yy in range(330, 700, 22):
        d.line((cx, yy, cx, yy + 11), fill=AMBER, width=3)
    text(d, (cx, 730), f"ceiling with a perfect transcript: {NUM['ceiling']:.0f}%", 26, AMBER, anchor="mm")
    text(d, (W / 2, 830), f"median {NUM['ms']:.0f} ms per command on a MacBook Air M2  ·  real human numbers: coming", 28, MUTED, anchor="mm")


def s_limits(img, t, x):
    d = ImageDraw.Draw(img)
    p = (t - x["start"]) / x["dur"]
    text(d, (W / 2, 230), "Honest limits", 60, INK, anchor="mm", weight=700)
    items = ["research preview, version 0.1", "tested on only a few real voices so far",
             "a little worse than its base model on long general speech", "nine command types today, not open conversation"]
    for i, s in enumerate(items):
        k = ease((p - 0.08 - i * 0.16) * 4)
        y = 380 + i * 105
        d.ellipse((560, y - 12, 584, y + 12), fill=C(AMBER, k))
        text(d, (620, y), s, 42, INK, anchor="lm", alpha=k)


def s_close(img, t, x):
    d = ImageDraw.Draw(img)
    p = (t - x["start"]) / max(1, x["end"] - x["start"])
    k = ease_out(p * 3)
    logo(d, W / 2 - 40, 400, 200, k, t)
    text(d, (W / 2, 560), "Hinglish voice commands, on device", 46, MUTED, anchor="mm", alpha=ease((p - 0.1) * 3))
    text(d, (W / 2, 660), "open source  ·  Apache 2.0  ·  free", 40, TEAL, anchor="mm", alpha=ease((p - 0.25) * 3))


SCENES = {"01_hook": s_hook, "02_problem": s_problem, "03_intro": s_intro, "04_sun": s_pipeline, "05_tiny": s_pipeline,
          "06_ground": s_pipeline, "07_demo": terminal, "08_numbers": s_numbers, "10_limits": s_limits, "11_close": s_close}


def frame(t):
    img = bg(t)
    x = scene_at(t)
    # fade between scenes
    fade = min(ease((t - x["start"] + 0.3) / 0.45), ease((x["end"] - t) / 0.35))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    SCENES.get(x["id"], s_close)(layer, t, x)
    if fade < 1:
        layer.putalpha(layer.getchannel("A").point(lambda v: int(v * max(0.0, fade))))
    img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    caption(img, t)
    g = min(1.0, t / 0.6, (TOTAL - t) / 0.8)
    if g < 1:
        img = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), img, max(0.0, g))
    return img


# ------------------------------------------------------------------ audio
def load_wav(p: Path) -> np.ndarray:
    import soundfile as sf

    a, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:
        idx = np.arange(0, len(a), sr / SR)
        a = np.interp(idx, np.arange(len(a)), a).astype("float32")
    return a


def soundtrack() -> np.ndarray:
    n = int(TOTAL * SR)
    out = np.zeros(n, "float32")
    # a quiet synthesised pad (no third party music): a slow major chord with a soft swell
    tt = np.arange(n) / SR
    pad = sum(np.sin(2 * np.pi * f * tt) * a for f, a in ((110, 0.5), (164.8, 0.35), (220, 0.3), (277.2, 0.18)))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * tt / 16)
    out += (pad * env * 0.025).astype("float32")
    out *= np.minimum(1, np.minimum(tt / 2, (TOTAL - tt) / 2)).astype("float32")
    for x in lines:
        if x["audio"]:
            a = load_wav(x["audio"])
            a = a / max(1e-6, np.abs(a).max()) * 0.85
            s = int(x["start"] * SR)
            out[s:s + len(a)] += a[: n - s]
    # demo clips: the real input audio, under the terminal scene
    if "07_demo" in SC:
        for i, ex in enumerate(demo[:3]):
            p = ROOT / ex["clip"]
            if p.exists():
                a = load_wav(p) * 0.8
                s = int((SC["07_demo"]["start"] + i * 3.0 + 0.1) * SR)
                out[s:s + len(a)] += a[: max(0, n - s)]
    return np.clip(out, -1, 1)


def render():
    import imageio_ffmpeg

    OUT.mkdir(parents=True, exist_ok=True)
    wav = OUT / "soundtrack.wav"
    a = (soundtrack() * 32767).astype("<i2")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(a.tobytes())
    mp4 = OUT / "bolo_launch.mp4"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(wav), "-c:v", "libx264", "-preset", "medium",
           "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
           str(mp4)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(TOTAL * FPS)
    for i in range(n):
        pr.stdin.write(frame(i / FPS).tobytes())
        if i % 300 == 0:
            print(f"{i}/{n} frames", flush=True)
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
            frame(tt).save(OUT / f"frame_{tt:05.1f}.png")
        print("timeline:", [(x["id"], round(x["start"], 1)) for x in lines], "total", round(TOTAL, 1))
    else:
        render()
