"""Numpy-only audio augmentation for synthetic (TTS) training speech.

Goal: make clean TTS audio look more like a laptop/phone mic in a room, so the ASR fine-tune does not
learn "TTS acoustics". All ops are float32 mono at 16 kHz, deterministic given a numpy Generator.

  speed       : resample-based speed perturbation (changes tempo and pitch, as in Kaldi 0.9/1.0/1.1)
  reverb      : synthetic room impulse response (exponential-decay noise tail, RT60 0.15-0.7 s)
  add_noise   : additive noise at a target SNR (coloured synthetic noise, or real noise clips if given)
  bandlimit   : telephone / cheap-mic band (FFT mask, e.g. 300-3400 Hz)
  gain        : random level change, with clipping protection
  augment     : a random chain of the above, returns (audio, params) so every sample is reproducible

No scipy/ffmpeg dependency on purpose (not available on the dev machine).
"""

from __future__ import annotations

import io
import wave

import numpy as np

SR = 16000


# ---------- IO ----------
def read_wav(path_or_bytes) -> tuple[np.ndarray, int]:
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    with wave.open(src) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError("only 16-bit PCM supported")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_wav(path, x: np.ndarray, sr: int = SR) -> None:
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Band-limited resampling via FFT (fine for short utterances)."""
    if sr_in == sr_out or len(x) == 0:
        return x.astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    if n_out < 2:
        return np.zeros(max(0, n_out), dtype=np.float32)
    X = np.fft.rfft(x)
    n_bins = n_out // 2 + 1
    Y = np.zeros(n_bins, dtype=complex)
    k = min(len(X), n_bins)
    Y[:k] = X[:k]
    return (np.fft.irfft(Y, n=n_out) * (n_out / len(x))).astype(np.float32)


# ---------- ops ----------
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def speed(x: np.ndarray, factor: float) -> np.ndarray:
    """factor > 1 = faster/shorter. Implemented as resampling then reinterpreting at SR."""
    if abs(factor - 1.0) < 1e-6:
        return x
    n_out = int(round(len(x) / factor))
    t = np.linspace(0, len(x) - 1, n_out)
    return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def room_ir(rng: np.random.Generator, rt60: float, sr: int = SR, direct_db: float = 0.0) -> np.ndarray:
    n = int(sr * min(1.0, rt60 * 1.2))
    t = np.arange(n) / sr
    decay = np.exp(-6.9078 * t / rt60)  # -60 dB at rt60
    tail = rng.standard_normal(n) * decay
    # a few early reflections
    for _ in range(4):
        d = rng.integers(int(0.002 * sr), int(0.03 * sr))
        if d < n:
            tail[d] += rng.uniform(0.3, 0.7) * (1 if rng.random() < 0.5 else -1)
    tail[0] = 0.0
    tail /= np.sqrt(np.sum(tail**2)) + 1e-9
    h = tail.copy()
    h[0] = 10 ** (direct_db / 20)
    return h.astype(np.float32)


def reverb(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = len(x) + len(h) - 1
    nfft = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(h, nfft), nfft)[: len(x)]
    return (y * (rms(x) / (rms(y) + 1e-9))).astype(np.float32)


def coloured_noise(rng: np.random.Generator, n: int, colour: str = "pink") -> np.ndarray:
    exp = {"white": 0.0, "pink": 1.0, "brown": 2.0}[colour]
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if n > 1 else 1.0
    spec /= f ** (exp / 2)
    y = np.fft.irfft(spec, n)
    return (y / (rms(y) + 1e-9)).astype(np.float32)


def hum(n: int, sr: int = SR, f0: float = 50.0) -> np.ndarray:
    t = np.arange(n) / sr
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3))
    return (y / (rms(y) + 1e-9)).astype(np.float32)


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    noise = noise[: len(x)]
    scale = rms(x) / (rms(noise) * 10 ** (snr_db / 20))
    return (x + scale * noise).astype(np.float32)


def bandlimit(x: np.ndarray, lo: float = 300.0, hi: float = 3400.0, sr: int = SR) -> np.ndarray:
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    # smooth (raised-cosine) edges, 100 Hz wide, to avoid ringing
    m = np.clip((f - (lo - 100)) / 100, 0, 1) * np.clip(((hi + 100) - f) / 100, 0, 1)
    m = 0.5 - 0.5 * np.cos(np.pi * m)
    return np.fft.irfft(X * m, len(x)).astype(np.float32)


def gain(x: np.ndarray, db: float) -> np.ndarray:
    y = x * 10 ** (db / 20)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    return (y / peak * 0.99 if peak > 0.99 else y).astype(np.float32)


# ---------- chain ----------
DEFAULT = {
    "p_speed": 0.6, "speeds": (0.9, 1.0, 1.1),
    "p_reverb": 0.5, "rt60": (0.15, 0.7),
    "p_noise": 0.8, "snr": (5.0, 30.0),
    "p_band": 0.2,
    "gain_db": (-12.0, 3.0),
}


def augment(x: np.ndarray, rng: np.random.Generator, noises: list[np.ndarray] | None = None,
            cfg: dict | None = None) -> tuple[np.ndarray, dict]:
    c = {**DEFAULT, **(cfg or {})}
    p: dict = {}
    if rng.random() < c["p_speed"]:
        p["speed"] = float(rng.choice(c["speeds"]))
        x = speed(x, p["speed"])
    if rng.random() < c["p_reverb"]:
        p["rt60"] = round(float(rng.uniform(*c["rt60"])), 3)
        x = reverb(x, room_ir(rng, p["rt60"]))
    if rng.random() < c["p_noise"]:
        p["snr_db"] = round(float(rng.uniform(*c["snr"])), 1)
        if noises and rng.random() < 0.7:
            i = int(rng.integers(len(noises)))
            nz = noises[i]
            off = int(rng.integers(max(1, len(nz) - len(x))))
            p["noise"] = f"clip{i}"
            nz = nz[off:]
        else:
            kind = str(rng.choice(["white", "pink", "brown", "hum"]))
            p["noise"] = kind
            nz = hum(len(x)) + 0.1 * coloured_noise(rng, len(x), "white") if kind == "hum" \
                else coloured_noise(rng, len(x), kind)
        x = add_noise(x, nz, p["snr_db"])
    if rng.random() < c["p_band"]:
        p["band"] = [300, 3400]
        x = bandlimit(x)
    p["gain_db"] = round(float(rng.uniform(*c["gain_db"])), 1)
    x = gain(x, p["gain_db"])
    return x, p
