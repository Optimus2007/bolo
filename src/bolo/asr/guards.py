"""Cheap checks that stop the assistant acting on a transcript the recogniser made up.

Research (docs/research/REAL_VOICE_ACCURACY.md): Whisper's own heuristics (compression ratio 2.4, avg logprob -1)
barely catch hallucinations on short commands; voice activity, a token cap from speech duration and a speaking-rate
check are the cheap, effective ones. The VAD takes ~10 ms; the rest is negligible.

  speech_seconds(x)        how much of the clip is actually speech (Silero VAD)
  max_tokens(sec)          decode budget for that much speech
  rate_ok(text, sec)       characters per second of speech within what people can say
"""
from __future__ import annotations

import numpy as np

SR = 16000
FRAME = 480  # 30 ms
MIN_SPEECH = 0.25  # seconds; below this nothing was said
# characters (letters/digits, no spaces) per second of detected speech; set from computer-voice clips
# (scripts/calibrate_guards.py writes results/speech/guards.json) with wide margins, never from human test clips
RATE_MIN, RATE_MAX = 3.8, 28.8  # results/speech/guards.json


_VAD = None


def speech_seconds(x: np.ndarray, sr: int = SR) -> float:
    """Seconds of speech, by Silero VAD (MIT, ~2 MB, ~10 ms per clip on CPU). Falls back to an energy VAD, which
    under-counts soft speech on laptop mics, if silero-vad is not installed."""
    global _VAD
    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad

        if _VAD is None:
            _VAD = load_silero_vad()
        ts = get_speech_timestamps(torch.from_numpy(np.asarray(x, dtype="float32")), _VAD, sampling_rate=sr)
        return sum(t["end"] - t["start"] for t in ts) / sr
    except ImportError:
        return energy_speech_seconds(x, sr)


def energy_speech_seconds(x: np.ndarray, sr: int = SR) -> float:
    x = np.asarray(x, dtype="float32")
    n = len(x) // FRAME
    if n == 0:
        return 0.0
    fr = x[: n * FRAME].reshape(n, FRAME)
    db = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-10)
    floor = np.percentile(db, 10)
    peak = np.percentile(db, 99)
    thr = max(floor + 10.0, peak - 35.0, -60.0)
    return float(np.sum(db > thr)) * FRAME / sr


def max_tokens(sec: float) -> int:
    """Hinglish commands are ~3 words/s, ~1.5 tokens per word; generous so real speech is never cut."""
    return int(min(64, 10 + 9 * sec))


def chars(text: str) -> int:
    return sum(c.isalnum() for c in text)


def rate_ok(text: str, sec: float, lo: float = RATE_MIN, hi: float = RATE_MAX) -> bool:
    if sec <= 0:
        return False
    r = chars(text) / sec
    return lo <= r <= hi
