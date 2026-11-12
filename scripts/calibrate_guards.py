"""Set the speaking-rate limits for bolo.asr.guards from computer-voice clips (never from human test clips).

  .venv/bin/python scripts/calibrate_guards.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bolo.asr.guards import chars, speech_seconds  # noqa: E402
from bolo.audio.augment import SR, read_wav, resample  # noqa: E402

sets = [ROOT / "data/say/manifest.jsonl", ROOT / "data/say_test_v1/manifest.jsonl"]
tts = ROOT / "data/kaggle/out/bolo-tts-synth_check/bolo-tts"
rates, secs = [], []
for m in sets:
    for line in open(m):
        r = json.loads(line)
        x, sr = read_wav(str(m.parent / r["audio"]))
        s = speech_seconds(resample(x, sr, SR))
        ref = r.get("transcript") or r.get("asr_text") or r.get("text")
        if ref and s > 0:
            rates.append(chars(ref) / s)
            secs.append(s)
wavs = sorted((tts / "audio").glob("tts-dev-*.wav"))[:300]
man = {}
if (tts / "manifest.jsonl").exists():
    man = {Path(json.loads(l)["audio"]).name: json.loads(l) for l in open(tts / "manifest.jsonl")}
for w in wavs:
    r = man.get(w.name)
    if not r:
        continue
    x, sr = read_wav(str(w))
    s = speech_seconds(resample(x, sr, SR))
    if s > 0:
        rates.append(chars(r["asr_text"]) / s)
        secs.append(s)
rates = np.array(rates)
q = {p: round(float(np.percentile(rates, p)), 2) for p in (0, 0.5, 1, 5, 50, 95, 99, 99.5, 100)}
out = {"n": len(rates), "chars_per_speech_second": q, "speech_seconds_min": round(float(min(secs)), 2),
       "rate_min": round(q[0.5] * 0.5, 1), "rate_max": round(q[99.5] * 1.6, 1),
       "rule": "rate_min = 0.5 x 0.5th percentile, rate_max = 1.6 x 99.5th percentile, computer voices only"}
(ROOT / "results/speech/guards.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
