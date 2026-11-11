"""Real human Hinglish speech check: OpenSLR-104 Hindi-English TEST (CC BY-SA 4.0, spoken tutorials).

ASR-only (these are not commands, so no intent/action metrics). References are mixed script, often with
English words written in Devanagari (ऑफिस), which favours Devanagari-output models on raw WER; the
phonetic WER (after transliteration) is the fairer cross-script comparison. Neither is perfect: report both.

Download (443 MB): https://www.openslr.org/resources/104/Hindi-English_test.tar.gz -> data/public/openslr104/
Usage: python scripts/eval_openslr104.py [--n 300] [--asr whisper-base,swift,bolo-whisper]
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from bolo.audio.augment import read_wav, resample, SR
from bolo.audio.metrics import corpus_scores
from bolo.config import load_config
from bolo.normalize import Normalizer
from bolo.speech_eval import WhisperASR, asr_spec, pct

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/public/openslr104/test")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--asr", default="whisper-base,swift,bolo-whisper")
ap.add_argument("--out", default="results/speech/openslr104_test.json")
a = ap.parse_args()

root = Path(a.root)
T = root / "transcripts"
text = dict(line.split(" ", 1) for line in (T / "text").read_text(encoding="utf-8").splitlines() if " " in line)
segs = [line.split() for line in (T / "segments").read_text().splitlines()]
utt2spk = dict(line.split() for line in (T / "utt2spk").read_text().splitlines())
# stratified sample: round-robin across recordings, segments <= 28 s
by_rec = defaultdict(list)
for u, rec, s, e in segs:
    if u in text and float(e) - float(s) <= 28:
        by_rec[rec].append((u, rec, float(s), float(e)))
rng = random.Random(0)
for v in by_rec.values():
    rng.shuffle(v)
sample, recs = [], sorted(by_rec)
while len(sample) < a.n and any(by_rec.values()):
    for r in recs:
        if by_rec[r] and len(sample) < a.n:
            sample.append(by_rec[r].pop())
cache = {}


def audio(rec, s, e):
    if rec not in cache:
        x, sr = read_wav(str(root / f"{rec}.wav"))
        cache.clear()
        cache[rec] = resample(x, sr, SR) if sr != SR else x
    x = cache[rec]
    return x[int(s * SR): int(e * SR)]


sample.sort(key=lambda t: t[1])  # group by recording for the cache
refs = [text[u].strip() for u, *_ in sample]
dur = sum(e - s for _, _, s, e in sample)
nz = Normalizer.from_config(load_config())
res = {"dataset": "OpenSLR-104 Hindi-English test (CC BY-SA 4.0)", "n": len(sample), "minutes": round(dur / 60, 1),
       "recordings": len({t[1] for t in sample}), "speakers": len({utt2spk.get(t[0]) for t in sample}), "asr": {}}
utts = []
for name in a.asr.split(","):
    path, lang = asr_spec(name)
    if path.startswith("models/") and not Path(path).exists():
        print("skip", name)
        continue
    asr = WhisperASR(path, lang)
    asr(np.zeros(SR, np.float32))  # warm-up
    hyps, ms = [], []
    for u, rec, s, e in sample:
        x = audio(rec, s, e)
        t = time.perf_counter()
        hyps.append(asr(x))
        ms.append((time.perf_counter() - t) * 1000)
    sc = corpus_scores(refs, hyps, nz)
    sc.update(median_ms=pct(ms, 0.5), rtf=round(sum(ms) / 1000 / dur, 3), language=lang)
    res["asr"][name] = sc
    print(name, json.dumps(sc), flush=True)
    utts += [{"utt": t[0], "asr": name, "ref": r, "hyp": h} for t, r, h in zip(sample, refs, hyps)]
    del asr
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
Path("data/eval").mkdir(parents=True, exist_ok=True)
Path("data/eval/openslr104_utts.jsonl").write_text("".join(json.dumps(u, ensure_ascii=False) + "\n" for u in utts))
