"""Training rows for Bolo Tiny made from what the speech model actually hears.

For every TTS clip of the TRAIN split (never dev/test), run Bolo Sun with 5-beam search and keep each distinct
hypothesis that differs from the clean text. The gold intent is kept; each gold slot is re-located in the noisy
hypothesis by sound (best window of words, similarity >= MIN_SIM). Closed slots carry their value in the tag, so
"spotify pe gaana premium karo" becomes a row where "premium" is tagged action=previous. Rows where a slot cannot be
found are dropped (label not trustworthy). Output: data/text/v1/train_asr.jsonl, capped at the size of train.jsonl
so clean and noisy text are mixed about 1:1 (research: ASR-only training hurts clean accuracy).

  .venv/bin/python scripts/make_asr_noised.py --tts data/kaggle/out/bolo-tts-synth_check/bolo-tts
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bolo.audio.augment import SR, augment, read_wav, resample  # noqa: E402
from bolo.config import load_config  # noqa: E402
from bolo.grounding import similarity  # noqa: E402
from bolo.normalize import Normalizer  # noqa: E402
from bolo.pipeline import BEAMS, build_prompt  # noqa: E402
from bolo.speech_eval import WhisperASR  # noqa: E402

MIN_SIM = 0.5

ap = argparse.ArgumentParser()
ap.add_argument("--tts", required=True)
ap.add_argument("--data", default="data/text/v1")
ap.add_argument("--asr", default="models/bolo-whisper-fp16")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--name", default="train_asr", help="output file stem in --data")
ap.add_argument("--augment", action="store_true",
                help="add noise/echo/speed (bolo.audio.augment) before transcribing: needed when --asr was trained on these clips")
a = ap.parse_args()

cfg = load_config()
norm = Normalizer.from_config(cfg)
train = {r["id"]: r for r in map(json.loads, open(ROOT / a.data / "train.jsonl"))}
mp = Path(a.tts) / "manifest.jsonl"
if mp.exists():
    man = [json.loads(l) for l in open(mp)]
else:  # partial download (run cancelled): clip tts-train-00019 was made from text row train-00019
    man = [{"utt_id": f.stem, "source_id": f.stem[4:], "split": "train", "audio": f"audio/{f.name}",
            "intent": train[f.stem[4:]]["intent"], "asr_text": train[f.stem[4:]]["text"]}
           for f in sorted((Path(a.tts) / "audio").glob("tts-train-*.wav")) if f.stem[4:] in train]
    print(f"no manifest: {len(man)} train clips mapped by file name (mapping checked by hand on 20 clips)")
man = [m for m in man if m.get("split") == "train" and (Path(a.tts) / m["audio"]).exists()]
if a.limit:
    man = man[: a.limit]
asr = WhisperASR(str(ROOT / a.asr), "en", prompt=build_prompt("commands"))


def roman(s):
    return [norm.to_roman(w) for w, _, _ in norm.words(s)]


def locate(hyp_words, used, surface):
    """Best window of hypothesis words for a gold slot surface; returns (i, j, score)."""
    k = len(surface)
    best = (None, None, 0.0)
    for n in range(max(1, k - 1), k + 2):
        for i in range(0, len(hyp_words) - n + 1):
            if any(x in used for x in range(i, i + n)):
                continue
            sc = similarity(" ".join(w for w, _, _ in hyp_words[i:i + n]), " ".join(surface))
            if sc > best[2]:
                best = (i, i + n, sc)
    return best


rng = np.random.default_rng(0)
out, kept, dropped, same, stale, t0 = [], 0, 0, 0, 0, time.perf_counter()
for n, m in enumerate(man):
    g = train.get(m["source_id"])
    if g is None or g["intent"] != m["intent"] or similarity(" ".join(roman(m["asr_text"])), " ".join(roman(g["text"]))) < 0.8:
        stale += 1
        continue  # the clip was made from different text (data regenerated since): its labels do not apply
    x, sr = read_wav(str(Path(a.tts) / m["audio"]))
    x = resample(x, sr, SR)
    if a.augment:
        x, _ = augment(x, rng)
    hyps = asr.nbest(x, beams=BEAMS)
    clean = " ".join(roman(g["text"]))
    for h, _ in hyps:
        hw = [(norm.to_roman(w), s, e) for w, s, e in norm.words(h)]
        if " ".join(w for w, _, _ in hw) == clean:
            same += 1
            continue
        spans, used, ok = [], set(), True
        for sp in g["spans"]:
            i, j, sc = locate(hw, used, roman(g["text"][sp["start"]:sp["end"]]))
            if i is None or sc < MIN_SIM:
                ok = False
                break
            used.update(range(i, j))
            spans.append({"slot": sp["slot"], "start": hw[i][1], "end": hw[j - 1][2], "value": sp["value"]})
        if not ok:
            dropped += 1
            continue
        out.append({"id": f"{g['id']}-asr{len(out)}", "text": h, "script": "asr", "intent": g["intent"],
                    "slots": g["slots"], "spans": sorted(spans, key=lambda s: s["start"]),
                    "meta": {"source": "asr_nbest", "clip": m["utt_id"], "asr": a.asr, "augmented": a.augment}})
        kept += 1
    if n % 200 == 0:
        print(f"{n}/{len(man)} kept {kept} dropped {dropped} same-as-clean {same} ({time.perf_counter() - t0:.0f}s)", flush=True)
random.Random(0).shuffle(out)
out = out[: len(train)]
(ROOT / a.data / f"{a.name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")
print(f"wrote {len(out)} rows -> {a.data}/{a.name}.jsonl (kept {kept}, dropped {dropped}, same-as-clean {same}, stale clips {stale})")
