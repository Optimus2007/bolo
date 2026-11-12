"""A second, *different* synthetic speech set: macOS `say` voices reading Bolo test commands.

  python scripts/make_say_eval.py                     # data/say/: the 80 `read` recording prompts (tuning set)
  python scripts/make_say_eval.py --heldout 240       # data/say_test/: 240 other test rows (final check)
  python scripts/make_say_eval.py --heldout 240 --shift 1   # data/say_test_v1/: same commands, other voices

Useful as a TTS-mismatch check (Parler-trained models vs. an unseen synthesizer) and for testing `bolo eval`
without human audio. The held-out set never overlaps the recording prompts, so decoding settings tuned on
data/say/ can be checked on data/say_test/. Output is gitignored. macOS only, about 1 to 3 minutes.
"""
import argparse
import json
import random
import subprocess
import wave
from collections import defaultdict
from pathlib import Path

from bolo.record.prompts import build_prompts

VOICES_HI, VOICES_EN = ["Lekha"], ["Aman", "Tara", "Rishi"]
TEST = "data/text/v0/test.jsonl"

ap = argparse.ArgumentParser()
ap.add_argument("--heldout", type=int, default=0, help="make N clips from test rows not used as recording prompts")
ap.add_argument("--shift", type=int, default=0, help="rotate voices (1 = the same commands said by other voices)")
a = ap.parse_args()

used = build_prompts(TEST)
if a.heldout:
    OUT = Path("data/say_test" + (f"_v{a.shift}" if a.shift else ""))
    taken = {p["source_id"] for p in used}
    by_intent = defaultdict(list)
    for line in open(TEST, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] not in taken:
            by_intent[r["intent"]].append(r)
    rng = random.Random(11)
    for v in by_intent.values():
        rng.shuffle(v)
    intents, rows, i = sorted(by_intent), [], 0
    while len(rows) < a.heldout and any(by_intent.values()):
        pool = by_intent[intents[i % len(intents)]]
        i += 1
        if pool:
            rows.append(pool.pop())
    prompts = [{"prompt_id": f"heldout-{k:03d}", "source_id": r["id"], "hint": r["tts_text"], "asr_text": r["asr_text"],
                "intent": r["intent"], "slots": r["slots"]} for k, r in enumerate(rows)]
else:
    OUT = Path("data/say")
    prompts = [p for p in used if p["kind"] == "read"]

(OUT / "audio").mkdir(parents=True, exist_ok=True)
with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as f:
    for i, p in enumerate(prompts):
        i += a.shift
        hi = i % 2 == 0  # alternate: Hindi voice reads mixed-script text, en_IN voices read Roman
        voice = VOICES_HI[0] if hi else VOICES_EN[i // 2 % len(VOICES_EN)]
        text = p["hint"] if hi else p["asr_text"]
        rel = f"audio/{p['prompt_id']}.wav"
        subprocess.run(["say", "-v", voice, "-o", str(OUT / rel), "--data-format=LEI16@16000", text], check=True)
        with wave.open(str(OUT / rel)) as w:
            dur = w.getnframes() / w.getframerate()
        f.write(json.dumps({"utt_id": f"say__{p['prompt_id']}", "audio": rel, "voice": voice, "kind": "read",
                            "source_id": p["source_id"], "prompt_id": p["prompt_id"], "intent": p["intent"],
                            "slots": p["slots"], "transcript": p["asr_text"], "duration_s": round(dur, 3),
                            "tts_model": f"macos-say:{voice}"}, ensure_ascii=False) + "\n")
print(f"{len(prompts)} clips -> {OUT}")
