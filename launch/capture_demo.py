"""Run the real Bolo pipeline on a few held-out macOS voice clips and save the outputs for the video demo scene.
Picks, in order: an open terminal command, a volume command, and one where Bolo correctly declines.
Writes launch/demo.json. The clips are synthetic (macOS `say`), and the video says so."""
import json
from pathlib import Path

from bolo.config import load_config
from bolo.evaluate import _exact
from bolo.pipeline import Bolo
from bolo.speech_eval import environment, load_set

rows = load_set("data/say_test/manifest.jsonl", "data/text/v1/test.jsonl", "say_test")
b = Bolo.load(asr="models/bolo-whisper-fp16", env=environment(load_config()))
b.listen(rows[0]["_wav"])  # warm-up
want = [("open_terminal", "terminal.wav"), ("set_volume", "volume.wav"), ("none", "not_a_command.wav")]
out = []
for intent, name in want:
    for r in rows:
        if r["intent"] != intent or len(r["ref"]) > 48:
            continue
        o = b.listen(r["_wav"])
        if _exact(r, o):
            out.append({"clip": r["_wav"], "clip_name": name, "transcript": o["transcript"],
                        "intent": o["intent"], "slots": o["slots"], "abstain": o["abstain"], "reason": o["reason"],
                        "ms": o["ms"]["asr"] + o["ms"]["nlu"]})
            print(out[-1])
            break
Path("launch/demo.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
