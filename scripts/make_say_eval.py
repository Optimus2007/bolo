"""A second, *different* synthetic speech set: macOS `say` voices reading the 80 `read` recording prompts.

Useful as a TTS-mismatch check (Parler-trained models vs. an unseen synthesizer) and for testing
`bolo eval` without human audio. Output: data/say/ (gitignored). macOS only, about 1 minute.
"""
import json
import subprocess
from pathlib import Path

from bolo.record.prompts import build_prompts

OUT = Path("data/say")
VOICES_HI, VOICES_EN = ["Lekha"], ["Aman", "Tara", "Rishi"]
(OUT / "audio").mkdir(parents=True, exist_ok=True)
prompts = [p for p in build_prompts("data/text/v0/test.jsonl") if p["kind"] == "read"]
with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as f:
    for i, p in enumerate(prompts):
        hi = i % 2 == 0  # alternate: Hindi voice reads mixed-script text, en_IN voices read Roman
        voice = VOICES_HI[0] if hi else VOICES_EN[i // 2 % len(VOICES_EN)]
        text = p["hint"] if hi else p["asr_text"]
        rel = f"audio/{p['prompt_id']}.wav"
        subprocess.run(["say", "-v", voice, "-o", str(OUT / rel), "--data-format=LEI16@16000", text], check=True)
        import wave
        with wave.open(str(OUT / rel)) as w:
            dur = w.getnframes() / w.getframerate()
        f.write(json.dumps({"utt_id": f"say__{p['prompt_id']}", "audio": rel, "voice": voice, "kind": "read",
                            "source_id": p["source_id"], "prompt_id": p["prompt_id"], "intent": p["intent"],
                            "slots": p["slots"], "transcript": p["asr_text"], "duration_s": round(dur, 3),
                            "tts_model": f"macos-say:{voice}"}, ensure_ascii=False) + "\n")
print(f"{len(prompts)} clips -> {OUT}")
