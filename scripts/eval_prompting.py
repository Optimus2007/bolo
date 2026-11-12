"""Does a vocabulary prompt help Bolo Sun on rare names? Compares no prompt vs a prompt of known entity names,
on the macOS `say` set (80 clips). Writes results/speech/prompting.json."""
import json
from pathlib import Path

from bolo.config import load_config
from bolo.speech_eval import WhisperASR, load_set, run

cfg = load_config()
rows = load_set("data/say/manifest.jsonl", "data/text/v1/test.jsonl", "say")
P = cfg.open_pools
apps = ", ".join(e.value for e in P["apps"])
words = "terminal, claude code, folder, volume, brightness, battery, wifi, bluetooth"
prompts = {"none": None, "commands": words, "commands+apps": words + ", " + apps}
res = {}
for name, pr in prompts.items():
    out = run({"say_read": rows}, [f"bolo-fp16[{name}]"], ["tiny"], cfg,
              asr_factory=lambda _n, pr=pr: WhisperASR("models/bolo-whisper-fp16", "en", prompt=pr))
    A = out["sets"]["say_read"]["asr"][f"bolo-fp16[{name}]"]
    res[name] = {"wer": A["wer"]["wer"], "phonetic_wer": A["wer"]["phonetic_wer"],
                 "e2e_em": A["nlu"]["tiny"]["grounded"]["exact_match"], "ms": A["latency_ms"]["median"],
                 "prompt_chars": len(pr or "")}
    print(name, res[name], flush=True)
Path("results/speech/prompting.json").write_text(json.dumps(res, indent=1))
