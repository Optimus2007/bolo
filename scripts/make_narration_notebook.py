"""Write notebooks/04_narration.ipynb: the launch video narration with Indic Parler TTS (female voices, Apache 2.0).

Reads launch/narration.json. Lines whose text still contains a {PLACEHOLDER} or is empty are skipped.
Each line gets 3 takes; whisper-small transcribes each take and the most intelligible one is kept (lowest WER).
Push with `python scripts/kaggle_push.py narr`, then start it from the Kaggle UI (the gated model needs the
HF_TOKEN secret, which API-started runs cannot read).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_notebooks import HF_LOGIN, nb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = json.loads((ROOT / "launch/narration.json").read_text(encoding="utf-8"))
sw = json.loads((ROOT / "results/speech/swift_heldout.json").read_text())
fill = {"SWIFT": f"{sw['greedy']['e2e'] * 100:.0f}", "SWIFTR": f"{sw['beam5_rerank']['e2e'] * 100:.0f}"}
lines = []
for x in spec["lines"]:
    t = re.sub(r"\{([A-Z]+)\}", lambda m: fill.get(m.group(1), m.group(0)), x["text"])
    if t.strip() and not re.search(r"\{[A-Z]+\}", t):
        lines.append(x | {"text": t})
book = nb([
    ("markdown", "# Bolo 04: launch video narration (Indic Parler TTS)\n\nNeeds the `HF_TOKEN` secret. GPU T4. "
                 "Output: `/kaggle/working/narration/<id>.wav` (best take) and `takes/`, plus `scores.json`."),
    ("code", "!pip install -q git+https://github.com/huggingface/parler-tts.git soundfile jiwer"),
    ("code", 'import os\nos.environ["USE_TF"] = "0"; os.environ["USE_TORCH"] = "1"; os.environ["USE_JAX"] = "0"; os.environ["USE_FLAX"] = "0"'),
    ("code", "import json, os\nWORK = '/kaggle/working' if os.path.exists('/kaggle/working') else '/content'\n" + HF_LOGIN),
    ("code", "SPEC = " + json.dumps({"voices": spec["voices"], "lines": lines}, ensure_ascii=False, indent=1)),
    ("code", r'''
import torch, soundfile as sf, numpy as np, jiwer, re
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, pipeline
assert not SMOKE, "needs the gated Indic Parler model: start this run from the Kaggle UI with the HF_TOKEN secret"
dev = "cuda"
model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL).to(dev)
tok = AutoTokenizer.from_pretrained(MODEL)
desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
asr = pipeline("automatic-speech-recognition", "openai/whisper-small", device=0)
SR = model.config.sampling_rate
OUT = WORK + "/narration"; os.makedirs(OUT + "/takes", exist_ok=True)
norm = lambda s: re.sub(r"[^a-z0-9ऀ-ॿ ]", " ", s.lower()).split()
scores = {}
for line in SPEC["lines"]:
    d = desc_tok(SPEC["voices"][line["voice"]], return_tensors="pt").to(dev)
    p = tok(line["text"], return_tensors="pt").to(dev)
    takes = []
    for seed in range(3):
        torch.manual_seed(seed)
        with torch.no_grad():
            a = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                               prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask, do_sample=True)
        a = a.cpu().float().numpy().squeeze()
        f = f"{OUT}/takes/{line['id']}_s{seed}.wav"
        sf.write(f, a, SR)
        lang = "hi" if line["voice"] == "hinglish" else "en"
        hyp = asr(f, generate_kwargs={"language": lang, "task": "transcribe"})["text"]
        w = jiwer.wer(" ".join(norm(line["text"])) or "x", " ".join(norm(hyp)) or "x") if lang == "en" else None
        takes.append({"seed": seed, "wer": w, "dur": round(len(a) / SR, 2), "heard": hyp})
        print(line["id"], seed, w, round(len(a) / SR, 2), hyp[:90], flush=True)
    best = min(takes, key=lambda t: (t["wer"] if t["wer"] is not None else 0, abs(t["dur"] - len(line["text"]) / 14)))
    os.replace(f"{OUT}/takes/{line['id']}_s{best['seed']}.wav", f"{OUT}/{line['id']}.wav")
    scores[line["id"]] = {"best": best, "takes": takes}
json.dump({"sr": SR, "scores": scores}, open(OUT + "/scores.json", "w"), indent=1, ensure_ascii=False)
'''),
])
(ROOT / "notebooks/04_narration.ipynb").write_text(json.dumps(book, indent=1, ensure_ascii=False))
print(len(lines), "lines ->", "notebooks/04_narration.ipynb")
