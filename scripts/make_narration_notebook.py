"""Write notebooks/04_narration.ipynb: the launch video narration with Indic Parler TTS (female voices, Apache 2.0).

Reads launch/narration.json. Lines whose text still contains a {PLACEHOLDER} or is empty are skipped.
Each line gets 3 takes; whisper-small transcribes each take and the most intelligible one is kept (lowest WER).
With --story it reads launch/story.json instead: per-beat style descriptions, 4 takes, and among takes within
0.1 WER of the best it keeps the one with the most pitch variation (least monotone).
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
STORY = "--story" in sys.argv  # the cartoon: launch/story.json, one expressive style per beat
if STORY:
    sys.argv.remove("--story")
    st = json.loads((ROOT / "launch/story.json").read_text(encoding="utf-8"))
    spec = {"voices": {}, "lines": [{"id": b["id"], "voice": b["voice"], "text": b["text"], "say": b.get("say") or [b["text"]],
             "desc": f"{st['voices'][b['voice']]} speaks in {b['style']}, with very clear audio and no background noise."}
            for b in st["beats"] if b["voice"] != "none"]}
else:
    spec = json.loads((ROOT / "launch/narration.json").read_text(encoding="utf-8"))
sw = json.loads((ROOT / "results/speech/swift_heldout.json").read_text())
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def words(n: int) -> str:
    """TTS reads digits badly ("59" came out garbled), so numbers are spelled out."""
    return ONES[n] if n < 20 else TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])


res = ROOT / "results/speech/launch_numbers.json"  # v1 speech eval; until it exists the board line is skipped
ln = json.loads(res.read_text()) if res.exists() else {}
def words_big(n: int) -> str:
    h, r = divmod(n, 100)
    return ((ONES[h] + " hundred" + (" and " if r else "")) if h else "") + (words(r) if r or not h else "")


fill = {**({"N_WORDS": words_big(ln["n"]), "E2E_WORDS": words(round(ln["e2e"]))} if ln else {}), "SWIFT": words(round(sw["greedy"]["e2e"] * 100)), "SWIFTR": words(round(sw["beam5_rerank"]["e2e"] * 100))}
ONLY = sys.argv[1].split(",") if len(sys.argv) > 1 else None  # e.g. 08_numbers: regenerate just these lines
lines = []
for x in spec["lines"]:
    sub = lambda u: re.sub(r"\{([A-Z_]+)\}", lambda m: fill.get(m.group(1), m.group(0)), u)
    t = sub(x["text"])
    if t.strip() and not re.search(r"\{[A-Z_]+\}", t) and (ONLY is None or x["id"] in ONLY):
        lines.append(x | {"text": t, "say": [sub(u) for u in x.get("say") or [t]]})
book = nb([
    ("markdown", ("# Bolo 04: cartoon voices" if STORY else "# Bolo 04: launch video narration") + " (Indic Parler TTS)\n\n" "Needs the `HF_TOKEN` secret. GPU T4. "
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
import librosa
def pitch_var(a):  # spread of log pitch over voiced frames: higher = livelier, less monotone
    f0, v, _ = librosa.pyin(a, fmin=70, fmax=400, sr=SR, frame_length=2048)
    f0 = f0[v & ~np.isnan(f0)]
    return float(np.std(np.log(f0))) if len(f0) > 10 else 0.0
for line in SPEC["lines"]:
    d = desc_tok(line.get("desc") or SPEC["voices"][line["voice"]], return_tensors="pt").to(dev)
    takes = []
    says = line.get("say") or [line["text"]]
    per = 3 if len(says) > 1 else 4  # 3 takes per spelling when the name is in the line
    for seed, say in [(sd, sy) for sy in says for sd in range(per)]:
        p = tok(say, return_tensors="pt").to(dev)
        torch.manual_seed(seed)
        with torch.no_grad():
            a = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                               prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask, do_sample=True)
        a = a.cpu().float().numpy().squeeze()
        vi = says.index(say)
        f = f"{OUT}/takes/{line['id']}_v{vi}s{seed}.wav"
        sf.write(f, a, SR)
        lang = "hi" if line["voice"] in ("hinglish", "aarav") else "en"
        hyp = asr(f, generate_kwargs={"language": lang, "task": "transcribe"})["text"]
        NAME = re.compile(r"^(my|mai|mi|mee|mae|may)[a-z]*n[a-z]*$")  # any spelling of the name, ignored in WER
        strip = lambda ws: [w for w in ws if not NAME.match(w)]
        w = jiwer.wer(" ".join(strip(norm(line["text"]))) or "x", " ".join(strip(norm(hyp))) or "x") if lang == "en" else None
        name_ok = None
        if "Myna" in line["text"]:  # did it say MY-na (not Mina / Meena / Miner)?
            name_ok = bool(re.search(r"\b(myna|mynah|maina|mainah|mayna|my na|my nah|mai na|maena)\b", hyp.lower()))
        pv = pitch_var(a)
        takes.append({"seed": seed, "variant": vi, "file": f, "name_ok": name_ok, "wer": w, "pitch_var": round(pv, 3), "dur": round(len(a) / SR, 2), "heard": hyp})
        print(line["id"], vi, seed, name_ok, w, round(pv, 3), round(len(a) / SR, 2), hyp[:90], flush=True)
    wer = lambda t: t["wer"] if t["wer"] is not None else 0
    pool = [t for t in takes if t["name_ok"] is not False] or takes  # the name must sound right, if any take manages it
    ok = [t for t in pool if wer(t) <= min(map(wer, pool)) + 0.1 and t["dur"] < 2.5 * len(line["text"]) / 14 + 1]
    best = max(ok or takes, key=lambda t: t["pitch_var"])  # most expressive among the intelligible takes
    import shutil; shutil.copy(best["file"], f"{OUT}/{line['id']}.wav")
    scores[line["id"]] = {"best": best, "takes": takes}
json.dump({"sr": SR, "scores": scores}, open(OUT + "/scores.json", "w"), indent=1, ensure_ascii=False)
'''),
])
(ROOT / "notebooks/04_narration.ipynb").write_text(json.dumps(book, indent=1, ensure_ascii=False))
print(len(lines), "lines ->", "notebooks/04_narration.ipynb")
