"""Generate the Kaggle/Colab notebooks in notebooks/ (edit cells here, then re-run this script).

The heavy logic lives in the tested package (bolo.audio.tts, bolo.asr.whisper_ft); notebooks are thin.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"


def nb(cells):
    return {"cells": [{"cell_type": t, "metadata": {}, "source": s.strip("\n").splitlines(True),
                       **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


FIND_BUNDLE = r'''
import glob, os, sys, json
# Kaggle: bundle uploaded as a private dataset -> /kaggle/input/...; Colab: unzip bolo-bundle.zip into /content
cands = glob.glob("/kaggle/input/**/bolo-bundle/src/bolo/__init__.py", recursive=True) + \
        glob.glob("/content/**/bolo-bundle/src/bolo/__init__.py", recursive=True)
assert cands, "bolo-bundle not found: add the bolo-bundle dataset (Kaggle) or unzip it in /content (Colab)"
BUNDLE = cands[0].split("/src/bolo/")[0]
sys.path.insert(0, BUNDLE + "/src")
WORK = "/kaggle/working" if os.path.exists("/kaggle/working") else "/content"
print("bundle:", BUNDLE, "| work dir:", WORK)
'''

HF_LOGIN = r'''
# Indic Parler-TTS is gated: accept its terms on huggingface.co first, then add a READ token as a secret
# named HF_TOKEN (Kaggle: Add-ons -> Secrets; Colab: key icon -> Secrets).
token, errs = None, []
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    errs.append(f"kaggle: {type(e).__name__}: {e}")
if not token and not os.path.exists("/kaggle"):
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
    except Exception as e:
        errs.append(f"colab: {type(e).__name__}: {e}")
token = token or os.environ.get("HF_TOKEN")
if token:
    from huggingface_hub import login
    login(token=token)
    MODEL = "ai4bharat/indic-parler-tts"
else:
    # API-started runs cannot read Kaggle secrets: run a pipeline SMOKE TEST with the ungated Parler-TTS mini
    # (same codebase/architecture) and stop. The real run must be started from the Kaggle UI.
    print("NO HF_TOKEN:", " | ".join(errs), "-> smoke-test mode with parler-tts/parler-tts-mini-v1")
    MODEL = "parler-tts/parler-tts-mini-v1"
SMOKE = MODEL != "ai4bharat/indic-parler-tts"
'''

TTS = nb([
    ("markdown", """
# Bolo 01: synthetic Hinglish command speech (Indic Parler-TTS)

Generates 16 kHz training/dev/test speech for the Bolo command dataset with AI4Bharat **Indic Parler-TTS**
(Apache-2.0, gated). Voices are split by speaker: test voices (Aman, Rani, Priya, Vikas) never appear in train.

**Before running**
1. On huggingface.co, open `ai4bharat/indic-parler-tts` and accept the access conditions.
2. Create a read token and add it as a notebook secret named `HF_TOKEN`.
3. Kaggle: *Add input* the private dataset **bolo-bundle** (made by `scripts/make_kaggle_bundle.py`).
   Settings: **GPU T4 x2** (one is used), **Internet on**.
4. Use **Save Version -> Save & Run All (Commit)** so it runs in the background (it is resumable: if it
   stops, re-run and it continues from `manifest.jsonl`).

Expected time: unknown until measured; the progress log prints seconds/utterance and an ETA. Lower `N`
below to try a small run first (e.g. 64 per split, a few minutes).
Output: `/kaggle/working/bolo-tts/` (audio + `manifest.jsonl` + `rejected.jsonl`), roughly 0.5 GB.
"""),
    ("code", "!pip install -q git+https://github.com/huggingface/parler-tts.git soundfile"),
    ("code", r'''
# parler-tts pins an old transformers + protobuf 4.x; Kaggle's TensorFlow needs protobuf>=5 and old transformers
# imports TF if present -> ImportError. We never use TF/JAX, so switch them off before importing transformers.
import os
os.environ["USE_TF"] = "0"; os.environ["USE_TORCH"] = "1"; os.environ["USE_JAX"] = "0"; os.environ["USE_FLAX"] = "0"
'''),
    ("code", FIND_BUNDLE),
    ("code", HF_LOGIN),
    ("code", r'''
from bolo.audio import tts
N = {"train": 7000, "dev": 400, "test": 1200}   # max utterances per split; lower for a trial run
BATCH = 16                                     # lower to 8 if CUDA OOM
OUT = WORK + "/bolo-tts"
jobs = []
for split, n in N.items():
    rows = [json.loads(l) for l in open(f"{BUNDLE}/data/text/v1/{split}.jsonl", encoding="utf-8")][:n]
    jobs += tts.plan(rows, split, seed=0)
print(len(jobs), "jobs; example:", json.dumps(jobs[0], ensure_ascii=False)[:400])
'''),
    ("code", r'''
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if SMOKE:
    smoke = sorted(jobs, key=lambda j: len(j["tts_input"]))[:12] + [j for j in jobs if j["split"] == "train"][:8]  # shortest = edge cases
    print(tts.synthesize(smoke, WORK + "/smoke", batch_size=BATCH, device=DEVICE, model_id=MODEL))
    sm = [json.loads(l) for l in open(WORK + "/smoke/manifest.jsonl")] if os.path.exists(WORK + "/smoke/manifest.jsonl") else []
    print("SMOKE kept", len(sm), "durations", [r["duration_s"] for r in sm])
    raise SystemExit("SMOKE TEST DONE (pipeline OK if clips were kept); start the real run from the Kaggle UI")
# 1) self-check on a small slice first: stop early (and save GPU quota) if the TTS output looks broken
probe = [j for j in jobs if j["split"] == "train"][:48] + [j for j in jobs if j["split"] == "test"][:16]
st = tts.synthesize(probe, OUT, batch_size=BATCH, device=DEVICE, model_id=MODEL)
man = [json.loads(l) for l in open(OUT + "/manifest.jsonl")] if os.path.exists(OUT + "/manifest.jsonl") else []
rej = [json.loads(l) for l in open(OUT + "/rejected.jsonl")] if os.path.exists(OUT + "/rejected.jsonl") else []
rate = len(rej) / max(1, len(man) + len(rej))
print(f"probe: kept {len(man)}, rejected {len(rej)} ({rate:.0%})", [r["reject"] for r in rej][:10])
assert rate < 0.3, "too many TTS rejections in the probe: inspect rejected.jsonl before a full run"
# 2) full run (resumable; the probe utterances are skipped)
stats = tts.synthesize(jobs, OUT, batch_size=BATCH, device=DEVICE, model_id=MODEL)
print(stats)
'''),
    ("code", r'''
import collections
man = [json.loads(l) for l in open(OUT + "/manifest.jsonl")]
rej = [json.loads(l) for l in open(OUT + "/rejected.jsonl")] if os.path.exists(OUT + "/rejected.jsonl") else []
print("kept per split:", collections.Counter(r["split"] for r in man))
print("rejected:", collections.Counter(r["reject"] for r in rej))
print("hours:", round(sum(r["duration_s"] for r in man) / 3600, 2))
json.dump({"kept": collections.Counter(r["split"] for r in man), "rejected": collections.Counter(r["reject"] for r in rej),
           "hours": sum(r["duration_s"] for r in man) / 3600}, open(OUT + "/summary.json", "w"), indent=1)
'''),
    ("markdown", "Listen to a few before training on them (TTS can mispronounce English words in Devanagari context):"),
    ("code", r'''
from IPython.display import Audio, display
import random
for r in random.Random(0).sample(man, 6):
    print(r["speaker"], "|", r["tts_input"], "| target:", r["asr_text"])
    display(Audio(OUT + "/" + r["audio"]))
'''),
])

FT = nb([
    ("markdown", """
# Bolo 02: Whisper fine-tune for Roman-script command Hinglish

Base: **Oriserve/Whisper-Hindi2Hinglish-Swift** (whisper-base, 72.6M, Apache-2.0, Roman output).
Targets: Bolo's canonical Roman `asr_text`. Training audio: the synthetic speech from notebook 01, with
on-the-fly noise/reverb/speed/phone-band augmentation (`bolo.audio.augment`).

**Before running**
1. Kaggle *Add input*: the **bolo-bundle** dataset and the **output of notebook 01** (bolo-tts).
2. Settings: **GPU T4 x2**, **Internet on** (to download base models, ~0.3-1 GB).
3. Save Version -> Save & Run All. Roughly: baselines ~10 min, fine-tune ~20-40 min (estimate; check the log).

**Honesty notes**
- Every number here is on *synthetic* speech (held-out voices, held-out templates/values). The real
  benchmark is the human `recordings/` set, evaluated locally in Phase 5 (audio never leaves the laptop).
- WER is reported raw, and also `phonetic_wer`, which forgives Roman spelling variants (kholo/kolo).
  Devanagari output (plain Whisper, language=hi) is transliterated before scoring, which still penalises
  English loanwords (टर्मिनल -> "tarminal" != "terminal"). That is a real cost of Devanagari output for our use.

**Output** `/kaggle/working/bolo-whisper/` (~290 MB safetensors + processor) and `results_asr.json`.
Download both into `models/bolo-whisper/` and `results/asr/` locally (models/ is gitignored).
"""),
    ("code", FIND_BUNDLE + r'''
tm = glob.glob("/kaggle/input/**/bolo-tts/manifest.jsonl", recursive=True) + glob.glob(WORK + "/bolo-tts/manifest.jsonl")
assert tm, "add notebook 01's output (bolo-tts) as an input"
TTS_MAN = tm[0]
print("tts manifest:", TTS_MAN)
'''),
    ("code", r'''
import torch, time
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from bolo.asr import whisper_ft as W
from bolo.config import load_config
from bolo.normalize import Normalizer
nz = Normalizer.from_config(load_config())
rows = W.load_manifest(TTS_MAN, for_training=True)
split = {s: [r for r in rows if r["split"] == s] for s in ("train", "dev", "test")}
print({k: len(v) for k, v in split.items()})
DEV = "cuda"
RESULTS = {"data": {k: len(v) for k, v in split.items()}, "baselines": {}, "finetuned": {}}
'''),
    ("markdown", "## Baselines on the synthetic test set (held-out voices)"),
    ("code", r'''
BASELINES = [("openai/whisper-base", "hi"), ("openai/whisper-base", "en"), ("openai/whisper-small", "hi"),
             ("Oriserve/Whisper-Hindi2Hinglish-Swift", "en"), ("Oriserve/Whisper-Hindi2Hinglish-Swift", "hi")]
hyps_all = {}
for mid, lang in BASELINES:
    proc = WhisperProcessor.from_pretrained(mid)
    m = WhisperForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.float16).to(DEV)
    s, h = W.evaluate(m, proc, split["test"], nz, language=lang, device=DEV)
    RESULTS["baselines"][f"{mid}|{lang}"] = s
    hyps_all[f"{mid}|{lang}"] = h
    print(mid, lang, s)
    del m; torch.cuda.empty_cache()
'''),
    ("markdown", "## Fine-tune"),
    ("code", r'''
CFG = dict(epochs=5, batch_size=32, lr=1e-5, warmup=50, eval_every=150, seed=0, freeze_encoder=False)
proc = WhisperProcessor.from_pretrained(W.BASE)
model = WhisperForConditionalGeneration.from_pretrained(W.BASE)
OUTDIR = WORK + "/bolo-whisper"
t0 = time.time()
res = W.train(model, proc, split["train"], split["dev"], OUTDIR, device=DEV, normalizer=nz, **CFG)
RESULTS["train"] = {"config": CFG, "base": W.BASE, "minutes": round((time.time() - t0) / 60, 1), **res}
'''),
    ("code", r'''
best = WhisperForConditionalGeneration.from_pretrained(OUTDIR, torch_dtype=torch.float16).to(DEV)
bproc = WhisperProcessor.from_pretrained(OUTDIR)
s, h = W.evaluate(best, bproc, split["test"], nz, language="en", device=DEV)
RESULTS["finetuned"]["bolo-whisper|en"] = s
hyps_all["bolo-whisper|en"] = h
print(s)
# per-utterance outputs for error analysis (synthetic text only; no audio)
RESULTS["test_utts"] = [{"utt_id": r["utt_id"], "ref": W.target(r), "speaker": r["speaker"],
                         **{k: v[i] for k, v in hyps_all.items()}} for i, r in enumerate(split["test"])]
json.dump(RESULTS, open(WORK + "/results_asr.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in RESULTS.items() if k != "test_utts"}, indent=1)[:3000])
'''),
    ("markdown", "Sample errors from the fine-tuned model:"),
    ("code", r'''
bad = [u for u in RESULTS["test_utts"] if u["bolo-whisper|en"].lower().strip(" .?!") != u["ref"]]
for u in bad[:25]:
    print(f'{u["ref"]!r:45} -> {u["bolo-whisper|en"]!r}')
'''),
])

# --- ablation: can we keep the command gain and forget less? (frozen encoder, shorter training) ---
_ft_setup = [c for c in FT["cells"] if c["cell_type"] == "code"][:2]
ABL = nb([
    ("markdown", """
# Bolo 03: Whisper fine-tune ablation

Same data and code as notebook 02, no baselines. Variants:
- `freeze_enc`: encoder frozen, only the decoder trains (less forgetting of general acoustics?)
- `short`: full model, 2 epochs instead of 5
Each is scored on the same synthetic test (held-out voices). Output: `/kaggle/working/ablation/<variant>/` + `results_ablation.json`.
"""),
    ("code", "".join(_ft_setup[0]["source"])),
    ("code", "".join(_ft_setup[1]["source"])),
    ("code", r'''
VARIANTS = {"freeze_enc": dict(epochs=5, freeze_encoder=True, lr=1e-5),
            "short": dict(epochs=2, freeze_encoder=False, lr=1e-5)}
for name, v in VARIANTS.items():
    proc = WhisperProcessor.from_pretrained(W.BASE)
    model = WhisperForConditionalGeneration.from_pretrained(W.BASE)
    out = f"{WORK}/ablation/{name}"
    t0 = time.time()
    res = W.train(model, proc, split["train"], split["dev"], out, device=DEV, normalizer=nz, batch_size=32,
                  warmup=50, eval_every=150, seed=0, **v)
    best = WhisperForConditionalGeneration.from_pretrained(out, torch_dtype=torch.float16).to(DEV)
    s, _ = W.evaluate(best, WhisperProcessor.from_pretrained(out), split["test"], nz, language="en", device=DEV)
    RESULTS["finetuned"][name] = {**s, "config": v, "minutes": round((time.time() - t0) / 60, 1), "best_dev": res["best"]}
    print(name, RESULTS["finetuned"][name], flush=True)
    del model, best; torch.cuda.empty_cache()
json.dump(RESULTS, open(WORK + "/results_ablation.json", "w"), indent=1, ensure_ascii=False)
'''),
])

OUT.mkdir(exist_ok=True)
for name, book in [("01_tts_synth.ipynb", TTS), ("02_whisper_finetune.ipynb", FT), ("03_whisper_ablation.ipynb", ABL)]:
    (OUT / name).write_text(json.dumps(book, indent=1, ensure_ascii=False))
    print("wrote", OUT / name)
