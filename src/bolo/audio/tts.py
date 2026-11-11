"""Synthetic training speech from an open TTS teacher (AI4Bharat Indic Parler-TTS, Apache-2.0, gated).

Planning (`plan`) is pure python and tested locally; synthesis (`synthesize`) needs a GPU + parler_tts
and runs in notebooks/01_tts_synth.ipynb on Kaggle/Colab.

Design choices (logged in PROGRESS.md):
- Voices are split by speaker: TRAIN voices never appear in the synthetic TEST set, so synthetic-test WER
  is not "same voice" WER.
- Hindi voices read the mixed-script `tts_text` (Hindi words in Devanagari, English in Latin), which is
  what the TTS was trained on. Indian-English voices read the Roman `asr_text`, but only for
  English-heavy commands (en_ratio >= 0.5), where an English-accented speaker is realistic.
- Descriptions vary pace/pitch/recording quality so audio is not all studio-clean; further mic/room
  realism comes from `augment.py` at training time.
- Synthetic speech is TRAINING data. The headline speech eval is the human `recordings/` set.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from .augment import SR, resample, write_wav

# Voice names from the Indic Parler-TTS model card (Hindi: Rohit, Divya recommended; Aman, Rani available.
# English row: Thoma, Mary recommended; Indian-English accents supported).
HI_TRAIN, HI_TEST = ["Rohit", "Divya"], ["Aman", "Rani"]
EN_TRAIN, EN_TEST = ["Thoma", "Mary", "Kabir", "Sneha", "Raghav"], ["Priya", "Vikas"]

PACE = ["slightly slow", "moderate", "moderate", "slightly fast", "fast"]
PITCH = ["low-pitched", "moderate-pitched", "slightly high-pitched"]
STYLE = ["monotone", "slightly expressive", "conversational"]
QUALITY = [
    "The recording is very clear audio with almost no background noise.",
    "The recording is clear, with the speaker sounding close up.",
    "The recording sounds slightly distant, as if spoken a little away from the microphone.",
    "The recording has a little background noise.",
]


def describe(speaker: str, rng: random.Random) -> str:
    return (f"{speaker} speaks in a {rng.choice(STYLE)} tone at a {rng.choice(PACE)} pace with a "
            f"{rng.choice(PITCH)} voice. {rng.choice(QUALITY)}")


def plan(rows: list[dict], split: str, seed: int = 0, en_voice_p: float = 0.35) -> list[dict]:
    rng = random.Random(f"{seed}-{split}")
    hi = HI_TEST if split == "test" else HI_TRAIN
    en = EN_TEST if split == "test" else EN_TRAIN
    jobs = []
    for r in rows:
        en_heavy = (r["meta"].get("en_ratio") or 0) >= 0.5
        if en_heavy and rng.random() < en_voice_p:
            spk, text, voice_lang = rng.choice(en), r["asr_text"], "en"
        else:
            spk, text, voice_lang = rng.choice(hi), r["tts_text"], "hi"
        jobs.append({"utt_id": f"tts-{r['id']}", "source_id": r["id"], "split": split, "speaker": spk,
                     "voice_lang": voice_lang, "tts_input": text, "description": describe(spk, rng),
                     "asr_text": r["asr_text"], "intent": r["intent"], "slots": r["slots"]})
    return jobs


def trim_silence(x: np.ndarray, sr: int = SR, thresh_db: float = -40.0, pad_s: float = 0.15) -> np.ndarray:
    if len(x) == 0:
        return x
    frame = int(0.02 * sr)
    n = len(x) // frame
    if n == 0:
        return x
    e = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(e / (e.max() + 1e-12))
    on = np.where(db > thresh_db)[0]
    if len(on) == 0:
        return x
    pad = int(pad_s * sr)
    return x[max(0, on[0] * frame - pad): min(len(x), (on[-1] + 1) * frame + pad)]


def plausible(x: np.ndarray, text: str, sr: int = SR) -> tuple[bool, str]:
    """Catch common TTS failures: empty output, babbling to max length, clipped garbage."""
    dur = len(x) / sr
    per_char = dur / max(1, len(text))
    if dur < 0.4:
        return False, "too_short"
    if per_char > 0.25 or dur > 12:
        return False, "too_long"
    if per_char < 0.025:
        return False, "too_fast"
    if np.mean(np.abs(x) > 0.99) > 0.01:
        return False, "clipped"
    return True, "ok"


def synthesize(jobs: list[dict], out_dir: Path | str, batch_size: int = 16, device: str = "cuda",
               model_id: str = "ai4bharat/indic-parler-tts", seed: int = 0, log_every: int = 10) -> dict:
    """Resumable: skips utterances already in out_dir/manifest.jsonl."""
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    out = Path(out_dir)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    man_f, rej_f = out / "manifest.jsonl", out / "rejected.jsonl"
    done = set()
    for f in (man_f, rej_f):
        if f.exists():
            done |= {json.loads(line)["utt_id"] for line in f.read_text().splitlines() if line.strip()}
    todo = [j for j in jobs if j["utt_id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to synthesize")
    if not todo:
        return {"done": len(done), "new": 0}

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr_model = model.config.sampling_rate
    torch.manual_seed(seed)

    # sort by length so batches pad less
    todo.sort(key=lambda j: len(j["tts_input"]))
    t0, n_ok, n_bad = time.time(), 0, 0
    for b in range(0, len(todo), batch_size):
        batch = todo[b: b + batch_size]
        d = dtok([j["description"] for j in batch], return_tensors="pt", padding=True).to(device)
        p = tok([j["tts_input"] for j in batch], return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            gen = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                 prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask,
                                 do_sample=True, return_dict_in_generate=True)
        seqs = gen.sequences
        lens = getattr(gen, "audios_length", None)
        with open(man_f, "a") as mf, open(rej_f, "a") as rf:
            for i, j in enumerate(batch):
                a = seqs[i].float().cpu().numpy().squeeze()
                if lens is not None:
                    a = a[: int(lens[i])]
                a = trim_silence(resample(a, sr_model, SR))
                peak = float(np.max(np.abs(a))) if len(a) else 0.0
                if peak > 0:
                    a = a / peak * 0.9
                ok, why = plausible(a, j["asr_text"])
                rec = {**j, "audio": f"audio/{j['utt_id']}.wav", "duration_s": round(len(a) / SR, 3),
                       "sample_rate": SR, "tts_model": model_id, "tts_sr": sr_model}
                if ok:
                    write_wav(out / rec["audio"], a)
                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_ok += 1
                else:
                    rf.write(json.dumps({**rec, "reject": why}, ensure_ascii=False) + "\n")
                    n_bad += 1
        if (b // batch_size) % log_every == 0:
            el = time.time() - t0
            k = b + len(batch)
            print(f"{k}/{len(todo)}  ok={n_ok} rejected={n_bad}  {el / k:.2f}s/utt  "
                  f"eta {(len(todo) - k) * el / k / 60:.0f} min", flush=True)
    return {"done": len(done), "new_ok": n_ok, "new_rejected": n_bad}
