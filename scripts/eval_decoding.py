"""Does smarter decoding close the speech-to-action gap? (beam search, hotword boosting, N-best reranking)

  python scripts/eval_decoding.py tune   # grid on the 80 clip macOS set (data/say), picks settings
  python scripts/eval_decoding.py test   # baseline vs the chosen settings on the 240 clip held-out set (data/say_test)

Model: Bolo Sun fp16 with the default "commands" prompt, NLU: Bolo Tiny + grounding.
Hypotheses are cached in data/eval/decoding/ (gitignored). Summary JSON goes to results/speech/.
"""
import itertools
import json
import sys
import time
from pathlib import Path

from bolo.asr.decode import Rerank, TrieBoost, choose
from bolo.audio.augment import SR, read_wav, resample
from bolo.audio.metrics import corpus_scores
from bolo.config import load_config
from bolo.evaluate import _exact, finalize, score
from bolo.grounding import ground
from bolo.models.tiny import BoloTiny
from bolo.normalize import Normalizer
from bolo.pipeline import COMMAND_WORDS
from bolo.speech_eval import WhisperASR, environment, load_set

MODEL = "models/bolo-whisper-fp16"
CACHE = Path("data/eval/decoding")
CACHE.mkdir(parents=True, exist_ok=True)
cfg = load_config()
env = environment(cfg)
nz = Normalizer.from_config(cfg)
nlu = BoloTiny.load(cfg, "models/bolo-tiny/phono_views_aug-s0")
asr = WhisperASR(MODEL, "en", prompt=COMMAND_WORDS)
PHRASES = env.folders + env.files + env.apps + env.contacts + [w.strip() for w in COMMAND_WORDS.split(",")]


def audio(rows):
    out = []
    for r in rows:
        x, sr = read_wav(r["_wav"])
        out.append(resample(x, sr, SR))
    return out


def decode(name, rows, wavs, beams, bonus):
    f = CACHE / f"{name}_b{beams}_x{bonus}.json"
    if f.exists():
        return json.loads(f.read_text())
    boost = TrieBoost(asr.proc.tokenizer, PHRASES, bonus=bonus) if bonus else None
    res = []
    asr.nbest(wavs[0], beams=beams, boost=boost)  # warm-up
    for w in wavs:
        t = time.perf_counter()
        if beams == 1 and not boost:
            hyps = [(asr(w), 0.0)]
        else:
            hyps = asr.nbest(w, beams=beams, boost=boost)
        res.append({"hyps": hyps, "ms": (time.perf_counter() - t) * 1000})
    f.write_text(json.dumps(res, ensure_ascii=False))
    return res


def top1(text):
    p = finalize(nlu.predict([" ".join(text.lower().split())])[0], cfg.schema, 0.0)
    return finalize(ground(p, env), cfg.schema, 0.0) if p["intent"] != "none" else p


def evaluate(rows, dec, rr: Rerank | None):
    preds, texts = [], []
    oracle = 0
    for r, d in zip(rows, dec):
        if rr is None:
            texts.append(d["hyps"][0][0])
            preds.append(top1(d["hyps"][0][0]))
        else:
            c = choose([tuple(h) for h in d["hyps"]], nlu, cfg, env, rr)
            texts.append(c.text)
            preds.append(c.pred)
        oracle += any(_exact(r, top1(h[0])) for h in d["hyps"])
    n = len(rows)
    em = sum(_exact(r, p) for r, p in zip(rows, preds)) / n
    false = sum(p["intent"] != "none" and not _exact(r, p) for r, p in zip(rows, preds)) / n
    sc = score(rows, preds)
    w = corpus_scores([r["ref"] for r in rows], texts, nz)
    ms = sorted(d["ms"] for d in dec)
    return {"wer": w["wer"], "phonetic_wer": w["phonetic_wer"], "e2e": round(em, 4), "wrong_action": round(false, 4),
            "acted_on_none": sc["false_action_rate"], "over_abstain": sc["over_abstain_rate"],
            "nbest_oracle": round(oracle / n, 4), "ms_median": round(ms[n // 2], 1), "ms_p90": round(ms[int(n * 0.9)], 1)}


GRID = [Rerank(w_nlu=a, w_ground=b, w_none=c) for a, b, c in itertools.product((0.5, 1, 2), (0, 1, 2, 4), (0, 1, 3))]


def tune():
    rows = load_set("data/say/manifest.jsonl", "data/text/v0/test.jsonl", "say")
    wavs = audio(rows)
    out = {}
    for beams, bonus in [(1, 0), (3, 0), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4)]:
        dec = decode("say", rows, wavs, beams, bonus)
        key = f"beam{beams}_boost{bonus}"
        out[key + "_top1"] = evaluate(rows, dec, None)
        print(key, "top1", out[key + "_top1"], flush=True)
        if beams > 1:
            best = max(((evaluate(rows, dec, rr), rr) for rr in GRID), key=lambda x: (x[0]["e2e"], -x[0]["wrong_action"]))
            out[key + "_rerank"] = best[0] | {"rerank": best[1].__dict__}
            print(key, "rerank", out[key + "_rerank"], flush=True)
    Path("results/speech/decoding_tune.json").write_text(json.dumps(out, indent=1))


def test():
    t = json.loads(Path("results/speech/decoding_tune.json").read_text())
    rows = load_set("data/say_test/manifest.jsonl", "data/text/v0/test.jsonl", "say_test")
    wavs = audio(rows)
    cands = {k: v for k, v in t.items() if "rerank" in k}
    best = max(cands, key=lambda k: (cands[k]["e2e"], -cands[k]["wrong_action"]))
    bt = max((k for k in t if k.endswith("_top1") and "beam5" in k), key=lambda k: t[k]["e2e"])
    out = {"chosen_on_dev": best, "settings": cands[best]["rerank"]}
    for label, key, rr in [("greedy (current default)", "beam1_boost0_top1", None),
                           ("beam5", "beam5_boost0_top1", None),
                           ("beam5 + rerank", "beam5_boost0_rerank", Rerank(**t["beam5_boost0_rerank"]["rerank"])),
                           (f"best boost top1 ({bt})", bt, None),
                           (f"chosen ({best})", best, Rerank(**cands[best]["rerank"]))]:
        beams, bonus = int(key.split("_")[0][4:]), int(key.split("_")[1][5:])
        dec = decode("say_test", rows, wavs, beams, bonus)
        out[label] = evaluate(rows, dec, rr)
        print(label, out[label], flush=True)
    Path("results/speech/decoding_test.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"tune": tune, "test": test}[sys.argv[1]]()
