"""Fair baseline for the launch numbers: Swift (the model Bolo Sun started from) on the 240 held-out macOS commands,
with the same prompt and the same decoding options as Bolo. Writes results/speech/swift_heldout.json."""
import json
from pathlib import Path

from bolo.asr.decode import Rerank, choose
from bolo.audio.augment import SR, read_wav, resample
from bolo.audio.metrics import corpus_scores
from bolo.config import load_config
from bolo.evaluate import _exact, finalize
from bolo.grounding import ground
from bolo.models.tiny import BoloTiny
from bolo.normalize import Normalizer
from bolo.pipeline import COMMAND_WORDS, RERANK
from bolo.speech_eval import WhisperASR, environment, load_set

cfg = load_config()
env, nz = environment(cfg), Normalizer.from_config(cfg)
nlu = BoloTiny.load(cfg, "models/bolo-tiny/phono_views_aug-s0")
rows = load_set("data/say_test/manifest.jsonl", "data/text/v1/test.jsonl", "say_test")
asr = WhisperASR("Oriserve/Whisper-Hindi2Hinglish-Swift", "en", prompt=COMMAND_WORDS)
greedy, chosen, gt, ct = [], [], [], []
for r in rows:
    x, sr = read_wav(r["_wav"])
    x = resample(x, sr, SR)
    t = asr(x)
    p = finalize(nlu.predict([" ".join(t.lower().split())])[0], cfg.schema, 0.0)
    greedy.append(_exact(r, finalize(ground(p, env), cfg.schema, 0.0) if p["intent"] != "none" else p))
    gt.append(t)
    c = choose(asr.nbest(x, beams=5), nlu, cfg, env, Rerank(**RERANK))
    chosen.append(_exact(r, c.pred))
    ct.append(c.text)
refs = [r["ref"] for r in rows]
out = {"n": len(rows), "greedy": {"e2e": round(sum(greedy) / len(rows), 4), "wer": corpus_scores(refs, gt, nz)["wer"]},
       "beam5_rerank": {"e2e": round(sum(chosen) / len(rows), 4), "wer": corpus_scores(refs, ct, nz)["wer"]}}
print(out)
Path("results/speech/swift_heldout.json").write_text(json.dumps(out, indent=1))
