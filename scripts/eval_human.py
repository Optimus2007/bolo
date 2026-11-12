"""Evaluate the real pipeline (speech -> Bolo Tiny -> grounding) on human recordings from `bolo record`.

  .venv/bin/python scripts/eval_human.py                       # recordings/manifest.jsonl
  .venv/bin/python scripts/eval_human.py --out results/speech/human_v1.json

Aggregates go to --out (committable: no audio, no transcripts). Per-utterance hypotheses go to data/eval/human/
(gitignored), because they are derived from people's voices.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bolo.audio.metrics import corpus_scores  # noqa: E402
from bolo.config import load_config  # noqa: E402
from bolo.evaluate import _exact  # noqa: E402
from bolo.normalize import Normalizer  # noqa: E402
from bolo.pipeline import Bolo  # noqa: E402
from bolo.speech_eval import environment, load_set  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="recordings/manifest.jsonl")
ap.add_argument("--test", default="data/text/v1/test.jsonl")
ap.add_argument("--out", default="results/speech/human_v1.json")
ap.add_argument("--configs", default="bolo_sun_beam5,bolo_sun_greedy,swift_greedy")
a = ap.parse_args()

cfg = load_config()
env = environment(cfg)
rows = load_set(ROOT / a.manifest, ROOT / a.test)
CONFIGS = {
    "bolo_sun_beam5": dict(asr="models/bolo-whisper-fp16", beams=5),
    "bolo_sun_greedy": dict(asr="models/bolo-whisper-fp16", beams=1),
    "swift_greedy": dict(asr="Oriserve/Whisper-Hindi2Hinglish-Swift", beams=1),
}
norm = Normalizer.from_config(cfg)


def same_lenient(gold: dict, pred: dict) -> bool:
    """Same intent and slots, ignoring case, spaces and spelling-by-sound ("money control" == "moneycontrol")."""
    from bolo.normalize import phonetic_key

    k = lambda v: phonetic_key(str(v).replace(" ", ""))
    return gold["intent"] == pred["intent"] and {a: k(v) for a, v in gold["slots"].items()} == \
        {a: k(v) for a, v in (pred.get("slots") or {}).items()}
res = {"n": len(rows), "speakers": len({r["speaker"] for r in rows}),
       "kinds": {k: sum(r["set"].endswith(k) for r in rows) for k in ("read", "free")},
       "note": "human recordings; speech model trained on the 9-command set unless stated", "configs": {}}
detail_dir = ROOT / "data/eval/human"
detail_dir.mkdir(parents=True, exist_ok=True)
for name in a.configs.split(","):
    b = Bolo.load(env=env, **CONFIGS[name])
    out, t0 = [], time.perf_counter()
    for r in rows:
        p = b.listen(r["_wav"])
        out.append(p | {"utt_id": r["utt_id"], "kind": r["set"].split("_")[-1], "gold": {"intent": r["intent"], "slots": r["slots"]},
                        "ref": r["ref"], "ok": _exact({"intent": r["intent"], "slots": r["slots"]}, p),
                        "ok_lenient": same_lenient({"intent": r["intent"], "slots": r["slots"]}, p)})
    n = len(out)
    none = [o for o in out if o["gold"]["intent"] == "none"]
    cmd = [o for o in out if o["gold"]["intent"] != "none"]
    read = [o for o in out if o["kind"] == "read" and o["ref"]]
    blk = {
        "e2e": round(sum(o["ok"] for o in out) / n, 4),
        "e2e_lenient": round(sum(o["ok_lenient"] for o in out) / n, 4),
        "e2e_read": round(sum(o["ok"] for o in out if o["kind"] == "read") / max(1, sum(o["kind"] == "read" for o in out)), 4),
        "e2e_free": round(sum(o["ok"] for o in out if o["kind"] == "free") / max(1, sum(o["kind"] == "free" for o in out)), 4),
        "intent_acc": round(sum(o["intent"] == o["gold"]["intent"] for o in out) / n, 4),
        "acted_on_none": f"{sum(o['intent'] != 'none' for o in none)}/{len(none)}",
        "wrong_action": round(sum((o["intent"] != "none") and not o["ok"] for o in out) / n, 4),
        "over_abstain": round(sum(o["intent"] == "none" for o in cmd) / max(1, len(cmd)), 4),
        "wer_read": corpus_scores([o["ref"] for o in read], [o["transcript"] for o in read], norm) if read else None,
        "ms_mean": round(sum(o["ms"].get("asr", 0) + o["ms"].get("nlu", 0) for o in out) / n, 1),
    }
    res["configs"][name] = blk
    (detail_dir / f"{name}.jsonl").write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out))
    print(name, json.dumps(blk), f"({time.perf_counter() - t0:.0f}s)", flush=True)
Path(ROOT / a.out).write_text(json.dumps(res, indent=1))
print("->", a.out)
