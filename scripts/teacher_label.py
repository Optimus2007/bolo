"""Add MuRIL teacher soft intent probabilities to train (+views) and paraphrase rows.
  HF_HOME=models/hf_cache python scripts/teacher_label.py --teacher models/muril_aug-s0 train para_train
Writes <name>_kd.jsonl next to each input.
"""
import argparse
import json
from pathlib import Path

from bolo.config import load_config
from bolo.evaluate import load_rows
from bolo.models.encoder import EncoderTagger

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default="models/muril_aug-s0")
ap.add_argument("--data", default="data/text/v1")
ap.add_argument("names", nargs="+")
args = ap.parse_args()
cfg = load_config()
t = EncoderTagger.load(cfg, args.teacher)
for name in args.names:
    rows = load_rows(f"{args.data}/{name}.jsonl")
    texts = [r["text"] for r in rows] + [v["text"] for r in rows for v in r.get("views", [])]
    probs = t.intent_probs(texts)
    k = len(rows)
    for i, r in enumerate(rows):
        r["teacher_intent"] = [round(x, 5) for x in probs[i]]
    for r in rows:
        for v in r.get("views", []):
            v["teacher_intent"] = [round(x, 5) for x in probs[k]]
            k += 1
    agree = sum(cfg_i == r["intent"] for cfg_i, r in
                zip([list(cfg.schema["intents"])[max(range(len(p)), key=p.__getitem__)] for p in probs[:len(rows)]], rows))
    out = Path(f"{args.data}/{name}_kd.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} rows, {len(texts)} texts, teacher agrees with label on {agree / len(rows):.3f}")
