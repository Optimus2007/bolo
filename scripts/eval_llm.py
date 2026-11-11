"""Evaluate a fine-tuned MLX LLM (e.g. the 4-bit fused Qwen) on the text test set.
  python scripts/eval_llm.py models/qwen0.5b-bolo-4bit qwen0.5b-lora-4bit
"""
import json
import os
import sys
import time
from pathlib import Path

from bolo.config import load_config
from bolo.evaluate import choose_threshold, consistency, finalize, load_rows, score
from bolo.models.llm import LLMPredictor

path, tag = sys.argv[1], sys.argv[2]
cfg = load_config()
test = load_rows("data/text/v0/test.jsonl")
dev = load_rows("data/text/v0/dev.jsonl")[:300]
m = LLMPredictor(path, name=tag)
t0 = time.time()
thr = choose_threshold(dev, m.predict([r["text"] for r in dev]), cfg.schema)
raw = m.predict([r["text"] for r in test])
print(f"batched throughput: {len(test) / (time.time() - t0) * 0 + 0:.0f}", end="\r")
preds = [finalize(p, cfg.schema, thr) for p in raw]
res = {"threshold": thr, "test": score(test, preds)}
unseen = [i for i, r in enumerate(test) if not r["meta"].get("seen_in_train")]
res["test_unseen_text"] = {k: v for k, v in score([test[i] for i in unseen], [preds[i] for i in unseen]).items()
                           if k != "breakdown"}
sub = test[:300]
vp = [finalize(p, cfg.schema, thr) for p in m.predict([v["text"] for r in sub for v in r["views"]])]
res["script_consistency"] = consistency(sub, preds[:300], [vp[2 * i:2 * i + 2] for i in range(300)])
res["script_consistency_note"] = "computed on first 300 test rows"
m.batch_size = 1
times = []
for r in test[:50]:
    s = time.perf_counter()
    m.predict([r["text"]])
    times.append((time.perf_counter() - s) * 1000)
times.sort()
res["latency_ms"] = {"p50": round(times[25], 1), "p95": round(times[47], 1), "note": "MLX GPU, batch 1, 50 utts"}
res["parse_failures"] = sum(1 for p in raw if "raw" in p)
size = sum(f.stat().st_size for f in Path(path).glob("*.safetensors"))
res.update(model=tag, seed=0, params=494_033_000, size_mb=round(size / 1e6, 1), data="data/text/v0")
Path(f"results/text/{tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
t = res["test"]
print(f"EM {t['exact_match']:.3f} | heldout {t['breakdown']['template']['heldout']['exact_match']:.3f} | intent "
      f"{t['intent_acc']:.3f} | slotF1 {t['slot']['f1']:.3f} | FAR {t['false_action_rate']:.3f} | consist "
      f"{res['script_consistency']} | lat {res['latency_ms']} | parse_fail {res['parse_failures']} | thr {thr}")
