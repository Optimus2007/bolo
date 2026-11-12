"""Export a trained Bolo-tiny to ONNX (fp32 + int8) and check parity + CPU latency with ONNX Runtime."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from bolo.config import load_config
from bolo.evaluate import finalize, load_rows, score
from bolo.models.export import export_onnx, padded_ids
from bolo.models.tagging import decode
from bolo.models.tiny import BoloTiny

src = sys.argv[1] if len(sys.argv) > 1 else "models/bolo-tiny/phono_views_aug_asr-s0"
cfg = load_config()
tiny = BoloTiny.load(cfg, src)
out = Path("models/export/bolo-tiny")
info = export_onnx(tiny, out)
test = load_rows("data/text/v1/test.jsonl")
res = {"source": src, **info}
for name in ("bolo_tiny.onnx", "bolo_tiny.int8.onnx"):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(out / name), so, providers=["CPUExecutionProvider"])
    preds, times = [], []
    for r in test:
        words = [w for w, _, _ in tiny.norm.words(r["text"])][: tiny.tc.max_len] or ["<empty>"]
        t0 = time.perf_counter()
        ids = padded_ids(tiny, words).numpy()
        il, tl = sess.run(None, {"ids": ids})
        times.append((time.perf_counter() - t0) * 1000)
        p = torch.softmax(torch.tensor(il[0]) / tiny.temperature, -1)
        conf, idx = p.max(-1)
        intent = tiny.intents[idx.item()]
        tags = [tiny.labels[j] for j in tl[0].argmax(-1)[: len(words)]]
        preds.append(finalize({"intent": intent, "slots": decode(cfg, tiny.norm, tiny.numbers, intent, words, tags),
                               "confidence": conf.item()}, cfg.schema))
    s = score(test, preds)
    times.sort()
    res[name] = {"exact_match": s["exact_match"], "intent_acc": s["intent_acc"],
                 "latency_ms_p50_1thread": round(float(np.median(times)), 3),
                 "latency_ms_p95_1thread": round(times[int(0.95 * len(times))], 3)}
print(json.dumps(res, indent=1))
Path("results/export/bolo_tiny_onnx.json").write_text(json.dumps(res, indent=1))
