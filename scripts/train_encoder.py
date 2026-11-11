"""Fine-tune a pretrained encoder joint tagger and evaluate. Usage:
  HF_HOME=models/hf_cache python scripts/train_encoder.py --model google/muril-base-cased --epochs 4 --seed 0
"""
import argparse
import json
import os
from pathlib import Path

import torch

from bolo.config import load_config
from bolo.evaluate import load_rows, run
from bolo.models.encoder import EncoderTagger

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="google/muril-base-cased")
ap.add_argument("--tag", default="muril")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-train", type=int, default=None)
ap.add_argument("--views", action="store_true")
ap.add_argument("--data", default="data/text/v0")
args = ap.parse_args()

cfg = load_config()
train = load_rows(f"{args.data}/train.jsonl")[: args.max_train]
dev = load_rows(f"{args.data}/dev.jsonl")
test = load_rows(f"{args.data}/test.jsonl")
m = EncoderTagger(cfg, args.model)
print(f"{args.model}: {m.n_params():,} params on {m.device}")
info = m.fit(train, dev, epochs=args.epochs, seed=args.seed, use_views=args.views)
out = Path(f"models/{args.tag}-s{args.seed}")
m.save(out)
res = run(m, cfg.schema, test, dev_rows=dev, timing=False)
torch.set_num_threads(4)
m.to("cpu")
from bolo.evaluate import latency_ms
res["latency_ms"] = latency_ms(m, [r["text"] for r in test[:100]], n=100)
res.update(views=args.views, model=args.tag, base=args.model, seed=args.seed, params=m.n_params(), train=info,
           size_mb_fp32=round(os.path.getsize(out / "model.pt") / 1e6, 1), data=args.data)
Path(f"results/text/{args.tag}-s{args.seed}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
t = res["test"]
print(f"EM {t['exact_match']:.3f} | heldout-tpl {t['breakdown']['template']['heldout']['exact_match']:.3f} | "
      f"intent {t['intent_acc']:.3f} | slotF1 {t['slot']['f1']:.3f} | FAR {t['false_action_rate']:.3f} | "
      f"consist {res.get('script_consistency')} | lat {res['latency_ms']}")
