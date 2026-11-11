"""Train + evaluate Bolo-tiny variants (ablation). Usage:
  python scripts/train_tiny.py --variants plain phono phono_views phono_consistency --epochs 20 --seeds 0
Writes models/bolo-tiny/<variant>-s<seed>/ (gitignored) and results/text/tiny_<variant>-s<seed>.json.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

from bolo.config import load_config
from bolo.evaluate import load_rows, run
from bolo.models.tiny import BoloTiny, TinyConfig

VARIANTS = {
    "plain": dict(features="plain", use_views=False, consistency=0.0),
    "phono": dict(features="phono", use_views=False, consistency=0.0),
    "phono_views": dict(features="phono", use_views=True, consistency=0.0),
    "phono_consistency": dict(features="phono", use_views=True, consistency=1.0),
    "plain_consistency": dict(features="plain", use_views=True, consistency=1.0),
    "phono_views_aug": dict(features="phono", use_views=True, consistency=0.0, augment=True),
    "phono_cons03_aug": dict(features="phono", use_views=True, consistency=0.3, augment=True),
    "plain_views_aug": dict(features="plain", use_views=True, consistency=0.0, augment=True),
    "phono_aug": dict(features="phono", use_views=False, consistency=0.0, augment=True),
}

ap = argparse.ArgumentParser()
ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--seeds", nargs="+", type=int, default=[0])
ap.add_argument("--data", default="data/text/v0")
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()
torch.set_num_threads(args.threads)

cfg = load_config()
train = load_rows(f"{args.data}/train.jsonl")
dev = load_rows(f"{args.data}/dev.jsonl")
test = load_rows(f"{args.data}/test.jsonl")
Path("results/text").mkdir(parents=True, exist_ok=True)

for seed in args.seeds:
    for name in args.variants:
        v = VARIANTS[name]
        m = BoloTiny(cfg, TinyConfig(features=v["features"]))
        print(f"== {name} seed {seed}: {m.n_params():,} params")
        info = m.fit(train, dev, epochs=args.epochs, use_views=v["use_views"], consistency=v["consistency"],
                     augment=v.get("augment", False),
                     seed=seed, log=lambda s: print("  ", s) if s.startswith(("epoch 1/", f"epoch {args.epochs}/")) else None)
        out_dir = Path(f"models/bolo-tiny/{name}-s{seed}")
        m.save(out_dir)
        res = run(m, cfg.schema, test, dev_rows=dev)
        res.update(model=f"bolo-tiny/{name}", seed=seed, params=m.n_params(), train=info,
                   size_mb_fp32=round(os.path.getsize(out_dir / "model.pt") / 1e6, 2),
                   data=args.data)
        Path(f"results/text/tiny_{name}-s{seed}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        t = res["test"]
        print(f"   EM {t['exact_match']:.3f} | intent {t['intent_acc']:.3f} | slotF1 {t['slot']['f1']:.3f} | "
              f"FAR {t['false_action_rate']:.3f} | consist {res.get('script_consistency')} | "
              f"deva EM {t['breakdown']['script']['deva']['exact_match']:.3f} | thr {res['threshold']} | "
              f"{info['train_seconds']}s")
