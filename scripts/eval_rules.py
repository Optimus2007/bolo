"""Evaluate the rule baseline -> results/text/rules.json"""
import json
from pathlib import Path

from bolo.baselines.rules import RuleBaseline
from bolo.config import load_config
from bolo.evaluate import load_rows, run

cfg = load_config()
test = load_rows("data/text/v1/test.jsonl")
res = run(RuleBaseline(cfg), cfg.schema, test, threshold=0.0)
res.update(model="rules", seed=0, params=0, size_mb_fp32=0.0, data="data/text/v1")
Path("results/text").mkdir(parents=True, exist_ok=True)
Path("results/text/rules.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
print("EM", res["test"]["exact_match"])
