"""Evaluate saved systems on the reviewed LLM-paraphrased test set (strict and grounded)."""
import json
from pathlib import Path

from bolo.baselines.rules import RuleBaseline
from bolo.config import load_config
from bolo.evaluate import finalize, load_rows, score
from bolo.grounding import Environment, ground

cfg = load_config()
rows = load_rows("data/text/v1/para_test_reviewed.jsonl")
P = cfg.open_pools
env = Environment([e.value for e in P["folders"]], [e.value for e in P["files"]],
                  [e.value for e in P["apps"]], [e.value for e in P["contacts"]])


def systems():
    yield "rules", RuleBaseline(cfg), 0.0
    from bolo.models.tiny import BoloTiny
    for v in ("phono_views_aug", "phono_views_aug_para"):
        for s in (0, 1, 2):
            d = f"models/bolo-tiny/{v}-s{s}"
            thr = json.loads(Path(f"results/text/tiny_{v}-s{s}.json").read_text())["threshold"]
            yield f"bolo-tiny/{v}-s{s}", BoloTiny.load(cfg, d), thr
    from bolo.models.encoder import EncoderTagger
    yield "muril_aug-s0", EncoderTagger.load(cfg, "models/muril_aug-s0"), \
        json.loads(Path("results/text/muril_aug-s0.json").read_text())["threshold"]
    from bolo.models.llm import LLMPredictor
    yield "qwen0.5b-lora-4bit", LLMPredictor("models/qwen0.5b-bolo-4bit"), \
        json.loads(Path("results/text/qwen0.5b-lora-4bit.json").read_text())["threshold"]


out = {}
for name, m, thr in systems():
    preds = [finalize(p, cfg.schema, thr) for p in m.predict([r["text"] for r in rows])]
    g = [finalize(ground(p, env), cfg.schema) if p["intent"] != "none" else p for p in preds]
    s, sg = score(rows, preds), score(rows, g)
    out[name] = {"n": len(rows), "exact_match": s["exact_match"], "intent_acc": s["intent_acc"],
                 "slot_f1": s["slot"]["f1"], "false_action_rate": s["false_action_rate"],
                 "grounded_exact_match": sg["exact_match"], "grounded_false_action_rate": sg["false_action_rate"]}
    print(name, out[name])
Path("results/para_test.json").write_text(json.dumps(out, indent=1))
