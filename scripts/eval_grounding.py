"""Evaluate environment grounding on top of each text model.

Environment = every folder/file/app/contact in the lexicon (all splits): it plays the role of the user's
real machine, which is known at run time (not training data).
Two conditions:
  present: the spoken entity exists on the device -> measure exact match after grounding
  absent : for each action row with an open slot, the entity is removed from the device -> the
           correct behaviour is to abstain (ask), measure the abstention rate
"""
import json
from pathlib import Path

from bolo.baselines.rules import RuleBaseline
from bolo.config import load_config
from bolo.evaluate import finalize, load_rows, score
from bolo.grounding import SLOT_KIND, Environment, ground

cfg = load_config()
test = load_rows("data/text/v1/test.jsonl")
P = cfg.open_pools
env = Environment([e.value for e in P["folders"]], [e.value for e in P["files"]],
                  [e.value for e in P["apps"]], [e.value for e in P["contacts"]])


def predictors():
    yield "rules", RuleBaseline(cfg), 0.0
    from bolo.models.tiny import BoloTiny
    yield "bolo-tiny/phono_views_aug-s0", BoloTiny.load(cfg, "models/bolo-tiny/phono_views_aug-s0"), 0.0
    try:
        from bolo.models.encoder import EncoderTagger
        r = json.loads(Path("results/text/muril_aug-s0.json").read_text())
        yield "muril_aug-s0", EncoderTagger.load(cfg, "models/muril_aug-s0"), r["threshold"]
    except Exception as e:  # noqa: BLE001
        print("skip muril:", e)


def open_slot_values(r):
    return [(k, v) for k, v in r["slots"].items() if (r["intent"], k) in SLOT_KIND]


results = {}
for name, model, thr in predictors():
    raw = model.predict([r["text"] for r in test])
    base = [finalize(p, cfg.schema, thr) for p in raw]
    grounded = [finalize(ground(p, env), cfg.schema, 0.0) if p["intent"] != "none" else p for p in base]
    s0, s1 = score(test, base), score(test, grounded)
    # absent condition
    absent_rows = [i for i, r in enumerate(test) if open_slot_values(r)]
    abstained = 0
    for i in absent_rows:
        r = test[i]
        missing = {v for _, v in open_slot_values(r)}
        env_minus = Environment([x for x in env.folders if x not in missing], [x for x in env.files if x not in missing],
                                [x for x in env.apps if x not in missing], [x for x in env.contacts if x not in missing])
        p = base[i]
        g = ground(p, env_minus) if p["intent"] != "none" else p
        abstained += g["intent"] == "none"
    results[name] = {
        "strict": {"exact_match": s0["exact_match"], "intent_acc": s0["intent_acc"], "slot_f1": s0["slot"]["f1"],
                   "false_action_rate": s0["false_action_rate"],
                   "deva_em": s0["breakdown"]["script"]["deva"]["exact_match"]},
        "grounded": {"exact_match": s1["exact_match"], "intent_acc": s1["intent_acc"], "slot_f1": s1["slot"]["f1"],
                     "false_action_rate": s1["false_action_rate"],
                     "deva_em": s1["breakdown"]["script"]["deva"]["exact_match"],
                     "over_abstain_rate": s1["over_abstain_rate"]},
        "absent_entity_abstain_rate": round(abstained / len(absent_rows), 4),
        "absent_rows": len(absent_rows),
    }
    print(name, json.dumps(results[name]))
Path("results/grounding.json").write_text(json.dumps(results, indent=1))
