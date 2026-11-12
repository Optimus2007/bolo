"""Does Bolo get better with use? A predict-then-learn simulation of one user.

Stream: the 240 held-out commands (data/say_test), then the same commands again said by other voices
(data/say_test_v1), each pass in a shuffled order. For every clip Bolo first acts (scored), then the simulated
user gives feedback: accepts a right action or says what they meant. Compared:
  static      tuned reranker, never learns
  aliases     only the correction memory
  bandit      only the online reranker weights (policy gradient) + usage prior
  adaptive    all of it
Writes results/speech/adapt_sim.json. Run scripts/eval_decoding.py tune first.
"""
import copy
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import eval_decoding as E  # noqa: E402
from bolo.adapt import Memory  # noqa: E402
from bolo.asr.decode import Rerank  # noqa: E402
from bolo.evaluate import _exact  # noqa: E402

t = json.loads(Path("results/speech/decoding_tune.json").read_text())
cands = {k: v for k, v in t.items() if "rerank" in k}
best = max(cands, key=lambda k: (cands[k]["e2e"], -cands[k]["wrong_action"]))
beams, bonus = int(best.split("_")[0][4:]), int(best.split("_")[1][5:])
rr = Rerank(**cands[best]["rerank"])

stream = []
for name in ("say_test", "say_test_v1"):
    rows = E.load_set(f"data/{name}/manifest.jsonl", "data/text/v0/test.jsonl", name)
    dec = E.decode(name, rows, E.audio(rows), beams, bonus)
    pas = list(zip(rows, dec))
    random.Random(5).shuffle(pas)
    stream.append(pas)


def simulate(learn_alias: bool, learn_w: bool) -> dict:
    mem = Memory.from_rerank(rr)
    out = {}
    for i, pas in enumerate(stream):
        ok = false = 0
        for r, d in pas:
            top, cs = mem.choose([tuple(h) for h in d["hyps"]], E.nlu, E.cfg, E.env)
            good = _exact(r, top["pred"])
            ok += good
            false += top["pred"]["intent"] != "none" and not good
            gold = {"intent": r["intent"], "slots": r["slots"]}
            m2 = copy.deepcopy(mem)
            mem.feedback(cs, gold, top)
            if not learn_alias:
                mem.aliases = m2.aliases
            if not learn_w:
                mem.w, mem.counts = m2.w, m2.counts
        out[f"pass{i + 1}"] = {"e2e": round(ok / len(pas), 4), "wrong_action": round(false / len(pas), 4)}
    out["final_w"] = {k: round(v, 3) for k, v in mem.w.items()}
    out["aliases_learned"] = len(mem.aliases)
    return out


res = {"decoding": best, "start_weights": rr.__dict__}
for name, a, w in [("static", False, False), ("aliases", True, False), ("bandit", False, True), ("adaptive", True, True)]:
    res[name] = simulate(a, w)
    print(name, res[name], flush=True)
Path("results/speech/adapt_sim.json").write_text(json.dumps(res, indent=1))
