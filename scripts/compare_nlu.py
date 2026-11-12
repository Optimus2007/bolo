"""Compare two Bolo Tiny variants end to end on the same clips (paired, per seed), from eval_human.py detail files.

  .venv/bin/python scripts/compare_nlu.py phono_views_aug phono_views_aug_asr --seeds 0 1 2 --sets human say
Prints mean right-action / wrong-action / over-abstain / acted-on-none per variant, and for each seed the clips fixed
and broken by B vs A with an exact McNemar p-value (two-sided binomial on the discordant pairs).
"""
import argparse
import json
from math import comb
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
ap.add_argument("--sets", nargs="+", default=["human", "say"])
ap.add_argument("--root", default="data/eval/s5")
ap.add_argument("--out")
args = ap.parse_args()


def load(v, s, st):
    return {json.loads(l)["utt_id"]: json.loads(l) for l in open(Path(args.root) / f"{v}_s{s}_{st}" / "bolo_sun_beam5.jsonl")}


def mcnemar(fixed, broken):
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def stats(rows):
    rows = list(rows.values())
    n = len(rows)
    none = [r for r in rows if r["gold"]["intent"] == "none"]
    cmd = [r for r in rows if r["gold"]["intent"] != "none"]
    return {"right": sum(r["ok"] for r in rows) / n,
            "wrong_action": sum(r["intent"] != "none" and not r["ok"] for r in rows) / n,
            "over_abstain": sum(r["intent"] == "none" for r in cmd) / max(1, len(cmd)),
            "acted_on_none": sum(r["intent"] != "none" for r in none) / max(1, len(none)), "n": n, "n_none": len(none)}


res = {}
for st in args.sets:
    res[st] = {"per_seed": [], "mean": {}}
    for v in (args.a, args.b):
        ss = [stats(load(v, s, st)) for s in args.seeds]
        res[st]["mean"][v] = {k: round(sum(x[k] for x in ss) / len(ss), 4) for k in ss[0] if k not in ("n", "n_none")}
        res[st]["mean"][v]["n"], res[st]["mean"][v]["n_none"] = ss[0]["n"], ss[0]["n_none"]
    for s in args.seeds:
        A, B = load(args.a, s, st), load(args.b, s, st)
        fixed = sum(1 for u in A if not A[u]["ok"] and B[u]["ok"])
        broken = sum(1 for u in A if A[u]["ok"] and not B[u]["ok"])
        res[st]["per_seed"].append({"seed": s, "fixed": fixed, "broken": broken, "p": round(mcnemar(fixed, broken), 4)})
    print(f"== {st} (n={res[st]['mean'][args.a]['n']}, not-a-command rows={res[st]['mean'][args.a]['n_none']})")
    for v in (args.a, args.b):
        m = res[st]["mean"][v]
        print(f"   {v:24s} right {m['right']*100:5.1f}%  wrong {m['wrong_action']*100:5.1f}%  "
              f"over-abstain {m['over_abstain']*100:5.1f}%  acted-on-none {m['acted_on_none']*100:5.1f}%")
    for p in res[st]["per_seed"]:
        print(f"   seed {p['seed']}: fixed {p['fixed']}, broken {p['broken']}, McNemar p={p['p']}")
if args.out:
    Path(args.out).write_text(json.dumps(res, indent=1))
