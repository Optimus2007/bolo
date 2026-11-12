"""Adaptation from use: Bolo gets better for one person as they accept or correct its actions.

This is a contextual bandit, not full RL. Each command is one decision: Bolo sees the context (the N-best
transcripts of the audio), picks an action, and gets a reward (the user accepts it, or says what they meant).
There is no long chain of decisions, so model based RL or meta RL would add machinery without a problem to solve.

Three things are learned, all from text and actions only. Audio is never stored.

1. Aliases. If Bolo heard "doctor" and the user meant the app "docker", the pair is remembered and the next
   "doctor" in that slot is read as "docker" before grounding.
2. A usage prior. Entities the user really uses get a small bonus: prior(e) = log(1 + count(e)).
3. Reranker weights. Each candidate action a gets a score s(a) = w . f(a) with features
   f = [asr log prob, log NLU confidence, grounding score, is_abstain, usage prior], and the policy is
   pi(a) = softmax(s). When the user tells us the right action a*, we take one gradient step on log pi(a*):
       w <- w + lr * (f(a*) - sum_a pi(a) f(a))
   which is the policy gradient (REINFORCE) update with full feedback. If a* is not in the N-best at all,
   only the alias is learned.

Store it wherever you like (`Memory.load(path)`); nothing is written unless you call `save`. The file holds
command texts and entity names, so treat it as personal data: it is gitignored by default under data/.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

FEATURES = ("asr", "nlu", "ground", "none", "prior")
ALIAS_SLOTS = {"folder", "path", "app", "name"}


@dataclass
class Memory:
    w: dict = field(default_factory=lambda: {"asr": 1.0, "nlu": 1.0, "ground": 1.0, "none": -1.0, "prior": 0.0})
    aliases: dict = field(default_factory=dict)   # "slot:heard" -> entity
    counts: dict = field(default_factory=dict)    # entity -> times used
    lr: float = 0.3
    alias_min_sim: float = 0.6
    updates: int = 0

    # ---------- persistence ----------
    @classmethod
    def load(cls, path: str | Path) -> "Memory":
        p = Path(path)
        return cls(**json.loads(p.read_text(encoding="utf-8"))) if p.exists() else cls()

    def save(self, path: str | Path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.__dict__, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def from_rerank(cls, rr) -> "Memory":
        return cls(w={"asr": 1.0, "nlu": rr.w_nlu, "ground": rr.w_ground, "none": -rr.w_none, "prior": 0.0})

    # ---------- use ----------
    def apply_aliases(self, pred: dict) -> dict:
        slots = {k: self.aliases.get(f"{k}:{str(v).lower()}", v) for k, v in pred["slots"].items()}
        return pred | {"slots": slots}

    def prior(self, pred: dict) -> float:
        vals = [str(v) for v in pred["slots"].values()]
        return max((math.log1p(self.counts.get(v, 0)) for v in vals), default=0.0)

    def score(self, f: dict) -> float:
        return sum(self.w[k] * f[k] for k in FEATURES)

    def candidates(self, hyps, nlu, cfg, env) -> list[dict]:
        """One entry per distinct action: its best features and the transcript that produced it."""
        from .evaluate import finalize
        from .grounding import ground

        texts = [" ".join(t.lower().replace("।", " ").split()) for t, _ in hyps]
        raws = nlu.predict(texts)
        by_action: dict = {}
        for (t, lp), raw in zip(hyps, raws):
            p = finalize(raw, cfg.schema, 0.0)
            heard = dict(p["slots"])
            p = self.apply_aliases(p)
            g = 1.0
            if p["intent"] != "none" and env is not None:
                gg = ground(p, env)
                p = finalize(gg, cfg.schema, 0.0)
                sc = [v["score"] for v in gg["grounding"].values() if isinstance(v, dict) and "score" in v]
                g = min(sc) if sc else 1.0
            f = {"asr": lp, "nlu": math.log(max(1e-6, float(raw.get("confidence", 1.0)))), "ground": g,
                 "none": float(p["intent"] == "none"), "prior": self.prior(p)}
            key = (p["intent"], tuple(sorted((k, str(v)) for k, v in p["slots"].items())))
            c = {"key": key, "pred": p, "heard": heard, "text": t, "f": f, "s": self.score(f)}
            if key not in by_action or c["s"] > by_action[key]["s"]:
                by_action[key] = c
        return sorted(by_action.values(), key=lambda c: -c["s"])

    def choose(self, hyps, nlu, cfg, env) -> tuple[dict, list[dict]]:
        cands = self.candidates(hyps, nlu, cfg, env)
        return cands[0], cands

    # ---------- learn ----------
    def feedback(self, cands: list[dict], correct: dict, chosen: dict | None = None):
        """correct: the action the user wanted ({"intent", "slots"}); chosen: the candidate Bolo picked."""
        key = (correct["intent"], tuple(sorted((k, str(v)) for k, v in correct["slots"].items())))
        for v in correct["slots"].values():
            self.counts[str(v)] = self.counts.get(str(v), 0) + 1
        if chosen and chosen["pred"]["intent"] == correct["intent"]:
            from .grounding import similarity

            for k, v in chosen["heard"].items():
                want = correct["slots"].get(k)
                # only names (not numbers or fixed choices like up/down), and only when it sounds alike:
                # "doctor" -> "docker" is a mishearing worth remembering, "cpu" -> "bluetooth" is not
                if k in ALIAS_SLOTS and want is not None and str(want) != str(v) \
                        and similarity(str(v), str(want)) >= self.alias_min_sim:
                    self.aliases[f"{k}:{str(v).lower()}"] = want
        hit = [c for c in cands if c["key"] == key]
        if not hit or len(cands) < 2:
            return
        m = max(c["s"] for c in cands)
        ps = [math.exp(c["s"] - m) for c in cands]
        z = sum(ps)
        for k in FEATURES:
            if k == "asr":  # keep the acoustic weight fixed at 1 so the scale stays anchored
                continue
            expect = sum(p / z * c["f"][k] for p, c in zip(ps, cands))
            self.w[k] += self.lr * (hit[0]["f"][k] - expect)
        self.updates += 1
