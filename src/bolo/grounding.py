"""Environment-grounded slot resolution.

At action time the assistant knows what actually exists on the device (folders, files, apps, contacts).
Each predicted open slot value is matched against those entities with a Hinglish-aware phonetic
similarity. This:
  1. repairs spelling/transliteration (कैगल -> "kaigal" -> kaggle),
  2. resolves app-vs-folder ambiguity ("bitwarden kholo": is there an app or a folder by that name?),
  3. abstains when nothing plausible exists, instead of acting on a guess.
Similarity = max(SequenceMatcher ratio over phonetic keys, ratio over consonant skeletons).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache

from .normalize import phonetic_key, skeleton


@lru_cache(maxsize=200000)
def _keys(s: str) -> tuple[str, str]:
    ws = s.lower().split()
    return " ".join(phonetic_key(w) for w in ws), " ".join(skeleton(w) for w in ws)


def similarity(a: str, b: str) -> float:
    ka, sa = _keys(a)
    kb, sb = _keys(b)
    if ka == kb:
        return 1.0
    return max(SequenceMatcher(None, ka, kb).ratio(), 0.9 * SequenceMatcher(None, sa, sb).ratio())


@dataclass
class Environment:
    folders: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    apps: list[str] = field(default_factory=list)
    contacts: list[str] = field(default_factory=list)

    def best(self, value: str, kind: str) -> tuple[str | None, float]:
        pool = {"folder": self.folders, "path": self.folders + self.files, "app": self.apps,
                "name": self.contacts}[kind]
        best, score = None, 0.0
        for e in pool:
            s = similarity(value, e)
            if s > score:
                best, score = e, s
        return best, score


SLOT_KIND = {("open_terminal", "folder"): "folder", ("open_claude_in_folder", "folder"): "folder",
             ("open_file_or_folder", "path"): "path", ("open_app", "app"): "app", ("call_contact", "name"): "name"}


def ground(pred: dict, env: Environment, threshold: float = 0.72, margin: float = 0.05) -> dict:
    """Return a new prediction with open slot values snapped to existing entities (or `none`)."""
    out = {"intent": pred["intent"], "slots": dict(pred["slots"]), "confidence": pred.get("confidence", 1.0),
           "grounding": {}}
    intent = out["intent"]
    # app <-> file/folder disambiguation by what exists
    if intent in ("open_app", "open_file_or_folder"):
        val = out["slots"].get("app") or out["slots"].get("path")
        if val:
            a, sa = env.best(val, "app")
            p, sp = env.best(val, "path")
            if intent == "open_app" and sp >= threshold and sp > sa + margin:
                out.update(intent="open_file_or_folder", slots={"path": val})
                out["grounding"]["switched"] = "open_app->open_file_or_folder"
            elif intent == "open_file_or_folder" and sa >= threshold and sa > sp + margin:
                out.update(intent="open_app", slots={"app": val})
                out["grounding"]["switched"] = "open_file_or_folder->open_app"
    for slot, value in list(out["slots"].items()):
        kind = SLOT_KIND.get((out["intent"], slot))
        if kind is None:
            continue
        ent, s = env.best(str(value), kind)
        out["grounding"][slot] = {"heard": value, "matched": ent, "score": round(s, 3)}
        if ent is None or s < threshold:
            return {"intent": "none", "slots": {}, "confidence": out["confidence"],
                    "grounding": {**out["grounding"], "abstain": f"no {kind} like {value!r}"}}
        out["slots"][slot] = ent
    return out
