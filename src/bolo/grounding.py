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

    def pool(self, kind: str) -> list[str]:
        return {"folder": self.folders, "path": self.folders + self.files, "app": self.apps,
                "name": self.contacts}[kind]

    def best(self, value: str, kind: str) -> tuple[str | None, float]:
        pool = self.pool(kind)
        best, score = None, 0.0
        for e in pool:
            s = similarity(value, e)
            if s > score:
                best, score = e, s
        return best, score


def snap_site(value: str, words: list[str], sites: list[str], min_sim: float = 0.85) -> str:
    """Snap a website to a known one, using the words around it: speech models often split one-word names
    ("money control", "you tube", "linked in") and the tagger may then keep only part ("money").
    Tries every 1-3 word window of the command that overlaps the predicted value, joined without spaces.
    Unknown sites pass through unchanged."""
    vw = set(str(value).lower().split())
    cands = {str(value).lower()}
    for n in (1, 2, 3):
        for i in range(len(words) - n + 1):
            win = words[i:i + n]
            if vw & set(win):
                cands.add(" ".join(win))
    best, score = value, 0.0
    for c in cands:
        cj = c.replace(" ", "")
        for site in sites:
            sc = similarity(cj, site.replace(" ", ""))
            if sc > score:
                best, score = site, sc
    return best if score >= min_sim else value


SLOT_KIND = {("open_terminal", "folder"): "folder", ("open_claude_in_folder", "folder"): "folder",
             ("open_file_or_folder", "path"): "path", ("open_app", "app"): "app", ("call_contact", "name"): "name",
             ("close_app", "app"): "app", ("find_file", "path"): "path", ("send_message", "name"): "name"}


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
        if not env.pool(kind):  # nothing known of this kind (e.g. contacts were not shared): cannot verify, pass through
            out["grounding"][slot] = {"heard": value, "matched": None, "verified": False}
            continue
        ent, s = env.best(str(value), kind)
        out["grounding"][slot] = {"heard": value, "matched": ent, "score": round(s, 3)}
        if ent is None or s < threshold:
            return {"intent": "none", "slots": {}, "confidence": out["confidence"],
                    "grounding": {**out["grounding"], "abstain": f"no {kind} like {value!r}"}}
        out["slots"][slot] = ent
    return out
