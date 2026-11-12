"""Routines: teach Myna one phrase that does several commands.

    bolo sikhao "coding mode" "vs code kholo" "projects folder mein terminal kholo" "dnd on karo"
    bolo do "coding mode on karo" --run      # opens VS Code, a terminal in projects, turns on Do Not Disturb
    bolo sikhao --list
    bolo sikhao --forget "coding mode"

Each step is a normal command, understood once when it is taught (so you see exactly what Myna understood before
saving it) and grounded again every time the routine runs (folders can move). A heard phrase matches a routine
only if, after the usual "karo / chalu karo / on karo" words are taken off, it sounds like the routine's name as a
whole, so "coding mode" never fires on "coding folder kholo". Saved next to the assistant name in
~/.config/bolo/settings.json, on this computer only.
"""
from __future__ import annotations

import json
import re

from .assistant import _load, settings_path

MAX_STEPS = 8
MAX_ROUTINES = 50
NAME_RE = re.compile(r"^[a-z0-9ऀ-ॿ][a-z0-9ऀ-ॿ ]{1,38}[a-z0-9ऀ-ॿ]$")
# words around a routine name that only mean "start it"
TRIGGER = {"karo", "kar", "do", "dena", "de", "chalu", "shuru", "start", "on", "kardo", "kar do", "karna", "please",
           "plz", "bhai", "yaar", "zara", "jara", "abhi", "wala", "waala", "mode"}
MIN_SIM = 0.86


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\sऀ-ॿ]", " ", s.lower()).split())


def _strip_trigger(words: list[str], keep: set[str]) -> list[str]:
    while words and words[-1] in TRIGGER and words[-1] not in keep:
        words = words[:-1]
    while words and words[0] in TRIGGER and words[0] not in keep:
        words = words[1:]
    return words


def get_routines() -> dict[str, list[dict]]:
    return _load().get("routines") or {}


def _save(routines: dict) -> None:
    s = _load()
    if routines:
        s["routines"] = routines
    else:
        s.pop("routines", None)
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")


def teach(name: str, steps: list[dict]) -> str:
    """steps: [{"text", "intent", "slots"}] as understood. Returns the saved name."""
    n = norm(name)
    if not NAME_RE.match(n) or len(n.split()) > 4:
        raise ValueError("a routine name is 3-40 letters or numbers, at most 4 words")
    if not steps or len(steps) > MAX_STEPS:
        raise ValueError(f"a routine has 1-{MAX_STEPS} steps")
    bad = [s["text"] for s in steps if s.get("intent") in (None, "none")]
    if bad:
        raise ValueError("not understood, so not saved: " + "; ".join(repr(b) for b in bad))
    r = get_routines()
    if n not in r and len(r) >= MAX_ROUTINES:
        raise ValueError(f"at most {MAX_ROUTINES} routines")
    r[n] = [{"text": s["text"], "intent": s["intent"], "slots": s.get("slots") or {}} for s in steps]
    _save(r)
    return n


def forget(name: str) -> bool:
    r = get_routines()
    hit = r.pop(norm(name), None)
    _save(r)
    return hit is not None


def match(text: str, routines: dict | None = None) -> str | None:
    """The routine this phrase starts, or None."""
    from .grounding import similarity

    routines = get_routines() if routines is None else routines
    words = norm(text).split()
    best, best_s = None, 0.0
    for name in routines:
        nw = name.split()
        said = _strip_trigger(words, set(nw))
        if not said or abs(len(said) - len(nw)) > 0:
            continue
        s = 1.0 if said == nw else similarity(" ".join(said), name)
        if s > best_s:
            best, best_s = name, s
    return best if best_s >= MIN_SIM else None
