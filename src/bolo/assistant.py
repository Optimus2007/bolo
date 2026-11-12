"""The assistant's name (default "Myna"), chosen by the user, and removing it from commands.

Bolo is the name of the models; the assistant on top of them is called Myna unless the user renames it:

    bolo name            # show
    bolo name Jarvis     # rename
    bolo name --reset    # back to Myna

People address an assistant by name ("Myna, volume kam karo", "hey myna terminal kholo", "मैना ..."). The command
model was never trained with the name in front, so `strip_name` removes it first. It is matched loosely (by sound,
so "maina", "mynah" and "मैना" all count), but it is never removed when the next word shows it is a person being
talked about ("rahul ko call karo" stays as it is even if the assistant is called Rahul).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

DEFAULT_NAME = "Myna"
CALLS = {"hey", "hi", "hello", "ok", "okay", "arre", "are", "arey", "suno", "sun", "oye", "oi", "ae", "ay", "yo"}
# words that, right after a name, mean the name is the object of the command, not the one being addressed
POSTPOSITIONS = {"ko", "se", "ka", "ki", "ke", "ne", "mein", "me", "main", "par", "pe", "wala", "wali", "wale", "tak", "ka"}
HONORIFICS = {"ji", "jee"}
NAME_RE = re.compile(r"^[A-Za-zऀ-ॿ][A-Za-zऀ-ॿ' -]{0,22}[A-Za-zऀ-ॿ]$")
MIN_SIM = 0.75


def settings_path() -> Path:
    return Path(os.environ.get("BOLO_SETTINGS") or Path.home() / ".config" / "bolo" / "settings.json")


def _load() -> dict:
    p = settings_path()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def get_name() -> str:
    return _load().get("assistant_name") or DEFAULT_NAME


def set_name(name: str | None) -> str:
    """Save a new name (None = back to the default). Returns the name now in use."""
    s = _load()
    if name is None:
        s.pop("assistant_name", None)
    else:
        name = " ".join(name.split())
        if not NAME_RE.match(name) or len(name.split()) > 3:
            raise ValueError("a name is 2-24 letters (Roman or Devanagari), at most 3 words")
        s["assistant_name"] = name
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    return get_name()


def _roman(w: str, to_roman) -> str:
    w = re.sub(r"[^\wऀ-ॿ]", "", w)
    return (to_roman(w) if to_roman else w).lower()


def _is_name(words: list[str], name_words: list[str]) -> bool:
    from .grounding import similarity

    return len(words) == len(name_words) and similarity(" ".join(words), " ".join(name_words)) >= MIN_SIM


def strip_name(text: str, name: str | None = None, to_roman=None) -> tuple[str, bool]:
    """Remove a leading ("[hey] Myna[ ji], ...") or trailing ("..., Myna") address. Returns (text, removed)."""
    name = name or get_name()
    toks = text.split()
    if not toks:
        return text, False
    rom = [_roman(t, to_roman) for t in toks]
    nw = [_roman(w, to_roman) for w in name.split()]
    n = len(nw)
    i = 1 if rom and rom[0] in CALLS else 0
    if _is_name(rom[i:i + n], nw):
        j = i + n
        if j < len(rom) and rom[j] in HONORIFICS:
            j += 1
        nxt = rom[j] if j < len(rom) else ""
        comma = toks[j - 1].endswith((",", "!", "।"))
        if len(rom) - j >= 1 and (comma or i == 1 or nxt not in POSTPOSITIONS):
            return " ".join(toks[j:]), True
    if len(rom) > n + 1 and _is_name(rom[-n:], nw) and (toks[-n - 1].endswith(",") or rom[-n - 1] not in POSTPOSITIONS):
        k = len(toks) - n
        if rom[k - 1] in CALLS:
            k -= 1
        return " ".join(toks[:k]).rstrip(","), True
    return text, False
