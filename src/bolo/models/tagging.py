"""Shared word-level BIO tagging scheme for joint intent + slot models.

Open slots (folder, path, name, app, level) are tagged by slot name; their value is read from the span.
Closed slots carry their value in the tag (e.g. `target=battery`, `direction=up`, `callapp=whatsapp`), so the
model classifies the value directly and no lookup table is needed.
"""

from __future__ import annotations

from ..config import Config
from ..normalize import Normalizer, phonetic_key

OPEN_TAGS = ["folder", "path", "name", "app", "level", "amount", "task", "query", "site", "message"]
NUMBER_SLOTS = ("level", "amount")
CALLAPP_INTENTS = ("call_contact", "send_message")


def closed_slots(cfg: Config) -> dict[str, list[str]]:
    """slot name -> closed values, for closed slots that carry their value in the tag (call apps are separate)."""
    st, out = cfg.schema["slot_types"], {}
    for spec in cfg.schema["intents"].values():
        for slot, sdef in (spec.get("slots") or {}).items():
            vals = st[sdef["type"]].get("values")
            if vals and sdef["type"] != "call_app":
                out[slot] = sorted(set(out.get(slot, [])) | {str(v) for v in vals})
    return out


def label_names(cfg: Config) -> list[str]:
    st = cfg.schema["slot_types"]
    closed = [f"{slot}={v}" for slot, vals in sorted(closed_slots(cfg).items()) for v in vals]
    closed += [f"callapp={v}" for v in st["call_app"]["values"]]
    labels = ["O"]
    for t in OPEN_TAGS + closed:
        labels += [f"B-{t}", f"I-{t}"]
    return labels


CLOSED_TAG_SLOTS = ("target", "direction", "action", "setting", "state", "unit", "engine")


def span_tag(intent: str, slot: str, value) -> str:
    if slot in CLOSED_TAG_SLOTS:
        return f"{slot}={value}"
    if slot == "app" and intent in CALLAPP_INTENTS:
        return f"callapp={value}"
    return slot


def word_tags(norm: Normalizer, text: str, spans: list[dict], intent: str) -> tuple[list[str], list[str]]:
    words = norm.words(text)
    tags = ["O"] * len(words)
    for sp in spans:
        t = span_tag(intent, sp["slot"], sp["value"])
        first = True
        for i, (_w, s, e) in enumerate(words):
            if s < sp["end"] and e > sp["start"]:
                tags[i] = ("B-" if first else "I-") + t
                first = False
    return [w for w, _, _ in words], tags


class NumberParser:
    def __init__(self, cfg: Config):
        self.words: dict[str, int] = {}
        for v, (roman, _dev) in cfg.numbers_hi.items():
            self.words[phonetic_key(roman)] = v
        for v, w in cfg.numbers_en.items():
            if " " not in w:
                self.words[phonetic_key(w)] = v

    def parse(self, roman_words: list[str]) -> int | None:
        keys = [phonetic_key(w) for w in roman_words]
        for i, (w, k) in enumerate(zip(roman_words, keys)):
            if w.isdigit():
                return max(0, min(100, int(w)))
            if k in self.words:
                v = self.words[k]
                if i + 1 < len(keys) and keys[i + 1] == phonetic_key("five") and v in (20, 70):
                    v += 5
                return v
        return None


def decode(cfg: Config, norm: Normalizer, numbers: NumberParser, intent: str,
           words: list[str], tags: list[str]) -> dict:
    """Turn word tags into a slot dict valid-shaped for `intent` (schema validation happens later)."""
    spans: list[tuple[str, list[str]]] = []
    for w, t in zip(words, tags):
        if t == "O":
            continue
        bio, name = t[:1], t[2:]
        if bio == "B" or not spans or spans[-1][0] != name:
            spans.append((name, [w]))
        else:
            spans[-1][1].append(w)
    allowed = (cfg.schema["intents"].get(intent, {}).get("slots") or {})
    slots: dict = {}
    for name, ws in spans:
        if "=" in name:
            key, value = name.split("=", 1)
            slot = "app" if key == "callapp" else key
            if slot in allowed and slot not in slots:
                slots[slot] = value
            continue
        roman = [norm.to_roman(w) for w in ws]
        if name in NUMBER_SLOTS:
            v = numbers.parse(roman)
            if v is not None and name in allowed and name not in slots:
                slots[name] = v
            continue
        if name in allowed and name not in slots:
            slots[name] = " ".join(roman).lower()
    return slots
