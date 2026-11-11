"""Template-based Hinglish command generator.

Each example is rendered once as a token sequence, then realised as:
  - text      : what a user might type (Roman / Devanagari / mixed script, with spelling noise)
  - asr_text  : canonical lowercase Roman, no noise or punctuation (the ASR training target)
  - tts_text  : mixed script (Hindi words in Devanagari, English in Latin), clean input for a TTS teacher
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import Config, PoolEntry, Template, Tok, load_config
from .schema import validate_label
from .splits import SPLITS, template_heldout, value_split

SCRIPT_WEIGHTS = {"roman": 0.6, "deva": 0.2, "mixed": 0.2}
NOISE_P = 0.2  # per-word probability of a spelling variant (Roman words only)
JOIN_P = 0.15  # "kar do" -> "kardo"
JOIN_HEADS = {"kar", "khol", "laga", "badha", "rakh", "chala"}
JOIN_TAILS = {"do", "de", "lo"}
MIN_TEMPLATES_FOR_HOLDOUT = 8

DEFAULT_SIZES = {"train": 5000, "dev": 750, "test": 1000}
INTENT_WEIGHTS = {
    "open_terminal": 1.0,
    "open_claude_in_folder": 1.0,
    "call_contact": 1.0,
    "open_app": 1.0,
    "check_status": 0.9,
    "open_file_or_folder": 1.0,
    "set_volume": 0.8,
    "set_brightness": 0.7,
    "none": 1.2,
}

SLOT_RE = re.compile(r"^\{([^}:]+)(?::([^}]+))?\}$")
CLASS_RE = re.compile(r"^<([^>?]+)(\?)?>$")

TYPE_POOL = {
    "folder": "folders",
    "path": "paths",
    "contact": "contacts",
    "app": "apps",
    "call_app": "call_apps",
    "status_target": "status_targets",
    "volume_direction": "volume_dir_hi",
    "brightness_direction": "brightness_dir_hi",
}


@dataclass
class Rendered:
    toks: list[Tok]
    slots: dict
    template: Template
    fillers: list[str]


def _norm_key(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


class Generator:
    def __init__(self, cfg: Config, seed: int = 13):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.split_values: dict[str, dict[str, list[PoolEntry]]] = {}
        for name, entries in cfg.open_pools.items():
            by_split: dict[str, list[PoolEntry]] = {s: [] for s in SPLITS}
            for e in entries:
                by_split[value_split(e.value)].append(e)
            self.split_values[name] = by_split
        self.heldout = {}
        for intent, tpls in cfg.templates.items():
            enough = len(tpls) >= MIN_TEMPLATES_FOR_HOLDOUT
            for t in tpls:
                self.heldout[t.id] = enough and template_heldout(t.text)

    # ---------- rendering ----------
    def _level(self, tpl: Template) -> tuple[int, Tok]:
        r = self.rng.random()
        english = tpl.lang == "en"
        p_digit, p_hi = (0.6, 0.0) if english else (0.45, 0.4)
        if r < p_digit:
            value = self.rng.randint(0, 100) if self.rng.random() < 0.3 else self.rng.choice(range(0, 101, 5))
            return value, Tok(str(value), str(value), "num")
        if r < p_digit + p_hi:
            value = self.rng.choice(sorted(self.cfg.numbers_hi))
            roman, dev = self.cfg.numbers_hi[value]
            return value, Tok(roman, dev, "hi")
        value = self.rng.choice(sorted(self.cfg.numbers_en))
        return value, Tok(self.cfg.numbers_en[value], None, "en")

    def _fill_slot(self, tpl: Template, slot: str, pool: str | None, split: str, labeled: bool):
        if labeled:
            stype_name = self.cfg.schema["intents"][tpl.intent]["slots"][slot]["type"]
            if stype_name == "level":
                value, tok = self._level(tpl)
                return value, [Tok(tok.roman, tok.dev, tok.lang, slot)]
            pool = pool or TYPE_POOL[stype_name]
        if pool in self.split_values:
            cands = self.split_values[pool][split]
            if tpl.kind:
                cands = [e for e in cands if e.kind == tpl.kind]
            e = self.rng.choice(cands)
            lang = "ne" if pool == "contacts" else "en"
            return e.value, [Tok(e.roman, e.dev, lang, slot if labeled else None)]
        closed = self.cfg.closed_pools[pool]
        canon_opts = sorted(closed)
        if slot in tpl.only:
            canon_opts = [c for c in canon_opts if c in tpl.only[slot]]
        canon = self.rng.choice(canon_opts)
        phrase = self.rng.choice(closed[canon])
        return canon, [Tok(t.roman, t.dev, t.lang, slot if labeled else None) for t in phrase]

    def render(self, tpl: Template, split: str) -> Rendered:
        toks: list[Tok] = []
        slots: dict = {}
        fillers: list[str] = []
        for raw in tpl.tokens:
            m = SLOT_RE.match(raw)
            if m:
                name, pool = m.group(1), m.group(2)
                labeled = name != "_"
                value, stoks = self._fill_slot(tpl, name, pool, split, labeled)
                if labeled:
                    slots[name] = value
                else:
                    fillers.append(value)
                toks.extend(stoks)
                continue
            m = CLASS_RE.match(raw)
            if m:
                name, optional = m.group(1), bool(m.group(2))
                if optional and self.rng.random() < 0.5:
                    continue
                phrase = self.rng.choice(self.cfg.classes[name])
                toks.extend(self.cfg.word(w) for w in phrase)
                continue
            toks.append(self.cfg.word(raw))
        if tpl.intent == "none":
            slots = {}
        return Rendered(toks, slots, tpl, fillers)

    def realise(self, r: Rendered, script: str, noisy: bool) -> tuple[str, list[dict]]:
        """Turn tokens into a string in the given script; return (text, slot spans)."""
        parts: list[tuple[str, str | None]] = []
        for i, t in enumerate(r.toks):
            if script == "roman":
                s = t.roman
            elif script == "deva":
                s = t.dev or t.roman
            else:  # mixed: Hindi words and names in Devanagari, English in Latin
                s = (t.dev or t.roman) if t.lang in ("hi", "ne") else t.roman
            if noisy and t.slot is None and s == t.roman and t.roman in self.cfg.spelling:
                if self.rng.random() < NOISE_P:
                    s = self.rng.choice(self.cfg.spelling[t.roman])
            parts.append((s, t.slot))
        text = ""
        spans: list[dict] = []
        for i, (s, slot) in enumerate(parts):
            if i > 0:
                prev_roman, cur_roman = r.toks[i - 1].roman, r.toks[i].roman
                joined = (noisy and script == "roman" and slot is None and parts[i - 1][1] is None
                          and prev_roman in JOIN_HEADS and cur_roman in JOIN_TAILS
                          and self.rng.random() < JOIN_P)
                text += "" if joined else " "
            start = len(text)
            text += s
            if slot is not None:
                if spans and spans[-1]["slot"] == slot and spans[-1]["end"] == start - 1:
                    spans[-1]["end"] = len(text)
                else:
                    spans.append({"slot": slot, "start": start, "end": len(text)})
        return text, spans

    def _decorate(self, text: str, script: str, tpl: Template) -> str:
        if not text:
            return text
        if script == "roman" and self.rng.random() < 0.15:
            text = text[0].upper() + text[1:]
        r = self.rng.random()
        question = any(w in tpl.tokens for w in ("kya", "what's", "how", "can", "could", "kab", "kaise"))
        if question and r < 0.3:
            text += "?"
        elif r < 0.08:
            text += "।" if script == "deva" else "."
        elif r < 0.12:
            text += "!"
        return text

    def example(self, tpl: Template, split: str) -> dict:
        r = self.render(tpl, split)
        script = "roman" if tpl.lang == "en" else self.rng.choices(
            list(SCRIPT_WEIGHTS), weights=list(SCRIPT_WEIGHTS.values()))[0]
        text, spans = self.realise(r, script, noisy=True)
        text = self._decorate(text, script, tpl)
        asr_text, _ = self.realise(r, "roman", noisy=False)
        tts_text, _ = self.realise(r, "mixed", noisy=False)
        words = [t for t in r.toks if t.slot is None and t.lang in ("hi", "en")]
        en = sum(1 for t in words if t.lang == "en")
        en_ratio = round(en / len(words), 3) if words else None
        validate_label(self.cfg.schema, tpl.intent, r.slots)
        return {
            "text": text,
            "script": script,
            "intent": tpl.intent,
            "slots": r.slots,
            "spans": [dict(s, value=r.slots[s["slot"]]) for s in spans],
            "asr_text": asr_text.lower(),
            "tts_text": tts_text,
            "meta": {
                "template_id": tpl.id,
                "template_heldout": self.heldout[tpl.id],
                "reason": tpl.reason,
                "fillers": r.fillers,
                "en_ratio": en_ratio,
            },
        }

    # ---------- dataset ----------
    def _templates_for(self, intent: str, split: str) -> list[Template]:
        tpls = self.cfg.templates[intent]
        if split == "test":
            return tpls
        return [t for t in tpls if not self.heldout[t.id]]

    def _quotas(self, n: int) -> dict[str, int]:
        intents = [i for i in INTENT_WEIGHTS if i in self.cfg.templates]
        total = sum(INTENT_WEIGHTS[i] for i in intents)
        q = {i: int(n * INTENT_WEIGHTS[i] / total) for i in intents}
        for i in intents[: n - sum(q.values())]:
            q[i] += 1
        return q

    def _fill(self, intent: str, split: str, quota: int, keys: set[str]) -> list[dict]:
        rows: list[dict] = []
        tpls = self._templates_for(intent, split)
        weights = [t.weight for t in tpls]
        attempts = 0
        while len(rows) < quota and attempts < 100 * quota:
            attempts += 1
            tpl = self.rng.choices(tpls, weights=weights)[0]
            ex = self.example(tpl, split)
            key = _norm_key(ex["asr_text"])
            if key in keys:
                continue
            keys.add(key)
            ex["_key"] = key
            rows.append(ex)
        return rows

    def generate(self, sizes: dict[str, int] | None = None) -> dict[str, list[dict]]:
        """Per-intent quotas; dedup within each split.

        Open-vocabulary values never cross splits (value-based split), so texts containing them can't
        leak. Closed-vocabulary texts (e.g. "battery kitni hai") may also occur in train; those dev/test
        rows get meta.seen_in_train = True so evaluation can report the unseen subset separately.
        """
        sizes = sizes or DEFAULT_SIZES
        out: dict[str, list[dict]] = {}
        train_keys: set[str] = set()
        for split in ("train", "dev", "test"):
            keys: set[str] = set()
            rows: list[dict] = []
            quotas = self._quotas(sizes[split])
            # fill in order; give any shortfall (saturated intents) to the next intents
            short = 0
            for intent in quotas:
                got = self._fill(intent, split, quotas[intent] + short, keys)
                short = quotas[intent] + short - len(got)
                rows.extend(got)
            if short:
                rows.extend(self._fill("open_terminal", split, short, keys))
            self.rng.shuffle(rows)
            for k, ex in enumerate(rows):
                key = ex.pop("_key")
                ex["id"] = f"{split}-{k:05d}"
                ex["split"] = split
                ex["meta"]["seen_in_train"] = split != "train" and key in train_keys
            if split == "train":
                train_keys = keys
            out[split] = rows
        return out


def stats(rows: list[dict]) -> dict:
    def bucket(r):
        x = r["meta"]["en_ratio"]
        if x is None:
            return "n/a"
        return "hi-heavy(<0.34)" if x < 0.34 else ("balanced" if x < 0.67 else "en-heavy(>=0.67)")

    return {
        "n": len(rows),
        "intents": dict(Counter(r["intent"] for r in rows).most_common()),
        "scripts": dict(Counter(r["script"] for r in rows).most_common()),
        "none_reasons": dict(Counter(r["meta"]["reason"] for r in rows if r["intent"] == "none").most_common()),
        "mixing_density": dict(Counter(bucket(r) for r in rows).most_common()),
        "heldout_templates": sum(1 for r in rows if r["meta"]["template_heldout"]),
        "seen_in_train": sum(1 for r in rows if r["meta"].get("seen_in_train")),
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_dataset(out_dir: Path | str, seed: int = 13, sizes: dict[str, int] | None = None,
                  config_dir: Path | str | None = None) -> dict:
    cfg = load_config(config_dir) if config_dir else load_config()
    gen = Generator(cfg, seed=seed)
    data = gen.generate(sizes)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in data.items():
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from .config import DEFAULT_CONFIG_DIR
    cdir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    manifest = {
        "generator": "bolo.generator v0 (templates only)",
        "seed": seed,
        "sizes": {s: len(r) for s, r in data.items()},
        "config_sha256": {p.name: _sha(p) for p in sorted(cdir.glob("*.yaml"))},
        "stats": {s: stats(r) for s, r in data.items()},
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest
