"""Load and validate Bolo YAML configs (schema, lexicon, templates)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
# repo checkout: <repo>/configs ; pip-installed wheel: configs are packaged inside bolo/configs
DEFAULT_CONFIG_DIR = (REPO_ROOT / "configs") if (REPO_ROOT / "configs" / "schema.yaml").exists() \
    else Path(__file__).resolve().parent / "configs"

TOKEN_RE = re.compile(r"^(\{[^}]+\}|<[^>]+>|[^\s{}<>]+)$")


# Value pools in lexicon.yaml, by how they are used.
OPEN_POOLS = ("folders", "files", "contacts", "apps", "tasks", "tasks_en", "queries", "sites", "messages", "messages_en")
SURFACE_POOLS = ("call_apps", "status_targets", "settings", "time_units", "time_units_en", "engines")
PHRASE_POOLS = ("volume_dir_hi", "volume_dir_en", "brightness_dir_hi", "brightness_dir_en", "media_actions_hi",
                "media_actions_en", "toggle_states_hi", "toggle_states_en")


@dataclass(frozen=True)
class Tok:
    """A rendered unit: one word, or a whole multi-word slot surface."""

    roman: str
    dev: str | None
    lang: str  # hi | en | ne (name) | num
    slot: str | None = None


@dataclass(frozen=True)
class PoolEntry:
    value: str  # canonical label value
    roman: str  # spoken surface (Roman)
    dev: str | None
    kind: str | None = None  # folder | file (paths only)


@dataclass
class Template:
    intent: str
    index: int
    text: str
    tokens: list[str]
    weight: float = 1.0
    lang: str | None = None
    kind: str | None = None
    only: dict[str, list[str]] = field(default_factory=dict)
    reason: str | None = None

    @property
    def id(self) -> str:
        return f"{self.intent}:{self.index}"


@dataclass
class Config:
    schema: dict
    hindi: dict[str, str]
    english: dict[str, str | None]
    classes: dict[str, list[list[str]]]
    spelling: dict[str, list[str]]
    open_pools: dict[str, list[PoolEntry]]
    closed_pools: dict[str, dict[str, list[list[Tok]]]]
    numbers_hi: dict[int, tuple[str, str]]
    numbers_en: dict[int, str]
    templates: dict[str, list[Template]]

    def word(self, w: str) -> Tok:
        if w in self.hindi:
            return Tok(w, self.hindi[w], "hi")
        if w in self.english:
            return Tok(w, self.english[w], "en")
        raise KeyError(f"word not in lexicon: {w!r}")

    def slot_type(self, intent: str, slot: str) -> dict:
        sdef = self.schema["intents"][intent]["slots"][slot]
        return self.schema["slot_types"][sdef["type"]]


def _split_entry(s: str) -> tuple[str, str | None]:
    if "|" in s:
        roman, dev = s.split("|", 1)
        return roman.strip(), dev.strip()
    return s.strip(), None


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_dir: Path | str = DEFAULT_CONFIG_DIR) -> Config:
    config_dir = Path(config_dir)
    schema = _load_yaml(config_dir / "schema.yaml")
    lex = _load_yaml(config_dir / "lexicon.yaml")
    tpl = _load_yaml(config_dir / "templates.yaml")

    hindi = {str(k): v for k, v in lex["hindi_words"].items()}
    english = {str(k): v for k, v in lex["english_words"].items()}
    overlap = set(hindi) & set(english)
    if overlap:
        raise ValueError(f"words in both hindi and english lists: {sorted(overlap)}")

    classes = {name: [str(p).split() for p in phrases] for name, phrases in lex["classes"].items()}
    spelling = {str(k): [str(x) for x in v] for k, v in lex["spelling"].items()}

    pools = lex["pools"]
    open_pools: dict[str, list[PoolEntry]] = {}
    for name in OPEN_POOLS:
        if name not in pools:
            continue
        entries = []
        for raw in pools[name]:
            roman, dev = _split_entry(str(raw))
            entries.append(PoolEntry(roman, roman, dev, kind=name[:-1] if name in ("folders", "files") else None))
        open_pools[name] = entries
    open_pools["paths"] = open_pools["folders"] + open_pools["files"]

    lang_for = {"contacts": "ne"}
    closed_pools: dict[str, dict[str, list[list[Tok]]]] = {}
    # Surface pools: each surface is one whole-phrase token.
    for name in SURFACE_POOLS:
        if name not in pools:
            continue
        closed_pools[name] = {}
        for canon, surfaces in pools[name].items():
            toks = []
            for s in surfaces:
                roman, dev = _split_entry(str(s))
                toks.append([Tok(roman, dev, lang_for.get(name, "en"))])
            closed_pools[name][str(canon)] = toks
    # Phrase pools: rendered word by word through the dictionary.
    cfg_partial = Config(schema, hindi, english, classes, spelling, open_pools, {}, {}, {}, {})
    for name in PHRASE_POOLS:
        if name not in pools:
            continue
        closed_pools[name] = {
            str(canon): [[cfg_partial.word(w) for w in str(p).split()] for p in phrases]
            for canon, phrases in pools[name].items()
        }

    numbers_hi = {int(k): _split_entry(str(v)) for k, v in lex["numbers_hi"].items()}
    numbers_en = {int(k): str(v) for k, v in lex["numbers_en"].items()}

    templates: dict[str, list[Template]] = {}
    for intent, items in tpl.items():
        if intent not in schema["intents"]:
            raise ValueError(f"templates for unknown intent {intent!r}")
        out = []
        for i, item in enumerate(items):
            if isinstance(item, str):
                item = {"t": item}
            text = str(item["t"])
            tokens = text.split()
            for t in tokens:
                if not TOKEN_RE.match(t):
                    raise ValueError(f"bad token {t!r} in template {text!r}")
            out.append(
                Template(
                    intent=intent,
                    index=i,
                    text=text,
                    tokens=tokens,
                    weight=float(item.get("w", 1.0)),
                    lang=item.get("lang"),
                    kind=item.get("kind"),
                    only={k: [str(x) for x in v] for k, v in (item.get("only") or {}).items()},
                    reason=item.get("reason"),
                )
            )
        templates[intent] = out

    return Config(schema, hindi, english, classes, spelling, open_pools, closed_pools,
                  numbers_hi, numbers_en, templates)
