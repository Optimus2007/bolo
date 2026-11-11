import re

import pytest

from bolo.config import load_config
from bolo.generator import CLASS_RE, SLOT_RE, Generator
from bolo.schema import LabelError, validate_label
from bolo.splits import value_split

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
OPEN_SLOTS = {"folder", "path", "name", "app"}  # `app` is open only for open_app (call_contact uses closed)
SIZES = {"train": 900, "dev": 150, "test": 250}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def data(cfg):
    return Generator(cfg, seed=7).generate(SIZES)


def test_every_template_token_resolves(cfg):
    for intent, tpls in cfg.templates.items():
        slots = cfg.schema["intents"][intent]["slots"] or {}
        for t in tpls:
            for tok in t.tokens:
                m = SLOT_RE.match(tok)
                if m:
                    name, pool = m.group(1), m.group(2)
                    if name == "_":
                        assert pool, f"unlabeled filler needs a pool: {t.text}"
                    else:
                        assert name in slots, f"{intent}: unknown slot {name} in {t.text}"
                    if pool:
                        assert pool in cfg.open_pools or pool in cfg.closed_pools, pool
                    continue
                m = CLASS_RE.match(tok)
                if m:
                    assert m.group(1) in cfg.classes, f"unknown class {tok} in {t.text}"
                    continue
                cfg.word(tok)  # raises KeyError if missing


def test_class_and_spelling_words_in_lexicon(cfg):
    for phrases in cfg.classes.values():
        for p in phrases:
            for w in p:
                cfg.word(w)
    for w in cfg.spelling:
        cfg.word(w)


def test_open_pools_disjoint_and_unique(cfg):
    names = {k: [e.value for e in v] for k, v in cfg.open_pools.items() if k != "paths"}
    for k, vals in names.items():
        assert len(vals) == len(set(vals)), f"duplicate value in {k}"
    sets = {k: set(v) for k, v in names.items()}
    keys = list(sets)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert not (sets[a] & sets[b]), f"{a} and {b} share values: {sets[a] & sets[b]}"


def test_each_open_pool_has_values_in_every_split(cfg):
    for name in ("folders", "files", "contacts", "apps"):
        splits = {value_split(e.value) for e in cfg.open_pools[name]}
        assert splits == {"train", "dev", "test"}, name


def test_deterministic(cfg):
    a = Generator(cfg, seed=3).generate({"train": 50, "dev": 10, "test": 10})
    b = Generator(cfg, seed=3).generate({"train": 50, "dev": 10, "test": 10})
    assert a == b


def test_sizes_and_all_intents_present(data, cfg):
    for split, n in SIZES.items():
        assert len(data[split]) == n
        assert {r["intent"] for r in data[split]} == set(cfg.templates)


def test_labels_valid(data, cfg):
    for rows in data.values():
        for r in rows:
            validate_label(cfg.schema, r["intent"], r["slots"])
            if r["intent"] == "none":
                assert r["slots"] == {} and r["spans"] == []
                assert r["meta"]["reason"]


def test_spans_point_at_slot_surfaces(data):
    for rows in data.values():
        for r in rows:
            assert {s["slot"] for s in r["spans"]} == set(r["slots"])
            for s in r["spans"]:
                piece = r["text"][s["start"]:s["end"]]
                assert piece and piece == piece.strip()
                closed_app = r["intent"] == "call_contact" and s["slot"] == "app"
                if r["script"] == "roman" and s["slot"] in OPEN_SLOTS and not closed_app:
                    assert piece.lower() == str(s["value"])


def test_asr_text_is_clean_roman(data):
    for rows in data.values():
        for r in rows:
            assert not DEVANAGARI.search(r["asr_text"]), r["asr_text"]
            assert r["asr_text"] == r["asr_text"].lower()
            assert not re.search(r"[?.!।]$", r["asr_text"])


def test_open_slot_values_unseen_across_splits(data, cfg):
    def values(split):
        out = set()
        for r in data[split]:
            for k, v in r["slots"].items():
                if k in OPEN_SLOTS and not (r["intent"] == "call_contact" and k == "app"):
                    out.add(v)
        return out

    train, dev, test = values("train"), values("dev"), values("test")
    assert not (train & test), train & test
    assert not (train & dev), train & dev
    assert not (dev & test), dev & test
    assert test  # non-empty


def test_unlabeled_fillers_respect_split(data):
    n = 0
    for split, rows in data.items():
        for r in rows:
            for v in r["meta"]["fillers"]:
                assert value_split(v) == split, (v, split)
                n += 1
    assert n > 0


def test_no_duplicate_texts_within_split(data):
    for rows in data.values():
        keys = [r["asr_text"] for r in rows]
        assert len(keys) == len(set(keys))


def test_heldout_templates_only_in_test(data):
    for split in ("train", "dev"):
        assert not any(r["meta"]["template_heldout"] for r in data[split])
    assert any(r["meta"]["template_heldout"] for r in data["test"])


def test_rows_with_open_values_never_seen_in_train(data):
    for r in data["test"]:
        if r["spans"] and any(s["slot"] in ("folder", "path", "name") for s in r["spans"]):
            assert not r["meta"]["seen_in_train"]


def test_scripts_cover_all_three(data):
    scripts = {r["script"] for r in data["train"]}
    assert scripts == {"roman", "deva", "mixed"}
    deva = [r for r in data["train"] if r["script"] == "deva"]
    # a "deva" row can be all-Latin if none of its words has a Devanagari form (e.g. a bare app name)
    assert sum(bool(DEVANAGARI.search(r["text"])) for r in deva) / len(deva) > 0.95


def test_views_share_labels_and_spans_are_valid(data):
    for r in data["test"]:
        assert len(r["views"]) == 2
        for v in r["views"]:
            assert {s["slot"] for s in v["spans"]} == set(r["slots"])
            for s in v["spans"]:
                assert v["text"][s["start"]:s["end"]].strip()


@pytest.mark.parametrize(
    "intent,slots",
    [
        ("open_app", {}),
        ("set_volume", {}),
        ("set_volume", {"level": 10, "direction": "up"}),
        ("set_volume", {"level": 150}),
        ("check_status", {"target": "weather"}),
        ("none", {"app": "chrome"}),
        ("fly_to_moon", {}),
    ],
)
def test_schema_rejects_bad_labels(cfg, intent, slots):
    with pytest.raises(LabelError):
        validate_label(cfg.schema, intent, slots)
