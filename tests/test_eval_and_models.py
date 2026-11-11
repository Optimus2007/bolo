import pytest

from bolo.baselines.rules import RuleBaseline
from bolo.config import load_config
from bolo.evaluate import choose_threshold, consistency, finalize, score
from bolo.generator import Generator
from bolo.models.tagging import NumberParser, decode, label_names, word_tags
from bolo.normalize import Normalizer


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def rows(cfg):
    return Generator(cfg, seed=11).generate({"train": 10, "dev": 10, "test": 300})["test"]


def test_perfect_predictions_score_perfectly(rows):
    preds = [{"intent": r["intent"], "slots": r["slots"], "confidence": 1.0} for r in rows]
    s = score(rows, preds)
    assert s["exact_match"] == 1.0 and s["intent_acc"] == 1.0 and s["slot"]["f1"] == 1.0
    assert s["false_action_rate"] == 0.0 and s["abstain"]["f1"] == 1.0


def test_always_none_has_no_false_actions_but_low_em(rows):
    preds = [{"intent": "none", "slots": {}, "confidence": 1.0} for _ in rows]
    s = score(rows, preds)
    assert s["false_action_rate"] == 0.0
    assert s["over_abstain_rate"] == 1.0
    assert s["exact_match"] < 0.3


def test_finalize_abstains_on_invalid_and_low_confidence(cfg):
    assert finalize({"intent": "open_app", "slots": {}, "confidence": 0.99}, cfg.schema)["intent"] == "none"
    assert finalize({"intent": "open_app", "slots": {"app": "chrome"}, "confidence": 0.3}, cfg.schema, 0.5)["intent"] == "none"
    ok = finalize({"intent": "open_app", "slots": {"app": "chrome"}, "confidence": 0.9}, cfg.schema, 0.5)
    assert ok["intent"] == "open_app"


def test_lenient_slot_match_accepts_spelling_variants(rows):
    r = {"intent": "call_contact", "slots": {"name": "sakshi"}, "meta": {}, "script": "deva"}
    p = {"intent": "call_contact", "slots": {"name": "sakshee"}, "confidence": 1.0}
    s = score([r], [p])
    assert s["exact_match"] == 0.0 and s["exact_match_lenient"] == 1.0


def test_consistency_metric():
    p = {"intent": "open_app", "slots": {"app": "zoom"}}
    q = {"intent": "open_app", "slots": {"app": "jum"}}  # same phonetic key -> consistent
    assert consistency([{}], [p], [[q, p]]) == 1.0
    assert consistency([{}], [p], [[{"intent": "none", "slots": {}}]]) == 0.0


def test_threshold_choice_is_in_range(cfg, rows):
    preds = [{"intent": r["intent"], "slots": r["slots"], "confidence": 0.9} for r in rows]
    assert 0.0 <= choose_threshold(rows, preds, cfg.schema) <= 0.9


def test_tagging_roundtrip_recovers_gold_slots(cfg, rows):
    """Encoding gold spans as word tags and decoding them must give back the gold slots
    (strictly for Roman rows; Devanagari values are transliterated, so compare leniently)."""
    norm = Normalizer.from_config(cfg)
    nums = NumberParser(cfg)
    labels = set(label_names(cfg))
    from bolo.evaluate import _pairs

    for r in rows:
        words, tags = word_tags(norm, r["text"], r["spans"], r["intent"])
        assert set(tags) <= labels
        if r["intent"] == "none":
            assert set(tags) == {"O"} or not tags
            continue
        slots = decode(cfg, norm, nums, r["intent"], words, tags)
        if r["script"] == "roman":
            assert _pairs(slots, False) == _pairs(r["slots"], False), (r["text"], slots, r["slots"])
        else:
            assert set(slots) == set(r["slots"]), (r["text"], slots, r["slots"])


def test_rule_baseline_examples(cfg):
    rb = RuleBaseline(cfg)
    p = rb.predict_one("bolo folder mein terminal kholo")
    assert p["intent"] == "open_terminal" and p["slots"] == {"folder": "bolo"}
    p = rb.predict_one("राहुल को कॉल करो")
    assert p["intent"] == "call_contact" and p["slots"]["name"] == "rahul"
    assert rb.predict_one("volume pachaas kar do")["slots"] == {"level": 50}
    assert rb.predict_one("terminal band karo")["intent"] == "none"


def test_bolo_tiny_trains_and_exports(cfg, tmp_path):
    torch = pytest.importorskip("torch")
    from bolo.models.tiny import BoloTiny, TinyConfig

    data = Generator(cfg, seed=5).generate({"train": 200, "dev": 50, "test": 50})
    m = BoloTiny(cfg, TinyConfig(buckets=1024, d_model=32, emb_dim=16, ff=64, layers=1))
    m.fit(data["train"], data["dev"], epochs=2, augment=True, use_views=True, log=lambda s: None)
    preds = m.predict([r["text"] for r in data["test"]])
    assert len(preds) == 50 and all(0.0 <= p["confidence"] <= 1.0 for p in preds)
    m.save(tmp_path / "m")
    m2 = BoloTiny.load(cfg, tmp_path / "m")
    assert m2.predict(["bolo folder mein terminal kholo"]) == m.predict(["bolo folder mein terminal kholo"])
    pytest.importorskip("onnxruntime")
    from bolo.models.export import export_onnx

    info = export_onnx(m, tmp_path / "onnx")
    assert info["onnx_int8_mb"] < info["onnx_fp32_mb"]
