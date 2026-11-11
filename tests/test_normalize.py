import pytest

from bolo.config import load_config
from bolo.normalize import Normalizer, phonetic_key, skeleton


@pytest.fixture(scope="module")
def norm():
    return Normalizer.from_config(load_config())


@pytest.mark.parametrize("variants", [
    ["kholo", "kolo", "khollo"],
    ["mein", "me", "mai", "mei", "main"],
    ["hai", "he"],
    ["awaaz", "awaz", "aawaz", "avaaz"],
    ["zyada", "jyada"],
    ["rahul", "raahul"],
    ["nahi", "nahin"],
])
def test_spelling_variants_share_key(variants):
    assert len({phonetic_key(v) for v in variants}) == 1


@pytest.mark.parametrize("a,b", [("karo", "kro"), ("terminal", "tarminal"), ("battery", "betri")])
def test_vowel_dropping_variants_share_skeleton(a, b):
    assert skeleton(a) == skeleton(b)


@pytest.mark.parametrize("dev,roman", [
    ("राहुल", "rahul"), ("विक्रम", "vikram"), ("कमरा", "kamra"), ("करना", "karna"),
    ("बदलो", "badlo"), ("अमित", "amit"), ("गुप्ता", "gupta"),
])
def test_rule_transliteration(norm, dev, roman):
    assert norm.to_roman(dev) == roman


def test_known_words_use_dictionary(norm):
    assert norm.to_roman("खोलो") == "kholo"
    assert norm.to_roman("टर्मिनल") == "terminal"
    assert norm.to_roman("में") == "mein"


def test_tokenize_offsets(norm):
    text = "ज़रा bolo में terminal kholo?"
    toks = norm.tokenize(text)
    assert [t for t, _, _ in toks] == ["ज़रा", "bolo", "में", "terminal", "kholo", "?"]
    for t, s, e in toks:
        assert text[s:e] == t
    assert norm.roman_words(text) == ["zara", "bolo", "mein", "terminal", "kholo"]
