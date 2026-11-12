import pytest

from bolo.assistant import DEFAULT_NAME, get_name, set_name, strip_name
from bolo.config import load_config
from bolo.normalize import Normalizer

R = Normalizer.from_config(load_config()).to_roman


@pytest.fixture(autouse=True)
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("BOLO_SETTINGS", str(tmp_path / "settings.json"))


def test_default_set_reset():
    assert get_name() == DEFAULT_NAME == "Myna"
    assert set_name("  Jarvis ") == "Jarvis" and get_name() == "Jarvis"
    assert set_name(None) == "Myna"
    for bad in ("", "x", "a" * 30, "rm -rf /", "one two three four", "R2D2"):
        with pytest.raises(ValueError):
            set_name(bad)


@pytest.mark.parametrize("text,want", [
    ("Myna, volume kam karo", "volume kam karo"),
    ("hey myna terminal kholo", "terminal kholo"),
    ("maina screenshot lo", "screenshot lo"),            # misspelt by the recogniser
    ("मैना वॉल्यूम कम करो", "वॉल्यूम कम करो"),            # Devanagari
    ("myna ji wifi band karo", "wifi band karo"),
    ("volume kam karo, myna", "volume kam karo"),
    ("arre myna, gaana pause karo", "gaana pause karo"),
])
def test_strips_address(text, want):
    assert strip_name(text, "Myna", R) == (want, True)


@pytest.mark.parametrize("text", [
    "myna ko call karo",          # a contact called Myna: the name is the object
    "call myna",
    "volume kam karo",
    "mina se baat karao",
    "myna",                       # just the name: nothing to strip to
])
def test_keeps_non_address(text):
    assert strip_name(text, "Myna", R) == (text, False)


def test_custom_name_vs_contact():
    assert strip_name("Rahul, terminal kholo", "Rahul", R) == ("terminal kholo", True)
    assert strip_name("rahul ko call karo", "Rahul", R) == ("rahul ko call karo", False)
    assert strip_name("hey friday lock screen", "Friday", R) == ("lock screen", True)
    assert strip_name("chotu bhai, volume badhao", "Chotu Bhai", R) == ("volume badhao", True)
