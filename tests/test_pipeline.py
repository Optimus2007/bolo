from pathlib import Path

import pytest

from bolo.grounding import Environment
from bolo.pipeline import scan_mac

MODEL = Path("models/bolo-tiny/phono_views_aug-s0")


def test_scan_is_read_only_and_finds_things(tmp_path):
    (tmp_path / "Rust_Learning").mkdir()
    (tmp_path / "Rust_Learning" / "notes.txt").write_text("x")
    (tmp_path / ".secret").mkdir()
    (tmp_path / "Library").mkdir()
    before = sorted(p.as_posix() for p in tmp_path.rglob("*"))
    env = scan_mac(tmp_path)
    assert "rust learning" in env.folders and "notes" in env.files
    assert ".secret" not in env.folders and "library" not in env.folders
    assert sorted(p.as_posix() for p in tmp_path.rglob("*")) == before  # nothing created/changed


@pytest.mark.skipif(not MODEL.exists(), reason="train bolo-tiny first")
def test_understand_grounds_or_abstains():
    from bolo.pipeline import Bolo

    b = Bolo.load()
    r = b.understand("rust learning folder mein terminal kholo")
    assert r["intent"] == "open_terminal" and r["slots"].get("folder") and not r["abstain"]
    env = Environment(folders=["rust learning", "photos"], files=[], apps=["vlc"], contacts=[])
    g = b.understand("rust lerning folder mein terminal kholo", env)
    assert g["slots"]["folder"] == "rust learning"  # snapped to what exists
    missing = b.understand("kaggle folder mein terminal kholo", env)
    assert missing["abstain"] and "kaggle" in missing["reason"]
    assert b.understand("aaj mausam kaisa hai")["abstain"]
