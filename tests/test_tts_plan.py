import json
from pathlib import Path

import numpy as np
import pytest

from bolo.audio import tts

TEST = Path("data/text/v0/test.jsonl")


@pytest.mark.skipif(not TEST.exists(), reason="run `bolo generate` first")
def test_plan_speaker_split_and_text_choice():
    rows = [json.loads(line) for line in open(TEST)][:300]
    test_jobs = tts.plan(rows, "test")
    train_jobs = tts.plan(rows, "train")
    assert {j["speaker"] for j in test_jobs} <= set(tts.HI_TEST + tts.EN_TEST)
    assert not {j["speaker"] for j in train_jobs} & set(tts.HI_TEST + tts.EN_TEST)
    for j, r in zip(test_jobs, rows):
        assert j["speaker"] in j["description"]
        assert j["tts_input"] == (r["asr_text"] if j["voice_lang"] == "en" else r["tts_text"])
        if j["voice_lang"] == "en":
            assert (r["meta"].get("en_ratio") or 0) >= 0.5
    assert tts.plan(rows, "test") == test_jobs  # deterministic


def test_trim_and_plausible():
    sr = 16000
    x = np.zeros(sr * 3, dtype=np.float32)
    x[sr: 2 * sr] = 0.5 * np.sin(np.arange(sr) * 0.1)
    y = tts.trim_silence(x)
    assert 1.0 < len(y) / sr < 1.5
    assert tts.plausible(y, "terminal kholo")[0]
    assert tts.plausible(np.zeros(sr * 11, np.float32) + 0.1, "hi")[1] == "too_long"
    assert tts.plausible(np.zeros(100, np.float32), "terminal kholo")[1] == "too_short"
    assert tts.plausible(np.full(int(0.9 * sr), 0.1, np.float32), "vlc")[0]  # short word, natural padding
