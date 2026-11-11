import json
from pathlib import Path

import numpy as np
import pytest

from bolo.audio.augment import write_wav
from bolo.config import load_config
from bolo.speech_eval import load_set, run, to_markdown

TEST = Path("data/text/v0/test.jsonl")
pytestmark = pytest.mark.skipif(not TEST.exists(), reason="run `bolo generate` first")


def test_eval_with_stub_asr(tmp_path):
    gold = [json.loads(line) for line in open(TEST)][:12]
    (tmp_path / "audio").mkdir()
    with open(tmp_path / "manifest.jsonl", "w") as f:
        for i, g in enumerate(gold):
            write_wav(tmp_path / f"audio/{i}.wav", np.zeros(8000, np.float32))
            f.write(json.dumps({"utt_id": f"u{i}", "audio": f"audio/{i}.wav", "speaker_id": "me", "kind": "read",
                                "source_id": g["id"], "prompt_id": f"p{i}", "intent": g["intent"], "slots": g["slots"],
                                "transcript": g["asr_text"], "duration_s": 0.5}) + "\n")
    rows = load_set(tmp_path / "manifest.jsonl", TEST)
    assert {r["set"] for r in rows} == {"human_read"} and rows[0]["ref"] == gold[0]["asr_text"]
    refs = {r["_wav"]: r["ref"] for r in rows}
    order = iter([r["ref"] for r in rows[:1]] + [r["ref"] for r in rows])  # 1 warm-up call

    class Perfect:
        def __call__(self, wav):
            return next(order)

    res = run({"human_read": rows}, ["perfect"], ["rules"], load_config(), utt_out=tmp_path / "u.jsonl",
              asr_factory=lambda name: Perfect())
    S = res["sets"]["human_read"]
    assert S["asr"]["perfect"]["wer"]["wer"] == 0
    # perfect ASR must equal the oracle-transcript result
    assert S["asr"]["perfect"]["nlu"]["rules"]["grounded"] == S["oracle"]["rules"]["grounded"]
    assert "human_read" in to_markdown(res) and (tmp_path / "u.jsonl").exists()
    assert refs
