import io
import json
import threading
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from bolo.record.prompts import build_prompts, describe_goal
from bolo.record.server import Store, make_handler, validate_wav

TEST = Path("data/text/v0/test.jsonl")
pytestmark = pytest.mark.skipif(not TEST.exists(), reason="run `bolo generate` first")


def wav(seconds=1.0, sr=16000, ch=1):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(ch); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(seconds * sr) * ch)
    return b.getvalue()


def test_prompts_from_test_only_and_stratified():
    ps = build_prompts(TEST)
    test_ids = {json.loads(line)["id"] for line in open(TEST)}
    assert len(ps) == 120 and all(p["source_id"] in test_ids for p in ps)
    assert len({p["source_id"] for p in ps}) == 120
    assert len({p["intent"] for p in ps if p["kind"] == "free"}) == 9
    assert all(p["asr_text"] for p in ps if p["kind"] == "read")
    assert describe_goal("call_contact", {"name": "ankita", "app": "whatsapp"}) == "Call ankita on whatsapp"


def test_validate_wav():
    assert abs(validate_wav(wav(1.0)) - 1.0) < 1e-6
    for bad in (wav(1.0, sr=44100), wav(1.0, ch=2), wav(0.1), wav(20), b"junk"):
        with pytest.raises(ValueError):
            validate_wav(bad)


def test_store_consent_save_withdraw(tmp_path):
    st = Store(tmp_path, TEST)
    with pytest.raises(ValueError):
        st.add_speaker("x", {"record": True}, {})
    sid = st.add_speaker("Speaker 07!", {"record": True, "project_use": True}, {"region": "Bihar", "email": "no"})
    assert sid == "speaker-07"
    sp = st.speaker(sid)
    assert sp["consent"]["public_release"] is False and "email" not in sp["meta"]
    p = st.prompts[0]
    with pytest.raises(ValueError):
        st.save("ghost", p["prompt_id"], wav())
    e = st.save(sid, p["prompt_id"], wav(), transcript="Kuch Bhi")
    assert (tmp_path / e["audio"]).exists() and e["consent"]["project_use"]
    st.save(sid, p["prompt_id"], wav(2.0))  # re-record replaces
    assert len(st.manifest()) == 1 and st.manifest()[0]["duration_s"] == 2.0
    assert st.withdraw(sid) == 1
    assert st.manifest() == [] and st.speaker(sid) is None and not (tmp_path / "audio" / sid).exists()


def test_http_roundtrip(tmp_path):
    st = Store(tmp_path, TEST)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(st))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def post(path, data):
        return json.loads(urllib.request.urlopen(urllib.request.Request(base + path, data=data, method="POST")).read())

    try:
        assert b"Bolo Recorder" in urllib.request.urlopen(base + "/").read()
        with pytest.raises(urllib.error.HTTPError):
            post("/api/speaker", json.dumps({"pseudonym": "a", "consent": {}}).encode())
        sid = post("/api/speaker", json.dumps({"pseudonym": "a", "consent": {"record": 1, "project_use": 1}}).encode())["speaker_id"]
        pid = st.prompts[1]["prompt_id"]
        assert post(f"/api/upload?speaker={sid}&prompt_id={pid}", wav())["ok"]
        done = json.loads(urllib.request.urlopen(f"{base}/api/prompts?speaker={sid}").read())["done"]
        assert done == [pid]
        assert post("/api/withdraw", json.dumps({"speaker_id": sid}).encode())["removed"] == 1
    finally:
        httpd.shutdown()


def test_human_recordings_cannot_be_used_for_training(tmp_path):
    from bolo.asr import whisper_ft as W

    st = Store(tmp_path / "recordings", TEST)
    sid = st.add_speaker("friend", {"record": True, "project_use": True}, {})
    st.save(sid, st.prompts[0]["prompt_id"], wav())
    man = tmp_path / "recordings" / "manifest.jsonl"
    assert len(W.load_manifest(man)) == 1  # evaluation use is fine
    with pytest.raises(W.HumanAudioError):
        W.load_manifest(man, for_training=True)
    other = tmp_path / "copy.jsonl"
    other.write_text(man.read_text())  # copied elsewhere: still caught by its consent field
    with pytest.raises(W.HumanAudioError):
        W.load_manifest(other, for_training=True)
    rows = W.load_manifest(man)
    with pytest.raises(W.HumanAudioError):
        W.train(None, None, rows, [], tmp_path / "out")
