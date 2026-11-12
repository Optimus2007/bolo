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
from bolo.record.server import TUNNEL_RE, AuthError, Store, make_handler, validate_wav

TEST = Path("data/text/v1/test.jsonl")
pytestmark = pytest.mark.skipif(not TEST.exists(), reason="run `bolo generate` first")
CONSENT = {"record": True, "project_use": True}
N_TYPES = len({json.loads(line)["intent"] for line in open(TEST)}) if TEST.exists() else 0  # 20 commands + none


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
    assert len({p["intent"] for p in ps if p["kind"] == "free"}) == N_TYPES
    assert all(p["asr_text"] for p in ps if p["kind"] == "read")
    assert describe_goal("call_contact", {"name": "ankita", "app": "whatsapp"}) == "Call ankita on whatsapp"


def test_validate_wav():
    assert abs(validate_wav(wav(1.0)) - 1.0) < 1e-6
    for bad in (wav(1.0, sr=44100), wav(1.0, ch=2), wav(0.1), wav(20), b"junk"):
        with pytest.raises(ValueError):
            validate_wav(bad)


def test_store_consent_keys_save_withdraw(tmp_path):
    st = Store(tmp_path, TEST)
    with pytest.raises(ValueError):
        st.add_speaker("x", {"record": True}, {})
    sid, key = st.add_speaker("Speaker 07!", CONSENT, {"region": "Bihar", "voice": "female", "email": "no"})
    assert sid == "speaker-07"
    sp = st.speaker(sid)
    assert sp["consent"]["public_release"] is False and sp["consent"]["evaluation_only"] is True
    assert "email" not in sp["meta"] and sp["meta"]["voice"] == "female" and key not in json.dumps(sp)
    # same pseudonym from someone else -> new id, never a takeover
    sid2, key2 = st.add_speaker("speaker 07", CONSENT, {})
    assert sid2 == "speaker-07-2" and key2 != key
    p = st.prompts_for(sid)[0]
    with pytest.raises(ValueError):
        st.save("ghost", key, p["prompt_id"], wav())
    with pytest.raises(AuthError):
        st.save(sid, key2, p["prompt_id"], wav())  # wrong person's key
    e = st.save(sid, key, p["prompt_id"], wav(), transcript="Kuch Bhi", mic={"input_rate": 48000, "junk": 1})
    assert (tmp_path / e["audio"]).exists() and e["consent"]["project_use"] and "junk" not in e["mic"]
    st.save(sid, key, p["prompt_id"], wav(2.0))  # re-record replaces
    assert len(st.manifest()) == 1 and st.manifest()[0]["duration_s"] == 2.0
    with pytest.raises(AuthError):
        st.withdraw(sid, key2)  # cannot delete someone else's recordings
    assert st.withdraw(sid, key) == 1
    assert st.manifest() == [] and st.speaker(sid) is None and not (tmp_path / "audio" / sid).exists()
    assert st.withdraw(sid2, owner=True) == 0  # owner (CLI) can always delete


def _serve(st, invite=None):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(st, invite))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _req(base, path, data=None, headers=None):
    method = "POST" if data is not None else "GET"
    r = urllib.request.Request(base + path, data=data, method=method, headers=headers or {})
    return json.loads(urllib.request.urlopen(r).read())


def test_http_roundtrip_local(tmp_path):
    st = Store(tmp_path, TEST)
    httpd, base = _serve(st)
    try:
        assert b"Bolo Recorder" in urllib.request.urlopen(base + "/").read()
        assert _req(base, "/api/session")["shared"] is False
        with pytest.raises(urllib.error.HTTPError):
            _req(base, "/api/speaker", json.dumps({"pseudonym": "a", "consent": {}}).encode())
        j = _req(base, "/api/speaker", json.dumps({"pseudonym": "a", "consent": CONSENT}).encode())
        sid, H = j["speaker_id"], {"X-Speaker-Key": j["speaker_key"]}
        pid = st.prompts_for(sid)[1]["prompt_id"]
        assert _req(base, f"/api/upload?speaker={sid}&prompt_id={pid}", wav(), H)["ok"]
        assert _req(base, f"/api/prompts?speaker={sid}", headers=H)["done"] == [pid]
        assert _req(base, f"/api/prompts?speaker={sid}")["done"] == []  # no key -> nothing revealed
        with pytest.raises(urllib.error.HTTPError) as e:
            _req(base, f"/api/upload?speaker={sid}&prompt_id={pid}", wav(), {"X-Speaker-Key": "wrong"})
        assert e.value.code == 403
        assert _req(base, "/api/withdraw", json.dumps({"speaker_id": sid}).encode(), H)["removed"] == 1
    finally:
        httpd.shutdown()


def test_share_mode_requires_invite(tmp_path):
    st = Store(tmp_path, TEST)
    httpd, base = _serve(st, invite="s3cret")
    try:
        for path in ("/", "/api/session", "/?invite=wrong"):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(base + path)
            assert e.value.code == 403
        with pytest.raises(urllib.error.HTTPError):
            _req(base, "/api/speaker", json.dumps({"pseudonym": "a", "consent": CONSENT}).encode())
        assert b"Bolo Recorder" in urllib.request.urlopen(base + "/?invite=s3cret").read()
        H = {"X-Invite": "s3cret"}
        assert _req(base, "/api/session", headers=H)["shared"] is True
        j = _req(base, "/api/speaker", json.dumps({"pseudonym": "a", "consent": CONSENT}).encode(), H)
        H2 = {**H, "X-Speaker-Key": j["speaker_key"]}
        pid = st.prompts_for(j["speaker_id"])[0]["prompt_id"]
        assert _req(base, f"/api/upload?speaker={j['speaker_id']}&prompt_id={pid}", wav(), H2)["ok"]
        assert st.manifest()[0]["via"] == "link"
    finally:
        httpd.shutdown()


def test_tunnel_url_regex():
    line = "2026-11-11 INF |  https://quiet-river-1234.trycloudflare.com                  |"
    assert TUNNEL_RE.search(line).group(0) == "https://quiet-river-1234.trycloudflare.com"


def test_human_recordings_cannot_be_used_for_training(tmp_path):
    from bolo.asr import whisper_ft as W

    st = Store(tmp_path / "recordings", TEST)
    sid, key = st.add_speaker("friend", CONSENT, {})
    st.save(sid, key, st.prompts_for(sid)[0]["prompt_id"], wav())
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


def test_short_balanced_assignment_covers_everything(tmp_path):
    st = Store(tmp_path, TEST, n_read=40, n_free=15)
    ids, keys = [], []
    for name in ("a", "b", "c"):
        sid, k = st.add_speaker(name, CONSENT, {})
        ids.append(sid); keys.append(k)
    sets = [st.prompts_for(s) for s in ids]
    for ps in sets:
        assert sum(p["kind"] == "read" for p in ps) == 40 and sum(p["kind"] == "free" for p in ps) == 15
        assert len({p["intent"] for p in ps if p["kind"] == "read"}) == N_TYPES  # every command type for every speaker
    reads = [{p["prompt_id"] for p in ps if p["kind"] == "read"} for ps in sets]
    assert len(reads[0] | reads[1]) >= 76  # two speakers cover almost all 80 read prompts
    assert len(reads[0] | reads[1] | reads[2]) == 80  # three cover all of them
    frees = set().union(*({p["prompt_id"] for p in ps if p["kind"] == "free"} for ps in sets))
    assert len(frees) >= 40  # three speakers cover all 40 bonus prompts
    other = next(p for p in st.prompts if p["prompt_id"] not in {q["prompt_id"] for q in sets[0]})
    with pytest.raises(ValueError):
        st.save(ids[0], keys[0], other["prompt_id"], wav())  # only your own prompts
