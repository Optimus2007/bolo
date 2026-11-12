"""Myna's personality, routines and the local app: replies, matching, privacy/safety checks. Nothing is executed."""
import json
import random
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from bolo.actions import plan
from bolo.persona import hinglish_status, mascot, outcome, reply, speak_step


@pytest.fixture(autouse=True)
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("BOLO_SETTINGS", str(tmp_path / "settings.json"))


def test_replies_for_every_command():
    from test_actions import EXAMPLES

    for intent, slots in EXAMPLES:
        p = plan({"intent": intent, "slots": slots}, None, "windows")
        for done in (False, True):
            mood, text = reply(p, done=done, rng=random.Random(0))
            assert text and mood in ("happy", "thinking", "oops") and "None" not in text, (intent, text)
    assert reply(None, heard={"reason": "no speech heard"})[0] == "sleepy"


def test_replies_avoid_gendered_verbs():
    bad = (" karungi", " karunga", " doongi", " doonga", " gayi hoon", " gaya hoon", " rahi hoon", " raha hoon")
    from test_actions import EXAMPLES

    for intent, slots in EXAMPLES:
        for done in (False, True):
            t = reply(plan({"intent": intent, "slots": slots}, None, "mac"), done=done)[1]
            assert not any(b in t.lower() for b in bad), t


def test_status_in_hinglish_and_failures_are_honest():
    assert hinglish_status("battery 80%") == "Battery 80% hai."
    assert hinglish_status("53 GB free of 245 GB").startswith("53 GB khaali")
    p = plan({"intent": "open_app", "slots": {"app": "chrome"}}, None, "mac")
    assert outcome(p, ["that did not work (boom)"])[0] == "oops"


def test_mascot_plain_has_name_and_no_escape_codes():
    s = mascot("happy", "Ho gaya!", "Jarvis", color=False)
    assert "Jarvis: Ho gaya!" in s and "\033" not in s


def test_speaking_never_puts_text_in_code():
    evil = 'x"; rm -rf ~ $(whoami)'
    for os_name in ("mac", "windows"):
        st = speak_step(evil, os_name, voices=[])
        assert evil not in " ".join(st["argv"])
        assert st.get("stdin") == evil or st.get("env", {}).get("BOLO_SAY") == evil


def test_routines_teach_match_forget():
    from bolo.routines import forget, get_routines, match, teach

    steps = [{"text": "chrome kholo", "intent": "open_app", "slots": {"app": "chrome"}},
             {"text": "dnd on karo", "intent": "toggle_setting", "slots": {"setting": "do_not_disturb", "state": "on"}}]
    assert teach("Coding Mode!", steps) == "coding mode"
    assert match("coding mode on karo") == "coding mode"
    assert match("coding mode chalu kar do") == "coding mode"
    assert match("koding mod shuru karo") == "coding mode"
    assert match("coding folder kholo") is None and match("mode on karo") is None
    assert match("chrome kholo") is None
    with pytest.raises(ValueError):
        teach("bad", [{"text": "blah", "intent": "none", "slots": {}}])
    assert forget("coding mode") and get_routines() == {}


class FakeBolo:
    name = "Myna"

    def _strip(self, t):
        return t, False

    def understand(self, text, env=None):
        if "battery" in text:
            return {"intent": "check_status", "slots": {"target": "battery"}, "transcript": text}
        if "band" in text:
            return {"intent": "close_app", "slots": {"app": "chrome"}, "transcript": text}
        if "timer" in text:
            return {"intent": "set_timer", "slots": {"amount": 1, "unit": "seconds"}, "transcript": text}
        return {"intent": "none", "slots": {}, "transcript": text, "reason": "not a supported command"}

    def listen(self, path):
        return self.understand("battery")


def test_app_preview_confirm_and_timers():
    from bolo.app.server import Myna

    ran = []
    fake = lambda p, **k: ran.append(p["intent"]) or ["battery 50%"]  # noqa: E731
    m = Myna(FakeBolo(), None, "mac", allow_run=False, perform=fake)
    r = m.text("battery kitni hai")
    assert r["plan_id"] and m.do(r["plan_id"])["mood"] == "thinking" and not ran  # preview mode never runs
    m.allow_run = True
    r = m.text("chrome band karo")
    assert r["confirm"] and m.do(r["plan_id"]).get("need_confirm") and not ran
    assert m.do(r["plan_id"], confirmed=True) and ran == ["close_app"]
    assert m.do(r["plan_id"])["mood"] == "unsure"  # a plan runs once
    assert m.text("kuch bhi")["mood"] == "unsure"
    done = threading.Event()
    m.schedule(0, "Timer done")
    for _ in range(50):
        if m.state()["events"]:
            done.set()
            break
        threading.Event().wait(0.02)
    assert done.is_set() and m.state(since=1)["events"] == []
    with pytest.raises(ValueError):
        m.audio(b"not a wav")


def test_app_http_refuses_other_sites():
    from bolo.app.server import Myna, make_handler

    m = Myna(FakeBolo(), None, "mac")
    port = [0]
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(m, "tok123", port))
    port[0] = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port[0]}/s/tok123/"

    def call(path, headers=None, data=None):
        req = urllib.request.Request(base + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, b""

    try:
        st, body = call("")
        assert st == 200 and b"tok123" in body
        assert call("api/text", data=b'{"text":"battery"}')[0] == 403
        assert call("api/text", {"X-Myna-Token": "tok123", "Origin": "https://evil.example"}, b'{"text":"battery"}')[0] == 403
        assert call("api/text", {"X-Myna-Token": "tok123", "Host": "evil.example"}, b'{"text":"battery"}')[0] == 403
        st, body = call("api/text", {"X-Myna-Token": "tok123"}, b'{"text":"battery"}')
        assert st == 200 and json.loads(body)["intent"] == "check_status"
        assert urllib.request.urlopen(f"http://127.0.0.1:{port[0]}/s/tok123/").status == 200
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"http://127.0.0.1:{port[0]}/s/wrong/")
    finally:
        srv.shutdown()


def test_help_question():
    from bolo.persona import is_help

    for t in ("kya kar sakte ho", "tum kya kya kar sakte ho?", "help", "Myna kya kar sakti ho", "madad"):
        assert is_help(t), t
    for t in ("chrome kholo", "battery kitni hai", "helpline number search karo"):
        assert not is_help(t), t
