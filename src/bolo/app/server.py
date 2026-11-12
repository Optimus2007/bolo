"""`bolo app`: Myna on your screen. A local page with the mascot, push-to-talk, replies, routines and timers.

    bolo app                 # opens http://127.0.0.1:<port>/s/<token>/ in your browser; preview mode
    bolo app --run           # allowed to really do things (still asks before closing apps / sleeping)

Private by design:
- binds to 127.0.0.1 only; nothing is reachable from the network
- a random token is in the page address and must come back with every request, and the Host header must be
  127.0.0.1/localhost, so another website open in the browser cannot send commands to Myna
- audio is held in memory for one request, written to a temp file only for the recogniser, and deleted at once;
  nothing is logged to disk
Timers and reminders run here in the background while the app is open.
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import urlparse

MAX_WAV = 1_200_000  # ~35 s of 16 kHz mono 16-bit; the page sends at most 15 s
MAX_TIMERS = 20


class Myna:
    """Everything the page can ask for, with no HTTP in it (tests drive this directly)."""

    def __init__(self, bolo, env=None, os_name: str | None = None, allow_run: bool = False, perform=None):
        from ..actions import current_os

        self.bolo, self.env = bolo, env
        self.os = os_name or current_os()
        self.allow_run = allow_run
        self._perform = perform
        self.lock = threading.Lock()
        self.plans: dict[str, dict] = {}
        self.events: list[dict] = []
        self.timers: dict[str, dict] = {}

    # ---- understanding
    def heard(self, out: dict) -> dict:
        from ..actions import describe, plan
        from ..persona import reply
        from ..routines import get_routines, match

        base = {"heard": out.get("transcript", ""), "intent": out["intent"], "slots": out["slots"],
                "confidence": out.get("confidence"), "ms": out.get("ms"), "can_run": self.allow_run}
        name = match(self.bolo._strip(out["transcript"])[0]) if out.get("transcript") else None
        if name:
            steps = get_routines()[name]
            plans = []
            for st in steps:
                o = self.bolo.understand(st["text"])
                plans.append(plan(o, self.env, self.os) if o["intent"] != "none" else
                             {"ok": False, "say": f"skipped {st['text']!r}", "steps": [], "intent": "none"})
            pid = self._keep({"routine": name, "plans": plans})
            return base | {"routine": name, "mood": "happy", "plan_id": pid, "confirm": any(p.get("confirm") for p in plans),
                           "reply": f"{name.title()} routine: {len(steps)} kaam" + (
                               "." if all(p.get("ok", True) for p in plans) else
                               f", par is computer pe {sum(1 for p in plans if p.get('ok', True))} hi ho sakte hain."),
                           "steps": [describe(s) for p in plans for s in p["steps"]],
                           "notes": [p["say"] for p in plans if not p.get("ok", True)]}
        from ..persona import HELP, is_help

        if out["intent"] == "none" and is_help(self.bolo._strip(out.get("transcript", ""))[0]):
            return base | {"mood": "happy", "reply": "Bahut kuch! Ye try karo:", "help": HELP, "steps": []}
        if out["intent"] == "none":
            mood, text = reply(None, heard=out)
            return base | {"mood": mood, "reply": text, "steps": [], "reason": out.get("reason")}
        p = plan(out, self.env, self.os)
        mood, text = reply(p, done=False)
        if p.get("ok", True):
            text = text.replace(" --run lagao toh kar doon.", "")
        return base | {"mood": mood, "reply": text, "plan_id": self._keep({"plans": [p]}) if p.get("ok") else None,
                       "confirm": bool(p.get("confirm")), "steps": [describe(s) for s in p["steps"]],
                       "notes": [p["partial"]] if p.get("partial") else []}

    def text(self, s: str) -> dict:
        s = " ".join(str(s).split())[:300]
        with self.lock:
            return self.heard(self.bolo.understand(s))

    def audio(self, wav: bytes) -> dict:
        if not wav or len(wav) > MAX_WAV or wav[:4] != b"RIFF":
            raise ValueError("send a 16-bit WAV of at most 15 seconds")
        fd, path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(wav)
            with self.lock:
                return self.heard(self.bolo.listen(path))
        finally:
            os.unlink(path)  # the recording is gone as soon as it is understood

    def _keep(self, item: dict) -> str:
        pid = secrets.token_hex(6)
        self.plans[pid] = item | {"at": time.time()}
        for k in [k for k, v in self.plans.items() if time.time() - v["at"] > 600]:  # forget after 10 min
            self.plans.pop(k, None)
        return pid

    # ---- doing
    def do(self, pid: str, confirmed: bool = False) -> dict:
        from ..actions import perform
        from ..persona import outcome

        if not self.allow_run:
            return {"mood": "thinking", "reply": "Abhi preview mode hai. `bolo app --run` se kholo toh kar doon.", "said": []}
        item = self.plans.pop(pid, None)
        if item is None:
            return {"mood": "unsure", "reply": "Ye purana ho gaya, ek baar phir bolo.", "said": []}
        plans = [p for p in item["plans"] if p.get("ok", True) and p["steps"]]
        if any(p.get("confirm") for p in plans) and not confirmed:
            self.plans[pid] = item
            return {"mood": "unsure", "reply": "Pakka? Confirm karo.", "need_confirm": True, "said": []}
        run = self._perform or perform
        said, results = [], []
        for p in plans:
            r = run(p, execute=True, confirm=lambda q: True, log=lambda *_: None, schedule=self.schedule)
            said += r
            results.append(outcome(p, r))
        oops = [x for x in results if x[0] == "oops"]
        if oops or not results:
            last = oops[0] if oops else ("oops", "Kuch karne ko mila hi nahi.")
        elif item.get("routine"):
            last = ("happy", f"{item['routine'].title()} routine ho gaya!")
        else:
            last = results[-1]
        return {"mood": last[0], "reply": last[1], "said": said}

    # ---- timers (only while the app is open)
    def schedule(self, seconds: int, text: str) -> str:
        if len(self.timers) >= MAX_TIMERS:
            return "too many timers running"
        tid = secrets.token_hex(4)
        due = time.time() + seconds

        def ring():
            self.timers.pop(tid, None)
            msg = "Time ho gaya!" if text == "Timer done" else f"Yaad hai? {text}"
            self.events.append({"n": len(self.events) + 1, "kind": "timer", "mood": "happy", "reply": msg})

        t = threading.Timer(seconds, ring)
        t.daemon = True
        self.timers[tid] = {"due": due, "text": text, "t": t}
        t.start()
        return f"timer set for {seconds} s"

    def cancel(self, tid: str) -> bool:
        t = self.timers.pop(tid, None)
        if t:
            t["t"].cancel()
        return t is not None

    def state(self, since: int = 0) -> dict:
        now = time.time()
        return {"events": [e for e in self.events if e["n"] > since],
                "timers": [{"id": k, "left": max(0, round(v["due"] - now)), "text": v["text"]} for k, v in self.timers.items()],
                "can_run": self.allow_run, "name": self.bolo.name, "os": self.os}


def make_handler(myna: Myna, token: str, port_ref: list):
    page = resources.files("bolo.app").joinpath("page.html").read_text(encoding="utf-8")
    prefix = f"/s/{token}"

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, body: bytes, ctype: str, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; "
                             "style-src 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=HTTPStatus.OK):
            self._send(json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

        def _allowed(self) -> str | None:
            """The path inside the session, or None if this request did not come from the Myna page."""
            host = (self.headers.get("Host") or "").lower()
            if host not in (f"127.0.0.1:{port_ref[0]}", f"localhost:{port_ref[0]}"):
                return None  # DNS rebinding: another site's name pointing at 127.0.0.1
            path = urlparse(self.path).path
            if not path.startswith(prefix + "/"):
                return None
            path = path[len(prefix):]
            if path.startswith("/api/"):
                if not hmac.compare_digest(self.headers.get("X-Myna-Token", ""), token):
                    return None
                origin = self.headers.get("Origin")
                if origin and origin.lower() not in (f"http://127.0.0.1:{port_ref[0]}", f"http://localhost:{port_ref[0]}"):
                    return None
            return path

        def _body(self, limit: int) -> bytes:
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n > limit:
                raise ValueError("too large")
            return self.rfile.read(n)

        def do_GET(self):
            path = self._allowed()
            if path is None:
                return self._send(b"Not here.", "text/plain; charset=utf-8", HTTPStatus.FORBIDDEN)
            if path == "/":
                return self._send(page.replace("__TOKEN__", token).encode(), "text/html; charset=utf-8")
            if path.startswith("/api/state"):
                q = urlparse(self.path).query
                since = int(q.split("since=", 1)[1].split("&")[0]) if "since=" in q else 0
                return self._json(myna.state(since))
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            path = self._allowed()
            if path is None:
                return self._json({"error": "forbidden"}, HTTPStatus.FORBIDDEN)
            try:
                if path == "/api/listen":
                    return self._json(myna.audio(self._body(MAX_WAV)))
                d = json.loads(self._body(20_000) or b"{}")
                if path == "/api/text":
                    return self._json(myna.text(d.get("text", "")))
                if path == "/api/do":
                    return self._json(myna.do(str(d.get("plan_id", "")), bool(d.get("confirmed"))))
                if path == "/api/cancel":
                    return self._json({"ok": myna.cancel(str(d.get("id", "")))})
            except (ValueError, json.JSONDecodeError) as e:
                return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    return H


def serve(asr: str | None, port: int = 0, allow_run: bool = False, open_browser: bool = True):
    from ..actions import current_os, scan_system
    from ..pipeline import Bolo

    print("Myna is waking up (loading the speech model)...")
    env = scan_system()
    bolo = Bolo.load(asr=asr, env=env, prompt="commands+apps")
    myna = Myna(bolo, env, current_os(), allow_run=allow_run)
    token = secrets.token_urlsafe(16)
    port_ref = [port]
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(myna, token, port_ref))
    port_ref[0] = httpd.server_address[1]
    url = f"http://127.0.0.1:{port_ref[0]}/s/{token}/"
    print(f"\n  Myna is here: {url}\n  ({'can really do things; asks before risky ones' if allow_run else 'preview mode: shows what it would do'}"
          f"; local only, Ctrl+C to stop)\n", flush=True)
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nPhir milenge! Myna is off now.")
    finally:
        httpd.server_close()
