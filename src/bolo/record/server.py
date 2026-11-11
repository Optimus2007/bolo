"""`bolo record`: a consent-first web tool for collecting the human-spoken EVAL set.

Local mode (default): binds to 127.0.0.1; the speaker records on this laptop.
Share mode (`--share`): the same server, still bound to 127.0.0.1, is exposed through a free Cloudflare
quick tunnel (HTTPS, needed for microphone access on phones). Audio is sent over the encrypted link and
stored ONLY on this laptop in `recordings/`; nothing is stored in any cloud.

Protection in share mode:
- an invite code (random, per session) is required on every request; the link carries it
- each speaker gets a private key at consent time; uploads and "withdraw" need that key, so one person cannot
  record as, or delete, someone else
- a pseudonym that is already taken gets a new id instead of overwriting the existing speaker

A speaker picks a pseudonym and gives explicit consent before any recording: record + evaluation-only use are
required; public_release is a separate opt-in (default off). Recordings are EVALUATION-ONLY; training code
refuses them (bolo.asr.whisper_ft.HumanAudioError). Never commit or publish anyone's audio (see docs/AUDIO.md).
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import re
import secrets
import shutil
import subprocess
import threading
import time
import wave
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .prompts import PROMPT_VERSION, build_prompts

TOOL_VERSION = "bolo-record 0.2"
SR = 16000
MIN_S, MAX_S = 0.3, 15.0
MAX_SPEAKERS = 200
SLUG = re.compile(r"[^a-z0-9_-]+")


def slugify(name: str) -> str:
    s = SLUG.sub("-", name.strip().lower()).strip("-")
    return s[:40] or "speaker"


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class AuthError(Exception):
    pass


class Store:
    def __init__(self, root: Path, test_path: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "audio").mkdir(exist_ok=True)
        self.lock = threading.Lock()
        pf = self.root / f"prompts_{PROMPT_VERSION}.jsonl"
        if not pf.exists():
            prompts = build_prompts(test_path)
            pf.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts), encoding="utf-8")
        self.prompts = [json.loads(line) for line in pf.read_text(encoding="utf-8").splitlines()]
        self.by_id = {p["prompt_id"]: p for p in self.prompts}

    # --- speakers ---
    def _speakers(self) -> dict:
        f = self.root / "speakers.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}

    def _save_speakers(self, d: dict):
        (self.root / "speakers.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

    def add_speaker(self, pseudonym: str, consent: dict, meta: dict, via: str = "local") -> tuple[str, str]:
        """Returns (speaker_id, private key). The key is shown to the speaker's browser only once."""
        if not (consent.get("record") and consent.get("project_use")):
            raise ValueError("recording requires consent to record and to evaluation use")
        allowed_meta = {k: str(v)[:60] for k, v in (meta or {}).items()
                        if k in ("voice", "age_range", "region", "first_language", "mic") and v}
        key = secrets.token_urlsafe(18)
        with self.lock:
            sp = self._speakers()
            if len(sp) >= MAX_SPEAKERS:
                raise ValueError("speaker limit reached")
            base = sid = slugify(pseudonym)
            n = 2
            while sid in sp:  # never take over an existing speaker
                sid = f"{base}-{n}"
                n += 1
            sp[sid] = {"pseudonym": pseudonym[:40], "key_sha256": _hash(key), "via": via, "consent": {
                "record": True, "project_use": True, "evaluation_only": True,
                "public_release": bool(consent.get("public_release")),
                "given_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, "meta": allowed_meta}
            self._save_speakers(sp)
        return sid, key

    def speaker(self, sid: str) -> dict | None:
        return self._speakers().get(sid)

    def check_key(self, sid: str, key: str) -> dict:
        sp = self.speaker(sid)
        if sp is None:
            raise ValueError("unknown speaker (give consent first)")
        if not key or not hmac.compare_digest(sp.get("key_sha256", ""), _hash(key)):
            raise AuthError("wrong speaker key")
        return sp

    # --- manifest ---
    def manifest(self) -> list[dict]:
        f = self.root / "manifest.jsonl"
        if not f.exists():
            return []
        return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _write_manifest(self, rows: list[dict]):
        (self.root / "manifest.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    def done(self, sid: str) -> list[str]:
        return [r["prompt_id"] for r in self.manifest() if r["speaker_id"] == sid]

    def save(self, sid: str, key: str, prompt_id: str, wav_bytes: bytes, transcript: str = "",
             mic: dict | None = None) -> dict:
        sp = self.check_key(sid, key)
        p = self.by_id.get(prompt_id)
        if p is None:
            raise ValueError("unknown prompt")
        dur = validate_wav(wav_bytes)
        d = self.root / "audio" / sid
        d.mkdir(parents=True, exist_ok=True)
        rel = f"audio/{sid}/{prompt_id}.wav"
        (self.root / rel).write_bytes(wav_bytes)
        mic = {k: v for k, v in (mic or {}).items() if k in ("echoCancellation", "noiseSuppression",
                                                              "autoGainControl", "input_rate", "mobile")}
        entry = {
            "utt_id": f"{sid}__{prompt_id}", "speaker_id": sid, "prompt_id": prompt_id, "kind": p["kind"],
            "source_id": p["source_id"], "audio": rel, "sample_rate": SR, "duration_s": round(dur, 3),
            "intent": p["intent"], "slots": p["slots"],
            "transcript": p["asr_text"] if p["kind"] == "read" else (transcript.strip().lower()[:300] or None),
            "transcript_source": "prompt" if p["kind"] == "read" else ("speaker_typed" if transcript.strip() else None),
            "consent": sp["consent"], "speaker_meta": sp["meta"], "mic": mic, "via": sp.get("via", "local"),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "tool": TOOL_VERSION,
        }
        with self.lock:
            rows = [r for r in self.manifest() if r["utt_id"] != entry["utt_id"]]  # re-record replaces
            rows.append(entry)
            self._write_manifest(rows)
        return entry

    def withdraw(self, sid: str, key: str | None = None, owner: bool = False) -> int:
        """Delete a speaker's audio + manifest rows + speaker entry. Needs their key unless owner=True (CLI)."""
        if not owner:
            self.check_key(sid, key or "")
        with self.lock:
            rows = self.manifest()
            keep = [r for r in rows if r["speaker_id"] != sid]
            removed = len(rows) - len(keep)
            for r in rows:
                if r["speaker_id"] == sid:
                    (self.root / r["audio"]).unlink(missing_ok=True)
            d = self.root / "audio" / sid
            if d.exists():
                for f in d.iterdir():
                    f.unlink()
                d.rmdir()
            self._write_manifest(keep)
            sp = self._speakers()
            sp.pop(sid, None)
            self._save_speakers(sp)
        return removed


def validate_wav(data: bytes) -> float:
    try:
        with wave.open(io.BytesIO(data)) as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != SR:
                raise ValueError("expected 16 kHz mono 16-bit WAV")
            dur = w.getnframes() / SR
    except (wave.Error, EOFError) as e:
        raise ValueError(f"not a WAV file: {e}") from e
    if not MIN_S <= dur <= MAX_S:
        raise ValueError(f"duration {dur:.2f}s outside [{MIN_S}, {MAX_S}]")
    return dur


def make_handler(store: Store, invite: str | None = None):
    """invite=None -> local mode (no code needed). Otherwise every request must carry the invite code."""
    page = resources.files("bolo.record").joinpath("page.html").read_text(encoding="utf-8")
    shared = invite is not None

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, body: bytes, ctype: str, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=HTTPStatus.OK):
            self._send(json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length", 0))
            if n > 5_000_000:
                raise ValueError("upload too large")
            return self.rfile.read(n)

        def _invite_ok(self, q: dict) -> bool:
            if not shared:
                return True
            got = self.headers.get("X-Invite") or q.get("invite", "")
            return hmac.compare_digest(str(got), invite)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if not self._invite_ok(q):
                return self._send(b"This recording link is invalid or has expired.", "text/plain; charset=utf-8",
                                  HTTPStatus.FORBIDDEN)
            if u.path == "/":
                self._send(page.encode(), "text/html; charset=utf-8")
            elif u.path == "/api/session":
                self._json({"shared": shared, "n_prompts": len(store.prompts)})
            elif u.path == "/api/prompts":
                sid, key = q.get("speaker", ""), self.headers.get("X-Speaker-Key", "")
                try:
                    done = store.done(sid) if sid and store.check_key(sid, key) else []
                except (ValueError, AuthError):
                    done = []
                self._json({"prompts": store.prompts, "done": done})
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if not self._invite_ok(q):
                return self._json({"error": "invalid invite"}, HTTPStatus.FORBIDDEN)
            key = self.headers.get("X-Speaker-Key", "")
            try:
                if u.path == "/api/speaker":
                    d = json.loads(self._body() or b"{}")
                    sid, k = store.add_speaker(d.get("pseudonym", ""), d.get("consent", {}), d.get("meta", {}),
                                               via="link" if shared else "local")
                    self._json({"speaker_id": sid, "speaker_key": k, "done": []})
                elif u.path == "/api/upload":
                    mic = json.loads(q.get("mic", "{}") or "{}")
                    e = store.save(q.get("speaker", ""), key, q.get("prompt_id", ""), self._body(),
                                   q.get("transcript", ""), mic)
                    self._json({"ok": True, "duration_s": e["duration_s"]})
                elif u.path == "/api/withdraw":
                    d = json.loads(self._body() or b"{}")
                    self._json({"removed": store.withdraw(slugify(d.get("speaker_id", "")), key)})
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except AuthError as e:
                self._json({"error": str(e)}, HTTPStatus.FORBIDDEN)
            except (ValueError, json.JSONDecodeError) as e:
                self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)

    return H


TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def start_tunnel(port: int, timeout: float = 45.0) -> tuple[subprocess.Popen, str]:
    exe = shutil.which("cloudflared") or "/opt/homebrew/bin/cloudflared"
    if not Path(exe).exists():
        raise RuntimeError("cloudflared not found: `brew install cloudflared`")
    proc = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0 = time.time()
    for line in proc.stdout:  # cloudflared prints the public URL in its log
        m = TUNNEL_RE.search(line)
        if m:
            threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()  # drain
            return proc, m.group(0)
        if time.time() - t0 > timeout:
            break
    proc.terminate()
    raise RuntimeError("could not start the Cloudflare tunnel (check internet)")


def serve(root: Path | str = "recordings", test_path: Path | str = "data/text/v0/test.jsonl",
          port: int = 8765, open_browser: bool = True, share: bool = False, minutes: int = 120):
    store = Store(Path(root), Path(test_path))
    invite = secrets.token_urlsafe(12) if share else None
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store, invite))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print("Reminder: recordings are EVALUATION-ONLY. Never commit or release anyone's audio without consent. "
          "recordings/ is gitignored.")
    tunnel = None
    try:
        if share:
            tunnel, base = start_tunnel(port)
            link = f"{base}/?invite={invite}"
            print("\n  Share this link (works only while this command runs):\n\n    " + link + "\n")
            print(f"  Audio is stored only in {Path(root).resolve()}. Link auto-closes in {minutes} min "
                  "(Ctrl+C to close now). Keep the laptop awake (e.g. `caffeinate -d` in another terminal).")
        else:
            url = f"http://127.0.0.1:{port}/"
            print(f"bolo record: {url}  (local only)")
            if open_browser:
                import webbrowser

                webbrowser.open(url)
        deadline = time.time() + minutes * 60 if share else float("inf")
        while time.time() < deadline:
            time.sleep(1)
        print("Link expired; shutting down.")
    except KeyboardInterrupt:
        pass
    finally:
        if tunnel:
            tunnel.terminate()
        httpd.shutdown()
        httpd.server_close()
        n = len(store.manifest())
        print(f"Closed. {n} recordings in {Path(root).resolve()}")
