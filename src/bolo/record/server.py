"""`bolo record`: a local, offline web tool for collecting the human-spoken EVAL set.

- Binds to 127.0.0.1 only; nothing leaves the machine.
- A speaker picks a pseudonym and gives explicit consent before any recording.
  record + project_use are required; public_release is a separate opt-in (default off).
- Audio is captured in the browser and encoded as 16 kHz mono 16-bit WAV (no ffmpeg needed).
- Everything lives in `recordings/` (gitignored): prompts, speakers.json, manifest.jsonl, audio/<speaker>/.
- "Withdraw" deletes a speaker's audio and manifest entries.

Never commit or publish anyone's audio without their consent (see docs/AUDIO.md).
"""

from __future__ import annotations

import io
import json
import re
import threading
import time
import wave
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .prompts import PROMPT_VERSION, build_prompts

TOOL_VERSION = "bolo-record 0.1"
SR = 16000
MIN_S, MAX_S = 0.3, 15.0
SLUG = re.compile(r"[^a-z0-9_-]+")


def slugify(name: str) -> str:
    s = SLUG.sub("-", name.strip().lower()).strip("-")
    return s[:40] or "speaker"


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

    def add_speaker(self, pseudonym: str, consent: dict, meta: dict) -> str:
        if not (consent.get("record") and consent.get("project_use")):
            raise ValueError("recording requires consent to record and to project use")
        sid = slugify(pseudonym)
        allowed_meta = {k: str(v)[:60] for k, v in (meta or {}).items()
                        if k in ("age_range", "region", "first_language", "mic") and v}
        with self.lock:
            sp = self._speakers()
            sp[sid] = {"pseudonym": pseudonym[:40], "consent": {
                "record": True, "project_use": True, "public_release": bool(consent.get("public_release")),
                "given_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, "meta": allowed_meta}
            self._save_speakers(sp)
        return sid

    def speaker(self, sid: str) -> dict | None:
        return self._speakers().get(sid)

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

    def save(self, sid: str, prompt_id: str, wav_bytes: bytes, transcript: str = "", mic: dict | None = None) -> dict:
        sp = self.speaker(sid)
        if sp is None:
            raise ValueError("unknown speaker (give consent first)")
        p = self.by_id.get(prompt_id)
        if p is None:
            raise ValueError("unknown prompt")
        dur = validate_wav(wav_bytes)
        d = self.root / "audio" / sid
        d.mkdir(parents=True, exist_ok=True)
        rel = f"audio/{sid}/{prompt_id}.wav"
        (self.root / rel).write_bytes(wav_bytes)
        entry = {
            "utt_id": f"{sid}__{prompt_id}", "speaker_id": sid, "prompt_id": prompt_id, "kind": p["kind"],
            "source_id": p["source_id"], "audio": rel, "sample_rate": SR, "duration_s": round(dur, 3),
            "intent": p["intent"], "slots": p["slots"],
            "transcript": p["asr_text"] if p["kind"] == "read" else (transcript.strip().lower()[:300] or None),
            "transcript_source": "prompt" if p["kind"] == "read" else ("speaker_typed" if transcript.strip() else None),
            "consent": sp["consent"], "speaker_meta": sp["meta"], "mic": mic or {},
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "tool": TOOL_VERSION,
        }
        with self.lock:
            rows = [r for r in self.manifest() if r["utt_id"] != entry["utt_id"]]  # re-record replaces
            rows.append(entry)
            self._write_manifest(rows)
        return entry

    def withdraw(self, sid: str) -> int:
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


def make_handler(store: Store):
    page = resources.files("bolo.record").joinpath("page.html").read_text(encoding="utf-8")

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _json(self, obj, status=HTTPStatus.OK):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length", 0))
            if n > 5_000_000:
                raise ValueError("upload too large")
            return self.rfile.read(n)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                body = page.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/prompts":
                sid = parse_qs(u.query).get("speaker", [""])[0]
                self._json({"prompts": store.prompts, "done": store.done(sid) if sid else []})
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/api/speaker":
                    d = json.loads(self._body() or b"{}")
                    sid = store.add_speaker(d.get("pseudonym", ""), d.get("consent", {}), d.get("meta", {}))
                    self._json({"speaker_id": sid, "done": store.done(sid)})
                elif u.path == "/api/upload":
                    mic = json.loads(q.get("mic", "{}") or "{}")
                    e = store.save(q.get("speaker", ""), q.get("prompt_id", ""), self._body(),
                                   q.get("transcript", ""), mic)
                    self._json({"ok": True, "duration_s": e["duration_s"]})
                elif u.path == "/api/withdraw":
                    d = json.loads(self._body() or b"{}")
                    self._json({"removed": store.withdraw(slugify(d.get("speaker_id", "")))})
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, json.JSONDecodeError) as e:
                self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)

    return H


def serve(root: Path | str = "recordings", test_path: Path | str = "data/text/v0/test.jsonl",
          port: int = 8765, open_browser: bool = True):
    store = Store(Path(root), Path(test_path))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store))
    url = f"http://127.0.0.1:{port}/"
    print(f"bolo record: {url}  (data in {Path(root).resolve()}, local only)")
    print("Reminder: never commit or release anyone's audio without their explicit consent. "
          "recordings/ is gitignored.")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
