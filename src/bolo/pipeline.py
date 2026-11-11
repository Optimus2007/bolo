"""One call from speech (or text) to a safe, grounded action.

    from bolo.pipeline import Bolo
    bolo = Bolo.load()                         # Bolo-tiny + (optional) a Whisper ASR model
    bolo.understand("rust learning folder mein terminal kholo")
    bolo.listen("command.wav")                 # audio file -> transcript -> action

Result (a plain dict, easy to hand to an agent):
    {"intent": "open_terminal", "slots": {"folder": "rust learning"}, "confidence": 0.97,
     "abstain": False, "reason": None, "transcript": "...", "grounding": {...}, "ms": {"asr": 120.1, "nlu": 1.2}}
When Bolo is not sure, or the folder/app/contact does not exist on this machine, it abstains
(intent "none", abstain True, with a reason) instead of guessing. Bolo never executes anything itself:
the caller decides what to run, and should ask the user before anything risky.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .config import load_config
from .evaluate import finalize
from .grounding import Environment, ground


def scan_mac(home: Path | str | None = None, max_depth: int = 2, max_entries: int = 5000) -> Environment:
    """Read-only scan of what exists on this Mac: folders/files under the home dir (shallow), and apps.
    Hidden folders and Library are skipped. Nothing is modified. Contacts are not read (pass them yourself)."""
    home = Path(home or Path.home())
    folders, files = [], []
    skip = {"Library", "Applications", "node_modules", ".Trash"}

    def walk(d: Path, depth: int):
        if depth > max_depth or len(folders) + len(files) > max_entries:
            return
        try:
            entries = sorted(d.iterdir())
        except (PermissionError, OSError):
            return
        for e in entries:
            if e.name.startswith(".") or e.name in skip:
                continue
            try:
                if e.is_dir() and not e.is_symlink():
                    folders.append(e.name.lower().replace("_", " ").replace("-", " "))
                    walk(e, depth + 1)
                elif e.is_file():
                    files.append(e.stem.lower().replace("_", " ").replace("-", " "))
            except OSError:
                continue

    walk(home, 0)
    apps = []
    for root in (Path("/Applications"), home / "Applications", Path("/System/Applications")):
        if root.exists():
            apps += [a.stem.lower() for a in root.glob("*.app")]
    uniq = lambda xs: sorted(set(x for x in xs if x))  # noqa: E731
    return Environment(uniq(folders), uniq(files), uniq(apps), [])


@dataclass
class Bolo:
    nlu: object
    cfg: object
    asr: object | None = None
    env: Environment | None = None
    threshold: float = 0.0

    @classmethod
    def load(cls, nlu_path: str = "models/bolo-tiny/phono_views_aug-s0", asr: str | None = None,
             asr_language: str = "en", env: Environment | None = None) -> "Bolo":
        """asr: a Whisper model id or local dir (e.g. "models/bolo-whisper"); None = text only."""
        from .models.tiny import BoloTiny

        cfg = load_config()
        nlu = BoloTiny.load(cfg, nlu_path)
        asr_model = None
        if asr:
            from .speech_eval import WhisperASR
            asr_model = WhisperASR(asr, asr_language)
        return cls(nlu=nlu, cfg=cfg, asr=asr_model, env=env)

    def understand(self, text: str, env: Environment | None = None) -> dict:
        t0 = time.perf_counter()
        raw = self.nlu.predict([" ".join(text.lower().replace("।", " ").split())])[0]
        pred = finalize(raw, self.cfg.schema, self.threshold)
        env = env or self.env
        if env is not None and pred["intent"] != "none":
            g = ground(pred, env)
            pred = finalize(g, self.cfg.schema, 0.0) | {"grounding": g["grounding"]}
        ms = (time.perf_counter() - t0) * 1000
        abstain = pred["intent"] == "none"
        reason = pred.get("abstain_reason") or (pred.get("grounding") or {}).get("abstain")
        if abstain and not reason:
            reason = "not a supported command"
        return {"intent": pred["intent"], "slots": pred["slots"], "confidence": round(float(pred.get("confidence", 1.0)), 4),
                "abstain": abstain, "reason": reason if abstain else None, "transcript": text,
                "grounding": pred.get("grounding"), "ms": {"nlu": round(ms, 2)}}

    def listen(self, wav_path: str | Path, env: Environment | None = None) -> dict:
        if self.asr is None:
            raise RuntimeError("no ASR model loaded: Bolo.load(asr='models/bolo-whisper') or a Whisper model id")
        from .audio.augment import SR, read_wav, resample

        x, sr = read_wav(str(wav_path))
        x = resample(x, sr, SR)
        t0 = time.perf_counter()
        text = self.asr(x)
        asr_ms = (time.perf_counter() - t0) * 1000
        out = self.understand(text, env)
        out["ms"]["asr"] = round(asr_ms, 1)
        return out
