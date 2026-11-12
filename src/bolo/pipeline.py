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

import json
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


BEAMS = 5  # beam search + rerank: +8.3 points correct actions on held-out macOS voices (results/speech/decoding_test.json)
RERANK = {"w_nlu": 0.5, "w_ground": 2.0, "w_none": 1.0}
COMMAND_WORDS = "terminal, claude code, folder, volume, brightness, battery, wifi, bluetooth"


def looped(text: str) -> bool:
    """Whisper sometimes gets stuck repeating itself ("cpu cpu cpu cpu ..."). Such a transcript is not something the
    user said, so it must never be acted on."""
    w = text.lower().split()
    if len(w) < 4:
        return False
    run = 1
    for i in range(1, len(w)):
        run = run + 1 if w[i] == w[i - 1] else 1
        if run >= 4:
            return True
    for n in (2, 3):  # a phrase repeated 3+ times in a row
        for i in range(len(w) - 3 * n + 1):
            if w[i:i + n] == w[i + n:i + 2 * n] == w[i + 2 * n:i + 3 * n]:
                return True
    return len(w) >= 10 and len(set(w)) / len(w) < 0.3


def build_prompt(kind: str | None, env: Environment | None = None, max_chars: int = 600) -> str | None:
    if kind is None:
        return None
    if kind == "commands":
        return COMMAND_WORDS
    if kind == "commands+apps":
        apps = ", ".join((env.apps if env else [])[:80])
        return (COMMAND_WORDS + (", " + apps if apps else ""))[:max_chars]
    return kind


@dataclass
class Bolo:
    nlu: object
    cfg: object
    asr: object | None = None
    env: Environment | None = None
    threshold: float = 0.0
    beams: int = 1
    rerank: object | None = None
    memory: object | None = None
    name: str | None = None  # what the user calls the assistant (default: settings, else "Myna")

    def __post_init__(self):
        from .assistant import get_name
        from .normalize import Normalizer

        self.name = self.name or get_name()
        self._to_roman = Normalizer.from_config(self.cfg).to_roman

    def _strip(self, text: str) -> tuple[str, bool]:
        from .assistant import strip_name

        return strip_name(text, self.name, self._to_roman)

    @classmethod
    def load(cls, nlu_path: str = "models/bolo-tiny/phono_views_aug-s0", asr: str | None = None,
             asr_language: str = "en", env: Environment | None = None, prompt: str | None = "commands",
             beams: int = BEAMS, memory=None) -> "Bolo":
        """asr: a Whisper model id or local dir (e.g. "models/bolo-whisper-fp16"); None = text only.
        prompt: vocabulary hint for the recogniser. "commands" (default) = core command words; "commands+apps" also
        adds the app names found in `env` (helps rare names); None = no prompt; any other string is used as is.
        Measured on the macOS voice set: WER 13.3% -> 12.4% ("commands") / 11.2% ("commands+apps"); see results/.
        beams: 1 = greedy (fastest); >1 = beam search, and the N best transcripts are reranked by what they mean
        (NLU confidence + grounding), see bolo.asr.decode. memory: an optional bolo.adapt.Memory that learns
        from `feedback()`."""
        from .models.tiny import BoloTiny

        cfg = load_config()
        nlu = BoloTiny.load(cfg, nlu_path)
        tf = Path(nlu_path) / "threshold.json"  # confidence cut-off tuned on dev (see scripts/rescore_tiny.py)
        thr = json.loads(tf.read_text())["threshold"] if tf.exists() else 0.0
        asr_model = None
        if asr:
            from .speech_eval import WhisperASR
            from .assistant import get_name

            p = build_prompt(prompt, env)
            asr_model = WhisperASR(asr, asr_language, prompt=(get_name() + ", " + p) if p and prompt != p else p)
        from .asr.decode import Rerank

        return cls(nlu=nlu, cfg=cfg, asr=asr_model, env=env, beams=beams, rerank=Rerank(**RERANK), memory=memory,
                   threshold=thr)

    def understand(self, text: str, env: Environment | None = None) -> dict:
        t0 = time.perf_counter()
        clean = lambda s: " ".join(s.lower().replace("।", " ").split())
        bare, named = self._strip(text)
        raw = self.nlu.predict([clean(bare)])[0]
        if named and raw["intent"] == "none":  # the "name" may have been part of the command after all
            alt = self.nlu.predict([clean(text)])[0]
            vals = [str(v) for v in (alt.get("slots") or {}).values()]
            if alt["intent"] != "none" and not any(self._strip(v + " x")[1] for v in vals):  # name must not become a slot
                raw, named = alt, False
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
                "addressed": named, "grounding": pred.get("grounding"), "ms": {"nlu": round(ms, 2)}}

    def listen(self, wav_path: str | Path, env: Environment | None = None) -> dict:
        if self.asr is None:
            raise RuntimeError("no ASR model loaded: Bolo.load(asr='models/bolo-whisper') or a Whisper model id")
        from .audio.augment import SR, read_wav, resample

        from .asr.guards import MIN_SPEECH, max_tokens, rate_ok, speech_seconds

        x, sr = read_wav(str(wav_path))
        x = resample(x, sr, SR)
        t0 = time.perf_counter()
        env = env or self.env
        sec = speech_seconds(x)
        if sec < MIN_SPEECH:
            return self._didnt_catch("", "no speech heard")
        ok = lambda h: not looped(h) and rate_ok(h, sec)
        if self.beams <= 1:
            text = self.asr(x, max_new=max_tokens(sec))
            asr_ms = (time.perf_counter() - t0) * 1000
            out = self.understand(text, env) if ok(text) else self._didnt_catch(text)
            out["ms"]["asr"] = round(asr_ms, 1)
            return out
        hyps = self.asr.nbest(x, beams=self.beams, max_new=max_tokens(sec))
        heard = hyps
        hyps = [(self._strip(h)[0], sc) for h, sc in hyps if ok(h)]
        if not hyps:
            out = self._didnt_catch(heard[0][0] if heard else "")
            out["ms"]["asr"] = round((time.perf_counter() - t0) * 1000, 1)
            return out
        asr_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        if self.memory is not None:
            top, cands = self.memory.choose(hyps, self.nlu, self.cfg, env)
            pred, text = top["pred"], top["text"]
            self._last = cands, top
        else:
            from .asr.decode import choose

            c = choose(hyps, self.nlu, self.cfg, env, self.rerank, self.threshold)
            pred, text = c.pred, c.text
        abstain = pred["intent"] == "none"
        reason = pred.get("abstain_reason") or (pred.get("grounding") or {}).get("abstain")
        return {"intent": pred["intent"], "slots": pred["slots"], "confidence": round(float(pred.get("confidence", 1.0)), 4),
                "abstain": abstain, "reason": (reason or "not a supported command") if abstain else None,
                "transcript": text, "alternatives": [h for h, _ in heard], "grounding": pred.get("grounding"),
                "ms": {"asr": round(asr_ms, 1), "nlu": round((time.perf_counter() - t1) * 1000, 2)}}

    def _didnt_catch(self, text: str, why: str = "could not hear that clearly") -> dict:
        return {"intent": "none", "slots": {}, "confidence": 0.0, "abstain": True,
                "reason": why, "transcript": text,
                "addressed": False, "grounding": None, "ms": {"nlu": 0.0}}

    def feedback(self, intent: str, slots: dict | None = None):
        """Tell Bolo what the last `listen()` should have done (or confirm it). Needs memory=Memory()."""
        if self.memory is None or not getattr(self, "_last", None):
            return
        cands, top = self._last
        self.memory.feedback(cands, {"intent": intent, "slots": slots or {}}, top)
