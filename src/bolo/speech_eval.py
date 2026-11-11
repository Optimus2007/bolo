"""`bolo eval`: speech -> text -> action benchmark.

For every audio set (human `recordings/`, synthetic TTS, macOS `say`, ...) and every ASR system:
  ASR      : WER / CER / phonetic WER (rows that have a reference transcript)
  NLU      : for each text model, intent accuracy, slot F1, exact match on the ASR output,
             strict and after environment grounding (the action that would actually run)
  oracle   : the same NLU on the reference transcript (separates ASR errors from NLU errors)
  latency  : ASR and NLU milliseconds per utterance (batch 1, this machine), real-time factor
Sets are NEVER pooled: human and synthetic numbers are reported side by side, never averaged together.

Privacy: the aggregate JSON/Markdown goes to results/speech/ (committable). Per-utterance hypotheses of
human speech go to data/eval/ (gitignored), never to results/.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from .audio.augment import SR
from .audio.metrics import corpus_scores
from .evaluate import finalize, score
from .grounding import Environment, ground

# name -> (model id or local path, decode language)
ASR_SYSTEMS = {
    "whisper-base": ("openai/whisper-base", "hi"),
    "whisper-small": ("openai/whisper-small", "hi"),
    "swift": ("Oriserve/Whisper-Hindi2Hinglish-Swift", "en"),
    "bolo-whisper": ("models/bolo-whisper", "en"),
}


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------- data ----------------
def load_set(manifest: Path | str, test_path: Path | str, name: str | None = None) -> list[dict]:
    """Rows from a recorder-style or TTS manifest, joined with the gold test row (labels + meta)."""
    manifest = Path(manifest)
    gold = {}
    for line in open(test_path, encoding="utf-8"):
        r = json.loads(line)
        gold[r["id"]] = r
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        g = gold.get(m.get("source_id"), {})
        kind = m.get("kind", "read")
        src = name or ("human" if "speaker_id" in m and "tts_model" not in m else "synthetic")
        rows.append({
            "utt_id": m.get("utt_id", m.get("audio")), "_wav": str(manifest.parent / m["audio"]),
            "set": f"{src}_{kind}", "speaker": m.get("speaker_id") or m.get("speaker") or m.get("voice"),
            "intent": m.get("intent", g.get("intent")), "slots": m.get("slots", g.get("slots", {})),
            "ref": m.get("transcript") if "transcript" in m else m.get("asr_text"),
            "script": "speech", "meta": g.get("meta", {}), "duration_s": m.get("duration_s"),
        })
    return rows


def environment(cfg) -> Environment:
    P = cfg.open_pools
    return Environment([e.value for e in P["folders"]], [e.value for e in P["files"]],
                       [e.value for e in P["apps"]], [e.value for e in P["contacts"]])


# ---------------- ASR ----------------
class WhisperASR:
    def __init__(self, path: str, language: str | None, dev: str | None = None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.dev = dev or device()
        self.proc = WhisperProcessor.from_pretrained(path)
        self.model = WhisperForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32).to(self.dev).eval()
        self.language = language
        self.torch = torch

    def __call__(self, wav) -> str:
        feats = self.proc.feature_extractor([wav], sampling_rate=SR, return_tensors="pt").input_features.to(self.dev)
        kw = {"task": "transcribe", "max_new_tokens": 64}
        if self.language:
            kw["language"] = self.language
        with self.torch.inference_mode():
            ids = self.model.generate(input_features=feats, **kw)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def run_asr(asr, rows: list[dict]) -> tuple[list[str], list[float]]:
    from .audio.augment import read_wav, resample

    hyps, ms = [], []
    for i, r in enumerate(rows):
        x, sr = read_wav(r["_wav"])
        x = resample(x, sr, SR)
        if i == 0:
            asr(x)  # warm-up, not timed
        t = time.perf_counter()
        hyps.append(asr(x))
        ms.append((time.perf_counter() - t) * 1000)
    return hyps, ms


# ---------------- NLU ----------------
def nlu_models(cfg, names: list[str]):
    for n in names:
        if n == "rules":
            from .baselines.rules import RuleBaseline
            yield n, RuleBaseline(cfg), 0.0
        elif n == "tiny":
            from .models.tiny import BoloTiny
            yield n, BoloTiny.load(cfg, "models/bolo-tiny/phono_views_aug-s0"), 0.0
        elif n == "muril":
            from .models.encoder import EncoderTagger
            thr = json.loads(Path("results/text/muril_aug-s0.json").read_text())["threshold"]
            yield n, EncoderTagger.load(cfg, "models/muril_aug-s0"), thr
        else:
            raise ValueError(f"unknown NLU model {n!r}")


def clean(text: str) -> str:
    return " ".join(text.lower().replace("।", " ").split())


def _summ(s: dict) -> dict:
    return {k: s[k] for k in ("n", "intent_acc", "exact_match", "exact_match_lenient", "false_action_rate",
                              "over_abstain_rate", "wrong_action_rate")} | {"slot_f1": s["slot"]["f1"]}


def nlu_block(model, thr, cfg, env, rows, texts) -> tuple[dict, list[dict], float]:
    t = time.perf_counter()
    raw = [model.predict([x])[0] for x in texts]  # batch 1 = realistic latency
    ms = (time.perf_counter() - t) * 1000 / max(1, len(texts))
    strict = [finalize(p, cfg.schema, thr) for p in raw]
    grounded = [finalize(ground(p, env), cfg.schema, 0.0) if p["intent"] != "none" else p for p in strict]
    return {"strict": _summ(score(rows, strict)), "grounded": _summ(score(rows, grounded))}, grounded, ms


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1) if xs else 0.0


# ---------------- main ----------------
def run(sets: dict[str, list[dict]], asr_names: list[str], nlu_names: list[str], cfg,
        utt_out: Path | None = None, asr_factory=None) -> dict:
    from .normalize import Normalizer

    env, nz = environment(cfg), Normalizer.from_config(cfg)
    nlus = list(nlu_models(cfg, nlu_names))
    asr_factory = asr_factory or (lambda name: WhisperASR(*ASR_SYSTEMS[name]))
    res = {"sets": {}, "asr_systems": {n: ASR_SYSTEMS.get(n) for n in asr_names}, "nlu": nlu_names,
           "device": device(), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    utts = []
    for sname, rows in sets.items():
        S = {"n": len(rows), "speakers": len({r["speaker"] for r in rows}),
             "hours": round(sum(r["duration_s"] or 0 for r in rows) / 3600, 3), "asr": {}, "oracle": {}}
        ref_rows = [r for r in rows if r["ref"]]
        if ref_rows:  # oracle: NLU on reference transcripts
            for nname, model, thr in nlus:
                blk, _, ms = nlu_block(model, thr, cfg, env, ref_rows, [clean(r["ref"]) for r in ref_rows])
                S["oracle"][nname] = {**blk, "nlu_ms": round(ms, 2)}
        for aname in asr_names:
            path, _ = ASR_SYSTEMS.get(aname, ("", None))
            if path.startswith("models/") and not Path(path).exists():
                print(f"skip {aname}: {path} not found")
                continue
            print(f"[{sname}] ASR {aname} on {len(rows)} utts ...", flush=True)
            asr = asr_factory(aname)
            hyps, ms = run_asr(asr, rows)
            del asr
            A = {"latency_ms": {"median": pct(ms, 0.5), "p90": pct(ms, 0.9)},
                 "rtf": round(sum(ms) / 1000 / max(1e-9, sum(r["duration_s"] or 0 for r in rows)), 3), "nlu": {}}
            idx = [i for i, r in enumerate(rows) if r["ref"]]
            if idx:
                A["wer"] = corpus_scores([rows[i]["ref"] for i in idx], [hyps[i] for i in idx], nz)
            for nname, model, thr in nlus:
                blk, preds, nms = nlu_block(model, thr, cfg, env, rows, [clean(h) for h in hyps])
                e2e = [m + nms for m in ms]
                A["nlu"][nname] = {**blk, "nlu_ms": round(nms, 2),
                                   "e2e_ms": {"median": pct(e2e, 0.5), "p90": pct(e2e, 0.9)}}
                for r, h, p in zip(rows, hyps, preds):
                    utts.append({"set": sname, "utt_id": r["utt_id"], "asr": aname, "nlu": nname, "ref": r["ref"],
                                 "hyp": h, "gold": {"intent": r["intent"], "slots": r["slots"]},
                                 "pred": {"intent": p["intent"], "slots": p["slots"]}})
            S["asr"][aname] = A
        res["sets"][sname] = S
    if utt_out:
        utt_out.parent.mkdir(parents=True, exist_ok=True)
        utt_out.write_text("".join(json.dumps(u, ensure_ascii=False) + "\n" for u in utts), encoding="utf-8")
    return res


def to_markdown(res: dict) -> str:
    L = [f"# Speech benchmark ({res['created']}, device={res['device']})", "",
         "Sets are reported separately and never pooled. E2E EM = exact action after grounding.", ""]
    for sname, S in res["sets"].items():
        L += [f"## {sname}  (n={S['n']}, speakers={S['speakers']}, {S['hours'] * 60:.1f} min audio)", "",
              "| ASR | WER | CER | phon. WER | NLU | intent acc | EM strict | **E2E EM (grounded)** | false-action | ASR ms (med) | RTF |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for nname, o in S["oracle"].items():
            L.append(f"| *oracle transcript* | 0 | 0 | 0 | {nname} | {o['strict']['intent_acc']:.3f} | "
                     f"{o['strict']['exact_match']:.3f} | {o['grounded']['exact_match']:.3f} | "
                     f"{o['grounded']['false_action_rate']} | – | – |")
        for aname, A in S["asr"].items():
            w = A.get("wer", {})
            for nname, B in A["nlu"].items():
                L.append(f"| {aname} | {w.get('wer', '–')} | {w.get('cer', '–')} | {w.get('phonetic_wer', '–')} | {nname} | "
                         f"{B['strict']['intent_acc']:.3f} | {B['strict']['exact_match']:.3f} | "
                         f"{B['grounded']['exact_match']:.3f} | {B['grounded']['false_action_rate']} | "
                         f"{A['latency_ms']['median']} | {A['rtf']} |")
        L.append("")
    return "\n".join(L)
