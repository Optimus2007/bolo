"""Small causal LM (e.g. Qwen2.5-0.5B-Instruct) fine-tuned with LoRA on MLX to emit JSON.

Prompt:  "Convert the Hinglish voice command to JSON.\nCommand: <text>\nJSON:"
Target:  ' {"intent": ..., "slots": {...}}'
Confidence = geometric-mean token probability of the generated JSON.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from ..config import Config
from ..normalize import Normalizer
from .tagging import NumberParser, decode, label_names, word_tags
from .tiny import Augmenter

PROMPT = "Convert the Hinglish voice command to JSON.\nCommand: {text}\nJSON:"


def target(intent: str, slots: dict) -> str:
    return " " + json.dumps({"intent": intent, "slots": slots}, ensure_ascii=False)


def build_sft(cfg: Config, rows: list[dict], out_path: Path, aug_copies: int = 1, seed: int = 0) -> int:
    """Write mlx-lm completion-format JSONL. Each row once as-is, plus `aug_copies` augmented copies
    (slot-value randomization / word dropout / fillers, relabeled via the tagger decode)."""
    rng = random.Random(seed)
    norm = Normalizer.from_config(cfg)
    numbers = NumberParser(cfg)
    labels = label_names(cfg)
    lid = {l: i for i, l in enumerate(labels)}
    fillers = sorted({w for c in ("pre", "post") for p in cfg.classes[c] for w in p})
    aug = Augmenter(labels, sorted(cfg.english), fillers)
    n = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"prompt": PROMPT.format(text=r["text"]),
                                "completion": target(r["intent"], r["slots"])}, ensure_ascii=False) + "\n")
            n += 1
            words, tags = word_tags(norm, r["text"], r["spans"], r["intent"])
            for _ in range(aug_copies):
                w2, t2 = aug(rng, words, [lid[t] for t in tags])
                if not w2 or w2 == words:
                    continue
                slots = {} if r["intent"] == "none" else decode(cfg, norm, numbers, r["intent"], w2,
                                                                 [labels[i] for i in t2])
                if set(slots) != set(r["slots"]):
                    continue  # augmentation broke a slot; skip
                f.write(json.dumps({"prompt": PROMPT.format(text=" ".join(w2)),
                                    "completion": target(r["intent"], slots)}, ensure_ascii=False) + "\n")
                n += 1
    return n


def _parse(text: str) -> dict | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class LLMPredictor:
    def __init__(self, model_path: str, adapter_path: str | None = None, name: str = "llm",
                 batch_size: int = 32, max_tokens: int = 48):
        from mlx_lm import load

        self.name = name
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        self.batch_size = batch_size
        self.max_tokens = max_tokens

    def predict(self, texts: list[str]) -> list[dict]:
        from mlx_lm import batch_generate

        out = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            # mlx-lm's completion datasets train through the chat template, so inference must match
            prompts = [self.tokenizer.apply_chat_template([{"role": "user", "content": PROMPT.format(text=t)}],
                                                          add_generation_prompt=True) for t in chunk]
            resp = batch_generate(self.model, self.tokenizer, prompts, max_tokens=self.max_tokens,
                                  return_logprobs=True)
            for j, text in enumerate(resp.texts):
                obj = _parse(text)
                conf = 0.0
                lps = getattr(resp, "logprobs", None)
                if lps is not None and obj is not None:
                    toks = lps[j]
                    vals = [float(x) for x in toks][: max(1, len(self.tokenizer.encode(json.dumps(obj))))]
                    conf = math.exp(sum(vals) / len(vals)) if vals else 0.0
                if not isinstance(obj, dict) or not isinstance(obj.get("slots", {}), dict):
                    out.append({"intent": "none", "slots": {}, "confidence": 0.0, "raw": text})
                    continue
                out.append({"intent": str(obj.get("intent", "none")), "slots": obj.get("slots") or {},
                            "confidence": round(conf, 4)})
        return out
