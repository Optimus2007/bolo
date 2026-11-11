"""Pretrained-encoder joint intent + slot tagger (e.g. google/muril-base-cased).

Same word-level BIO scheme, views and augmentation as Bolo-tiny, so the comparison isolates the
encoder (pretrained 236M-param MuRIL vs from-scratch 0.7M-param PhonoHash).
Word labels go on each word's first sub-token; intent from the [CLS] vector.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..normalize import Normalizer
from .tagging import NumberParser, decode, label_names, word_tags
from .tiny import Augmenter


class _Joint(nn.Module):
    def __init__(self, backbone, hidden: int, n_intents: int, n_tags: int):
        super().__init__()
        self.backbone = backbone
        # MuRIL's final LayerNorm has gain ~0.015, so its last-layer output is almost the same constant
        # bias vector for every input (linear probes barely learn). Use the second-to-last layer instead.
        self.layer = -2
        self.ln = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(0.1)
        self.intent_head = nn.Linear(hidden, n_intents)
        self.tag_head = nn.Linear(hidden, n_tags)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h = self.drop(self.ln(out.hidden_states[self.layer]))
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.intent_head(pooled), self.tag_head(h)


class EncoderTagger:
    def __init__(self, cfg: Config, model_name: str = "google/muril-base-cased", device: str | None = None,
                 max_len: int = 64):
        from transformers import AutoModel, AutoTokenizer

        self.cfg = cfg
        self.name = model_name
        self.norm = Normalizer.from_config(cfg)
        self.numbers = NumberParser(cfg)
        self.intents = list(cfg.schema["intents"])
        self.labels = label_names(cfg)
        self.label_id = {l: i for i, l in enumerate(self.labels)}
        self.tok = AutoTokenizer.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name)
        self.model = _Joint(backbone, backbone.config.hidden_size, len(self.intents), len(self.labels))
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.max_len = max_len
        self.temperature = 1.0

    def _encode(self, word_lists: list[list[str]], tag_lists: list[list[int]] | None = None):
        enc = self.tok(word_lists, is_split_into_words=True, truncation=True, max_length=self.max_len,
                       padding=True, return_tensors="pt")
        first_idx = []  # per example: sub-token index of each word's first piece
        tag_t = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
        for b in range(len(word_lists)):
            wids = enc.word_ids(b)
            seen, firsts = set(), []
            for j, w in enumerate(wids):
                if w is not None and w not in seen:
                    seen.add(w)
                    firsts.append(j)
                    if tag_lists is not None and w < len(tag_lists[b]):
                        tag_t[b, j] = tag_lists[b][w]
            first_idx.append(firsts)
        return enc, tag_t, first_idx

    def fit(self, train_rows, dev_rows, epochs: int = 4, batch_size: int = 32, lr: float = 5e-5,
            use_views: bool = True, augment: bool = True, seed: int = 0, log=print) -> dict:
        torch.manual_seed(seed)
        rng = random.Random(seed)
        items = []
        for r in train_rows:
            srcs = [(r["text"], r["spans"])] + ([(v["text"], v["spans"]) for v in r.get("views", [])] if use_views else [])
            for text, spans in srcs:
                w, t = word_tags(self.norm, text, spans, r["intent"])
                items.append((w or ["<empty>"], [self.label_id[x] for x in t] or [0], self.intents.index(r["intent"])))
        aug = None
        if augment:
            fillers = sorted({w for c in ("pre", "post") for p in self.cfg.classes[c] for w in p})
            aug = Augmenter(self.labels, sorted(self.cfg.english), fillers)
        head_params = (list(self.model.intent_head.parameters()) + list(self.model.tag_head.parameters())
                       + list(self.model.ln.parameters()))
        opt = torch.optim.AdamW([{"params": self.model.backbone.parameters(), "lr": lr},
                                 {"params": head_params, "lr": lr * 20}], weight_decay=0.01)
        steps = epochs * math.ceil(len(items) / batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr * 20], total_steps=steps, pct_start=0.1)
        t0, hist = time.time(), []
        for ep in range(epochs):
            self.model.train()
            rng.shuffle(items)
            tot = 0.0
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                if aug:
                    batch = [(*aug(rng, w, t), y) for w, t, y in batch]
                enc, tag_t, _ = self._encode([b[0] for b in batch], [b[1] for b in batch])
                il, tl = self.model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
                y = torch.tensor([b[2] for b in batch], device=self.device)
                loss = F.cross_entropy(il, y) + F.cross_entropy(tl.reshape(-1, tl.shape[-1]),
                                                                tag_t.to(self.device).reshape(-1), ignore_index=-100)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item() * len(batch)
            hist.append(round(tot / len(items), 4))
            log(f"epoch {ep + 1}/{epochs} loss {hist[-1]:.4f} ({time.time() - t0:.0f}s)")
        self.fit_temperature(dev_rows)
        return {"loss": hist, "train_seconds": round(time.time() - t0, 1), "temperature": self.temperature}

    @torch.no_grad()
    def _forward_texts(self, texts):
        self.model.eval()
        outs = []
        for i in range(0, len(texts), 64):
            wl = [[w for w, _, _ in self.norm.words(t)] or ["<empty>"] for t in texts[i:i + 64]]
            enc, _, firsts = self._encode(wl)
            il, tl = self.model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
            outs.append((wl, il.float().cpu(), tl.float().cpu(), firsts))
        return outs

    @torch.no_grad()
    def fit_temperature(self, dev_rows):
        logits = torch.cat([il for _, il, _, _ in self._forward_texts([r["text"] for r in dev_rows])])
        ys = torch.tensor([self.intents.index(r["intent"]) for r in dev_rows])
        self.temperature = min(((F.cross_entropy(logits / t, ys).item(), t) for t in
                                [0.5 + 0.1 * i for i in range(31)]))[1]

    @torch.no_grad()
    def predict(self, texts: list[str]) -> list[dict]:
        preds = []
        for wl, il, tl, firsts in self._forward_texts(texts):
            probs = F.softmax(il / self.temperature, -1)
            for b, words in enumerate(wl):
                conf, idx = probs[b].max(-1)
                intent = self.intents[idx.item()]
                tag_ids = tl[b].argmax(-1).tolist()
                tags = [self.labels[tag_ids[j]] for j in firsts[b]]
                ws = words[:len(tags)]
                slots = decode(self.cfg, self.norm, self.numbers, intent, ws, tags) if ws != ["<empty>"] else {}
                preds.append({"intent": intent, "slots": slots, "confidence": round(conf.item(), 4)})
        return preds

    @torch.no_grad()
    def intent_probs(self, texts: list[str]) -> list[list[float]]:
        out = []
        for _, il, _, _ in self._forward_texts(texts):
            out += F.softmax(il / self.temperature, -1).tolist()
        return out

    @classmethod
    def load(cls, cfg: Config, model_dir, device: str | None = None) -> "EncoderTagger":
        d = Path(model_dir)
        meta = json.loads((d / "config.json").read_text())
        obj = cls(cfg, meta["base"], device=device)
        obj.model.load_state_dict(torch.load(d / "model.pt", map_location="cpu", weights_only=True))
        obj.model.to(obj.device)
        obj.temperature = meta["temperature"]
        return obj

    def to(self, device: str):
        self.device = device
        self.model.to(device)
        return self

    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def save(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out / "model.pt")
        (out / "config.json").write_text(json.dumps({"base": self.name, "temperature": self.temperature,
                                                     "intents": self.intents, "labels": self.labels}, indent=2))
