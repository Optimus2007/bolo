"""Bolo-tiny: a from-scratch, vocabulary-free joint intent + slot model for edge devices.

Token representation = "PhonoHash": hashed character n-grams of several views of each word
  r: raw lowercase surface (Latin or Devanagari)
  t: Roman transliteration (Devanagari -> Roman)
  k: Hinglish phonetic key  (kholo/kolo/khollo -> kolo)
  s: consonant skeleton     (karo/kro -> kr)
each hashed twice (Bloom-style) into a small table and mean-pooled (an EmbeddingBag). The `plain`
variant uses only `r` (ordinary char-n-gram hashing), as the ablation baseline.
A 2-layer Transformer encoder feeds an intent head (masked mean pool) and a BIO tagging head.
"""

from __future__ import annotations

import json
import math
import random
import time
import zlib
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..normalize import Normalizer, phonetic_key, skeleton
from .tagging import NumberParser, decode, label_names, word_tags


@dataclass
class TinyConfig:
    features: str = "phono"  # phono | plain
    buckets: int = 8192
    emb_dim: int = 64
    d_model: int = 96
    layers: int = 2
    heads: int = 4
    ff: int = 192
    dropout: float = 0.1
    max_len: int = 40
    ngrams: tuple = (2, 3, 4)
    n_hashes: int = 2


class Featurizer:
    def __init__(self, tc: TinyConfig, norm: Normalizer):
        self.tc = tc
        self.norm = norm
        self._cache: dict[str, list[int]] = {}

    def _grams(self, s: str) -> list[str]:
        s2 = f"<{s}>"
        out = [s2]
        for n in self.tc.ngrams:
            out += [s2[i:i + n] for i in range(max(0, len(s2) - n + 1))]
        return out

    def word_ids(self, word: str) -> list[int]:
        if word in self._cache:
            return self._cache[word]
        raw = word.lower()
        views = [("r", raw)]
        if self.tc.features == "phono":
            roman = self.norm.to_roman(word)
            views += [("t", roman), ("k", phonetic_key(roman)), ("s", skeleton(roman))]
        ids = []
        for tag, s in views:
            for g in self._grams(s):
                for h in range(self.tc.n_hashes):
                    ids.append(1 + zlib.crc32(f"{h}|{tag}|{g}".encode()) % (self.tc.buckets - 1))
        self._cache[word] = ids
        return ids


class TinyModel(nn.Module):
    def __init__(self, tc: TinyConfig, n_intents: int, n_tags: int):
        super().__init__()
        self.tc = tc
        self.emb = nn.EmbeddingBag(tc.buckets, tc.emb_dim, mode="mean")
        self.proj = nn.Linear(tc.emb_dim, tc.d_model)
        self.pos = nn.Embedding(tc.max_len, tc.d_model)
        layer = nn.TransformerEncoderLayer(tc.d_model, tc.heads, tc.ff, tc.dropout, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, tc.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(tc.d_model)
        self.intent_head = nn.Linear(tc.d_model, n_intents)
        self.tag_head = nn.Linear(tc.d_model, n_tags)

    def forward(self, flat_ids, offsets, lengths):
        B, L = len(lengths), max(lengths)
        tok = self.emb(flat_ids, offsets)  # (num_tokens, E)
        x = torch.zeros(B, L, tok.shape[-1])
        mask = torch.ones(B, L, dtype=torch.bool)  # True = pad
        k = 0
        for b, n in enumerate(lengths):
            x[b, :n] = tok[k:k + n]
            mask[b, :n] = False
            k += n
        h = self.proj(x) + self.pos(torch.arange(L).unsqueeze(0))
        h = self.norm(self.enc(h, src_key_padding_mask=mask))
        keep = (~mask).unsqueeze(-1).float()
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1)
        return self.intent_head(pooled), self.tag_head(h)


_SYL = [("ka", "का"), ("ki", "कि"), ("ko", "को"), ("ra", "रा"), ("ri", "री"), ("ma", "मा"), ("mi", "मी"),
        ("na", "ना"), ("ni", "नि"), ("ta", "ता"), ("to", "टो"), ("sa", "सा"), ("si", "सी"), ("la", "ला"),
        ("lo", "लो"), ("pa", "पा"), ("pu", "पु"), ("da", "दा"), ("de", "दे"), ("ga", "गा"), ("go", "गो"),
        ("ba", "बा"), ("bi", "बि"), ("va", "वा"), ("ve", "वे"), ("ja", "जा"), ("ha", "हा"), ("ya", "या"),
        ("an", "अन"), ("ar", "अर"), ("in", "इन"), ("sh", "श"), ("ch", "च"), ("n", "न"), ("r", "र"), ("t", "ट")]


class Augmenter:
    """Training-time augmentation for unseen slot values and unseen phrasing.

    - slot-value randomization: replace open-slot spans (folder/path/name/app) with random 1-3 word
      pseudo-words (Roman or Devanagari, matching the original script) or random English words, so the
      model must read context rather than memorize value -> intent associations;
    - word dropout on non-slot words and random filler insertion, for unseen phrasings.
    """

    OPEN = {"folder", "path", "name", "app"}

    def __init__(self, label_names: list[str], english_words: list[str], fillers: list[str],
                 p_value: float = 0.5, p_drop: float = 0.1, p_filler: float = 0.15):
        self.labels = label_names
        self.english = english_words
        self.fillers = fillers
        self.p_value, self.p_drop, self.p_filler = p_value, p_drop, p_filler
        self.o = label_names.index("O")

    def _pseudo(self, rng: random.Random, deva: bool) -> str:
        parts = rng.choices(_SYL, k=rng.randint(2, 4))
        return "".join(d if deva else r for r, d in parts)

    def _value(self, rng: random.Random, deva: bool) -> list[str]:
        n = rng.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
        out = []
        for _ in range(n):
            out.append(rng.choice(self.english) if rng.random() < 0.3 else self._pseudo(rng, deva))
        return out

    def __call__(self, rng: random.Random, words: list[str], tags: list[int]):
        if rng.random() < self.p_value:
            new_w, new_t, i = [], [], 0
            while i < len(words):
                lab = self.labels[tags[i]]
                if lab.startswith("B-") and lab[2:] in self.OPEN:
                    j = i + 1
                    while j < len(words) and self.labels[tags[j]] == "I-" + lab[2:]:
                        j += 1
                    deva = any("ऀ" <= ch <= "ॿ" for w in words[i:j] for ch in w)
                    val = self._value(rng, deva)
                    new_w += val
                    new_t += [tags[i]] + [self.labels.index("I-" + lab[2:])] * (len(val) - 1)
                    i = j
                else:
                    new_w.append(words[i])
                    new_t.append(tags[i])
                    i += 1
            words, tags = new_w, new_t
        if len(words) > 2 and self.p_drop > 0:
            keep = [not (t == self.o and rng.random() < self.p_drop) for t in tags]
            if any(keep):
                words = [w for w, k in zip(words, keep) if k]
                tags = [t for t, k in zip(tags, keep) if k]
        if rng.random() < self.p_filler:
            pos = rng.randint(0, len(words))
            words = words[:pos] + [rng.choice(self.fillers)] + words[pos:]
            tags = tags[:pos] + [self.o] + tags[pos:]
        return words, tags


def _batch(feat: Featurizer, seqs: list[list[str]], max_len: int):
    flat, offsets, lengths = [], [], []
    for words in seqs:
        words = words[:max_len] or ["<empty>"]
        lengths.append(len(words))
        for w in words:
            offsets.append(len(flat))
            flat.extend(feat.word_ids(w))
    return torch.tensor(flat), torch.tensor(offsets), lengths


class BoloTiny:
    """Train / predict wrapper (predictor interface used by bolo.evaluate)."""

    name = "bolo-tiny"

    def __init__(self, cfg: Config, tc: TinyConfig | None = None):
        self.cfg = cfg
        self.tc = tc or TinyConfig()
        self.norm = Normalizer.from_config(cfg)
        self.feat = Featurizer(self.tc, self.norm)
        self.numbers = NumberParser(cfg)
        self.intents = list(cfg.schema["intents"])
        self.labels = label_names(cfg)
        self.label_id = {l: i for i, l in enumerate(self.labels)}
        self.model = TinyModel(self.tc, len(self.intents), len(self.labels))
        self.temperature = 1.0

    # ----- data -----
    def encode_row(self, text: str, spans: list[dict], intent: str, soft=None):
        words, tags = word_tags(self.norm, text, spans, intent)
        words, tags = words[: self.tc.max_len], tags[: self.tc.max_len]
        if not words:
            words, tags = ["<empty>"], ["O"]
        return words, [self.label_id[t] for t in tags], self.intents.index(intent), soft

    def groups_from_rows(self, rows: list[dict], use_views: bool) -> list[list[tuple]]:
        groups = []
        for r in rows:
            g = [self.encode_row(r["text"], r["spans"], r["intent"], r.get("teacher_intent"))]
            if use_views:
                g += [self.encode_row(v["text"], v["spans"], r["intent"], v.get("teacher_intent"))
                      for v in r.get("views", [])]
            groups.append(g)
        return groups

    # ----- training -----
    def fit(self, train_rows: list[dict], dev_rows: list[dict], epochs: int = 20, batch_size: int = 64,
            lr: float = 2e-3, use_views: bool = False, consistency: float = 0.0, seed: int = 0,
            augment: bool = False, kd_alpha: float = 0.0, kd_temp: float = 2.0, log=print) -> dict:
        torch.manual_seed(seed)
        rng = random.Random(seed)
        groups = self.groups_from_rows(train_rows, use_views or consistency > 0)
        aug = None
        if augment:
            fillers = sorted({w for c in ("pre", "post") for p in self.cfg.classes[c] for w in p})
            fillers += [self.cfg.hindi[w] for w in fillers if w in self.cfg.hindi]
            aug = Augmenter(self.labels, sorted(self.cfg.english), fillers)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * math.ceil(len(groups) / batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
        hist = []
        t0 = time.time()
        for ep in range(epochs):
            self.model.train()
            rng.shuffle(groups)
            tot = 0.0
            for i in range(0, len(groups), batch_size):
                gb = groups[i:i + batch_size]
                if aug is not None:
                    gb = [[(*aug(rng, w, t), y, sft) for (w, t, y, sft) in g] for g in gb]
                items = [x for g in gb for x in g]
                seqs = [w for w, _, _, _ in items]
                flat, offs, lens = _batch(self.feat, seqs, self.tc.max_len)
                il, tl = self.model(flat, offs, lens)
                y_int = torch.tensor([y for _, _, y, _ in items])
                y_tag = torch.full(tl.shape[:2], -100, dtype=torch.long)
                for b, (_, t, _, _) in enumerate(items):
                    y_tag[b, :len(t)] = torch.tensor(t)
                loss = F.cross_entropy(il, y_int) + F.cross_entropy(tl.reshape(-1, tl.shape[-1]),
                                                                    y_tag.reshape(-1), ignore_index=-100)
                if kd_alpha > 0:
                    idx = [b for b, it in enumerate(items) if it[3] is not None]
                    if idx:
                        tp = torch.tensor([items[b][3] for b in idx]).clamp_min(1e-6)
                        tp = (tp.log() / kd_temp).softmax(-1)  # re-temper teacher probs
                        sp = F.log_softmax(il[idx] / kd_temp, -1)
                        loss = loss + kd_alpha * kd_temp ** 2 * F.kl_div(sp, tp, reduction="batchmean")
                if consistency > 0:
                    loss = loss + consistency * self._consistency_loss(gb, il, tl, lens)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item() * len(gb)
            hist.append(round(tot / len(groups), 4))
            log(f"epoch {ep + 1}/{epochs} loss {hist[-1]:.4f} ({time.time() - t0:.0f}s)")
        self.fit_temperature(dev_rows)
        return {"loss": hist, "train_seconds": round(time.time() - t0, 1), "temperature": self.temperature}

    def _consistency_loss(self, gb, il, tl, lens):
        """Symmetric KL between each view and the group mean (intent), and between views with equal
        word counts (tags)."""
        losses = []
        k = 0
        for g in gb:
            n = len(g)
            if n < 2:
                k += n
                continue
            lp = F.log_softmax(il[k:k + n], -1)
            mean = lp.exp().mean(0, keepdim=True).clamp_min(1e-8)
            losses.append(F.kl_div(lp, mean.expand_as(lp), reduction="batchmean") +
                          F.kl_div(mean.log().expand_as(lp), lp.exp(), reduction="batchmean"))
            L = lens[k]
            same = [j for j in range(n) if lens[k + j] == L]
            if len(same) > 1:
                tp = F.log_softmax(tl[[k + j for j in same], :L], -1)
                tm = tp.exp().mean(0, keepdim=True).clamp_min(1e-8)
                losses.append(F.kl_div(tp, tm.expand_as(tp), reduction="batchmean") / L)
            k += n
        return torch.stack(losses).mean() if losses else torch.tensor(0.0)

    @torch.no_grad()
    def fit_temperature(self, dev_rows: list[dict]):
        logits, ys = self._intent_logits([r["text"] for r in dev_rows]), torch.tensor(
            [self.intents.index(r["intent"]) for r in dev_rows])
        best = (1.0, float("inf"))
        for t in [0.5 + 0.1 * i for i in range(31)]:
            nll = F.cross_entropy(logits / t, ys).item()
            if nll < best[1]:
                best = (t, nll)
        self.temperature = round(best[0], 2)

    # ----- inference -----
    @torch.no_grad()
    def _intent_logits(self, texts):
        self.model.eval()
        out = []
        for i in range(0, len(texts), 256):
            seqs = [[w for w, _, _ in self.norm.words(t)] for t in texts[i:i + 256]]
            flat, offs, lens = _batch(self.feat, seqs, self.tc.max_len)
            il, _ = self.model(flat, offs, lens)
            out.append(il)
        return torch.cat(out)

    @torch.no_grad()
    def predict(self, texts: list[str]) -> list[dict]:
        self.model.eval()
        preds = []
        for i in range(0, len(texts), 256):
            chunk = texts[i:i + 256]
            seqs = [[w for w, _, _ in self.norm.words(t)][: self.tc.max_len] for t in chunk]
            flat, offs, lens = _batch(self.feat, seqs, self.tc.max_len)
            il, tl = self.model(flat, offs, lens)
            probs = F.softmax(il / self.temperature, -1)
            tag_ids = tl.argmax(-1)
            for b, words in enumerate(seqs):
                conf, idx = probs[b].max(-1)
                intent = self.intents[idx.item()]
                tags = [self.labels[j] for j in tag_ids[b, :len(words)].tolist()]
                slots = decode(self.cfg, self.norm, self.numbers, intent, words, tags) if words else {}
                preds.append({"intent": intent, "slots": slots, "confidence": round(conf.item(), 4)})
        return preds

    # ----- persistence -----
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def save(self, out_dir: Path | str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out / "model.pt")
        meta = {"tiny_config": asdict(self.tc), "intents": self.intents, "labels": self.labels,
                "temperature": self.temperature}
        (out / "config.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, cfg: Config, model_dir: Path | str) -> "BoloTiny":
        d = Path(model_dir)
        meta = json.loads((d / "config.json").read_text())
        tcd = meta["tiny_config"]
        tcd["ngrams"] = tuple(tcd["ngrams"])
        obj = cls(cfg, TinyConfig(**tcd))
        obj.model.load_state_dict(torch.load(d / "model.pt", weights_only=True))
        obj.temperature = meta["temperature"]
        return obj
