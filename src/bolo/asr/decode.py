"""Smarter decoding for Bolo Sun: beam search, hotword boosting, and N-best reranking by meaning.

Plain Whisper decoding keeps one guess and ignores what exists on the machine. Here:

1. Beam search keeps the `beams` best partial transcripts instead of one.
2. Hotword boosting (`TrieBoost`) adds `bonus` to the log score of any token that continues a known phrase
   (app, folder, file and contact names plus core command words). The bonus is only paid while a phrase is
   being followed; a hypothesis that leaves the phrase does not keep it, because it is the beam's summed score
   that decides, and partial phrases rarely beat a clean reading unless the audio supports them.
   In maths: score(y) = sum_t log p(y_t | y_<t, audio) + bonus * (number of tokens inside known phrases).
3. N-best reranking: every returned hypothesis is parsed by the NLU and grounded against the environment.
   The final choice maximises
       S = asr_logprob / length  +  w_nlu * log p_nlu(intent)  +  w_ground * grounding_score  -  w_none * [abstain]
   and hypotheses that lead to the same action pool their probability (softmax over S, summed per action).

Nothing here trains anything; all weights are a handful of numbers tuned on a dev set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


class TrieBoost:
    """A HF LogitsProcessor: bonus for tokens that start or continue a known phrase (token-id prefix tree)."""

    def __init__(self, tokenizer, phrases: list[str], bonus: float = 2.0, start_bonus: float | None = None):
        self.bonus = bonus
        self.start_bonus = bonus * 0.5 if start_bonus is None else start_bonus
        self.trie: dict = {}
        self.max_len = 1
        for p in {p.strip().lower() for p in phrases if p and p.strip()}:
            for variant in (" " + p,):  # Whisper words after the first carry a leading space
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if not ids:
                    continue
                node = self.trie
                for t in ids:
                    node = node.setdefault(t, {})
                self.max_len = max(self.max_len, len(ids))
        self.roots = list(self.trie)

    def _children(self, seq: list[int]) -> set[int]:
        out: set[int] = set()
        for k in range(1, min(self.max_len, len(seq)) + 1):
            node = self.trie
            for t in seq[-k:]:
                node = node.get(t)
                if node is None:
                    break
            if node:
                out.update(node)
        return out

    def __call__(self, input_ids, scores):
        import torch

        if self.start_bonus:
            scores[:, self.roots] += self.start_bonus
        if self.bonus:
            for b, seq in enumerate(input_ids.tolist()):
                ch = self._children(seq)
                if ch:
                    idx = torch.tensor(sorted(ch), device=scores.device)
                    scores[b, idx] += self.bonus - self.start_bonus * torch.isin(
                        idx, torch.tensor(self.roots, device=scores.device)).float()
        return scores


@dataclass
class Rerank:
    w_nlu: float = 1.0       # weight on log p_nlu(intent)
    w_ground: float = 1.0    # weight on the grounding similarity of open slots (0..1)
    w_none: float = 1.0      # penalty when a hypothesis leads to "none" (abstain)
    temp: float = 1.0        # softmax temperature over hypotheses when pooling by action


@dataclass
class Choice:
    pred: dict
    text: str
    hyps: list[tuple[str, float]] = field(default_factory=list)
    support: float = 1.0      # pooled probability of the chosen action across the N-best


def _action_key(p: dict) -> tuple:
    return (p["intent"], tuple(sorted((k, str(v)) for k, v in p["slots"].items())))


def choose(hyps: list[tuple[str, float]], nlu, cfg, env, rr: Rerank, thr: float = 0.0) -> Choice:
    """hyps: [(text, length-normalised log prob)] best first. Returns the action with the highest pooled score."""
    from ..evaluate import finalize
    from ..grounding import ground

    texts = [" ".join(t.lower().replace("।", " ").split()) for t, _ in hyps]
    raws = nlu.predict(texts)
    cands = []
    for (t, lp), raw in zip(hyps, raws):
        p = finalize(raw, cfg.schema, thr)
        gscore = 1.0
        if p["intent"] != "none" and env is not None:
            g = ground(p, env)
            p = finalize(g, cfg.schema, 0.0) | {"grounding": g["grounding"]}
            sc = [v["score"] for v in g["grounding"].values() if isinstance(v, dict) and "score" in v]
            gscore = min(sc) if sc else 1.0
        conf = max(1e-6, float(raw.get("confidence", 1.0)))
        s = lp + rr.w_nlu * math.log(conf) + rr.w_ground * gscore - rr.w_none * (p["intent"] == "none")
        cands.append((s, t, p))
    m = max(s for s, _, _ in cands)
    pooled: dict[tuple, float] = {}
    first: dict[tuple, tuple] = {}
    for s, t, p in cands:
        k = _action_key(p)
        pooled[k] = pooled.get(k, 0.0) + math.exp((s - m) / rr.temp)
        if k not in first or s > first[k][0]:
            first[k] = (s, t, p)
    z = sum(pooled.values())
    best = max(pooled, key=pooled.get)
    _, t, p = first[best]
    return Choice(pred=p, text=t, hyps=hyps, support=pooled[best] / z)


def beam_search(model, feats, tok, prefix: list[int], beams: int = 5, max_new: int = 48, boost=None,
                suppress: list[int] | None = None, begin_suppress: list[int] | None = None) -> list[tuple[list[int], float]]:
    """A small, explicit beam search for Whisper (the transformers 5 beam path returns duplicate beams).

    prefix: decoder start tokens (optional <|startofprev|> prompt, then <|startoftranscript|><|en|><|transcribe|>
    <|notimestamps|>). Ranking uses log p + boost bonus; the returned score is the plain mean log p per token,
    so it stays comparable across hypotheses. Returns [(new token ids without eot, mean log p)] best first.
    """
    import torch

    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    ts_begin = tok.convert_tokens_to_ids("<|notimestamps|>") + 1
    dev = feats.device
    with torch.inference_mode():
        enc = model.model.encoder(feats).last_hidden_state
        V = model.config.vocab_size
        mask = torch.zeros(V, device=dev, dtype=torch.float32)
        mask[ts_begin:] = -float("inf")
        if suppress:
            mask[torch.tensor(suppress, device=dev)] = -float("inf")
        first_mask = mask.clone()
        if begin_suppress:
            first_mask[torch.tensor(begin_suppress, device=dev)] = -float("inf")
        seqs = [list(prefix)]
        raw = torch.zeros(1, device=dev)
        rank = torch.zeros(1, device=dev)
        past = None
        inp = torch.tensor([prefix], device=dev)
        done: list[tuple[list[int], float, float]] = []
        for step in range(max_new):
            out = model.model.decoder(input_ids=inp, encoder_hidden_states=enc.expand(len(seqs), -1, -1),
                                      past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = model.proj_out(out.last_hidden_state[:, -1]).float()
            lp = torch.log_softmax(logits + (first_mask if step == 0 else mask), -1)
            bonus = torch.zeros_like(lp)
            if boost is not None:
                bonus = boost(torch.tensor(seqs, device=dev), torch.zeros_like(lp))
            cand = (rank[:, None] + lp + bonus).view(-1)
            top = torch.topk(cand, min(2 * beams, cand.numel()))
            new_seqs, new_raw, new_rank, keep = [], [], [], []
            for score, idx in zip(top.values.tolist(), top.indices.tolist()):
                b, t = divmod(idx, V)
                r = raw[b].item() + lp[b, t].item()
                if t == eot:
                    n = len(seqs[b]) - len(prefix) + 1
                    done.append((seqs[b][len(prefix):], r / n, score / n))
                else:
                    new_seqs.append(seqs[b] + [t]); new_raw.append(r); new_rank.append(score); keep.append(b)
                if len(new_seqs) == beams:
                    break
            best_alive = max(new_rank) / (len(new_seqs[0]) - len(prefix) + 1) if new_seqs else -float("inf")
            if len(done) >= beams and max(d[2] for d in done) > best_alive or not new_seqs:
                break
            idx = torch.tensor(keep, device=dev)
            past.reorder_cache(idx)
            seqs, raw, rank = new_seqs, torch.tensor(new_raw, device=dev), torch.tensor(new_rank, device=dev)
            inp = torch.tensor([[s[-1]] for s in seqs], device=dev)
        if not done:
            done = [(s[len(prefix):], r / max(1, len(s) - len(prefix)), k) for s, r, k in zip(seqs, raw.tolist(), rank.tolist())]
    done.sort(key=lambda d: -d[2])
    return [(d[0], d[1]) for d in done[:beams]]
