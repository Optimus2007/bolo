"""ASR metrics for Roman-script Hinglish (no jiwer dependency).

Hinglish has no standard Roman spelling ("karo"/"karoo", "kholo"/"kolo"), so besides plain WER/CER we
report a *phonetic WER*: words are compared after `phonetic_key`, which collapses common spelling
variants. Hypotheses in Devanagari (e.g. from plain Whisper with language=hi) are transliterated to
Roman first, so every model is scored in the same space. Report all three; never only the kindest.
"""

from __future__ import annotations

import re
import unicodedata

from ..normalize import Normalizer, phonetic_key

_PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def norm_words(text: str, normalizer: Normalizer | None = None) -> list[str]:
    t = unicodedata.normalize("NFC", text or "").lower().replace("।", " ")  # danda
    nz = normalizer or Normalizer()
    return [w for w in (_PUNCT.sub("", x) for x in nz.roman_words(t)) if w]


def corpus_scores(refs: list[str], hyps: list[str], normalizer: Normalizer | None = None) -> dict:
    we = wn = ce = cn = pe = 0
    exact = 0
    for r, h in zip(refs, hyps, strict=True):
        rw, hw = norm_words(r, normalizer), norm_words(h, normalizer)
        we += edit_distance(rw, hw)
        wn += len(rw)
        rc, hc = list(" ".join(rw)), list(" ".join(hw))
        ce += edit_distance(rc, hc)
        cn += len(rc)
        pe += edit_distance([phonetic_key(w) for w in rw], [phonetic_key(w) for w in hw])
        exact += rw == hw
    n = max(1, len(refs))
    return {"wer": round(we / max(1, wn), 4), "cer": round(ce / max(1, cn), 4),
            "phonetic_wer": round(pe / max(1, wn), 4), "sentence_exact": round(exact / n, 4), "n": len(refs)}
