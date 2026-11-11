"""Hinglish text normalization: tokenization, Devanagari->Roman, phonetic keys.

Design goals: deterministic, dependency-free, fast enough for on-device use, and script-invariant
(the same word typed as "kholo", "kolo", "khollo" or "खोलो" should land on the same or nearby keys).

- `deva_to_roman`: rule-based Hinglish-style romanization with schwa deletion. It uses only the
  function-word dictionary from the lexicon (never slot-value pools, which would leak test values).
- `phonetic_key`: collapses common Romanized-Hindi spelling variation (aspiration, long vowels,
  w/v, z/j, c/k, doubled letters, trailing h/n).
- `skeleton`: consonant skeleton of the key (karo/kro -> "kr").
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
TOKEN_RE = re.compile(r"[ऀ-ॿ‌‍]+|[A-Za-z0-9'\-]+|[^\sA-Za-z0-9ऀ-ॿ]")
PUNCT = set("?.!,।;:\"()")

_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh",
    "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d",
    "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
    "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
}
_NUKTA = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "फ": "f", "ड": "d", "ढ": "dh"}
_PRECOMPOSED = {"क़": "क़", "ख़": "ख़", "ग़": "ग़", "ज़": "ज़", "ड़": "ड़",
                "ढ़": "ढ़", "फ़": "फ़"}
_VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e",
           "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e"}
_MATRAS = {"ा": "a", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "े": "e", "ै": "ai",
           "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e"}
_VIRAMA = "्"
_NASAL = {"ं": "n", "ँ": "n"}
_VISARGA = "ः"
_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}


@dataclass
class _Akshara:
    cons: str  # romanized consonant ("" for independent vowel)
    vowel: str | None  # explicit vowel; None = inherent schwa; "" = virama (no vowel)
    nasal: str = ""


def _parse(word: str) -> list[_Akshara]:
    for k, v in _PRECOMPOSED.items():
        word = word.replace(k, v)
    out: list[_Akshara] = []
    i = 0
    while i < len(word):
        ch = word[i]
        if ch in _CONS:
            cons = _CONS[ch]
            if i + 1 < len(word) and word[i + 1] == "़":
                cons = _NUKTA.get(ch, cons)
                i += 1
            out.append(_Akshara(cons, None))
        elif ch in _VOWELS:
            out.append(_Akshara("", _VOWELS[ch]))
        elif ch in _MATRAS and out:
            out[-1].vowel = _MATRAS[ch]
        elif ch == _VIRAMA and out:
            out[-1].vowel = ""
        elif ch in _NASAL and out:
            out[-1].nasal = "n"
        elif ch == _VISARGA and out:
            out[-1].nasal = "h"
        elif ch in _DIGITS:
            out.append(_Akshara(_DIGITS[ch], ""))
        i += 1
    return out


def _translit_word(word: str) -> str:
    aks = _parse(word)
    n = len(aks)
    # Schwa deletion (simplified Hindi rule): drop the final inherent 'a', and a medial inherent 'a'
    # in a VC_CV context when the previous syllable kept its vowel.
    keep = [a.vowel not in (None, "") or a.cons == "" for a in aks]  # explicit vowels always kept
    schwa = [a.vowel is None and a.cons != "" for a in aks]
    deleted = [False] * n
    for i in range(n - 1, -1, -1):  # right to left
        if not schwa[i]:
            continue
        if i == n - 1:
            deleted[i] = n > 1
        elif i > 0 and aks[i].nasal == "":
            next_voiced = not deleted[i + 1] and aks[i + 1].vowel != ""
            prev_voiced = keep[i - 1] or schwa[i - 1]
            deleted[i] = next_voiced and prev_voiced and aks[i + 1].cons != ""
    parts = []
    for i, a in enumerate(aks):
        if a.vowel is None:
            v = "" if deleted[i] else "a"
        else:
            v = a.vowel
        parts.append(a.cons + v + a.nasal)
    return "".join(parts)


class Normalizer:
    """Roman/Devanagari/mixed Hinglish -> Roman tokens + phonetic features."""

    def __init__(self, dev_to_roman: dict[str, str] | None = None):
        # Exact reverse dictionary for known function words (from the lexicon); everything else is
        # transliterated by rule.
        self.dev_to_roman = dev_to_roman or {}

    @classmethod
    def from_config(cls, cfg) -> "Normalizer":
        rev: dict[str, str] = {}
        for roman, dev in cfg.hindi.items():
            rev.setdefault(dev, roman)
        for roman, dev in cfg.english.items():
            if dev:
                rev.setdefault(dev, roman)
        return cls(rev)

    def to_roman(self, word: str) -> str:
        if not DEVANAGARI_RE.search(word):
            return word.lower()
        w = unicodedata.normalize("NFC", word)
        if w in self.dev_to_roman:
            return self.dev_to_roman[w]
        return _translit_word(w)

    def tokenize(self, text: str) -> list[tuple[str, int, int]]:
        """Return [(token, start, end)] with punctuation as separate tokens."""
        return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]

    def words(self, text: str) -> list[tuple[str, int, int]]:
        """Like tokenize but drops punctuation."""
        return [t for t in self.tokenize(text) if t[0] not in PUNCT]

    def roman_words(self, text: str) -> list[str]:
        return [self.to_roman(w) for w, _, _ in self.words(text)]


@lru_cache(maxsize=65536)
def phonetic_key(word: str) -> str:
    w = re.sub(r"[^a-z0-9]", "", word.lower())
    if not w or w.isdigit():
        return w
    w = w.replace("chh", "C").replace("ch", "C")
    w = w.replace("ow", "o")
    for a, b in (("ph", "f"), ("kh", "k"), ("gh", "g"), ("jh", "j"), ("th", "t"), ("dh", "d"),
                 ("bh", "b"), ("sh", "s"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("w", "v"),
                 ("z", "j"), ("c", "k")):
        w = w.replace(a, b)
    w = w.replace("C", "c")
    for a, b in (("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u"), ("aa", "a"), ("ai", "e"),
                 ("ei", "e"), ("ay", "e"), ("au", "o"), ("ou", "u")):
        w = w.replace(a, b)
    w = re.sub(r"(.)\1+", r"\1", w)
    if len(w) > 1 and w.endswith("y"):
        w = w[:-1] + "i"
    if len(w) > 2 and w[-1] in "hn" and w[-2] in "aeiou":
        w = w[:-1]
    elif len(w) > 1 and w.endswith("h"):
        w = w[:-1]
    return w


@lru_cache(maxsize=65536)
def skeleton(word: str) -> str:
    k = phonetic_key(word)
    if not k:
        return k
    return k[0] + re.sub(r"[aeiou]", "", k[1:])
