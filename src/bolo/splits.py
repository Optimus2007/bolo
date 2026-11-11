"""Deterministic, value-based splitting.

Open-vocabulary slot values (folders, files, contacts, apps) are assigned to exactly one split by a
stable hash of the value, so test utterances only contain values never seen in train. Adding new
values later never moves existing ones between splits.
"""

from __future__ import annotations

import hashlib

SPLITS = ("train", "dev", "test")


def _bucket(key: str, n: int) -> int:
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % n


def value_split(value: str) -> str:
    b = _bucket("value:" + value, 20)
    if b < 14:
        return "train"
    if b < 17:
        return "dev"
    return "test"


def template_heldout(template_text: str) -> bool:
    """~20% of templates are reserved for test (only applied to intents with enough templates)."""
    return _bucket("template:" + template_text, 5) == 0
