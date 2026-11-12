"""Text-level evaluation for Bolo intent+slot predictors.

A predictor is any object with `predict(texts: list[str]) -> list[dict]` returning
{"intent": str, "slots": dict, "confidence": float}. `finalize` applies the shared abstention policy:
schema-invalid outputs and low-confidence outputs become `none`.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from .normalize import phonetic_key
from .schema import LabelError, validate_label


def load_rows(path: Path | str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def finalize(pred: dict, schema: dict, threshold: float = 0.0) -> dict:
    out = {"intent": pred.get("intent", "none"), "slots": dict(pred.get("slots") or {}),
           "confidence": float(pred.get("confidence", 1.0)), "abstain_reason": None}
    if out["intent"] == "none":
        out["slots"] = {}
        return out
    try:
        validate_label(schema, out["intent"], out["slots"])
    except LabelError as e:
        out.update(intent="none", slots={}, abstain_reason=f"invalid: {e}")
        return out
    if out["confidence"] < threshold:
        out.update(intent="none", slots={}, abstain_reason="low_confidence")
    return out


def _norm(v) -> str:
    return str(v).strip().lower()


def _lenient(v) -> str:
    return " ".join(phonetic_key(w) for w in _norm(v).split())


def _pairs(slots: dict, lenient: bool) -> set:
    f = _lenient if lenient else _norm
    return {(k, f(v)) for k, v in slots.items()}


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"p": round(p, 4), "r": round(r, 4), "f1": round(f, 4)}


def _exact(gold: dict, pred: dict, lenient: bool = False) -> bool:
    return gold["intent"] == pred["intent"] and _pairs(gold["slots"], lenient) == _pairs(pred["slots"], lenient)


def _mix_bucket(x):
    if x is None:
        return "n/a"
    return "hi-heavy" if x < 0.34 else ("balanced" if x < 0.67 else "en-heavy")


def score(rows: list[dict], preds: list[dict]) -> dict:
    assert len(rows) == len(preds)
    n = len(rows)
    intent_ok = sum(r["intent"] == p["intent"] for r, p in zip(rows, preds))
    res = {"n": n, "intent_acc": round(intent_ok / n, 4)}
    for lenient in (False, True):
        tp = fp = fn = 0
        for r, p in zip(rows, preds):
            g, q = _pairs(r["slots"], lenient), _pairs(p["slots"], lenient)
            tp += len(g & q)
            fp += len(q - g)
            fn += len(g - q)
        res["slot_lenient" if lenient else "slot"] = _prf(tp, fp, fn)
    res["exact_match"] = round(sum(_exact(r, p) for r, p in zip(rows, preds)) / n, 4)
    res["exact_match_lenient"] = round(sum(_exact(r, p, True) for r, p in zip(rows, preds)) / n, 4)

    gold_none = [r["intent"] == "none" for r in rows]
    pred_none = [p["intent"] == "none" for p in preds]
    tp = sum(g and q for g, q in zip(gold_none, pred_none))
    fp = sum((not g) and q for g, q in zip(gold_none, pred_none))
    fn = sum(g and not q for g, q in zip(gold_none, pred_none))
    res["abstain"] = _prf(tp, fp, fn)
    n_none = sum(gold_none)
    res["false_action_rate"] = round(fn / n_none, 4) if n_none else None  # acted when it should not
    res["over_abstain_rate"] = round(fp / (n - n_none), 4) if n - n_none else None
    # wrong action = acted, but the action is not exactly right (includes acting on a none)
    acted = [(r, p) for r, p in zip(rows, preds) if p["intent"] != "none"]
    res["wrong_action_rate"] = round(sum(not _exact(r, p) for r, p in acted) / len(acted), 4) if acted else None

    # risk-coverage over confidence (correct = exact match, incl. correctly abstaining)
    order = sorted(range(n), key=lambda i: -preds[i]["confidence"])
    errs, aurc = 0, 0.0
    for k, i in enumerate(order, 1):
        errs += not _exact(rows[i], preds[i])
        aurc += errs / k
    res["aurc"] = round(aurc / n, 4)

    # breakdowns
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        m = r.get("meta", {})
        groups["script"][r.get("script", "?")].append(i)
        groups["template"]["heldout" if m.get("template_heldout") else "seen"].append(i)
        groups["text_seen_in_train"][str(bool(m.get("seen_in_train")))].append(i)
        groups["mixing"][_mix_bucket(m.get("en_ratio"))].append(i)
        if r["intent"] == "none":
            groups["none_reason"][m.get("reason") or "?"].append(i)
    bd = {}
    for gname, buckets in groups.items():
        bd[gname] = {}
        for b, idx in sorted(buckets.items()):
            em = sum(_exact(rows[i], preds[i]) for i in idx) / len(idx)
            bd[gname][b] = {"n": len(idx), "exact_match": round(em, 4)}
            if gname == "none_reason":
                bd[gname][b]["abstain_recall"] = round(sum(preds[i]["intent"] == "none" for i in idx) / len(idx), 4)
    res["breakdown"] = bd
    return res


def consistency(rows: list[dict], main_preds: list[dict], view_preds: list[list[dict]]) -> float:
    """Fraction of utterances where the model gives the same (lenient) action for every script view."""
    same = 0
    for p, vps in zip(main_preds, view_preds):
        key = (p["intent"], frozenset(_pairs(p["slots"], True)))
        same += all((v["intent"], frozenset(_pairs(v["slots"], True))) == key for v in vps)
    return round(same / len(rows), 4)


MAX_FAR = 0.03  # acting on something that is not a command is worse than asking again


def choose_threshold(rows: list[dict], raw_preds: list[dict], schema: dict, max_far: float | None = MAX_FAR) -> float:
    """Pick the confidence threshold with the best exact match on (dev) rows among those whose false action rate
    (acting on a `none` row) is at most `max_far`; if none qualifies, the one with the lowest false action rate.
    max_far=None: plain best exact match (the old rule; it often picked 0, i.e. never abstain on confidence)."""
    none_idx = [i for i, r in enumerate(rows) if r["intent"] == "none"]
    cands = []
    for t in [x / 100 for x in range(0, 100, 2)]:
        preds = [finalize(p, schema, t) for p in raw_preds]
        em = sum(_exact(r, p) for r, p in zip(rows, preds))
        far = sum(preds[i]["intent"] != "none" for i in none_idx) / max(1, len(none_idx))
        cands.append((t, em, far))
    ok = [c for c in cands if max_far is None or c[2] <= max_far]
    if ok:
        return max(ok, key=lambda c: (c[1], -c[0]))[0]
    return min(cands, key=lambda c: (c[2], -c[1]))[0]


def latency_ms(predictor, texts: list[str], n: int = 200) -> dict:
    texts = (texts * (n // max(len(texts), 1) + 1))[:n]
    predictor.predict(texts[:5])  # warm-up
    times = []
    for t in texts:
        s = time.perf_counter()
        predictor.predict([t])
        times.append((time.perf_counter() - s) * 1000)
    times.sort()
    return {"p50": round(statistics.median(times), 3), "p95": round(times[int(0.95 * len(times)) - 1], 3)}


def run(predictor, schema: dict, test_rows: list[dict], dev_rows: list[dict] | None = None,
        threshold: float | None = None, with_views: bool = True, timing: bool = True) -> dict:
    if threshold is None:
        threshold = 0.0
        if dev_rows is not None:
            dev_raw = predictor.predict([r["text"] for r in dev_rows])
            threshold = choose_threshold(dev_rows, dev_raw, schema)
    raw = predictor.predict([r["text"] for r in test_rows])
    preds = [finalize(p, schema, threshold) for p in raw]
    out = {"threshold": threshold, "test": score(test_rows, preds)}
    unseen = [i for i, r in enumerate(test_rows) if not r["meta"].get("seen_in_train")]
    out["test_unseen_text"] = {k: v for k, v in score([test_rows[i] for i in unseen],
                                                       [preds[i] for i in unseen]).items() if k != "breakdown"}
    if with_views and all("views" in r for r in test_rows):
        flat = [v["text"] for r in test_rows for v in r["views"]]
        vraw = predictor.predict(flat)
        vp = [finalize(p, schema, threshold) for p in vraw]
        k = len(test_rows[0]["views"])
        view_preds = [vp[i * k:(i + 1) * k] for i in range(len(test_rows))]
        out["script_consistency"] = consistency(test_rows, preds, view_preds)
        view_rows = [dict(r, script=v["script"], text=v["text"]) for r in test_rows for v in r["views"]]
        out["views_exact_match"] = score(view_rows, vp)["exact_match"]
    if timing:
        out["latency_ms"] = latency_ms(predictor, [r["text"] for r in test_rows[:200]])
    return out
