"""Aggregate results/text/*.json (mean ± std over seeds) -> results/text/SUMMARY.md"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

rows = defaultdict(list)
for p in sorted(Path("results/text").glob("*.json")):
    r = json.loads(p.read_text())
    rows[r["model"]].append(r)


def ms(vals, pct=True):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    k = 100 if pct else 1
    m = st.mean(vals) * k
    return f"{m:.1f} ± {st.stdev(vals) * k:.1f}" if len(vals) > 1 else f"{m:.1f}"


cols = [
    ("Exact match", lambda r: r["test"]["exact_match"]),
    ("EM unseen-text", lambda r: r["test_unseen_text"]["exact_match"]),
    ("EM held-out tpl", lambda r: r["test"]["breakdown"]["template"]["heldout"]["exact_match"]),
    ("EM Devanagari", lambda r: r["test"]["breakdown"]["script"]["deva"]["exact_match"]),
    ("Intent acc", lambda r: r["test"]["intent_acc"]),
    ("Slot F1", lambda r: r["test"]["slot"]["f1"]),
    ("Slot F1 lenient", lambda r: r["test"]["slot_lenient"]["f1"]),
    ("Abstain F1", lambda r: r["test"]["abstain"]["f1"]),
    ("False-action rate ↓", lambda r: r["test"]["false_action_rate"]),
    ("AURC ↓", lambda r: r["test"]["aurc"]),
    ("Script consistency", lambda r: r.get("script_consistency")),
]
lines = ["# Text intent+slot results (synthetic test set, v0)", "",
         "Mean ± std over seeds (n = number of seeds). All numbers in %. Test = 1,000 rows with unseen slot",
         "values; 'unseen-text' excludes rows whose exact text occurs in train; 'held-out tpl' = phrasings never",
         "seen in training. Latency = single utterance, CPU, p50.", "",
         "| Model | n | Params | " + " | ".join(c for c, _ in cols) + " | Latency p50 (ms) |",
         "|---" * (len(cols) + 4) + "|"]
order = sorted(rows, key=lambda m: -st.mean(r["test"]["exact_match"] for r in rows[m]))
for m in order:
    rs = rows[m]
    params = f"{rs[0]['params'] / 1e6:.2f}M" if rs[0]["params"] else "–"
    lat = st.mean(r["latency_ms"]["p50"] for r in rs if "latency_ms" in r)
    lines.append(f"| {m} | {len(rs)} | {params} | " + " | ".join(ms([f(r) for r in rs]) for _, f in cols)
                 + f" | {lat:.2f} |")
Path("results/text/SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
