"""Paraphrase generated commands with a local instruction-tuned LLM (MLX), keeping labels.

Labels are preserved by construction: the LLM is told to keep every slot phrase verbatim, and any
paraphrase that does not contain all slot surfaces exactly is dropped. Slot spans are re-located in
the paraphrase. Sources are Roman-script rows from ONE split, so slot values never cross splits.

  HF_HOME=models/hf_cache python scripts/paraphrase.py --split train --n 1200 --k 3
  HF_HOME=models/hf_cache python scripts/paraphrase.py --split test --n 300 --k 1
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from bolo.config import load_config
from bolo.evaluate import load_rows

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
ap.add_argument("--split", default="train")
ap.add_argument("--n", type=int, default=1200, help="number of source rows")
ap.add_argument("--k", type=int, default=3, help="paraphrases requested per source")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", default="data/text/v0")
args = ap.parse_args()

SYSTEM = ("You rewrite short voice commands that young people in India say to a laptop assistant, in "
          "Hinglish (Hindi-English code-mixing written in Roman script). Be natural and varied: change word "
          "order, verbs, politeness, fillers and the amount of English vs Hindi. Never change the meaning.")
USER = ("Command: \"{text}\"\n"
        "{keep}"
        "Write {k} different natural ways to say this command. One per line, no numbering, no quotes, "
        "no explanations.")
NEG_USER = ("Command: \"{text}\"\n"
            "This command is unclear, incomplete, or something the assistant cannot do. Write {k} different "
            "natural Hinglish ways to say the same thing, keeping it equally unclear or out of scope. "
            "One per line, no numbering, no quotes, no explanations.")

cfg = load_config()
rows = [r for r in load_rows(f"{args.data}/{args.split}.jsonl") if r["script"] == "roman"]
rng = random.Random(args.seed)
by_intent = defaultdict(list)
for r in rows:
    by_intent[r["intent"]].append(r)
per = args.n // len(by_intent)
sources = []
for intent, rs in sorted(by_intent.items()):
    rng.shuffle(rs)
    sources += rs[:per]
rng.shuffle(sources)


def surfaces(r):
    return [(s["slot"], r["text"][s["start"]:s["end"]].lower(), s["value"]) for s in r["spans"]]


def prompt_for(tok, r):
    text = re.sub(r"[?.!।]+$", "", r["text"]).lower()
    if r["intent"] == "none":
        user = NEG_USER.format(text=text, k=args.k)
    else:
        keep = ", ".join(f'"{s}"' for _, s, _ in surfaces(r))
        user = USER.format(text=text, k=args.k,
                           keep=f"Keep these words exactly as written: {keep}.\n" if keep else "")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True)


LINE_JUNK = re.compile(r"^\s*(\d+[.)]|[-*•])\s*")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def clean_lines(text):
    out = []
    for line in text.split("\n"):
        line = LINE_JUNK.sub("", line).strip().strip('"').strip("'").strip()
        if not line or ":" in line or len(line.split()) > 25 or len(line) < 3:
            continue
        if re.search(r"[^\w\sऀ-ॿ?!.,'\-]", line):
            continue
        out.append(line)
    return out


from mlx_lm import batch_generate, load  # noqa: E402

model, tok = load(args.model)
out_rows, seen = [], set()
stats = defaultdict(int)
t0 = time.time()
for i in range(0, len(sources), args.batch):
    chunk = sources[i:i + args.batch]
    prompts = [prompt_for(tok, r) for r in chunk]
    resp = batch_generate(model, tok, prompts, max_tokens=40 * args.k + 20)
    for r, text in zip(chunk, resp.texts):
        src_norm = re.sub(r"[^\w\s]", "", r["text"].lower()).strip()
        for line in clean_lines(text)[: args.k]:
            stats["generated"] += 1
            low = line.lower()
            key = re.sub(r"[^\w\s]", "", low).strip()
            if key == src_norm or key in seen:
                stats["dup"] += 1
                continue
            spans, ok = [], True
            for slot, surf, value in surfaces(r):
                m = re.search(rf"(?<!\w){re.escape(surf)}(?!\w)", low)
                if not m:
                    ok = False
                    break
                spans.append({"slot": slot, "start": m.start(), "end": m.end(), "value": value})
            if not ok:
                stats["slot_lost"] += 1
                continue
            seen.add(key)
            stats["kept"] += 1
            out_rows.append({
                "id": f"para-{args.split}-{len(out_rows):05d}", "split": args.split, "text": line,
                "script": "mixed" if DEVANAGARI.search(line) else "roman", "intent": r["intent"],
                "slots": r["slots"], "spans": spans, "views": [],
                "meta": {"source_id": r["id"], "paraphrase": True, "reason": r["meta"]["reason"],
                         "template_id": r["meta"]["template_id"], "template_heldout": r["meta"]["template_heldout"],
                         "en_ratio": None, "seen_in_train": False, "fillers": []},
            })
    print(f"{i + len(chunk)}/{len(sources)} sources, kept {stats['kept']} ({time.time() - t0:.0f}s)", flush=True)

out = Path(f"{args.data}/para_{args.split}.jsonl")
with open(out, "w", encoding="utf-8") as f:
    for r in out_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
meta = {"model": args.model, "split": args.split, "sources": len(sources), "k": args.k, **stats,
        "seconds": round(time.time() - t0)}
Path(f"{args.data}/para_{args.split}.manifest.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta))
