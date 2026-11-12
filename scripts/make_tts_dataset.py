"""Package downloaded TTS clips (a finished or partial run of notebook 01) as a Kaggle dataset folder with a manifest.

The cancelled v1 run left audio but no manifest; the jobs are deterministic (bolo.audio.tts.plan, seed 0, same text
rows as the bundle), so the manifest is rebuilt from the plan for the clips that exist. Synthetic speech only.

  .venv/bin/python scripts/make_tts_dataset.py --src data/kaggle/out/bolo-tts-synth_check/bolo-tts --out data/kaggle/tts_v1
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bolo.audio import tts  # noqa: E402

N = {"train": 7000, "dev": 400, "test": 1200}  # as in notebook 01
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--data", default="data/text/v1")
a = ap.parse_args()

src, out = Path(a.src), Path(a.out) / "bolo-tts"
(out / "audio").mkdir(parents=True, exist_ok=True)
jobs = []
for split, n in N.items():
    rows = [json.loads(l) for l in open(ROOT / a.data / f"{split}.jsonl", encoding="utf-8")][:n]
    jobs += tts.plan(rows, split, seed=0)
man, missing = [], 0
for j in jobs:
    w = src / "audio" / f"{j['utt_id']}.wav"
    if not w.exists():
        missing += 1
        continue
    shutil.copy2(w, out / "audio" / w.name)
    man.append({**j, "audio": f"audio/{w.name}", "tts_model": "ai4bharat/indic-parler-tts",
                "note": "manifest rebuilt from tts.plan(seed=0); run was cancelled before writing one"})
(out / "manifest.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in man) + "\n")
by = {s: sum(m["split"] == s for m in man) for s in N}
spk = {s: sorted({m["speaker"] for m in man if m["split"] == s}) for s in N}
print(f"{len(man)} clips -> {out} {by}; not generated: {missing}")
print("speakers:", {s: len(v) for s, v in spk.items()}, "train/test overlap:", set(spk["train"]) & set(spk["test"]))
