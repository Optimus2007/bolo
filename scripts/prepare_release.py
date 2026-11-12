"""Stage (do NOT publish) the release artifacts into data/release/ (gitignored).

  data/release/bolo-sun/      Hugging Face model repo: float16 weights + processor + README (model card with metadata)
  data/release/bolo-tiny/     Hugging Face model repo: safetensors + config + int8 ONNX + README
  data/release/bolocmd-v0/    Hugging Face dataset repo: synthetic text train/dev/test + README (datasheet)

Only release-safe things are copied: no human recordings, no audio, nothing derived from the Qwen Research
licensed paraphrases. Uploading is a manual decision for the owner (see docs/RELEASE_CHECKLIST.md).
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/release"
FORBIDDEN = ("recordings", "para_", "_kd", ".wav")


def safe(p: Path):
    assert not any(f in str(p) for f in FORBIDDEN), f"refusing to stage {p}"
    return p


def card(meta: dict, body_path: Path) -> str:
    lines = ["---"]
    for k, v in meta.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            lines += [f"  - {x}" for x in v]
        else:
            lines.append(f"{k}: {v}")
    lines.append("---\n")
    return "\n".join(lines) + body_path.read_text(encoding="utf-8")


def bolo_sun():
    src = ROOT / "models/bolo-whisper-v1-fp16"
    if not src.exists():
        print("skip bolo-sun (models/bolo-whisper-v1-fp16 missing)")
        return
    d = OUT / "bolo-sun"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for f in src.iterdir():
        if f.name != "train_log.json":
            shutil.copy(safe(f), d / f.name)
    (d / "README.md").write_text(card({
        "license": "apache-2.0", "language": ["hi", "en"], "base_model": "Oriserve/Whisper-Hindi2Hinglish-Swift",
        "pipeline_tag": "automatic-speech-recognition", "library_name": "transformers",
        "tags": ["whisper", "hinglish", "code-switching", "voice-commands", "on-device"],
    }, ROOT / "docs/MODEL_CARD_bolo_sun.md"))


def bolo_tiny():
    import torch
    from safetensors.torch import save_file

    src = ROOT / "models/bolo-tiny/phono_views_aug_asr-s0"
    d = OUT / "bolo-tiny"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    sd = torch.load(safe(src / "model.pt"), weights_only=True)
    save_file({k: v.contiguous() for k, v in sd.items()}, d / "model.safetensors")
    shutil.copy(src / "config.json", d / "config.json")
    onnx = ROOT / "models/export/bolo-tiny/bolo_tiny.int8.onnx"
    if onnx.exists():
        shutil.copy(onnx, d / onnx.name)
    (d / "README.md").write_text(card({
        "license": "apache-2.0", "language": ["hi", "en"], "pipeline_tag": "token-classification",
        "tags": ["hinglish", "intent-classification", "slot-filling", "on-device", "onnx"],
    }, ROOT / "docs/MODEL_CARD_bolo_tiny.md"))


def dataset():
    src = ROOT / "data/text/v1"
    d = OUT / "bolocmd-v0"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "manifest.json"):
        shutil.copy(safe(src / name), d / name)
    (d / "README.md").write_text(card({
        "license": "cc-by-4.0", "language": ["hi", "en"], "task_categories": ["token-classification", "text-classification"],
        "tags": ["hinglish", "code-switching", "voice-commands", "synthetic"], "size_categories": ["1K<n<10K"],
    }, ROOT / "docs/DATASHEET.md"))


if __name__ == "__main__":
    bolo_sun()
    bolo_tiny()
    dataset()
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"{p.stat().st_size / 1e6:8.2f} MB  {p.relative_to(OUT)}")
    print(json.dumps({"staged_in": str(OUT), "published": False}))
