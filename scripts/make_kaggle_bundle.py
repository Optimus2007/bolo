"""Bundle code + synthetic text data for Kaggle/Colab (the repo is private, so notebooks can't git clone).

Output: data/kaggle/bolo-bundle.zip (gitignored). Upload it as a PRIVATE Kaggle dataset named
`bolo-bundle`. Contains only code, configs and the synthetic text splits -- never recordings/ or audio.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/kaggle/bolo-bundle.zip"
INCLUDE = ["src/bolo", "configs", "data/text/v0/train.jsonl", "data/text/v0/dev.jsonl",
           "data/text/v0/test.jsonl", "data/text/v0/manifest.json", "pyproject.toml", "LICENSE"]
BLOCK = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".opus", ".pyc")

OUT.parent.mkdir(parents=True, exist_ok=True)
n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for inc in INCLUDE:
        p = ROOT / inc
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file()) if p.exists() else []
        for f in files:
            rel = f.relative_to(ROOT)
            if f.suffix in BLOCK or "__pycache__" in rel.parts or "recordings" in rel.parts:
                continue
            z.write(f, Path("bolo-bundle") / rel)
            n += 1
print(f"{OUT} : {n} files, {OUT.stat().st_size / 1e6:.1f} MB")
