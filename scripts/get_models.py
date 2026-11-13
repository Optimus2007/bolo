"""Download Bolo's trained models from the GitHub release into models/.

    uv run python scripts/get_models.py          # the default stack: Bolo Sun v1 (142 MB) + Bolo Tiny (75 MB zip)
    uv run python scripts/get_models.py --all    # also the earlier and rejected speech models (for reproducing results)

Every file is checked against the SHA-256 below before it is unpacked. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

RELEASE = "https://github.com/Optimus2007/bolo/releases/download/weights-v1"
FILES = {  # name: (sha256, default?, what it is)
    "bolo-whisper-v1-fp16.zip": ("c9357d11395da27d94f2f645999e919e032874e1396c227561eccb4ec6683de2", True,
                                 "Bolo Sun v1, the default speech model"),
    "bolo-tiny.zip": ("f0cc4364492101e4a75fb64fb25a49528744edf7c8ece027ad9c35e2748c1ac5", True,
                      "Bolo Tiny: every seed and ablation checkpoint, plus the 0.8 MB int8 ONNX"),
    "bolo-whisper-fp16.zip": ("f8c54669ccd6302940538221407e9448863f2a8a93c9092ba46671a35d126723", False,
                              "Bolo Sun v0 (9 commands, no room/noise augmentation)"),
    "bolo-whisper-v1e10-fp16.zip": ("88212f57fcc87380013214282b78f7d725f4cad36587757a0c225d7104284703", False,
                                    "Bolo Sun v1 trained 10 epochs: better on synthetic, worse on real speech (rejected)"),
    "bolo-whisper-wise050-fp16.zip": ("0beaf32cdca29e7029a7ba43520bd8169d30c0e95b620d2f9dd58d37114e3fe9", False,
                                      "50/50 weight blend of Swift and Bolo Sun v0 (forgets less, learns less)"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="also download the earlier and rejected models")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    cache = root / "models" / ".downloads"
    cache.mkdir(parents=True, exist_ok=True)
    for name, (digest, default, what) in FILES.items():
        if not (default or a.all):
            continue
        z = cache / name
        if not (z.exists() and sha256(z) == digest):
            print(f"downloading {name}: {what}")
            urllib.request.urlretrieve(f"{RELEASE}/{name}", z)
        if sha256(z) != digest:
            z.unlink()
            print(f"  checksum mismatch for {name}; deleted the download, please try again", file=sys.stderr)
            return 1
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():  # only ever unpack inside models/
                if not m.startswith("models/") or ".." in Path(m).parts:
                    print(f"  refusing unexpected path {m!r} in {name}", file=sys.stderr)
                    return 1
            zf.extractall(root)
        print(f"  ok: {name}")
    print("Models are in models/. Try: uv run bolo app")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
