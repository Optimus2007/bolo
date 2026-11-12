"""Push Bolo's data bundle and notebooks to Kaggle (everything PRIVATE), and fetch results.

  python scripts/kaggle_push.py dataset            # create/update private dataset <user>/bolo-bundle
  python scripts/kaggle_push.py tts [--trial] [--no-gpu]
  python scripts/kaggle_push.py ttsdata          # private dataset bolo-tts-v1 from data/kaggle/tts_v1
  python scripts/kaggle_push.py ft  [--trial]
  python scripts/kaggle_push.py status
  python scripts/kaggle_push.py fetch tts|ft       # download outputs into data/kaggle/out/<kernel>/

Uses the official kaggle CLI (run via `uvx --with six kaggle`) and the credentials already in ~/.kaggle.
Never uploads recordings/ or any human audio. Never sets anything public.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/kaggle"
KAGGLE = ["uvx", "--with", "six", "kaggle"]
STORY = False  # narr: cartoon voices from launch/story.json (--story)
ONLY = None  # narr: comma separated line ids to regenerate (--only)
KERNELS = {"tts": ("bolo-tts-synth", "01_tts_synth.ipynb"), "ft": ("bolo-whisper-ft", "02_whisper_finetune.ipynb"),
           "abl": ("bolo-whisper-ablation", "03_whisper_ablation.ipynb"), "narr": ("bolo-narration", "04_narration.ipynb")}
TRIAL_N = {"tts": ('N = {"train": 7000, "dev": 400, "test": 1200}', 'N = {"train": 64, "dev": 16, "test": 32}'),
           "ft": ('CFG = dict(epochs=5,', 'CFG = dict(max_steps=20, epochs=1,')}


def kg(*args, check=True) -> str:
    r = subprocess.run(KAGGLE + list(args), capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if check and r.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{out}")
    return out


def user() -> str:
    for f in (Path.home() / ".kaggle/kaggle.json", Path.home() / ".kaggle/credentials.json"):
        if f.exists():
            d = json.loads(f.read_text())
            for k in ("username", "user", "userName"):
                if d.get(k):
                    return d[k]
    line = kg("datasets", "list", "--mine").splitlines()[2]
    return line.split("/")[0]


def dataset():
    subprocess.run([sys.executable, str(ROOT / "scripts/make_kaggle_bundle.py")], check=True)
    d = WORK / "upload_bundle"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copy(WORK / "bolo-bundle.zip", d / "bolo-bundle.zip")
    u = user()
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": "bolo-bundle", "id": f"{u}/bolo-bundle", "licenses": [{"name": "apache-2.0"}]}))
    exists = "bolo-bundle" in kg("datasets", "list", "--mine")
    if exists:
        print(kg("datasets", "version", "-p", str(d), "-m", "update", "-r", "zip"))
    else:  # private is the default; never pass --public
        print(kg("datasets", "create", "-p", str(d), "-r", "zip"))


def tts_dataset(src: Path = ROOT / "data/kaggle/tts_v1"):
    """Upload the synthetic clips (computer voices only, never recordings/) as a private dataset."""
    assert (src / "bolo-tts" / "manifest.jsonl").exists(), "run scripts/make_tts_dataset.py first"
    assert "recordings" not in str(src.resolve())
    u = user()
    (src / "dataset-metadata.json").write_text(json.dumps(
        {"title": "bolo-tts-v1", "id": f"{u}/bolo-tts-v1", "licenses": [{"name": "apache-2.0"}]}))
    if "bolo-tts-v1" in kg("datasets", "list", "--mine"):
        print(kg("datasets", "version", "-p", str(src), "-m", "update", "-r", "zip"))
    else:  # private is the default; never pass --public
        print(kg("datasets", "create", "-p", str(src), "-r", "zip"))


def kernel(which: str, trial: bool, gpu: bool):
    subprocess.run([sys.executable, str(ROOT / "scripts/make_notebooks.py")], check=True)
    if which == "narr":
        subprocess.run([sys.executable, str(ROOT / "scripts/make_narration_notebook.py")] + (["--story"] if STORY else []) + ([ONLY] if ONLY else []), check=True)
    slug, nbname = KERNELS[which]
    u = user()
    d = WORK / f"kernel_{slug}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    book = json.loads((ROOT / "notebooks" / nbname).read_text())
    if trial:
        a, b = TRIAL_N[which]
        hits = 0
        for c in book["cells"]:
            new = [line.replace(a, b) for line in c["source"]]
            hits += new != c["source"]
            c["source"] = new
        assert hits, f"trial marker not found in {nbname}"
    (d / nbname).write_text(json.dumps(book, indent=1, ensure_ascii=False))
    meta = {"id": f"{u}/{slug}", "title": slug, "code_file": nbname, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
            "dataset_sources": [f"{u}/bolo-bundle"], "kernel_sources": [], "competition_sources": []}
    if which in ("ft", "abl"):
        meta["dataset_sources"].append(f"{u}/bolo-tts-v1")
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(kg("kernels", "push", "-p", str(d)))


def status():
    u = user()
    for slug, _ in KERNELS.values():
        print(slug, "->", kg("kernels", "status", f"{u}/{slug}", check=False).splitlines()[-1])


def fetch(which: str):
    slug = KERNELS[which][0]
    out = WORK / "out" / slug
    out.mkdir(parents=True, exist_ok=True)
    print(kg("kernels", "output", f"{user()}/{slug}", "-p", str(out)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dataset", "ttsdata", "tts", "ft", "abl", "narr", "status", "fetch"])
    ap.add_argument("which", nargs="?")
    ap.add_argument("--trial", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--only", help="narr: comma separated line ids")
    ap.add_argument("--story", action="store_true", help="narr: the cartoon's voices")
    a = ap.parse_args()
    ONLY, STORY = a.only, a.story
    if a.cmd == "dataset":
        dataset()
    elif a.cmd == "ttsdata":
        tts_dataset()
    elif a.cmd in ("tts", "ft", "abl", "narr"):
        kernel(a.cmd, a.trial, not a.no_gpu)
    elif a.cmd == "status":
        status()
    else:
        fetch(a.which)
