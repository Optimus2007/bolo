"""`bolo` command-line interface."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _cmd_generate(args: argparse.Namespace) -> None:
    from .generator import write_dataset

    sizes = {"train": args.train, "dev": args.dev, "test": args.test}
    manifest = write_dataset(args.out, seed=args.seed, sizes=sizes)
    print(json.dumps({"out": str(args.out), "sizes": manifest["sizes"]}, ensure_ascii=False))
    if args.sample_out:
        _write_sample(Path(args.out), Path(args.sample_out), args.sample_n, args.seed)
        print(f"sample -> {args.sample_out}")


def _write_sample(data_dir: Path, sample_path: Path, n: int, seed: int) -> None:
    rng = random.Random(seed)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sample_path, "w", encoding="utf-8") as out:
        for split in ("train", "dev", "test"):
            rows = (data_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            for line in rng.sample(rows, min(n, len(rows))):
                out.write(line + "\n")


def _cmd_stats(args: argparse.Namespace) -> None:
    manifest = json.loads((Path(args.data) / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps(manifest["stats"], ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="bolo", description="Hinglish voice-command toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate the synthetic text dataset")
    g.add_argument("--out", default="data/text/v0")
    g.add_argument("--seed", type=int, default=13)
    g.add_argument("--train", type=int, default=5000)
    g.add_argument("--dev", type=int, default=750)
    g.add_argument("--test", type=int, default=1000)
    g.add_argument("--sample-out", default=None, help="also write a small sample JSONL (safe to commit)")
    g.add_argument("--sample-n", type=int, default=40, help="rows per split in the sample")
    g.set_defaults(func=_cmd_generate)

    s = sub.add_parser("stats", help="print dataset statistics")
    s.add_argument("--data", default="data/text/v0")
    s.set_defaults(func=_cmd_stats)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
