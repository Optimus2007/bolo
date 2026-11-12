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


def _cmd_record(args: argparse.Namespace) -> None:
    from .record.server import serve

    serve(root=args.root, test_path=args.test, port=args.port, open_browser=not args.no_browser,
          share=args.share, minutes=args.minutes, n_read=args.n_read, n_free=args.n_free)


def _cmd_eval(args: argparse.Namespace) -> None:
    from .config import load_config
    from .speech_eval import load_set, run, to_markdown

    sets: dict = {}
    for spec in args.manifest:
        name, _, path = spec.rpartition("=")
        for r in load_set(path, args.test, name or None):
            sets.setdefault(r["set"], []).append(r)
    if args.limit:
        sets = {k: v[: args.limit] for k, v in sets.items()}
    print({k: len(v) for k, v in sets.items()})
    out = Path(args.out)
    res = run(sets, args.asr.split(","), args.nlu.split(","), load_config(),
              utt_out=Path("data/eval") / (out.stem + "_utts.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    md = to_markdown(res)
    out.with_suffix(".md").write_text(md)
    print(md)


def _cmd_understand(args: argparse.Namespace) -> None:
    from .pipeline import Bolo, scan_mac

    env = scan_mac() if args.ground else None
    b = Bolo.load(asr=args.asr if args.audio else None, env=env, prompt=None if args.no_prompt else
                  ("commands+apps" if env else "commands"), **({"beams": args.beams} if args.beams else {}))
    out = b.listen(args.input) if args.audio else b.understand(args.input)
    print(json.dumps(out, ensure_ascii=False, indent=1))


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

    r = sub.add_parser("record", help="local web tool to record the human-spoken eval set (consent-first)")
    r.add_argument("--root", default="recordings", help="where audio + manifest are stored (gitignored)")
    r.add_argument("--test", default="data/text/v0/test.jsonl", help="test split the prompts are drawn from")
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--no-browser", action="store_true")
    r.add_argument("--share", action="store_true",
                   help="share a private HTTPS link (Cloudflare quick tunnel); audio is stored only on this laptop")
    r.add_argument("--minutes", type=int, default=120, help="auto-close the shared link after this many minutes")
    r.add_argument("--n-read", type=int, default=40, help="read prompts per speaker (main part, ~4 min)")
    r.add_argument("--n-free", type=int, default=15, help="optional 'say it your way' bonus prompts per speaker")
    r.set_defaults(func=_cmd_record)

    e = sub.add_parser("eval", help="speech -> text -> action benchmark (ASR + NLU + grounding + latency)")
    e.add_argument("--manifest", nargs="+", default=["recordings/manifest.jsonl"],
                   help="manifest paths, optionally NAME=path (e.g. say=data/say/manifest.jsonl)")
    e.add_argument("--test", default="data/text/v0/test.jsonl")
    e.add_argument("--asr", default="whisper-base,swift,bolo-whisper",
                   help="comma list from: whisper-base, whisper-small, swift, bolo-whisper")
    e.add_argument("--nlu", default="rules,tiny", help="comma list from: rules, tiny, muril")
    e.add_argument("--limit", type=int, default=0, help="max utterances per set (0 = all)")
    e.add_argument("--out", default="results/speech/benchmark.json")
    e.set_defaults(func=_cmd_eval)

    u = sub.add_parser("understand", help="text (or a .wav with --audio) -> intent + slots (never executes anything)")
    u.add_argument("input", help="a Hinglish command, or a path to a 16-bit WAV with --audio")
    u.add_argument("--audio", action="store_true")
    default_asr = next((m for m in ("models/bolo-whisper-fp16", "models/bolo-whisper") if Path(m).exists()),
                       "Oriserve/Whisper-Hindi2Hinglish-Swift")
    u.add_argument("--asr", default=default_asr, help="Whisper model id or local dir")
    u.add_argument("--ground", action="store_true",
                   help="match folders/files/apps against what exists on this Mac (read-only scan of ~ and /Applications)")
    u.add_argument("--no-prompt", action="store_true", help="do not give the recogniser a vocabulary hint")
    u.add_argument("--beams", type=int, default=0, help="1 = greedy; >1 = beam search + rerank by meaning (default: built in)")
    u.set_defaults(func=_cmd_understand)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
