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


def _cmd_name(args: argparse.Namespace) -> None:
    from .assistant import DEFAULT_NAME, get_name, set_name, settings_path

    if args.reset or args.new:
        try:
            name = set_name(None if args.reset else " ".join(args.new))
        except ValueError as e:
            raise SystemExit(f"bolo name: {e}")
        print(f"The assistant is now called {name}. Say \"{name}, volume kam karo\" or just \"volume kam karo\".")
        print(f"(saved in {settings_path()})")
    else:
        name = get_name()
        print(name + ("" if name != DEFAULT_NAME else "  (default; change it with `bolo name <new name>`)"))


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


def _countdown(seconds: int, text: str, name: str, speak_on: bool, os_name: str) -> str:
    """`bolo do --run` timers: count down in this terminal (Ctrl+C cancels). The Myna app runs them in the background."""
    import time

    from .persona import mascot, speak

    end = time.time() + seconds
    try:
        while (left := end - time.time()) > 0:
            print(f"\r   ⏳ {int(left) // 60:02d}:{int(left) % 60:02d}  (Ctrl+C to cancel) ", end="", flush=True)
            time.sleep(min(1.0, left))
    except KeyboardInterrupt:
        print()
        return "timer cancelled"
    print("\a")
    msg = "Time ho gaya!" if text == "Timer done" else f"Yaad hai? {text}"
    print(mascot("happy", msg, name))
    if speak_on:
        speak(msg, os_name, wait=True)
    return "timer finished"


def _cmd_do(args: argparse.Namespace) -> None:
    from .actions import current_os, perform, plan, scan_system
    from .persona import mascot, outcome, reply, speak
    from .pipeline import Bolo
    from .routines import get_routines, match

    here = current_os()
    os_name = args.os or here
    if args.run and os_name != here:
        raise SystemExit(f"--run only works for this computer ({here}); drop --os to run, or drop --run to preview")
    env = scan_system() if os_name == here else None  # another OS's files are not on this machine
    b = Bolo.load(asr=args.asr if args.audio else None, env=env, prompt="commands+apps" if env else "commands")
    out = b.listen(args.input) if args.audio else b.understand(args.input)
    talk = args.speak and args.run
    color = False if args.plain else None

    def show(mood, text):
        print(mascot(mood, text, b.name, color=color))
        if talk:
            speak(text, os_name)

    sched = (lambda sec, text: _countdown(sec, text, b.name, talk, os_name)) if args.run else None
    confirm = (lambda q: True) if args.yes else None
    routine = match(b._strip(out["transcript"])[0]) if out["transcript"] else None
    if routine:
        steps = get_routines()[routine]
        print(f"heard: {out['transcript']!r} -> routine {routine!r} ({len(steps)} steps)")
        show("happy" if args.run else "thinking", f"{routine.title()} routine" + (" shuru!" if args.run else ", --run lagao toh chalu karoon."))
        for st in steps:
            o = b.understand(st["text"])
            if o["intent"] == "none":
                print(f"   skipped {st['text']!r}: {o.get('reason')}")
                continue
            p = plan(o, env, os_name)
            perform(p, execute=args.run, confirm=confirm, schedule=sched, log=lambda m: print("   " + m))
        return
    print(f"heard: {out['transcript']!r} -> {out['intent']} {json.dumps(out['slots'], ensure_ascii=False)}")
    if out["intent"] == "none":
        show(*reply(None, heard=out))
        print(f"   (not doing anything: {out.get('reason')})")
        return
    p = plan(out, env, os_name)
    if not args.run or not p.get("ok", True):
        show(*reply(p, done=False))
        perform(p, execute=False, log=lambda m: print("   " + m))
        return
    timer = p["intent"] in ("set_timer", "set_reminder")
    if timer:
        show(*reply(p, done=True))
    said = perform(p, execute=True, confirm=confirm, schedule=sched, log=lambda m: None)
    if said == ["cancelled"]:
        show("thinking", "Theek hai, nahi kiya.")
        return
    mood, text = outcome(p, said)
    if not timer:
        show(mood, text)
    if p["intent"] not in ("check_status",) and not text.startswith("Hmm"):
        for x in said:
            print("   " + x)
    if p.get("partial"):
        print(f"   note: {p['partial']}")


def _cmd_sikhao(args: argparse.Namespace) -> None:
    from .persona import mascot
    from .pipeline import Bolo
    from .routines import forget, get_routines, teach

    if args.list or (not args.name and not args.forget):
        r = get_routines()
        if not r:
            print('No routines yet. Teach one: bolo sikhao "coding mode" "vs code kholo" "dnd on karo"')
        for n, steps in r.items():
            print(f"{n}:  " + "  ->  ".join(s["text"] for s in steps))
        return
    if args.forget:
        print("forgotten" if forget(args.forget) else f"no routine called {args.forget!r}")
        return
    if not args.steps:
        raise SystemExit("bolo sikhao: give the routine's name, then each command in quotes")
    b = Bolo.load()
    steps = []
    for t in args.steps:
        o = b.understand(t)
        print(f"   {t!r} -> {o['intent']} {json.dumps(o['slots'], ensure_ascii=False)}")
        steps.append({"text": t, "intent": o["intent"], "slots": o["slots"]})
    try:
        n = teach(args.name, steps)
    except ValueError as e:
        raise SystemExit(f"bolo sikhao: {e}")
    print(mascot("happy", f"Seekh liya! Ab bolo \"{n} chalu karo\".", b.name))


def _cmd_app(args: argparse.Namespace) -> None:
    from .app.server import serve

    serve(asr=args.asr, port=args.port, allow_run=args.run, open_browser=not args.no_browser)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="bolo", description="Hinglish voice-command toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate the synthetic text dataset")
    g.add_argument("--out", default="data/text/v1")
    g.add_argument("--seed", type=int, default=13)
    g.add_argument("--train", type=int, default=9000)
    g.add_argument("--dev", type=int, default=1200)
    g.add_argument("--test", type=int, default=2000)
    g.add_argument("--sample-out", default=None, help="also write a small sample JSONL (safe to commit)")
    g.add_argument("--sample-n", type=int, default=40, help="rows per split in the sample")
    g.set_defaults(func=_cmd_generate)

    s = sub.add_parser("stats", help="print dataset statistics")
    s.add_argument("--data", default="data/text/v1")
    s.set_defaults(func=_cmd_stats)

    r = sub.add_parser("record", help="local web tool to record the human-spoken eval set (consent-first)")
    r.add_argument("--root", default="recordings", help="where audio + manifest are stored (gitignored)")
    r.add_argument("--test", default="data/text/v1/test.jsonl", help="test split the prompts are drawn from")
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
    e.add_argument("--test", default="data/text/v1/test.jsonl")
    e.add_argument("--asr", default="whisper-base,swift,bolo-whisper",
                   help="comma list from: whisper-base, whisper-small, swift, bolo-whisper")
    e.add_argument("--nlu", default="rules,tiny", help="comma list from: rules, tiny, muril")
    e.add_argument("--limit", type=int, default=0, help="max utterances per set (0 = all)")
    e.add_argument("--out", default="results/speech/benchmark.json")
    e.set_defaults(func=_cmd_eval)

    u = sub.add_parser("understand", help="text (or a .wav with --audio) -> intent + slots (never executes anything)")
    u.add_argument("input", help="a Hinglish command, or a path to a 16-bit WAV with --audio")
    u.add_argument("--audio", action="store_true")
    default_asr = next((m for m in ("models/bolo-whisper-v1-fp16", "models/bolo-whisper-fp16", "models/bolo-whisper") if Path(m).exists()),
                       "Oriserve/Whisper-Hindi2Hinglish-Swift")
    u.add_argument("--asr", default=default_asr, help="Whisper model id or local dir")
    u.add_argument("--ground", action="store_true",
                   help="match folders/files/apps against what exists on this computer (read-only scan of home and installed apps)")
    u.add_argument("--no-prompt", action="store_true", help="do not give the recogniser a vocabulary hint")
    u.add_argument("--beams", type=int, default=0, help="1 = greedy; >1 = beam search + rerank by meaning (default: built in)")
    u.set_defaults(func=_cmd_understand)

    d = sub.add_parser("do", help="understand a command and do it on this computer (preview only unless --run)")
    d.add_argument("input", help="a Hinglish command, or a path to a 16-bit WAV with --audio")
    d.add_argument("--audio", action="store_true")
    d.add_argument("--asr", default=default_asr, help="Whisper model id or local dir")
    d.add_argument("--run", action="store_true", help="actually do it (default: only show what would happen)")
    d.add_argument("--yes", action="store_true", help="do not ask before closing apps or sleeping")
    d.add_argument("--os", choices=["windows", "linux", "mac"], help="preview the plan for another OS")
    d.add_argument("--speak", action="store_true", help="Myna says the reply out loud (with --run; offline system voice)")
    d.add_argument("--plain", action="store_true", help="no colours")
    d.set_defaults(func=_cmd_do)

    ap = sub.add_parser("app", help="Myna on your screen: talk to it in the browser (local only)")
    ap.add_argument("--run", action="store_true", help="let Myna really do things (default: preview)")
    ap.add_argument("--asr", default=default_asr, help="Whisper model id or local dir")
    ap.add_argument("--port", type=int, default=0, help="0 = any free port")
    ap.add_argument("--no-browser", action="store_true")
    ap.set_defaults(func=_cmd_app)

    sk = sub.add_parser("sikhao", help="teach Myna a routine: one phrase that does several commands")
    sk.add_argument("name", nargs="?", help='the phrase, e.g. "coding mode"')
    sk.add_argument("steps", nargs="*", help='each command in quotes, e.g. "vs code kholo" "dnd on karo"')
    sk.add_argument("--list", action="store_true")
    sk.add_argument("--forget", metavar="NAME")
    sk.set_defaults(func=_cmd_sikhao)

    nm = sub.add_parser("name", help="show or change what you call the assistant (default: Myna)")
    nm.add_argument("new", nargs="*", help="new name, e.g. Jarvis")
    nm.add_argument("--reset", action="store_true", help="back to the default name")
    nm.set_defaults(func=_cmd_name)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
