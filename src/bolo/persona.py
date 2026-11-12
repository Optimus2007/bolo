"""Myna's personality: short Hinglish replies, a small terminal mascot with moods, and speaking aloud offline.

    from bolo.persona import reply, mascot, speak_step
    reply(plan_dict, done=True)          # "Chrome khol diya!"
    print(mascot("happy", "Chrome khol diya!"))

Replies are in Roman Hinglish and avoid gendered verb forms ("khol diya", "ho gaya", "nahi mila"), so they fit
whatever name and voice the user picks. Speaking uses the voice already on the computer (macOS `say`, Windows
SAPI, Linux espeak-ng / spd-say): offline, and the text goes in as stdin or an env var, never as code.
"""
from __future__ import annotations

import os
import random
import shutil
import sys

UNIT_HI = {"seconds": "second", "minutes": "minute", "hours": "ghante"}
STATUS_HI = {"battery": "battery", "wifi": "Wi-Fi", "internet": "internet", "storage": "storage", "git": "git",
             "cpu": "CPU", "ram": "RAM", "bluetooth": "Bluetooth"}
SETTING_HI = {"wifi": "Wi-Fi", "bluetooth": "Bluetooth", "do_not_disturb": "Do Not Disturb", "dark_mode": "dark mode"}
MEDIA_HI = {"play": "Gaana chala diya", "pause": "Pause kar diya", "next": "Agla gaana laga diya", "previous": "Pichla gaana laga diya"}
OK_WORDS = ["Ho gaya!", "Done!", "Lo ji!", "Bilkul!"]
UNSURE = ["Samajh nahi aaya, ek baar phir bolo?", "Thoda clear bolo na, phir se?", "Ye wala samjha nahi. Dobara?"]


def _lvl(s: dict, what: str) -> str:
    if s.get("level") is not None:
        return f"{what} {s['level']}% pe kar diya"
    return {"up": f"{what} badha diya", "down": f"{what} kam kar diya", "mute": "Mute kar diya",
            "unmute": "Unmute kar diya", "max": f"{what} full kar diya", "min": f"{what} sabse kam kar diya"
            }.get(str(s.get("direction")), f"{what} badal diya")


def _done(intent: str, s: dict) -> str:
    g = lambda k: str(s.get(k, "")).strip()  # noqa: E731
    return {
        "open_terminal": "Terminal khol diya" + (f", {g('folder')} folder mein" if g("folder") else ""),
        "open_claude_in_folder": "Claude Code chalu" + (f", {g('folder')} mein" if g("folder") else ""),
        "open_app": f"{g('app').title()} khol diya", "close_app": f"{g('app').title()} band kar diya",
        "open_file_or_folder": f"{g('path')} khol diya", "find_file": f"{g('path')} ki khoj ho gayi",
        "set_volume": _lvl(s, "Volume"), "set_brightness": _lvl(s, "Brightness"),
        "media_control": MEDIA_HI.get(g("action"), "Ho gaya"), "lock_screen": "Screen lock kar di",
        "sleep_mac": "Good night! Computer so raha hai", "take_screenshot": "Screenshot le liya",
        "toggle_setting": f"{SETTING_HI.get(g('setting'), g('setting'))} {g('state')} kar diya",
        "check_status": f"{STATUS_HI.get(g('target'), g('target'))} dekh liya",
        "web_search": f"\"{g('query')}\" search kar diya" + (" YouTube pe" if g("engine") == "youtube" else ""),
        "open_website": f"{g('site')} khol diya",
        "set_timer": f"{g('amount')} {UNIT_HI.get(g('unit'), g('unit'))} ka timer laga diya",
        "set_reminder": f"Reminder laga diya: {g('task')}",
        "call_contact": f"{g('name').title()} ko call ke liye app khol diya",
        "send_message": f"{g('name').title()} ka message copy kar diya, app khol diya",
    }.get(intent, "Ho gaya")


def _would(intent: str, s: dict) -> str:
    return _done(intent, s).replace(" khol diya", " kholna hai").replace(" kar diya", " karna hai") \
        .replace(" kar di", " karni hai").replace(" laga diya", " lagana hai").replace(" le liya", " lena hai") \
        .replace(" dekh liya", " dekhna hai").replace(" chala diya", " chalana hai").replace(" ki khoj ho gayi", " dhoondhna hai")


def reply(p: dict | None, done: bool = False, rng: random.Random | None = None, heard: dict | None = None) -> tuple[str, str]:
    """(mood, text) for a plan from bolo.actions.plan. mood: happy | unsure | sleepy | thinking | oops."""
    rng = rng or random.Random()
    if p is None:
        why = (heard or {}).get("reason") or ""
        if "no speech" in why:
            return "sleepy", "Kuch suna hi nahi... mic on hai?"
        if heard and heard.get("grounding") and (heard["grounding"] or {}).get("abstain"):
            return "unsure", "Ye naam is computer pe nahi mila. Thoda aur batao?"
        return "unsure", rng.choice(UNSURE)
    if not p.get("ok", True):
        say = p.get("say", "")
        if "could not find" in say:
            what = say.split("could not find", 1)[1].replace("on this computer", "").strip()
            return "oops", f"Oho, {what} nahi mila is computer pe."
        return "oops", "Ye abhi nahi kar sakte, sorry!"
    s, intent = p.get("slots") or {}, p["intent"]
    if not done:
        w = _would(intent, s)
        return "thinking", w[:1].upper() + w[1:] + ". --run lagao toh kar doon."
    text = _done(intent, s)
    text = text[:1].upper() + text[1:]
    if intent not in ("check_status", "find_file", "sleep_mac"):
        text = rng.choice(OK_WORDS) + " " + text
    return "happy", text


PROBLEM = ("did not work", "not installed", "not found", "not open", "no battery", "could not", "can't")


STATUS_RE = [  # what bolo.actions.status says -> how Myna says it
    (r"^(\d+) GB free of (\d+) GB$", r"\1 GB khaali hai, \2 GB mein se."),
    (r"^internet is working$", "Internet chal raha hai."), (r"^no internet$", "Internet nahi chal raha."),
    (r"^battery (\d+)%$", r"Battery \1% hai."), (r"^CPU (?:about )?(\d+)% busy$", r"CPU \1% busy hai."),
    (r"^memory (\d+)% used$", r"RAM \1% bhari hui hai."), (r"^connected to (.+)$", r"\1 Wi-Fi se connected ho."),
    (r"^not connected to Wi-Fi$", "Wi-Fi se connected nahi ho."), (r"^bluetooth (on|off)$", r"Bluetooth \1 hai."),
    (r"^found (.+)$", r"Mil gaya: \1"), (r"^no (.+) under your home folder$", r"\1 kahin nahi mila."),
    (r"^this folder is not a git repository$", "Ye folder git repo nahi hai."),
    (r"^(.+), (\d+) changed files$", r"Branch \1, \2 files badli hain."),
]


def hinglish_status(x: str) -> str:
    import re

    for pat, rep in STATUS_RE:
        if re.match(pat, x):
            return re.sub(pat, rep, x)
    return x[:1].upper() + x[1:]


def outcome(p: dict, said: list[str], rng: random.Random | None = None) -> tuple[str, str]:
    """(mood, text) after a plan ran; `said` is what bolo.actions.perform reported."""
    bad = [x for x in said if any(w in x for w in PROBLEM)]
    if bad:
        return "oops", "Hmm, poora nahi hua: " + bad[0]
    if p["intent"] in ("check_status", "find_file") and said:
        return "happy", hinglish_status(said[0])
    return reply(p, done=True, rng=rng)


HELP_RE = r"^(myna |hey |ok )?(tum |aap |tu )?(kya kya|kya) (kar sakte|kar sakti|kar sakta|karte|kar sakta) ?(ho|hai|hain|hu)?\b|^(help|madad|help karo|commands?|kya bol sakte)\b"
HELP = [("Apps", "chrome kholo · spotify band karo"), ("Folders", "downloads folder kholo · projects mein terminal kholo"),
        ("Awaaz, roshni", "volume chaalis pe kar do · brightness kam karo · agla gaana"),
        ("Settings", "dark mode on karo · wifi band karo · screen lock karo · screenshot lo"),
        ("Sawaal", "battery kitni hai · storage kitna bacha hai · internet chal raha hai kya"),
        ("Internet", "youtube pe lofi chalao · github kholo · python list sort search karo"),
        ("Time", "5 minute ka timer lagao · 2 ghante baad chai yaad dilana"),
        ("Log", "mummy ko whatsapp pe message bhejo ki main late hoon · rahul ko call karo"),
        ("Routines", 'bolo sikhao "coding mode" "vs code kholo" "dnd on karo"')]


def is_help(text: str) -> bool:
    import re

    return bool(re.search(HELP_RE, " ".join(text.lower().replace("?", " ").split())))


def help_text() -> str:
    return "Ye sab bol sakte ho:\n" + "\n".join(f"   {k:<14} {v}" for k, v in HELP)


# ---- the terminal mascot (same bird as the launch cartoon: teal body, crest, big eyes, yellow beak)
EYES = {"happy": "^ ^", "unsure": "o ?", "sleepy": "- -", "thinking": "o o", "oops": "x x", "listening": "O O"}
TOP = {"happy": "  ,/|\\,  ", "unsure": "  ,/|\\, ?", "sleepy": "  ,/|\\, z", "thinking": "  ,/|\\, .",
       "oops": "  ,/|\\, !", "listening": "  ,/|\\, )))"}


def _color() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"


def mascot(mood: str, text: str, name: str = "Myna", color: bool | None = None) -> str:
    color = _color() if color is None else color
    t = (lambda c, s: f"\033[{c}m{s}\033[0m") if color else (lambda c, s: s)
    teal, amber, bold = "38;5;43", "38;5;214", "1"
    eyes = EYES.get(mood, EYES["thinking"])
    lines = [t(teal, TOP.get(mood, TOP["thinking"])),
             t(teal, " ( ") + eyes + t(teal, " ) ") + "  " + t(bold, name + ":") + " " + text,
             t(teal, "  (") + t(amber, " v ") + t(teal, ")")]
    return "\n".join(lines)


# ---- speaking aloud
WIN_SAY = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
           "$v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -in 'en-IN','hi-IN' } | "
           "Select-Object -First 1; if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; $s.Speak($env:BOLO_SAY)")
MAC_VOICES = ("Rishi", "Aman", "Tara")  # Indian English voices read Roman Hinglish best


def speak_step(text: str, os_name: str, voices: list[str] | None = None) -> dict | None:
    """A run step (same format as bolo.actions plans) that says `text`, or None if this computer has no voice."""
    if os_name == "mac":
        v = next((x for x in MAC_VOICES if voices is None or x in voices), None)
        return {"do": "run", "argv": ["say", *(["-v", v] if v else []), "-f", "-"], "stdin": text}
    if os_name == "windows":
        return {"do": "run", "argv": ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", WIN_SAY],
                "env": {"BOLO_SAY": text}}
    if shutil.which("espeak-ng"):
        return {"do": "run", "argv": ["espeak-ng", "-v", "en-in", "--stdin"], "stdin": text}
    if shutil.which("spd-say"):
        return {"do": "run", "argv": ["spd-say", "-l", "en", "-e"], "stdin": text}
    return None


def mac_voices() -> list[str]:
    import subprocess

    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [ln.split()[0] for ln in out.splitlines() if ln.strip()]


def speak(text: str, os_name: str, wait: bool = False) -> bool:
    import subprocess

    st = speak_step(text, os_name, mac_voices() if os_name == "mac" else None)
    if st is None or shutil.which(st["argv"][0]) is None:
        return False
    try:
        pr = subprocess.Popen(st["argv"], stdin=subprocess.PIPE, text=True, env={**os.environ, **st.get("env", {})},
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if wait:
            pr.communicate(st.get("stdin", ""), timeout=30)
        else:
            pr.stdin.write(st.get("stdin", ""))
            pr.stdin.close()
    except (OSError, subprocess.SubprocessError):
        return False
    return True
