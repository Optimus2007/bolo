"""Recording prompts, drawn only from the held-out TEST split.

Two kinds of prompts:
  read : the speaker reads a test sentence (shown in Roman; Devanagari shown as a hint)
  free : the speaker is shown only the *goal* ("Call Ankita on WhatsApp") and says it however they
         naturally would. Free prompts measure real phrasing variation, which no template covers. The
         transcript is optional (typed by the speaker afterwards); the label is the goal.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

PROMPT_VERSION = "v1"  # v1: prompts drawn from the 20 command dataset


def describe_goal(intent: str, slots: dict) -> str:
    s = slots
    if intent == "open_terminal":
        return f"Open a terminal in the “{s['folder']}” folder" if "folder" in s else "Open a new terminal"
    if intent == "open_claude_in_folder":
        return f"Start Claude Code in the “{s['folder']}” folder" if "folder" in s else "Start Claude Code"
    if intent == "call_contact":
        return f"Call {s['name']}" + (f" on {s['app']}" if "app" in s else "")
    if intent == "open_app":
        return f"Open the app {s['app']}"
    if intent == "check_status":
        return {"battery": "Ask how much battery is left", "wifi": "Ask whether Wi-Fi is connected",
                "internet": "Ask whether the internet is working", "storage": "Ask how much disk space is left",
                "git": "Ask for the git status", "cpu": "Ask about CPU usage", "ram": "Ask about RAM / memory usage",
                "bluetooth": "Ask whether Bluetooth is on"}[s["target"]]
    if intent == "open_file_or_folder":
        return f"Open “{s['path']}” (a file or folder)"
    if intent in ("set_volume", "set_brightness"):
        what = "volume" if intent == "set_volume" else "screen brightness"
        if "level" in s:
            return f"Set the {what} to {s['level']}%"
        return {"up": f"Turn the {what} up", "down": f"Turn the {what} down", "mute": "Mute the sound",
                "unmute": "Unmute the sound", "max": f"Put the {what} to maximum",
                "min": f"Put the {what} to minimum"}[s["direction"]]
    if intent == "media_control":
        return {"play": "Ask it to play the music", "pause": "Ask it to pause the music",
                "next": "Ask for the next song", "previous": "Ask for the previous song"}[s["action"]]
    if intent == "lock_screen":
        return "Ask it to lock the screen"
    if intent == "sleep_mac":
        return "Ask it to put the laptop to sleep"
    if intent == "take_screenshot":
        return "Ask it to take a screenshot"
    if intent == "toggle_setting":
        name = {"wifi": "Wi-Fi", "bluetooth": "Bluetooth", "do_not_disturb": "Do Not Disturb",
                "dark_mode": "dark mode"}[s["setting"]]
        return f"Turn {name} {s['state']}"
    if intent == "set_timer":
        return f"Set a timer for {s['amount']} {s['unit']}"
    if intent == "set_reminder":
        when = f" in {s['amount']} {s['unit']}" if "amount" in s else ""
        return f"Ask it to remind you{when}: “{s['task']}”"
    if intent == "web_search":
        return f"Search {'YouTube' if s.get('engine') == 'youtube' else 'the web'} for “{s['query']}”"
    if intent == "open_website":
        return f"Open the website {s['site']}"
    if intent == "close_app":
        return f"Close the app {s['app']}"
    if intent == "find_file":
        return f"Ask where “{s['path']}” is on the laptop"
    if intent == "send_message":
        via = f" on {s['app']}" if "app" in s else ""
        return f"Message {s['name']}{via}: “{s['message']}”"
    return "Say something the assistant can't do (e.g. ask about the weather, or ask it to delete a file)"


def build_prompts(test_path: Path | str, n_read: int = 80, n_free: int = 40, seed: int = 7) -> list[dict]:
    rows = [json.loads(line) for line in open(test_path, encoding="utf-8")]
    rng = random.Random(seed)
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_intent[r["intent"]].append(r)
    intents = sorted(by_intent)
    prompts, used = [], set()

    def take(k, kind):
        out = []
        i = 0
        while len(out) < k:
            intent = intents[i % len(intents)]
            i += 1
            pool = [r for r in by_intent[intent] if r["id"] not in used]
            if not pool:
                continue
            r = rng.choice(pool)
            used.add(r["id"])
            out.append({
                "prompt_id": f"{PROMPT_VERSION}-{kind}-{len(out):03d}",
                "kind": kind,
                "source_id": r["id"],
                "show": r["asr_text"] if kind == "read" else describe_goal(r["intent"], r["slots"]),
                "hint": r["tts_text"] if kind == "read" else "",
                "intent": r["intent"],
                "slots": r["slots"],
                "asr_text": r["asr_text"] if kind == "read" else None,
                "reason": r["meta"].get("reason"),
            })
        return out

    prompts += take(n_read, "read")
    prompts += take(n_free, "free")
    rng.shuffle(prompts)
    return prompts
