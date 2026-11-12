"""Command (intent + slots) -> a plan of concrete steps for Windows, Linux or macOS. Building a plan runs nothing.

Safety rules, enforced here and tested in tests/test_actions.py:
  * never a shell: every step is an argv list (or a file/URL to open);
  * words that came from speech never become code: they go in as their own argv item, as the working folder,
    as stdin, or as an environment variable that a fixed PowerShell/AppleScript script reads;
  * unknown websites become a search, never a guessed address;
  * closing an app or putting the computer to sleep needs a yes first (`confirm`);
  * when an OS cannot do something without admin rights or an extra tool, the plan says so (`partial`)
    and does the nearest safe thing (usually opening the right settings page).

A step is one of:
  {"do": "run", "argv": [...], "cwd"?: str, "env"?: {...}, "stdin"?: str}
  {"do": "open", "target": path or URI}   (os.startfile / open / xdg-open)
  {"do": "url", "url": ...}               (default browser)
  {"do": "status", "target": ...}         (computed by bolo.actions.run)
  {"do": "find", "name": ...}             (search under home, show the best match)
  {"do": "schedule", "seconds": n, "text": ...}   (needs the Myna background app)
"""
from __future__ import annotations

import datetime as _dt
import shutil
from pathlib import Path
from urllib.parse import quote_plus

from .scan import current_os, norm_name
from .sites import SEARCH, SITES

CONFIRM = {"close_app", "sleep_mac"}
UNIT_S = {"seconds": 1, "minutes": 60, "hours": 3600}

# ---------- helpers ----------


def ps(script: str, **env) -> dict:
    """A fixed PowerShell script; inputs only via $env:BOLO_<NAME>."""
    return {"do": "run", "argv": ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                                  "-Command", script], "env": {f"BOLO_{k.upper()}": str(v) for k, v in env.items()}}


def osa(*lines: str, args: list[str] | None = None) -> dict:
    """A fixed AppleScript; inputs only as `argv` items."""
    argv = ["osascript"]
    for ln in lines:
        argv += ["-e", ln]
    return {"do": "run", "argv": argv + list(args or [])}


def run(*argv: str, **kw) -> dict:
    return {"do": "run", "argv": list(argv), **kw}


WIN_KEYS = r"""$s='[DllImport("user32.dll")] public static extern void keybd_event(byte b, byte s, uint f, UIntPtr e);'
$k=Add-Type -MemberDefinition $s -Name K -Namespace BoloKeys -PassThru
foreach($p in ($env:BOLO_KEYS -split ',')){ $vk,$n=$p -split 'x'; for($i=0;$i -lt [int]$n;$i++){
 $k::keybd_event([byte][int]$vk,0,0,[UIntPtr]::Zero); $k::keybd_event([byte][int]$vk,0,2,[UIntPtr]::Zero) } }"""
VK = {"vol_mute": 0xAD, "vol_down": 0xAE, "vol_up": 0xAF, "next": 0xB0, "prev": 0xB1, "play_pause": 0xB3}


def win_keys(*presses: tuple[str, int]) -> dict:
    return ps(WIN_KEYS, keys=",".join(f"{VK[k]}x{int(n)}" for k, n in presses))


WIN_BRIGHT = r"""$m=Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightnessMethods -ErrorAction Stop
if($env:BOLO_LEVEL -ne ''){ $n=[int]$env:BOLO_LEVEL } else {
 $c=(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness).CurrentBrightness; $n=[int]$c+[int]$env:BOLO_DELTA }
$n=[math]::Max(0,[math]::Min(100,$n)); $m | Invoke-CimMethod -MethodName WmiSetBrightness -Arguments @{Timeout=1;Brightness=[byte]$n} | Out-Null"""

WIN_SHOT = r"""Add-Type -AssemblyName System.Windows.Forms,System.Drawing
$b=[System.Windows.Forms.SystemInformation]::VirtualScreen; $bmp=New-Object System.Drawing.Bitmap $b.Width,$b.Height
$g=[System.Drawing.Graphics]::FromImage($bmp); $g.CopyFromScreen($b.Left,$b.Top,0,0,$bmp.Size)
New-Item -ItemType Directory -Force -Path (Split-Path $env:BOLO_PATH) | Out-Null; $bmp.Save($env:BOLO_PATH)"""

# graceful: CloseMainWindow lets the app ask "save changes?"; matched on the process name only, never window titles
WIN_CLOSE = r"""$n=[WildcardPattern]::Escape($env:BOLO_PROC)
$p=Get-Process | Where-Object { $_.MainWindowHandle -ne 0 -and $_.ProcessName -like $n }
if(-not $p){ exit 3 }; $p | ForEach-Object { [void]$_.CloseMainWindow() }"""

WIN_DARK = r"""$k='HKCU:\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize'
Set-ItemProperty -Path $k -Name AppsUseLightTheme -Value ([int]$env:BOLO_LIGHT) -Type DWord
Set-ItemProperty -Path $k -Name SystemUsesLightTheme -Value ([int]$env:BOLO_LIGHT) -Type DWord"""

# process names for common apps (Windows); others fall back to the spoken name without spaces
WIN_PROC = {"chrome": "chrome", "google chrome": "chrome", "edge": "msedge", "microsoft edge": "msedge",
            "firefox": "firefox", "brave": "brave", "vs code": "Code", "visual studio code": "Code",
            "word": "WINWORD", "excel": "EXCEL", "powerpoint": "POWERPNT", "outlook": "OUTLOOK",
            "microsoft teams": "ms-teams", "teams": "ms-teams", "whatsapp": "WhatsApp", "spotify": "Spotify",
            "discord": "Discord", "zoom": "Zoom", "telegram": "Telegram", "slack": "slack", "notepad": "notepad",
            "vlc": "vlc", "obs": "obs64", "notion": "Notion", "obsidian": "Obsidian", "postman": "Postman",
            "pycharm": "pycharm64", "intellij": "idea64", "android studio": "studio64", "cursor": "Cursor",
            "arc browser": "Arc", "arc": "Arc", "steam": "steam", "blender": "blender", "figma": "Figma",
            "task manager": "Taskmgr", "file explorer": "explorer", "calculator": "CalculatorApp"}
WEB_APP = {"whatsapp": "https://web.whatsapp.com", "microsoft teams": "https://teams.microsoft.com",
           "zoom": "https://app.zoom.us", "telegram": "https://web.telegram.org", "discord": "https://discord.com/app",
           "skype": "https://web.skype.com"}
CALL_APP = {"whatsapp": "whatsapp", "teams": "microsoft teams", "zoom": "zoom", "skype": "skype",
            "discord": "discord", "telegram": "telegram", "signal": "signal", "facetime": "facetime",
            "google meet": None, "phone": None}


def _where(env, name: str | None) -> str | None:
    if not name or env is None:
        return None
    paths = (getattr(env, "where", {}) or {}).get(norm_name(str(name)))
    return min(paths, key=lambda p: (len(Path(p).parts), p)) if paths else None


def _which(os_name: str, exe: str) -> bool:
    return os_name == current_os() and shutil.which(exe) is not None


def _shot_path(os_name: str) -> str:
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H.%M.%S")
    base = Path.home() / ("Desktop" if os_name == "mac" else "Pictures/Screenshots")
    return str(base / f"Myna {stamp}.png")


def _url(u: str) -> list[dict]:
    return [{"do": "url", "url": u}]


def _copy(os_name: str, text: str) -> dict:
    if os_name == "windows":
        return run("clip.exe", stdin=text)
    if os_name == "mac":
        return run("pbcopy", stdin=text)
    return run("wl-copy", stdin=text) if _which(os_name, "wl-copy") else run("xclip", "-selection", "clipboard", stdin=text)


# ---------- per command ----------


def _terminal(os_name, folder, claude=False):
    cwd = folder or str(Path.home())
    if os_name == "windows":
        if _which(os_name, "wt") or os_name != current_os():
            return [run("wt.exe", "-d", ".", *(["cmd.exe", "/k", "claude"] if claude else []), cwd=cwd)]
        return [run("cmd.exe", "/c", "start", "Myna", "cmd.exe", *(["/k", "claude"] if claude else []), cwd=cwd)]
    if os_name == "mac":
        if claude:
            return [osa("on run argv", 'tell application "Terminal"', "activate",
                        'do script "cd " & quoted form of (item 1 of argv) & " && claude"', "end tell", "end run",
                        args=[cwd])]
        return [run("open", "-a", "Terminal", cwd)]
    if _which(os_name, "gnome-terminal") or os_name != current_os():
        return [run("gnome-terminal", f"--working-directory={cwd}", *(["--", "claude"] if claude else []))]
    return [run("x-terminal-emulator", *(["-e", "claude"] if claude else []), cwd=cwd)]


def _open_app(os_name, env, app):
    target = (getattr(env, "launch", {}) or {}).get(norm_name(app)) if env is not None else None
    if os_name == "windows":
        if target is None and env is not None:  # never ShellExecute a name the scan did not find (could start anything)
            return []
        return [{"do": "open", "target": target or f"<Start menu: {app}>"}]
    if os_name == "mac":
        return [run("open", "-a", target or app)]
    return [run("gtk-launch", target or norm_name(app).replace(" ", "-"))]


def _close_app(os_name, env, app):
    n = norm_name(app)
    if os_name == "windows":
        return [ps(WIN_CLOSE, proc=WIN_PROC.get(n, n.replace(" ", "")))]
    if os_name == "mac":
        target = (getattr(env, "launch", {}) or {}).get(n) if env is not None else None
        name = Path(target).stem if target else app
        return [osa("on run argv", "tell application (item 1 of argv) to quit", "end run", args=[name])]
    return [run("pkill", "-TERM", "-i", "-x", n.replace(" ", ""))]


def _volume(os_name, level, direction):
    if os_name == "windows":  # each key press moves Windows volume by 2
        if level is not None:
            return [win_keys(("vol_down", 50), ("vol_up", round(int(level) / 2)))]
        return [win_keys({"up": ("vol_up", 5), "down": ("vol_down", 5), "max": ("vol_up", 50),
                          "mute": ("vol_mute", 1), "unmute": ("vol_mute", 1)}[direction])]
    if os_name == "mac":
        if level is not None:
            return [osa("on run argv", "set volume output volume ((item 1 of argv) as integer)", "end run",
                        args=[str(int(level))])]
        return [osa({"up": "set volume output volume ((output volume of (get volume settings)) + 10)",
                     "down": "set volume output volume ((output volume of (get volume settings)) - 10)",
                     "max": "set volume output volume 100", "mute": "set volume with output muted",
                     "unmute": "set volume without output muted"}[direction])]
    sink = "@DEFAULT_SINK@"
    if level is not None:
        return [run("pactl", "set-sink-volume", sink, f"{int(level)}%")]
    return [{"up": run("pactl", "set-sink-volume", sink, "+10%"), "down": run("pactl", "set-sink-volume", sink, "-10%"),
             "max": run("pactl", "set-sink-volume", sink, "100%"), "mute": run("pactl", "set-sink-mute", sink, "1"),
             "unmute": run("pactl", "set-sink-mute", sink, "0")}[direction]]


def _brightness(os_name, level, direction):
    if direction in ("max", "min"):
        level, direction = (100 if direction == "max" else 10), None
    if os_name == "windows":
        return [ps(WIN_BRIGHT, level="" if level is None else int(level),
                   delta={"up": 10, "down": -10}.get(direction, 0))]
    if os_name == "mac":  # no built-in command for a level: brightness keys (about 6% each)
        if level is not None:
            return [osa("on run argv", 'tell application "System Events"', "repeat 16 times", "key code 145",
                        "end repeat", "repeat ((item 1 of argv) as integer) times", "key code 144", "end repeat",
                        "end tell", "end run", args=[str(round(int(level) / 6.25))])]
        code = {"up": "144", "down": "145"}[direction]
        return [osa('tell application "System Events"', "repeat 2 times", f"key code {code}", "end repeat", "end tell")]
    if level is not None:
        return [run("brightnessctl", "set", f"{int(level)}%")]
    return [run("brightnessctl", "set", {"up": "+10%", "down": "10%-"}[direction])]


def _media(os_name, action):
    if os_name == "windows":
        return [win_keys(({"play": "play_pause", "pause": "play_pause", "next": "next", "previous": "prev"}[action], 1))]
    if os_name == "mac":
        verb = {"play": "play", "pause": "pause", "next": "next track", "previous": "previous track"}[action]
        return [osa('if application "Spotify" is running then', f'tell application "Spotify" to {verb}', "else",
                    f'tell application "Music" to {verb}', "end if")]
    return [run("playerctl", {"play": "play", "pause": "pause", "next": "next", "previous": "previous"}[action])]


def _toggle(os_name, setting, state):
    on = state == "on"
    if os_name == "windows":
        if setting == "dark_mode":
            return [ps(WIN_DARK, light=0 if on else 1)], None
        page = {"wifi": "ms-settings:network-wifi", "bluetooth": "ms-settings:bluetooth",
                "do_not_disturb": "ms-settings:notifications"}[setting]
        return [{"do": "open", "target": page}], "Windows needs admin rights to switch this directly; opening its settings page"
    if os_name == "mac":
        if setting == "wifi":
            return [run("networksetup", "-setairportpower", "en0", "on" if on else "off")], None
        if setting == "dark_mode":
            return [osa('tell application "System Events" to tell appearance preferences to set dark mode to '
                        + ("true" if on else "false"))], None
        if setting == "bluetooth" and _which(os_name, "blueutil"):
            return [run("blueutil", "--power", "1" if on else "0")], None
        page = {"bluetooth": "x-apple.systempreferences:com.apple.BluetoothSettings",
                "do_not_disturb": "x-apple.systempreferences:com.apple.Focus-Settings.extension"}[setting]
        return [{"do": "open", "target": page}], "macOS has no built-in command for this; opening its settings page"
    if setting == "wifi":
        return [run("nmcli", "radio", "wifi", state)], None
    if setting == "bluetooth":
        return [run("bluetoothctl", "power", state)], None
    if setting == "dark_mode":
        return [run("gsettings", "set", "org.gnome.desktop.interface", "color-scheme",
                    "prefer-dark" if on else "default")], None
    return [run("gsettings", "set", "org.gnome.desktop.notifications", "show-banners", "false" if on else "true")], None


def _system(os_name, intent):
    if intent == "lock_screen":
        return {"windows": [run("rundll32.exe", "user32.dll,LockWorkStation")],
                "mac": [run("pmset", "displaysleepnow")], "linux": [run("loginctl", "lock-session")]}[os_name]
    if intent == "sleep_mac":
        return {"windows": [run("rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0")],
                "mac": [run("pmset", "sleepnow")], "linux": [run("systemctl", "suspend")]}[os_name]
    path = _shot_path(os_name)
    return {"windows": [ps(WIN_SHOT, path=path)], "mac": [run("screencapture", "-x", path)],
            "linux": [run("gnome-screenshot", "-f", path)]}[os_name]


def _say(intent: str, s: dict) -> str:
    return {
        "open_terminal": "Opening a terminal" + (f" in {s.get('folder')}" if s.get("folder") else ""),
        "open_claude_in_folder": "Starting Claude Code" + (f" in {s.get('folder')}" if s.get("folder") else ""),
        "open_app": f"Opening {s.get('app')}", "close_app": f"Closing {s.get('app')}",
        "open_file_or_folder": f"Opening {s.get('path')}", "find_file": f"Looking for {s.get('path')}",
        "set_volume": "Volume " + (f"{s['level']}%" if s.get("level") is not None else str(s.get("direction"))),
        "set_brightness": "Brightness " + (f"{s['level']}%" if s.get("level") is not None else str(s.get("direction"))),
        "media_control": str(s.get("action", "")).capitalize(), "lock_screen": "Locking the screen",
        "sleep_mac": "Putting the computer to sleep", "take_screenshot": "Taking a screenshot",
        "toggle_setting": f"Turning {str(s.get('setting', '')).replace('_', ' ')} {s.get('state')}",
        "check_status": f"Checking {s.get('target')}", "web_search": f"Searching for {s.get('query')}",
        "open_website": f"Opening {s.get('site')}",
        "set_timer": f"Timer for {s.get('amount')} {s.get('unit')}", "set_reminder": f"Reminder: {s.get('task')}",
        "call_contact": f"Calling {s.get('name')}", "send_message": f"Message to {s.get('name')}",
    }.get(intent, intent)


def plan(pred: dict, env=None, os_name: str | None = None) -> dict:
    """pred: {"intent", "slots"} after grounding. Returns {"intent", "say", "steps", "confirm", "partial", "ok"}."""
    os_name = os_name or (getattr(env, "os", None) if env is not None else None) or current_os()
    intent, s = pred.get("intent"), dict(pred.get("slots") or {})
    out = {"intent": intent, "slots": s, "os": os_name, "say": _say(intent, s), "steps": [],
           "confirm": intent in CONFIRM, "partial": None, "ok": True}
    if intent in (None, "none"):
        return out | {"ok": False, "say": "Nothing to do"}
    if intent in ("open_terminal", "open_claude_in_folder"):
        folder = _where(env, s.get("folder"))
        if s.get("folder") and not folder:
            out["partial"] = f"no folder called {s['folder']!r} found; using the home folder"
        out["steps"] = _terminal(os_name, folder, claude=intent == "open_claude_in_folder")
    elif intent == "open_app":
        out["steps"] = _open_app(os_name, env, s["app"])
        if not out["steps"]:
            return out | {"ok": False, "say": f"I could not find {s['app']} on this computer"}
    elif intent == "close_app":
        out["steps"] = _close_app(os_name, env, s["app"])
    elif intent == "open_file_or_folder":
        p = _where(env, s.get("path"))
        if not p:
            return out | {"ok": False, "say": f"I could not find {s.get('path')}"}
        out["steps"] = [{"do": "open", "target": p}]
    elif intent == "find_file":
        out["steps"] = [{"do": "find", "name": str(s.get("path", ""))}]
    elif intent == "set_volume":
        out["steps"] = _volume(os_name, s.get("level"), s.get("direction"))
        if os_name == "windows" and s.get("direction") in ("mute", "unmute"):
            out["partial"] = "Windows has one mute key: it switches mute on or off"
    elif intent == "set_brightness":
        out["steps"] = _brightness(os_name, s.get("level"), s.get("direction"))
        if os_name == "windows":
            out["partial"] = "works on laptop screens; external monitors usually ignore it"
    elif intent == "media_control":
        out["steps"] = _media(os_name, s["action"])
    elif intent in ("lock_screen", "sleep_mac", "take_screenshot"):
        out["steps"] = _system(os_name, intent)
        if intent == "sleep_mac" and os_name == "windows":
            out["partial"] = "if hibernation is turned on, Windows hibernates instead of sleeping"
    elif intent == "toggle_setting":
        out["steps"], out["partial"] = _toggle(os_name, s["setting"], s["state"])
    elif intent == "check_status":
        out["steps"] = [{"do": "status", "target": s["target"]}]
    elif intent == "web_search":
        out["steps"] = _url(SEARCH.get(s.get("engine") or "google", SEARCH["google"]).format(quote_plus(str(s["query"]))))
    elif intent == "open_website":
        site = norm_name(str(s["site"]))
        if site in SITES:
            out["steps"] = _url(SITES[site])
        else:
            out["steps"] = _url(SEARCH["google"].format(quote_plus(site)))
            out["partial"] = f"{site!r} is not a site I know, so I searched for it instead of guessing its address"
    elif intent in ("set_timer", "set_reminder"):
        secs = int(s["amount"]) * UNIT_S[s["unit"]] if s.get("amount") and s.get("unit") else None
        out["steps"] = [{"do": "schedule", "seconds": secs, "text": s.get("task") or "Timer done"}]
        out["partial"] = "runs while Myna is open (`bolo app`, or `bolo do --run` counts down in the terminal)"
    elif intent in ("call_contact", "send_message"):
        app = CALL_APP.get(s.get("app") or "whatsapp")
        steps = []
        if intent == "send_message" and s.get("message"):
            steps.append(_copy(os_name, str(s["message"])))
        app_steps = _open_app(os_name, env, app) if app else []
        if app and env is not None and app not in (getattr(env, "launch", {}) or {}) and app in WEB_APP:
            app_steps = _url(WEB_APP[app])  # not installed: its web version
        if app_steps:
            steps += app_steps
        elif app in WEB_APP:
            steps += _url(WEB_APP[app])
        elif s.get("app") == "google meet":
            steps += _url("https://meet.google.com")
        out["steps"] = steps
        out["partial"] = ("contacts are not linked yet: I opened the app" +
                          (" and copied the message, so paste it after picking the contact" if intent == "send_message" else
                           "; pick the contact there"))
    else:
        return out | {"ok": False, "say": f"I can't do {intent} yet"}
    return out
