"""Carry out a plan from bolo.actions.plan. Dry run unless `execute=True`; never a shell.

    from bolo.actions import plan, perform
    p = plan({"intent": "open_app", "slots": {"app": "chrome"}}, env)
    perform(p)                          # prints what it would do
    perform(p, execute=True)            # does it (asks first for close_app / sleep)
"""
from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import webbrowser
from pathlib import Path

from .scan import current_os, norm_name

TIMEOUT = 20


def describe(step: dict) -> str:
    d = step["do"]
    if d == "run":
        argv = step["argv"]
        if argv[0] == "powershell.exe":
            shown = [argv[0], "<fixed script>"]
        elif argv[0] == "osascript":  # fixed script lines come in "-e" pairs; anything after them is input
            i = 1
            while i < len(argv) and argv[i] == "-e":
                i += 2
            shown = [argv[0], "<fixed script>", *argv[i:]]
        else:
            shown = argv
        extra = "".join(f" [{k}={v}]" for k, v in (step.get("env") or {}).items())
        return " ".join(shown) + (f"  (in {step['cwd']})" if step.get("cwd") else "") + extra + \
            ("  (stdin: text)" if "stdin" in step else "")
    if d == "open":
        return f"open {step['target']}"
    if d == "url":
        return f"browser: {step['url']}"
    if d == "status":
        return f"check {step['target']}"
    if d == "find":
        return f"search home for {step['name']!r}"
    if d == "schedule":
        return f"in {step['seconds']} s: {step['text']!r}"
    return str(step)


def _out(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _ps(script: str) -> str:
    return _out(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script]).strip()


def status(target: str, os_name: str | None = None) -> str:
    os_name = os_name or current_os()
    if target == "storage":
        u = shutil.disk_usage(Path.home())
        return f"{u.free / 1e9:.0f} GB free of {u.total / 1e9:.0f} GB"
    if target == "internet":
        try:
            socket.create_connection(("1.1.1.1", 53), timeout=2).close()
            return "internet is working"
        except OSError:
            return "no internet"
    if target == "battery":
        if os_name == "windows":
            v = _ps("(Get-CimInstance Win32_Battery).EstimatedChargeRemaining")
        elif os_name == "mac":
            m = re.search(r"(\d+)%", _out(["pmset", "-g", "batt"]))
            v = m.group(1) if m else ""
        else:
            caps = sorted(Path("/sys/class/power_supply").glob("BAT*/capacity"))
            v = caps[0].read_text().strip() if caps else ""
        return f"battery {v}%" if v else "no battery found"
    if target == "cpu":
        if os_name == "windows":
            v = _ps("(Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average")
            return f"CPU {v}% busy" if v else "could not read CPU"
        load = os.getloadavg()[0] / (os.cpu_count() or 1) * 100
        return f"CPU about {min(100, load):.0f}% busy"
    if target == "ram":
        if os_name == "windows":
            v = _ps("$o=Get-CimInstance Win32_OperatingSystem; [int](100-100*$o.FreePhysicalMemory/$o.TotalVisibleMemorySize)")
            return f"memory {v}% used" if v else "could not read memory"
        if os_name == "linux":
            mi = dict(re.findall(r"(\w+):\s+(\d+)", Path("/proc/meminfo").read_text()))
            return f"memory {100 - 100 * int(mi['MemAvailable']) / int(mi['MemTotal']):.0f}% used"
        total = int(_out(["sysctl", "-n", "hw.memsize"]) or 0)
        vm = dict(re.findall(r"Pages (free|inactive|speculative):\s+(\d+)", _out(["vm_stat"])))
        free = sum(int(v) for v in vm.values()) * 16384
        return f"memory {100 - 100 * free / total:.0f}% used" if total else "could not read memory"
    if target == "wifi":
        if os_name == "windows":
            m = re.search(r"^\s*SSID\s*:\s*(.+)$", _out(["netsh", "wlan", "show", "interfaces"]), re.M)
        elif os_name == "mac":
            m = re.search(r"Current Wi-Fi Network:\s*(.+)", _out(["networksetup", "-getairportnetwork", "en0"]))
        else:
            m = re.search(r"^yes:(.+)$", _out(["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"]), re.M)
        return f"connected to {m.group(1).strip()}" if m else "not connected to Wi-Fi"
    if target == "bluetooth":
        if os_name == "linux":
            m = re.search(r"Powered:\s*(\w+)", _out(["bluetoothctl", "show"]))
            return f"bluetooth {'on' if m and m.group(1) == 'yes' else 'off'}" if m else "could not read bluetooth"
        if os_name == "mac" and shutil.which("blueutil"):
            return "bluetooth " + ("on" if _out(["blueutil", "-p"]).strip() == "1" else "off")
        return "can't read bluetooth here yet"
    if target == "git":
        out = _out(["git", "status", "--short", "--branch"])
        if not out:
            return "this folder is not a git repository"
        lines = out.strip().splitlines()
        return f"{lines[0].lstrip('# ')}, {len(lines) - 1} changed files"
    return f"can't check {target} yet"


def find(name: str, home: Path | None = None, max_depth: int = 4, limit: int = 5) -> list[str]:
    """Breadth-first search of home for names that sound like `name`."""
    from ..grounding import similarity
    from .scan import SKIP

    home = home or Path.home()
    target, hits, level = norm_name(name), [], [home]
    for _ in range(max_depth):
        nxt = []
        for d in level:
            try:
                entries = list(d.iterdir())
            except OSError:
                continue
            for e in entries:
                if e.name.startswith((".", "$")) or e.name in SKIP:
                    continue
                s = similarity(norm_name(e.stem if e.is_file() else e.name), target)
                if s >= 0.75:
                    hits.append((s, len(e.parts), str(e)))
                if e.is_dir() and not e.is_symlink():
                    nxt.append(e)
        level = nxt
    return [p for _, _, p in sorted(hits, key=lambda h: (-h[0], h[1]))[:limit]]


def _open(target: str, os_name: str) -> None:
    if os_name == "windows":
        os.startfile(target)  # type: ignore[attr-defined]  # ShellExecute: files, folders, .lnk, ms-settings: URIs
    else:
        subprocess.Popen(["open" if os_name == "mac" else "xdg-open", target])


def _reveal(path: str, os_name: str) -> None:
    if os_name == "windows":
        subprocess.Popen(["explorer.exe", f"/select,{path}"])
    elif os_name == "mac":
        subprocess.Popen(["open", "-R", path])
    else:
        subprocess.Popen(["xdg-open", str(Path(path).parent)])


def perform(p: dict, execute: bool = False, confirm=None, log=print) -> list[str]:
    """Run (or, by default, only describe) a plan. `confirm(question) -> bool` is asked for risky commands."""
    lines = [describe(s) for s in p["steps"]]
    if not p.get("ok", True):
        log(p["say"])
        return [p["say"]]
    if not execute:
        log(f"[dry run] {p['say']}" + (f"  ({p['partial']})" if p.get("partial") else ""))
        for ln in lines:
            log("   would: " + ln)
        return lines
    if p["os"] != current_os():
        raise RuntimeError(f"this plan is for {p['os']}, this computer runs {current_os()}")
    if p.get("confirm"):
        ask = confirm or (lambda q: input(q + " [y/N] ").strip().lower() in ("y", "yes", "haan", "ha"))
        if not ask(f"{p['say']}?"):
            log("ok, not doing it")
            return ["cancelled"]
    log(p["say"] + (f"  ({p['partial']})" if p.get("partial") else ""))
    said = []
    for s in p["steps"]:
        d = s["do"]
        if d == "run":
            if shutil.which(s["argv"][0]) is None:
                said.append(f"{s['argv'][0]} is not installed")
                continue
            r = subprocess.run(s["argv"], cwd=s.get("cwd"), env={**os.environ, **(s.get("env") or {})},
                               input=s.get("stdin"), text=True, capture_output=True, timeout=TIMEOUT)
            if r.returncode == 3 and p["intent"] == "close_app":
                said.append(f"{p['slots'].get('app')} is not open")
            elif r.returncode != 0:
                said.append(f"that did not work ({(r.stderr or r.stdout).strip()[:120]})")
        elif d == "open":
            if s["target"].startswith("<"):  # a preview placeholder, never a real target
                said.append("that app was not found on this computer")
                continue
            _open(s["target"], p["os"])
        elif d == "url":
            webbrowser.open(s["url"])
        elif d == "status":
            said.append(status(s["target"], p["os"]))
        elif d == "find":
            hits = find(s["name"])
            if hits:
                _reveal(hits[0], p["os"])
                said.append("found " + hits[0] + (f" (+{len(hits) - 1} more)" if len(hits) > 1 else ""))
            else:
                said.append(f"no {s['name']} under your home folder")
        elif d == "schedule":
            said.append("timers need the Myna background app, which is not built yet")
    for x in said:
        log("   " + x)
    return said
