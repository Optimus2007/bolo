"""Read-only scan of what exists on this computer (Windows, Linux or macOS): apps, and folders/files under home.

Grounding matches spoken names against these; the action step then needs where each one really is, so the scan
also keeps `where` (name -> full paths) and `launch` (app name -> what to start). Nothing is modified, nothing is
sent anywhere. Contacts are not read.
"""
from __future__ import annotations

import os
import platform
import re
from pathlib import Path

from ..grounding import Environment

SKIP = {"Library", "Applications", "node_modules", ".Trash", "AppData", "Application Data", "$Recycle.Bin",
        "__pycache__", "venv", ".venv", "site-packages"}
# vendor words people usually leave out: "chrome kholo" means Google Chrome
VENDORS = ("google ", "microsoft ", "mozilla ", "adobe ", "jetbrains ", "apple ")
ALIASES = {"visual studio code": ["vs code", "code"], "file explorer": ["explorer", "files"],
           "windows terminal": ["terminal"], "windows powershell": ["powershell"], "microsoft edge": ["edge"],
           "microsoft teams": ["teams"], "word": ["ms word"], "excel": ["ms excel"], "powerpoint": ["ppt"],
           "settings": ["system settings"], "task manager": ["activity monitor"],
           "system settings": ["settings"], "activity monitor": ["task manager"]}
# Windows apps that have no Start-menu shortcut file (built in or from the Store): opened by their URI
WINDOWS_BUILTIN = {"settings": "ms-settings:", "calculator": "calculator:", "camera": "microsoft.windows.camera:",
                   "microsoft store": "ms-windows-store:", "file explorer": "explorer.exe",
                   "task manager": "taskmgr.exe", "notepad": "notepad.exe", "paint": "mspaint.exe",
                   "control panel": "control.exe", "snipping tool": "ms-screenclip:", "clock": "ms-clock:",
                   "photos": "ms-photos:", "mail": "outlookmail:", "maps": "bingmaps:"}


def current_os() -> str:
    s = platform.system().lower()
    return {"darwin": "mac", "windows": "windows"}.get(s, "linux")


def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("_", " ").replace("-", " ")).strip()


def _add_app(apps: dict[str, str], name: str, target: str) -> None:
    n = norm_name(name)
    if not n or n in apps:
        return
    apps[n] = target
    for v in VENDORS:
        if n.startswith(v) and n[len(v):] not in apps:
            apps[n[len(v):]] = target
    for a in ALIASES.get(n, []):
        apps.setdefault(a, target)


def windows_apps(env: dict | None = None) -> dict[str, str]:
    env = env or os.environ
    apps: dict[str, str] = {}
    roots = [Path(env.get("ProgramData", r"C:\ProgramData")), Path(env.get("APPDATA", ""))]
    for r in roots:
        sm = r / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        if not sm.exists():
            continue
        for lnk in sorted(sm.rglob("*.lnk")):
            if re.search(r"uninstall|readme|help|documentation|website", lnk.stem, re.I):
                continue
            _add_app(apps, lnk.stem, str(lnk))
    for n, t in WINDOWS_BUILTIN.items():
        _add_app(apps, n, t)
    return apps


def linux_apps(home: Path) -> dict[str, str]:
    """.desktop files (system, user, flatpak, snap). Launched with `gtk-launch <id>`."""
    apps: dict[str, str] = {}
    dirs = [home / ".local/share/applications", Path("/usr/share/applications"), Path("/usr/local/share/applications"),
            Path("/var/lib/flatpak/exports/share/applications"), home / ".local/share/flatpak/exports/share/applications",
            Path("/var/lib/snapd/desktop/applications")]
    for d in dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.desktop")):
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            entry = text.split("[Desktop Entry]", 1)[-1].split("\n[", 1)[0]
            if re.search(r"^(NoDisplay|Hidden)\s*=\s*true", entry, re.M | re.I):
                continue
            m = re.search(r"^Name\s*=\s*(.+)$", entry, re.M)
            if m:
                _add_app(apps, m.group(1), f.stem)
    return apps


def mac_apps(home: Path) -> dict[str, str]:
    apps: dict[str, str] = {}
    for root in (Path("/Applications"), home / "Applications", Path("/System/Applications"),
                 Path("/System/Applications/Utilities")):
        if root.exists():
            for a in sorted(root.glob("*.app")):
                _add_app(apps, a.stem, str(a))
    return apps


def scan_files(home: Path, max_depth: int = 2, max_entries: int = 5000):
    folders, files, where = [], [], {}

    def walk(d: Path, depth: int):
        if depth > max_depth or len(folders) + len(files) > max_entries:
            return
        try:
            entries = sorted(d.iterdir())
        except (PermissionError, OSError):
            return
        for e in entries:
            if e.name.startswith((".", "$")) or e.name in SKIP:
                continue
            try:
                if e.is_dir() and not e.is_symlink():
                    n = norm_name(e.name)
                    folders.append(n)
                    where.setdefault(n, []).append(str(e))
                    walk(e, depth + 1)
                elif e.is_file():
                    n = norm_name(e.stem)
                    files.append(n)
                    where.setdefault(n, []).append(str(e))
            except OSError:
                continue

    walk(home, 0)
    return folders, files, where


def scan_system(home: Path | str | None = None, os_name: str | None = None, max_depth: int = 2,
                max_entries: int = 5000) -> Environment:
    home = Path(home or Path.home())
    os_name = os_name or current_os()
    folders, files, where = scan_files(home, max_depth, max_entries)
    apps = {"windows": windows_apps, "linux": lambda: linux_apps(home), "mac": lambda: mac_apps(home)}[os_name]()
    uniq = lambda xs: sorted(set(x for x in xs if x))  # noqa: E731
    return Environment(uniq(folders), uniq(files), uniq(apps), [], where=where, launch=apps, os=os_name)
