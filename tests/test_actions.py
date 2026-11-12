"""The action step: every command has a plan on every OS, plans never use a shell, and words from speech never
become code. Nothing here runs a command."""
import pytest

from bolo.actions import describe, perform, plan
from bolo.actions.scan import _add_app, linux_apps, norm_name, scan_system, windows_apps
from bolo.grounding import Environment

OSES = ["windows", "linux", "mac"]
EVIL = 'x"; rm -rf ~; echo "$(whoami)` & del /q C:\\ | calc'

EXAMPLES = [
    ("open_terminal", {"folder": "projects"}), ("open_terminal", {}), ("open_claude_in_folder", {"folder": "projects"}),
    ("call_contact", {"name": "rahul", "app": "whatsapp"}), ("call_contact", {"name": "rahul"}),
    ("open_app", {"app": "chrome"}), ("check_status", {"target": "battery"}),
    ("open_file_or_folder", {"path": "projects"}), ("set_volume", {"level": 40}),
    ("set_volume", {"direction": "up"}), ("set_volume", {"direction": "mute"}),
    ("set_brightness", {"level": 70}), ("set_brightness", {"direction": "down"}), ("set_brightness", {"direction": "max"}),
    ("media_control", {"action": "next"}), ("media_control", {"action": "pause"}), ("lock_screen", {}),
    ("sleep_mac", {}), ("take_screenshot", {}),
    *[("toggle_setting", {"setting": s, "state": st}) for s in ("wifi", "bluetooth", "do_not_disturb", "dark_mode")
      for st in ("on", "off")],
    ("set_timer", {"amount": 5, "unit": "minutes"}), ("set_reminder", {"task": "chai peeni hai", "amount": 2, "unit": "hours"}),
    ("web_search", {"query": "python list sort"}), ("web_search", {"query": "lofi", "engine": "youtube"}),
    ("open_website", {"site": "github"}), ("open_website", {"site": "some random site"}),
    ("close_app", {"app": "chrome"}), ("find_file", {"path": "resume"}),
    ("send_message", {"name": "mummy", "message": "main late hoon"}),
]


def env_for(os_name):
    return Environment(folders=["projects"], files=[], apps=["chrome"], where={"projects": [f"/home/u/projects"]},
                       launch={"chrome": "C:/x/Google Chrome.lnk" if os_name == "windows" else "google-chrome"},
                       os=os_name)


def all_steps(p):
    return p["steps"]


@pytest.mark.parametrize("os_name", OSES)
@pytest.mark.parametrize("intent,slots", EXAMPLES)
def test_every_command_has_a_plan(os_name, intent, slots):
    p = plan({"intent": intent, "slots": slots}, env_for(os_name), os_name)
    assert p["ok"] and p["steps"], (intent, slots, p)
    for s in p["steps"]:
        assert s["do"] in {"run", "open", "url", "status", "find", "schedule"}
        if s["do"] == "run":
            assert isinstance(s["argv"], list) and all(isinstance(a, str) for a in s["argv"])
            assert s["argv"][0] not in ("sh", "bash", "zsh", "cmd", "/bin/sh") or s["argv"][:2] == ["cmd.exe", "/c"]
        describe(s)


@pytest.mark.parametrize("os_name", OSES)
@pytest.mark.parametrize("intent,slot", [("open_terminal", "folder"), ("open_claude_in_folder", "folder"),
                                         ("open_app", "app"), ("close_app", "app"), ("web_search", "query"),
                                         ("open_website", "site"), ("send_message", "message"), ("find_file", "path")])
def test_speech_never_becomes_code(os_name, intent, slot):
    """A hostile value may only appear as a whole argv item, a cwd, stdin, an env var, or URL-encoded."""
    slots = {slot: EVIL} | ({"name": "rahul"} if intent == "send_message" else {})
    env = env_for(os_name)
    env.where[norm_name(EVIL)] = [EVIL]  # pretend a folder with that name exists
    p = plan({"intent": intent, "slots": slots}, env, os_name)
    for s in p["steps"]:
        if s["do"] == "run":
            argv = s["argv"]
            code = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a in ("-e", "-Command")]
            assert not any("rm -rf" in c for c in code), "value spliced into a script"
            for a in argv:
                if "rm -rf" in a:  # allowed only as the whole value (or --working-directory=<value>)
                    assert a in (EVIL, norm_name(EVIL)) or a.endswith(EVIL), a
        if s["do"] == "url":
            assert "rm -rf" not in s["url"] and " " not in s["url"]


def test_unknown_site_is_searched_not_guessed():
    p = plan({"intent": "open_website", "slots": {"site": "paytmm"}}, None, "windows")
    assert p["steps"][0]["url"].startswith("https://www.google.com/search?q=") and p["partial"]
    p = plan({"intent": "open_website", "slots": {"site": "moneycontrol"}}, None, "windows")
    assert p["steps"][0]["url"] == "https://www.moneycontrol.com" and not p["partial"]


def test_risky_commands_ask_first():
    for intent in ("close_app", "sleep_mac"):
        p = plan({"intent": intent, "slots": {"app": "chrome"} if intent == "close_app" else {}}, None, "windows")
        assert p["confirm"]
    assert not plan({"intent": "lock_screen", "slots": {}}, None, "windows")["confirm"]


def test_dry_run_by_default_and_wrong_os_refused():
    p = plan({"intent": "lock_screen", "slots": {}}, None, "windows" if __import__("platform").system() != "Windows" else "mac")
    lines = perform(p, log=lambda *_: None)
    assert lines and all("Lock" in x or "rundll32" in x or "pmset" in x for x in lines)
    with pytest.raises(RuntimeError):
        perform(p, execute=True, log=lambda *_: None)


def test_missing_folder_falls_back_home_and_says_so():
    p = plan({"intent": "open_terminal", "slots": {"folder": "nahi hai"}}, env_for("windows"), "windows")
    assert p["partial"] and p["steps"][0].get("cwd")


def test_windows_volume_level_is_key_presses():
    p = plan({"intent": "set_volume", "slots": {"level": 40}}, None, "windows")
    assert p["steps"][0]["env"]["BOLO_KEYS"] == "174x50,175x20"  # all the way down, then 20 x 2% up


def test_app_names_get_vendor_free_aliases():
    apps = {}
    _add_app(apps, "Google Chrome", "c.lnk")
    _add_app(apps, "Visual Studio Code", "v.lnk")
    assert apps["chrome"] == "c.lnk" and apps["vs code"] == "v.lnk" and apps["google chrome"] == "c.lnk"


def test_windows_start_menu_scan(tmp_path):
    sm = tmp_path / "ProgramData" / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    (sm / "Tools").mkdir(parents=True)
    for n in ("Google Chrome.lnk", "Tools/Uninstall Foo.lnk", "Tools/Spotify.lnk"):
        (sm / n).write_text("")
    apps = windows_apps({"ProgramData": str(tmp_path / "ProgramData"), "APPDATA": str(tmp_path / "none")})
    assert "chrome" in apps and "spotify" in apps and not any("uninstall" in a for a in apps)
    assert apps["settings"] == "ms-settings:" and "file explorer" in apps


def test_linux_desktop_scan(tmp_path):
    d = tmp_path / ".local/share/applications"
    d.mkdir(parents=True)
    (d / "firefox.desktop").write_text("[Desktop Entry]\nName=Firefox Web Browser\nExec=firefox\n")
    (d / "hidden.desktop").write_text("[Desktop Entry]\nName=Secret\nNoDisplay=true\n")
    apps = linux_apps(tmp_path)
    assert apps["firefox web browser"] == "firefox" and "secret" not in apps


def test_scan_keeps_real_paths(tmp_path):
    (tmp_path / "ML_Notes").mkdir()
    (tmp_path / "ML_Notes" / "week-1.pdf").write_text("")
    env = scan_system(tmp_path, os_name="linux")
    assert "ml notes" in env.folders and env.where["ml notes"] == [str(tmp_path / "ML_Notes")]
    assert env.where["week 1"] == [str(tmp_path / "ML_Notes" / "week-1.pdf")] and env.os == "linux"


def test_windows_never_opens_an_unscanned_name():
    env = env_for("windows")
    p = plan({"intent": "open_app", "slots": {"app": "cmd"}}, env, "windows")
    assert not p["ok"] and not p["steps"]
    p = plan({"intent": "open_app", "slots": {"app": "chrome"}}, env, "windows")
    assert p["steps"] == [{"do": "open", "target": "C:/x/Google Chrome.lnk"}]
