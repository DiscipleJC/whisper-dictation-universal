#!/usr/bin/env python3
"""
Whisper Dictation — cross-platform system-tray controller.

Shows a tray / menu-bar icon with live status and today's stats, and lets you
pause (privacy mode), restart the daemon, and open the settings/log. It talks to
the running dictation daemon over its local IPC (http://127.0.0.1:18765), so it
works on macOS, Linux and Windows via pystray — as long as the active backend
exposes that IPC.
"""
import json
import os
import platform
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import pystray
from PIL import Image, ImageDraw

IPC = "http://127.0.0.1:18765"
POLL_SEC = 1.5
PROJECT_DIR = Path(__file__).parent.resolve()
SYSTEM = platform.system()


def _log_path():
    if SYSTEM == "Darwin":
        return Path.home() / "Library" / "Logs" / "whisper-dictation.log"
    return PROJECT_DIR / "whisper-dictation.log"


def _open_path(p):
    p = str(p)
    try:
        if SYSTEM == "Darwin":
            subprocess.run(["open", p], check=False)
        elif SYSTEM == "Windows":
            os.startfile(p)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", p], check=False)
    except Exception:
        pass


def ipc_get(path):
    try:
        with urllib.request.urlopen(f"{IPC}{path}", timeout=2) as r:
            return json.load(r)
    except Exception:
        return None


def ipc_post(path):
    try:
        req = urllib.request.Request(f"{IPC}{path}", method="POST", data=b"")
        with urllib.request.urlopen(req, timeout=2) as r:
            return json.load(r)
    except Exception:
        return None


def _dot(rgba):
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse((14, 14, 50, 50), fill=rgba)
    return img


ICONS = {
    "idle":         _dot((150, 150, 150, 255)),
    "recording":    _dot((220, 60, 60, 255)),
    "transcribing": _dot((235, 170, 40, 255)),
    "offline":      _dot((120, 120, 120, 90)),
}


class Tray:
    def __init__(self):
        self.data = None  # latest /state, or None when the daemon is offline

    # ── dynamic menu text ────────────────────────────────────────────────────
    def status(self, _=None):
        if not self.data:
            return "● Offline — daemon not running"
        return {
            "idle":         "● Idle",
            "recording":    "🔴 Recording",
            "transcribing": "⏳ Transcribing",
        }.get(self.data.get("state"), "● Unknown")

    def stats(self, _=None):
        s = (self.data or {}).get("stats_today") or {}
        return f"Today: {s.get('count', 0)} · {s.get('words', 0)} words"

    def model(self, _=None):
        return f"Model: {(self.data or {}).get('model', '—').split('/')[-1]}"

    def language(self, _=None):
        return f"Language: {(self.data or {}).get('language', '—')}"

    def latency(self, _=None):
        avg = ((self.data or {}).get("stats_today") or {}).get("avg_sec")
        return f"Avg latency: {avg:.1f}s" if avg else "Avg latency: —"

    # ── actions ──────────────────────────────────────────────────────────────
    def privacy_checked(self, _=None):
        return bool((self.data or {}).get("privacy_mode"))

    def toggle_privacy(self, icon, item):
        ipc_post("/privacy/off" if self.privacy_checked() else "/privacy/on")

    def restart(self, icon, item):
        if SYSTEM == "Darwin":
            subprocess.run(
                ["launchctl", "kickstart", "-k",
                 f"gui/{os.getuid()}/com.whisper-dictation"], check=False)
        else:
            ipc_post("/restart")

    def open_settings(self, icon, item):
        local = PROJECT_DIR / "local_settings.py"
        _open_path(local if local.exists()
                   else PROJECT_DIR / "local_settings.example.py")

    def open_log(self, icon, item):
        _open_path(_log_path())

    def quit(self, icon, item):
        icon.stop()

    # ── build + run ──────────────────────────────────────────────────────────
    def _menu(self):
        info = lambda text: pystray.MenuItem(text, None, enabled=False)
        return pystray.Menu(
            info(self.status),
            pystray.Menu.SEPARATOR,
            info(self.stats),
            info(self.model),
            info(self.language),
            info(self.latency),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Privacy mode (mic closed when idle)",
                             self.toggle_privacy, checked=self.privacy_checked),
            pystray.MenuItem("Restart daemon", self.restart),
            pystray.MenuItem("Open settings", self.open_settings),
            pystray.MenuItem("Open log", self.open_log),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self.quit),
        )

    def _poll(self, icon):
        while True:
            self.data = ipc_get("/state")
            st = self.data.get("state") if self.data else "offline"
            icon.icon = ICONS.get(st, ICONS["idle"])
            icon.title = f"Whisper Dictation ({st})"
            try:
                icon.update_menu()
            except Exception:
                pass
            time.sleep(POLL_SEC)

    def run(self):
        icon = pystray.Icon("whisper-dictation", ICONS["offline"],
                            "Whisper Dictation", self._menu())
        threading.Thread(target=self._poll, args=(icon,), daemon=True).start()
        icon.run()


if __name__ == "__main__":
    Tray().run()
