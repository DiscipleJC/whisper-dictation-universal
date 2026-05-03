#!/usr/bin/env python3
"""
Whisper Dictation — Universal Installer
Supported: macOS Apple Silicon, macOS Intel, Linux, Windows
Usage: python3 install.py
"""

import sys
import os
import platform
import subprocess
import shutil
from pathlib import Path

# ── Platform ──────────────────────────────────────────────────────────────────

SYSTEM          = platform.system()    # Darwin | Linux | Windows
MACHINE         = platform.machine()   # arm64  | x86_64 | AMD64
IS_MACOS        = SYSTEM == "Darwin"
IS_LINUX        = SYSTEM == "Linux"
IS_WINDOWS      = SYSTEM == "Windows"
IS_APPLE_SILICON = IS_MACOS and MACHINE == "arm64"

PROJECT_DIR = Path(__file__).parent.resolve()
VENV_DIR    = PROJECT_DIR / "venv"
DICTATE_PY  = PROJECT_DIR / "dictate.py"

# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'─' * 52}")
    print(f"  {title}")
    print(f"{'─' * 52}")

def ok(msg):   print(f"  ✅  {msg}")
def warn(msg): print(f"  ⚠️   {msg}")
def info(msg): print(f"  ℹ️   {msg}")

def fail(msg):
    print(f"\n  ❌  {msg}\n")
    sys.exit(1)

def run(cmd, check=True, capture=False):
    return subprocess.run(cmd, check=check,
                          capture_output=capture, text=True)

def pip():
    base = VENV_DIR / ("Scripts" if IS_WINDOWS else "bin")
    return base / ("pip.exe" if IS_WINDOWS else "pip")

def python():
    base = VENV_DIR / ("Scripts" if IS_WINDOWS else "bin")
    return base / ("python.exe" if IS_WINDOWS else "python3")

# ── Step 1: Python version ────────────────────────────────────────────────────

def check_python():
    section("Step 1 — Python version")
    v = sys.version_info
    if v < (3, 10):
        fail(f"Python 3.10+ required. Found: {v.major}.{v.minor}")
    ok(f"Python {v.major}.{v.minor}.{v.micro}")

# ── Step 2: Platform ──────────────────────────────────────────────────────────

def check_platform():
    section("Step 2 — Platform detection")
    if IS_APPLE_SILICON:
        ok("macOS Apple Silicon (M-series)")
        info("Backend: mlx_whisper — optimised for Apple Neural Engine")
    elif IS_MACOS:
        ok("macOS Intel")
        info("Backend: faster-whisper (CPU)")
        warn("Transcription is slower without Apple Silicon — consider 'tiny' or 'base' model")
    elif IS_LINUX:
        ok("Linux")
        info("Backend: faster-whisper")
        info("NVIDIA GPU detected automatically by faster-whisper if CUDA is installed")
    elif IS_WINDOWS:
        ok("Windows")
        info("Backend: faster-whisper")
        warn("For GPU acceleration install CUDA 11.x + cuDNN 8.x before this installer")
    else:
        fail(f"Unsupported platform: {SYSTEM} / {MACHINE}")

# ── Step 3: Virtual environment ───────────────────────────────────────────────

def create_venv():
    section("Step 3 — Virtual environment")
    if VENV_DIR.exists():
        warn(f"venv already exists — skipping creation")
        warn(f"Delete {VENV_DIR} and re-run to start fresh")
    else:
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
        ok(f"Created: {VENV_DIR}")

# ── Step 4: Packages ──────────────────────────────────────────────────────────

def install_packages():
    section("Step 4 — Installing packages")

    common = ["sounddevice", "pynput", "pyperclip", "numpy"]

    if IS_APPLE_SILICON:
        backend = ["mlx-whisper"]
    else:
        backend = ["faster-whisper"]
        if not IS_APPLE_SILICON:
            info("Note: dictate.py uses mlx_whisper API (Apple Silicon only).")
            info("On this platform you will need a platform-specific dictate script.")
            info("See: https://github.com/DiscipleJC/whisper-dictation")

    packages = common + backend
    info(f"Packages: {', '.join(packages)}")

    run([str(pip()), "install", "--upgrade", "pip"], capture=True)
    run([str(pip()), "install"] + packages)
    ok("All packages installed")

# ── Step 5: Autostart ─────────────────────────────────────────────────────────

def _macos_plist_content():
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.whisper-dictation</string>

    <key>ProgramArguments</key>
    <array>
        <string>{python()}</string>
        <string>{DICTATE_PY}</string>
    </array>

    <key>WorkingDirectory</key>
    <string>{PROJECT_DIR}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>{Path.home()}/Library/Logs/whisper-dictation.log</string>

    <key>StandardErrorPath</key>
    <string>{Path.home()}/Library/Logs/whisper-dictation.error.log</string>
</dict>
</plist>
"""

def setup_macos():
    section("Step 5 — macOS Launch Agent")

    launch_agents = Path.home() / "Library" / "LaunchAgents"
    launch_agents.mkdir(exist_ok=True)
    plist_dst = launch_agents / "com.whisper-dictation.plist"

    plist_dst.write_text(_macos_plist_content())
    os.chmod(plist_dst, 0o644)
    ok(f"Installed: {plist_dst}")

    run(["launchctl", "unload", str(plist_dst)], check=False, capture=True)
    run(["launchctl", "load", str(plist_dst)])
    ok("Launch Agent loaded — Whisper Dictation is running")

def _linux_service_content():
    return f"""[Unit]
Description=Whisper Dictation — push-to-talk voice input
After=graphical-session.target

[Service]
Type=simple
ExecStart={python()} {DICTATE_PY}
WorkingDirectory={PROJECT_DIR}
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
"""

def setup_linux():
    section("Step 5 — Linux systemd user service")

    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_dir.mkdir(parents=True, exist_ok=True)
    service_path = service_dir / "whisper-dictation.service"
    service_path.write_text(_linux_service_content())

    run(["systemctl", "--user", "daemon-reload"])
    run(["systemctl", "--user", "enable", "--now", "whisper-dictation"])
    ok("systemd service installed and started")

def _windows_bat_content():
    return f'@echo off\nstart /min "" "{python()}" "{DICTATE_PY}"\n'

def setup_windows():
    section("Step 5 — Windows autostart")

    bat_path = PROJECT_DIR / "start-whisper-dictation.bat"
    bat_path.write_text(_windows_bat_content())

    startup = (Path(os.environ.get("APPDATA", ""))
               / "Microsoft" / "Windows" / "Start Menu"
               / "Programs" / "Startup")
    dst = startup / "whisper-dictation.bat"
    shutil.copy2(bat_path, dst)
    ok(f"Startup shortcut: {dst}")
    warn("Run start-whisper-dictation.bat to start now, or log out and back in")

def setup_autostart():
    if IS_MACOS:
        setup_macos()
    elif IS_LINUX:
        setup_linux()
    elif IS_WINDOWS:
        setup_windows()

# ── Step 6: Accessibility (macOS) ─────────────────────────────────────────────

def guide_accessibility():
    if not IS_MACOS:
        return

    section("Step 6 — Accessibility permission (macOS)")

    import glob
    pattern = ("/opt/homebrew/Cellar/python@3.12/*/Frameworks/"
               "Python.framework/Versions/3.12/Resources/Python.app")
    matches = glob.glob(pattern)
    python_app = matches[0] if matches else None

    print()
    print("  pynput needs Accessibility access to monitor the keyboard.")
    print()
    print("  Open:  System Settings → Privacy & Security → Accessibility → +")
    if python_app:
        print(f"\n  Add this path (Cmd+Shift+G in the file picker):")
        print(f"  {python_app}")
    else:
        print("  Add: /opt/homebrew/Cellar/python@3.12/<version>/")
        print("       Frameworks/Python.framework/Versions/3.12/Resources/Python.app")
    print()
    print("  After adding — restart the agent:")
    print("  launchctl unload ~/Library/LaunchAgents/com.whisper-dictation.plist")
    print("  launchctl load   ~/Library/LaunchAgents/com.whisper-dictation.plist")

# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary():
    section("Done — Whisper Dictation installed")
    print()
    print("  Hotkey : Hold RIGHT OPTION (Alt) → speak → release → text appears")
    print()
    if IS_MACOS:
        print("  Logs   : ~/Library/Logs/whisper-dictation.log")
        print("  Stop   : launchctl unload ~/Library/LaunchAgents/com.whisper-dictation.plist")
        print("  Start  : launchctl load   ~/Library/LaunchAgents/com.whisper-dictation.plist")
    elif IS_LINUX:
        print("  Status : systemctl --user status whisper-dictation")
        print("  Stop   : systemctl --user stop  whisper-dictation")
        print("  Start  : systemctl --user start whisper-dictation")
    elif IS_WINDOWS:
        print("  To start now: run start-whisper-dictation.bat")
        print("  Autostart active on next login")
    print()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 52)
    print("  Whisper Dictation — Universal Installer v1.0")
    print("=" * 52)

    check_python()
    check_platform()
    create_venv()
    install_packages()
    setup_autostart()
    guide_accessibility()
    print_summary()

if __name__ == "__main__":
    main()
