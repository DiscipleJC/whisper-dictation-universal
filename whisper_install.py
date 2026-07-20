#!/usr/bin/env python3
"""
Whisper Dictation — Universal Installer
Supported: macOS Apple Silicon, macOS Intel, Linux, Windows
Usage: python3 whisper_install.py
"""

import sys
if sys.version_info < (3, 10):
    sys.exit(
        f"\n❌ Python 3.10+ required (the installer creates a venv with the same Python).\n"
        f"   You ran this with Python {sys.version_info.major}.{sys.version_info.minor}.\n"
        f"   Install a newer Python and re-run:\n"
        f"   brew install python@3.12         # macOS\n"
        f"   python3.12 whisper_install.py    # use the new Python explicitly\n"
    )

import os
import platform
import subprocess
import shutil
from pathlib import Path

# ── Platform ──────────────────────────────────────────────────────────────────

SYSTEM           = platform.system()    # Darwin | Linux | Windows
MACHINE          = platform.machine()   # arm64  | x86_64 | AMD64
IS_MACOS         = SYSTEM == "Darwin"
IS_LINUX         = SYSTEM == "Linux"
IS_WINDOWS       = SYSTEM == "Windows"
IS_APPLE_SILICON = IS_MACOS and MACHINE == "arm64"

PROJECT_DIR = Path(__file__).parent.resolve()
VENV_DIR    = PROJECT_DIR / "venv"

# Set after choose_backend() — Apple Silicon fixed to mlx script
DICTATE_SCRIPT = PROJECT_DIR / "whisper_dictate_macos_m.py"
BACKEND        = "mlx"   # mlx | faster_whisper | openai_api
MODEL          = "mlx-community/whisper-medium-mlx-4bit"  # updated per backend

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

# ── Step 2: Platform + backend choice ────────────────────────────────────────

def choose_backend():
    global DICTATE_SCRIPT, BACKEND, MODEL

    section("Step 2 — Platform & backend")

    if IS_APPLE_SILICON:
        ok("macOS Apple Silicon (M-series)")
        info("Backend: mlx_whisper — optimised for Apple Neural Engine")
        DICTATE_SCRIPT = PROJECT_DIR / "whisper_dictate_macos_m.py"
        BACKEND = "mlx"
        MODEL   = "mlx-community/whisper-medium-mlx-4bit"
        return

    if IS_MACOS:
        ok("macOS Intel")
        warn("Apple Neural Engine not available — choose transcription backend:")
    elif IS_LINUX:
        ok("Linux")
        info("NVIDIA GPU is used automatically if CUDA is installed")
    elif IS_WINDOWS:
        ok("Windows")
        warn("For GPU acceleration install CUDA 11.x + cuDNN 8.x first")
    else:
        fail(f"Unsupported platform: {SYSTEM} / {MACHINE}")

    print()
    print("  Choose transcription backend:")
    print("    [1] Local model (faster-whisper) — free, private, works offline")
    print("         Needs a decent CPU; first run downloads ~500MB model")
    print("    [2] OpenAI Whisper API — fast on any hardware, ~$0.006/min")
    print("         Requires internet + OpenAI API key")
    print()

    while True:
        choice = input("  Enter choice [1]: ").strip() or "1"
        if choice in ("1", "2"):
            break
        print("  Please enter 1 or 2.")

    if choice == "1":
        info("Backend: faster-whisper (local)")
        DICTATE_SCRIPT = PROJECT_DIR / "dictate_faster_whisper.py"
        BACKEND = "faster_whisper"
        MODEL   = "Systran/faster-whisper-medium"
    else:
        info("Backend: OpenAI Whisper API (cloud)")
        DICTATE_SCRIPT = PROJECT_DIR / "whisper_dictate_openai_api.py"
        BACKEND = "openai_api"
        MODEL   = "whisper-1 (OpenAI cloud)"
        _prompt_api_key()

def _prompt_api_key():
    env_path = PROJECT_DIR / ".env"

    # Check if key already saved
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY=") and len(line) > 20:
                ok("OpenAI API key already saved in .env")
                return

    print()
    print("  Enter your OpenAI API key (starts with sk-):")
    key = input("  API key: ").strip()
    if not key.startswith("sk-"):
        warn("Key doesn't start with 'sk-' — saved anyway, check if correct")

    env_path.write_text(f"OPENAI_API_KEY={key}\n")
    os.chmod(env_path, 0o600)
    ok(f"API key saved to {env_path}")

# ── Step 3: Virtual environment ───────────────────────────────────────────────

def create_venv():
    section("Step 3 — Virtual environment")
    if VENV_DIR.exists():
        warn("venv already exists — skipping creation")
        warn(f"Delete {VENV_DIR} and re-run to start fresh")
    else:
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
        ok(f"Created: {VENV_DIR}")

# ── Step 4: Packages ──────────────────────────────────────────────────────────

def install_packages():
    section("Step 4 — Installing packages")

    common = ["sounddevice", "pynput", "pyperclip", "numpy"]

    if BACKEND == "mlx":
        backend_pkgs = ["mlx-whisper"]
    elif BACKEND == "faster_whisper":
        backend_pkgs = ["faster-whisper"]
    else:  # openai_api
        backend_pkgs = ["openai"]

    packages = common + backend_pkgs
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
        <string>{DICTATE_SCRIPT}</string>
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
ExecStart={python()} {DICTATE_SCRIPT}
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
    return f'@echo off\nstart /min "" "{python()}" "{DICTATE_SCRIPT}"\n'

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

# ── Step 6/7: Accessibility + Input Monitoring (macOS) ────────────────────────

def _find_python_app():
    """Locate the Homebrew Python.app bundle that LaunchAgent will use."""
    import glob
    pattern = ("/opt/homebrew/Cellar/python@3.12/*/Frameworks/"
               "Python.framework/Versions/3.12/Resources/Python.app")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def _try(cmd, **kwargs):
    """Run a command, ignore failures. Used for best-effort UX helpers."""
    try:
        subprocess.run(cmd, check=False, **kwargs)
    except Exception:
        pass


def _ux_assist(python_app, pane_url, pane_name):
    """Best-effort: copy path to clipboard, open System Settings pane,
    reveal Python.app in Finder. All optional, fail silently."""
    if python_app:
        _try(["pbcopy"], input=python_app.encode())
        print(f"  ✓ Python.app path copied to clipboard")
        print(f"     ({python_app})")
    _try(["open", pane_url])
    print(f"  ✓ Opened System Settings → {pane_name}")
    if python_app:
        _try(["open", "-R", python_app])
        print(f"  ✓ Revealed Python.app in Finder — drag-and-drop works too")


def guide_accessibility():
    if not IS_MACOS:
        return

    section("Step 6 — Accessibility permission (macOS)")

    python_app = _find_python_app()

    print()
    print("  pynput needs Accessibility access to monitor the keyboard.")
    print()

    _ux_assist(
        python_app,
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
        "Privacy & Security → Accessibility",
    )

    print()
    print("  In the Accessibility list:")
    print("    1. Click +")
    print("    2. Press Cmd+V (path is already in clipboard)")
    print("       — OR drag Python.app from the Finder window")
    print("       — OR press Cmd+Shift+G and paste the path")
    print("    3. Click Open, make sure the checkbox is ON")


def guide_input_monitoring():
    if not IS_MACOS:
        return

    section("Step 7 — Input Monitoring permission (macOS 14+)")

    python_app = _find_python_app()

    print()
    print("  CRITICAL on macOS Sonoma/Sequoia/Tahoe (14+):")
    print("  pynput needs Input Monitoring permission IN ADDITION to Accessibility.")
    print("  Without it the LaunchAgent runs but the hotkey is silently ignored.")
    print()

    _ux_assist(
        python_app,
        "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent",
        "Privacy & Security → Input Monitoring",
    )

    print()
    print("  Same procedure as Accessibility — add the same Python.app path,")
    print("  make sure the checkbox is ON.")
    print()
    print("  After granting BOTH permissions — reload the agent:")
    print("  launchctl kickstart -k gui/$(id -u)/com.whisper-dictation")

# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary():
    section("Done — Whisper Dictation installed")
    print()
    print("  Hotkey : Hold RIGHT OPTION (Alt) → speak → release → text appears")
    print(f"  Model  : {MODEL}")
    print(f"  Script : {DICTATE_SCRIPT.name}")
    print()
    if IS_MACOS:
        print("  Logs   : ~/Library/Logs/whisper-dictation.log")
        print("  Stop   : launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.whisper-dictation.plist")
        print("  Start  : launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.whisper-dictation.plist")
        print("  Restart: launchctl kickstart -k gui/$(id -u)/com.whisper-dictation")
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
    print("  Whisper Dictation — Universal Installer v1.1")
    print("=" * 52)

    check_python()
    choose_backend()
    create_venv()
    install_packages()
    setup_autostart()
    guide_accessibility()
    guide_input_monitoring()
    print_summary()

if __name__ == "__main__":
    main()
