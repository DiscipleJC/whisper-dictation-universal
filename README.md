# 🎙 Whisper Dictation Universal

Push-to-talk voice dictation for macOS, Linux, and Windows.  
Hold a hotkey → speak → release → text pastes into any app.

Works in: terminal, Telegram, WhatsApp, browser, Claude Code, Slack — anywhere.

---

## Quick Start — macOS Apple Silicon (M1–M5)

```bash
# 0. Check Python version (must be 3.10+)
python3 --version
# If < 3.10:  brew install python@3.12

# 1. Clone
git clone https://github.com/DiscipleJC/whisper-dictation-universal.git
cd whisper-dictation-universal

# 2. Install (creates venv, installs deps, sets up autostart)
python3 whisper_install.py

# 3. Run
source venv/bin/activate
python whisper_dictate_macos_m.py
```

> **First run:** the Whisper model (~400 MB) downloads automatically and is cached locally. This takes 1–2 minutes — wait for the `=====` header to appear before speaking.

> **Hit an error?** See [Troubleshooting](#troubleshooting) below.

---

## How it works

```
Hold RIGHT OPTION (Alt) → speak → release
              ↓
     Whisper transcribes locally
              ↓
  Text pastes into the active app
```

---

## Prerequisites

### macOS

**Homebrew** (if not installed):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Python 3.10+** (if not installed):
```bash
brew install python@3.12
```

Check your version:
```bash
python3 --version   # must be 3.10 or higher
```

### Linux / Windows

- Python 3.10+ from [python.org](https://www.python.org/downloads/)
- Linux: `sudo apt install python3-dev portaudio19-dev` (Debian/Ubuntu)

---

## Backends

| Backend | Script | Platform | Cost | Requires |
|---|---|---|---|---|
| **mlx-whisper** | `whisper_dictate_macos_m.py` | macOS Apple Silicon (M1–M5) | Free | — |
| **faster-whisper** | `dictate_faster_whisper.py` | macOS Intel, Linux, Windows | Free | CPU / NVIDIA GPU |
| **OpenAI Whisper API** | `whisper_dictate_openai_api.py` | Any platform | ~$0.006/min | OpenAI API key |

> **Apple M1 / M2 / M3 / M4 / M5** — the `mlx-whisper` backend uses the Apple Neural Engine for fast, fully offline transcription.

---

## Default model

| Backend | Default model | Size | First-run download |
|---|---|---|---|
| mlx-whisper | `whisper-medium-mlx-4bit` | ~400 MB | Auto, ~1–2 min |
| faster-whisper | `medium` | ~500 MB | Auto, ~2–3 min |
| OpenAI API | `whisper-1` (cloud) | — | — |

Model is downloaded automatically on first run and cached locally. Subsequent runs start instantly.

---

## Install

### Option A — Universal installer (recommended)

Detects your platform and backend automatically:

```bash
python3 whisper_install.py
```

It will:
1. Detect platform (Apple Silicon / Intel / Linux / Windows)
2. Create a virtual environment
3. Install required packages
4. Set up autostart (LaunchAgent on macOS, systemd on Linux, Startup folder on Windows)
5. Guide you through Accessibility permission (macOS)

---

### Option B — Manual install

> If you're not sure which path to take, use **Option A** above. Manual install is for users who want full control over each step.

Run these commands **in order** — do not skip the version check or the venv verification:

```bash
# 1. Check Python version — must be 3.10+
python3 --version
# If lower → install: brew install python@3.12

# 2. Create virtual environment
#    Use python3.12 explicitly if you just installed it via Homebrew
python3 -m venv venv

# 3. Activate venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 4. Verify venv is active — the prompt should show "(venv)" prefix
which python
# Expected: .../whisper-dictation-universal/venv/bin/python
# If it shows a system path → venv is not active, stop and re-activate

# 5. Upgrade pip
pip install --upgrade pip
```

Then install dependencies for your backend (still inside the activated venv):

```bash
# macOS Apple Silicon (M1–M5)
pip install -r requirements.txt
python -c "import mlx_whisper; print('OK')"

# macOS Intel / Linux / Windows
pip install sounddevice pynput pyperclip numpy faster-whisper
python -c "import faster_whisper; print('OK')"

# OpenAI API backend (any platform)
pip install sounddevice pynput pyperclip numpy openai
python -c "import openai; print('OK')"
```

If the `python -c "import ..."` line prints `OK` — install succeeded.  
If you get `ModuleNotFoundError` → see [Troubleshooting](#troubleshooting).

---

## Run

You can run the script in **two ways**. Pick the one that fits your usage:

### Option 1 — Foreground (simplest, terminal must stay open)

```bash
source venv/bin/activate
python whisper_dictate_macos_m.py    # macOS Apple Silicon
python dictate_faster_whisper.py     # macOS Intel / Linux / Windows
python whisper_dictate_openai_api.py # OpenAI API backend
```

Terminal must stay open. Closing it stops the script. Good for testing or occasional use.

### Option 2 — Background autostart (set-and-forget)

LaunchAgent runs the script in background, starts automatically at every login, restarts if it crashes. **No terminal needed.** See [Autostart](#autostart-macos) below. Recommended for daily use.

> **First run only:** wait 1–2 minutes while the model downloads. You'll see the startup header when it's ready.

---

## Accessibility permission (macOS)

pynput requires Accessibility access to listen for global hotkeys.  
**Which binary you grant access to depends on how you run the script:**

### Running from terminal (foreground)

Grant Accessibility to **Terminal** (or iTerm, Warp, etc):

```
System Settings → Privacy & Security → Accessibility → + → add Terminal
```

### Running as LaunchAgent (background autostart) — IMPORTANT

LaunchAgent launches Python **directly via launchd**, not through Terminal — so it does **not** inherit Terminal's Accessibility permission. You must grant it to the **Homebrew Python.app** specifically (the signed application bundle that macOS Accessibility can recognize):

1. Find your exact Python.app path — it depends on the installed version. Run in terminal:
   ```bash
   ls -d /opt/homebrew/Cellar/python@3.12/*/Frameworks/Python.framework/Versions/3.12/Resources/Python.app
   ```
   It will print something like `/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/Resources/Python.app`. Copy that path.
2. Open `System Settings → Privacy & Security → Accessibility`
3. Click `+`
4. In the file picker, press `Cmd+Shift+G` to enter a path
5. Paste the path from step 1, click Open
6. Make sure the checkbox is **on**

After granting — restart the script or LaunchAgent (`launchctl unload` + `launchctl load`).

> **Why Python.app and not `venv/bin/python3.12`?** The venv's `python3.12` is a symlink without its own code signature. macOS Accessibility matches by signed application bundle, and the Homebrew `Python.app` is the entry that is reliably recognized.

> **macOS 15 Sequoia / macOS 26 Tahoe — you may need BOTH entries.** On newer macOS the system can add a *second* item to the permission list named `python3.12` (often shown with a generic/❓ icon) in addition to `Python` (the `.app`, rocket icon). On these versions, granting only `Python.app` can be **insufficient** — the hotkey stays dead until you also enable the `python3.12` entry. If pressing the hotkey does nothing after you granted `Python.app`, open the list again and toggle **both** `Python` and `python3.12` ON, in **both** Accessibility and Input Monitoring, then reload the LaunchAgent. (Reported on macOS 26 with Homebrew `python@3.12` 3.12.13.)

> Many users hit this: it works in terminal but silently fails after LaunchAgent setup. The cause is always: Accessibility granted to Terminal, but LaunchAgent runs Python directly.

---

## Input Monitoring permission (macOS Sonoma/Sequoia/Tahoe — required for LaunchAgent)

> **Critical** on macOS 14+ (Sonoma and newer). Without this, the LaunchAgent will run with no errors but the hotkey is **silently ignored**.

In addition to Accessibility, recent macOS versions also require **Input Monitoring** for any process that captures keyboard events via pynput. The setup mirrors Accessibility exactly:

### Running from terminal

Grant Input Monitoring to **Terminal** (same path as for Accessibility):

```
System Settings → Privacy & Security → Input Monitoring → + → add Terminal
```

### Running as LaunchAgent

Grant Input Monitoring to the **same Homebrew Python.app** you used for Accessibility:

1. Open `System Settings → Privacy & Security → Input Monitoring`
2. Click `+`
3. Press `Cmd+Shift+G` and paste the path:
   ```
   /opt/homebrew/Cellar/python@3.12/<version>/Frameworks/Python.framework/Versions/3.12/Resources/Python.app
   ```
4. Click Open, make sure the checkbox is **on**
5. Reload the LaunchAgent:
   ```bash
   launchctl unload ~/Library/LaunchAgents/com.whisper-dictation.plist
   launchctl load ~/Library/LaunchAgents/com.whisper-dictation.plist
   ```

> **Both Accessibility AND Input Monitoring are required** for the LaunchAgent path. Granting only Accessibility is the most common cause of "hotkey silently does nothing" after Accessibility is correctly configured.

---

## Microphone permission (macOS)

On first use macOS will ask for microphone access — click **Allow**.  
If you missed it: `System Settings → Privacy & Security → Microphone → enable Terminal`

---

## OpenAI API key (cloud backend only)

Skip this section if you use `mlx-whisper` or `faster-whisper` (offline backends).

The `whisper_dictate_openai_api.py` script reads your key from a `.env` file in the repo root:

```bash
# Create .env in the repo root
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

Get a key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

> `.env` is listed in `.gitignore` — your key will not be committed. Never share it or paste it into chats/issues.

---

## Configuration

Edit the script to change hotkey, language, or model:

```python
HOTKEY   = Key.alt_r   # Hotkey: Right Option / Right Alt
LANGUAGE = None        # None = auto-detect | "ru" | "en" | "uk" | etc.
MODEL    = "mlx-community/whisper-medium-mlx-4bit"  # mlx-whisper model
```

### Available mlx-whisper models

| Model | Size | Latency |
|---|---|---|
| `whisper-base-mlx` | 74 MB | ~0.3s |
| `whisper-small-mlx` | 244 MB | ~0.5s |
| `whisper-medium-mlx-4bit` | 400 MB | ~0.8s ← default |
| `whisper-large-v3-turbo` | 1.6 GB | ~1s |
| `whisper-large-v3-mlx` | 1.5 GB | ~1.5s |

**Want better accuracy?** On an M-Pro / M-Max (or any Mac with 16 GB+ RAM)
switch the default to `large-v3-turbo`:

```python
MODEL = "mlx-community/whisper-large-v3-turbo"
```

It is noticeably more accurate on uncommon words, names, and word endings,
and still runs comfortably under real time on Apple Silicon. The default
stays `medium` so it downloads fast and runs well on entry-level Macs (8 GB);
`turbo` is the recommended upgrade when you have the headroom.

> **Tip — domain vocabulary:** if you frequently dictate specialised terms
> (product names, jargon), add them to `INITIAL_PROMPT` in the script. Whisper
> uses it as a hint and spells those words correctly far more often.

### Supported languages

99 languages including Russian (`ru`), Ukrainian (`uk`), English (`en`), Spanish (`es`), and more.  
Set `LANGUAGE = None` for automatic detection.

---

## Autostart (macOS)

Configure the script to run in background and start at every login. Once set up, you can close the terminal completely.

### Quick setup — if you used the universal installer (Option A)

Already done. `whisper_install.py` created the LaunchAgent automatically at `~/Library/LaunchAgents/com.whisper-dictation.plist` with your real paths. Skip to **Critical step** below.

### Manual setup — if you used Manual install (Option B)

From inside the repo folder with venv activated:

```bash
# 1. Generate the LaunchAgent file with your real paths
sed -e "s|/PATH/TO/VENV|$(pwd)/venv|g" \
    -e "s|/PATH/TO/PROJECT|$(pwd)|g" \
    -e "s|YOUR_USERNAME|$(whoami)|g" \
    whisper-dictation-universal.plist \
    > ~/Library/LaunchAgents/com.whisper-dictation.plist

# 2. Load it (starts now + at every future login)
launchctl load ~/Library/LaunchAgents/com.whisper-dictation.plist

# 3. Verify it's running
launchctl list | grep whisper-dictation
# Expected: 12345  0  com.whisper-dictation
# Second column = 0 means running successfully
```

### Critical step — TWO permissions required

> ⚠️ The LaunchAgent needs **both** of these permissions granted to the **Homebrew Python.app** (not Terminal). Granting only one is the #1 cause of "everything looks fine but hotkey does nothing":
>
> 1. **Accessibility** — see [Accessibility permission](#accessibility-permission-macos) for path and steps
> 2. **Input Monitoring** — see [Input Monitoring permission](#input-monitoring-permission-macos-sonomasequoiatahoe--required-for-launchagent) (required on macOS 14+)
>
> Both paths point to the same Python.app file. After granting both, reload the LaunchAgent (`launchctl unload` + `launchctl load`).

### Management

```bash
# Stop
launchctl unload ~/Library/LaunchAgents/com.whisper-dictation.plist

# Start
launchctl load ~/Library/LaunchAgents/com.whisper-dictation.plist
```

Logs:

```
~/Library/Logs/whisper-dictation.log
~/Library/Logs/whisper-dictation.error.log
```

### Uninstall

```bash
launchctl unload ~/Library/LaunchAgents/com.whisper-dictation.plist
rm ~/Library/LaunchAgents/com.whisper-dictation.plist
```

---

## File transcription

Transcribe any audio or video file from the command line:

```bash
python whisper_transcribe.py <file> [options]
```

Supported formats: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`

> **Requires ffmpeg** for video files and non-WAV audio. Install once:
> ```bash
> brew install ffmpeg        # macOS
> sudo apt install ffmpeg    # Ubuntu / Debian
> ```

---

## Cost

**$0/month** for mlx-whisper and faster-whisper backends.  
Runs entirely on your machine. No API calls, no cloud, no subscriptions.

OpenAI API backend: ~$0.006 per minute of audio.

---

## Troubleshooting

### `source: no such file or directory: venv/bin/activate`

You tried to activate a venv that doesn't exist yet. Create it first:

```bash
python3 -m venv venv
source venv/bin/activate
```

### `Defaulting to user installation because normal site-packages is not writeable`

This message means your venv is **not active** — pip is installing into your global user site-packages instead.

Fix: stop, activate the venv, then re-run pip:

```bash
source venv/bin/activate
which python   # must point to ./venv/bin/python
pip install -r requirements.txt
```

The prompt should show `(venv)` prefix when active.

### `ModuleNotFoundError: No module named 'mlx_whisper'`

Two common causes:

**1. Python version too old.** `mlx-whisper` requires Python 3.10+ — on Python 3.9 `pip install` may silently skip it without a clear error.

```bash
python3 --version              # if 3.9 or lower → upgrade
brew install python@3.12
rm -rf venv                    # remove the broken venv
python3.12 -m venv venv        # recreate with the new Python
source venv/bin/activate
pip install -r requirements.txt
python -c "import mlx_whisper; print('OK')"
```

**2. Packages installed outside venv.** Check that venv is active:

```bash
which python                   # must point to ./venv/bin/python
```

If it points to a system Python — activate the venv and reinstall.

### Hotkey doesn't trigger recording

Grant Accessibility access. **Which binary depends on how you run the script** — see [Accessibility permission](#accessibility-permission-macos) for details:

- Running from terminal → grant to **Terminal**
- Running as LaunchAgent → grant to the **Homebrew Python.app** (find exact path with `ls -d /opt/homebrew/Cellar/python@3.12/*/Frameworks/Python.framework/Versions/3.12/Resources/Python.app`)

Restart the script (or `launchctl unload && launchctl load` for LaunchAgent) after granting.

### Hotkey works in terminal but silently fails after LaunchAgent setup

You granted Accessibility to Terminal, but LaunchAgent runs Python directly without Terminal as parent. Terminal's permission doesn't apply.

Fix: grant Accessibility to the Homebrew **Python.app** specifically (see [Accessibility permission](#accessibility-permission-macos) for the exact path). Then reload the LaunchAgent:

```bash
launchctl unload ~/Library/LaunchAgents/com.whisper-dictation.plist
launchctl load ~/Library/LaunchAgents/com.whisper-dictation.plist
```

### Hotkey doesn't catch — Accessibility is granted, error log is empty, but nothing happens

You have Python.app in Accessibility with the checkbox ON, the LaunchAgent is running (`launchctl list | grep whisper-dictation` shows status `0`), and `~/Library/Logs/whisper-dictation.error.log` has no `not trusted!` messages. But pressing the hotkey produces nothing — no log entries, no recording.

**Most likely cause on macOS Sonoma/Sequoia/Tahoe: missing Input Monitoring permission.** pynput on macOS 14+ requires **both** Accessibility AND Input Monitoring. Accessibility alone is not enough.

Fix: grant Input Monitoring to the **same Python.app** path (see [Input Monitoring permission](#input-monitoring-permission-macos-sonomasequoiatahoe--required-for-launchagent)). Then reload the LaunchAgent.

**Diagnostic trick — foreground test:** Stop the LaunchAgent (`launchctl unload ...`), then run the script directly in your terminal (`source venv/bin/activate; python whisper_dictate_macos_m.py`). If the hotkey **works in foreground** but **not as LaunchAgent**, the issue is permissions for Python.app specifically — Terminal already has these system-wide permissions, while LaunchAgent runs Python directly and needs them granted to Python.app.

**Toggle trick (if both permissions are granted but still doesn't work):** macOS occasionally "sticks" permissions in an inconsistent state. Toggle Python OFF in Accessibility (or Input Monitoring), wait 5 seconds, toggle it back ON, then reload the LaunchAgent.

### No microphone input / silence detected

Grant Microphone access:  
`System Settings → Privacy & Security → Microphone → enable Terminal`

### Cleanup if you installed packages globally by accident

If pip showed "Defaulting to user installation" and you want to remove those globally-installed packages:

```bash
pip3 uninstall sounddevice pynput pyperclip numpy mlx-whisper \
  cffi pycparser pyobjc-core pyobjc-framework-Cocoa \
  pyobjc-framework-ApplicationServices pyobjc-framework-CoreText \
  pyobjc-framework-Quartz
```

Then redo the install correctly inside an active venv.

---

## License

MIT

---

Created by [@DiscipleJC](https://github.com/DiscipleJC)
