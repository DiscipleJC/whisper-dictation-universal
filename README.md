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

```bash
source venv/bin/activate
python whisper_dictate_macos_m.py    # macOS Apple Silicon
python dictate_faster_whisper.py     # macOS Intel / Linux / Windows
python whisper_dictate_openai_api.py # OpenAI API backend
```

> **First run only:** wait 1–2 minutes while the model downloads. You'll see the startup header when it's ready.

---

## Accessibility permission (macOS)

pynput requires Accessibility access to listen for global hotkeys:

```
System Settings → Privacy & Security → Accessibility → + → add Terminal (or Python)
```

After granting — restart the script or LaunchAgent.

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
| `whisper-large-v3-mlx` | 1.5 GB | ~1.5s |

### Supported languages

99 languages including Russian (`ru`), Ukrainian (`uk`), English (`en`), Spanish (`es`), and more.  
Set `LANGUAGE = None` for automatic detection.

---

## Autostart (macOS)

The installer creates a LaunchAgent at:

```
~/Library/LaunchAgents/com.whisper-dictation.plist
```

Manage it manually:

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

Grant Accessibility access:  
`System Settings → Privacy & Security → Accessibility → + → add Terminal (or Python)`

Restart the script after granting permission.

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
