# 🎙 Whisper Dictation Universal

Push-to-talk voice dictation for macOS, Linux, and Windows.  
Hold a hotkey → speak → release → text pastes into any app.

Works in: terminal, Telegram, WhatsApp, browser, Claude Code, Slack — anywhere.

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

## Backends

| Backend | Script | Platform | Cost | Requires |
|---|---|---|---|---|
| **mlx-whisper** | `whisper_dictate_macos_m.py` | macOS Apple Silicon (M1–M5) | Free | — |
| **faster-whisper** | `dictate_faster_whisper.py` | macOS Intel, Linux, Windows | Free | CPU / NVIDIA GPU |
| **OpenAI Whisper API** | `whisper_dictate_openai_api.py` | Any platform | ~$0.006/min | OpenAI API key |

> **Apple M1 / M2 / M3 / M4 / M5** — the `mlx-whisper` backend uses the Apple Neural Engine for fast, fully offline transcription.

---

## Default model

| Backend | Default model | Size | Latency |
|---|---|---|---|
| mlx-whisper | `whisper-medium-mlx-4bit` | ~400 MB | ~0.8s |
| faster-whisper | `medium` | ~500 MB | ~1.5s |
| OpenAI API | `whisper-1` (cloud) | — | ~1–2s |

Model is downloaded automatically on first run and cached locally.

---

## Requirements

- Python 3.10+
- Microphone access
- macOS: Accessibility permission (for global hotkey)

---

## Install

The universal installer detects your platform and backend automatically:

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

## Manual install

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# macOS Apple Silicon (M1–M5)
pip install sounddevice pynput pyperclip numpy mlx-whisper

# macOS Intel / Linux / Windows
pip install sounddevice pynput pyperclip numpy faster-whisper

# OpenAI API backend (any platform)
pip install sounddevice pynput pyperclip numpy openai
```

---

## Run

```bash
source venv/bin/activate
python whisper_dictate_macos_m.py    # macOS Apple Silicon
python dictate_faster_whisper.py     # macOS Intel / Linux / Windows
python whisper_dictate_openai_api.py # OpenAI API backend
```

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

## Accessibility permission (macOS)

pynput requires Accessibility access to listen for global hotkeys:

```
System Settings → Privacy & Security → Accessibility → + → add Terminal (or Python)
```

After granting — restart the LaunchAgent.

---

## File transcription

Transcribe any audio or video file from the command line:

```bash
python whisper_transcribe.py <file> [options]
```

Supported formats: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`

---

## Cost

**$0/month** for mlx-whisper and faster-whisper backends.  
Runs entirely on your machine. No API calls, no cloud, no subscriptions.

OpenAI API backend: ~$0.006 per minute of audio.

---

## License

MIT
