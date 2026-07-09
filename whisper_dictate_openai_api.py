#!/usr/bin/env python3
"""
Whisper Dictation — OpenAI Whisper API backend
Works on any platform (macOS, Linux, Windows)
Requires: OPENAI_API_KEY in .env file
Hotkey: RIGHT ALT → speak → release → text appears
"""

import sys
if sys.version_info < (3, 10):
    sys.exit(
        f"\n❌ Python 3.10+ required.\n"
        f"   You have Python {sys.version_info.major}.{sys.version_info.minor}.\n"
        f"   See README → Install → step 1 for upgrade instructions.\n"
    )

import io
import time
import wave
import threading
import numpy as np
import sounddevice as sd
import pyperclip
from pathlib import Path
from pynput import keyboard
from pynput.keyboard import Key, Controller

# Load API key from .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            import os
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
            break

try:
    from openai import OpenAI
    client = OpenAI()
except Exception as e:
    print(f"❌ OpenAI client error: {e}")
    print("   Set OPENAI_API_KEY in .env file")
    raise SystemExit(1)

# --- Settings ---
HOTKEY   = Key.alt_r
LANGUAGE = None    # None = auto-detect | "ru" | "en" | etc.
RATE     = 16000
# ----------------

# Domain vocabulary hint so technical terms are spelled correctly.
# Whisper keeps only the LAST ~224 tokens of the prompt, so put the most
# important terms at the end.
INITIAL_PROMPT = (
    "Claude Code, OpenAI, Python, JavaScript, TypeScript, GitHub, Docker, "
    "Kubernetes, API, REST, JSON, SQL, React, Node.js, Linux, macOS, Windows, "
    "Telegram, Slack, Zoom, YouTube, GPT, LLM, AI, ML, CPU, GPU, SSD, RAM."
)

print("=" * 48)
print("  Whisper Dictation  |  OpenAI Whisper API")
print("=" * 48)
print(f"  Hotkey  : RIGHT ALT (hold)")
print(f"  Language: {LANGUAGE or 'auto-detect'}")
print(f"  Backend : OpenAI whisper-1 (cloud)")
print("=" * 48)
print("  Cost: ~$0.006 / minute of audio")
print("  Ctrl+C to quit.")
print()
print("  ℹ️  Hotkey not working? Verify BOTH permissions for Python:")
print("     System Settings → Privacy & Security → Accessibility AND Input Monitoring")
print()

kb      = Controller()
_active = False
_frames = []
_lock   = threading.Lock()


def _callback(indata, frames, t, status):
    with _lock:
        if _active:
            _frames.append(indata.copy())


def _to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    buf.seek(0)
    return buf.read()


def _transcribe():
    with _lock:
        frames = list(_frames)

    if not frames:
        return

    audio = np.concatenate(frames).flatten()

    if audio.shape[0] < RATE * 0.3:
        print("⚠️  Too short\n", flush=True)
        return

    print("⏳ Sending to OpenAI...", flush=True)
    wav_bytes = _to_wav_bytes(audio)

    try:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", wav_bytes, "audio/wav"),
            language=LANGUAGE,
            prompt=INITIAL_PROMPT,
        )
        text = response.text.strip()
    except Exception as e:
        print(f"❌ API error: {e}\n", flush=True)
        return

    if text:
        print(f"✅ {text}\n", flush=True)
        pyperclip.copy(text)
        time.sleep(0.1)
        kb.type(text)
    else:
        print("⚠️  No speech detected\n", flush=True)


def on_press(key):
    global _active, _frames
    if key == HOTKEY and not _active:
        with _lock:
            _frames = []
        _active = True
        print("🎙  Recording...", flush=True)


def on_release(key):
    global _active
    if key == HOTKEY and _active:
        _active = False
        threading.Thread(target=_transcribe, daemon=True).start()


with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", callback=_callback):
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
