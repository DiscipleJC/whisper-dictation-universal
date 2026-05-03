#!/usr/bin/env python3
"""
Whisper Dictation — Linux / Windows / macOS Intel
Backend: faster-whisper (local model, runs on CPU or NVIDIA GPU)
Hotkey: RIGHT ALT → speak → release → text appears
"""

import time
import threading
import numpy as np
import sounddevice as sd
import pyperclip
from faster_whisper import WhisperModel
from pynput import keyboard
from pynput.keyboard import Key, Controller

# --- Settings ---
HOTKEY    = Key.alt_r
MODEL     = "medium"      # tiny | base | small | medium | large-v3
LANGUAGE  = None          # None = auto-detect
RATE      = 16000
DEVICE    = "auto"        # auto | cpu | cuda
COMPUTE   = "auto"        # auto | int8 | float16 | float32
# ----------------

print("=" * 48)
print("  Whisper Dictation  |  faster-whisper")
print("=" * 48)
print(f"  Hotkey  : RIGHT ALT (hold)")
print(f"  Model   : {MODEL}")
print(f"  Language: {LANGUAGE or 'auto-detect'}")
print(f"  Device  : {DEVICE}")
print("=" * 48)
print("  Loading model — first run downloads it...")

model = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE)
print("  Model ready. Ctrl+C to quit.\n")

kb      = Controller()
_active = False
_frames = []
_lock   = threading.Lock()


def _callback(indata, frames, t, status):
    with _lock:
        if _active:
            _frames.append(indata.copy())


def _transcribe():
    with _lock:
        frames = list(_frames)

    if not frames:
        return

    audio = np.concatenate(frames).flatten()

    if audio.shape[0] < RATE * 0.3:
        print("⚠️  Too short\n", flush=True)
        return

    print("⏳ Transcribing...", flush=True)
    segments, _ = model.transcribe(
        audio,
        language=LANGUAGE,
        beam_size=5,
        vad_filter=True,
    )
    text = " ".join(s.text for s in segments).strip()

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
