#!/usr/bin/env python3
"""
Whisper Dictation — push-to-talk для любого приложения на macOS M-серии
Удерживай RIGHT OPTION → говори → отпусти → текст вставляется куда угодно
"""

import sys
import time
import threading
import numpy as np
import sounddevice as sd
import mlx_whisper
import pyperclip
from pynput import keyboard
from pynput.keyboard import Key, Controller


# --- Настройки ---
HOTKEY   = Key.alt_r                         # Правый Option (Alt)
LANGUAGE = None                              # None = авто-детект (русский + английский вместе)
MODEL    = "mlx-community/whisper-medium-mlx-4bit"  # medium 4-bit = качество medium, меньше памяти, чуть быстрее
RATE     = 16000
# -----------------

kb      = Controller()
_active = False
_frames = []
_lock   = threading.Lock()


def _callback(indata, frames, t, status):
    with _lock:
        if _active:
            _frames.append(indata.copy())


def _transcribe():
    global _active

    with _lock:
        frames = list(_frames)

    if not frames:
        return

    audio = np.concatenate(frames).flatten()

    if audio.shape[0] < RATE * 0.3:  # < 0.3 сек — слишком коротко, игнорируем
        print("⚠️  Слишком короткая запись\n", flush=True)
        return

    print("⏳ Транскрибирую...", flush=True)
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MODEL,
        language=LANGUAGE,
        no_speech_threshold=0.3,
        condition_on_previous_text=False,
        initial_prompt="Привет! Как дела? Всё хорошо. Claude Code, API, Python, GitHub, Telegram, AI-Crew, mlx_whisper.",
    )

    text = result["text"].strip()
    if text:
        print(f"✅ {text}\n", flush=True)
        pyperclip.copy(text)
        time.sleep(0.1)
        kb.type(text)
    else:
        print("⚠️  Текст не распознан\n", flush=True)


def on_press(key):
    global _active, _frames
    if key == HOTKEY and not _active:
        with _lock:
            _frames = []
        _active = True
        print("🎙  Запись...", flush=True)


def on_release(key):
    global _active
    if key == HOTKEY and _active:
        _active = False
        threading.Thread(target=_transcribe, daemon=True).start()


print("=" * 45)
print("  Whisper Dictation  |  mlx-whisper M-series")
print("=" * 45)
print(f"  Hotkey : RIGHT OPTION (удерживай)")
print(f"  Язык   : {LANGUAGE or 'авто-детект'}")
print(f"  Модель : {MODEL.split('/')[-1]}")
print("=" * 45)
print("  Первый запуск скачает модель (~290MB)")
print("  Ctrl+C для выхода\n")

with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", callback=_callback):
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
