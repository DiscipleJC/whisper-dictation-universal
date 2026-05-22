#!/usr/bin/env python3
"""
Whisper Dictation — menu bar app (prototype)
- Right Option (hold) = push-to-talk (как раньше)
- Click icon → menu → Toggle recording = режим Google Assistant (click to start, click to stop)
- Privacy mode = микрофон физически открыт только во время записи
"""

import json
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path

import mlx_whisper
import numpy as np
import pyperclip
import rumps
import sounddevice as sd
from AppKit import NSImage, NSImageSymbolConfiguration
from pynput import keyboard
from pynput.keyboard import Controller, Key, KeyCode

HOTKEY = Key.alt_r
LANGUAGE = None
MODEL = "mlx-community/whisper-medium-mlx-4bit"
RATE = 16000
STATS_LOG = Path.home() / "Library" / "Logs" / "whisper-dictation-stats.jsonl"

SYMBOL_IDLE = "mic"
SYMBOL_REC = "mic.fill"
SYMBOL_TX = "ellipsis"
SYMBOL_PRIVACY_IDLE = "mic.slash"

FALLBACK_TITLE = "🎙"  # if SF Symbols unavailable

INITIAL_PROMPT = (
    "Claude Code, OpenAI, Python, JavaScript, TypeScript, GitHub, Docker, "
    "Kubernetes, API, REST, JSON, SQL, React, Node.js, Linux, macOS, Windows, "
    "Telegram, Slack, Zoom, YouTube, GPT, LLM, AI, ML, CPU, GPU, SSD, RAM."
)


class DictationCore:
    def __init__(self, on_state_change):
        self.kb = Controller()
        self._stream = None
        self._frames = []
        self._lock = threading.Lock()
        self._active = False
        self.privacy_mode = False
        self.on_state_change = on_state_change

    def _audio_callback(self, indata, frames, t, status):
        with self._lock:
            if self._active:
                self._frames.append(indata.copy())

    def _open_stream(self):
        if self._stream is None:
            self._stream = sd.InputStream(
                samplerate=RATE, channels=1, dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()

    def _close_stream(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def apply_idle_stream_policy(self):
        if self.privacy_mode:
            self._close_stream()
        else:
            self._open_stream()

    def set_privacy_mode(self, enabled):
        self.privacy_mode = enabled
        if not self._active:
            self.apply_idle_stream_policy()

    def is_active(self):
        return self._active

    def start_recording(self):
        if self._active:
            return
        with self._lock:
            self._frames = []
        self._open_stream()
        self._active = True
        self.on_state_change("recording")

    def stop_recording(self):
        if not self._active:
            return
        self._active = False
        self.on_state_change("transcribing")
        threading.Thread(target=self._do_transcribe, daemon=True).start()

    def toggle(self):
        if self._active:
            self.stop_recording()
        else:
            self.start_recording()

    def _do_transcribe(self):
        try:
            with self._lock:
                frames = list(self._frames)
            if not frames:
                return
            audio = np.concatenate(frames).flatten()
            if audio.shape[0] < RATE * 0.3:
                return
            audio_sec = audio.shape[0] / RATE
            t0 = time.monotonic()
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=MODEL,
                language=LANGUAGE,
                no_speech_threshold=0.3,
                condition_on_previous_text=False,
                initial_prompt=INITIAL_PROMPT,
            )
            transcribe_sec = time.monotonic() - t0
            text = result["text"].strip()
            if not text:
                return
            self._log_stats(text, audio_sec, transcribe_sec, result.get("language"))
            pyperclip.copy(text)
            time.sleep(0.1)
            v_key = KeyCode.from_vk(9)
            with self.kb.pressed(Key.cmd):
                self.kb.press(v_key)
                self.kb.release(v_key)
        finally:
            if self.privacy_mode:
                self._close_stream()
            self.on_state_change("idle")

    def _log_stats(self, text, audio_sec, transcribe_sec, language):
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "words": len(text.split()),
            "chars": len(text),
            "audio_sec": round(audio_sec, 2),
            "transcribe_sec": round(transcribe_sec, 2),
            "language": language,
        }
        try:
            with STATS_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass


def today_stats():
    if not STATS_LOG.exists():
        return 0, 0
    today = date.today().isoformat()
    count = 0
    words = 0
    for line in STATS_LOG.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("ts", "").startswith(today):
            count += 1
            words += entry.get("words", 0)
    return count, words


def _sf_symbol_image(name, point_size=16.0):
    """Get SF Symbol as NSImage. Returns None if symbol unavailable."""
    img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, None)
    if img is None:
        return None
    cfg = NSImageSymbolConfiguration.configurationWithPointSize_weight_(
        point_size, 0.0  # NSFontWeightRegular
    )
    img = img.imageWithSymbolConfiguration_(cfg)
    img.setTemplate_(True)  # adapts to dark/light menu bar
    return img


class WhisperMenuBarApp(rumps.App):
    def __init__(self):
        super().__init__("WhisperDictation", quit_button=None)
        self.core = DictationCore(on_state_change=self._on_state_change)

        self.toggle_item = rumps.MenuItem(
            "Start recording", callback=self._on_toggle_click
        )
        self.privacy_item = rumps.MenuItem(
            "Privacy mode (mic off when idle)", callback=self._on_privacy_toggle
        )
        self.stats_item = rumps.MenuItem(
            "Today's stats", callback=self._on_stats_click
        )
        self.menu = [
            self.toggle_item,
            None,
            self.stats_item,
            self.privacy_item,
            None,
            rumps.MenuItem("Quit", callback=rumps.quit_application),
        ]

        self.core.apply_idle_stream_policy()

        self._kbd_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._kbd_listener.start()

    def _on_toggle_click(self, _sender):
        self.core.toggle()

    def _on_stats_click(self, _sender):
        count, words = today_stats()
        rumps.notification(
            title="Whisper Dictation",
            subtitle=f"Today: {count} recordings, {words} words",
            message="",
        )

    def _on_privacy_toggle(self, sender):
        sender.state = not sender.state
        self.core.set_privacy_mode(bool(sender.state))
        # Refresh icon if currently idle (mic ↔ mic.slash)
        if not self.core.is_active():
            self._apply_icon("idle")

    def _on_state_change(self, state):
        labels = {"idle": "Start recording", "recording": "Stop recording",
                  "transcribing": "Transcribing…"}
        if state in labels:
            self.toggle_item.title = labels[state]
        self._apply_icon(state)

    def _apply_icon(self, state):
        if state == "recording":
            symbol = SYMBOL_REC
        elif state == "transcribing":
            symbol = SYMBOL_TX
        else:  # idle
            symbol = SYMBOL_PRIVACY_IDLE if self.core.privacy_mode else SYMBOL_IDLE
        img = _sf_symbol_image(symbol)
        if img is not None:
            self._icon_nsimage = img
            self._title = ""  # ensure no text shows alongside the image
            try:
                self._nsapp.setStatusBarIcon()
            except AttributeError:
                pass
        else:
            self.title = FALLBACK_TITLE

    def _on_key_press(self, key):
        if key == HOTKEY and not self.core.is_active():
            self.core.start_recording()

    def _on_key_release(self, key):
        if key == HOTKEY and self.core.is_active():
            self.core.stop_recording()


if __name__ == "__main__":
    # Ensure macOS treats this as an Accessory app (status-bar only, no Dock icon).
    # Without this, status item may not render when launched from a non-bundled
    # terminal process.
    try:
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
        NSApplication.sharedApplication().setActivationPolicy_(
            NSApplicationActivationPolicyAccessory
        )
    except Exception as e:
        print(f"⚠️  could not set activation policy: {e}", flush=True)

    WhisperMenuBarApp().run()
