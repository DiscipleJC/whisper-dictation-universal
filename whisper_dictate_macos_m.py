#!/usr/bin/env python3
"""
Whisper Dictation — push-to-talk + IPC daemon
- Right Option (hold) → push-to-talk
- HTTP IPC on 127.0.0.1:18765 для Hammerspoon (toggle / state / privacy)
"""

import http.server
import json
import socketserver
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx_whisper
import numpy as np
import pyperclip
import sounddevice as sd
from pynput import keyboard
from pynput.keyboard import Controller, Key, KeyCode

HOTKEY = Key.alt_r
LANGUAGE = None
MODEL = "mlx-community/whisper-medium-mlx-4bit"
RATE = 16000
STATS_LOG = Path.home() / "Library" / "Logs" / "whisper-dictation-stats.jsonl"
IPC_HOST = "127.0.0.1"
IPC_PORT = 18765

INITIAL_PROMPT = (
    "Claude Code, OpenAI, Python, JavaScript, TypeScript, GitHub, Docker, "
    "Kubernetes, API, REST, JSON, SQL, React, Node.js, Linux, macOS, Windows, "
    "Telegram, Slack, Zoom, YouTube, GPT, LLM, AI, ML, CPU, GPU, SSD, RAM."
)


class Daemon:
    def __init__(self):
        self.kb = Controller()
        self._lock = threading.Lock()
        self._frames = []
        self._state = "idle"  # idle | recording | transcribing
        self._stream = None
        self._privacy_mode = True  # default ON — стрим открыт только во время записи
        # При privacy_mode=True _open_persistent_stream() ничего не делает,
        # стрим откроется по требованию в start_recording().

    @property
    def state(self):
        return self._state

    @property
    def privacy_mode(self):
        return self._privacy_mode

    def _audio_callback(self, indata, frames, t, status):
        with self._lock:
            if self._state == "recording":
                self._frames.append(indata.copy())

    def _open_persistent_stream(self):
        if self._stream is not None or self._privacy_mode:
            return
        self._stream = sd.InputStream(
            samplerate=RATE, channels=1, dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    def _open_stream_for_recording(self):
        if self._stream is not None:
            return True
        try:
            self._stream = sd.InputStream(
                samplerate=RATE, channels=1, dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            return True
        except Exception as e:
            print(f"⚠️  PortAudio open failed: {e}", flush=True)
            # Cleanup частично-созданного стрима
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            return False

    def _close_stream(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_privacy_mode(self, enabled):
        self._privacy_mode = bool(enabled)
        if self._state == "idle":
            if self._privacy_mode:
                self._close_stream()
            else:
                self._open_persistent_stream()

    def start_recording(self):
        if self._state != "idle":
            return False
        with self._lock:
            self._frames = []
        if not self._open_stream_for_recording():
            print("⚠️  Запись не стартовала (PortAudio error)", flush=True)
            return False
        self._state = "recording"
        print("🎙  Запись...", flush=True)
        return True

    def stop_recording(self):
        if self._state != "recording":
            return False
        self._state = "transcribing"
        threading.Thread(target=self._do_transcribe, daemon=True).start()
        return True

    def toggle(self):
        if self._state == "idle":
            return self.start_recording()
        if self._state == "recording":
            return self.stop_recording()
        return False  # transcribing — игнор

    def _do_transcribe(self):
        try:
            with self._lock:
                frames = list(self._frames)
            if not frames:
                return
            audio = np.concatenate(frames).flatten()
            if audio.shape[0] < RATE * 0.3:
                print("⚠️  Слишком короткая запись\n", flush=True)
                return
            audio_sec = audio.shape[0] / RATE
            print("⏳ Транскрибирую...", flush=True)
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
                print("⚠️  Текст не распознан\n", flush=True)
                return
            print(f"✅ {text}\n", flush=True)
            self._log_stats(text, audio_sec, transcribe_sec, result.get("language"))
            pyperclip.copy(text)
            time.sleep(0.1)
            v_key = KeyCode.from_vk(9)
            with self.kb.pressed(Key.cmd):
                self.kb.press(v_key)
                self.kb.release(v_key)
        finally:
            self._state = "idle"
            if self._privacy_mode:
                self._close_stream()

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
        except OSError as e:
            print(f"⚠️  stats log failed: {e}", flush=True)


daemon = Daemon()


def stats_today():
    if not STATS_LOG.exists():
        return {"count": 0, "words": 0}
    today = datetime.now().strftime("%Y-%m-%d")
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
    return {"count": count, "words": words}


class IPCHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs):
        return  # silent

    def _respond(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/state":
            self._respond(200, {
                "state": daemon.state,
                "privacy_mode": daemon.privacy_mode,
                "stats_today": stats_today(),
            })
        elif self.path == "/stats/today":
            self._respond(200, stats_today())
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/toggle":
            ok = daemon.toggle()
            self._respond(200, {"ok": ok, "state": daemon.state})
        elif self.path == "/privacy/on":
            daemon.set_privacy_mode(True)
            self._respond(200, {"ok": True, "privacy_mode": True})
        elif self.path == "/privacy/off":
            daemon.set_privacy_mode(False)
            self._respond(200, {"ok": True, "privacy_mode": False})
        else:
            self._respond(404, {"error": "not found"})


def start_ipc_server():
    # allow_reuse_address — пережить TIME_WAIT после рестарта daemon
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    try:
        server = socketserver.ThreadingTCPServer((IPC_HOST, IPC_PORT), IPCHandler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"  IPC    : http://{IPC_HOST}:{IPC_PORT}", flush=True)
        return server
    except OSError as e:
        print(f"⚠️  IPC server not started: {e}", flush=True)
        return None


def on_press(key):
    if key == HOTKEY:
        daemon.start_recording()


def on_release(key):
    if key == HOTKEY:
        daemon.stop_recording()


print("=" * 45)
print("  Whisper Dictation  |  mlx-whisper M-series")
print("=" * 45)
print(f"  Hotkey : RIGHT OPTION (push-to-talk)")
print(f"  Язык   : {LANGUAGE or 'авто-детект'}")
print(f"  Модель : {MODEL.split('/')[-1]}")
start_ipc_server()
print("=" * 45)
print("  Ctrl+C для выхода\n")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
