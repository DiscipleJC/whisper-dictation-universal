#!/usr/bin/env python3
"""
Whisper Dictation — push-to-talk + IPC daemon
- Right Option (hold) → push-to-talk
- HTTP IPC on 127.0.0.1:18765 для Hammerspoon (toggle / state / privacy)
"""

import sys
if sys.version_info < (3, 10):
    sys.exit(
        f"\n❌ Python 3.10+ required (mlx-whisper needs 3.10+).\n"
        f"   You have Python {sys.version_info.major}.{sys.version_info.minor}.\n"
        f"   See README → Install → step 1 for upgrade instructions:\n"
        f"   brew install python@3.12  &&  python3.12 -m venv venv\n"
    )

import ctypes
import ctypes.util
import http.server
import json
import os
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

# Освобождает PortAudio handle: после IDLE_RESTART_SEC без записей процесс
# завершает себя через os._exit(), LaunchAgent (KeepAlive=true) поднимает
# свежий PID с чистым Core Audio session — оранжевый индикатор гаснет.
# 900s (15 мин): модель дольше остаётся "тёплой" в памяти → меньше холодных
# стартов на нажатии. Освежённый процесс прогревает модель в фоне (prewarm_model),
# поэтому холодная загрузка ~5-7s после рестарта не попадает на хоткей.
IDLE_RESTART_SEC = 900
WATCHDOG_PERIOD_SEC = 5

# Pause media while recording and resume afterwards by sending the system
# Play/Pause media key — this controls whatever is playing system-wide
# (browser tabs like YouTube, Music, Spotify, etc.). Best-effort.
# Set AUTO_PAUSE_MEDIA = False in local_settings.py to disable.
AUTO_PAUSE_MEDIA = True

INITIAL_PROMPT = (
    "Claude Code, OpenAI, Python, JavaScript, TypeScript, GitHub, Docker, "
    "Kubernetes, API, REST, JSON, SQL, React, Node.js, Linux, macOS, Windows, "
    "Telegram, Slack, Zoom, YouTube, GPT, LLM, AI, ML, CPU, GPU, SSD, RAM."
)

# Optional local overrides — keep a personal model choice and private domain
# vocabulary out of version control. Create local_settings.py (gitignored;
# copy from local_settings.example.py) to set MODEL and/or extend INITIAL_PROMPT.
try:
    import local_settings as _local
    MODEL = getattr(_local, "MODEL", None) or MODEL
    AUTO_PAUSE_MEDIA = getattr(_local, "AUTO_PAUSE_MEDIA", AUTO_PAUSE_MEDIA)
    _extra_prompt = getattr(_local, "EXTRA_PROMPT", "")
    if _extra_prompt:
        INITIAL_PROMPT = f"{INITIAL_PROMPT} {_extra_prompt}"
except ImportError:
    pass


# Media-key support via pyobjc. Sending the system Play/Pause key controls
# whatever is currently playing (browser, Music, Spotify). Degrades silently
# if pyobjc is unavailable.
try:
    from AppKit import NSEvent as _NSEvent
    from Quartz import CGEventPost as _CGEventPost
    _MEDIA_OK = True
except Exception:
    _MEDIA_OK = False

_NX_KEYTYPE_PLAY = 16


def _send_media_play_pause():
    """Send the system Play/Pause media key. Returns True if sent."""
    if not _MEDIA_OK:
        return False
    try:
        for down in (True, False):
            ev = _NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                14, (0, 0), 0xA00 if down else 0xB00, 0, 0, None, 8,
                (_NX_KEYTYPE_PLAY << 16) | ((0xA if down else 0xB) << 8), -1)
            _CGEventPost(0, ev.CGEvent())
        return True
    except Exception:
        return False


# CoreAudio: is the default output device actively playing audio? Used to only
# toggle media when something is really playing (so pressing the hotkey while
# nothing plays won't accidentally start a paused track). Best-effort.
try:
    _CoreAudio = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
except Exception:
    _CoreAudio = None


class _AudioAddr(ctypes.Structure):
    _fields_ = [("sel", ctypes.c_uint32),
                ("scope", ctypes.c_uint32),
                ("elem", ctypes.c_uint32)]


def _fourcc(s):
    return (ord(s[0]) << 24) | (ord(s[1]) << 16) | (ord(s[2]) << 8) | ord(s[3])


def _audio_is_playing():
    """True if the default output device is actively playing audio.
    Fails open (returns True) so media control still works if detection breaks."""
    if _CoreAudio is None:
        return True
    try:
        a1 = _AudioAddr(_fourcc('dOut'), _fourcc('glob'), 0)
        dev = ctypes.c_uint32(0)
        sz = ctypes.c_uint32(4)
        if _CoreAudio.AudioObjectGetPropertyData(
                1, ctypes.byref(a1), 0, None, ctypes.byref(sz),
                ctypes.byref(dev)) != 0:
            return True
        a2 = _AudioAddr(_fourcc('gone'), _fourcc('glob'), 0)
        run = ctypes.c_uint32(0)
        sz2 = ctypes.c_uint32(4)
        if _CoreAudio.AudioObjectGetPropertyData(
                dev.value, ctypes.byref(a2), 0, None, ctypes.byref(sz2),
                ctypes.byref(run)) != 0:
            return True
        return bool(run.value)
    except Exception:
        return True


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
        self._last_activity = time.monotonic()
        self._media_toggled = False

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

    def _pause_media(self):
        """Pause whatever is playing — but only if audio is actually playing,
        so the hotkey never accidentally starts a paused track."""
        self._media_toggled = False
        if AUTO_PAUSE_MEDIA and _audio_is_playing() and _send_media_play_pause():
            self._media_toggled = True

    def _resume_media(self):
        """Resume by toggling Play/Pause again (only if we toggled it)."""
        if self._media_toggled:
            _send_media_play_pause()
            self._media_toggled = False

    def set_privacy_mode(self, enabled):
        self._privacy_mode = bool(enabled)
        self._last_activity = time.monotonic()
        if self._state == "idle":
            if self._privacy_mode:
                self._close_stream()
            else:
                self._open_persistent_stream()

    def seconds_since_activity(self):
        return time.monotonic() - self._last_activity

    def start_recording(self):
        if self._state != "idle":
            return False
        with self._lock:
            self._frames = []
        if not self._open_stream_for_recording():
            print("⚠️  Запись не стартовала (PortAudio error)", flush=True)
            return False
        self._state = "recording"
        self._last_activity = time.monotonic()
        print("🎙  Запись...", flush=True)
        self._pause_media()
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
            self._last_activity = time.monotonic()
            if self._privacy_mode:
                self._close_stream()
            self._resume_media()

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


def prewarm_model():
    """Загрузить веса модели в память в фоне при старте процесса.

    mlx_whisper кэширует модель после первого transcribe(), поэтому холодная
    загрузка (~5-7s чтения весов с диска в RAM) случается на первом вызове.
    Прогоняем короткий тихий буфер при запуске — холодный старт уходит в фон,
    а не на первое нажатие хоткея пользователя. Запускается и при каждом
    перезапуске демона (idle-watchdog), пока процесс простаивает."""
    def loop():
        try:
            t0 = time.monotonic()
            silent = np.zeros(int(RATE * 0.5), dtype=np.float32)
            mlx_whisper.transcribe(
                silent,
                path_or_hf_repo=MODEL,
                language=LANGUAGE,
                no_speech_threshold=0.3,
                condition_on_previous_text=False,
            )
            print(f"  Prewarm: модель в памяти за {time.monotonic() - t0:.1f}s",
                  flush=True)
        except Exception as e:
            print(f"⚠️  prewarm failed: {e}", flush=True)
    threading.Thread(target=loop, daemon=True).start()


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
        elif self.path == "/restart":
            self._respond(200, {"ok": True})
            # Defer exit so the HTTP response is fully flushed to the caller.
            def _exit_soon():
                time.sleep(0.15)
                print("🔄 /restart received — exiting for clean mic release", flush=True)
                os._exit(0)
            threading.Thread(target=_exit_soon, daemon=True).start()
        else:
            self._respond(404, {"error": "not found"})


def start_idle_watchdog():
    """После IDLE_RESTART_SEC без записей завершает процесс через os._exit().
    LaunchAgent (KeepAlive=true) поднимает свежий PID с чистым Core Audio session,
    оранжевый индикатор гаснет."""
    def loop():
        while True:
            time.sleep(WATCHDOG_PERIOD_SEC)
            if daemon.state != "idle":
                continue
            if daemon.seconds_since_activity() >= IDLE_RESTART_SEC:
                print(
                    f"⏰ idle {IDLE_RESTART_SEC}s — restart for clean mic release",
                    flush=True,
                )
                os._exit(0)
    threading.Thread(target=loop, daemon=True).start()


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
start_idle_watchdog()
prewarm_model()
print(f"  Idle restart: {IDLE_RESTART_SEC}s")
print("=" * 45)
print("  Ctrl+C для выхода")
print()
print("  ℹ️  Hotkey not working? Verify BOTH permissions for Python:")
print("     System Settings → Privacy & Security → Accessibility AND Input Monitoring")
print()

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
