#!/usr/bin/env python3
"""
Whisper Dictation — push-to-talk + IPC daemon
- Right Option (hold) → push-to-talk
- HTTP IPC on 127.0.0.1:18765 for Hammerspoon (toggle / state / privacy)
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
import re
import socketserver
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import mlx_whisper
import numpy as np
import pyperclip
import sounddevice as sd
from pynput import keyboard
from pynput.keyboard import Controller, Key, KeyCode

HOTKEY = Key.alt_r
# Hands-free recording, for anything too long to hold a key through (a meeting,
# a long note). Holding the RIGHT Option for minutes turns every other key into
# Option+key and Cmd+Tab into "hide the app you are leaving". Two ways to start
# a recording that no key has to hold; both stop the same way:
#
#   TOGGLE_HOTKEY — a chord: tap it to start, tap it again to stop. Spelled as
#   modifiers + one key: "ctrl", "alt" (= LEFT Option; the right one stays
#   push-to-talk), "cmd", "shift", then a letter or digit. The key is matched
#   by its hardware position, so it works in any keyboard layout. "" disables.
#
#   Double tap — two quick taps of the push-to-talk key itself. A tap is a
#   press released within DOUBLE_TAP_MAX_HOLD_SEC; the second one has to come
#   within DOUBLE_TAP_SEC of that release. The price: a quick single tap ends
#   DOUBLE_TAP_SEC later than before (the recording keeps running while we
#   wait for a possible second tap, so nothing is cut off if it comes). A
#   normal hold-and-speak dictation is not delayed. DOUBLE_TAP_SEC = 0 disables.
#
# One tap of the push-to-talk key (or the chord, or IPC /toggle) stops a
# hands-free recording. All three are overridable in local_settings.py.
TOGGLE_HOTKEY = "ctrl+alt+r"
DOUBLE_TAP_SEC = 0.45
DOUBLE_TAP_MAX_HOLD_SEC = 0.3
LANGUAGE = None
# NOTE: not the 4-bit build — mlx-community's 4bit/8bit turbo repos ship
# model.safetensors, which mlx-whisper <= 0.4.3 cannot load (it expects
# weights.safetensors). See README "Available mlx-whisper models" for the
# local-copy workaround if you need the smaller model.
MODEL = "mlx-community/whisper-large-v3-turbo"
RATE = 16000
# Explicit block size: left at 0 (PortAudio's choice), macOS 26/27 delivers
# 15-sample blocks — ~1000 callbacks/s, and the 128-block pre-roll buffer then
# held only ~0.12 s instead of PREROLL_SEC. 512 samples = 32 ms per block.
BLOCKSIZE = 512
# A recording held at least this long that got no live audio at all means the
# mic stream has stalled (open, but Core Audio stopped calling back).
STALL_SEC = 0.5
STATS_LOG = Path.home() / "Library" / "Logs" / "whisper-dictation-stats.jsonl"
IPC_HOST = "127.0.0.1"
IPC_PORT = 18765

# Idle self-restart (os._exit + LaunchAgent KeepAlive relaunch) is DISABLED.
# Across sleep/wake launchd did not reliably relaunch, leaving the daemon dead
# and the hotkey silently unresponsive. Privacy mode already closes the mic
# stream when idle, so the restart is unnecessary. The /restart IPC endpoint
# still allows a manual restart. Set IDLE_RESTART_SEC > 0 to re-enable.
IDLE_RESTART_SEC = 0
WATCHDOG_PERIOD_SEC = 5

# Keep the mic stream "warm" for this many seconds after a dictation, and seed
# each recording with a short pre-roll, so back-to-back dictations aren't clipped
# by cold-start latency / a too-fast first word. 0 = close immediately (max
# privacy, but the very start of fast speech can be lost). Overridable.
KEEP_WARM_SEC = 8
PREROLL_SEC = 0.3

# Pause media while recording and resume afterwards by sending the system
# Play/Pause media key — this controls whatever is playing system-wide
# (browser tabs like YouTube, Music, Spotify, etc.). Best-effort.
#
# OFF by default: the media key only reaches a real player when something is
# actually playing, but the CoreAudio "is playing" gate (_audio_is_playing)
# is unreliable on macOS 15.4+/26 (kAudioDevicePropertyDeviceIsRunningSomewhere
# reports the device as busy even in silence), so the key fires on every
# dictation. With no now-playing app registered, macOS routes a stray
# Play/Pause key to *launching Music.app* ("Choose Music Library" dialog).
# Enable AUTO_PAUSE_MEDIA = True in local_settings.py if you want auto-pause
# and accept that trade-off.
AUTO_PAUSE_MEDIA = False

# Convert spoken punctuation commands ("новая строка", "запятая", "new line",
# "comma", ...) into real punctuation in the transcript. Off by default because
# some commands are also ordinary words (e.g. "точка зрения"). Enable in
# local_settings.py with SPOKEN_PUNCTUATION = True.
SPOKEN_PUNCTUATION = False

# Safety guard for long dictations. Auto-paste sends Cmd+V into whatever window
# is frontmost when you release the hotkey — during a multi-minute recording the
# focus has often drifted away from where you started, so the paste misfires
# (and _restore_clipboard then wipes the text). If the recorded audio is longer
# than this many seconds, DON'T auto-paste: leave the transcript on the
# clipboard (without restoring the previous clip) and show a notification, so
# you paste it deliberately with Cmd+V. 0 disables the guard (always auto-paste).
# Overridable in local_settings.py.
AUTO_PASTE_MAX_SEC = 300

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
    SPOKEN_PUNCTUATION = getattr(_local, "SPOKEN_PUNCTUATION", SPOKEN_PUNCTUATION)
    KEEP_WARM_SEC = getattr(_local, "KEEP_WARM_SEC", KEEP_WARM_SEC)
    AUTO_PASTE_MAX_SEC = getattr(_local, "AUTO_PASTE_MAX_SEC", AUTO_PASTE_MAX_SEC)
    TOGGLE_HOTKEY = getattr(_local, "TOGGLE_HOTKEY", TOGGLE_HOTKEY)
    DOUBLE_TAP_SEC = getattr(_local, "DOUBLE_TAP_SEC", DOUBLE_TAP_SEC)
    DOUBLE_TAP_MAX_HOLD_SEC = getattr(_local, "DOUBLE_TAP_MAX_HOLD_SEC", DOUBLE_TAP_MAX_HOLD_SEC)
    _extra_prompt = getattr(_local, "EXTRA_PROMPT", "")
    if _extra_prompt:
        INITIAL_PROMPT = f"{INITIAL_PROMPT} {_extra_prompt}"
except ImportError:
    pass


# ── Hands-free chord (TOGGLE_HOTKEY) ─────────────────────────────────────────
# The chord is recognised from the modifier state the OS reports at the moment
# the key arrives (CGEventSourceFlagsState / CGEventSourceKeyState — instant,
# non-blocking calls), not from our own bookkeeping of modifier presses: a lost
# modifier release can then never leave a "stuck" chord. pynput's
# darwin_intercept is NOT used on purpose — it runs AFTER on_press (so it would
# report the previous event's flags) and turns the tap from listen-only into an
# active one.
_FLAG_OF = {"ctrl": 0x40000, "alt": 0x80000, "cmd": 0x100000, "shift": 0x20000}
_FLAG_ALL = sum(_FLAG_OF.values())
_FLAG_LEFT_ALT, _FLAG_RIGHT_ALT = 0x20, 0x40   # NX_DEVICEL/RALTKEYMASK
_MODIFIER_ALIASES = {"control": "ctrl", "option": "alt", "opt": "alt",
                     "command": "cmd", "meta": "cmd"}
# macOS virtual key codes (hardware positions) for letters and digits. With
# Control held, pynput reports the key with no readable character, and the
# character also changes with the layout — the position does not.
_VK = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8,
    "v": 9, "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "9": 25, "7": 26,
    "8": 28, "0": 29, "o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38,
    "k": 40, "n": 45, "m": 46,
}
_LABELS = {"ctrl": "Ctrl", "alt": "Option", "cmd": "Cmd", "shift": "Shift"}
_VK_LEFT_ALT, _VK_RIGHT_ALT = 58, 61


def _parse_toggle_hotkey(spec):
    """"ctrl+alt+r" → (required flag mask, vk, label) or None when disabled or
    invalid. label = "Ctrl+Option+R" for messages."""
    spec = (spec or "").strip().lower()
    if not spec:
        return None
    parts = [p.strip() for p in spec.split("+") if p.strip()]
    if len(parts) < 2:
        return None
    *mods, key = parts
    mask, names = 0, []
    for m in mods:
        m = _MODIFIER_ALIASES.get(m, m)
        if m not in _FLAG_OF or m in names:
            return None
        mask |= _FLAG_OF[m]
        names.append(m)
    if key not in _VK:
        return None
    label = "+".join(_LABELS[n] for n in names) + "+" + key.upper()
    return (mask, _VK[key], label)


def _is_toggle_chord(key, flags, chord):
    """True when the pressed `key` with modifier `flags` completes the chord:
    exactly the required modifiers are down (no extra Cmd/Shift/…). "alt"
    means the LEFT Option: when the device bits say only the RIGHT one is down,
    that press belongs to push-to-talk, not to the chord."""
    if chord is None:
        return False
    mask, vk, _ = chord
    if getattr(key, "vk", None) != vk:
        return False
    if (flags & _FLAG_ALL) != mask:
        return False
    if mask & _FLAG_OF["alt"] and (flags & (_FLAG_LEFT_ALT | _FLAG_RIGHT_ALT)) == _FLAG_RIGHT_ALT:
        return False
    return True


def _current_flags():
    """Modifier flags the OS holds down right now (session state = hardware
    plus posted events), with the left/right Option device bits filled in from
    the per-key state. 0 when Quartz is unavailable, so the chord never fires
    by accident. Called inside the event-tap callback: must stay instant."""
    try:
        from Quartz import (CGEventSourceFlagsState, CGEventSourceKeyState,
                            kCGEventSourceStateCombinedSessionState as _S)
        flags = int(CGEventSourceFlagsState(_S)) & _FLAG_ALL
        if CGEventSourceKeyState(_S, _VK_LEFT_ALT):
            flags |= _FLAG_LEFT_ALT
        if CGEventSourceKeyState(_S, _VK_RIGHT_ALT):
            flags |= _FLAG_RIGHT_ALT
        return flags
    except Exception:
        return 0


_TOGGLE_CHORD = _parse_toggle_hotkey(TOGGLE_HOTKEY)
TOGGLE_HOTKEY_LABEL = _TOGGLE_CHORD[2] if _TOGGLE_CHORD else ""


def _notify(title, message):
    """Best-effort macOS notification via osascript. Never raises."""
    try:
        safe_title = title.replace('"', "'")
        safe_msg = message.replace('"', "'")
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{safe_msg}" with title "{safe_title}"'],
            check=False, capture_output=True, timeout=5,
        )
    except Exception:
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


# Spoken punctuation: phrase -> mark. Multi-word phrases must come before any
# single word they contain (e.g. "точка с запятой" before "точка").
_SPOKEN_PUNCT = [
    ("новый абзац", "\n\n"), ("новая строка", "\n"),
    ("точка с запятой", ";"), ("вопросительный знак", "?"),
    ("восклицательный знак", "!"), ("открыть скобку", "("),
    ("закрыть скобку", ")"), ("кавычки", '"'),
    ("запятая", ","), ("двоеточие", ":"), ("точка", "."), ("тире", "—"),
    ("new paragraph", "\n\n"), ("new line", "\n"),
    ("question mark", "?"), ("exclamation mark", "!"),
    ("exclamation point", "!"), ("full stop", "."),
    ("open paren", "("), ("close paren", ")"),
    ("semicolon", ";"), ("comma", ","), ("colon", ":"),
    ("period", "."), ("dash", "—"),
]
_SPOKEN_PUNCT_RE = [
    (re.compile(rf"\b{re.escape(p)}\b", re.IGNORECASE), s) for p, s in _SPOKEN_PUNCT
]


# Signature Whisper hallucinations — phrases the model "hears" in breath or
# noise ("Субтитры создавал DimaTorzok", "Thanks for watching"). They come out
# with HIGH avg_logprob (the model is confident — the phrases are frequent in
# its training data), so the probability-based segment filter cannot catch
# them; match the text instead. A segment is dropped only when it contains
# nothing but such a phrase.
_HALLUCINATION_RE = re.compile(
    r"^\s*(?:"
    r"субтитры\s+(?:сделал|делал|создавал|создал|подготовил)\s+\S+"
    r"|редактор субтитров\b.*|корректор\s+\S{1,20}"
    r"|продолжение следует"
    r"|спасибо за просмотр\w*"
    r"|подписывайтесь на\s.*канал.*"
    r"|(?:thank you|thanks) for watching\w*"
    r"|thank\s+you(?:\s+(?:so|very)\s+much)?"
    r"|thanks(?:\s+(?:so|very)\s+much)?"
    r"|bye(?:[-\s]*bye)?"
    r"|subtitles by\b.*|.*amara\.org.*"
    r")[\s.!?…,]*$",
    re.IGNORECASE,
)


def _apply_spoken_punctuation(text):
    """Replace spoken punctuation commands with real marks, then tidy spacing."""
    if not SPOKEN_PUNCTUATION:
        return text
    for rx, sym in _SPOKEN_PUNCT_RE:
        text = rx.sub(sym, text)
    text = re.sub(r"[ \t]+([,.;:!?)»])", r"\1", text)   # no space before these
    text = re.sub(r"([(«])[ \t]+", r"\1", text)          # no space after these
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)         # trim around newlines
    text = re.sub(r"[ \t]{2,}", " ", text)               # collapse double spaces
    return text.strip()


class Daemon:
    def __init__(self):
        self.kb = Controller()
        self._lock = threading.Lock()
        self._frames = []
        self._state = "idle"  # idle | recording | transcribing
        self._stream = None
        self._privacy_mode = True  # default ON — stream open only while recording
        # With privacy_mode=True _open_persistent_stream() is a no-op; the
        # stream is opened on demand in start_recording().
        self._last_activity = time.monotonic()
        self._media_toggled = False
        self._prebuf = deque(maxlen=128)
        # Guards opening/closing the stream, so the keep-warm watchdog can't
        # close it in the middle of start_recording() reusing it.
        self._stream_lock = threading.RLock()
        self._last_cb = 0.0       # monotonic time of the last audio callback
        self._rec_samples = 0     # live samples captured by this recording
        self._rec_started = 0.0
        self._rec_held = 0.0      # how long the hotkey was held
        self._dead_streak = 0     # recordings in a row with no live audio
        self._gen = 0             # id of the current stream; older ones are ignored
        self._abandoned = False   # a stream close hung inside PortAudio
        self._closing = 0         # stream shutdowns still running
        # True while a recording started by the chord, a double tap or IPC
        # /toggle is running: releasing the push-to-talk key must not end it.
        self._hands_free = False
        # Double tap: token (= _rec_started) of a recording whose quick first
        # tap was released and which is waiting DOUBLE_TAP_SEC for a second
        # one. The lock is only held for a few assignments, never across audio
        # or a notification.
        self._tap_pending = None
        self._tap_lock = threading.Lock()

    @property
    def state(self):
        return self._state

    @property
    def privacy_mode(self):
        return self._privacy_mode

    @property
    def hands_free(self):
        return self._hands_free

    def _audio_callback(self, gen, indata, frames, t, status):
        block = indata.copy()
        # Check and write under one lock: generation bumps happen under it
        # too, so a callback from a replaced stream can't slip in between.
        with self._lock:
            if gen != self._gen:
                return  # a replaced/abandoned stream woke up — not our audio
            self._last_cb = time.monotonic()
            self._prebuf.append(block)
            if self._state == "recording":
                self._frames.append(block)
                self._rec_samples += len(block)

    def _preroll_blocks(self):
        """Recent pre-buffer blocks (~PREROLL_SEC) so the start of speech isn't
        clipped when the stream was already warm. Empty if the stream was cold."""
        if PREROLL_SEC <= 0:
            return []
        need = int(PREROLL_SEC * RATE)
        out, total = [], 0
        for block in reversed(list(self._prebuf)):
            out.append(block)
            total += len(block)
            if total >= need:
                break
        out.reverse()
        return out

    def _new_stream(self):
        with self._lock:
            self._gen += 1
            gen = self._gen
        stream = sd.InputStream(
            samplerate=RATE, channels=1, dtype="float32", blocksize=BLOCKSIZE,
            callback=lambda *a: self._audio_callback(gen, *a),
        )
        try:
            stream.start()
        except Exception:
            with self._lock:
                self._gen += 1  # it may have half-started: drop its audio
                self._prebuf.clear()
            self._shutdown_stream(stream)
            raise
        self._last_cb = time.monotonic()
        return stream

    def _open_persistent_stream(self):
        with self._stream_lock:
            if self._stream is not None or self._privacy_mode:
                return
            if self._abandoned:
                self._restart_daemon(idle=True)  # an open would hang too
            self._stream = self._new_stream()

    def _stream_is_stale(self):
        """An open stream that has stopped calling back. A live stream calls
        back every BLOCKSIZE/RATE (32 ms); half a second of silence means
        Core Audio has dropped it."""
        return (self._stream is not None
                and time.monotonic() - self._last_cb > STALL_SEC)

    def _open_stream_for_recording(self):
        if self._stream is not None and self._stream_is_stale():
            print("⚠️  Warm mic stream stopped delivering audio — reopening",
                  flush=True)
            self._close_stream()
        if self._abandoned:
            # PortAudio is stuck in a hung close: opening a new stream would
            # block too — here on the hotkey thread, freezing every later key
            # press with no log line at all. Only a fresh process recovers.
            self._restart_daemon()
        if self._stream is not None:
            return True
        try:
            self._stream = self._new_stream()
            return True
        except Exception as e:
            print(f"⚠️  PortAudio open failed: {e}", flush=True)
            # Clean up a partially-created stream
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            return False

    def _close_stream(self):
        """Detach the stream first, then shut it down off-thread with a
        timeout: on a stalled Core Audio stream stop() can block forever,
        which used to freeze the keep-warm watchdog and leave the dead stream
        in place for every following recording ("Recording too short")."""
        with self._stream_lock:
            stream, self._stream = self._stream, None
            with self._lock:
                self._gen += 1  # silence its callback even if close() hangs
                # The next stream starts cold: its pre-roll must not replay
                # the tail of this one.
                self._prebuf.clear()
                if stream is not None:
                    self._closing += 1
            # Still under the stream lock: an open waits out the shutdown
            # (at most 2 s) and then sees _abandoned, instead of entering
            # PortAudio while abort() may be hung there.
            if stream is not None:
                self._shutdown_stream(stream, registered=True)

    def _shutdown_stream(self, stream, registered=False):
        """abort() + close() off-thread, waiting at most 2 s."""
        def shutdown():
            try:
                stream.abort()   # abort, not stop: don't wait to drain buffers
                stream.close()
            except Exception as e:
                print(f"⚠️  mic stream close failed: {e}", flush=True)
            finally:
                with self._lock:
                    self._closing -= 1

        if not registered:
            with self._lock:
                self._closing += 1

        t = threading.Thread(target=shutdown, daemon=True)
        t.start()
        t.join(timeout=2)
        if t.is_alive():
            # PortAudio is now stuck in that thread: an in-process reset
            # (Pa_Terminate) could hang on it too, so recovery must restart.
            self._abandoned = True
            print("⚠️  mic stream close hung — abandoned it", flush=True)

    @staticmethod
    def _restart_daemon(idle=False):
        """Exit so the LaunchAgent (KeepAlive) relaunches a fresh process with
        a clean Core Audio session — the recovery that always works.
        idle=True: nobody is dictating, so restart quietly (no notification)."""
        if idle:
            print("⚠️  Mic stream close hung while idle — restarting the "
                  "daemon", flush=True)
            os._exit(1)
        print("⚠️  Microphone not responding — restarting the daemon, "
              "please repeat in a few seconds", flush=True)
        _notify("Whisper Dictation",
                "Microphone stopped responding — restarting, "
                "please repeat in a few seconds")
        os._exit(1)

    def _reset_audio(self):
        """Reinitialise PortAudio to recover a stale Core Audio session after
        sleep (recordings come back empty). Avoids a full process restart."""
        try:
            # Held through terminate, so no stream can open or start closing
            # while PortAudio is torn down.
            with self._stream_lock:
                self._close_stream()
                if self._abandoned or self._closing:
                    # A close is hung or still running (maybe on another
                    # thread): Pa_Terminate would race or hang on it.
                    self._restart_daemon()
                sd._terminate()
                sd._initialize()
            print("🔁 Audio reinitialised (after sleep)", flush=True)
        except Exception as e:
            print(f"⚠️  audio reset failed: {e}", flush=True)

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
        with self._stream_lock:
            return self._start_recording_locked()

    def _start_recording_locked(self):
        if self._state != "idle":
            return False
        with self._lock:
            # A stale stream's pre-buffer is old audio — don't seed with it.
            self._frames = [] if self._stream_is_stale() else self._preroll_blocks()
            self._rec_samples = 0
        if not self._open_stream_for_recording():
            print("⚠️  Recording did not start (PortAudio error)", flush=True)
            return False
        self._rec_started = time.monotonic()
        self._tap_pending = None
        self._state = "recording"
        self._last_activity = time.monotonic()
        print("🎙  Recording...", flush=True)
        self._pause_media()
        return True

    def stop_recording(self):
        if self._state != "recording":
            return False
        self._rec_held = time.monotonic() - self._rec_started
        self._hands_free = False
        self._tap_pending = None
        self._state = "transcribing"
        threading.Thread(target=self._do_transcribe, daemon=True).start()
        return True

    def toggle(self):
        """Hands-free toggle (chord or IPC /toggle): start a recording that no
        key has to hold, or stop the one running. Notifications go to a thread:
        osascript can take a moment and this runs in the hotkey callback."""
        if self._state == "idle":
            if not self.start_recording():
                return False
            self._hands_free = True
            hint = TOGGLE_HOTKEY_LABEL or "toggle"
            print(f"🎙  Hands-free recording — {hint} again to stop", flush=True)
            self._notify_async("Whisper Dictation",
                               f"Recording… press {hint} again to stop")
            return True
        if self._state == "recording":
            was_hands_free = self._hands_free
            if not self.stop_recording():
                return False
            if was_hands_free:
                self._notify_async("Whisper Dictation",
                                   "Recording stopped — transcribing")
            return True
        return False  # transcribing — ignore

    def hotkey_down(self):
        """The push-to-talk key went down. Three meanings, by state: it ends a
        hands-free recording, it is the second tap of a double tap (the
        recording the first tap started simply carries on, hands-free), or it
        starts an ordinary push-to-talk recording."""
        with self._tap_lock:
            if self._state == "recording" and self._hands_free:
                if not self.stop_recording():
                    return False
                self._notify_async("Whisper Dictation",
                                   "Recording stopped — transcribing")
                return True
            if self._state == "recording" and self._tap_pending is not None:
                self._tap_pending = None
                self._hands_free = True
                print("🎙  Hands-free recording (double tap) — tap the hotkey "
                      "once to stop", flush=True)
                self._notify_async("Whisper Dictation",
                                   "Recording… tap right Option once to stop")
                return True
        return self.start_recording()

    def hotkey_up(self):
        """The push-to-talk key was released. A hands-free recording ignores
        it. A quick tap does not stop yet — it waits DOUBLE_TAP_SEC for a
        second tap, on a timer thread; anything held longer stops at once."""
        with self._tap_lock:
            if self._hands_free or self._state != "recording":
                return False
            held = time.monotonic() - self._rec_started
            if DOUBLE_TAP_SEC > 0 and held < DOUBLE_TAP_MAX_HOLD_SEC:
                token = self._rec_started
                self._tap_pending = token
                t = threading.Timer(DOUBLE_TAP_SEC, self._tap_expired, args=(token,))
                t.daemon = True
                t.start()
                return True
            return self.stop_recording()

    def _tap_expired(self, token):
        """Timer thread: no second tap came, so the quick tap ends the way a
        released push-to-talk does. The token keeps a stale timer from
        stopping a newer recording."""
        with self._tap_lock:
            if (self._tap_pending is not None and self._tap_pending == token
                    and self._state == "recording" and not self._hands_free):
                self.stop_recording()

    @staticmethod
    def _notify_async(title, message):
        threading.Thread(target=_notify, args=(title, message), daemon=True).start()

    def _do_transcribe(self):
        try:
            with self._lock:
                frames = list(self._frames)
            if self._rec_held >= STALL_SEC and self._rec_samples == 0:
                # The hotkey was held, but the mic delivered nothing (only the
                # pre-roll, if any): Core Audio has wedged the stream. First
                # try the in-process reset; if the next recording is dead too,
                # restart — a fresh process always recovers, and KeepAlive
                # relaunches the daemon in ~3 s.
                self._dead_streak += 1
                if self._dead_streak == 1 and not self._abandoned:
                    print("⚠️  Microphone delivered no audio — resetting audio, "
                          "please repeat", flush=True)
                    self._reset_audio()
                    return
                self._restart_daemon()
            if self._rec_samples:
                self._dead_streak = 0
            if not frames:
                if self._abandoned:
                    self._restart_daemon()
                print("⚠️  Empty recording — resetting audio after sleep, please repeat",
                      flush=True)
                self._reset_audio()
                return
            audio = np.concatenate(frames).flatten()
            if audio.shape[0] < RATE * 0.3:
                print("⚠️  Recording too short\n", flush=True)
                return
            audio_sec = audio.shape[0] / RATE
            print("⏳ Transcribing...", flush=True)
            t0 = time.monotonic()
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=MODEL,
                language=LANGUAGE,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                initial_prompt=INITIAL_PROMPT,
            )
            transcribe_sec = time.monotonic() - t0
            # Drop hallucinated segments before pasting. Thresholds are the
            # official openai/whisper values: no_speech_prob > 0.6 with
            # avg_logprob < -1.0 = speech invented over breath/noise ("Thanks
            # for watching"); compression_ratio >= 2.4 = a repetition loop.
            segments = [
                s for s in result["segments"]
                if not (s["no_speech_prob"] > 0.6 and s["avg_logprob"] < -1.0)
                and s["compression_ratio"] < 2.4
                and not _HALLUCINATION_RE.match(s["text"].strip())
            ]
            text = "".join(s["text"] for s in segments).strip()
            if not text:
                print("⚠️  No text recognised\n", flush=True)
                return
            text = _apply_spoken_punctuation(text)
            print(f"✅ {text}\n", flush=True)
            self._log_stats(text, audio_sec, transcribe_sec, result.get("language"))
            # Trailing space so consecutive dictations don't run together.
            payload = text + " "
            if AUTO_PASTE_MAX_SEC and audio_sec > AUTO_PASTE_MAX_SEC:
                # Long recording: focus has likely drifted from where dictation
                # started, so a blind Cmd+V would misfire. Leave the transcript
                # on the clipboard (do NOT restore the previous clip — that would
                # wipe it) and notify, so the user pastes it deliberately.
                pyperclip.copy(payload)
                mins = audio_sec / 60
                print(f"📋 Long recording ({mins:.1f} min) — transcript left on "
                      f"clipboard, press Cmd+V to paste\n", flush=True)
                _notify("Whisper Dictation",
                        f"Long transcript ready ({mins:.1f} min) — press Cmd+V to paste")
            else:
                # Paste at the cursor via the clipboard, then hand the clipboard
                # back to whatever the user had before, so dictation never leaves
                # its text stuck in the clipboard (hijacking the next Cmd+V).
                try:
                    prev_clip = pyperclip.paste()
                except Exception:
                    prev_clip = None
                pyperclip.copy(payload)
                time.sleep(0.1)
                v_key = KeyCode.from_vk(9)
                with self.kb.pressed(Key.cmd):
                    self.kb.press(v_key)
                    self.kb.release(v_key)
                if prev_clip is not None:
                    threading.Thread(
                        target=self._restore_clipboard,
                        args=(prev_clip, payload),
                        daemon=True,
                    ).start()
        finally:
            self._state = "idle"
            self._last_activity = time.monotonic()
            # Keep the stream warm for KEEP_WARM_SEC (closed later by the
            # keep-warm watchdog); close immediately only if keep-warm is off.
            if self._privacy_mode and KEEP_WARM_SEC <= 0:
                self._close_stream()
            self._resume_media()

    @staticmethod
    def _restore_clipboard(prev_clip, ours, delay=0.4):
        """Give the frontmost app time to read the pasteboard for Cmd+V, then
        put the user's previous clipboard back. Guarded: only restore if the
        clipboard still holds OUR dictated text — if the user copied something
        else in the meantime, leave their copy alone (no clobber)."""
        time.sleep(delay)
        try:
            if pyperclip.paste() == ours:
                pyperclip.copy(prev_clip)
        except Exception:
            pass

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
    """Load the model weights into memory in the background at startup.

    mlx_whisper caches the model after the first transcribe(), so the cold load
    (~5-7s of reading weights from disk into RAM) happens on the first call.
    Running a short silent buffer at startup moves that cold start into the
    background instead of onto the user's first hotkey press. Also runs on every
    daemon restart while the process is idle."""
    def loop():
        try:
            t0 = time.monotonic()
            silent = np.zeros(int(RATE * 0.5), dtype=np.float32)
            mlx_whisper.transcribe(
                silent,
                path_or_hf_repo=MODEL,
                language=LANGUAGE,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
            )
            print(f"  Prewarm: model in memory in {time.monotonic() - t0:.1f}s",
                  flush=True)
        except Exception as e:
            print(f"⚠️  prewarm failed: {e}", flush=True)
    threading.Thread(target=loop, daemon=True).start()


def stats_today():
    if not STATS_LOG.exists():
        return {"count": 0, "words": 0, "avg_sec": None}
    today = datetime.now().strftime("%Y-%m-%d")
    count = 0
    words = 0
    total_sec = 0.0
    for line in STATS_LOG.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("ts", "").startswith(today):
            count += 1
            words += entry.get("words", 0)
            total_sec += entry.get("transcribe_sec", 0) or 0
    avg_sec = round(total_sec / count, 2) if count else None
    return {"count": count, "words": words, "avg_sec": avg_sec}


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
                "model": MODEL,
                "language": LANGUAGE or "auto",
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
    """After IDLE_RESTART_SEC with no recordings, exit via os._exit(); the
    LaunchAgent (KeepAlive=true) relaunches a fresh PID with a clean Core Audio
    session so the orange mic indicator clears. Disabled when IDLE_RESTART_SEC <= 0."""
    if IDLE_RESTART_SEC <= 0:
        return
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


def start_keepwarm_watchdog():
    """Close the warm mic stream after KEEP_WARM_SEC of inactivity (privacy)."""
    if KEEP_WARM_SEC <= 0:
        return
    def loop():
        while True:
            time.sleep(2)
            try:
                # Under the stream lock, so a hotkey press can't reuse the
                # stream while it is being closed.
                with daemon._stream_lock:
                    if (daemon.state == "idle" and daemon.privacy_mode
                            and daemon._stream is not None
                            and daemon.seconds_since_activity() >= KEEP_WARM_SEC):
                        daemon._close_stream()
                        if daemon._abandoned:
                            # Restart now, while idle, rather than on the
                            # next hotkey press.
                            daemon._restart_daemon(idle=True)
            except Exception as e:
                # Never let one failure kill the watchdog thread for good.
                print(f"⚠️  keep-warm watchdog: {e}", flush=True)
    threading.Thread(target=loop, daemon=True).start()


def start_ipc_server():
    # allow_reuse_address — survive TIME_WAIT after a daemon restart
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
        daemon.hotkey_down()
    elif _TOGGLE_CHORD is not None and getattr(key, "vk", None) == _TOGGLE_CHORD[1] \
            and _is_toggle_chord(key, _current_flags(), _TOGGLE_CHORD):
        daemon.toggle()


def on_release(key):
    # What a release means (stop, wait for a second tap, or nothing during a
    # hands-free recording) is decided in Daemon.hotkey_up().
    if key == HOTKEY:
        daemon.hotkey_up()


print("=" * 45)
print("  Whisper Dictation  |  mlx-whisper M-series")
print("=" * 45)
print(f"  Hotkey : RIGHT OPTION (push-to-talk)")
print(f"  Hands-free: {TOGGLE_HOTKEY_LABEL + ' — tap to start, tap again to stop' if _TOGGLE_CHORD else 'chord off (TOGGLE_HOTKEY empty or invalid: ' + repr(TOGGLE_HOTKEY) + ')'}")
print(f"  Double tap: {'RIGHT OPTION twice — hands-free recording, one more tap stops it' if DOUBLE_TAP_SEC > 0 else 'off (DOUBLE_TAP_SEC = 0)'}")
print(f"  Language: {LANGUAGE or 'auto-detect'}")
print(f"  Model   : {MODEL.split('/')[-1]}")
start_ipc_server()
start_idle_watchdog()
start_keepwarm_watchdog()
prewarm_model()
print(f"  Idle restart: {'disabled' if IDLE_RESTART_SEC <= 0 else str(IDLE_RESTART_SEC) + 's'}")
print(f"  Keep-warm   : {'off' if KEEP_WARM_SEC <= 0 else str(KEEP_WARM_SEC) + 's (preroll ' + str(PREROLL_SEC) + 's)'}")
print(f"  Auto-paste  : {'always' if AUTO_PASTE_MAX_SEC <= 0 else 'up to ' + str(AUTO_PASTE_MAX_SEC) + 's, longer → clipboard + notify'}")
print("=" * 45)
print("  Ctrl+C to quit")
print()
print("  ℹ️  Hotkey not working? Verify BOTH permissions for Python:")
print("     System Settings → Privacy & Security → Accessibility AND Input Monitoring")
print()

# background agent: no Dock icon (the Python "rocket"), no app switcher
try:
    from AppKit import NSApplication
    NSApplication.sharedApplication().setActivationPolicy_(1)  # Accessory
except Exception:
    pass

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
