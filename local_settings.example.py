"""Local overrides for whisper_dictate_macos_m.py (example).

Copy this file to `local_settings.py` (which is gitignored) to keep a personal
model choice and private domain vocabulary out of version control:

    cp local_settings.example.py local_settings.py

Then edit local_settings.py. Both settings below are optional — remove either
one to fall back to the script's default.
"""

# Override the transcription model — an HF repo id or an absolute path to a
# local model folder. The script's default is whisper-large-v3-turbo (~1.6 GB);
# on a tight-memory Mac, whisper-medium-mlx-4bit (~400 MB) is the lighter
# fallback. (The 4-bit turbo build needs a local-copy workaround with
# mlx-whisper <= 0.4.3 — see README "Available mlx-whisper models".)
MODEL = "mlx-community/whisper-medium-mlx-4bit"

# Extra words appended to INITIAL_PROMPT — your own product names, jargon, etc.
# Whisper uses it as a hint and spells these terms correctly far more often.
# Note: Whisper keeps only the LAST ~224 tokens of the combined prompt, so put
# the most important terms at the END of this string.
EXTRA_PROMPT = "MyProduct, MyService, SomeJargon."

# Pause media while dictating and resume afterwards, by sending the system
# Play/Pause key (works with browser tabs/YouTube, Music, Spotify). Default False:
# on macOS 15.4+/26 the "is something playing" gate is unreliable, so the key
# can fire in silence and launch Music.app ("Choose Music Library"). Set to True
# to opt back in.
AUTO_PAUSE_MEDIA = False

# Convert spoken punctuation ("новая строка", "запятая", "new line", "comma",
# ...) into real marks. Default False (some commands are also ordinary words,
# e.g. "точка зрения"). Set True to enable.
SPOKEN_PUNCTUATION = True

# Keep the mic stream warm for N seconds after a dictation (+ a short pre-roll)
# so the next dictation's first word isn't clipped by cold-start latency.
# Trade-off: the mic indicator stays on a few seconds after you stop.
# Set 0 to close the mic immediately after each dictation (max privacy).
KEEP_WARM_SEC = 8

# Safety guard for long dictations. Auto-paste sends Cmd+V into whatever window
# is frontmost when you release the hotkey; during a multi-minute recording the
# focus has usually drifted away from where you started, so the paste misfires
# into the wrong window (and the clipboard is then restored, wiping the text).
# If the recorded audio is longer than this many seconds, the transcript is left
# on the clipboard (not restored) and a notification is shown, so you paste it
# deliberately with Cmd+V. 300 s = 5 min. Set 0 to disable the guard (always
# auto-paste). The transcript is always saved to the log regardless.
AUTO_PASTE_MAX_SEC = 300

# Same-application guard. If the keyboard focus has moved to a DIFFERENT
# application between pressing the hotkey and the transcript being ready, the
# text is not pasted (it would land in the wrong window): it is left on the
# clipboard and a notification is shown, like the long-recording guard above.
# Two windows of the same application are not told apart. Set False to always
# paste into whatever is frontmost.
PASTE_ONLY_IF_SAME_APP = True

# How the transcript is inserted. Chromium-based apps (Chrome, Brave, Edge and
# Electron apps such as Slack) can drop a Cmd+V that follows a clipboard
# restore, so by default the text is TYPED into them as keystrokes and the
# clipboard is not touched; every other app still gets Cmd+V.
#   "chromium" — type into Chromium/Electron apps, paste elsewhere (default)
#   "always"   — type everywhere
#   "never"    — always paste via Cmd+V
TYPE_INSTEAD_OF_PASTE = "chromium"

# Hands-free recording — for a meeting or a long note, where holding the right
# Option for minutes is impractical (every other key becomes Option+key).
# TOGGLE_HOTKEY is a chord: tap to start, tap again to stop. Spelled as
# modifiers + one key: "ctrl", "alt" (= LEFT Option; the right one stays the
# push-to-talk key), "cmd", "shift", then a letter or digit. Matched by key
# position, so any keyboard layout works. Set "" to disable the chord.
TOGGLE_HOTKEY = "ctrl+alt+r"

# The same hands-free recording by tapping the right Option twice; one more tap
# stops it. A tap is a press released within DOUBLE_TAP_MAX_HOLD_SEC, the
# second tap must come within DOUBLE_TAP_SEC of that release. A quick single
# tap therefore ends DOUBLE_TAP_SEC later than before; normal hold-and-speak
# dictation is not delayed. Set DOUBLE_TAP_SEC = 0 to disable the double tap.
DOUBLE_TAP_SEC = 0.45
DOUBLE_TAP_MAX_HOLD_SEC = 0.3

# Meeting transcript by speaker (2026-11-18). After a HANDS-FREE recording of
# DIARIZE_MIN_SEC or longer the transcript is also split by voice —
# "[mm:ss] Speaker 2: ..." — by whisper_diarize.py in a separate process. The
# plain transcript is on the clipboard at once; the by-speaker text follows as
# a file in MEETINGS_DIR and on the clipboard, with a notification. Needs an
# interpreter with sherpa-onnx (pip install sherpa-onnx numpy) and two ONNX
# models from https://github.com/k2-fsa/sherpa-onnx/releases — a speaker
# segmentation model and a speaker embedding model. Everything runs locally.
# DIARIZE_ASK_SPEAKERS = True asks how many people spoke (more reliable when
# you know); otherwise DIARIZE_AUTO_THRESHOLD decides (lower = more voices).
# DIARIZE_HANDS_FREE = True
# DIARIZE_PYTHON = "~/diarization/venv/bin/python"
# DIARIZE_SEGMENTATION_MODEL = "~/diarization/models/segmentation.onnx"
# DIARIZE_EMBEDDING_MODEL = "~/diarization/models/embedding.onnx"
# DIARIZE_MIN_SEC = 60
# DIARIZE_THREADS = 6
# DIARIZE_ASK_SPEAKERS = False
# DIARIZE_AUTO_THRESHOLD = 0.5
# DIARIZE_LABEL = "Speaker"
# MEETINGS_DIR = "~/Documents/whisper-meetings"

# Meeting mode: a hands-free recording (double tap / chord / /toggle) is a
# meeting whatever its length — plain transcript to the clipboard and to a file
# in MEETINGS_DIR, by-speaker text to a file only, nothing pasted into a window.
# MEETING_MODE = True
