"""Local overrides for whisper_dictate_macos_m.py (example).

Copy this file to `local_settings.py` (which is gitignored) to keep a personal
model choice and private domain vocabulary out of version control:

    cp local_settings.example.py local_settings.py

Then edit local_settings.py. Both settings below are optional — remove either
one to fall back to the script's default.
"""

# Override the transcription model. On an M-Pro / M-Max or any 16 GB+ Mac,
# large-v3-turbo is noticeably more accurate while still fast.
MODEL = "mlx-community/whisper-large-v3-turbo"

# Extra words appended to INITIAL_PROMPT — your own product names, jargon, etc.
# Whisper uses it as a hint and spells these terms correctly far more often.
EXTRA_PROMPT = "MyProduct, MyService, SomeJargon."

# Pause media while dictating and resume afterwards, by sending the system
# Play/Pause key (works with browser tabs/YouTube, Music, Spotify). Default True.
# Set to False to leave playback untouched.
AUTO_PAUSE_MEDIA = True

# Convert spoken punctuation ("новая строка", "запятая", "new line", "comma",
# ...) into real marks. Default False (some commands are also ordinary words,
# e.g. "точка зрения"). Set True to enable.
SPOKEN_PUNCTUATION = True

# Keep the mic stream warm for N seconds after a dictation (+ a short pre-roll)
# so the next dictation's first word isn't clipped by cold-start latency.
# Trade-off: the mic indicator stays on a few seconds after you stop.
# Set 0 to close the mic immediately after each dictation (max privacy).
KEEP_WARM_SEC = 8
