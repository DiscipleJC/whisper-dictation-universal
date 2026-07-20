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
