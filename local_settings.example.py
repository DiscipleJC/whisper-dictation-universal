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
