#!/usr/bin/env python3
"""
Whisper Transcriber — audio/video file transcription
Platforms: macOS Apple Silicon, macOS Intel, Linux, Windows
Usage: python transcribe.py <file> [options]
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ── Platform ──────────────────────────────────────────────────────────────────

SYSTEM           = platform.system()
MACHINE          = platform.machine()
IS_APPLE_SILICON = SYSTEM == "Darwin" and MACHINE == "arm64"

# ── Supported formats ─────────────────────────────────────────────────────────

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALL_EXTS   = AUDIO_EXTS | VIDEO_EXTS

# ── Default models ────────────────────────────────────────────────────────────

MODELS = {
    "mlx":     "mlx-community/whisper-large-v3-turbo",
    "faster":  "medium",
    "openai":  "whisper-1",
}

# Universal tech vocabulary — helps Whisper recognize proper nouns across all 99 languages
INITIAL_PROMPT = (
    "Claude Code, OpenAI, Python, JavaScript, TypeScript, GitHub, Docker, "
    "Kubernetes, API, REST, JSON, SQL, React, Node.js, Linux, macOS, Windows, "
    "Telegram, Slack, Zoom, YouTube, GPT, LLM, AI, ML, CPU, GPU, SSD, RAM."
)

# ── ffmpeg helpers ────────────────────────────────────────────────────────────

def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("❌  ffmpeg not found. Install it:")
        if SYSTEM == "Darwin":
            print("    brew install ffmpeg")
        elif SYSTEM == "Linux":
            print("    sudo apt install ffmpeg")
        else:
            print("    https://ffmpeg.org/download.html")
        sys.exit(1)


def _to_wav(input_path: Path) -> Path:
    """Convert any audio/video to 16kHz mono WAV in a temp file."""
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(input_path),
            "-ar", "16000", "-ac", "1", "-f", "wav",
            str(tmp),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"❌  ffmpeg error:\n{result.stderr}")
        sys.exit(1)
    return tmp

# ── Format writers ────────────────────────────────────────────────────────────

def _ts_srt(s: float) -> str:
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(sec):02d},{int((s % 1) * 1000):03d}"


def _ts_vtt(s: float) -> str:
    return _ts_srt(s).replace(",", ".")


def _write_txt(segments: list, path: Path):
    path.write_text(" ".join(s["text"] for s in segments).strip(), encoding="utf-8")


def _write_srt(segments: list, path: Path):
    lines = []
    for i, s in enumerate(segments, 1):
        lines += [str(i), f"{_ts_srt(s['start'])} --> {_ts_srt(s['end'])}",
                  s["text"].strip(), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_vtt(segments: list, path: Path):
    lines = ["WEBVTT", ""]
    for s in segments:
        lines += [f"{_ts_vtt(s['start'])} --> {_ts_vtt(s['end'])}",
                  s["text"].strip(), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(segments: list, path: Path):
    path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")


WRITERS = {"txt": _write_txt, "srt": _write_srt, "vtt": _write_vtt, "json": _write_json}

# ── Backends ──────────────────────────────────────────────────────────────────

def _transcribe_mlx(wav: Path, language, model: str) -> list:
    import mlx_whisper
    # NOTE: never pass beam_size — mlx_whisper raises NotImplementedError for any non-None value
    result = mlx_whisper.transcribe(
        str(wav),
        path_or_hf_repo=model,
        language=language,
        no_speech_threshold=0.3,
        condition_on_previous_text=False,
        initial_prompt=INITIAL_PROMPT,
    )
    return [{"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in result["segments"]]


def _transcribe_faster(wav: Path, language, model: str) -> list:
    from faster_whisper import WhisperModel
    wm = WhisperModel(model, device="auto", compute_type="auto")
    segments, _ = wm.transcribe(
        str(wav),
        language=language,
        beam_size=5,
        vad_filter=True,
        initial_prompt=INITIAL_PROMPT,
    )
    return [{"start": s.start, "end": s.end, "text": s.text} for s in segments]


def _transcribe_openai(wav: Path, language) -> list:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
                break

    from openai import OpenAI
    client = OpenAI()
    with open(wav, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    return [{"start": s.start, "end": s.end, "text": s.text}
            for s in response.segments]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Whisper Transcriber — audio/video to text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python transcribe.py meeting.mp4
  python transcribe.py lecture.mp3 --format srt
  python transcribe.py interview.wav --language ru --format txt --format json
  python transcribe.py video.mov --backend faster --model large-v3
        """,
    )
    parser.add_argument("input", help="Audio or video file")
    parser.add_argument("--language", "-l", default=None,
                        help="ISO language code: en, ru, uk, zh, de, fr, es, ja, ko, ar ... "
                             "All 99 Whisper languages supported. Default: auto-detect.")
    parser.add_argument("--format", "-f", action="append", dest="formats",
                        choices=["txt", "srt", "vtt", "json"], default=None,
                        help="Output format (repeatable). Default: txt + srt")
    parser.add_argument("--output", "-o", default=None,
                        help="Output base path (no extension)")
    parser.add_argument("--backend", "-b", default="auto",
                        choices=["auto", "mlx", "faster", "openai"],
                        help="Backend (default: auto — mlx on Apple Silicon, faster elsewhere)")
    parser.add_argument("--model", "-m", default=None,
                        help="Model override (e.g. large-v3, medium, tiny)")

    args = parser.parse_args()

    formats     = args.formats or ["txt", "srt"]
    input_path  = Path(args.input)
    output_base = Path(args.output) if args.output else input_path.with_suffix("")

    if not input_path.exists():
        print(f"❌  File not found: {input_path}")
        sys.exit(1)

    if input_path.suffix.lower() not in ALL_EXTS:
        print(f"❌  Unsupported format: {input_path.suffix}")
        print(f"    Supported: {', '.join(sorted(ALL_EXTS))}")
        sys.exit(1)

    # Resolve backend
    backend = args.backend
    if backend == "auto":
        backend = "mlx" if IS_APPLE_SILICON else "faster"

    model = args.model or MODELS[backend]

    print("=" * 50)
    print("  Whisper Transcriber")
    print("=" * 50)
    print(f"  File    : {input_path.name}")
    print(f"  Backend : {backend}  ({model.split('/')[-1]})")
    print(f"  Language: {args.language or 'auto-detect (99 languages)'}")
    print(f"  Formats : {', '.join(formats)}")
    print("=" * 50)

    # Convert to 16kHz mono WAV
    _check_ffmpeg()
    print("⏳  Converting to WAV...", flush=True)
    wav = _to_wav(input_path)

    try:
        print("⏳  Transcribing...", flush=True)
        if backend == "mlx":
            segments = _transcribe_mlx(wav, args.language, model)
        elif backend == "faster":
            segments = _transcribe_faster(wav, args.language, model)
        else:
            segments = _transcribe_openai(wav, args.language)
    finally:
        wav.unlink(missing_ok=True)

    if not segments:
        print("⚠️   No speech detected")
        sys.exit(0)

    # Write output files
    for fmt in formats:
        out = output_base.with_suffix(f".{fmt}")
        WRITERS[fmt](segments, out)
        print(f"✅  {out}")

    words    = len(" ".join(s["text"] for s in segments).split())
    duration = segments[-1]["end"]
    print(f"\n  Words   : {words}")
    print(f"  Duration: {duration:.1f}s")
    print()


if __name__ == "__main__":
    main()
