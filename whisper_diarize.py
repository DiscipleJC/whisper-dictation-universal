"""Who spoke when: turn a meeting recording plus its recognised words into
"[mm:ss] Speaker N: text" paragraphs.

Runs as a SEPARATE process, started by whisper_dictate_macos_m.py after a
hands-free recording (DIARIZE_HANDS_FREE). It is deliberately not imported by
the daemon: sherpa-onnx lives in its own interpreter (DIARIZE_PYTHON), the
models are loaded only for the minute or two the job takes, and a crash or a
hang here cannot touch the hotkey. Everything is local: two ONNX models read
from disk, no network.

  python whisper_diarize.py --wav rec.wav --words words.json --out text.txt \
      --seg segmentation.onnx --emb embedding.onnx [--speakers 5]

--words is a JSON list of {"start": sec, "end": sec, "word": " text"} — what
mlx-whisper returns with word_timestamps=True. --speakers 0 lets the
clustering threshold decide how many voices there are (less reliable than a
given number). The last line printed is a JSON summary for the daemon.
"""
import argparse
import json
import sys
import time
import wave


def read_wav(path):
    import numpy as np
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("need 16-bit PCM")
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels())[:, 0]
    return np.ascontiguousarray(pcm.astype(np.float32) / 32768.0), rate


def diarize(audio, rate, seg, emb, speakers, threshold, threads):
    """[(start, end, speaker index from 0)], sorted by start."""
    import sherpa_onnx
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb, num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=speakers if speakers > 0 else -1, threshold=threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not cfg.validate():
        raise ValueError("sherpa-onnx rejected the configuration (model paths?)")
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    if rate != sd.sample_rate:
        raise ValueError(f"need {sd.sample_rate} Hz audio, got {rate}")
    return [(r.start, r.end, r.speaker) for r in sd.process(audio).sort_by_start_time()]


def renumber(turns):
    """Number the voices 1, 2, 3… in the order they first speak — the raw
    cluster ids have gaps and no meaning."""
    order = {}
    for _, _, sp in turns:
        order.setdefault(sp, len(order) + 1)
    return [(s, e, order[sp]) for s, e, sp in turns]


def speaker_of(word, turns):
    """The voice whose turn overlaps the word most; a word that fell into a
    gap between turns goes to the nearest turn. None when there are no turns."""
    best, best_overlap = None, 0.0
    for s, e, sp in turns:
        overlap = min(word["end"], e) - max(word["start"], s)
        if overlap > best_overlap:
            best, best_overlap = sp, overlap
    if best is None and turns:
        mid = (word["start"] + word["end"]) / 2
        best = min(turns, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])))[2]
    return best


def stamp(sec):
    sec = int(sec)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def paragraphs(words, turns, label="Speaker"):
    """Consecutive words of one voice become one paragraph."""
    paras = []
    for w in words:
        sp = speaker_of(w, turns)
        if paras and paras[-1][0] == sp:
            paras[-1][2].append(w["word"])
        else:
            paras.append([sp, w["start"], [w["word"]]])
    out = []
    for sp, start, ws in paras:
        text = "".join(ws).strip()
        if text:
            who = f"{label} {sp}" if sp is not None else f"{label} ?"
            out.append(f"[{stamp(start)}] {who}: {text}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--words", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--speakers", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--label", default="Speaker")
    a = ap.parse_args()

    t0 = time.monotonic()
    audio, rate = read_wav(a.wav)
    with open(a.words, encoding="utf-8") as f:
        words = json.load(f)
    turns = renumber(diarize(audio, rate, a.seg, a.emb, a.speakers, a.threshold, a.threads))
    paras = paragraphs(words, turns, a.label)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(paras) + "\n")
    print(json.dumps({"speakers": len({t[2] for t in turns}), "paragraphs": len(paras),
                      "sec": round(time.monotonic() - t0, 1)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # the daemon reads stderr's last line into its log
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
